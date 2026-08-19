"""Сквозной прогон: база → панель → вселенная → фактор → портфель → метрики.

Проверяет стыки, которые не видны в модульных тестах: пересчёт цены открытия,
календарь ребалансировок, чтение справочника. Данные синтетические и построены
так, что правильный ответ известен заранее.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from factorbot.backtest.costs import CostModel
from factorbot.backtest.engine import run_backtest
from factorbot.data.panel import load_price_panel
from factorbot.data.schema import create_all
from factorbot.factors.momentum import momentum
from factorbot.portfolio import PortfolioRules
from factorbot.report import metrics as M
from factorbot.universe import UniverseRules

N_NAMES = 24
DAYS = pd.bdate_range("2003-01-01", periods=800, name="date")
START, END = pd.Timestamp("2004-06-01"), DAYS[-1]


def _trending_prices(seed: int = 7) -> pd.DataFrame:
    """Ряды с устойчивыми трендами: половина бумаг растёт, половина падает.

    Тренд задан на всю историю, поэтому прошлый победитель остаётся победителем.
    На таких данных momentum обязан обыграть обратную к нему стратегию — иначе
    где-то перепутан знак, окно или дата.
    """
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.0006, 0.0006, N_NAMES)
    prices = {}
    for i, drift in enumerate(drifts):
        noise = rng.normal(0.0, 0.004, len(DAYS))
        prices[1000 + i] = 50.0 * np.exp(np.cumsum(drift + noise))
    return pd.DataFrame(prices, index=DAYS)


@pytest.fixture
def db(tmp_path):
    """DuckDB с ценами и справочником — как после factorbot-build."""
    closes = _trending_prices()
    path = tmp_path / "in_sample.duckdb"
    conn = duckdb.connect(str(path))
    create_all(conn)

    rows = []
    for permaticker in closes.columns:
        series = closes[permaticker]
        opens = series.shift(1).fillna(series)
        for day in DAYS:
            price = float(series.loc[day])
            rows.append({
                "permaticker": int(permaticker), "ticker": f"T{permaticker}",
                "date": day.date(), "open": float(opens.loc[day]),
                "high": price, "low": price, "close": price, "closeadj": price,
                "volume": 1e6, "dollar_volume": 200e6,
            })
    conn.register("_p", pd.DataFrame(rows))
    conn.execute("INSERT INTO prices SELECT * FROM _p")
    conn.unregister("_p")

    securities = pd.DataFrame([{
        "permaticker": int(p), "ticker": f"T{p}", "name": f"Company {p}",
        "exchange": "NYSE", "sector": f"Sector{i % 6}", "industry": "X",
        "siccode": "1234", "category": "Domestic Common Stock", "is_delisted": False,
        "first_price_date": DAYS[0].date(), "last_price_date": DAYS[-1].date(),
    } for i, p in enumerate(closes.columns)])
    conn.register("_s", securities)
    conn.execute("INSERT INTO securities SELECT * FROM _s")
    conn.unregister("_s")

    conn.close()
    return path


def _open_panel(db):
    conn = duckdb.connect(str(db), read_only=True)
    try:
        return load_price_panel(conn), conn.execute("SELECT * FROM securities").df()
    finally:
        conn.close()


def _run(panel, securities, *, sign: float):
    def score(p, as_of, universe):
        return sign * momentum(p, as_of).reindex(universe.index)

    return run_backtest(
        panel, securities, score_fn=score,
        universe_rules=UniverseRules(),
        portfolio_rules=PortfolioRules(top_n=6, buffer_rank=9, max_sector_weight=0.34),
        cost_model=CostModel(),
        start=START, end=END,
    )


def test_panel_recovers_the_adjusted_open(db):
    """openadj = open × closeadj / close. Смешивать сырое открытие со
    скорректированным закрытием нельзя (ТЗ 4.1)."""
    panel, _ = _open_panel(db)
    assert not panel.openadj.isna().all().any()
    assert panel.openadj.shape == panel.closeadj.shape


def test_rebalance_calendar_is_monthly(db):
    panel, _ = _open_panel(db)
    months = panel.month_end_dates()
    assert len(months) == pytest.approx(len(DAYS) / 21, rel=0.15)
    for day in months:
        assert panel.next_trading_day(day) is None or panel.next_trading_day(day) > day


def test_full_run_produces_a_usable_report(db):
    panel, securities = _open_panel(db)
    result = _run(panel, securities, sign=1.0)

    assert result.n_rebalances > 12
    assert result.equity_net.notna().all()
    assert (result.equity_net > 0).all()
    assert result.universe_size.min() > 0

    for weights in result.weights.values():
        assert weights.sum() == pytest.approx(1.0)
        assert (weights > 0).all()

    net = M.summarize(result)
    assert np.isfinite(net.cagr)
    assert np.isfinite(net.sharpe)
    assert net.max_drawdown <= 0.0
    assert net.annual_turnover >= 0.0


def test_momentum_beats_its_opposite_on_trending_data(db):
    """Системная проверка знака и таймингов: на рядах с устойчивыми трендами
    импульс обязан обыграть обратную стратегию."""
    panel, securities = _open_panel(db)
    forward = _run(panel, securities, sign=1.0)
    reverse = _run(panel, securities, sign=-1.0)
    assert forward.equity_net.iloc[-1] > reverse.equity_net.iloc[-1]


def test_costs_show_up_as_a_gap_between_gross_and_net(db):
    """ТЗ 8: отчёт обязан показывать доходность до и после издержек."""
    panel, securities = _open_panel(db)
    result = _run(panel, securities, sign=1.0)
    assert result.equity_gross.iloc[-1] > result.equity_net.iloc[-1]
    assert result.costs.sum() > 0
