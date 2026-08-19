"""Режимный фильтр (ТЗ 7.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import first_execution_day, make_panel, make_securities, trading_days

from factorbot.backtest.costs import CostModel
from factorbot.backtest.engine import run_backtest
from factorbot.data.panel import PricePanel
from factorbot.portfolio import PortfolioRules
from factorbot.regime import RegimeFilter, RegimeRules, build_regime_filter
from factorbot.universe import UniverseRules

DAYS = trading_days(600, start="2004-01-01")
WINDOW = 200


def _series(values: list[float], days: pd.DatetimeIndex = DAYS) -> pd.Series:
    return pd.Series(values, index=days)


# --------------------------------------------------------------------------- #
# Сигнал
# --------------------------------------------------------------------------- #


def test_rising_market_is_risk_on():
    prices = _series([100.0 + i for i in range(len(DAYS))])
    assert not RegimeFilter(prices, sma_window=WINDOW).is_risk_off(DAYS[-1])


def test_falling_market_is_risk_off():
    prices = _series([100.0 + len(DAYS) - i for i in range(len(DAYS))])
    assert RegimeFilter(prices, sma_window=WINDOW).is_risk_off(DAYS[-1])


def test_price_exactly_at_the_average_counts_as_risk_on():
    """Строгое неравенство ТЗ 7.1: защита включается ниже средней, а не на ней."""
    prices = _series([100.0] * len(DAYS))
    assert not RegimeFilter(prices, sma_window=WINDOW).is_risk_off(DAYS[-1])


def test_short_history_defaults_to_risk_on():
    """Отсутствие сигнала — не повод сидеть в деньгах: фильтр защитный оверлей,
    а не разрешение на вход."""
    days = trading_days(50)
    prices = pd.Series([100.0 - i for i in range(50)], index=days)
    assert not RegimeFilter(prices, sma_window=WINDOW).is_risk_off(days[-1])


def test_signal_uses_only_prices_up_to_the_signal_day():
    """Заглянуть в завтрашнюю цену здесь особенно соблазнительно: фильтр
    срабатывает на резких движениях, и один день разницы систематически
    улучшает результат на каждом развороте."""
    values = [100.0 + i for i in range(len(DAYS))]
    as_of = DAYS[400]
    prices = _series(values)

    crashed = list(values)
    for i in range(401, len(DAYS)):
        crashed[i] = 1.0                          # обвал уже после дня сигнала

    assert RegimeFilter(prices, sma_window=WINDOW).is_risk_off(as_of) == (
        RegimeFilter(_series(crashed), sma_window=WINDOW).is_risk_off(as_of)
    )


def test_window_length_changes_the_verdict():
    """Порог SMA — параметр карты чувствительности ТЗ 9.2.2, а не константа."""
    # Долгий подъём, затем неглубокая коррекция: цена ниже короткой средней,
    # но всё ещё заметно выше длинной.
    values = [50.0 + 0.3 * i for i in range(500)]
    values += [values[-1] - 0.2 * i for i in range(len(DAYS) - 500)]
    prices = _series(values)

    assert RegimeFilter(prices, sma_window=50).is_risk_off(DAYS[-1])
    assert not RegimeFilter(prices, sma_window=400).is_risk_off(DAYS[-1])


# --------------------------------------------------------------------------- #
# Защитный актив
# --------------------------------------------------------------------------- #


def test_risk_off_portfolio_is_fully_in_the_defensive_asset():
    prices = _series([100.0] * len(DAYS))
    flt = RegimeFilter(prices, sma_window=WINDOW, risk_off_permaticker=7,
                       risk_off_prices=prices)
    weights = flt.risk_off_weights(DAYS[-1])
    assert weights.to_dict() == {7: 1.0}


def test_defensive_asset_that_does_not_exist_yet_leaves_capital_in_cash(caplog):
    """SHY начал торговаться в июле 2002, а in-sample начинается в 1999 (ТЗ 9.1).
    Уходить некуда, и это надо видеть в логе, а не подставлять что-то другое."""
    late = pd.Series([np.nan] * 500 + [80.0] * 100, index=DAYS)
    flt = RegimeFilter(_series([100.0] * len(DAYS)), sma_window=WINDOW,
                       risk_off_permaticker=7, risk_off_prices=late)

    with caplog.at_level("WARNING"):
        weights = flt.risk_off_weights(DAYS[10])
    assert weights.empty
    assert any("Защитный актив недоступен" in r.getMessage() for r in caplog.records)


def test_defensive_asset_is_used_once_it_starts_trading():
    late = pd.Series([np.nan] * 500 + [80.0] * 100, index=DAYS)
    flt = RegimeFilter(_series([100.0] * len(DAYS)), sma_window=WINDOW,
                       risk_off_permaticker=7, risk_off_prices=late)
    assert flt.risk_off_weights(DAYS[-1]).to_dict() == {7: 1.0}


# --------------------------------------------------------------------------- #
# Сборка из базы
# --------------------------------------------------------------------------- #


def _panel_with_market(stock_prices: list[float], market_prices: list[float],
                       bond_prices: list[float] | None = None) -> PricePanel:
    closes = {1: stock_prices, 900: market_prices}
    if bond_prices is not None:
        closes[901] = bond_prices
    return make_panel(closes, DAYS)


def _securities_with_market(include_bond: bool = True) -> pd.DataFrame:
    rows = make_securities([1])
    extra = make_securities([900, 901] if include_bond else [900],
                            category="Domestic Common Stock ETF")
    extra.loc[extra["permaticker"] == 900, "ticker"] = "SPY"
    if include_bond:
        extra.loc[extra["permaticker"] == 901, "ticker"] = "SHY"
    return pd.concat([rows, extra], ignore_index=True)


def test_disabled_rules_build_no_filter():
    panel = _panel_with_market([50.0] * len(DAYS), [100.0] * len(DAYS), [80.0] * len(DAYS))
    rules = RegimeRules(enabled=False)
    assert build_regime_filter(panel, _securities_with_market(), rules) is None


def test_missing_benchmark_is_an_error_not_a_silent_disable():
    """Прогон назывался бы «с фильтром» и шёл бы без него — худший из исходов."""
    panel = make_panel({1: [50.0] * len(DAYS)}, DAYS)
    with pytest.raises(ValueError, match="Бенчмарк"):
        build_regime_filter(panel, make_securities([1]), RegimeRules())


def test_missing_defensive_asset_is_only_a_warning(caplog):
    panel = _panel_with_market([50.0] * len(DAYS), [100.0] * len(DAYS))
    with caplog.at_level("WARNING"):
        flt = build_regime_filter(panel, _securities_with_market(include_bond=False),
                                  RegimeRules())
    assert flt is not None
    assert any("Защитный актив" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Через движок
# --------------------------------------------------------------------------- #


START, END = pd.Timestamp("2005-06-01"), DAYS[-1]
NO_COSTS = CostModel(0.0, 0.0, 0.0)


def _run(panel, securities, *, regime):
    return run_backtest(
        panel, securities,
        score_fn=lambda p, d, u: pd.Series(1.0, index=u.index),
        universe_rules=UniverseRules(),
        portfolio_rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
        cost_model=NO_COSTS, start=START, end=END, regime=regime,
    )


def _crashing_market():
    """Акция и рынок падают со второй половины истории, облигации стоят."""
    turn = 350
    stock = [100.0] * turn + [100.0 - 0.2 * i for i in range(len(DAYS) - turn)]
    market = list(stock)
    bonds = [80.0] * len(DAYS)
    return _panel_with_market(stock, market, bonds), _securities_with_market()


def test_filter_moves_the_portfolio_into_bonds_during_a_crash():
    panel, securities = _crashing_market()
    flt = build_regime_filter(panel, securities, RegimeRules(sma_window=WINDOW))
    result = _run(panel, securities, regime=flt)

    assert result.risk_off_dates, "фильтр обязан был сработать на падающем рынке"
    last = list(result.weights.values())[-1]
    assert list(last.index) == [901], "в защите портфель должен быть в SHY"


def test_filter_reduces_the_drawdown_it_was_added_for():
    panel, securities = _crashing_market()
    without = _run(panel, securities, regime=None)
    with_filter = _run(panel, securities,
                       regime=build_regime_filter(panel, securities,
                                                  RegimeRules(sma_window=WINDOW)))

    assert with_filter.equity_net.iloc[-1] > without.equity_net.iloc[-1]
    assert with_filter.risk_off_share > 0


def test_filter_changes_nothing_while_the_market_is_above_its_average():
    rising = [50.0 + 0.1 * i for i in range(len(DAYS))]
    panel = _panel_with_market(rising, rising, [80.0] * len(DAYS))
    securities = _securities_with_market()

    without = _run(panel, securities, regime=None)
    with_filter = _run(panel, securities,
                       regime=build_regime_filter(panel, securities,
                                                  RegimeRules(sma_window=WINDOW)))

    assert with_filter.risk_off_dates == []
    assert with_filter.equity_net.iloc[-1] == pytest.approx(without.equity_net.iloc[-1])


def test_risk_off_without_a_defensive_asset_holds_cash_flat():
    """Капитал в деньгах под нулевую ставку: занижает результат защиты, то есть
    ошибается против фильтра, а не в его пользу."""
    turn = 350
    stock = [100.0] * turn + [100.0 - 0.2 * i for i in range(len(DAYS) - turn)]
    panel = _panel_with_market(stock, list(stock))
    securities = _securities_with_market(include_bond=False)

    result = _run(panel, securities,
                  regime=build_regime_filter(panel, securities,
                                             RegimeRules(sma_window=WINDOW)))
    assert result.risk_off_dates
    tail = result.equity_net.loc[result.risk_off_dates[-1]:]
    assert tail.std() == pytest.approx(0.0, abs=1e-12), "в деньгах капитал не меняется"


def test_switching_into_the_defensive_asset_costs_money():
    panel, securities = _crashing_market()
    flt = build_regime_filter(panel, securities, RegimeRules(sma_window=WINDOW))
    result = run_backtest(
        panel, securities,
        score_fn=lambda p, d, u: pd.Series(1.0, index=u.index),
        universe_rules=UniverseRules(),
        portfolio_rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
        cost_model=CostModel(), start=START, end=END, regime=flt,
    )
    switch = result.risk_off_dates[0]
    execution = first_execution_day(DAYS, switch)
    assert result.costs.loc[execution] > 0, "уход в защиту — это сделка, она платная"
