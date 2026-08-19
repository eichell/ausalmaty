"""Value (ТЗ 6.2, 6.3) и обязательный тест ТЗ 12 про отрицательную прибыль."""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest
from helpers import make_panel, trading_days

from factorbot.data import pit
from factorbot.factors.value import (
    COMPONENTS,
    ValueRules,
    compute_yields,
    market_cap,
    value_score,
)

DAYS = trading_days(400, start="2004-01-01")
AS_OF = DAYS[-1]

#: Цена у всех $10, чтобы капитализация читалась глазами: 10 × sharesbas.
PRICE = 10.0


def _panel(permatickers, price: float = PRICE):
    return make_panel({p: [price] * len(DAYS) for p in permatickers}, DAYS)


def _flow(permaticker: int, *, netinc=None, revenue=None, opcf=None, capex=None,
          available_from=date(2004, 5, 10)) -> dict:
    return {
        "permaticker": permaticker, "ticker": f"T{permaticker}", "dimension": "ART",
        "reportperiod": date(2004, 3, 31), "calendardate": date(2004, 3, 31),
        "available_from": available_from, "revenue_ttm": revenue,
        "netinc_ttm": netinc, "opcf_ttm": opcf, "capex_ttm": capex,
        "equity": None, "debt": None, "cash": None, "assets": None, "sharesbas": None,
    }


def _stock(permaticker: int, *, equity=None, shares=100.0,
           available_from=date(2004, 5, 10)) -> dict:
    return {
        "permaticker": permaticker, "ticker": f"T{permaticker}", "dimension": "ARQ",
        "reportperiod": date(2004, 3, 31), "calendardate": date(2004, 3, 31),
        "available_from": available_from, "revenue_ttm": None, "netinc_ttm": None,
        "opcf_ttm": None, "capex_ttm": None, "equity": equity, "debt": 0.0,
        "cash": 0.0, "assets": None, "sharesbas": shares,
    }


def _conn(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    pit.load_fundamentals(conn, pd.DataFrame(rows))
    return conn


# --------------------------------------------------------------------------- #
# Капитализация
# --------------------------------------------------------------------------- #


def test_market_cap_is_price_times_shares():
    panel = _panel([1001])
    caps = market_cap(panel, AS_OF, pd.Series({1001: 250.0}))
    assert caps[1001] == pytest.approx(PRICE * 250.0)


def test_market_cap_uses_the_unadjusted_price():
    """Скорректированная цена пересчитана от сегодняшней базы, а число акций —
    «тогдашнее». Перемножив их, у компании со сплитом получим капитализацию,
    отличающуюся в разы."""
    panel = _panel([1001])
    adjusted = panel.closeadj.copy()
    adjusted.loc[:, 1001] = 1.0                 # как если бы после сплитов
    panel = type(panel)(adjusted, panel.openadj, panel.close_unadj, panel.dollar_volume)

    caps = market_cap(panel, AS_OF, pd.Series({1001: 100.0}))
    assert caps[1001] == pytest.approx(PRICE * 100.0)


def test_zero_or_missing_shares_give_no_market_cap():
    panel = _panel([1001, 1002])
    caps = market_cap(panel, AS_OF, pd.Series({1001: 0.0, 1002: np.nan}))
    assert caps.isna().all()


# --------------------------------------------------------------------------- #
# Формулы ТЗ 6.2
# --------------------------------------------------------------------------- #


@pytest.fixture
def single_company():
    conn = _conn([
        _flow(1001, netinc=200.0, revenue=5000.0, opcf=300.0, capex=-50.0),
        _stock(1001, equity=800.0, shares=100.0),
    ])
    yield conn
    conn.close()


def test_yields_match_the_spec_formulas(single_company):
    yields = compute_yields(single_company, _panel([1001]), AS_OF, [1001])
    cap = PRICE * 100.0                          # = 1000
    row = yields.loc[1001]

    assert row["earnings_yield"] == pytest.approx(200.0 / cap)
    assert row["sales_yield"] == pytest.approx(5000.0 / cap)
    assert row["book_to_market"] == pytest.approx(800.0 / cap)


def test_free_cash_flow_subtracts_capital_expenditure(single_company):
    """У поставщика отток отрицателен, поэтому в коде плюс. Ошибка знака здесь
    удваивает FCF вместо вычитания и делает фактор любителем капиталоёмких."""
    yields = compute_yields(single_company, _panel([1001]), AS_OF, [1001])
    cap = PRICE * 100.0
    assert yields.loc[1001, "fcf_yield"] == pytest.approx((300.0 - 50.0) / cap)
    assert yields.loc[1001, "fcf_yield"] < 300.0 / cap


def test_all_four_components_are_returned(single_company):
    yields = compute_yields(single_company, _panel([1001]), AS_OF, [1001])
    assert list(yields.columns) == list(COMPONENTS)


# --------------------------------------------------------------------------- #
# Отрицательная прибыль (обязательный тест ТЗ 12)
# --------------------------------------------------------------------------- #


def test_loss_making_company_lands_at_the_bottom_not_the_top():
    """ТЗ 12: компания с отрицательной прибылью попадает в нижний дециль
    earnings_yield, а не в верхний.

    Ради этого ТЗ 6.2 и требует доходностей вместо мультипликаторов. Тест
    показывает разницу прямо: по E/P убыточная компания худшая, по P/E она
    выглядела бы самой дешёвой из всех.
    """
    profits = {1000 + i: float(50 * i) for i in range(1, 10)}   # от 50 до 450
    profits[1099] = -300.0                                       # убыток

    rows = []
    for permaticker, netinc in profits.items():
        rows.append(_flow(permaticker, netinc=netinc, revenue=5000.0,
                          opcf=netinc, capex=-10.0))
        rows.append(_stock(permaticker, equity=800.0, shares=100.0))

    conn = _conn(rows)
    try:
        yields = compute_yields(conn, _panel(list(profits)), AS_OF, list(profits))
    finally:
        conn.close()

    ranked = yields["earnings_yield"].sort_values()
    assert ranked.index[0] == 1099, "убыточная компания обязана быть худшей"

    # А теперь то, чего ТЗ велит избегать: тот же ряд через мультипликатор.
    price_to_earnings = 1.0 / yields["earnings_yield"]
    assert price_to_earnings.sort_values().index[0] == 1099, (
        "по P/E убыточная компания выглядит самой дешёвой — ровно поэтому "
        "ранжирование идёт по доходностям"
    )


def test_loss_making_company_gets_the_lowest_z_score():
    profits = {1000 + i: float(50 * i) for i in range(1, 10)}
    profits[1099] = -300.0
    rows = []
    for permaticker, netinc in profits.items():
        rows.append(_flow(permaticker, netinc=netinc, revenue=5000.0,
                          opcf=netinc, capex=-10.0))
        rows.append(_stock(permaticker, equity=800.0, shares=100.0))

    conn = _conn(rows)
    try:
        yields = compute_yields(conn, _panel(list(profits)), AS_OF, list(profits))
    finally:
        conn.close()

    sectors = pd.Series({p: "Technology" for p in profits})
    scores = value_score(yields, sectors)
    assert scores.idxmin() == 1099


# --------------------------------------------------------------------------- #
# Пропуски и исключения (ТЗ 6.3, 5)
# --------------------------------------------------------------------------- #


def _yields_frame(data: dict[int, dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(data, orient="index")
    frame.index.name = "permaticker"
    return frame.reindex(columns=list(COMPONENTS))


SECTORS = pd.Series({p: "Technology" for p in range(1, 12)})


def test_two_components_are_enough_to_stay_in_the_ranking():
    """ТЗ 6.3: балл по имеющимся, если не хватает одного или двух."""
    data = {p: {"earnings_yield": 0.01 * p, "sales_yield": 0.02 * p,
                "book_to_market": np.nan, "fcf_yield": np.nan} for p in range(1, 12)}
    scores = value_score(_yields_frame(data), SECTORS)
    assert scores.notna().all()


def test_one_component_is_not_enough():
    """ТЗ 6.3: при отсутствии трёх и более — исключается из ранжирования."""
    data = {p: {"earnings_yield": 0.01 * p, "sales_yield": np.nan,
                "book_to_market": np.nan, "fcf_yield": np.nan} for p in range(1, 12)}
    scores = value_score(_yields_frame(data), SECTORS)
    assert scores.isna().all()


def test_a_thin_company_is_excluded_not_scored_low():
    """Разница принципиальная: заниженный балл всё равно участвует в отборе,
    а исключение — нет."""
    data = {p: {"earnings_yield": 0.01 * p, "sales_yield": 0.02 * p,
                "book_to_market": 0.3, "fcf_yield": 0.05} for p in range(1, 11)}
    data[11] = {"earnings_yield": 0.5, "sales_yield": np.nan,
                "book_to_market": np.nan, "fcf_yield": np.nan}
    scores = value_score(_yields_frame(data), SECTORS)
    assert pd.isna(scores.loc[11])
    assert scores.drop(11).notna().all()


def test_financial_sector_is_excluded_from_value():
    """ТЗ 5: балансовые метрики финансов несопоставимы с остальными секторами."""
    data = {p: {"earnings_yield": 0.01 * p, "sales_yield": 0.02 * p,
                "book_to_market": 0.3, "fcf_yield": 0.05} for p in range(1, 12)}
    sectors = SECTORS.copy()
    sectors.loc[5] = "Financials"
    scores = value_score(_yields_frame(data), sectors)
    assert 5 not in scores.index


def test_components_are_averaged_as_z_scores_not_as_raw_yields():
    """Сырые доходности усреднять нельзя: sales_yield у торговых компаний
    измеряется единицами и задавил бы остальные три компонента."""
    data = {}
    for p in range(1, 12):
        data[p] = {"earnings_yield": 0.10 - 0.005 * p,   # убывает: лучший — первый
                   "sales_yield": 0.01 * p,               # растёт
                   "book_to_market": 0.5, "fcf_yield": 0.05}
    data[1]["sales_yield"] = 50.0                         # выброс в тысячи раз
    scores = value_score(_yields_frame(data), SECTORS)

    raw_mean = _yields_frame(data).mean(axis=1)
    assert raw_mean.idxmax() == 1, "среднее сырых величин определяется выбросом"
    assert scores.idxmax() != 1 or scores.loc[1] < raw_mean.loc[1]


# --------------------------------------------------------------------------- #
# PIT (ТЗ 4.8)
# --------------------------------------------------------------------------- #


def test_report_published_after_the_rebalance_date_is_not_used():
    """Ключевое требование проекта: отчёт, ещё не раскрытый на дату сигнала,
    в расчёт фактора попасть не может."""
    conn = _conn([
        _flow(1001, netinc=100.0, revenue=1000.0, opcf=100.0, capex=-10.0,
              available_from=date(2004, 5, 10)),
        _flow(1001, netinc=999.0, revenue=9999.0, opcf=999.0, capex=-10.0,
              available_from=date(2099, 1, 1)),
        _stock(1001, equity=800.0, shares=100.0),
    ])
    try:
        yields = compute_yields(conn, _panel([1001]), AS_OF, [1001])
    finally:
        conn.close()
    assert yields.loc[1001, "earnings_yield"] == pytest.approx(100.0 / (PRICE * 100.0))


def test_company_without_any_disclosed_report_is_absent():
    conn = _conn([
        _flow(1001, netinc=100.0, revenue=1000.0, opcf=100.0, capex=-10.0,
              available_from=date(2099, 1, 1)),
        _stock(1001, equity=800.0, shares=100.0, available_from=date(2099, 1, 1)),
    ])
    try:
        yields = compute_yields(conn, _panel([1001]), AS_OF, [1001])
    finally:
        conn.close()
    assert yields.empty


def test_balance_without_flows_still_yields_book_to_market():
    """Компания отчиталась только по ARQ: три компонента из четырёх отсутствуют,
    но book_to_market посчитать можно — и она честно выбывает по ТЗ 6.3."""
    conn = _conn([_stock(1001, equity=800.0, shares=100.0)])
    try:
        yields = compute_yields(conn, _panel([1001]), AS_OF, [1001])
    finally:
        conn.close()
    assert yields.loc[1001, "book_to_market"] == pytest.approx(0.8)
    assert yields.loc[1001, ["earnings_yield", "sales_yield", "fcf_yield"]].isna().all()


def test_empty_universe_gives_an_empty_frame():
    conn = _conn([_stock(1001, equity=1.0)])
    try:
        assert compute_yields(conn, _panel([1001]), AS_OF, []).empty
    finally:
        conn.close()
    assert value_score(pd.DataFrame(), pd.Series(dtype="string")).empty


def test_rules_come_from_the_config_not_from_the_code():
    """ТЗ 11: констант в коде быть не должно."""
    from pathlib import Path

    from factorbot.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "strategy.yaml")
    rules = ValueRules.from_config(cfg.factors, cfg.universe)

    assert rules.components == COMPONENTS
    assert rules.min_components == 2              # ТЗ 6.3: нужно два из четырёх
    assert rules.excluded_sector == "Financials"  # ТЗ 5
