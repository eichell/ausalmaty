"""Вселенная (ТЗ 5). Главный риск здесь — survivorship bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_panel, make_securities, trading_days

from factorbot.universe import UniverseRules, build_universe, eligible_securities

RULES = UniverseRules()
DAYS = trading_days(500, start="2003-01-01")


def _flat_panel(permatickers, price=50.0, **kwargs):
    return make_panel({p: [price] * len(DAYS) for p in permatickers}, DAYS, **kwargs)


def test_delisted_flag_never_filters_the_universe():
    """ТЗ 4.1: отбор по `is_delisted` — это survivorship bias.

    Флаг относится к сегодняшнему дню; применённый к 2005 году, он выкидывает
    компании, которые тогда прекрасно торговались и были бы куплены.
    """
    panel = _flat_panel([1001, 1002])
    securities = make_securities([1001, 1002], delisted={1002})
    universe = build_universe(panel, securities, DAYS[-1], RULES)
    assert 1002 in universe.index


def test_penny_stock_is_excluded():
    panel = make_panel({1001: [50.0] * len(DAYS), 1002: [4.0] * len(DAYS)}, DAYS)
    universe = build_universe(panel, make_securities([1001, 1002]), DAYS[-1], RULES)
    assert set(universe.index) == {1001}


def test_price_threshold_uses_the_unadjusted_series():
    """Скорректированная цена пересчитана от сегодняшней базы: акция за $40 в 1999
    году может числиться в панели за $3, и порог ТЗ 5 отсёк бы её задним числом."""
    panel = make_panel({1001: [3.0] * len(DAYS)}, DAYS)
    raw = panel.close.copy()
    raw.loc[:, 1001] = 40.0
    panel = type(panel)(panel.closeadj, panel.openadj, raw, panel.dollar_volume)

    universe = build_universe(panel, make_securities([1001]), DAYS[-1], RULES)
    assert 1001 in universe.index


def test_illiquid_name_is_excluded():
    panel = _flat_panel([1001, 1002], dollar_volume={1001: 100e6, 1002: 1e6})
    universe = build_universe(panel, make_securities([1001, 1002]), DAYS[-1], RULES)
    assert set(universe.index) == {1001}


def test_short_history_is_excluded():
    """Нужно 14 месяцев истории — иначе momentum считать не из чего (ТЗ 5)."""
    prices = {1001: [50.0] * len(DAYS), 1002: [np.nan] * (len(DAYS) - 60) + [50.0] * 60}
    universe = build_universe(
        make_panel(prices, DAYS), make_securities([1001, 1002]), DAYS[-1], RULES
    )
    assert set(universe.index) == {1001}


def test_name_not_trading_on_the_rebalance_date_is_excluded():
    prices = {1001: [50.0] * len(DAYS), 1002: [50.0] * (len(DAYS) - 1) + [np.nan]}
    universe = build_universe(
        make_panel(prices, DAYS), make_securities([1001, 1002]), DAYS[-1], RULES
    )
    assert set(universe.index) == {1001}


# --------------------------------------------------------------------------- #
# Справочник
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("category", [
    "Domestic Common Stock ETF", "ADR Common Stock", "Domestic Stock Warrant",
    "Canadian Preferred Stock", "ETD", "Domestic Fund", "Unit",
])
def test_non_common_stock_is_excluded(category):
    """ТЗ 5: ETF, фонды, трасты, ADR, SPAC — не наша вселенная."""
    securities = make_securities([1001], category=category)
    assert len(eligible_securities(securities, RULES)) == 0


def test_domestic_common_stock_is_kept():
    securities = make_securities([1001], category="Domestic Common Stock")
    assert list(eligible_securities(securities, RULES)) == [1001]


@pytest.mark.parametrize("exchange", ["OTC", "PINK", "BATS"])
def test_secondary_exchanges_are_excluded(exchange):
    securities = make_securities([1001], exchange=exchange)
    assert len(eligible_securities(securities, RULES)) == 0


def test_sector_is_carried_into_the_universe():
    panel = _flat_panel([1001])
    securities = make_securities([1001], sectors={1001: "Energy"})
    universe = build_universe(panel, securities, DAYS[-1], RULES)
    assert universe.loc[1001, "sector"] == "Energy"


def test_missing_sector_becomes_its_own_group_not_a_hole():
    panel = _flat_panel([1001])
    securities = make_securities([1001])
    securities.loc[:, "sector"] = None
    universe = build_universe(panel, securities, DAYS[-1], RULES)
    assert universe.loc[1001, "sector"] == "Unknown"


def test_non_trading_date_is_an_error_not_a_silent_empty_result():
    panel = _flat_panel([1001])
    with pytest.raises(ValueError, match="не торговый день"):
        build_universe(panel, make_securities([1001]), pd.Timestamp("2003-01-05"), RULES)
