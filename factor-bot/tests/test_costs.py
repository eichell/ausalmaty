"""Издержки (ТЗ 8)."""

from __future__ import annotations

import pandas as pd
import pytest

from factorbot.backtest.costs import BPS, CostModel, rebalance_cost, turnover

MODEL = CostModel()
LIQUID = pd.Series({1: 100e6, 2: 100e6, 3: 100e6, 4: 100e6})
ILLIQUID = pd.Series({1: 1e6, 2: 1e6, 3: 1e6, 4: 1e6})


def test_liquid_and_illiquid_rates_match_the_spec():
    rates = MODEL.rate_per_side(pd.Series({1: 100e6, 2: 1e6}))
    assert rates[1] == pytest.approx(10 * BPS)
    assert rates[2] == pytest.approx(25 * BPS)


def test_unknown_volume_is_priced_as_illiquid():
    """Ошибиться в сторону дорогой оценки безопаснее, чем занизить издержки."""
    rates = MODEL.rate_per_side(pd.Series({1: float("nan")}))
    assert rates[1] == pytest.approx(25 * BPS)


def test_threshold_is_exclusive_at_fifty_million():
    rates = MODEL.rate_per_side(pd.Series({1: 50e6, 2: 50e6 + 1}))
    assert rates[1] == pytest.approx(25 * BPS)
    assert rates[2] == pytest.approx(10 * BPS)


def test_full_replacement_of_the_portfolio_is_one_hundred_percent_turnover():
    before = pd.Series({1: 0.5, 2: 0.5})
    after = pd.Series({3: 0.5, 4: 0.5})
    assert turnover(before, after) == pytest.approx(1.0)


def test_unchanged_portfolio_costs_nothing():
    weights = pd.Series({1: 0.5, 2: 0.5})
    assert turnover(weights, weights) == pytest.approx(0.0)
    assert rebalance_cost(weights, weights, LIQUID, MODEL) == pytest.approx(0.0)


def test_both_sides_of_a_swap_are_paid_for():
    """Продажа одной бумаги ради покупки другой — два платежа, но один оборот."""
    before = pd.Series({1: 1.0})
    after = pd.Series({2: 1.0})
    cost = rebalance_cost(before, after, LIQUID, MODEL)
    assert cost == pytest.approx(2 * 10 * BPS)
    assert turnover(before, after) == pytest.approx(1.0)


def test_illiquid_names_cost_two_and_a_half_times_more():
    before = pd.Series({1: 1.0})
    after = pd.Series({2: 1.0})
    assert rebalance_cost(before, after, ILLIQUID, MODEL) == pytest.approx(
        2.5 * rebalance_cost(before, after, LIQUID, MODEL)
    )


def test_commission_is_added_on_top_of_slippage():
    model = CostModel(commission_bps=5.0)
    before, after = pd.Series({1: 1.0}), pd.Series({2: 1.0})
    assert rebalance_cost(before, after, LIQUID, model) == pytest.approx(2 * 15 * BPS)


def test_first_purchase_from_cash_is_charged_once():
    cost = rebalance_cost(pd.Series(dtype="float64"), pd.Series({1: 1.0}), LIQUID, MODEL)
    assert cost == pytest.approx(10 * BPS)
