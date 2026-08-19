"""Метрики отчёта (ТЗ 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorbot.report import metrics as M


def _equity(values: list[float], start: str = "2000-01-03") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values), name="date")
    return pd.Series(values, index=idx, name="equity")


def test_cagr_on_a_known_doubling():
    """Ровно два года, рост вдвое → 2**0.5 − 1 ≈ 41.4%."""
    idx = pd.DatetimeIndex(["2000-01-01", "2002-01-01"])
    equity = pd.Series([1.0, 2.0], index=idx)
    assert M.cagr(equity) == pytest.approx(2**0.5 - 1, rel=1e-3)


def test_flat_equity_has_zero_return_and_zero_volatility():
    equity = _equity([1.0] * 250)
    assert M.cagr(equity) == pytest.approx(0.0)
    assert M.volatility(equity.pct_change().dropna()) == pytest.approx(0.0)


def test_sharpe_scales_by_root_252():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0004, 0.01, 5000))
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert M.sharpe(returns) == pytest.approx(expected)


def test_sortino_ignores_upside_volatility():
    """Ряд с крупными положительными выбросами наказываться не должен."""
    calm = pd.Series([0.001] * 100 + [-0.001] * 100)
    spiky = pd.Series([0.05] * 100 + [-0.001] * 100)
    assert M.sortino(spiky) > M.sortino(calm)


def test_sortino_is_infinite_without_losing_days():
    assert M.sortino(pd.Series([0.01] * 50)) == float("inf")


def test_max_drawdown_matches_the_hand_computed_value():
    equity = _equity([100.0, 120.0, 60.0, 90.0, 130.0])
    assert M.max_drawdown(equity) == pytest.approx(-0.5)


def test_drawdown_is_zero_on_a_monotone_curve():
    assert M.max_drawdown(_equity([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_underwater_period_that_never_recovers_still_counts():
    """Инвестор находится в этой просадке прямо сейчас — не считать её нечестно."""
    equity = pd.Series(
        [100.0] + [50.0] * 400, index=pd.bdate_range("2000-01-03", periods=401)
    )
    assert M.longest_underwater_months(equity) > 18


def test_underwater_measures_the_longest_stretch_not_the_last():
    values = [100.0, 90.0, 100.0] + [90.0] * 200 + [200.0, 190.0]
    equity = pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))
    assert M.longest_underwater_months(equity) > 9


# --------------------------------------------------------------------------- #
# Сравнение с бенчмарком (ТЗ 10 — самая важная цифра)
# --------------------------------------------------------------------------- #


def test_underperformance_is_measured_on_the_relative_curve():
    """Стратегия растёт, но медленнее индекса. Просадки в абсолюте нет, а
    отставание есть — именно его инвестор и не выдерживает."""
    days = pd.bdate_range("2000-01-03", periods=500)
    strategy = pd.Series(np.linspace(1.0, 1.5, 500), index=days)
    benchmark = pd.Series(np.linspace(1.0, 3.0, 500), index=days)

    assert M.max_drawdown(strategy) == pytest.approx(0.0)
    assert M.longest_underwater_months(M.relative_equity(strategy, benchmark)) > 20


def test_matching_the_benchmark_leaves_no_underperformance():
    days = pd.bdate_range("2000-01-03", periods=300)
    curve = pd.Series(np.linspace(1.0, 2.0, 300), index=days)
    assert M.longest_underwater_months(M.relative_equity(curve, curve)) == pytest.approx(0.0)


def test_yearly_returns_are_split_by_calendar_year():
    days = pd.bdate_range("2000-01-03", "2002-12-31")
    equity = pd.Series(np.linspace(1.0, 4.0, len(days)), index=days)
    table = M.yearly_returns(equity)
    assert list(table.index) == [2000, 2001, 2002]
    assert (table["strategy"] > 0).all()


def test_yearly_table_shows_excess_over_the_benchmark():
    days = pd.bdate_range("2000-01-03", "2001-12-31")
    strategy = pd.Series(np.linspace(1.0, 2.0, len(days)), index=days)
    benchmark = pd.Series(np.linspace(1.0, 1.5, len(days)), index=days)
    table = M.yearly_returns(strategy, benchmark)
    assert (table["excess"] > 0).all()


# --------------------------------------------------------------------------- #
# Признаки, что что-то не так (ТЗ 9.3)
# --------------------------------------------------------------------------- #


def test_impossibly_good_sharpe_is_flagged():
    metrics = M.Metrics(
        cagr=0.3, volatility=0.12, sharpe=2.5, sortino=3.0, max_drawdown=-0.4,
        max_drawdown_months=10, max_underperformance_months=5, annual_turnover=1.0,
        n_rebalances=100, average_holding_months=6, total_return=5.0, years=14,
    )
    assert any("Sharpe" in flag for flag in M.sanity_warnings(metrics))


def test_suspiciously_shallow_drawdown_is_flagged():
    metrics = M.Metrics(
        cagr=0.12, volatility=0.15, sharpe=0.8, sortino=1.1, max_drawdown=-0.10,
        max_drawdown_months=6, max_underperformance_months=5, annual_turnover=1.0,
        n_rebalances=100, average_holding_months=6, total_return=3.0, years=14,
    )
    assert any("просадка" in flag for flag in M.sanity_warnings(metrics))


def test_realistic_result_raises_no_flags():
    """Ориентир ТЗ 9.3: Sharpe 0.6–0.9, просадка 35–50%."""
    metrics = M.Metrics(
        cagr=0.13, volatility=0.18, sharpe=0.75, sortino=1.05, max_drawdown=-0.42,
        max_drawdown_months=28, max_underperformance_months=19, annual_turnover=1.8,
        n_rebalances=168, average_holding_months=6.7, total_return=4.2, years=14,
    )
    assert M.sanity_warnings(metrics) == []
