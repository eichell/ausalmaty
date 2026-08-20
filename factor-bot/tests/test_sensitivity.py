"""Защита от переподгонки (ТЗ 9.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorbot.config import Section, with_overrides
from factorbot.sensitivity import (
    DEFAULT_SWEEPS,
    ShuffleTest,
    SweepSpec,
    run_sweep,
    shuffle_scores,
    shuffle_test,
)

# --------------------------------------------------------------------------- #
# Подмена параметра в конфиге
# --------------------------------------------------------------------------- #


BASE = Section({"portfolio": {"top_n": 30, "buffer_rank": 45},
                "factors": {"momentum": {"lookback_days": 252}}})


def test_override_changes_the_value_and_leaves_the_rest_alone():
    changed = with_overrides(BASE, {"portfolio.top_n": 20})
    assert changed.portfolio.top_n == 20
    assert changed.portfolio.buffer_rank == 45
    assert changed.factors.momentum.lookback_days == 252


def test_override_does_not_touch_the_original():
    with_overrides(BASE, {"portfolio.top_n": 20})
    assert BASE.portfolio.top_n == 30


def test_nested_path_works():
    changed = with_overrides(BASE, {"factors.momentum.lookback_days": 189})
    assert changed.factors.momentum.lookback_days == 189


def test_unknown_parameter_is_an_error():
    """Опечатка в пути обязана падать: молча созданный ключ ничего бы не изменил,
    и карта показывала бы плоскую линию вместо чувствительности."""
    with pytest.raises(KeyError, match="top_nn"):
        with_overrides(BASE, {"portfolio.top_nn": 20})


def test_unknown_section_is_an_error():
    with pytest.raises(KeyError, match="portfolios"):
        with_overrides(BASE, {"portfolios.top_n": 20})


# --------------------------------------------------------------------------- #
# Связанные параметры
# --------------------------------------------------------------------------- #


def test_buffer_scales_with_portfolio_size():
    """При портфеле в 50 бумаг буфер в 45 был бы меньше самого портфеля."""
    spec = next(s for s in DEFAULT_SWEEPS if s.path == "portfolio.top_n")
    assert spec.overrides(50)["portfolio.buffer_rank"] == 75
    assert spec.overrides(30)["portfolio.buffer_rank"] == 45   # как в ТЗ 7


def test_default_grid_matches_the_spec():
    """ТЗ 9.2.2: окно 9/12/15 мес., портфель 20/30/40/50, порог SMA 150/200/250."""
    grid = {s.path: list(s.values) for s in DEFAULT_SWEEPS}
    assert grid["factors.momentum.lookback_days"] == [189, 252, 315]   # 9/12/15 × 21
    assert grid["portfolio.top_n"] == [20, 30, 40, 50]
    assert grid["regime_filter.sma_window"] == [150, 200, 250]


# --------------------------------------------------------------------------- #
# Плато против пика (ТЗ 9.2.2)
# --------------------------------------------------------------------------- #


SPEC = SweepSpec("тест", "portfolio.top_n", [10, 20, 30, 40])


def _runner(by_value: dict) -> callable:
    def run(overrides):
        return pd.Series({"sharpe": by_value[overrides["portfolio.top_n"]],
                          "cagr": 0.1, "max_drawdown": -0.3, "turnover": 1.0})
    return run


def test_flat_results_are_a_plateau():
    verdict = run_sweep(_runner({10: 0.80, 20: 0.85, 30: 0.90, 40: 0.83}), SPEC)
    assert verdict.is_plateau
    assert verdict.best_value == 30


def test_a_lone_spike_is_flagged_as_overfitting():
    """Хорошо при одном значении и провал у соседей — это подгонка, а не эффект."""
    verdict = run_sweep(_runner({10: 0.05, 20: 0.10, 30: 1.90, 40: 0.08}), SPEC)
    assert not verdict.is_plateau
    assert "ПИК" in verdict.summary()


def test_no_signal_anywhere_is_reported_as_such():
    verdict = run_sweep(_runner({10: -0.2, 20: -0.3, 30: -0.1, 40: -0.4}), SPEC)
    assert not verdict.is_plateau
    assert "сигнала нет" in verdict.summary()


def test_best_at_the_edge_uses_its_only_neighbour():
    verdict = run_sweep(_runner({10: 0.90, 20: 0.85, 30: 0.60, 40: 0.55}), SPEC)
    assert verdict.best_value == 10
    assert verdict.neighbour_ratio == pytest.approx(0.85 / 0.90)


def test_table_keeps_every_value_for_the_report():
    verdict = run_sweep(_runner({10: 0.8, 20: 0.9, 30: 0.7, 40: 0.6}), SPEC)
    assert list(verdict.table.index) == [10, 20, 30, 40]
    assert "turnover" in verdict.table.columns


# --------------------------------------------------------------------------- #
# Перемешанные данные (ТЗ 9.2.4)
# --------------------------------------------------------------------------- #


def test_shuffling_keeps_the_values_but_breaks_the_mapping():
    rng = np.random.default_rng(0)
    original = pd.Series({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
    wrapped = shuffle_scores(rng)(lambda p, d, u: original)

    shuffled = wrapped(None, None, None)
    assert sorted(shuffled.to_numpy()) == sorted(original.to_numpy())
    assert list(shuffled.index) == list(original.index)


def test_shuffling_actually_changes_the_order():
    rng = np.random.default_rng(1)
    original = pd.Series({i: float(i) for i in range(50)})
    wrapped = shuffle_scores(rng)(lambda p, d, u: original)
    assert not wrapped(None, None, None).equals(original)


def test_real_result_above_every_shuffle_gives_the_lowest_possible_p_value():
    test = ShuffleTest(real=1.5, shuffled=[0.1, -0.2, 0.05, 0.0, 0.2], metric="sharpe")
    assert test.p_value == pytest.approx(1 / 6)


def test_too_few_shuffles_cannot_reach_significance():
    """Минимальное достижимое p равно 1/(n+1): чтобы получить 0.05, нужно не
    меньше девятнадцати перестановок. Отсюда и значение по умолчанию."""
    few = ShuffleTest(real=99.0, shuffled=[0.0] * 5, metric="sharpe")
    enough = ShuffleTest(real=99.0, shuffled=[0.0] * 20, metric="sharpe")
    assert not few.has_edge
    assert enough.has_edge


def test_shuffled_close_to_real_is_normal_for_a_long_only_portfolio():
    """Перемешанный Sharpe близок к настоящему у любой long-only стратегии:
    обоими движет рыночная бета. Само по себе это не утечка."""
    test = ShuffleTest(real=1.5, shuffled=[1.4, 1.45, 1.5, 1.42], metric="sharpe",
                       baseline=1.45)
    assert not test.leak_suspected
    assert not test.has_edge
    assert "Преимущества над случайным отбором не видно" in test.summary()


def test_random_selection_beating_the_market_by_far_suggests_a_leak():
    """Вот это действительно подозрительно: случайный отбор из вселенной —
    примерно равновзвешенный индекс, и обгонять рынок вдвое он не должен."""
    test = ShuffleTest(real=3.0, shuffled=[2.8, 2.9, 3.1], metric="sharpe",
                       baseline=0.5)
    assert test.leak_suspected
    assert "УТЕЧКА" in test.summary()


def test_without_a_benchmark_the_question_is_left_open():
    """Делать вид, что ответ есть, нельзя."""
    test = ShuffleTest(real=1.5, shuffled=[1.4, 1.45], metric="sharpe")
    assert test.leak_suspected is None
    assert "проверить нечем" in test.summary()


def test_p_value_is_one_when_every_shuffle_wins():
    test = ShuffleTest(real=0.1, shuffled=[0.5, 0.6, 0.7], metric="sharpe")
    assert test.p_value == pytest.approx(1.0)


def test_shuffle_test_runs_the_real_case_once_and_the_rest_shuffled():
    calls = []

    def runner(wrapper):
        calls.append(wrapper)
        return pd.Series({"sharpe": 1.0 if wrapper is None else 0.0})

    result = shuffle_test(runner, n_shuffles=5, metric="sharpe")
    assert len(calls) == 6
    assert calls.count(None) == 1
    assert result.real == pytest.approx(1.0)
    assert result.shuffled == [0.0] * 5
