"""Композит факторов (ТЗ 6.3, 9.2.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorbot.factors.composite import CompositeRules, combine

RULES = CompositeRules(weights={"momentum": 0.5, "value": 0.5})


def test_equal_weights_give_the_average_of_the_two_scores():
    scores = combine({
        "momentum": pd.Series({1: 2.0, 2: -1.0}),
        "value": pd.Series({1: 0.0, 2: 1.0}),
    }, RULES)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)


def test_unequal_weights_are_applied_as_given():
    rules = CompositeRules(weights={"momentum": 0.75, "value": 0.25})
    scores = combine({
        "momentum": pd.Series({1: 4.0}),
        "value": pd.Series({1: 0.0}),
    }, rules)
    assert scores[1] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Неполный набор компонентов
# --------------------------------------------------------------------------- #


def test_missing_value_leaves_the_stock_ranked_by_momentum_alone():
    """ТЗ 5: финансы исключены из value, но для momentum оставлены.

    Буквальная формула дала бы NaN и выкинула бумагу из портфеля вовсе — то есть
    исключила бы её и из momentum, вопреки требованию.
    """
    scores = combine({
        "momentum": pd.Series({1: 2.0, 2: 1.0}),
        "value": pd.Series({1: np.nan, 2: 1.0}),
    }, RULES)
    assert scores[1] == pytest.approx(2.0)
    assert scores[2] == pytest.approx(1.0)


def test_missing_component_is_not_treated_as_zero():
    """Подстановка нуля вместо пропуска — это заполнение пропусков, запрещённое
    ТЗ 6.3, и она систематически занижала бы балл таких бумаг."""
    scores = combine({
        "momentum": pd.Series({1: 2.0}),
        "value": pd.Series({1: np.nan}),
    }, RULES)
    assert scores[1] != pytest.approx(1.0), "0.5 * 2 + 0.5 * 0 — так нельзя"
    assert scores[1] == pytest.approx(2.0)


def test_missing_momentum_leaves_the_stock_ranked_by_value_alone():
    scores = combine({
        "momentum": pd.Series({1: np.nan}),
        "value": pd.Series({1: -1.5}),
    }, RULES)
    assert scores[1] == pytest.approx(-1.5)


def test_stock_without_any_component_is_excluded():
    scores = combine({
        "momentum": pd.Series({1: np.nan}),
        "value": pd.Series({1: np.nan}),
    }, RULES)
    assert pd.isna(scores[1])


def test_requiring_both_components_excludes_the_partial_ones():
    strict = CompositeRules(weights={"momentum": 0.5, "value": 0.5}, min_components=2)
    scores = combine({
        "momentum": pd.Series({1: 2.0, 2: 1.0}),
        "value": pd.Series({1: np.nan, 2: 1.0}),
    }, strict)
    assert pd.isna(scores[1])
    assert scores[2] == pytest.approx(1.0)


def test_indexes_are_unioned_not_intersected():
    scores = combine({
        "momentum": pd.Series({1: 1.0}),
        "value": pd.Series({2: 2.0}),
    }, RULES)
    assert set(scores.index) == {1, 2}
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Отказы и конфиг
# --------------------------------------------------------------------------- #


def test_component_without_a_weight_is_an_error_not_a_silent_drop():
    """Молча проигнорировать компонент значит изменить стратегию без следа
    в конфиге — то есть сделать прогон невоспроизводимым."""
    with pytest.raises(KeyError, match="quality"):
        combine({"quality": pd.Series({1: 1.0})}, RULES)


def test_empty_input_gives_an_empty_series():
    assert combine({}, RULES).empty


def test_weights_come_from_the_config_and_sum_to_one():
    """ТЗ 9.2.3: 50/50 — априорный выбор, не результат подбора по бэктесту."""
    from pathlib import Path

    from factorbot.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "strategy.yaml")
    rules = CompositeRules.from_config(cfg.factors)
    assert rules.weights == {"momentum": 0.5, "value": 0.5}


def test_weights_that_do_not_sum_to_one_are_rejected():
    class _Cfg:
        weights = {"momentum": 0.7, "value": 0.5}

    with pytest.raises(ValueError, match="1.0"):
        CompositeRules.from_config(_Cfg())


# --------------------------------------------------------------------------- #
# Свойства композита
# --------------------------------------------------------------------------- #


def test_composite_sits_between_its_two_halves():
    momentum = pd.Series({1: 3.0, 2: -2.0, 3: 0.5})
    value = pd.Series({1: -1.0, 2: 2.0, 3: 0.5})
    scores = combine({"momentum": momentum, "value": value}, RULES)

    lower = pd.concat([momentum, value], axis=1).min(axis=1)
    upper = pd.concat([momentum, value], axis=1).max(axis=1)
    assert (scores >= lower - 1e-12).all()
    assert (scores <= upper + 1e-12).all()


def test_a_stock_good_at_both_beats_a_stock_good_at_one():
    scores = combine({
        "momentum": pd.Series({"оба": 1.5, "один": 3.0}),
        "value": pd.Series({"оба": 1.5, "один": -1.0}),
    }, RULES)
    assert scores["оба"] > scores["один"]
