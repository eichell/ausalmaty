"""Deflated Sharpe Ratio (ТЗ 9.2.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorbot.report.deflated_sharpe import (
    count_experiments,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_variance_from_trials,
)


def _returns(mean: float, sigma: float, n: int = 2000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sigma, n))


# --------------------------------------------------------------------------- #
# Порог перебора
# --------------------------------------------------------------------------- #


def test_more_trials_raise_the_bar():
    """Максимум двадцати случайных величин выше максимума двух — по устройству."""
    low = expected_max_sharpe(sharpe_variance=0.04, n_trials=2)
    high = expected_max_sharpe(sharpe_variance=0.04, n_trials=200)
    assert high > low > 0


def test_a_single_trial_needs_no_correction():
    assert expected_max_sharpe(sharpe_variance=0.04, n_trials=1) == pytest.approx(0.0)


def test_no_spread_between_trials_means_no_bar():
    assert expected_max_sharpe(sharpe_variance=0.0, n_trials=100) == pytest.approx(0.0)


def test_wider_spread_raises_the_bar():
    narrow = expected_max_sharpe(sharpe_variance=0.01, n_trials=50)
    wide = expected_max_sharpe(sharpe_variance=0.25, n_trials=50)
    assert wide > narrow


# --------------------------------------------------------------------------- #
# Сама поправка
# --------------------------------------------------------------------------- #


def test_strong_result_from_one_trial_survives():
    dsr = deflated_sharpe_ratio(_returns(0.0012, 0.01), n_trials=1)
    assert dsr.probability > 0.95
    assert dsr.survives


def test_the_same_result_after_many_trials_may_not_survive():
    """Один и тот же Sharpe значит разное после одного прогона и после трёхсот."""
    returns = _returns(0.00035, 0.01)
    few = deflated_sharpe_ratio(returns, n_trials=1, sharpe_variance=0.02)
    many = deflated_sharpe_ratio(returns, n_trials=300, sharpe_variance=0.02)
    assert few.probability > many.probability


def test_pure_noise_does_not_survive():
    dsr = deflated_sharpe_ratio(_returns(0.0, 0.01), n_trials=50, sharpe_variance=0.02)
    assert dsr.probability < 0.95
    assert not dsr.survives


def test_negative_skew_lowers_the_verdict():
    """Долго по чуть-чуть, изредка много потерять — обычная форма для акций,
    и обычный Sharpe её переоценивает."""
    rng = np.random.default_rng(3)
    calm = pd.Series(rng.normal(0.0008, 0.01, 2000))

    skewed = calm.copy()
    skewed.iloc[::200] -= 0.06                    # редкие крупные убытки
    skewed += (0.06 / 200)                        # то же среднее

    a = deflated_sharpe_ratio(calm, n_trials=20, sharpe_variance=0.02)
    b = deflated_sharpe_ratio(skewed, n_trials=20, sharpe_variance=0.02)
    assert b.skewness < a.skewness
    assert b.probability < a.probability


def test_too_few_observations_is_an_error():
    with pytest.raises(ValueError, match="три наблюдения"):
        deflated_sharpe_ratio(pd.Series([0.01, 0.02]), n_trials=1)


def test_summary_names_both_numbers_that_matter():
    dsr = deflated_sharpe_ratio(_returns(0.0008, 0.01), n_trials=17, sharpe_variance=0.02)
    text = dsr.summary()
    assert "DSR" in text and "17 испытаниях" in text


# --------------------------------------------------------------------------- #
# Связь с журналом и картой чувствительности
# --------------------------------------------------------------------------- #


def test_trials_are_counted_from_the_journal(tmp_path):
    """ТЗ 9.2.1: вот зачем ведут журнал испытаний."""
    log = tmp_path / "experiments.log"
    log.write_text(
        "# комментарий\n\n2026-01-01 | a | b | c\n2026-01-02 | a | b | c\n",
        encoding="utf-8",
    )
    assert count_experiments(log) == 2


def test_missing_journal_counts_as_one_trial(tmp_path):
    assert count_experiments(tmp_path / "нет.log") == 1


def test_empty_journal_counts_as_one_trial(tmp_path):
    log = tmp_path / "experiments.log"
    log.write_text("# только комментарий\n", encoding="utf-8")
    assert count_experiments(log) == 1


def test_spread_comes_from_the_sensitivity_map():
    sharpes = pd.Series([0.8, 0.9, 0.7, 1.1, 0.6])
    assert sharpe_variance_from_trials(sharpes) == pytest.approx(sharpes.var(ddof=1))


def test_a_single_trial_has_no_spread():
    assert sharpe_variance_from_trials(pd.Series([0.8])) == pytest.approx(0.0)
