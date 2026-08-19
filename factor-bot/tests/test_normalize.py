"""Нормализация факторов (ТЗ 6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorbot.normalize import normalize_within_sector, rank_to_z, winsorize


def test_winsorize_clips_both_tails():
    values = pd.Series([-1000.0, *range(1, 99), 1000.0], dtype="float64")
    clipped = winsorize(values)
    assert clipped.min() > -1000.0
    assert clipped.max() < 1000.0


def test_rank_to_z_is_symmetric_and_finite_at_the_edges():
    z = rank_to_z(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.isfinite(z).all(), "Φ⁻¹(0) = −∞ — нужен сдвиг на полранга"
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.iloc[0] == pytest.approx(-z.iloc[-1])


def test_rank_beats_zscore_on_an_outlier():
    """Ради этого в ТЗ и стоит ранг: одна экстремальная бумага не должна
    расплющивать шкалу для всех остальных."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    z = rank_to_z(values)
    # Расстояние между соседями остаётся сопоставимым, выброс не съедает шкалу.
    gaps = z.diff().dropna()
    assert gaps.max() / gaps.min() < 3


def test_ordering_is_preserved():
    values = pd.Series({10: 0.5, 20: -0.2, 30: 1.7})
    z = rank_to_z(values)
    assert z.sort_values().index.tolist() == [20, 10, 30]


def test_single_name_sector_gets_the_average_score_not_infinity():
    values = pd.Series({10: 0.5})
    z = rank_to_z(values)
    assert z.loc[10] == pytest.approx(0.0)


def test_missing_values_stay_missing():
    """ТЗ 6.3: пропуски не заполнять."""
    values = pd.Series({10: 1.0, 20: np.nan, 30: 3.0})
    sectors = pd.Series({10: "Tech", 20: "Tech", 30: "Tech"})
    z = normalize_within_sector(values, sectors)
    assert pd.isna(z.loc[20])
    assert z.notna().sum() == 2


def test_normalization_happens_inside_the_sector_not_across_it():
    """Слабейшая бумага сильного сектора не должна обгонять лидера слабого."""
    values = pd.Series({1: 10.0, 2: 11.0, 3: 12.0, 4: 0.1, 5: 0.2, 6: 0.3})
    sectors = pd.Series({1: "Energy", 2: "Energy", 3: "Energy",
                         4: "Utilities", 5: "Utilities", 6: "Utilities"})
    z = normalize_within_sector(values, sectors)

    assert z.loc[3] == pytest.approx(z.loc[6])      # лидеры обоих секторов равны
    assert z.loc[1] == pytest.approx(z.loc[4])      # и аутсайдеры тоже
    assert z.loc[1] < z.loc[6]


def test_unknown_sector_is_a_group_of_its_own_not_a_crash():
    values = pd.Series({1: 1.0, 2: 2.0, 3: 3.0})
    sectors = pd.Series({1: "Tech", 2: None, 3: "Tech"})
    z = normalize_within_sector(values, sectors)
    assert z.notna().all()
