"""Momentum (ТЗ 6.1) и обязательные тесты ТЗ 12."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_panel, trading_days

from factorbot.factors.momentum import momentum, momentum_diagnostics


def test_known_answer_on_a_synthetic_series():
    """ТЗ 12: ряд с известным ответом.

    Цена растёт на 1% в день. За окно с t−252 по t−21 это 231 шаг, значит
    mom_12_1 = 1.01**231 − 1.
    """
    days = trading_days(300)
    prices = [100.0 * 1.01**i for i in range(300)]
    panel = make_panel({1001: prices}, days)

    got = momentum(panel, days[-1])[1001]
    assert got == pytest.approx(1.01**231 - 1, rel=1e-9)


def test_last_month_is_skipped_not_included():
    """Пропуск последнего месяца обязателен (ТЗ 6.1): всплеск в последние 21 день
    не должен попадать в сигнал."""
    days = trading_days(300)
    prices = [100.0] * 300
    flat = make_panel({1001: list(prices)}, days)

    spiked = list(prices)
    for i in range(279, 300):          # последние 21 день — рост вдвое
        spiked[i] = 200.0
    spiked_panel = make_panel({1001: spiked}, days)

    assert momentum(flat, days[-1])[1001] == pytest.approx(0.0)
    assert momentum(spiked_panel, days[-1])[1001] == pytest.approx(0.0)


def test_two_for_one_split_is_not_a_fifty_percent_drop():
    """ТЗ 12: обработка сплита 2:1 — momentum не должен зафиксировать −50%.

    Скорректированный ряд сплита не видит по построению. Тест сравнивает его с
    нескорректированным, где цена в середине окна делится пополам: именно так
    выглядела бы ошибка, если бы фактор считали по сырой цене.
    """
    days = trading_days(300)
    adjusted = [100.0] * 300
    raw = [100.0] * 150 + [50.0] * 150

    assert momentum(make_panel({1001: adjusted}, days), days[-1])[1001] == pytest.approx(0.0)
    assert momentum(make_panel({1001: raw}, days), days[-1])[1001] == pytest.approx(-0.5)


def test_short_history_gives_nan_not_a_number():
    """Бумаге без 252 дней истории импульс не приписывается (ТЗ 5, 6.3)."""
    days = trading_days(100)
    panel = make_panel({1001: [100.0 + i for i in range(100)]}, days)
    assert np.isnan(momentum(panel, days[-1])[1001])


def test_short_gap_is_bridged_but_long_one_is_not():
    """Праздник закрывается переносом цены, месячная дыра — нет."""
    days = trading_days(300)
    short_gap = [100.0 * 1.001**i for i in range(300)]
    short_gap[278] = np.nan                       # один пропущенный день
    assert not np.isnan(momentum(make_panel({1001: short_gap}, days), days[-1])[1001])

    long_gap = [100.0 * 1.001**i for i in range(300)]
    for i in range(255, 280):
        long_gap[i] = np.nan
    assert np.isnan(momentum(make_panel({1001: long_gap}, days), days[-1])[1001])


def test_window_shorter_than_the_skip_is_rejected():
    days = trading_days(300)
    panel = make_panel({1001: [100.0] * 300}, days)
    with pytest.raises(ValueError, match="назад во времени"):
        momentum(panel, days[-1], lookback_days=10, skip_days=21)


def test_diagnostics_return_half_year_momentum_and_volatility():
    days = trading_days(300)
    prices = [100.0 * 1.005**i for i in range(300)]
    panel = make_panel({1001: prices}, days)

    diag = momentum_diagnostics(panel, days[-1])
    assert diag.loc[1001, "mom_6_1"] == pytest.approx(1.005**105 - 1, rel=1e-9)
    # Идеально гладкий ряд: дневная доходность постоянна, разброса нет.
    assert diag.loc[1001, "volatility"] == pytest.approx(0.0, abs=1e-12)


def test_flat_series_of_zeros_does_not_divide_by_zero():
    days = trading_days(300)
    panel = make_panel({1001: [0.0] * 300}, days)
    assert pd.isna(momentum(panel, days[-1])[1001])
