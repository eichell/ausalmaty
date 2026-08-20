"""Сверка с контрольным источником DAILY (ТЗ 4.4)."""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest
from helpers import make_panel, trading_days

from factorbot.data import pit
from factorbot.data.daily_check import (
    NAME_TOLERANCE,
    compare_on_date,
    sample_dates,
    their_multiples,
)
from factorbot.data.schema import create_all

DAYS = trading_days(400, start="2004-01-01")
AS_OF = DAYS[-1]
PRICE = 10.0
SHARES = 100.0                       # капитализация = 1000 у каждой бумаги
NAMES = list(range(1001, 1021))


def _reports(netinc: dict[int, float]) -> pd.DataFrame:
    rows = []
    for permaticker in NAMES:
        rows.append({
            "permaticker": permaticker, "ticker": f"T{permaticker}", "dimension": "ART",
            "reportperiod": date(2004, 3, 31), "calendardate": date(2004, 3, 31),
            "available_from": date(2004, 5, 10), "revenue_ttm": 2000.0,
            "netinc_ttm": netinc[permaticker], "opcf_ttm": 100.0, "capex_ttm": -10.0,
            "equity": None, "debt": None, "cash": None, "assets": None, "sharesbas": None,
        })
        rows.append({
            "permaticker": permaticker, "ticker": f"T{permaticker}", "dimension": "ARQ",
            "reportperiod": date(2004, 3, 31), "calendardate": date(2004, 3, 31),
            "available_from": date(2004, 5, 10), "revenue_ttm": None, "netinc_ttm": None,
            "opcf_ttm": None, "capex_ttm": None, "equity": 500.0, "debt": 0.0,
            "cash": 0.0, "assets": None, "sharesbas": SHARES,
        })
    return pd.DataFrame(rows)


def _conn(control: pd.DataFrame, netinc: dict[int, float] | None = None):
    conn = duckdb.connect(":memory:")
    create_all(conn)
    pit.load_fundamentals(conn, _reports(netinc or dict.fromkeys(NAMES, 100.0)))
    if not control.empty:
        conn.register("_d", control)
        conn.execute("INSERT INTO daily_control SELECT * FROM _d")
        conn.unregister("_d")
    return conn


def _control(**overrides) -> pd.DataFrame:
    """Контрольные значения, по умолчанию совпадающие с нашими."""
    base = {"marketcap": PRICE * SHARES, "ev": 0.0,
            "pe": (PRICE * SHARES) / 100.0, "pb": (PRICE * SHARES) / 500.0,
            "ps": (PRICE * SHARES) / 2000.0}
    base.update(overrides)
    return pd.DataFrame([
        {"permaticker": p, "date": AS_OF.date(), **base} for p in NAMES
    ])


def _panel():
    return make_panel({p: [PRICE] * len(DAYS) for p in NAMES}, DAYS)


def _by_metric(verdicts):
    return {v.metric: v for v in verdicts}


# --------------------------------------------------------------------------- #
# Согласие
# --------------------------------------------------------------------------- #


def test_matching_values_raise_no_flags():
    conn = _conn(_control())
    try:
        verdicts = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))
    finally:
        conn.close()

    assert set(verdicts) == {"marketcap", "pe", "pb", "ps"}
    assert not any(v.systematic for v in verdicts.values())
    assert verdicts["pb"].median_rel_diff == pytest.approx(0.0)


def test_scattered_disagreement_is_not_called_systematic():
    """Расхождение по отдельным бумагам объясняется разницей определений:
    у поставщика другая прибыль, другой капитал, возможно разводнённые акции."""
    control = _control()
    rng = np.random.default_rng(0)
    noise = rng.normal(1.0, 0.25, len(control))    # разброс без сдвига
    control["pb"] = control["pb"] * noise

    conn = _conn(control)
    try:
        verdict = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))["pb"]
    finally:
        conn.close()

    assert not verdict.systematic
    assert verdict.share_beyond > 0.3, "хвост должен быть виден"


# --------------------------------------------------------------------------- #
# Систематический сдвиг — то, ради чего сверка существует
# --------------------------------------------------------------------------- #


def test_shifted_median_is_flagged():
    """Сдвиг всей выборки разницей определений не объясняется (ТЗ 4.4)."""
    control = _control()
    control["pb"] = control["pb"] * 1.30

    conn = _conn(control)
    try:
        verdict = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))["pb"]
    finally:
        conn.close()

    assert verdict.systematic
    assert verdict.median_rel_diff < -0.2
    assert "искать ошибку у себя" in verdict.summary()


def test_small_shift_stays_below_the_threshold():
    control = _control()
    control["ps"] = control["ps"] * 1.02
    conn = _conn(control)
    try:
        verdict = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))["ps"]
    finally:
        conn.close()
    assert not verdict.systematic


# --------------------------------------------------------------------------- #
# Единицы измерения
# --------------------------------------------------------------------------- #


def test_constant_factor_of_a_million_is_units_not_an_error():
    """У поставщика капитализация в миллионах, у нас в долларах. Без этой
    проверки первый прогон сообщил бы о расхождении в миллион раз."""
    control = _control(marketcap=(PRICE * SHARES) / 1e6)
    conn = _conn(control)
    try:
        verdict = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))["marketcap"]
    finally:
        conn.close()

    assert verdict.unit_ratio == pytest.approx(1e6)
    assert not verdict.systematic
    assert "единицы измерения, а не ошибка" in verdict.summary()


def test_a_scattered_million_fold_gap_is_not_treated_as_units():
    """Единицы дают постоянное отношение. Разнобой в миллион раз — это ошибка."""
    control = _control()
    rng = np.random.default_rng(1)
    control["marketcap"] = control["marketcap"] / 1e6 * rng.uniform(0.5, 2.0, len(control))

    conn = _conn(control)
    try:
        verdict = _by_metric(compare_on_date(conn, _panel(), AS_OF, NAMES))["marketcap"]
    finally:
        conn.close()

    assert verdict.unit_ratio is None
    assert verdict.systematic


# --------------------------------------------------------------------------- #
# Пропуски и выборка дат
# --------------------------------------------------------------------------- #


def test_loss_making_company_has_no_price_to_earnings():
    """При отрицательной прибыли обратное отношение бессмысленно — ровно то,
    ради чего ТЗ 6.2 требует считать доходности."""
    netinc = dict.fromkeys(NAMES, 100.0)
    netinc[1001] = -50.0
    conn = _conn(_control(), netinc=netinc)
    try:
        from factorbot.data.daily_check import our_multiples
        ours = our_multiples(conn, _panel(), AS_OF, NAMES)
    finally:
        conn.close()
    assert pd.isna(ours.loc[1001, "pe"])
    assert ours.loc[1002, "pe"] == pytest.approx(10.0)


def test_without_control_data_there_is_nothing_to_compare(caplog):
    conn = _conn(pd.DataFrame())
    try:
        with caplog.at_level("WARNING"):
            assert compare_on_date(conn, _panel(), AS_OF, NAMES) == []
    finally:
        conn.close()
    assert any("сверять нечего" in r.getMessage() for r in caplog.records)


def test_control_query_returns_only_the_requested_date():
    conn = _conn(_control())
    try:
        assert len(their_multiples(conn, AS_OF)) == len(NAMES)
        assert their_multiples(conn, DAYS[0]).empty
    finally:
        conn.close()


def test_dates_are_spread_across_the_whole_history():
    """Одна дата ничего не доказывает: расхождение могло появиться после
    конкретного корпоративного действия."""
    panel = _panel()
    picked = sample_dates(panel, 5)
    assert len(picked) == 5
    assert picked[0] < picked[-1]
    assert picked[0] in panel.month_end_dates()


def test_sampling_more_than_available_returns_everything():
    panel = _panel()
    assert len(sample_dates(panel, 999)) == len(panel.month_end_dates())


def test_name_tolerance_is_looser_than_the_systematic_threshold():
    """Пороги разного назначения: по бумаге расхождения нормальны, по медиане нет."""
    from factorbot.data.daily_check import SYSTEMATIC_THRESHOLD

    assert NAME_TOLERANCE > SYSTEMATIC_THRESHOLD
