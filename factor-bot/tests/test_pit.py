"""Гейт этапа 1 (ТЗ 12, 13). Пока эти тесты не зелёные, factors/ не создаётся."""

from __future__ import annotations

from datetime import date

import duckdb
import pytest
from conftest import INSERT_SQL

from factorbot.data.pit import (
    FUNDAMENTALS_DDL,
    LookAheadError,
    get_fundamentals,
)


def _row(df, permaticker: int):
    sub = df.loc[df["permaticker"] == permaticker]
    assert len(sub) == 1, f"ожидалась одна строка для {permaticker}, получено {len(sub)}"
    return sub.iloc[0]


# --------------------------------------------------------------------------- #
# 1. Ключевой тест проекта (ТЗ 12)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("as_of", [date(2005, 6, 1), date(2005, 8, 10), date(2005, 9, 1)])
def test_no_records_from_the_future(pit_conn, as_of):
    """Запрос на дату T не возвращает записей с available_from > T."""
    for dim in ("ART", "ARQ"):
        df = get_fundamentals(pit_conn, as_of, dim)
        assert (df["available_from"].dt.date <= as_of).all()


# --------------------------------------------------------------------------- #
# 2. Амендменты
# --------------------------------------------------------------------------- #


def test_amendment_visible_only_after_its_own_filing(pit_conn):
    """До подачи 10-Q/A рынок знал 412.0, после — 388.5."""
    before = get_fundamentals(pit_conn, date(2005, 6, 1), "ART")
    assert _row(before, 1001)["netinc_ttm"] == pytest.approx(412.0)

    after = get_fundamentals(pit_conn, date(2005, 9, 1), "ART")
    assert _row(after, 1001)["netinc_ttm"] == pytest.approx(388.5)


def test_later_period_wins_over_later_amendment(pit_conn):
    """Ловушка сортировки: амендмент к Q1 (2005-08-22) подан позже отчёта за Q2
    (2005-08-05). Сортировка по одному available_from вернула бы устаревший Q1."""
    df = get_fundamentals(pit_conn, date(2005, 9, 1), "ART")
    row = _row(df, 1002)
    assert row["reportperiod"].date() == date(2005, 6, 30)
    assert row["netinc_ttm"] == pytest.approx(120.0)


def test_amendment_applied_when_no_newer_period_exists(pit_conn):
    """Обратная сторона: если свежего периода нет, амендмент всё же применяется."""
    df = get_fundamentals(pit_conn, date(2005, 8, 25), "ART", permatickers=[1001])
    assert _row(df, 1001)["netinc_ttm"] == pytest.approx(388.5)


# --------------------------------------------------------------------------- #
# 3. Коллизии ключа
# --------------------------------------------------------------------------- #


def test_same_available_from_for_two_periods_does_not_break_selection(pit_conn):
    """Опоздавший эмитент раскрыл Q1 и Q2 одним днём. Выбирается более поздний
    отчётный период, выборка не падает и не удваивает строку."""
    df = get_fundamentals(pit_conn, date(2005, 8, 6), "ART")
    row = _row(df, 1003)
    assert row["reportperiod"].date() == date(2005, 6, 30)
    assert row["netinc_ttm"] == pytest.approx(55.0)


def test_primary_key_blocks_exact_duplicate():
    """Полный дубль по четырём полям — ошибка загрузки, а не тихое слияние."""
    conn = duckdb.connect(":memory:")
    conn.execute(FUNDAMENTALS_DDL)
    row = (2001, "ZZZ", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 5, 10),
           1.0, 1.0, 1.0, -1.0, None, None, None, None, None)
    conn.execute(INSERT_SQL, list(row))
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(INSERT_SQL, list(row))
    conn.close()


# --------------------------------------------------------------------------- #
# 4. Пропуски и устаревание
# --------------------------------------------------------------------------- #


def test_missing_dimension_yields_absence_not_nan_row(pit_conn):
    """1004 не отчитывается по ART. Он должен отсутствовать, а не приходить
    строкой из NaN (ТЗ 6.3: пропуски не заполнять)."""
    df = get_fundamentals(pit_conn, date(2005, 9, 1), "ART")
    assert 1004 not in set(df["permaticker"])
    assert not df["netinc_ttm"].isna().any()

    arq = get_fundamentals(pit_conn, date(2005, 9, 1), "ARQ")
    assert 1004 in set(arq["permaticker"])


def test_no_rows_before_first_disclosure(pit_conn):
    """До первой публикации компании в выборке нет вообще."""
    df = get_fundamentals(pit_conn, date(2005, 5, 9), "ART")
    assert set(df["permaticker"]) == {1005}


def test_stale_flag_marks_companies_that_stopped_reporting(pit_conn):
    df = get_fundamentals(pit_conn, date(2005, 9, 1), "ART", staleness_months=9)
    assert _row(df, 1005)["is_stale"]
    assert not _row(df, 1002)["is_stale"]


# --------------------------------------------------------------------------- #
# 5. Защита от MR-измерений (ТЗ 4.3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", ["MRQ", "MRY", "MRT", "arq", "", "ARY"])
def test_non_as_reported_dimension_is_rejected(pit_conn, dim):
    with pytest.raises(ValueError, match="as-reported"):
        get_fundamentals(pit_conn, date(2005, 9, 1), dim)


# --------------------------------------------------------------------------- #
# 6. Пост-условие срабатывает само по себе
# --------------------------------------------------------------------------- #


def test_postcondition_catches_leak_independently_of_query(monkeypatch, pit_conn):
    """Если кто-то изменит SQL и снимет фильтр, assert на выходе всё равно упадёт."""
    import factorbot.data.pit as pit

    broken = pit._SELECT_SQL.replace(
        "available_from <= $as_of", "available_from <= $as_of + INTERVAL 1 YEAR"
    )
    monkeypatch.setattr(pit, "_SELECT_SQL", broken)
    with pytest.raises(LookAheadError):
        pit.get_fundamentals(pit_conn, date(2005, 6, 1), "ART")


def test_one_row_per_permaticker(pit_conn):
    df = get_fundamentals(pit_conn, date(2005, 9, 1), "ART")
    assert not df["permaticker"].duplicated().any()


def test_empty_permaticker_list_returns_empty_frame(pit_conn):
    df = get_fundamentals(pit_conn, date(2005, 9, 1), "ART", permatickers=[])
    assert df.empty
    assert "is_stale" in df.columns
