"""Схемы ТЗ 4.7 и граница PIT-доступа ТЗ 4.8."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from factorbot.data import pit
from factorbot.data.schema import create_all


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    create_all(c)
    yield c
    c.close()


EXPECTED_TABLES = {
    "prices", "fundamentals", "securities", "corp_actions",
    "alpaca_map", "universe", "daily_control",
}


def test_create_all_builds_every_table_from_the_spec(conn):
    got = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()}
    assert EXPECTED_TABLES <= got


def test_create_all_is_idempotent(conn):
    create_all(conn)  # повторный вызов на непустой базе не должен падать
    conn.execute("SELECT 1")


def test_prices_key_rejects_two_rows_for_one_day(conn):
    row = [1001, "AAA", date(2005, 3, 31), 10.0, 11.0, 9.0, 10.5, 10.5, 1000.0, 10500.0]
    conn.execute("INSERT INTO prices VALUES (" + ",".join(["?"] * 10) + ")", row)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute("INSERT INTO prices VALUES (" + ",".join(["?"] * 10) + ")", row)


# --------------------------------------------------------------------------- #
# Запись фундаментала идёт только через pit.py (ТЗ 4.8)
# --------------------------------------------------------------------------- #


def _frame(**over) -> pd.DataFrame:
    base = {
        "permaticker": 1001, "ticker": "AAA", "dimension": "ART",
        "reportperiod": date(2005, 3, 31), "calendardate": date(2005, 3, 31),
        "available_from": date(2005, 5, 10), "revenue_ttm": 5000.0,
        "netinc_ttm": 412.0, "opcf_ttm": 600.0, "capex_ttm": -150.0,
        "equity": None, "debt": None, "cash": None, "assets": None, "sharesbas": None,
    }
    base.update(over)
    return pd.DataFrame([base])


def test_load_fundamentals_writes_and_is_readable_through_pit(conn):
    assert pit.load_fundamentals(conn, _frame()) == 1
    df = pit.get_fundamentals(conn, date(2005, 6, 1), "ART")
    assert len(df) == 1
    assert df.iloc[0]["netinc_ttm"] == pytest.approx(412.0)


def test_load_fundamentals_drops_most_recent_reported_dimensions(conn):
    """MR* не должны попадать в базу вообще (ТЗ 4.3)."""
    mixed = pd.concat([_frame(), _frame(dimension="MRT", netinc_ttm=999.0)], ignore_index=True)
    assert pit.load_fundamentals(conn, mixed) == 1
    stored = conn.execute("SELECT DISTINCT dimension FROM fundamentals").fetchall()
    assert stored == [("ART",)]


def test_load_fundamentals_rejects_frame_with_missing_columns(conn):
    with pytest.raises(ValueError, match="колонки"):
        pit.load_fundamentals(conn, _frame().drop(columns=["capex_ttm"]))


def test_load_fundamentals_keeps_last_duplicate_instead_of_failing_the_batch(conn):
    dup = pd.concat([_frame(), _frame(netinc_ttm=500.0)], ignore_index=True)
    assert pit.load_fundamentals(conn, dup) == 1
    df = pit.get_fundamentals(conn, date(2005, 6, 1), "ART")
    assert df.iloc[0]["netinc_ttm"] == pytest.approx(500.0)


def test_copy_between_respects_the_visibility_boundary(conn):
    """Срез для файла периода режется по available_from, как и PIT-выборка (ТЗ 9.1)."""
    early = _frame()
    late = _frame(reportperiod=date(2005, 6, 30), available_from=date(2005, 8, 5),
                  netinc_ttm=120.0)
    pit.load_fundamentals(conn, pd.concat([early, late], ignore_index=True))

    dst = duckdb.connect(":memory:")
    try:
        moved = pit.copy_fundamentals_between(conn, dst, available_from_max=date(2005, 7, 1))
        assert moved == 1
        assert dst.execute("SELECT count(*) FROM fundamentals").fetchone()[0] == 1
    finally:
        dst.close()
