"""Сквозная сборка базы на подставном поставщике: сети нет, шаги настоящие."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest
from test_sharadar import TICKERS_RAW

from factorbot.data import pit
from factorbot.data.build import build_full_database
from factorbot.data.provider import DataProvider

SEP_RAW = pd.DataFrame([
    {"ticker": "AAA", "date": "2004-06-01", "open": 9.5, "high": 10.5, "low": 9.0,
     "close": 10.0, "closeadj": 5.0, "closeunadj": 10.0, "volume": 1_000.0},
    {"ticker": "AAA", "date": "2010-06-01", "open": 20.0, "high": 21.0, "low": 19.0,
     "close": 20.0, "closeadj": 20.0, "closeunadj": 20.0, "volume": 2_000.0},
    # 2006 год символ не принадлежал никому — строка обязана исчезнуть.
    {"ticker": "AAA", "date": "2006-06-01", "open": 1.0, "high": 1.0, "low": 1.0,
     "close": 1.0, "closeadj": 1.0, "closeunadj": 1.0, "volume": 1.0},
])

SF1_RAW = pd.DataFrame([
    {"ticker": "AAA", "dimension": "ART", "reportperiod": "2004-06-30",
     "calendardate": "2004-06-30", "datekey": "2004-08-10", "lastupdated": "2019-01-01",
     "revenue": 5000.0, "netinc": 412.0, "ncfo": 600.0, "capex": -150.0,
     "equity": 1000.0, "debt": 300.0, "cashneq": 120.0, "assets": 2200.0,
     "sharesbas": 50.0},
    {"ticker": "AAA", "dimension": "MRT", "reportperiod": "2004-06-30",
     "calendardate": "2004-06-30", "datekey": "2004-08-10", "lastupdated": "2019-01-01",
     "revenue": 9999.0, "netinc": 9999.0, "ncfo": 1.0, "capex": -1.0,
     "equity": 1.0, "debt": 1.0, "cashneq": 1.0, "assets": 1.0, "sharesbas": 1.0},
])

ACTIONS_RAW = pd.DataFrame([
    {"ticker": "AAA", "date": "2004-09-01", "action": "split", "value": 2.0},
])

DAILY_RAW = pd.DataFrame([
    {"ticker": "AAA", "date": "2004-06-01", "marketcap": 500.0, "ev": 600.0,
     "pe": 12.0, "pb": 1.5, "ps": 0.9},
])


class FakeProvider(DataProvider):
    name = "fake"
    _tables = {"TICKERS": TICKERS_RAW, "SEP": SEP_RAW, "SF1": SF1_RAW,
               "ACTIONS": ACTIONS_RAW, "DAILY": DAILY_RAW}

    def available_tables(self):
        return tuple(self._tables)

    def fetch_table(self, table: str, *, force: bool = False) -> pd.DataFrame:
        return self._tables[table].copy()


@pytest.fixture
def built(tmp_path):
    path = tmp_path / "full.duckdb"
    counts = build_full_database(FakeProvider(), path)
    return path, counts


def test_pipeline_fills_every_table(built):
    _, counts = built
    assert counts["securities"] == 2
    assert counts["prices"] == 2          # строка из «ничьего» промежутка отброшена
    assert counts["fundamental_rows"] == 1
    assert counts["corp_actions"] == 1
    assert counts["daily_control"] == 1


def test_prices_are_attributed_to_the_right_owner_of_the_symbol(built):
    path, _ = built
    conn = duckdb.connect(str(path), read_only=True)
    try:
        rows = dict(conn.execute("SELECT date, permaticker FROM prices").fetchall())
    finally:
        conn.close()
    assert rows[date(2004, 6, 1)] == 1001
    assert rows[date(2010, 6, 1)] == 2002


def test_most_recent_reported_row_is_absent_from_the_built_database(built):
    path, _ = built
    conn = duckdb.connect(str(path), read_only=True)
    try:
        df = pit.get_fundamentals(conn, date(2004, 9, 1), "ART")
    finally:
        conn.close()
    assert len(df) == 1
    assert df.iloc[0]["netinc_ttm"] == pytest.approx(412.0)


def test_rebuild_replaces_instead_of_appending(tmp_path):
    path = tmp_path / "full.duckdb"
    build_full_database(FakeProvider(), path)
    counts = build_full_database(FakeProvider(), path)
    assert counts["prices"] == 2
