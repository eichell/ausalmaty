"""Физическое разделение выборки и замок hold-out (ТЗ 9.1)."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from factorbot.data import periods, pit
from factorbot.data.schema import create_all


def _prices(rows: list[tuple[int, date, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"permaticker": pt, "ticker": "AAA", "date": d, "open": px, "high": px,
         "low": px, "close": px, "closeadj": px, "volume": 1000.0,
         "dollar_volume": px * 1000.0}
        for pt, d, px in rows
    ])


def _fundamental(reportperiod: date, available_from: date, netinc: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "permaticker": 1001, "ticker": "AAA", "dimension": "ART",
        "reportperiod": reportperiod, "calendardate": reportperiod,
        "available_from": available_from, "revenue_ttm": 100.0, "netinc_ttm": netinc,
        "opcf_ttm": 10.0, "capex_ttm": -2.0, "equity": None, "debt": None,
        "cash": None, "assets": None, "sharesbas": None,
    }])


@pytest.fixture
def full_db(tmp_path):
    """Полная база: по одной цене на год с 2011 по 2021 и три отчёта."""
    path = tmp_path / "full.duckdb"
    conn = duckdb.connect(str(path))
    create_all(conn)

    prices = _prices([(1001, date(y, 6, 1), float(y)) for y in range(2011, 2022)])
    conn.register("_p", prices)
    conn.execute("INSERT INTO prices SELECT * FROM _p")
    conn.unregister("_p")

    for rp, af, ni in [
        (date(2011, 3, 31), date(2011, 5, 10), 11.0),
        (date(2015, 3, 31), date(2015, 5, 10), 15.0),
        (date(2021, 3, 31), date(2021, 5, 10), 21.0),
    ]:
        pit.load_fundamentals(conn, _fundamental(rp, af, ni))

    conn.close()
    return path


def test_split_writes_one_file_per_period(full_db, tmp_path):
    written = periods.split_database(full_db, tmp_path / "out")
    assert set(written) == {"in_sample", "validation", "holdout"}
    for path in written.values():
        assert path.exists()


def test_holdout_prices_stay_out_of_the_in_sample_file(full_db, tmp_path):
    """Главное требование ТЗ 9.1: 2020+ не попадает в файл разработки."""
    out = periods.split_database(full_db, tmp_path / "out")
    conn = duckdb.connect(str(out["in_sample"]), read_only=True)
    try:
        years = {r[0].year for r in conn.execute("SELECT date FROM prices").fetchall()}
    finally:
        conn.close()
    assert max(years) == 2012
    assert not {2013, 2020, 2021} & years


def test_warmup_brings_prior_prices_so_momentum_is_computable(full_db, tmp_path):
    """Валидация начинается в 2013, но 252 дня истории должны лежать рядом."""
    out = periods.split_database(full_db, tmp_path / "out", warmup_days=430)
    conn = duckdb.connect(str(out["validation"]), read_only=True)
    try:
        years = {r[0].year for r in conn.execute("SELECT date FROM prices").fetchall()}
    finally:
        conn.close()
    assert 2012 in years, "нет разогрева: momentum на первой ребалансировке не посчитать"
    assert max(years) == 2019


def test_fundamentals_keep_full_history_below_the_upper_bound(full_db, tmp_path):
    """Нижней границы у отчётности нет: иначе давно не отчитавшаяся компания
    исчезает из вселенной из-за одного лишь разрезания файлов."""
    out = periods.split_database(full_db, tmp_path / "out")
    conn = duckdb.connect(str(out["validation"]), read_only=True)
    try:
        df = pit.get_fundamentals(conn, date(2019, 12, 31), "ART")
        available = conn.execute("SELECT count(*) FROM fundamentals").fetchone()[0]
    finally:
        conn.close()
    assert available == 2, "отчёты 2011 и 2015 должны остаться, отчёт 2021 — нет"
    assert df.iloc[0]["netinc_ttm"] == pytest.approx(15.0)


def test_future_fundamentals_never_reach_an_earlier_period_file(full_db, tmp_path):
    out = periods.split_database(full_db, tmp_path / "out")
    conn = duckdb.connect(str(out["in_sample"]), read_only=True)
    try:
        rows = conn.execute("SELECT max(available_from) FROM fundamentals").fetchone()[0]
    finally:
        conn.close()
    assert rows <= date(2012, 12, 31)


# --------------------------------------------------------------------------- #
# Замок
# --------------------------------------------------------------------------- #


def test_holdout_cannot_be_opened_by_accident(full_db, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    periods.split_database(full_db, out_dir)
    monkeypatch.delenv(periods.HOLDOUT_UNLOCK_ENV, raising=False)
    with pytest.raises(periods.HoldoutLocked):
        periods.open_period("holdout", out_dir)


def test_holdout_opens_when_the_lock_is_removed_on_purpose(full_db, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    periods.split_database(full_db, out_dir)
    monkeypatch.setenv(periods.HOLDOUT_UNLOCK_ENV, "1")
    conn = periods.open_period("holdout", out_dir)
    try:
        years = {r[0].year for r in conn.execute("SELECT date FROM prices").fetchall()}
    finally:
        conn.close()
    assert {2020, 2021} <= years


def test_in_sample_needs_no_key(full_db, tmp_path):
    out_dir = tmp_path / "out"
    periods.split_database(full_db, out_dir)
    conn = periods.open_period("in_sample", out_dir)
    conn.close()


def test_missing_period_file_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="factorbot-build"):
        periods.open_period("in_sample", tmp_path)
