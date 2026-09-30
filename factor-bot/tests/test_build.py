"""Сквозная сборка базы на подставном поставщике: сети нет, шаги настоящие."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from test_sharadar import ALL_TICKERS_RAW, TICKERS_RAW

from factorbot.config import load_config
from factorbot.data import pit, sharadar
from factorbot.data.build import benchmark_instruments, build_full_database, preflight
from factorbot.data.provider import DataProvider
from factorbot.universe import UniverseRules, eligible_securities

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
     "calendardate": "2004-06-30", "date": "2004-08-10", "lastupdated": "2019-01-01",
     "revenue": 5000.0, "netinc": 412.0, "ncfo": 600.0, "capex": -150.0,
     "equity": 1000.0, "debt": 300.0, "cashneq": 120.0, "assets": 2200.0,
     "sharesbas": 50.0},
    {"ticker": "AAA", "dimension": "MRT", "reportperiod": "2004-06-30",
     "calendardate": "2004-06-30", "date": "2004-08-10", "lastupdated": "2019-01-01",
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
    _tables = {"tickers": TICKERS_RAW, "stocks": SEP_RAW, "fundamentals": SF1_RAW,
               "actions": ACTIONS_RAW, "daily": DAILY_RAW}

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


# --------------------------------------------------------------------------- #
# Урезанный тариф ключа (ТЗ 4.1, 4.4)
# --------------------------------------------------------------------------- #


class RestrictedProvider(FakeProvider):
    """Ключ без подписки на ACTIONS и DAILY — типичный бесплатный тариф."""

    name = "restricted"
    forbidden = {"actions", "daily"}

    def fetch_table(self, table: str, *, force: bool = False) -> pd.DataFrame:
        if table in self.forbidden:
            raise sharadar.SubscriptionError(f"{table}: нет доступа по тарифу")
        return super().fetch_table(table, force=force)


def test_build_survives_a_restricted_key(tmp_path):
    counts = build_full_database(RestrictedProvider(), tmp_path / "full.duckdb")
    assert counts["prices"] == 2
    assert counts["fundamental_rows"] == 1
    assert counts["corp_actions"] == 0
    assert counts["daily_control"] == 0


def test_missing_actions_is_warned_about_not_swallowed(tmp_path, caplog):
    """Без ACTIONS банкротство выглядит как исчезновение из выборки (ТЗ 4.1)."""
    with caplog.at_level("WARNING"):
        build_full_database(RestrictedProvider(), tmp_path / "full.duckdb")
    assert any("ТЗ 4.1" in r.getMessage() for r in caplog.records)


class _Access:
    def __init__(self, ok: dict[str, bool]) -> None:
        self._ok = ok

    def check_access(self):
        return {t: sharadar.TableAccess(t, self._ok.get(t, True)) for t in sharadar.TABLES}


def test_preflight_passes_when_the_required_tables_are_there():
    access = preflight(_Access({"actions": False, "daily": False}))
    assert access["stocks"].ok
    assert not access["actions"].ok


def test_preflight_refuses_to_start_without_the_required_tables():
    with pytest.raises(sharadar.SubscriptionError, match="sharadar.com"):
        preflight(_Access({"fundamentals": False}))


# --------------------------------------------------------------------------- #
# Частота запросов к поставщику
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


RATE_LIMITED = _Response(429, {"quandl_error": {
    "code": "QELx06",
    "message": "You have exceeded the API speed limit and your account has "
               "temporarily been disabled.",
}})


def test_rate_limit_stops_the_probe_instead_of_hammering(monkeypatch):
    """Бесплатный ключ отключается целиком. Продолжать проверку значит продлевать
    блокировку — и разбираться потом часами."""
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return RATE_LIMITED

    monkeypatch.setattr(sharadar.requests, "get", fake_get)
    provider = sharadar.SharadarProvider(api_key="x", probe_interval_s=0)

    with pytest.raises(sharadar.RateLimitError, match="QELx06"):
        provider.check_access()
    assert len(calls) == 1, "после отказа по частоте запросов больше быть не должно"


def test_subscription_refusal_does_not_stop_the_probe(monkeypatch):
    """Отказ по тарифу касается одной таблицы: остальные проверить надо."""
    forbidden = _Response(403, {"quandl_error": {
        "code": "QEPx05", "message": "You have attempted to view a table you have "
                                     "not subscribed to."}})
    monkeypatch.setattr(sharadar.requests, "get", lambda url, **kw: forbidden)
    provider = sharadar.SharadarProvider(api_key="x", probe_interval_s=0)

    access = provider.check_access()
    assert len(access) == len(sharadar.TABLES)
    assert not any(a.ok for a in access.values())


def test_probe_without_a_key_says_so_instead_of_calling_out(monkeypatch):
    """Без ключа проверка обязана вернуть отказ, а не пойти в сеть.

    Пустая строка в конструкторе — не гарантия: провайдер подставляет ключ из
    окружения. Раньше этот тест делал настоящий запрос к поставщику боевым
    ключом и расходовал его лимит.
    """
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    monkeypatch.delenv("NASDAQ_DATA_LINK_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("тест без ключа не имеет права ходить в сеть")

    monkeypatch.setattr(sharadar.requests, "get", forbidden)

    provider = sharadar.SharadarProvider(api_key="", probe_interval_s=0)
    state = provider.probe_table("stocks")
    assert not state.ok
    assert "SHARADAR_API_KEY" in state.detail


# --------------------------------------------------------------------------- #
# Бенчмарк и защитный актив (ТЗ 7.1, 10)
# --------------------------------------------------------------------------- #

FUND_PRICES_RAW = pd.DataFrame([
    {"ticker": "SPY", "date": "2004-06-01", "open": 111.0, "high": 112.0, "low": 110.0,
     "close": 111.5, "closeadj": 80.0, "closeunadj": 111.5, "volume": 5e7},
    {"ticker": "SHY", "date": "2004-06-01", "open": 81.0, "high": 81.2, "low": 80.9,
     "close": 81.0, "closeadj": 60.0, "closeunadj": 81.0, "volume": 1e6},
])


class FundProvider(FakeProvider):
    """Тот же поставщик, но со справочником фондов и их ценами."""

    _tables = dict(FakeProvider._tables, tickers=ALL_TICKERS_RAW)

    def fetch_fund_prices(self, tickers, *, force: bool = False) -> pd.DataFrame:
        wanted = {t.upper() for t in tickers}
        return FUND_PRICES_RAW.loc[FUND_PRICES_RAW["ticker"].isin(wanted)].copy()


@pytest.fixture
def built_with_funds(tmp_path):
    path = tmp_path / "full.duckdb"
    counts = build_full_database(FundProvider(), path, instruments=["SPY", "SHY"])
    return path, counts


def test_benchmark_lands_in_the_same_price_table_as_stocks(built_with_funds):
    """Иначе панель, режимный фильтр и отчёт читали бы цены из разных мест."""
    path, counts = built_with_funds
    assert counts["fund_prices"] == 2
    conn = duckdb.connect(str(path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT s.ticker FROM prices p JOIN securities s USING (permaticker) "
            "WHERE s.ticker IN ('SPY', 'SHY')"
        ).fetchall()
    finally:
        conn.close()
    assert sorted(r[0] for r in rows) == ["SHY", "SPY"]


def test_benchmark_does_not_enter_the_universe(built_with_funds):
    """ETF отсеивается по категории (ТЗ 5), а не по таблице, из которой приехал."""
    path, _ = built_with_funds
    conn = duckdb.connect(str(path), read_only=True)
    try:
        securities = conn.execute("SELECT * FROM securities").df()
    finally:
        conn.close()
    eligible = eligible_securities(securities, UniverseRules())
    assert 118691 not in set(eligible)
    assert 118012 not in set(eligible)


def test_build_without_instruments_touches_nothing(built):
    """Прежнее поведение сохранено: без списка бумаг шаг не выполняется."""
    _, counts = built
    assert counts["fund_prices"] == 0


def test_benchmark_list_comes_from_the_config_without_duplicates():
    # Путь от файла теста, а не от текущего каталога: pytest запускают и из
    # корня репозитория, и из factor-bot/, и тест не должен зависеть от того,
    # откуда его позвали.
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "strategy.yaml")
    assert benchmark_instruments(cfg) == ["SPY", "SHY"]
