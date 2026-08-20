"""Синхронизация с Alpaca (ТЗ 4.6). Сети нет — площадка подставная."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from factorbot.data.alpaca_sync import liquid_sample, reconcile, sync_tradability
from factorbot.data.schema import create_all

DAYS = pd.bdate_range("2016-01-04", periods=60)


class FakeVenue:
    """Читает только справочник и бары. Отправки ордеров нет и не должно быть."""

    name = "fake-alpaca"

    def __init__(self, assets: pd.DataFrame, bars: pd.DataFrame | None = None) -> None:
        self._assets = assets
        self._bars = bars if bars is not None else pd.DataFrame()
        self.requested: list[str] = []

    def tradable_assets(self) -> pd.DataFrame:
        return self._assets.copy()

    def daily_bars(self, symbols, start, end) -> pd.DataFrame:
        self.requested = list(symbols)
        return self._bars.copy()


ASSETS = pd.DataFrame([
    {"symbol": "AAA", "status": "active", "tradable": True, "fractionable": True},
    {"symbol": "BBB", "status": "active", "tradable": True, "fractionable": False},
    {"symbol": "OTCX", "status": "inactive", "tradable": False, "fractionable": False},
])


def _securities(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permaticker": r["permaticker"], "ticker": r["ticker"], "name": "X",
        "exchange": "NYSE", "sector": "Technology", "industry": "Y", "siccode": "1",
        "category": "Domestic Common Stock", "is_delisted": r.get("delisted", False),
        "first_price_date": DAYS[0].date(), "last_price_date": DAYS[-1].date(),
    } for r in rows])


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    create_all(c)
    yield c
    c.close()


def _load_securities(conn, securities: pd.DataFrame) -> None:
    conn.register("_s", securities)
    conn.execute("INSERT INTO securities SELECT * FROM _s")
    conn.unregister("_s")


def _load_prices(conn, series: dict[int, list[float]], volume: float = 100e6) -> None:
    rows = []
    for permaticker, prices in series.items():
        for day, price in zip(DAYS, prices, strict=True):
            rows.append({
                "permaticker": permaticker, "ticker": f"T{permaticker}",
                "date": day.date(), "open": price, "high": price, "low": price,
                "close": price, "closeadj": price, "close_unadj": price,
                "volume": 1e6, "dollar_volume": volume,
            })
    conn.register("_p", pd.DataFrame(rows))
    conn.execute("INSERT INTO prices SELECT * FROM _p")
    conn.unregister("_p")


# --------------------------------------------------------------------------- #
# Карта торгуемости (ТЗ 4.6.1)
# --------------------------------------------------------------------------- #


def test_map_is_written_to_the_database(conn):
    _load_securities(conn, _securities([
        {"permaticker": 1, "ticker": "AAA"}, {"permaticker": 2, "ticker": "OTCX"},
    ]))
    sync_tradability(conn, FakeVenue(ASSETS), checked_at=date(2024, 1, 2))

    stored = conn.execute("SELECT * FROM alpaca_map ORDER BY permaticker").df()
    assert len(stored) == 2
    assert bool(stored.set_index("permaticker").loc[1, "tradable"])
    assert not bool(stored.set_index("permaticker").loc[2, "tradable"])


def test_unavailable_names_stay_in_the_map(conn):
    """ТЗ 4.6.1: их надо отсеивать осознанно и логировать, а не терять."""
    _load_securities(conn, _securities([{"permaticker": 2, "ticker": "OTCX"}]))
    mapping = sync_tradability(conn, FakeVenue(ASSETS))
    assert len(mapping) == 1
    assert not bool(mapping.iloc[0]["tradable"])


def test_delisted_security_never_reaches_the_map(conn):
    """Символ ушедшей компании достаётся другой. Сопоставив их, мы получили бы
    карту, по которой заявка уходит не в ту бумагу (ТЗ 4.5)."""
    _load_securities(conn, _securities([
        {"permaticker": 1, "ticker": "AAA"},
        {"permaticker": 99, "ticker": "AAA", "delisted": True},
    ]))
    mapping = sync_tradability(conn, FakeVenue(ASSETS))
    assert list(mapping["permaticker"]) == [1]


def test_rerun_updates_instead_of_duplicating(conn):
    _load_securities(conn, _securities([{"permaticker": 1, "ticker": "AAA"}]))
    sync_tradability(conn, FakeVenue(ASSETS), checked_at=date(2024, 1, 2))
    sync_tradability(conn, FakeVenue(ASSETS), checked_at=date(2024, 6, 1))

    stored = conn.execute("SELECT * FROM alpaca_map").df()
    assert len(stored) == 1
    assert pd.Timestamp(stored.iloc[0]["checked_at"]).date() == date(2024, 6, 1)


def test_empty_catalogue_is_an_error(conn):
    with pytest.raises(RuntimeError, match="factorbot-build"):
        sync_tradability(conn, FakeVenue(ASSETS))


# --------------------------------------------------------------------------- #
# Выборка для сверки
# --------------------------------------------------------------------------- #


def test_sample_takes_the_most_liquid_first(conn):
    _load_securities(conn, _securities([
        {"permaticker": 1, "ticker": "AAA"}, {"permaticker": 2, "ticker": "BBB"},
    ]))
    _load_prices(conn, {1: [10.0] * len(DAYS)}, volume=10e6)
    _load_prices(conn, {2: [20.0] * len(DAYS)}, volume=500e6)
    sync_tradability(conn, FakeVenue(ASSETS))

    sample = liquid_sample(conn, DAYS[0].date(), DAYS[-1].date(), limit=1)
    assert list(sample["permaticker"]) == [2]


def test_untradable_names_are_not_sampled(conn):
    _load_securities(conn, _securities([{"permaticker": 2, "ticker": "OTCX"}]))
    _load_prices(conn, {2: [20.0] * len(DAYS)})
    sync_tradability(conn, FakeVenue(ASSETS))
    assert liquid_sample(conn, DAYS[0].date(), DAYS[-1].date(), limit=10).empty


# --------------------------------------------------------------------------- #
# Сверка цен (ТЗ 4.6.3)
# --------------------------------------------------------------------------- #


def _bars(symbol: str, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": symbol, "date": [d.date() for d in DAYS], "close": prices,
    })


def _prepared(conn, sep: list[float], alpaca: list[float]):
    _load_securities(conn, _securities([{"permaticker": 1, "ticker": "AAA"}]))
    _load_prices(conn, {1: sep})
    sync_tradability(conn, FakeVenue(ASSETS))
    return FakeVenue(ASSETS, _bars("AAA", alpaca))


def test_matching_series_produce_no_findings(conn):
    series = [10.0 * 1.001**i for i in range(len(DAYS))]
    venue = _prepared(conn, series, [v * 5 for v in series])   # другой уровень цен
    assert reconcile(conn, venue, start=DAYS[0].date()).empty


def test_unprocessed_corporate_action_shows_up(conn):
    """Ровно та ошибка, ради которой сверка и существует."""
    sep = [10.0] * len(DAYS)
    alpaca = [10.0] * 30 + [5.0] * (len(DAYS) - 30)      # у них сплит учтён, у нас нет
    venue = _prepared(conn, sep, alpaca)

    findings = reconcile(conn, venue, start=DAYS[0].date())
    assert len(findings) == 1
    assert findings.iloc[0]["rel_diff"] > 0.4


def test_only_the_sampled_symbols_are_requested(conn):
    series = [10.0] * len(DAYS)
    venue = _prepared(conn, series, series)
    reconcile(conn, venue, start=DAYS[0].date(), limit=5)
    assert venue.requested == ["AAA"]


def test_dates_before_the_overlap_are_lifted(conn, caplog):
    """История Alpaca начинается в 2016 году (ТЗ 4.6)."""
    series = [10.0] * len(DAYS)
    venue = _prepared(conn, series, series)
    with caplog.at_level("WARNING"):
        reconcile(conn, venue, start=date(2005, 1, 1))
    assert any("2016" in r.getMessage() for r in caplog.records)


def test_reconcile_without_a_map_says_what_to_run(conn):
    _load_securities(conn, _securities([{"permaticker": 1, "ticker": "AAA"}]))
    _load_prices(conn, {1: [10.0] * len(DAYS)})
    with pytest.raises(RuntimeError, match="alpaca_sync map"):
        reconcile(conn, FakeVenue(ASSETS), start=DAYS[0].date())
