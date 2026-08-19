"""Нормализация Sharadar (ТЗ 4.3, 4.5). Сети здесь нет — только чистые функции.

Главный проверяемый риск — переиспользование тикеров. Символ AAA принадлежит
1001 до 2005 года и 2002 с 2007-го. Соединение по `ticker` склеило бы две разные
компании в один непрерывный ряд цен, и momentum принял бы стык за движение.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factorbot.data import sharadar

TICKERS_RAW = pd.DataFrame([
    # table, permaticker, ticker, окна владения символом
    {"table": "SEP", "permaticker": 1001, "ticker": "AAA", "name": "Alpha Corp",
     "exchange": "NYSE", "sector": "Technology", "industry": "Software",
     "siccode": "7372", "category": "Domestic Common Stock", "isdelisted": "Y",
     "firstpricedate": "1999-01-04", "lastpricedate": "2005-12-30",
     "firstquarter": "1999-03-31", "lastquarter": "2005-09-30"},
    {"table": "SEP", "permaticker": 2002, "ticker": "AAA", "name": "Anew Inc",
     "exchange": "NASDAQ", "sector": "Healthcare", "industry": "Biotech",
     "siccode": "2836", "category": "Domestic Common Stock", "isdelisted": "N",
     "firstpricedate": "2007-01-03", "lastpricedate": "",
     "firstquarter": "2007-03-31", "lastquarter": ""},
    {"table": "SF1", "permaticker": 1001, "ticker": "AAA", "name": "Alpha Corp",
     "exchange": "NYSE", "sector": "Technology", "industry": "Software",
     "siccode": "7372", "category": "Domestic Common Stock", "isdelisted": "Y",
     "firstpricedate": "1999-01-04", "lastpricedate": "2005-12-30",
     "firstquarter": "1999-03-31", "lastquarter": "2005-09-30"},
    {"table": "SF1", "permaticker": 2002, "ticker": "AAA", "name": "Anew Inc",
     "exchange": "NASDAQ", "sector": "Healthcare", "industry": "Biotech",
     "siccode": "2836", "category": "Domestic Common Stock", "isdelisted": "N",
     "firstpricedate": "2007-01-03", "lastpricedate": "",
     "firstquarter": "2007-03-31", "lastquarter": ""},
])


@pytest.fixture
def sep_map():
    return sharadar.build_ticker_map(TICKERS_RAW, "SEP")


# --------------------------------------------------------------------------- #
# Карта тикеров
# --------------------------------------------------------------------------- #


def test_open_interval_for_still_listed_company(sep_map):
    """Бумага без lastpricedate не должна выпадать из карты."""
    row = sep_map.loc[sep_map["permaticker"] == 2002].iloc[0]
    assert row["valid_to"] > pd.Timestamp("2100-01-01")


def test_sf1_map_uses_quarter_window_not_price_window():
    m = sharadar.build_ticker_map(TICKERS_RAW, "SF1")
    row = m.loc[m["permaticker"] == 1001].iloc[0]
    assert row["valid_to"] == pd.Timestamp("2005-09-30")


def test_missing_source_table_is_an_error():
    with pytest.raises(sharadar.SharadarError):
        sharadar.build_ticker_map(TICKERS_RAW, "ACTIONS")


# --------------------------------------------------------------------------- #
# Переиспользование тикера (ТЗ 4.5)
# --------------------------------------------------------------------------- #


def test_reused_symbol_maps_to_different_companies_by_date(sep_map):
    df = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": ["2004-06-01", "2010-06-01"]})
    out = sharadar.attach_permaticker(df, sep_map, date_col="date")
    assert out["permaticker"].tolist() == [1001, 2002]


def test_date_in_the_gap_between_owners_is_dropped(sep_map):
    """2006 год символ не принадлежал никому. Строка не должна достаться никому."""
    df = pd.DataFrame({"ticker": ["AAA"], "date": ["2006-06-01"]})
    assert sharadar.attach_permaticker(df, sep_map, date_col="date").empty


def test_unknown_symbol_is_dropped_not_guessed(sep_map):
    df = pd.DataFrame({"ticker": ["ZZZZ"], "date": ["2004-06-01"]})
    assert sharadar.attach_permaticker(df, sep_map, date_col="date").empty


def test_overlapping_ownership_is_reported_not_silently_joined():
    """Противоречивый справочник обязан остановить загрузку, а не склеить истории."""
    broken = TICKERS_RAW.copy()
    broken.loc[broken["permaticker"] == 2002, "firstpricedate"] = "2000-01-01"
    tmap = sharadar.build_ticker_map(broken, "SEP")
    df = pd.DataFrame({"ticker": ["AAA"], "date": ["2004-06-01"]})
    with pytest.raises(sharadar.SharadarError, match="двум permaticker"):
        sharadar.attach_permaticker(df, tmap, date_col="date")


# --------------------------------------------------------------------------- #
# SEP → prices
# --------------------------------------------------------------------------- #


def test_dollar_volume_uses_unadjusted_price(sep_map):
    sep = pd.DataFrame([{
        "ticker": "AAA", "date": "2004-06-01", "open": 9.5, "high": 10.5, "low": 9.0,
        "close": 10.0, "closeadj": 5.0, "closeunadj": 20.0, "volume": 1_000.0,
    }])
    out = sharadar.normalize_sep(sep, sep_map)
    assert out.iloc[0]["dollar_volume"] == pytest.approx(20_000.0)
    assert out.iloc[0]["closeadj"] == pytest.approx(5.0)


def test_prices_are_deduplicated_per_day(sep_map):
    sep = pd.DataFrame([
        {"ticker": "AAA", "date": "2004-06-01", "open": 1, "high": 1, "low": 1,
         "close": 1, "closeadj": 1, "closeunadj": 1, "volume": 1},
        {"ticker": "AAA", "date": "2004-06-01", "open": 2, "high": 2, "low": 2,
         "close": 2, "closeadj": 2, "closeunadj": 2, "volume": 2},
    ])
    out = sharadar.normalize_sep(sep, sep_map)
    assert len(out) == 1
    assert out.iloc[0]["closeadj"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# SF1 → фундаментал (ТЗ 4.3)
# --------------------------------------------------------------------------- #


def _sf1_row(dimension="ART", **over):
    row = {
        "ticker": "AAA", "dimension": dimension, "reportperiod": "2004-06-30",
        "calendardate": "2004-06-30", "datekey": "2004-08-10", "lastupdated": "2019-01-01",
        "revenue": 5000.0, "netinc": 412.0, "ncfo": 600.0, "capex": -150.0,
        "equity": 1000.0, "debt": 300.0, "cashneq": 120.0, "assets": 2200.0,
        "sharesbas": 50.0,
    }
    row.update(over)
    return row


@pytest.fixture
def sf1_map():
    return sharadar.build_ticker_map(TICKERS_RAW, "SF1")


def test_flow_row_carries_only_flows(sf1_map):
    out = sharadar.normalize_sf1(pd.DataFrame([_sf1_row("ART")]), sf1_map)
    row = out.iloc[0]
    assert row["netinc_ttm"] == pytest.approx(412.0)
    assert pd.isna(row["equity"]) and pd.isna(row["sharesbas"])


def test_stock_row_carries_only_balances(sf1_map):
    out = sharadar.normalize_sf1(pd.DataFrame([_sf1_row("ARQ")]), sf1_map)
    row = out.iloc[0]
    assert row["equity"] == pytest.approx(1000.0)
    assert pd.isna(row["netinc_ttm"]) and pd.isna(row["revenue_ttm"])


def test_most_recent_reported_rows_never_reach_the_database(sf1_map):
    """ТЗ 4.3: MR* — прямой look-ahead, отбрасывается на загрузке."""
    raw = pd.DataFrame([_sf1_row("ART"), _sf1_row("MRT"), _sf1_row("MRQ"), _sf1_row("ARY")])
    out = sharadar.normalize_sf1(raw, sf1_map)
    assert set(out["dimension"]) == {"ART"}


def test_available_from_is_datekey_not_lastupdated(sf1_map):
    """`lastupdated` — момент правки у поставщика, а не публичного раскрытия."""
    out = sharadar.normalize_sf1(pd.DataFrame([_sf1_row()]), sf1_map)
    assert str(out.iloc[0]["available_from"]) == "2004-08-10"


def test_row_without_disclosure_date_is_unusable_and_dropped(sf1_map):
    raw = pd.DataFrame([_sf1_row(), _sf1_row(reportperiod="2004-03-31", datekey=None)])
    out = sharadar.normalize_sf1(raw, sf1_map)
    assert len(out) == 1


def test_capex_sign_is_preserved_as_cash_outflow(sf1_map):
    """Знак поставщика сохраняется: FCF = opcf + capex. См. README, отклонения."""
    out = sharadar.normalize_sf1(pd.DataFrame([_sf1_row()]), sf1_map)
    assert out.iloc[0]["capex_ttm"] == pytest.approx(-150.0)


def test_empty_after_dimension_filter_is_an_error(sf1_map):
    with pytest.raises(sharadar.SharadarError, match="ART/ARQ"):
        sharadar.normalize_sf1(pd.DataFrame([_sf1_row("MRQ")]), sf1_map)


# --------------------------------------------------------------------------- #
# TICKERS → securities
# --------------------------------------------------------------------------- #


def test_securities_has_one_row_per_permaticker_with_delisting_flag():
    out = sharadar.normalize_tickers(TICKERS_RAW)
    assert len(out) == 2
    assert out.set_index("permaticker").loc[1001, "is_delisted"]
    assert not out.set_index("permaticker").loc[2002, "is_delisted"]


def test_securities_ignores_rows_of_other_source_tables():
    """SF1-строки справочника не должны удваивать вселенную."""
    out = sharadar.normalize_tickers(TICKERS_RAW)
    assert out["permaticker"].duplicated().sum() == 0
