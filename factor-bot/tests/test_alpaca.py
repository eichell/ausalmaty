"""Торгуемость и сверка цен (ТЗ 4.6). Сети нет — только чистые функции."""

from __future__ import annotations

from datetime import date

import pandas as pd

from factorbot.data.alpaca import build_alpaca_map, reconcile_prices

SECURITIES = pd.DataFrame([
    {"permaticker": 1001, "ticker": "AAA"},
    {"permaticker": 1002, "ticker": "BBB"},
    {"permaticker": 1003, "ticker": "OTCX"},   # на Alpaca отсутствует
])

ASSETS = pd.DataFrame([
    {"symbol": "AAA", "status": "active", "tradable": True, "fractionable": True},
    {"symbol": "BBB", "status": "inactive", "tradable": True, "fractionable": False},
])


def test_unavailable_security_stays_in_the_map_as_not_tradable():
    """ТЗ 4.6.1: такие бумаги отсеиваются осознанно и логируются, а не теряются."""
    m = build_alpaca_map(SECURITIES, ASSETS, checked_at=date(2024, 1, 2)).set_index("permaticker")
    assert len(m) == 3
    assert not m.loc[1003, "tradable"]
    assert pd.isna(m.loc[1003, "alpaca_symbol"])


def test_inactive_status_is_not_tradable_even_with_the_flag_set():
    m = build_alpaca_map(SECURITIES, ASSETS, checked_at=date(2024, 1, 2)).set_index("permaticker")
    assert not m.loc[1002, "tradable"]
    assert m.loc[1001, "tradable"]


def test_map_records_the_reconciliation_date():
    m = build_alpaca_map(SECURITIES, ASSETS, checked_at=date(2024, 1, 2))
    assert (m["checked_at"] == date(2024, 1, 2)).all()


# --------------------------------------------------------------------------- #
# Сверка цен (ТЗ 4.6.3)
# --------------------------------------------------------------------------- #


def _sep(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "permaticker": 1001,
        "date": [date(2020, 1, d) for d in range(1, len(closes) + 1)],
        "closeadj": closes,
    })


def _bars(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": "AAA",
        "date": [date(2020, 1, d) for d in range(1, len(closes) + 1)],
        "close": closes,
    })


MAP = pd.DataFrame([{"permaticker": 1001, "alpaca_symbol": "AAA"}])


def test_matching_series_produce_no_findings():
    """Разный уровень цен — законен: базы корректировки у поставщиков разные."""
    bad = reconcile_prices(_sep([10.0, 11.0, 12.1]), _bars([50.0, 55.0, 60.5]), MAP)
    assert bad.empty


def test_unprocessed_split_shows_up_as_a_finding():
    """Alpaca учла сплит 2:1, SEP — нет. Ровно та ошибка, ради которой сверка есть."""
    bad = reconcile_prices(_sep([10.0, 11.0, 22.0]), _bars([10.0, 11.0, 11.0]), MAP)
    assert len(bad) == 1
    assert bad.iloc[0]["date"] == date(2020, 1, 3)


def test_small_divergence_below_tolerance_is_ignored():
    bad = reconcile_prices(_sep([10.0, 11.0]), _bars([10.0, 11.02]), MAP)
    assert bad.empty


def test_no_overlap_returns_empty_frame_with_the_diff_column():
    empty = pd.DataFrame({"symbol": [], "date": [], "close": []})
    out = reconcile_prices(_sep([10.0, 11.0]), empty, MAP)
    assert out.empty
    assert "rel_diff" in out.columns
