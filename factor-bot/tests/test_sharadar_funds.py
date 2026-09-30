"""Постраничное чтение цен фондов (ТЗ 7.1, 10). Транспорт подменён, сети нет.

Проверяется ровно один риск: молча усечённая история бенчмарка. Она не роняет
ничего — SMA-200 посчитается и по половине истории, режимный фильтр заработает,
и неверным будет только результат. Поэтому загрузка обязана либо дойти до конца
истории, либо остановиться с ошибкой.
"""

from __future__ import annotations

import pytest
import requests

from factorbot.data import sharadar


class FakeResponse:
    ok = True
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text


def csv_page(rows: list[tuple[str, float]]) -> str:
    head = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"
    body = [
        f"SPY,{day},{px},{px},{px},{px},1000,{px},{px},2026-09-18" for day, px in rows
    ]
    return "\n".join([head, *body]) + "\n"


@pytest.fixture
def provider(tmp_path):
    return sharadar.SharadarProvider(api_key="test-key", cache_dir=tmp_path)


def test_history_is_read_page_by_page_until_it_ends(provider, monkeypatch):
    """Поставщик вправе ограничить ответ; читаем, пока появляются новые дни."""
    pages = [
        csv_page([("2020-01-03", 3.0), ("2020-01-02", 2.0)]),
        csv_page([("2020-01-02", 2.0), ("2020-01-01", 1.0)]),
        csv_page([("2020-01-01", 1.0)]),
    ]
    seen: list[dict] = []

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        seen.append(dict(params or {}))
        return FakeResponse(pages[len(seen) - 1])

    monkeypatch.setattr(requests, "get", fake_get)
    out = provider._download_fund_history("SPY")

    assert sorted(out["date"]) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    # Вторая и третья страницы запрашиваются с верхней границей по дате.
    assert seen[0] == {"ticker": "SPY"}
    assert seen[1]["date.lte"] == "2020-01-02"


def test_key_goes_in_the_header_not_in_the_query_string(provider, monkeypatch):
    """В строке запроса ключ попал бы в логи прокси и в текст исключений."""
    captured: dict = {}

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        captured["params"] = dict(params or {})
        captured["headers"] = dict(headers or {})
        return FakeResponse(csv_page([("2020-01-01", 1.0)]))

    monkeypatch.setattr(requests, "get", fake_get)
    provider._download_fund_history("SPY")

    assert captured["headers"]["x-api-key"] == "test-key"
    assert "test-key" not in str(captured["params"])


def test_endless_pagination_stops_with_an_error(provider, monkeypatch):
    """Дефект справочника не должен превратиться в бесконечный цикл запросов."""
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        calls["n"] += 1
        # Каждая страница уводит историю на год назад: конца не наступает никогда.
        year = 2020 - calls["n"]
        return FakeResponse(csv_page([(f"{year}-12-31", 1.0), (f"{year}-01-01", 1.0)]))

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(sharadar.SharadarError, match="усечён"):
        provider._download_fund_history("SPY")
    assert calls["n"] == sharadar.MAX_FUND_PAGES


def test_empty_answer_is_an_error_not_an_empty_history(provider, monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: FakeResponse("ticker,date,open,high,low,close,volume,"
                                      "closeadj,closeunadj,lastupdated\n"),
    )
    with pytest.raises(sharadar.SharadarError, match="ни одной строки"):
        provider._download_fund_history("SPY")


def test_downloaded_history_is_cached_on_disk(provider, monkeypatch):
    """Повторный запуск сборки не должен снова идти в сеть."""
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        calls["n"] += 1
        return FakeResponse(csv_page([("2020-01-01", 1.0)]))

    monkeypatch.setattr(requests, "get", fake_get)
    first = provider.fetch_fund_prices(["SPY"])
    after_download = calls["n"]
    second = provider.fetch_fund_prices(["SPY"])

    assert calls["n"] == after_download
    assert len(second) == len(first) == 1


def test_repeated_tickers_are_fetched_once(provider, monkeypatch):
    """SPY стоит и бенчмарком отчёта, и бенчмарком режимного фильтра."""
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        calls["n"] += 1
        return FakeResponse(csv_page([("2020-01-01", 1.0)]))

    monkeypatch.setattr(requests, "get", fake_get)
    out = provider.fetch_fund_prices(["SPY", "spy"])
    assert len(out) == 1


def test_no_key_and_no_cache_says_what_to_do(tmp_path):
    provider = sharadar.SharadarProvider(api_key="", cache_dir=tmp_path)
    with pytest.raises(sharadar.SharadarError, match="SHARADAR_API_KEY"):
        provider.fetch_fund_prices(["SPY"])
