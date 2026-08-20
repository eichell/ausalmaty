"""Фикстуры для тестов PIT-слоя.

База синтетическая и намеренно крошечная. Тесты на реальной выгрузке SF1
недетерминированы: поставщик правит данные, и «зелёный» тест завтра краснеет
без изменений в коде. Здесь каждая строка написана руками под конкретный случай.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factorbot.data.pit import FUNDAMENTALS_DDL  # noqa: E402

# --------------------------------------------------------------------------- #
# Синтетические данные
# --------------------------------------------------------------------------- #
#
# 1001  амендмент к Q1: до 2005-08-22 рынок знал 412.0, после — 388.5
# 1002  амендмент к Q1 подан ПОЗЖЕ отчёта за Q2 — ловушка для сортировки
# 1003  опоздавший эмитент: два отчётных периода с одним available_from
# 1004  есть только ARQ, ART отсутствует — не должен возвращаться NaN-строкой
# 1005  последний отчёт за Q1 2004 — устаревший на 2005 год
#
# Колонки: permaticker, ticker, dimension, reportperiod, calendardate,
#          available_from, revenue_ttm, netinc_ttm, opcf_ttm, capex_ttm,
#          equity, debt, cash, assets, sharesbas

ROWS: list[tuple] = [
    # --- 1001: оригинал и амендмент одного периода -------------------------
    (1001, "AAA", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 5, 10),
     5000.0, 412.0, 600.0, -150.0, None, None, None, None, None),
    (1001, "AAA", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 8, 22),
     4950.0, 388.5, 590.0, -150.0, None, None, None, None, None),
    (1001, "AAA", "ARQ", date(2005, 3, 31), date(2005, 3, 31), date(2005, 5, 10),
     None, None, None, None, 1000.0, 300.0, 120.0, 2200.0, 50.0),

    # --- 1002: амендмент к Q1 позже, чем оригинал Q2 -----------------------
    (1002, "BBB", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 5, 10),
     800.0, 100.0, 130.0, -20.0, None, None, None, None, None),
    (1002, "BBB", "ART", date(2005, 6, 30), date(2005, 6, 30), date(2005, 8, 5),
     840.0, 120.0, 140.0, -22.0, None, None, None, None, None),
    (1002, "BBB", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 8, 22),
     790.0, 90.0, 125.0, -20.0, None, None, None, None, None),

    # --- 1003: два периода раскрыты одним днём -----------------------------
    (1003, "CCC", "ART", date(2005, 3, 31), date(2005, 3, 31), date(2005, 8, 5),
     300.0, 50.0, 60.0, -10.0, None, None, None, None, None),
    (1003, "CCC", "ART", date(2005, 6, 30), date(2005, 6, 30), date(2005, 8, 5),
     320.0, 55.0, 65.0, -11.0, None, None, None, None, None),

    # --- 1004: только балансовое измерение ---------------------------------
    (1004, "DDD", "ARQ", date(2005, 3, 31), date(2005, 3, 31), date(2005, 5, 10),
     None, None, None, None, 900.0, 400.0, 60.0, 1800.0, 30.0),

    # --- 1005: перестал отчитываться --------------------------------------
    (1005, "EEE", "ART", date(2004, 3, 31), date(2004, 3, 31), date(2004, 5, 10),
     150.0, 10.0, 18.0, -4.0, None, None, None, None, None),
]

INSERT_SQL = "INSERT INTO fundamentals VALUES (" + ", ".join(["?"] * 15) + ")"


@pytest.fixture
def pit_conn() -> duckdb.DuckDBPyConnection:
    """In-memory база с загруженными синтетическими строками."""
    conn = duckdb.connect(":memory:")
    conn.execute(FUNDAMENTALS_DDL)
    conn.executemany(INSERT_SQL, ROWS)
    yield conn
    conn.close()


# --------------------------------------------------------------------------- #
# Изоляция окружения
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Ни один тест не видит боевых ключей и не ходит с ними в сеть.

    Провайдеры по умолчанию берут ключ из окружения, поэтому тест, создавший
    клиент «без ключа», незаметно превращался в живой запрос к поставщику — с
    настоящим ключом, расходуя его лимит. Один такой тест уже успел это сделать.

    Заодно `os.environ` подменяется копией: `load_dotenv` пишет в него напрямую,
    и без копии переменная, выставленная одним тестом, доживала до конца прогона
    и меняла поведение следующих.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in (
        "NASDAQ_DATA_LINK_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY",
        "FACTORBOT_UNLOCK_HOLDOUT",
    ):
        os.environ.pop(key, None)
