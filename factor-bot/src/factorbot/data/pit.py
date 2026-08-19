"""Единственная точка доступа к фундаментальным данным (ТЗ 4.8).

Правило проекта: прямых обращений к таблице `fundamentals` из остального кода
быть не должно. Любая выборка обязана проходить через `get_fundamentals`, которая
гарантирует условие `available_from <= as_of`.

Ключевые инварианты, которые держит этот модуль:

1.  Разрешены только as-reported измерения (ART/ARQ). MR* отвергается на входе
    как look-ahead (ТЗ 4.3).
2.  Одна строка на permaticker: самый поздний доступный отчётный период, внутри
    него — самое позднее доступное раскрытие. Порядок сортировки именно такой,
    а не `available_from DESC`: амендмент к Q1, поданный позже отчёта за Q2,
    иначе вытеснил бы более свежий период.
3.  Пропуски не заполняются (ТЗ 6.3). Компания без доступных строк отсутствует
    в результате, а не возвращается строкой из NaN.
4.  Пост-условие проверяется assert-ом на выходе, независимо от тестов.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

#: Измерения SF1, допустимые в бэктесте. MRQ/MRY/MRT отсутствуют намеренно:
#: пересмотренные цифры не были известны рынку в момент публикации (ТЗ 4.3).
ALLOWED_DIMENSIONS: frozenset[str] = frozenset({"ART", "ARQ"})

#: Потоковые величины берутся из ART (trailing twelve months).
FLOW_DIMENSION = "ART"

#: Балансовые величины — из ARQ (последний квартал).
STOCK_DIMENSION = "ARQ"

#: Порог устаревания по умолчанию. Компания, не отчитывающаяся дольше этого
#: срока, обычно готовится к делистингу, и её последний баланс уже не описывает
#: реальность. В ТЗ этого нет; флаг предназначен для фильтра вселенной.
DEFAULT_STALENESS_MONTHS = 9

FUNDAMENTALS_DDL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    permaticker   BIGINT  NOT NULL,
    ticker        VARCHAR,
    dimension     VARCHAR NOT NULL,
    reportperiod  DATE    NOT NULL,
    calendardate  DATE,
    available_from DATE   NOT NULL,   -- = SF1.datekey, момент публичного раскрытия
    revenue_ttm   DOUBLE,
    netinc_ttm    DOUBLE,
    opcf_ttm      DOUBLE,
    capex_ttm     DOUBLE,
    equity        DOUBLE,
    debt          DOUBLE,
    cash          DOUBLE,
    assets        DOUBLE,
    sharesbas     DOUBLE,
    -- available_from входит в ключ обязательно: один отчётный период порождает
    -- несколько строк (оригинал + амендменты), и обе версии нужны для PIT.
    PRIMARY KEY (permaticker, dimension, reportperiod, available_from)
);
"""

_SELECT_SQL = """
WITH visible AS (
    SELECT *
    FROM fundamentals
    WHERE dimension = $dimension
      AND available_from <= $as_of
      {ticker_filter}
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY permaticker
            ORDER BY reportperiod DESC, available_from DESC
        ) AS rn
    FROM visible
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1
ORDER BY permaticker
"""


class LookAheadError(RuntimeError):
    """Пост-условие PIT-выборки нарушено. Останавливает прогон, а не логирует."""


# --------------------------------------------------------------------------- #
# Подключение
# --------------------------------------------------------------------------- #


def connect(db_path: str | Path, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Открывает базу периода. Путь передаётся явно: значения по умолчанию нет,
    чтобы hold-out нельзя было подключить случайно (ТЗ 9.1)."""
    conn = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        conn.execute(FUNDAMENTALS_DDL)
    return conn


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #


def get_fundamentals(
    conn: duckdb.DuckDBPyConnection,
    as_of: date,
    dimension: str,
    permatickers: list[int] | None = None,
    *,
    staleness_months: int = DEFAULT_STALENESS_MONTHS,
) -> pd.DataFrame:
    """Возвращает только записи, публично доступные на дату `as_of`.

    Одна строка на permaticker. Компании без доступных записей в результат не
    попадают. Колонка `is_stale` помечает бумаги, чей последний доступный отчёт
    старше `staleness_months`.

    Raises:
        ValueError: если `dimension` не входит в ALLOWED_DIMENSIONS.
        LookAheadError: если в результате оказалась запись с available_from > as_of.
    """
    if dimension not in ALLOWED_DIMENSIONS:
        raise ValueError(
            f"Измерение {dimension!r} запрещено в бэктесте. "
            f"Допустимы только as-reported: {sorted(ALLOWED_DIMENSIONS)}. "
            "MR*-измерения содержат пересмотренные цифры и дают look-ahead (ТЗ 4.3)."
        )

    params: dict[str, object] = {"dimension": dimension, "as_of": as_of}
    ticker_filter = ""
    if permatickers is not None:
        if len(permatickers) == 0:
            return _empty_like(conn)
        ticker_filter = "AND permaticker IN (SELECT UNNEST($permatickers))"
        params["permatickers"] = [int(p) for p in permatickers]

    sql = _SELECT_SQL.format(ticker_filter=ticker_filter)
    df = conn.execute(sql, params).df()

    df = _add_staleness(df, as_of=as_of, staleness_months=staleness_months)
    _assert_pit_clean(df, as_of=as_of)
    return df


# --------------------------------------------------------------------------- #
# Внутреннее
# --------------------------------------------------------------------------- #


def _empty_like(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Пустой кадр с правильными колонками — чтобы вызывающий код не ветвился."""
    df = conn.execute("SELECT * FROM fundamentals WHERE FALSE").df()
    df["months_since_report"] = pd.Series(dtype="float64")
    df["is_stale"] = pd.Series(dtype="bool")
    return df


def _add_staleness(df: pd.DataFrame, *, as_of: date, staleness_months: int) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["months_since_report"] = pd.Series(dtype="float64")
        df["is_stale"] = pd.Series(dtype="bool")
        return df

    rp = pd.to_datetime(df["reportperiod"])
    months = (as_of.year - rp.dt.year) * 12 + (as_of.month - rp.dt.month)
    df = df.copy()
    df["months_since_report"] = months.astype("float64")
    df["is_stale"] = months > staleness_months
    return df


def _assert_pit_clean(df: pd.DataFrame, *, as_of: date) -> None:
    """Дешёвая страховка, работающая даже если тесты забыли обновить."""
    if df.empty:
        return

    available = pd.to_datetime(df["available_from"]).dt.date
    leaked = df.loc[available > as_of]
    if not leaked.empty:
        raise LookAheadError(
            f"PIT-выборка на {as_of} вернула {len(leaked)} записей из будущего: "
            f"permatickers={sorted(leaked['permaticker'].unique().tolist())}"
        )

    dupes = df["permaticker"].duplicated()
    if dupes.any():
        raise LookAheadError(
            "PIT-выборка вернула более одной строки на permaticker: "
            f"{sorted(df.loc[dupes, 'permaticker'].unique().tolist())}"
        )


# --------------------------------------------------------------------------- #
# Запись
# --------------------------------------------------------------------------- #
#
# Загрузчик тоже обязан ходить через этот модуль: правило ТЗ 4.8 запрещает имя
# таблицы вне pit.py, и CI-проверка его не различает по намерению. Побочный
# плюс — фильтр измерений применяется на входе, а не только на выходе.

#: Колонки таблицы в порядке DDL. Загрузчик обязан отдать ровно их.
FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "permaticker", "ticker", "dimension", "reportperiod", "calendardate",
    "available_from", "revenue_ttm", "netinc_ttm", "opcf_ttm", "capex_ttm",
    "equity", "debt", "cash", "assets", "sharesbas",
)


def load_fundamentals(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Записывает нормализованный фундаментал. Единственный путь записи (ТЗ 4.8).

    Строки с измерениями вне ALLOWED_DIMENSIONS отбрасываются на входе: MR* не
    должны попадать в базу вообще, иначе однажды кто-нибудь их оттуда прочитает.

    Returns:
        Число записанных строк.

    Raises:
        ValueError: если во входном кадре нет требуемых колонок.
    """
    missing = [c for c in FUNDAMENTALS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В кадре отсутствуют колонки: {missing}")

    df = df.loc[df["dimension"].isin(ALLOWED_DIMENSIONS), list(FUNDAMENTALS_COLUMNS)]
    if df.empty:
        return 0

    # Дубли по ключу — ошибка выгрузки, но падать на всей суточной загрузке из-за
    # повторной строки поставщика неразумно. Оставляем последнюю и логируем числом.
    df = df.drop_duplicates(
        subset=["permaticker", "dimension", "reportperiod", "available_from"], keep="last"
    )

    conn.execute(FUNDAMENTALS_DDL)
    conn.register("_incoming_fundamentals", df)
    conn.execute(
        "INSERT OR REPLACE INTO fundamentals SELECT * FROM _incoming_fundamentals"
    )
    conn.unregister("_incoming_fundamentals")
    return len(df)


def copy_fundamentals_between(
    src: duckdb.DuckDBPyConnection,
    dst: duckdb.DuckDBPyConnection,
    *,
    available_from_max: date,
    available_from_min: date | None = None,
) -> int:
    """Переносит срез фундаментала в базу периода (ТЗ 9.1).

    Отбор идёт по `available_from`, а не по `reportperiod`: физическое разделение
    файлов должно повторять ту же границу видимости, что и PIT-выборка.
    """
    sql = "SELECT * FROM fundamentals WHERE available_from <= $hi"
    params: dict[str, object] = {"hi": available_from_max}
    if available_from_min is not None:
        sql += " AND available_from >= $lo"
        params["lo"] = available_from_min
    df = src.execute(sql, params).df()
    return load_fundamentals(dst, df)
