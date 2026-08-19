"""Физическое разделение выборки (ТЗ 9.1).

ТЗ требует не «не смотреть» на hold-out, а не загружать его в память во время
разработки. Дисциплина, которую нельзя нарушить по невнимательности, — это
отдельные файлы плюс явный ключ на открытие последнего из них.

    in_sample   1999–2012   разработка, отладка, любые эксперименты
    validation  2013–2019   не чаще одного раза на крупную версию
    holdout     2020–…      открыть один раз, в самом конце

Разогрев. Первая дата ребалансировки периода требует 252 торговых дня цен позади
себя (ТЗ 6.1), иначе momentum на ней не считается и период начинается с дыры.
Поэтому в файл периода кладутся цены за `warmup_days` до его начала. Утечки это
не создаёт: данные прошлого относительно начала периода, а не будущего.

Фундаментал кладётся целиком до верхней границы, без нижней. Компания, не
отчитывавшаяся два года, обязана остаться в базе со своим последним отчётом и
флагом устаревания, иначе разделение файлов само меняет вселенную.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from factorbot.data import pit
from factorbot.data.schema import create_all

log = logging.getLogger(__name__)

#: Разогрев: 14 месяцев истории цен из ТЗ 5 с запасом на выходные и праздники.
DEFAULT_WARMUP_DAYS = 430

#: Переменная окружения, снимающая замок с hold-out. Значение произвольно —
#: важно, что снять замок можно только осознанно, а не импортом модуля.
HOLDOUT_UNLOCK_ENV = "FACTORBOT_UNLOCK_HOLDOUT"


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date
    locked: bool = False

    def warmup_start(self, warmup_days: int) -> date:
        return self.start - timedelta(days=warmup_days)


PERIODS: dict[str, Period] = {
    "in_sample": Period("in_sample", date(1999, 1, 1), date(2012, 12, 31)),
    "validation": Period("validation", date(2013, 1, 1), date(2019, 12, 31)),
    "holdout": Period("holdout", date(2020, 1, 1), date(2100, 1, 1), locked=True),
}


class HoldoutLocked(RuntimeError):
    """Попытка открыть hold-out без явного снятия замка (ТЗ 9.1)."""


def period_path(out_dir: str | Path, name: str) -> Path:
    return Path(out_dir) / f"{name}.duckdb"


# --------------------------------------------------------------------------- #
# Запись
# --------------------------------------------------------------------------- #


def split_database(
    src_path: str | Path,
    out_dir: str | Path,
    *,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    periods: dict[str, Period] | None = None,
) -> dict[str, Path]:
    """Режет полную базу на файлы периодов. Возвращает пути созданных файлов."""
    periods = periods or PERIODS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = duckdb.connect(str(src_path), read_only=True)
    written: dict[str, Path] = {}

    try:
        for name, period in periods.items():
            target = period_path(out_dir, name)
            if target.exists():
                target.unlink()
            dst = duckdb.connect(str(target))
            try:
                create_all(dst)
                _copy_period(src, dst, period, warmup_days=warmup_days)
            finally:
                dst.close()
            written[name] = target
            log.info("Период %s → %s", name, target)
    finally:
        src.close()

    return written


def _copy_period(
    src: duckdb.DuckDBPyConnection,
    dst: duckdb.DuckDBPyConnection,
    period: Period,
    *,
    warmup_days: int,
) -> None:
    lo = period.warmup_start(warmup_days)
    hi = period.end

    _copy_by_date(src, dst, "prices", lo=lo, hi=hi)
    _copy_by_date(src, dst, "daily_control", lo=lo, hi=hi)
    _copy_by_date(src, dst, "corp_actions", lo=lo, hi=hi)

    # Справочники не датированы по строкам — переносятся целиком.
    for table in ("securities", "alpaca_map"):
        _copy_whole(src, dst, table)

    # Фундаментал — только через pit.py (ТЗ 4.8).
    n = pit.copy_fundamentals_between(src, dst, available_from_max=hi)
    log.debug("%s: перенесено строк фундаментала: %d", period.name, n)


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchall()
    return bool(rows)


def _copy_by_date(
    src: duckdb.DuckDBPyConnection, dst: duckdb.DuckDBPyConnection,
    table: str, *, lo: date, hi: date,
) -> None:
    if not _table_exists(src, table):
        return
    df = src.execute(
        f"SELECT * FROM {table} WHERE date BETWEEN $lo AND $hi", {"lo": lo, "hi": hi}
    ).df()
    if df.empty:
        return
    dst.register("_chunk", df)
    dst.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _chunk")
    dst.unregister("_chunk")


def _copy_whole(
    src: duckdb.DuckDBPyConnection, dst: duckdb.DuckDBPyConnection, table: str
) -> None:
    if not _table_exists(src, table):
        return
    df = src.execute(f"SELECT * FROM {table}").df()
    if df.empty:
        return
    dst.register("_chunk", df)
    dst.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _chunk")
    dst.unregister("_chunk")


# --------------------------------------------------------------------------- #
# Чтение
# --------------------------------------------------------------------------- #


def open_period(
    name: str, out_dir: str | Path, *, read_only: bool = True
) -> duckdb.DuckDBPyConnection:
    """Открывает базу периода. Hold-out требует снятия замка (ТЗ 9.1).

    Raises:
        KeyError: неизвестное имя периода.
        HoldoutLocked: попытка открыть заблокированный период без ключа.
        FileNotFoundError: файл периода не собран.
    """
    period = PERIODS[name]
    if period.locked and not os.environ.get(HOLDOUT_UNLOCK_ENV):
        raise HoldoutLocked(
            f"Период {name!r} закрыт (ТЗ 9.1): открывается один раз, в самом конце.\n"
            f"Осознанное снятие замка: {HOLDOUT_UNLOCK_ENV}=1.\n"
            "Правка кода после просмотра hold-out превращает его в очередной "
            "in-sample и уничтожает единственную честную проверку в проекте."
        )
    path = period_path(out_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"База периода не собрана: {path}. Запустите factorbot-build.")
    if period.locked:
        log.warning("ОТКРЫТ HOLD-OUT (%s). Результат принимается как есть, без правок кода.", path)
    return pit.connect(path, read_only=read_only)
