"""Сборка базы: выгрузка Sharadar → нормализация → файлы периодов (ТЗ 4, 9.1).

    python -m factorbot.data.build --config config/strategy.yaml

Порядок шагов не случаен. Сначала справочник: без интервальной карты
ticker → permaticker остальные таблицы соединять нечем (ТЗ 4.5). Затем цены и
отчётность. В конце — разрезание на периоды, потому что до него полная база
существует в одном файле, а после разработка работает только с in-sample.

Скрипт идемпотентен: повторный запуск за те же сутки берёт сырьё из кэша.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import duckdb

from factorbot.config import load_config
from factorbot.data import pit, sharadar
from factorbot.data.periods import split_database
from factorbot.data.schema import create_all

log = logging.getLogger("factorbot.build")


def build_full_database(
    provider: sharadar.SharadarProvider,
    db_path: str | Path,
    *,
    load_daily_control: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Собирает полную базу из сырых таблиц. Возвращает число строк по таблицам."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}
    try:
        create_all(conn)

        tickers_raw = provider.fetch_table("TICKERS", force=force)
        securities = sharadar.normalize_tickers(tickers_raw)
        counts["securities"] = _insert(conn, "securities", securities)

        sep_map = sharadar.build_ticker_map(tickers_raw, "SEP")
        sf1_map = sharadar.build_ticker_map(tickers_raw, "SF1")

        prices = sharadar.normalize_sep(provider.fetch_table("SEP", force=force), sep_map)
        counts["prices"] = _insert(conn, "prices", prices)

        # Фундаментал пишется только через pit.py (ТЗ 4.8).
        sf1 = sharadar.normalize_sf1(provider.fetch_table("SF1", force=force), sf1_map)
        counts["fundamental_rows"] = pit.load_fundamentals(conn, sf1)

        actions = sharadar.normalize_actions(
            provider.fetch_table("ACTIONS", force=force), sep_map
        )
        counts["corp_actions"] = _insert(conn, "corp_actions", actions)

        if load_daily_control:
            daily = sharadar.normalize_daily(
                provider.fetch_table("DAILY", force=force), sep_map
            )
            counts["daily_control"] = _insert(conn, "daily_control", daily)
    finally:
        conn.close()

    return counts


def _insert(conn: duckdb.DuckDBPyConnection, table: str, df) -> int:
    if df is None or df.empty:
        log.warning("%s: нечего писать", table)
        return 0
    conn.register("_chunk", df)
    conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _chunk")
    conn.unregister("_chunk")
    log.info("%s: записано %d строк", table, len(df))
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка базы factorbot (ТЗ 4, 9.1)")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--force", action="store_true", help="игнорировать кэш сырья")
    parser.add_argument("--skip-split", action="store_true",
                        help="не резать на периоды (только полная база)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    provider = sharadar.SharadarProvider(cache_dir=cfg.data.raw_dir)

    log.info("Сборка полной базы: %s", cfg.data.full_db)
    counts = build_full_database(
        provider, cfg.data.full_db,
        load_daily_control=cfg.data.load_daily_control, force=args.force,
    )
    for table, n in counts.items():
        log.info("  %-18s %10d", table, n)

    if not args.skip_split:
        log.info("Разрезание на периоды (ТЗ 9.1), разогрев %d дней", cfg.periods.warmup_days)
        written = split_database(
            cfg.data.full_db, cfg.data.processed_dir, warmup_days=cfg.periods.warmup_days
        )
        for name, path in written.items():
            log.info("  %-12s %s", name, path)
        log.info(
            "Hold-out собран и закрыт. Разработка идёт на %s.",
            Path(cfg.data.processed_dir) / "in_sample.duckdb",
        )

    log.info("Готово: %s", date.today().isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
