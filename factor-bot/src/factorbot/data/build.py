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
import pandas as pd

from factorbot.config import load_config, load_dotenv
from factorbot.data import pit, sharadar
from factorbot.data.periods import PERIODS, period_path, split_database
from factorbot.data.schema import create_all

log = logging.getLogger("factorbot.build")


def build_full_database(
    provider: sharadar.SharadarProvider,
    db_path: str | Path,
    *,
    instruments: list[str] | None = None,
    load_daily_control: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Собирает полную базу из сырых таблиц. Возвращает число строк по таблицам.

    Таблицы ТЗ 4.1 (`ACTIONS`) и ТЗ 4.4 (`DAILY`) на урезанном тарифе могут быть
    недоступны. Загрузка в этом случае продолжается, но пропуск фиксируется в
    логе предупреждением: без `ACTIONS` банкротство выглядит как исчезновение из
    выборки, а не как −100%, и результат завышается систематически.
    """
    instruments = list(instruments or [])
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}
    try:
        create_all(conn)

        tickers_raw = provider.fetch_table("tickers", force=force)
        securities = sharadar.normalize_tickers(tickers_raw)
        counts["securities"] = _insert(conn, "securities", securities)

        sep_map = sharadar.build_ticker_map(tickers_raw, "stocks")
        sf1_map = sharadar.build_ticker_map(tickers_raw, sharadar.FUNDAMENTALS_TABLE)

        counts["prices"] = _insert_streamed(
            conn, "prices", provider, "stocks",
            lambda chunk: sharadar.normalize_sep(chunk, sep_map), force=force,
        )

        # Фундаментал пишется только через pit.py (ТЗ 4.8).
        written = 0
        for chunk in provider.iter_table(sharadar.FUNDAMENTALS_TABLE, force=force):
            written += pit.load_fundamentals(
                conn, sharadar.normalize_sf1(chunk, sf1_map, strict=False)
            )
        if written == 0:
            raise sharadar.SharadarError(
                "В отчётности не оказалось ни одной строки с измерениями ART/ARQ "
                "(ТЗ 4.3): собирать value не из чего."
            )
        counts["fundamental_rows"] = written
        log.info("%s: записано %d строк", sharadar.FUNDAMENTALS_TABLE, written)

        counts.update(load_funds(
            conn, provider, tickers_raw, instruments, force=force,
        ))

        actions_raw = _fetch_optional(provider, "actions", force=force)
        if actions_raw is not None:
            counts["corp_actions"] = _insert(
                conn, "corp_actions", sharadar.normalize_actions(actions_raw, sep_map)
            )
        else:
            counts["corp_actions"] = 0
            log.warning(
                "actions недоступна: delisting returns считать не из чего (ТЗ 4.1). "
                "Бэктест на такой базе завышает доходность value-стратегии."
            )

        if load_daily_control:
            counts["daily_control"] = _insert_streamed(
                conn, "daily_control", provider, "daily",
                lambda chunk: sharadar.normalize_daily(chunk, sep_map),
                force=force, optional=True,
            )
    finally:
        conn.close()

    return counts


def benchmark_instruments(cfg) -> list[str]:
    """Тикеры, которые нужны вне вселенной: бенчмарк и защитный актив.

    Берутся из конфига, а не из списка в коде: если в ТЗ 7.1 поменяется защитный
    актив, база и стратегия не должны разъехаться молча.
    """
    names = [
        cfg.reporting.benchmark,
        cfg.regime_filter.benchmark,
        cfg.regime_filter.risk_off_asset,
    ]
    return list(dict.fromkeys(str(n).upper() for n in names if n))


def load_funds(
    conn: duckdb.DuckDBPyConnection,
    provider: sharadar.SharadarProvider,
    tickers_raw,
    instruments: list[str],
    *,
    force: bool = False,
) -> dict[str, int]:
    """Догружает в базу цены бенчмарка и защитного актива (ТЗ 7.1, 10).

    Они лежат в отдельной таблице поставщика (`funds`, прежний SFP) — `stocks`
    содержит только акции. Без этого шага база собирается целиком и выглядит
    исправной, а SPY в ней нет: сравнение с рынком пропускается предупреждением в
    логе, а режимный фильтр падает на первой же дате.
    """
    if not instruments:
        return {"fund_securities": 0, "fund_prices": 0}

    fund_map = sharadar.build_ticker_map(tickers_raw, sharadar.FUNDS_TABLE)
    securities = sharadar.normalize_tickers(
        tickers_raw, source_table=sharadar.FUNDS_TABLE, only=instruments
    )
    raw = provider.fetch_fund_prices(instruments, force=force)
    prices = sharadar.normalize_funds(raw, fund_map)

    counts = {
        "fund_securities": _insert(conn, "securities", securities),
        "fund_prices": _insert(conn, "prices", prices),
    }
    log.info("Вне вселенной загружены: %s", ", ".join(instruments))
    return counts


def add_funds_everywhere(
    cfg, provider: sharadar.SharadarProvider, *, force: bool = False
) -> dict[str, int]:
    """Догружает бенчмарк и защитный актив в уже собранные базы.

    Отдельный режим, а не пересборка: полная база — четыре гигабайта, и повторное
    разрезание на периоды стоит часа работы и всей оперативной памяти контейнера.
    Две бумаги за тридцать лет — пятнадцать тысяч строк.

    Про hold-out. Файл периода открывается на запись напрямую, минуя замок ТЗ 9.1.
    Замок защищает от чтения результатов, а здесь в файл только добавляются цены
    двух ETF; ни одна цифра оттуда не читается и не попадает в отчёт. Собирать
    hold-out заведомо неполным ради формальности означало бы, что его
    единственный разрешённый прогон пройдёт без режимного фильтра.
    """
    instruments = benchmark_instruments(cfg)
    if not instruments:
        log.warning("В конфиге не задан ни бенчмарк, ни защитный актив: нечего грузить.")
        return {}

    tickers_raw = provider.fetch_table("tickers", force=force)
    fund_map = sharadar.build_ticker_map(tickers_raw, sharadar.FUNDS_TABLE)
    securities = sharadar.normalize_tickers(
        tickers_raw, source_table=sharadar.FUNDS_TABLE, only=instruments
    )
    prices = sharadar.normalize_funds(
        provider.fetch_fund_prices(instruments, force=force), fund_map
    )
    log.info("Загружено из funds: %d строк цен по %s", len(prices), ", ".join(instruments))

    counts: dict[str, int] = {}
    targets: list[tuple[str, Path, date | None, date | None]] = [
        ("full", Path(cfg.data.full_db), None, None)
    ]
    for name, period in PERIODS.items():
        targets.append((
            name,
            period_path(cfg.data.processed_dir, name),
            period.warmup_start(cfg.periods.warmup_days),
            period.end,
        ))

    for name, path, lo, hi in targets:
        if not path.exists():
            log.warning("%s: файла нет (%s), пропускаю", name, path)
            continue
        window = prices
        if lo is not None:
            dates = pd.to_datetime(prices["date"])
            window = prices.loc[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))]
        conn = duckdb.connect(str(path))
        try:
            _insert(conn, "securities", securities, quiet=True)
            counts[name] = _insert(conn, "prices", window, quiet=True)
        finally:
            conn.close()
        log.info("  %-11s %s: +%d строк цен", name, path.name, counts.get(name, 0))

    return counts


def _fetch_optional(provider, table: str, *, force: bool):
    """Таблица, отсутствие которой не останавливает сборку (ТЗ 4.1, 4.4)."""
    try:
        return provider.fetch_table(table, force=force)
    except sharadar.SubscriptionError as exc:
        log.warning("%s пропущена: %s", table, exc)
        return None


def preflight(provider: sharadar.SharadarProvider) -> dict[str, sharadar.TableAccess]:
    """Печатает доступность таблиц и падает, если нет обязательных."""
    access = provider.check_access()
    for table in sharadar.TABLES:
        state = access[table]
        (log.info if state.ok else log.warning)("  %s", state)

    missing = [t for t in sharadar.REQUIRED_TABLES if not access[t].ok]
    if missing:
        raise sharadar.SubscriptionError(
            f"Без этих таблиц собирать нечего: {missing}. "
            "Проверьте подписку на https://sharadar.com/account"
        )
    return access


def _insert_streamed(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    provider,
    source: str,
    normalize,
    *,
    force: bool = False,
    optional: bool = False,
) -> int:
    """Грузит большую таблицу порциями: читает, нормализует, пишет и забывает.

    Полная история цен не помещается в память одним кадром — тридцать лет по
    строке на бумагу на торговый день. Порционная обработка ничего не меняет в
    логике: нормализация построчная, а дубликаты между порциями снимает
    первичный ключ таблицы.
    """
    total = 0
    try:
        chunks = provider.iter_table(source, force=force)
        for chunk in chunks:
            total += _insert(conn, table, normalize(chunk), quiet=True)
    except sharadar.SubscriptionError as exc:
        if not optional:
            raise
        log.warning("%s пропущена: %s", source, exc)
        return 0

    log.info("%s: записано %d строк", table, total)
    return total


def _insert(conn: duckdb.DuckDBPyConnection, table: str, df, *, quiet: bool = False) -> int:
    if df is None or df.empty:
        log.warning("%s: нечего писать", table)
        return 0
    conn.register("_chunk", df)
    conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _chunk")
    conn.unregister("_chunk")
    if not quiet:
        log.info("%s: записано %d строк", table, len(df))
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка базы factorbot (ТЗ 4, 9.1)")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--force", action="store_true", help="игнорировать кэш сырья")
    parser.add_argument("--skip-split", action="store_true",
                        help="не резать на периоды (только полная база)")
    parser.add_argument("--funds-only", action="store_true",
                        help="только догрузить бенчмарк и защитный актив "
                             "в уже собранные базы (ТЗ 7.1, 10)")
    parser.add_argument("--check-access", action="store_true",
                        help="только проверить, какие таблицы отдаёт ключ, и выйти")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    load_dotenv()
    cfg = load_config(args.config)
    provider = sharadar.SharadarProvider(cache_dir=cfg.data.raw_dir)

    log.info("Проверка доступа к таблицам Sharadar (ТЗ 4.2)")
    try:
        access = preflight(provider)
    except sharadar.RateLimitError as exc:
        log.error(
            "Ключ временно отключён поставщиком за частоту запросов: %s\n"
            "Это проходит само. Повторять сразу нельзя — счётчик продлевается.", exc,
        )
        return 3
    except sharadar.SubscriptionError as exc:
        # Тариф ключа — не программная ошибка, трейсбек тут только мешает.
        log.error("%s", exc)
        return 2
    if args.check_access:
        return 0

    if args.funds_only:
        log.info("Догрузка бенчмарка и защитного актива в собранные базы")
        add_funds_everywhere(cfg, provider, force=args.force)
        return 0
    if not all(access[t].ok for t in sharadar.OPTIONAL_TABLES):
        log.warning(
            "Часть таблиц вне тарифа. База соберётся, но полнота данных ниже "
            "требований ТЗ 4.1/4.4 — это ограничение результата, а не мелочь."
        )

    log.info("Сборка полной базы: %s", cfg.data.full_db)
    counts = build_full_database(
        provider, cfg.data.full_db,
        instruments=benchmark_instruments(cfg),
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
