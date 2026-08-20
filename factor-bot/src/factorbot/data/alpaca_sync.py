"""Синхронизация с Alpaca: торгуемость и сверка цен (ТЗ 4.6).

    python -m factorbot.data.alpaca_sync map        # /v2/assets → таблица alpaca_map
    python -m factorbot.data.alpaca_sync reconcile  # SEP.closeadj против баров Alpaca

Отправки ордеров здесь нет и не будет до прохождения проверок раздела 9 (ТЗ 2).
Это ограничение задачи, а не техническое: модуль исполнения — отдельная работа с
лимитами на эмитента и стоп-лоссами, и начинать её до честной проверки стратегии
значит торговать тем, что не проверено.

Обе операции пишут дату сверки. Карта торгуемости устаревает: бумаги появляются
и исчезают на площадке, и карта годовой давности — это список того, что торговалось
год назад.
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
from factorbot.data.alpaca import (
    PRICE_TOLERANCE,
    AlpacaVenue,
    build_alpaca_map,
    reconcile_prices,
)
from factorbot.data.schema import create_all

log = logging.getLogger("factorbot.alpaca")

#: Начало отрезка, на котором истории Sharadar и Alpaca пересекаются (ТЗ 4.6.3).
OVERLAP_START = date(2016, 1, 1)

#: Сколько самых ликвидных бумаг сверять по умолчанию. Сверка нужна не ради
#: полноты, а ради обнаружения систематической ошибки; на неликвиде расхождения
#: объясняются отсутствием сделок, а не ошибкой обработки корпоративных действий.
DEFAULT_SAMPLE = 50


def sync_tradability(
    conn: duckdb.DuckDBPyConnection, venue: AlpacaVenue, *, checked_at: date | None = None
) -> pd.DataFrame:
    """Обновляет карту `permaticker` → символ Alpaca (ТЗ 4.6.1).

    Returns:
        Записанная карта. Колонка `tradable` показывает, что заявку по бумаге
        вообще можно отправить.
    """
    securities = conn.execute("SELECT * FROM securities").df()
    if securities.empty:
        raise RuntimeError("В базе нет справочника бумаг: сначала factorbot-build.")

    assets = venue.tradable_assets()
    mapping = build_alpaca_map(securities, assets, checked_at=checked_at or date.today())

    create_all(conn)
    conn.register("_map", mapping)
    conn.execute("INSERT OR REPLACE INTO alpaca_map SELECT * FROM _map")
    conn.unregister("_map")

    tradable = int(mapping["tradable"].sum())
    log.info(
        "Карта Alpaca обновлена: %d бумаг, торгуемых %d, недоступных %d",
        len(mapping), tradable, len(mapping) - tradable,
    )
    return mapping


def liquid_sample(
    conn: duckdb.DuckDBPyConnection, start: date, end: date, limit: int
) -> pd.DataFrame:
    """Самые ликвидные торгуемые бумаги на отрезке — выборка для сверки."""
    return conn.execute(
        """
        SELECT p.permaticker, m.alpaca_symbol, avg(p.dollar_volume) AS adv
        FROM prices p JOIN alpaca_map m USING (permaticker)
        WHERE p.date BETWEEN $start AND $end
          AND m.tradable AND m.alpaca_symbol IS NOT NULL
        GROUP BY p.permaticker, m.alpaca_symbol
        ORDER BY adv DESC
        LIMIT $limit
        """,
        {"start": start, "end": end, "limit": limit},
    ).df()


def reconcile(
    conn: duckdb.DuckDBPyConnection,
    venue: AlpacaVenue,
    *,
    start: date = OVERLAP_START,
    end: date | None = None,
    limit: int = DEFAULT_SAMPLE,
    tolerance: float = PRICE_TOLERANCE,
) -> pd.DataFrame:
    """Сверяет `SEP.closeadj` с барами Alpaca на пересекающемся отрезке (ТЗ 4.6.3).

    Returns:
        Дни с расхождением больше порога. Пустой кадр — ожидаемый результат.
        Непустой означает ошибку обработки корпоративных действий с чьей-то
        стороны, и разбирать её нужно до того, как momentum примет её за сигнал.
    """
    end = end or date.today()
    if start < OVERLAP_START:
        log.warning(
            "История Alpaca начинается в 2016 году (ТЗ 4.6): дата %s поднята до %s",
            start, OVERLAP_START,
        )
        start = OVERLAP_START

    sample = liquid_sample(conn, start, end, limit)
    if sample.empty:
        raise RuntimeError(
            "Нечего сверять: карта Alpaca пуста. Сначала `alpaca_sync map`."
        )

    prices = conn.execute(
        """
        SELECT permaticker, date, closeadj
        FROM prices
        WHERE date BETWEEN $start AND $end
          AND permaticker IN (SELECT UNNEST($ids))
          AND closeadj IS NOT NULL
        ORDER BY permaticker, date
        """,
        {"start": start, "end": end,
         "ids": [int(p) for p in sample["permaticker"]]},
    ).df()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    symbols = [str(s) for s in sample["alpaca_symbol"]]
    log.info("Сверка %d бумаг за %s — %s", len(symbols), start, end)
    bars = venue.daily_bars(symbols, start, end)

    findings = reconcile_prices(prices, bars, sample, tolerance=tolerance)
    if findings.empty:
        log.info("Расхождений больше %.1f%% не найдено", 100 * tolerance)
    return findings


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Синхронизация с Alpaca (ТЗ 4.6)")
    parser.add_argument("command", choices=["map", "reconcile"])
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--db", default=None, help="база; по умолчанию полная из конфига")
    parser.add_argument("--live", action="store_true",
                        help="боевой эндпоинт вместо paper (на чтение справочника)")
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--start", default=OVERLAP_START.isoformat())
    parser.add_argument("--tolerance", type=float, default=PRICE_TOLERANCE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    load_dotenv()
    cfg = load_config(args.config)
    db_path = Path(args.db or cfg.data.full_db)
    if not db_path.exists():
        log.error("База %s не собрана. Сначала factorbot-build.", db_path)
        return 2

    venue = AlpacaVenue(paper=not args.live)
    if not (venue.key_id and venue.secret_key):
        log.error(
            "Не заданы ALPACA_API_KEY_ID и ALPACA_API_SECRET_KEY. "
            "Положите их в .env (файл в git не попадает)."
        )
        return 2

    conn = duckdb.connect(str(db_path))
    try:
        return _dispatch(args, conn, venue)
    except RuntimeError as exc:
        # Незаполненная карта или пустой справочник — ожидаемые состояния,
        # а не сбой программы. Трейсбек тут только прячет подсказку.
        log.error("%s", exc)
        return 2
    finally:
        conn.close()


def _dispatch(args, conn: duckdb.DuckDBPyConnection, venue: AlpacaVenue) -> int:
    if args.command == "map":
        mapping = sync_tradability(conn, venue)
        unavailable = mapping.loc[~mapping["tradable"]]
        print(f"\nВсего бумаг в карте:   {len(mapping)}")
        print(f"Торгуемых на Alpaca:   {int(mapping['tradable'].sum())}")
        print(f"Недоступных:           {len(unavailable)}")
        if len(unavailable):
            print("\nПримеры недоступных (их надо отсеивать перед заявками, ТЗ 4.6.1):")
            print(unavailable.head(10).to_string(index=False))
    else:
        findings = reconcile(
            conn, venue,
            start=date.fromisoformat(args.start),
            limit=args.limit, tolerance=args.tolerance,
        )
        if findings.empty:
            print(f"\nOK: расхождений больше {args.tolerance:.1%} нет.")
        else:
            print(f"\nРасхождения больше {args.tolerance:.1%}: {len(findings)} дней "
                  f"по {findings['permaticker'].nunique()} бумагам")
            print(findings.nlargest(15, "rel_diff").to_string(index=False))
            print(
                "\nЭто сигнал об ошибке обработки корпоративных действий — "
                "у поставщика или у нас (ТЗ 4.6.3). Разобрать до прогонов."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
