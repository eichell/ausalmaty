"""Сверка наших мультипликаторов с таблицей DAILY (ТЗ 4.4).

DAILY содержит готовые `pe`, `pb`, `ps`, `marketcap`. Брать их напрямую ТЗ
запрещает: неочевидно, на каком измерении они построены, а значит неочевидна их
PIT-чистота, и определения могут не совпадать с нашими. Всё считается самим.

Но как контрольный источник DAILY незаменим. Формулировка ТЗ 4.4 точная:
«если наш `pb` систематически расходится с их — искать ошибку у себя».

Ключевое слово — **систематически**. Расхождение по отдельной бумаге ожидаемо и
ни о чём не говорит: у поставщика другая прибыль, другой капитал, возможно
разводнённые акции вместо базовых. А вот сдвиг медианы по всей выборке объяснить
разницей определений нельзя — так выглядит ошибка в коде.

Поэтому проверка смотрит на медиану относительного расхождения по срезу, а не на
худшие случаи. Отдельно считается доля бумаг за порогом: если медиана на месте, а
хвост толстый, дело в определениях; если уехала медиана — дело в нас.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from factorbot.data.panel import PricePanel
from factorbot.factors.value import compute_yields, market_cap

log = logging.getLogger(__name__)

#: Сдвиг медианы, выше которого расхождение считается систематическим.
#: Порог мягкий: разница определений даёт единицы процентов, ошибка в коде —
#: обычно десятки или разы.
SYSTEMATIC_THRESHOLD = 0.05

#: Порог для отдельной бумаги. Здесь расхождения нормальны, и цифра нужна лишь
#: чтобы отличить «толстый хвост» от «уехала вся выборка».
NAME_TOLERANCE = 0.10

#: Мультипликаторы DAILY и обратные им доходности из ТЗ 6.2.
MULTIPLES = {
    "pe": "earnings_yield",
    "pb": "book_to_market",
    "ps": "sales_yield",
}


@dataclass(frozen=True)
class ControlVerdict:
    """Итог сверки по одному показателю на одну дату."""

    metric: str
    n: int
    median_rel_diff: float
    share_beyond: float
    systematic: bool
    unit_ratio: float | None = None

    def summary(self) -> str:
        if self.n == 0:
            return f"{self.metric}: сравнивать нечего"
        if self.unit_ratio is not None:
            # Разделитель разрядов меняется только в числе: общий replace по
            # строке съедал запятую в самом тексте.
            times = f"{self.unit_ratio:,.0f}".replace(",", "\u00a0")
            return (
                f"{self.metric}: значения отличаются ровно в {times} раз — "
                "это разные единицы измерения, а не ошибка расчёта"
            )
        verdict = (
            "СИСТЕМАТИЧЕСКОЕ РАСХОЖДЕНИЕ — искать ошибку у себя (ТЗ 4.4)"
            if self.systematic else "в пределах разницы определений"
        )
        return (
            f"{self.metric}: {self.n} бумаг, медиана расхождения "
            f"{self.median_rel_diff:+.1%}, за порогом {self.share_beyond:.0%} — {verdict}"
        )


def our_multiples(
    conn: duckdb.DuckDBPyConnection,
    panel: PricePanel,
    as_of: pd.Timestamp,
    permatickers: list[int],
) -> pd.DataFrame:
    """Наши `pe`, `pb`, `ps` и капитализация на дату — из цен и AR-измерений."""
    yields = compute_yields(conn, panel, as_of, permatickers)
    if yields.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=yields.index)
    for multiple, yield_name in MULTIPLES.items():
        values = yields[yield_name]
        # Обратное отношение определено только при положительной доходности:
        # у убыточной компании P/E бессмысленно, ради чего ТЗ 6.2 и требует
        # считать доходности. Здесь оно нужно лишь для сверки с поставщиком.
        out[multiple] = 1.0 / values.where(values > 0)

    caps = market_cap(panel, as_of, _shares(conn, panel, as_of, permatickers))
    out["marketcap"] = caps.reindex(out.index)
    return out


def _shares(
    conn: duckdb.DuckDBPyConnection,
    panel: PricePanel,
    as_of: pd.Timestamp,
    permatickers: list[int],
) -> pd.Series:
    """Число акций из последнего доступного отчёта — только через pit.py."""
    from factorbot.data import pit

    stocks = pit.get_fundamentals(
        conn, pd.Timestamp(as_of).date(), pit.STOCK_DIMENSION, permatickers
    )
    if stocks.empty:
        return pd.Series(dtype="float64")
    return stocks.set_index("permaticker")["sharesbas"]


def their_multiples(
    conn: duckdb.DuckDBPyConnection, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Контрольные значения поставщика на дату."""
    df = conn.execute(
        """
        SELECT permaticker, marketcap, pe, pb, ps
        FROM daily_control
        WHERE date = $as_of
        """,
        {"as_of": pd.Timestamp(as_of).date()},
    ).df()
    return df.set_index("permaticker") if not df.empty else pd.DataFrame()


def compare_on_date(
    conn: duckdb.DuckDBPyConnection,
    panel: PricePanel,
    as_of: pd.Timestamp,
    permatickers: list[int],
) -> list[ControlVerdict]:
    """Сверяет наши показатели с DAILY на одну дату (ТЗ 4.4)."""
    ours = our_multiples(conn, panel, as_of, permatickers)
    theirs = their_multiples(conn, as_of)
    if ours.empty or theirs.empty:
        log.warning("%s: сверять нечего (нет наших значений или DAILY)", as_of.date())
        return []

    verdicts = []
    for metric in ("marketcap", *MULTIPLES):
        if metric not in theirs.columns:
            continue
        pair = pd.DataFrame({
            "ours": ours[metric], "theirs": theirs[metric],
        }).dropna()
        pair = pair.loc[(pair["theirs"] != 0) & np.isfinite(pair["ours"])]
        verdicts.append(_verdict(metric, pair))
    return verdicts


def _verdict(metric: str, pair: pd.DataFrame) -> ControlVerdict:
    if pair.empty:
        return ControlVerdict(metric, 0, float("nan"), float("nan"), False)

    ratio = pair["ours"] / pair["theirs"]
    unit = _unit_mismatch(ratio)
    rel = ratio - 1.0
    median = float(rel.median())

    return ControlVerdict(
        metric=metric,
        n=len(pair),
        median_rel_diff=median,
        share_beyond=float((rel.abs() > NAME_TOLERANCE).mean()),
        systematic=bool(unit is None and abs(median) > SYSTEMATIC_THRESHOLD),
        unit_ratio=unit,
    )


def _unit_mismatch(ratio: pd.Series) -> float | None:
    """Постоянное отношение, кратное тысяче, — это единицы, а не ошибка.

    Капитализация у поставщика обычно в миллионах, у нас в долларах. Без этой
    проверки первый же прогон сообщил бы о расхождении в миллион раз и напугал
    бы на пустом месте.
    """
    median = float(ratio.median())
    if not np.isfinite(median) or median <= 0:
        return None
    # Разброс вокруг медианы должен быть мал: иначе это не единицы, а разнобой.
    spread = float((ratio / median - 1.0).abs().median())
    if spread > 0.01:
        return None
    for power in (1e-6, 1e-3, 1e3, 1e6, 1e9):
        if abs(median / power - 1.0) < 0.01:
            return power
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def sample_dates(panel: PricePanel, n: int) -> list[pd.Timestamp]:
    """Равномерно разбросанные даты ребалансировки — выборка для сверки.

    Одна дата ничего не доказывает: расхождение могло появиться после конкретного
    корпоративного действия. Несколько дат по всей истории показывают, сдвиг это
    или разовый случай.
    """
    months = panel.month_end_dates()
    if len(months) <= n:
        return list(months)
    step = len(months) / n
    return [months[int(i * step)] for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from factorbot.config import load_config
    from factorbot.data.panel import load_price_panel
    from factorbot.data.periods import PERIODS, open_period

    parser = argparse.ArgumentParser(description="Сверка с таблицей DAILY (ТЗ 4.4)")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--period", default="in_sample", choices=sorted(PERIODS))
    parser.add_argument("--dates", type=int, default=8, help="сколько дат проверить")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    period = PERIODS[args.period]
    conn = open_period(args.period, cfg.data.processed_dir)

    try:
        panel = load_price_panel(conn, end=period.end)
        if panel.trading_days.empty:
            print("В базе периода нет цен.")
            return 2

        found = False
        for as_of in sample_dates(panel, args.dates):
            verdicts = compare_on_date(conn, panel, as_of, list(panel.tickers))
            if not verdicts:
                continue
            print(f"\n--- {as_of.date()} ---")
            for verdict in verdicts:
                print("  " + verdict.summary())
                found = found or verdict.systematic
    finally:
        conn.close()

    if found:
        print(
            "\nЕсть систематическое расхождение с контрольным источником. "
            "ТЗ 4.4 на этот счёт однозначно: искать ошибку у себя."
        )
        return 1
    print("\nСистематических расхождений с DAILY не обнаружено.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
