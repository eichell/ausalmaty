"""Широкие панели цен для векторизованных расчётов (ТЗ 3).

Бэктестер свой, на pandas: месячная кросс-секционная ребалансировка в готовых
фреймворках прячет ровно те детали, которые здесь и решают достоверность.

Три панели, все проиндексированы торговыми днями, колонки — permaticker:

*   `closeadj` — поправка на сплиты и дивиденды. Единственная база для доходностей
    и для momentum (ТЗ 4.1): на нескорректированной цене сплит 2:1 выглядит как
    обвал на 50%.
*   `openadj` — цена открытия, приведённая тем же множителем. Исполнение идёт по
    открытию следующего дня (ТЗ 7), а `open` у поставщика поправлен только на
    сплиты. Смешивать его с `closeadj` нельзя: разница — это дивидендная
    доходность, которая молча утекла бы в результат.
*   `close` — цена как она была, без поправки на дивиденды. Нужна ровно для одного:
    порога «дороже $5» (ТЗ 5). Скорректированный ряд для этого не годится — он
    пересчитан от сегодняшней базы и в 1999 году показал бы другие деньги.
*   `dollar_volume` — для фильтра ликвидности (ТЗ 5) и выбора ставки издержек (ТЗ 8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PricePanel:
    """Цены в широком виде. Индекс — торговые дни, колонки — permaticker."""

    closeadj: pd.DataFrame
    openadj: pd.DataFrame
    close: pd.DataFrame
    dollar_volume: pd.DataFrame

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self.closeadj.index

    @property
    def tickers(self) -> pd.Index:
        return self.closeadj.columns

    def returns(self) -> pd.DataFrame:
        """Дневные доходности из скорректированной цены."""
        return self.closeadj.pct_change()

    def month_end_dates(self) -> list[pd.Timestamp]:
        """Последние торговые дни месяцев — даты ребалансировки (ТЗ 7)."""
        days = self.trading_days.to_series()
        return list(days.groupby([days.dt.year, days.dt.month]).max().sort_values())

    def next_trading_day(self, day: pd.Timestamp) -> pd.Timestamp | None:
        """День исполнения: следующий торговый после даты ребалансировки (ТЗ 7)."""
        later = self.trading_days[self.trading_days > day]
        return later[0] if len(later) else None


def load_price_panel(
    conn: duckdb.DuckDBPyConnection,
    start: date | None = None,
    end: date | None = None,
    permatickers: list[int] | None = None,
) -> PricePanel:
    """Собирает панели из таблицы цен.

    Границы дат обязательны на практике: в файл периода уже положен разогрев
    (ТЗ 9.1), и брать оттуда больше, чем нужно, незачем.
    """
    where = ["closeadj IS NOT NULL"]
    params: dict[str, object] = {}
    if start is not None:
        where.append("date >= $start")
        params["start"] = start
    if end is not None:
        where.append("date <= $end")
        params["end"] = end
    if permatickers is not None:
        if not permatickers:
            return _empty_panel()
        where.append("permaticker IN (SELECT UNNEST($permatickers))")
        params["permatickers"] = [int(p) for p in permatickers]

    sql = f"""
        SELECT permaticker, date, open, close, closeadj, dollar_volume
        FROM prices
        WHERE {' AND '.join(where)}
        ORDER BY date, permaticker
    """
    df = conn.execute(sql, params).df()
    if df.empty:
        return _empty_panel()

    df["date"] = pd.to_datetime(df["date"])

    # Множитель корректировки: closeadj / close. Дни с нулевой или отсутствующей
    # ценой закрытия множителя не дают — открытие по ним неизвестно.
    ratio = (df["closeadj"] / df["close"]).where(df["close"] > 0)
    df["openadj"] = df["open"] * ratio

    closeadj = df.pivot(index="date", columns="permaticker", values="closeadj")
    openadj = df.pivot(index="date", columns="permaticker", values="openadj")
    # День без цены открытия исполняется по закрытию. Иначе позиция теряет
    # движение этого дня целиком: и до открытия, и после него.
    openadj = openadj.fillna(closeadj)
    close = df.pivot(index="date", columns="permaticker", values="close")
    dollar_volume = df.pivot(index="date", columns="permaticker", values="dollar_volume")

    log.info(
        "Панель цен: %d дней × %d бумаг, %s — %s",
        len(closeadj), closeadj.shape[1],
        closeadj.index.min().date(), closeadj.index.max().date(),
    )
    return PricePanel(closeadj, openadj, close, dollar_volume)


def _empty_panel() -> PricePanel:
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return PricePanel(empty, empty.copy(), empty.copy(), empty.copy())


def load_sectors(conn: duckdb.DuckDBPyConnection) -> pd.Series:
    """permaticker → сектор. Нормализация факторов идёт внутри сектора (ТЗ 6.3)."""
    df = conn.execute("SELECT permaticker, sector FROM securities").df()
    return df.set_index("permaticker")["sector"]
