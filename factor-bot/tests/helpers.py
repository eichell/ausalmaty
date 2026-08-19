"""Сборка синтетических панелей для тестов движка.

Ряды пишутся руками под конкретный случай: на реальной выгрузке проверить, что
сплит 2:1 не съел половину импульса, невозможно — там всё сразу.
"""

from __future__ import annotations

import pandas as pd

from factorbot.data.panel import PricePanel


def trading_days(n: int, start: str = "2004-01-01") -> pd.DatetimeIndex:
    """n подряд идущих будних дней."""
    return pd.bdate_range(start=start, periods=n, name="date")


def make_panel(
    closes: dict[int, list[float] | pd.Series],
    days: pd.DatetimeIndex,
    *,
    opens: dict[int, list[float]] | None = None,
    dollar_volume: float | dict[int, float] = 100e6,
) -> PricePanel:
    """Панель из готовых рядов закрытия.

    Открытие по умолчанию равно вчерашнему закрытию: сделка по открытию не должна
    случайно получать сегодняшнее движение, если тест его не задаёт.
    """
    closeadj = pd.DataFrame({p: pd.Series(v, index=days) for p, v in closes.items()})
    closeadj.index.name = "date"

    if opens is None:
        openadj = closeadj.shift(1).fillna(closeadj)
    else:
        openadj = pd.DataFrame({p: pd.Series(v, index=days) for p, v in opens.items()})
    openadj.index.name = "date"

    if isinstance(dollar_volume, dict):
        volume = pd.DataFrame(
            {p: pd.Series(float(dollar_volume.get(p, 100e6)), index=days)
             for p in closeadj.columns}
        )
    else:
        volume = pd.DataFrame(float(dollar_volume), index=days, columns=closeadj.columns)
    volume.index.name = "date"

    return PricePanel(closeadj, openadj, closeadj.copy(), volume)


def make_securities(
    permatickers: list[int],
    *,
    sectors: dict[int, str] | None = None,
    delisted: set[int] | None = None,
    exchange: str = "NYSE",
    category: str = "Domestic Common Stock",
) -> pd.DataFrame:
    """Справочник под панель. Все бумаги — обыкновенные акции основной биржи."""
    sectors = sectors or {}
    delisted = delisted or set()
    return pd.DataFrame([
        {
            "permaticker": p,
            "ticker": f"T{p}",
            "name": f"Company {p}",
            "exchange": exchange,
            "sector": sectors.get(p, "Technology"),
            "industry": "Software",
            "siccode": "7372",
            "category": category,
            "is_delisted": p in delisted,
            "first_price_date": None,
            "last_price_date": None,
        }
        for p in permatickers
    ])


def first_execution_day(days: pd.DatetimeIndex, start: str | pd.Timestamp) -> pd.Timestamp:
    """День первой сделки: следующий торговый после первой даты ребалансировки.

    Сигнал считается в последний торговый день месяца, сделка идёт по открытию
    следующего дня (ТЗ 7) — тесты исполнения должны целиться именно в него.
    """
    start = pd.Timestamp(start)
    as_series = days.to_series()
    month_ends = as_series.groupby([as_series.dt.year, as_series.dt.month]).max()
    signal = min(d for d in month_ends if d >= start)
    return days[days.get_loc(signal) + 1]
