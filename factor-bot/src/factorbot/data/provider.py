"""Абстрактный поставщик сырых данных (ТЗ 11).

Смысл абстракции узкий и конкретный: ТЗ 4 разводит исследование (Sharadar) и
исполнение (Alpaca) намеренно, чтобы проект не зависел от одного поставщика.
Интерфейс описывает только загрузку сырых таблиц; нормализация к схемам ТЗ 4.7
живёт в реализации, потому что она у каждого поставщика своя.

Это не попытка предусмотреть будущих поставщиков впрок. Это граница, за которой
`build.py` не знает, откуда пришли данные.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Источник сырых исторических таблиц."""

    #: Имя поставщика для логов и таблицы сверки.
    name: str

    @abstractmethod
    def fetch_table(self, table: str, *, force: bool = False) -> pd.DataFrame:
        """Отдаёт сырую таблицу поставщика целиком.

        Args:
            table: имя таблицы в терминах поставщика.
            force: игнорировать локальный кэш и тянуть заново.
        """

    @abstractmethod
    def available_tables(self) -> tuple[str, ...]:
        """Таблицы, которые этот поставщик умеет отдавать."""


class ExecutionVenue(ABC):
    """Торговая площадка. В исследовательской части не участвует (ТЗ 4.6).

    Отправки ордеров здесь нет и не будет до прохождения проверок раздела 9
    (ТЗ 2). Разрешены только две операции: проверка торгуемости и сверка цен.
    """

    name: str

    @abstractmethod
    def tradable_assets(self) -> pd.DataFrame:
        """Список инструментов с полями tradable/status/fractionable (ТЗ 4.6.1)."""

    @abstractmethod
    def daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Дневные бары для сверки цен на пересекающемся отрезке (ТЗ 4.6.3)."""
