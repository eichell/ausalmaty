"""Управление риском позиции: стоп-лосс, лимит на бумагу, ограничитель сделок.

**В ТЗ этого нет.** Раздел 7 описывает портфель из равных весов с лимитом на
сектор и буфером, и никаких стоп-лоссов не предусматривает. Всё, что здесь, —
добавка сверх задания, и относиться к ней надо соответственно.

Поэтому весь модуль построен как отключаемый флаг, по умолчанию выключенный, а
результат обязан сравниваться с версией без него — ровно так же, как ТЗ 7.1
требует для режимного фильтра. Защита, которую никто не сравнивал с её
отсутствием, — это лишний параметр и лишняя ось переподгонки, а не защита.

Чего стоп-лосс здесь не делает
------------------------------

Он **не срабатывает внутри дня**. Внутридневные данные и логика вне объёма
работ (ТЗ 2), и притворяться, что мы знаем внутридневной минимум, нельзя.
Условие проверяется по цене закрытия, сделка проходит по открытию следующего дня
— та же дисциплина, что и у всей остальной стратегии (ТЗ 7). На реальном
брокерском стоп-ордере исполнение было бы хуже: он срабатывает по пути вниз.
Значит наша оценка стопа оптимистична, и это надо помнить.

Он **не спасает от гэпа**. Бумага, открывшаяся на 40% ниже, будет продана по
этой цене, а не по уровню стопа. Именно так это работает и в жизни.

Почему порог широкий
--------------------

Стоп в 5–10% на месячной кросс-секционной стратегии режет обычную волатильность
и превращается в генератор оборота. Хуже того, он работает прямо против value:
этот фактор по построению покупает то, что падало. Смысл имеет только широкий
порог, ловящий катастрофу отдельной бумаги — банкротство, мошенничество,
провалившееся испытание препарата, — а не рыночный шум.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

#: Порог, ниже которого стоп начинает ловить обычную волатильность, а не
#: катастрофу. Значение эмпирическое: дневная сигма ликвидной акции — порядка
#: 2%, месячная — около 9%, и стоп в 15% сработает на нормальном движении.
NARROW_STOP_WARNING = 0.15


@dataclass(frozen=True)
class StopLossRules:
    """Фиксированный стоп от цены входа. По умолчанию выключен."""

    enabled: bool = False
    #: Доля падения от цены входа, при которой позиция закрывается.
    threshold: float = 0.30
    #: Сколько месяцев бумага не рассматривается после срабатывания.
    quarantine_months: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError(
                f"Порог стопа {self.threshold} должен лежать между 0 и 1 "
                "(доля падения от цены входа)."
            )
        if self.enabled and self.threshold < NARROW_STOP_WARNING:
            log.warning(
                "Стоп %.0f%% уже месячной волатильности типичной акции: он будет "
                "срабатывать на обычном движении и работать против value, который "
                "по построению покупает падавшее.", 100 * self.threshold,
            )

    @classmethod
    def from_config(cls, cfg) -> StopLossRules:
        return cls(
            enabled=bool(cfg.enabled),
            threshold=float(cfg.threshold),
            quarantine_months=int(cfg.quarantine_months),
        )

    def stop_price(self, entry: float) -> float:
        return entry * (1.0 - self.threshold)

    def triggered(
        self, entries: dict[int, float], closes: pd.Series
    ) -> set[int]:
        """Какие позиции пробили стоп по цене закрытия.

        Цена закрытия, а не минимум дня: внутридневных данных у нас нет (ТЗ 2).
        """
        if not self.enabled:
            return set()
        hit = set()
        for permaticker, entry in entries.items():
            price = closes.get(permaticker)
            if price is not None and pd.notna(price) and price <= self.stop_price(entry):
                hit.add(permaticker)
        return hit

    def quarantine_until(self, day: pd.Timestamp) -> pd.Timestamp:
        """До какой даты бумага не рассматривается к покупке.

        Без карантина стоп превращается в дорогую карусель: продали по стопу,
        купили обратно на ближайшей ребалансировке, заплатили дважды за то, чтобы
        остаться при своих.
        """
        return day + pd.DateOffset(months=self.quarantine_months)


@dataclass(frozen=True)
class PositionLimits:
    """Лимит на одну бумагу.

    При равных весах и портфеле из 30 бумаг доля каждой равна 3.33%, и лимит
    ничего не меняет. Он нужен как предохранитель: если вселенная сузилась и
    бумаг набралось меньше двадцати, вес каждой уедет выше 5%, а концентрация
    вырастет именно тогда, когда рынок к этому меньше всего располагает.
    """

    #: Максимальная доля одной бумаги. None — ограничения нет.
    max_position_weight: float | None = None

    @classmethod
    def from_config(cls, cfg) -> PositionLimits:
        value = cfg.max_position_weight
        return cls(max_position_weight=None if value is None else float(value))

    def apply(self, weights: pd.Series) -> pd.Series:
        """Обрезает веса по лимиту. Остаток уходит в деньги, а не другим бумагам.

        Раздавать остаток означало бы нарушить тот же лимит у соседей или
        превратить портфель в неравновзвешенный, чего ТЗ 7 не предусматривает.
        """
        if self.max_position_weight is None or weights.empty:
            return weights
        capped = weights.clip(upper=self.max_position_weight)
        if float(capped.sum()) < float(weights.sum()) - 1e-12:
            log.info(
                "Лимит на бумагу %.1f%% связал: в деньгах остаётся %.1f%% портфеля",
                100 * self.max_position_weight, 100 * (1.0 - capped.sum()),
            )
        return capped


@dataclass(frozen=True)
class TradeThrottle:
    """Ограничитель числа стоп-выходов за скользящую неделю.

    Ребалансировку он не трогает: она происходит раз в месяц по построению, и
    откладывать её значило бы менять стратегию, а не ограничивать риск.
    Ограничиваются только стоп-выходы, которые могут случиться в любой день.

    Смысл — не в оптимизации, а в предохранителе: если за неделю стопы выбили
    половину портфеля, происходит либо обвал рынка (с ним разбирается режимный
    фильтр), либо ошибка в данных. И в том, и в другом случае лучше притормозить.
    """

    #: Сколько стоп-выходов допустимо за 7 календарных дней. None — без предела.
    max_trades_per_week: int | None = None

    @classmethod
    def from_config(cls, cfg) -> TradeThrottle:
        value = cfg.max_trades_per_week
        return cls(max_trades_per_week=None if value is None else int(value))

    def allowed(self, recent: list[pd.Timestamp], today: pd.Timestamp) -> int:
        """Сколько выходов можно провести сегодня с учётом недавних."""
        if self.max_trades_per_week is None:
            return 1_000_000
        window_start = today - pd.Timedelta(days=7)
        used = sum(1 for day in recent if day > window_start)
        return max(0, self.max_trades_per_week - used)
