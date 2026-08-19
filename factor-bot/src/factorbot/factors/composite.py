"""Композит факторов (ТЗ 6.3).

    composite = 0.5 * z_momentum + 0.5 * z_value_score

Веса заданы константой в конфиге и **не подбираются по результатам бэктеста**
(ТЗ 9.2.3). Подбор весов на двух факторах и коротком ряду почти гарантированно
даёт переподгонку: параметров мало, степеней свободы много, и лучший результат
объясняется перебором, а не сигналом.

Что делать с бумагой, у которой есть не все компоненты
------------------------------------------------------

ТЗ 5 исключает финансовый сектор из value, но требует оставить его для momentum.
Буквальная формула на такой бумаге даёт NaN, и она исчезает из портфеля вовсе —
то есть исключается и из momentum тоже, вопреки требованию. То же происходит с
бумагой, выбывшей из value-ранжирования по ТЗ 6.3.

Здесь вес недостающего компонента перераспределяется на имеющиеся: у финансовой
бумаги композит равен её z_momentum. Это ближе к тексту ТЗ, чем два очевидных
варианта — выкинуть бумагу или подставить ноль вместо пропуска (второе занижало
бы балл систематически и означало бы заполнение пропусков, запрещённое ТЗ 6.3).

Цена решения. Балл по одному фактору шумнее, чем среднее двух: усреднение гасит
дисперсию. Значит бумаги с одним компонентом чаще попадают в оба хвоста
распределения, а отбор берёт верхний — и финансы окажутся в портфеле чаще, чем
следует из их доли во вселенной. Эффект надо иметь в виду при разборе вклада по
секторам (ТЗ 10) и проверять на карте чувствительности (ТЗ 9.2.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositeRules:
    """Веса факторов. Априорный выбор, не результат подбора (ТЗ 9.2.3)."""

    weights: dict[str, float]
    #: Сколько компонентов должно быть, чтобы балл вообще считался.
    min_components: int = 1

    @classmethod
    def from_config(cls, factors_cfg) -> CompositeRules:
        weights = {name: float(value) for name, value in factors_cfg.weights.items()}
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Веса факторов должны давать 1.0, получено {total}.")
        return cls(weights=weights)


def combine(
    components: dict[str, pd.Series], rules: CompositeRules
) -> pd.Series:
    """Взвешенное среднее z-оценок с перераспределением весов на имеющиеся.

    Args:
        components: имя фактора → его z-оценки. Индексы могут не совпадать;
            результат считается по объединению.

    Returns:
        Балл по каждой бумаге. NaN у тех, у кого не набралось `min_components`.

    Raises:
        KeyError: если для компонента не задан вес — молча его игнорировать
            нельзя, это изменило бы стратегию без следа в конфиге.
    """
    missing_weights = [name for name in components if name not in rules.weights]
    if missing_weights:
        raise KeyError(
            f"Для компонентов {missing_weights} не заданы веса в конфиге. "
            f"Известны: {sorted(rules.weights)}."
        )

    if not components:
        return pd.Series(dtype="float64", name="composite")

    frame = pd.DataFrame(components)
    weights = pd.Series({name: rules.weights[name] for name in frame.columns})

    present = frame.notna()
    # Знаменатель — сумма весов тех компонентов, что есть у этой бумаги.
    weight_sum = present.mul(weights, axis=1).sum(axis=1)
    weighted = frame.mul(weights, axis=1).sum(axis=1, skipna=True)

    score = (weighted / weight_sum.where(weight_sum > 0)).where(
        present.sum(axis=1) >= rules.min_components
    )

    partial = int(((present.sum(axis=1) > 0) & (present.sum(axis=1) < len(frame.columns))).sum())
    if partial:
        log.debug("%d бумаг с неполным набором компонентов: вес перераспределён", partial)

    return score.rename("composite")
