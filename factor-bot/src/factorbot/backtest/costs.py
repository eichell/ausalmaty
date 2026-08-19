"""Издержки (ТЗ 8).

Не опция, а обязательная часть модели. Отчёт обязан показывать доходность до и
после издержек: разрыв между ними — прямой индикатор чувствительности стратегии
к обороту.

Ставки из ТЗ: 10 б.п. на сторону для бумаг с оборотом больше $50 млн в день,
25 б.п. для остальных. Комиссия по умолчанию нулевая (современные брокеры США),
но параметр оставлен: он понадобится в момент, когда брокер сменится.

Влияние на цену не моделируется. Это допущение живёт при капитале $1–10 млн;
выше него оно перестаёт работать, и цифры отчёта становятся оптимистичными.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Параметры ТЗ 8."""

    commission_bps: float = 0.0
    slippage_bps_liquid: float = 10.0
    slippage_bps_illiquid: float = 25.0
    liquid_dollar_volume_threshold: float = 50_000_000.0

    @classmethod
    def from_config(cls, cfg) -> CostModel:
        return cls(
            commission_bps=float(cfg.commission_bps),
            slippage_bps_liquid=float(cfg.slippage_bps_liquid),
            slippage_bps_illiquid=float(cfg.slippage_bps_illiquid),
            liquid_dollar_volume_threshold=float(cfg.liquid_dollar_volume_threshold),
        )

    def rate_per_side(self, dollar_volume: pd.Series) -> pd.Series:
        """Ставка на сторону по каждой бумаге, в долях.

        Неизвестный оборот трактуется как неликвид: ошибиться в сторону более
        дорогой оценки безопаснее, чем занизить издержки.
        """
        illiquid = dollar_volume.isna() | (
            dollar_volume <= self.liquid_dollar_volume_threshold
        )
        bps = pd.Series(self.slippage_bps_liquid, index=dollar_volume.index)
        bps = bps.where(~illiquid, self.slippage_bps_illiquid)
        return (bps + self.commission_bps) * BPS


def turnover(before: pd.Series, after: pd.Series) -> float:
    """Оборот ребалансировки: половина суммы модулей изменений весов.

    Половина — потому что продажа одной бумаги ради покупки другой это один
    оборот портфеля, а не два. Для издержек считается полная сумма: платят обе
    стороны сделки.
    """
    changes = _weight_changes(before, after)
    return float(changes.abs().sum() / 2)


def rebalance_cost(
    before: pd.Series, after: pd.Series, dollar_volume: pd.Series, model: CostModel
) -> float:
    """Издержки ребалансировки в долях капитала.

    Каждая изменённая доля веса — это сделка, и каждая сделка платит ставку своей
    бумаги. Продажи и покупки считаются отдельно, потому что платятся отдельно.
    """
    changes = _weight_changes(before, after)
    if changes.empty:
        return 0.0
    rates = model.rate_per_side(dollar_volume.reindex(changes.index))
    return float((changes.abs() * rates).sum())


def _weight_changes(before: pd.Series, after: pd.Series) -> pd.Series:
    names = before.index.union(after.index)
    return after.reindex(names).fillna(0.0) - before.reindex(names).fillna(0.0)
