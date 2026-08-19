"""Нормализация факторов (ТЗ 6.3).

Одинаковая процедура для всех факторов:

1.  Винзоризация по 1-му и 99-му процентилю на каждую дату.
2.  Ранг внутри сектора GICS → z-оценка через обратную нормальную функцию.
3.  Пропуски не заполняются.

Почему ранг, а не z-score напрямую: в фундаментальных данных много выбросов, и
одна компания с E/P = 40 растягивает шкалу так, что все остальные слипаются
около нуля. Ранг к такому нечувствителен по построению.

Винзоризация перед ранжированием выглядит избыточной — ранг и так не зависит от
величины выброса. Она нужна следующему шагу: диагностике и любым расчётам,
которые смотрят на сырое значение. Порядок из ТЗ сохранён.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Обрезает хвосты по процентилям текущего среза."""
    valid = values.dropna()
    if valid.empty:
        return values
    lo, hi = valid.quantile([lower, upper])
    return values.clip(lower=lo, upper=hi)


def rank_to_z(values: pd.Series) -> pd.Series:
    """Ранг → z через обратную нормальную функцию.

    Формула Ван дер Вардена: z = Φ⁻¹((rank − 0.5) / n). Сдвиг на полранга нужен,
    чтобы у крайних бумаг не получалось Φ⁻¹(0) = −∞.
    """
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    if len(valid) == 1:
        # Одна бумага в группе — сравнивать не с чем, средний балл честнее.
        return pd.Series(0.0, index=values.index).where(values.notna())

    ranks = valid.rank(method="average")
    z = pd.Series(norm.ppf((ranks - 0.5) / len(valid)), index=valid.index)
    return z.reindex(values.index)


def normalize_within_sector(
    values: pd.Series,
    sectors: pd.Series,
    *,
    winsorize_bounds: tuple[float, float] = (0.01, 0.99),
) -> pd.Series:
    """Полная процедура ТЗ 6.3 для одного фактора на одну дату.

    Args:
        values: значения фактора, индекс — permaticker.
        sectors: сектор каждой бумаги, тот же индекс.

    Returns:
        z-оценки. NaN остаётся NaN: пропуски не заполняются (ТЗ 6.3).
    """
    values = values.astype("float64")
    clipped = winsorize(values, *winsorize_bounds)
    sector = sectors.reindex(values.index).astype("string").fillna("Unknown")

    out = clipped.groupby(sector, dropna=False).transform(rank_to_z)
    out.name = f"z_{values.name}" if values.name else "z"
    return out
