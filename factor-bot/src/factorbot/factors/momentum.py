"""Momentum (ТЗ 6.1).

    mom_12_1 = adj_close[t - 21] / adj_close[t - 252] - 1

Пропуск последнего месяца обязателен: на горизонте до 21 дня работает
краткосрочный разворот, который частично гасит эффект импульса. Формула
намеренно записана в торговых днях, а не в календарных месяцах, — так окно
не «дышит» вместе с числом праздников в году.

Считается по скорректированной цене. На нескорректированном ряду сплит 2:1
выглядит как падение на 50%, и фактор уверенно ставит такую бумагу в аутсайдеры
(ТЗ 4.1, тест из ТЗ 12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorbot.data.panel import PricePanel

#: Сколько дней допустимо тянуть последнюю цену вперёд. Праздники и короткие
#: остановки торгов закрываются, длинная дыра — нет: у бумаги, не торговавшейся
#: месяц, импульса не существует, и подставлять ей старую цену нечестно.
MAX_STALE_DAYS = 5


def _price_at_offset(closeadj: pd.DataFrame, day: int, offset: int) -> pd.Series:
    """Цена за `offset` торговых дней до позиции `day`, с коротким ffill."""
    position = day - offset
    if position < 0:
        return pd.Series(np.nan, index=closeadj.columns)
    lo = max(0, position - MAX_STALE_DAYS)
    return closeadj.iloc[lo: position + 1].ffill().iloc[-1]


def momentum(
    panel: PricePanel,
    as_of: pd.Timestamp,
    *,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.Series:
    """Импульс на дату ребалансировки. NaN там, где истории не хватает."""
    if lookback_days <= skip_days:
        raise ValueError(
            f"Окно {lookback_days} должно быть длиннее пропуска {skip_days}: "
            "иначе формула считает доходность назад во времени."
        )
    if as_of not in panel.trading_days:
        raise ValueError(f"{as_of.date()} не торговый день панели")

    day = panel.trading_days.get_loc(as_of)
    recent = _price_at_offset(panel.closeadj, day, skip_days)
    distant = _price_at_offset(panel.closeadj, day, lookback_days)

    out = recent / distant.where(distant > 0) - 1.0
    out.name = "mom_12_1"
    return out


def momentum_diagnostics(
    panel: PricePanel,
    as_of: pd.Timestamp,
    *,
    short_lookback_days: int = 126,
    skip_days: int = 21,
    volatility_window: int = 126,
) -> pd.DataFrame:
    """Полугодовой импульс и волатильность (ТЗ 6.1).

    Для диагностики, не для основного сигнала. Нужны, чтобы понимать, чем именно
    объясняется результат: если весь эффект держится на одном окне и исчезает на
    соседнем, это подгонка, а не фактор (ТЗ 9.2.2).
    """
    day = panel.trading_days.get_loc(as_of)
    recent = _price_at_offset(panel.closeadj, day, skip_days)
    distant = _price_at_offset(panel.closeadj, day, short_lookback_days)

    window = panel.closeadj.iloc[max(0, day - volatility_window): day + 1]
    daily = window.pct_change()
    # Годовая волатильность: 252 торговых дня.
    vol = daily.std(skipna=True) * np.sqrt(252)

    return pd.DataFrame({
        "mom_6_1": recent / distant.where(distant > 0) - 1.0,
        "volatility": vol,
    })
