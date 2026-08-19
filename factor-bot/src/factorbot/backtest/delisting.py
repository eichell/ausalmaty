"""Доходность при делистинге (ТЗ 4.1).

Требование ТЗ: банкротство обязано давать −100%, а не исчезновение бумаги из
выборки. Разница принципиальная. Если позиция просто пропадает из расчёта,
бэктест никогда не фиксирует убыток, которого в жизни избежать было нельзя, и
результат завышается тем сильнее, чем чаще стратегия покупает дешёвые бумаги, —
то есть ровно для value.

Источник правды — ACTIONS. Когда таблица недоступна (урезанный тариф), остаётся
консервативное допущение: делистинг = −100%. Оно завышает убытки поглощённых
компаний, но ошибается в безопасную сторону, а не в приятную.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

#: Действия ACTIONS, означающие уход бумаги с биржи.
DELISTING_ACTIONS = ("delisted", "delisting")

#: Что предполагается, когда о судьбе бумаги ничего не известно.
DEFAULT_DELISTING_RETURN = -1.0


def build_delisting_returns(
    securities: pd.DataFrame,
    corp_actions: pd.DataFrame | None = None,
    last_prices: pd.Series | None = None,
) -> pd.Series:
    """Доходность последней сделки по каждой ушедшей бумаге.

    Args:
        securities: справочник; используется только флаг `is_delisted`.
        corp_actions: события ACTIONS. Без них всё считается как −100%.
        last_prices: последняя известная цена бумаги. Нужна, чтобы перевести
            выплату при делистинге в доходность.

    Returns:
        permaticker → доходность (−1.0 означает полную потерю позиции).
        Бумаги, которые всё ещё торгуются, в результат не попадают.
    """
    delisted = securities.loc[securities["is_delisted"].fillna(False).astype(bool)]
    out = pd.Series(
        DEFAULT_DELISTING_RETURN,
        index=pd.Index(delisted["permaticker"].astype("int64"), name="permaticker"),
        dtype="float64",
    )

    if corp_actions is None or corp_actions.empty:
        log.warning(
            "ACTIONS недоступна: делистинг всех %d бумаг считается как −100%% (ТЗ 4.1). "
            "Поглощения при этом занижены.", len(out),
        )
        return out

    events = corp_actions.loc[
        corp_actions["action"].astype("string").str.lower().isin(DELISTING_ACTIONS)
    ]
    if events.empty:
        return out

    # Значение действия — то, что получил держатель за бумагу. Ноль или пропуск
    # означают, что не получил ничего.
    if last_prices is None:
        log.warning("Нет последних цен: выплаты при делистинге перевести в доходность нечем.")
        return out

    last = events.sort_values("date").drop_duplicates("permaticker", keep="last")
    payout = pd.to_numeric(last.set_index("permaticker")["value"], errors="coerce")
    payout = payout.loc[payout > 0]

    reference = last_prices.reindex(payout.index)
    recovered = (payout / reference.where(reference > 0) - 1.0).dropna()
    # Ниже −100% уйти нельзя: акционер не доплачивает за банкротство.
    recovered = recovered.clip(lower=DEFAULT_DELISTING_RETURN)

    known = recovered.index.intersection(out.index)
    out.loc[known] = recovered.loc[known]
    log.info("Возврат при делистинге известен для %d бумаг из %d", len(known), len(out))
    return out


def apply_delisting(
    position_value: float, permaticker: int, delisting_returns: pd.Series
) -> float:
    """Стоимость позиции после ухода бумаги с биржи."""
    ret = delisting_returns.get(permaticker, DEFAULT_DELISTING_RETURN)
    return position_value * (1.0 + float(ret))
