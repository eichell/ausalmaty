"""Формирование портфеля (ТЗ 7).

Отбор — топ-N по баллу, веса равные. Взвешивание по баллу ТЗ отвергает: оно не
даёт устойчивого преимущества и увеличивает концентрацию.

Два правила, которые взаимодействуют и потому реализованы одним проходом:

*   **Буферизация.** Бумага, уже находящаяся в портфеле, удерживается, пока
    остаётся в топ-`buffer_rank`. Без буфера портфель каждый месяц переставляет
    бумаги, стоящие на границе отсечки, платя издержки за перестановку шума.
*   **Лимит на сектор.** Не более `max_sector_weight` портфеля. Лимит задан в
    ТЗ в весах, а отбирается — штуками, и это не одно и то же. Пока портфель
    набирается до `top_n`, вес бумаги равен 1/top_n и пересчёт тривиален. Но если
    вселенная узкая и позиций меньше, вес каждой растёт, и три бумаги из девяти
    дают сектору 33% при лимите 30%. Поэтому после отбора идёт вторая проверка —
    уже по фактическому числу позиций.

Порядок обхода — по убыванию балла, сначала удерживаемые. Поэтому «вытеснять
худшие по баллу бумаги сектора» выполняется само: до худших очередь не доходит.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioRules:
    """Параметры ТЗ 7. Значения приходят из конфига."""

    top_n: int = 30
    buffer_rank: int = 45
    max_sector_weight: float = 0.30

    @classmethod
    def from_config(cls, cfg) -> PortfolioRules:
        return cls(
            top_n=int(cfg.top_n),
            buffer_rank=int(cfg.buffer_rank),
            max_sector_weight=float(cfg.max_sector_weight),
        )

    @property
    def max_names_per_sector(self) -> int:
        """Лимит в бумагах для полного портфеля."""
        return self.names_allowed(self.top_n)

    def names_allowed(self, n_positions: int) -> int:
        """Сколько бумаг сектора помещается в лимит при `n_positions` позициях.

        Минимум одна: сектор, представленный единственной бумагой, иначе
        выпадал бы из портфеля целиком при любом лимите меньше 1/n.
        """
        return max(1, math.floor(n_positions * self.max_sector_weight))


def select_portfolio(
    scores: pd.Series,
    sectors: pd.Series,
    rules: PortfolioRules,
    held: pd.Index | None = None,
) -> pd.Index:
    """Отбирает бумаги в портфель на одну дату ребалансировки.

    Args:
        scores: балл по бумагам, индекс — permaticker. NaN исключается.
        sectors: сектор каждой бумаги.
        held: что лежит в портфеле сейчас — для буферизации.

    Returns:
        Индекс отобранных permaticker, по убыванию балла.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return pd.Index([], name="permaticker")

    sector = sectors.reindex(ranked.index).astype("string").fillna("Unknown")
    held = pd.Index([]) if held is None else pd.Index(held)

    # Буфер: удержать можно только то, что ещё в списке кандидатов и не провалилось
    # ниже buffer_rank. Бумага, выпавшая из вселенной, удержанию не подлежит.
    buffer_zone = ranked.index[: rules.buffer_rank]
    incumbents = [p for p in buffer_zone if p in held]
    challengers = [p for p in ranked.index if p not in set(incumbents)]

    selected: list[int] = []
    per_sector: dict[str, int] = {}
    for permaticker in incumbents + challengers:
        if len(selected) >= rules.top_n:
            break
        name_sector = sector.loc[permaticker]
        if per_sector.get(name_sector, 0) >= rules.max_names_per_sector:
            continue
        selected.append(permaticker)
        per_sector[name_sector] = per_sector.get(name_sector, 0) + 1

    selected = _enforce_weight_cap(selected, sector, rules)

    # Порядок по баллу, а не по тому, кто был удержан: так вывод читается.
    out = ranked.loc[selected].sort_values(ascending=False).index
    return pd.Index(out, name="permaticker")


def _enforce_weight_cap(
    selected: list[int], sector: pd.Series, rules: PortfolioRules
) -> list[int]:
    """Досматривает лимит сектора по фактическому числу позиций.

    Нужно только для неполного портфеля: при `top_n` позициях отбор уже уложился
    в лимит. Лишние бумаги убираются с конца — они и по баллу худшие, потому что
    `selected` собран по убыванию.

    Лимит бывает недостижим: при трёх секторах и пороге 30% любое распределение
    даёт хотя бы одному сектору 34%. Тогда убирать бумаги бессмысленно — от этого
    веса оставшихся только растут. В таком случае состав остаётся как есть, а в
    лог уходит предупреждение: это ограничение вселенной, а не ошибка отбора.
    """
    while selected:
        counts = Counter(str(sector.loc[p]) for p in selected)
        allowed = rules.names_allowed(len(selected))
        over = [s for s, c in counts.items() if c > allowed]
        if not over:
            return selected

        if len(counts) * allowed < len(selected):
            log.warning(
                "Лимит сектора %.0f%% недостижим: в портфеле %d бумаг из %d секторов. "
                "Состав оставлен как есть.",
                100 * rules.max_sector_weight, len(selected), len(counts),
            )
            return selected

        worst_sector = max(over, key=lambda s: counts[s])
        for permaticker in reversed(selected):
            if str(sector.loc[permaticker]) == worst_sector:
                selected.remove(permaticker)
                break
    return selected


def equal_weights(selected: pd.Index) -> pd.Series:
    """Равные веса. Сумма ровно 1 на каждую дату (ТЗ 12).

    Если бумаг меньше top_n — вес делится между имеющимися, а не остаётся в
    денежной части: ТЗ не предусматривает частичного выхода в деньги нигде,
    кроме режимного фильтра (ТЗ 7.1).
    """
    if len(selected) == 0:
        return pd.Series(dtype="float64", name="weight")
    weights = pd.Series(1.0 / len(selected), index=selected, name="weight")
    return weights


def sector_weights(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    """Доли секторов в портфеле — для проверки лимита и для отчёта (ТЗ 10)."""
    sector = sectors.reindex(weights.index).astype("string").fillna("Unknown")
    return weights.groupby(sector).sum().sort_values(ascending=False)
