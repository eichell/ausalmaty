"""Value (ТЗ 6.2, 6.3).

Считаются **доходности, а не мультипликаторы** — перевёрнутые отношения. Причина
конкретная: при отрицательной прибыли P/E уходит в бессмысленный минус, и
убыточная компания оказывается в одном ряду с самыми дешёвыми. E/P помещает её
туда, где ей и место, — в нижнюю часть шкалы, монотонно и без разрывов.

    earnings_yield = netinc_ttm  / market_cap
    sales_yield    = revenue_ttm / market_cap
    book_to_market = equity      / market_cap
    fcf_yield      = (opcf_ttm + capex_ttm) / market_cap

Знак capex. ТЗ 6.2 записывает FCF как `opcf − capex`, подразумевая capex
положительным. Поставщик отдаёт отток отрицательным, и это значение хранится как
есть, поэтому здесь стоит плюс. Перепутать знак тут — значит удвоить FCF вместо
вычитания и получить фактор, который уверенно любит капиталоёмкие компании.

Данные берутся строго через `pit.get_fundamentals` (ТЗ 4.8): потоковые величины
из ART, балансовые из ARQ (ТЗ 4.3). Никакого кэша между датами — на каждой дате
ребалансировки выборка своя, и пост-условие PIT проверяется заново.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from factorbot.data import pit
from factorbot.data.panel import PricePanel
from factorbot.normalize import normalize_within_sector

log = logging.getLogger(__name__)

#: Компоненты композита ТЗ 6.2, в порядке перечисления.
COMPONENTS: tuple[str, ...] = (
    "earnings_yield", "sales_yield", "book_to_market", "fcf_yield",
)

#: Сколько дней допустимо тянуть последнюю цену вперёд при расчёте капитализации.
MAX_STALE_DAYS = 5


@dataclass(frozen=True)
class ValueRules:
    """Параметры ТЗ 6.2–6.3."""

    components: tuple[str, ...] = COMPONENTS
    #: Минимум компонентов, при котором бумага остаётся в ранжировании.
    #: ТЗ 6.3: при отсутствии трёх и более из четырёх — исключается.
    min_components: int = 2
    #: Финансовый сектор исключается из value: Debt/Equity и балансовые метрики
    #: там несопоставимы с остальными секторами (ТЗ 5). Для momentum остаётся.
    excluded_sector: str = "Financials"

    @classmethod
    def from_config(cls, factors_cfg, universe_cfg) -> ValueRules:
        return cls(
            components=tuple(factors_cfg.value.components),
            min_components=int(factors_cfg.value.min_components),
            excluded_sector=str(universe_cfg.exclude_sector_from_value),
        )


def market_cap(panel: PricePanel, as_of: pd.Timestamp, shares: pd.Series) -> pd.Series:
    """Капитализация: цена на дату ребалансировки × акции из последнего отчёта.

    Цена берётся нескорректированная. Скорректированный ряд пересчитан от
    сегодняшней базы, а число акций — «тогдашнее»; перемножив их, для компании,
    делавшей сплит, получим капитализацию, отличающуюся в разы.
    """
    day = panel.trading_days.get_loc(as_of)
    lo = max(0, day - MAX_STALE_DAYS)
    price = panel.close_unadj.iloc[lo: day + 1].ffill().iloc[-1]

    shares = pd.to_numeric(shares, errors="coerce")
    shares = shares.where(shares > 0)
    return (price.reindex(shares.index) * shares).rename("market_cap")


def compute_yields(
    conn: duckdb.DuckDBPyConnection,
    panel: PricePanel,
    as_of: pd.Timestamp,
    permatickers: list[int],
) -> pd.DataFrame:
    """Четыре доходности ТЗ 6.2 на одну дату ребалансировки.

    Returns:
        Кадр с колонками COMPONENTS, индекс — permaticker. Строки есть только у
        бумаг, для которых известна капитализация; пропуски внутри строк
        сохраняются как NaN и не заполняются (ТЗ 6.3).
    """
    as_of_date = pd.Timestamp(as_of).date()
    flows = pit.get_fundamentals(conn, as_of_date, pit.FLOW_DIMENSION, permatickers)
    stocks = pit.get_fundamentals(conn, as_of_date, pit.STOCK_DIMENSION, permatickers)

    if stocks.empty:
        return _empty_yields()

    stocks = stocks.set_index("permaticker")
    caps = market_cap(panel, as_of, stocks["sharesbas"])
    caps = caps.loc[caps.notna() & (caps > 0)]
    if caps.empty:
        return _empty_yields()

    out = pd.DataFrame(index=caps.index)
    out.index.name = "permaticker"

    equity = pd.to_numeric(stocks["equity"], errors="coerce").reindex(caps.index)
    out["book_to_market"] = equity / caps

    if flows.empty:
        for name in ("earnings_yield", "sales_yield", "fcf_yield"):
            out[name] = np.nan
        return out[list(COMPONENTS)]

    flows = flows.set_index("permaticker")
    netinc = pd.to_numeric(flows["netinc_ttm"], errors="coerce").reindex(caps.index)
    revenue = pd.to_numeric(flows["revenue_ttm"], errors="coerce").reindex(caps.index)
    opcf = pd.to_numeric(flows["opcf_ttm"], errors="coerce").reindex(caps.index)
    capex = pd.to_numeric(flows["capex_ttm"], errors="coerce").reindex(caps.index)

    out["earnings_yield"] = netinc / caps
    out["sales_yield"] = revenue / caps
    # Плюс, а не минус: у поставщика отток капитала отрицателен. См. docstring.
    out["fcf_yield"] = (opcf + capex) / caps

    return out[list(COMPONENTS)]


def value_score(
    yields: pd.DataFrame, sectors: pd.Series, rules: ValueRules | None = None
) -> pd.Series:
    """Композит value: среднее z-оценок компонентов (ТЗ 6.3).

    Каждый компонент нормализуется отдельно — сначала винзоризация, затем ранг
    внутри сектора и перевод в z. Усреднять сырые доходности нельзя: у них разные
    масштабы, и sales_yield, который у торговых компаний измеряется единицами,
    просто задавил бы остальные три.

    Бумага с недостающими компонентами получает балл по имеющимся; если их
    осталось меньше `min_components` — выбывает из ранжирования, а не получает
    заниженный балл. Финансовый сектор исключается целиком (ТЗ 5).
    """
    rules = rules or ValueRules()
    if yields.empty:
        return pd.Series(dtype="float64", name="value_score")

    sector = sectors.reindex(yields.index).astype("string").fillna("Unknown")
    keep = sector.str.lower() != rules.excluded_sector.lower()

    frame = yields.loc[keep, list(rules.components)]
    if frame.empty:
        return pd.Series(dtype="float64", name="value_score")

    z = pd.DataFrame(index=frame.index)
    for component in rules.components:
        values = frame[component].rename(component)
        z[component] = normalize_within_sector(values, sector.loc[frame.index])

    available = z.notna().sum(axis=1)
    score = z.mean(axis=1, skipna=True).where(available >= rules.min_components)

    dropped = int((available < rules.min_components).sum())
    if dropped:
        log.debug(
            "%d бумаг вне value-ранжирования: меньше %d компонентов (ТЗ 6.3)",
            dropped, rules.min_components,
        )
    return score.rename("value_score")


def _empty_yields() -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype="float64") for c in COMPONENTS})
    out.index.name = "permaticker"
    return out
