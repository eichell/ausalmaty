"""Вселенная на дату ребалансировки (ТЗ 5).

Пересчитывается на каждую дату и только по данным, доступным на неё. Здесь легко
и незаметно занести look-ahead, поэтому важны два запрета:

*   **`is_delisted` не участвует в отборе.** Это признак «сегодня», а не «тогда».
    Фильтр по нему выкинул бы из истории все обанкротившиеся компании — ровно тот
    survivorship bias, который ТЗ 4.1 называет необсуждаемым (доходность
    value-стратегий на такой вселенной завышена систематически). Момент, когда
    бумага перестала торговаться, берётся из самих цен.
*   **Порог цены — по нескорректированному ряду.** Скорректированная цена
    пересчитана от сегодняшней базы: акция, стоившая в 1999 году $40, после
    двадцати лет дивидендов и сплитов может числиться в панели за $3.

Сектор, биржа и категория берутся из справочника как есть. Это допущение:
Sharadar хранит текущую классификацию, а не историческую. Компания, сменившая
сектор, ретроспективно нормализуется не с теми соседями. Эффект мал по сравнению
с ценой отдельного справочника истории GICS, но он есть, и это осознанный размен.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from factorbot.data.panel import PricePanel

log = logging.getLogger(__name__)

#: Категории Sharadar, которые ТЗ 5 исключает: ETF, фонды, трасты, ADR, SPAC.
#: Проверка по подстроке: у поставщика это «Domestic Common Stock»,
#: «ADR Common Stock», «ETD», «Canadian Stock Warrant» и десяток похожих.
EXCLUDED_CATEGORY_MARKERS = ("ETF", "ETN", "ETD", "FUND", "TRUST", "ADR", "SPAC",
                             "WARRANT", "PREFERRED", "UNIT")

#: Что оставляем: обыкновенные акции США.
REQUIRED_CATEGORY_MARKER = "COMMON STOCK"


@dataclass(frozen=True)
class UniverseRules:
    """Пороги ТЗ 5. Значения приходят из конфига, не из кода."""

    exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "AMEX")
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    dollar_volume_window: int = 60
    min_price_history_months: int = 14

    @classmethod
    def from_config(cls, cfg) -> UniverseRules:
        return cls(
            exchanges=tuple(cfg.exchanges),
            min_price=float(cfg.min_price),
            min_dollar_volume=float(cfg.min_dollar_volume),
            dollar_volume_window=int(cfg.dollar_volume_window),
            min_price_history_months=int(cfg.min_price_history_months),
        )


def eligible_securities(securities: pd.DataFrame, rules: UniverseRules) -> pd.Index:
    """Статический отбор по справочнику: биржа и тип бумаги (ТЗ 5).

    `is_delisted` здесь намеренно не используется.
    """
    df = securities.copy()
    exchange = df["exchange"].astype("string").str.upper()
    category = df["category"].astype("string").str.upper().fillna("")

    on_main_exchange = exchange.isin([e.upper() for e in rules.exchanges])
    is_common = category.str.contains(REQUIRED_CATEGORY_MARKER, na=False)
    excluded = pd.Series(False, index=df.index)
    for marker in EXCLUDED_CATEGORY_MARKERS:
        excluded |= category.str.contains(marker, na=False)

    keep = df.loc[on_main_exchange & is_common & ~excluded, "permaticker"]
    return pd.Index(keep.astype("int64").unique(), name="permaticker")


def build_universe(
    panel: PricePanel,
    securities: pd.DataFrame,
    as_of: pd.Timestamp,
    rules: UniverseRules,
) -> pd.DataFrame:
    """Вселенная на одну дату ребалансировки.

    Returns:
        Кадр, индексированный permaticker, с колонками `price`, `dollar_volume`,
        `sector`. Строка есть только у бумаг, прошедших все фильтры ТЗ 5.
    """
    if as_of not in panel.trading_days:
        raise ValueError(f"{as_of.date()} не торговый день панели")

    allowed = eligible_securities(securities, rules)
    columns = panel.tickers.intersection(allowed)
    if columns.empty:
        return _empty_universe()

    day = panel.trading_days.get_loc(as_of)
    price = panel.close_unadj.iloc[day].reindex(columns)
    traded_today = price.notna()

    window = panel.dollar_volume.iloc[
        max(0, day - rules.dollar_volume_window + 1): day + 1
    ].reindex(columns=columns)
    adv = window.mean(skipna=True)

    # История цен: первая котировка должна быть достаточно давно. Считаем по самой
    # панели, а не по справочнику — так бумага, чьи данные начинаются позже
    # заявленного, не пролезет.
    history_start = as_of - pd.DateOffset(months=rules.min_price_history_months)
    closeadj = panel.closeadj.reindex(columns=columns)
    first_seen = closeadj.apply(lambda col: col.first_valid_index())
    long_enough = first_seen.notna() & (pd.to_datetime(first_seen) <= history_start)

    keep = (
        traded_today
        & (price > rules.min_price)
        & (adv > rules.min_dollar_volume)
        & long_enough
    )
    selected = columns[keep.reindex(columns).fillna(False)]
    if len(selected) == 0:
        return _empty_universe()

    sectors = securities.set_index("permaticker")["sector"]
    out = pd.DataFrame({
        "price": price.reindex(selected),
        "dollar_volume": adv.reindex(selected),
        "sector": sectors.reindex(selected).astype("string").fillna("Unknown"),
    })
    out.index.name = "permaticker"
    return out


def _empty_universe() -> pd.DataFrame:
    out = pd.DataFrame({
        "price": pd.Series(dtype="float64"),
        "dollar_volume": pd.Series(dtype="float64"),
        "sector": pd.Series(dtype="string"),
    })
    out.index.name = "permaticker"
    return out


def save_universe(
    conn, members: dict[pd.Timestamp, pd.Index], *, replace: bool = True
) -> int:
    """Сохраняет состав вселенной по датам в таблицу `universe` (ТЗ 4.7).

    Вселенная считается на лету при каждом прогоне, и без записи невозможно
    ответить на вопрос «что вообще было доступно к покупке 30 июня 2007 года».
    А это первый вопрос при разборе любой странной сделки: бумага не куплена
    потому, что балл низкий, или потому, что её не было во вселенной вовсе.

    Args:
        replace: очистить таблицу перед записью. Иначе останутся строки прошлого
            прогона с другими порогами, и таблица перестанет соответствовать
            какому-либо одному прогону.

    Returns:
        Число записанных строк.
    """
    rows = [
        {"date": day.date() if hasattr(day, "date") else day, "permaticker": int(p)}
        for day, index in members.items()
        for p in index
    ]
    if not rows:
        log.warning("Вселенная пуста на всех датах: записывать нечего")
        return 0

    if replace:
        conn.execute("DELETE FROM universe")

    frame = pd.DataFrame(rows).drop_duplicates()
    conn.register("_universe", frame)
    conn.execute("INSERT OR REPLACE INTO universe SELECT * FROM _universe")
    conn.unregister("_universe")
    log.info("Записано в universe: %d строк на %d дат", len(frame), len(members))
    return len(frame)
