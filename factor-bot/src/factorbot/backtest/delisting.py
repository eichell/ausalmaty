"""Доходность при делистинге (ТЗ 4.1).

Требование ТЗ: банкротство обязано давать −100%, а не исчезновение бумаги из
выборки. Разница принципиальная. Если позиция просто пропадает из расчёта,
бэктест никогда не фиксирует убыток, которого в жизни избежать было нельзя, и
результат завышается тем сильнее, чем чаще стратегия покупает дешёвые бумаги, —
то есть ровно для value.

Источник правды — ACTIONS. Но читать оттуда можно далеко не любое поле, и это
главное, что здесь надо знать.

**Колонка `value` означает разное в разных событиях.** У события `delisted` это
капитализация компании в миллионах долларов, а не выплата на акцию. Проверяется
на минуту: у Lucent (LU) последняя цена $2.55 и `value` 11429.1, у Deutsche
Telekom $11.97 и 52205.0 — это $11.4 и $52.2 млрд капитализации. Разделив такое
на цену акции, получаешь делистинг с доходностью +448 000% — и именно это
делала первая версия модуля. Результат: 4712 бумаг с «доходностью» выше +1000%,
CAGR 1878% и положительный 2008 год. Симптом был виден сразу (ТЗ 9.3 прямо
называет такое поводом искать ошибку), но причина лежала не в движке.

Поэтому выплата берётся только из событий, где `value` действительно является
суммой на акцию, и это проверено на данных: у `acquisitioncash` отношение
`value` к последней цене имеет медиану 1.001 и 99-й процентиль 1.15. У событий
`delisted`, `acquisitionby`, `acquisitionof`, `voluntarydelisting`,
`regulatorydelisting`, `mergerto` то же отношение измеряется десятками — это
капитализация, и здесь эти поля не читаются никогда.

**Оплата акциями покупателя — не убыток.** В сделке за акции держатель получает
бумаги покупателя примерно на ту же сумму, по которой цель торговалась в
последний день. Считать такой уход как −100% значило бы занижать результат на
1755 бумагах in-sample. Поэтому для них доходность 0%: позиция конвертируется по
рынку, а дальше деньги ждут ближайшей ребалансировки. Это допущение; точная
альтернатива — вести позицию в бумагах покупателя, что требует карты
цель → покупатель, которой в ACTIONS нет.

**Смешанная оплата.** Если у бумаги есть и денежное, и акционное событие, `value`
денежного — только доплата к акциям (у 44 таких сделок отношение к цене около
0.19). Брать её за всю выплату значило бы получить −80% на удачном поглощении,
поэтому смешанные сделки считаются как акционные.

**Остаток.** Больше половины ушедших бумаг (8078 из 14658 in-sample) не имеет ни
одного события с известной оплатой. Для них остаётся требование ТЗ 4.1: −100%.
Это заведомо занижает результат — часть таких бумаг ушла на OTC, а не в ноль, —
но ошибается в безопасную сторону.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

#: Что предполагается, когда о судьбе бумаги ничего не известно (ТЗ 4.1).
DEFAULT_DELISTING_RETURN = -1.0

#: События, где `value` — сумма денег на акцию. Единственный источник выплаты.
CASH_CONSIDERATION = ("acquisitioncash", "acquisitionelectcash")

#: События, где держатель получает бумаги покупателя, а не деньги.
STOCK_CONSIDERATION = (
    "acquisitionstock", "acquisitionelectstock", "spacmerger", "mergerto",
)

#: Доходность при оплате акциями: позиция конвертируется по рыночной стоимости.
STOCK_CONSIDERATION_RETURN = 0.0

#: Ликвидация: держатель не получает ничего.
BANKRUPTCY_ACTIONS = ("bankruptcyliquidation",)

#: Полоса доверия для денежной выплаты. Сделка за деньги проходит по цене,
#: близкой к последней рыночной: на данных in-sample 99% отношений лежат между
#: 0.88 и 1.12. Выход за полосу означает, что прочитано не то поле или событие
#: сопоставлено не с той бумагой, и такая выплата не используется вовсе —
#: молча пустить её в расчёт значит повторить ту же ошибку в мелком масштабе.
CASH_RETURN_BOUNDS = (-0.9, 1.0)


def build_delisting_returns(
    securities: pd.DataFrame,
    corp_actions: pd.DataFrame | None = None,
    last_prices: pd.Series | None = None,
) -> pd.Series:
    """Доходность последней сделки по каждой ушедшей бумаге.

    Args:
        securities: справочник; используется только флаг `is_delisted`.
        corp_actions: события ACTIONS. Без них всё считается как −100%.
        last_prices: последняя **нескорректированная** цена бумаги. Выплата в
            ACTIONS указана в тогдашних долларах, а скорректированный ряд
            пересчитан от другой базы; смешивать их нельзя. Доходность при этом
            остаётся отношением и корректно применяется к позиции, оценённой по
            скорректированной цене.

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

    action = corp_actions["action"].astype("string").str.lower()

    # Порядок важен: акционная оплата рассматривается первой, потому что при
    # смешанной сделке денежная строка несёт только доплату.
    paid_in_stock = pd.Index(
        corp_actions.loc[action.isin(STOCK_CONSIDERATION), "permaticker"].unique()
    ).intersection(out.index)
    out.loc[paid_in_stock] = STOCK_CONSIDERATION_RETURN

    paid_in_cash, rejected = _cash_returns(
        corp_actions.loc[action.isin(CASH_CONSIDERATION)], last_prices,
        exclude=paid_in_stock,
    )
    known_cash = paid_in_cash.index.intersection(out.index)
    out.loc[known_cash] = paid_in_cash.loc[known_cash]

    # Ликвидация — это и есть −100%, то есть значение по умолчанию. Строка ниже
    # ничего не меняет численно и стоит здесь ради одного: если когда-нибудь
    # значение по умолчанию перестанет быть −100%, банкротство останется нулём.
    liquidated = pd.Index(
        corp_actions.loc[action.isin(BANKRUPTCY_ACTIONS), "permaticker"].unique()
    ).intersection(out.index).difference(paid_in_stock).difference(known_cash)
    out.loc[liquidated] = DEFAULT_DELISTING_RETURN

    unknown = len(out) - len(paid_in_stock) - len(known_cash) - len(liquidated)
    log.info(
        "Делистинг: деньгами %d, акциями %d, ликвидация %d, без данных %d "
        "(считаются как −100%%, ТЗ 4.1)",
        len(known_cash), len(paid_in_stock), len(liquidated), unknown,
    )
    if rejected:
        log.warning(
            "Выплат вне полосы доверия %s: %d — не использованы (см. модуль delisting).",
            CASH_RETURN_BOUNDS, rejected,
        )
    return out


def _cash_returns(
    events: pd.DataFrame, last_prices: pd.Series | None, *, exclude: pd.Index
) -> tuple[pd.Series, int]:
    """Доходность денежных поглощений. Возвращает серию и число отброшенных."""
    empty = pd.Series(dtype="float64", name="delisting_return")
    if events.empty:
        return empty, 0
    if last_prices is None:
        log.warning("Нет последних цен: выплаты при делистинге перевести в доходность нечем.")
        return empty, 0

    last = events.sort_values("date").drop_duplicates("permaticker", keep="last")
    payout = pd.to_numeric(last.set_index("permaticker")["value"], errors="coerce")
    payout = payout.loc[payout > 0].drop(index=exclude, errors="ignore")

    reference = last_prices.reindex(payout.index)
    ret = (payout / reference.where(reference > 0) - 1.0).dropna()

    low, high = CASH_RETURN_BOUNDS
    trusted = ret.loc[(ret >= low) & (ret <= high)]
    return trusted.rename("delisting_return"), len(ret) - len(trusted)


def apply_delisting(
    position_value: float, permaticker: int, delisting_returns: pd.Series
) -> float:
    """Стоимость позиции после ухода бумаги с биржи."""
    ret = delisting_returns.get(permaticker, DEFAULT_DELISTING_RETURN)
    return position_value * (1.0 + float(ret))
