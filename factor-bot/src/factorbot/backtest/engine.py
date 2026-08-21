"""Движок бэктеста (ТЗ 7).

Собственный, векторизованный не до конца намеренно: дневной цикл по позициям
читается и проверяется, а месячная кросс-секционная ребалансировка от векторизации
почти ничего не выигрывает — 168 ребалансировок за четырнадцать лет in-sample.
Готовые фреймворки ТЗ 3 отвергает по той же причине: они прячут детали, которые
здесь и решают достоверность.

Как устроен день исполнения. Сигнал считается на последний торговый день месяца,
сделки проходят по цене открытия следующего дня (ТЗ 7). Поэтому этот день
разбивается на три шага:

1.  Старые позиции переоцениваются от вчерашнего закрытия до сегодняшнего открытия.
2.  По открытию проходит ребалансировка, и с оборота списываются издержки (ТЗ 8).
3.  Новые позиции живут от открытия до закрытия.

Так между сигналом и сделкой всегда стоит ночь, и ни одна цена, по которой
торгуем, не участвовала в расчёте того, что покупать.

Доходность до издержек получается делением на накопленный множитель издержек, а
не вторым прогоном: веса в обоих мирах совпадают, издержки только масштабируют
капитал, поэтому результат точный, а не приближённый.

Кривая эквити начинается с первой реальной позиции, а не с первой даты периода.
В начале выборки вселенная пуста: бумагам нужно 14 месяцев истории цен (ТЗ 5).
Плоский участок до первой покупки — не результат стратегии, но в метрики он
попадает и занижает и волатильность, и просадку, и CAGR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from factorbot.backtest.costs import CostModel, rebalance_cost, turnover
from factorbot.backtest.delisting import DEFAULT_DELISTING_RETURN
from factorbot.data.panel import PricePanel
from factorbot.portfolio import PortfolioRules, equal_weights, select_portfolio
from factorbot.regime import RegimeFilter
from factorbot.risk import PositionLimits, StopLossRules, TradeThrottle
from factorbot.universe import UniverseRules, build_universe

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Результат прогона. Всё, что нужно отчёту ТЗ 10, и ничего сверх."""

    equity_net: pd.Series
    equity_gross: pd.Series
    weights: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    universe_size: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    delisted_hits: int = 0
    #: Даты ребалансировки, на которых сработал режимный фильтр (ТЗ 7.1).
    risk_off_dates: list[pd.Timestamp] = field(default_factory=list)
    #: Состав вселенной на каждую дату сигнала — для таблицы `universe` (ТЗ 4.7).
    universe_members: dict[pd.Timestamp, pd.Index] = field(default_factory=dict)
    #: Число сделок на каждой ребалансировке (ТЗ 10).
    trades: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    #: Сработавшие стоп-лоссы: (дата исполнения, permaticker). Вне ТЗ.
    stops: list[tuple[pd.Timestamp, int]] = field(default_factory=list)
    #: Сколько раз ограничитель откладывал стоп-выход.
    throttled: int = 0

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    @property
    def n_trades(self) -> int:
        """Всего сделок за прогон (ТЗ 10).

        Сделкой считается любое изменение веса бумаги: и вход, и выход, и
        доведение доли до целевой. Именно за них платятся издержки ТЗ 8, поэтому
        считать иначе значило бы отчитываться не о том, что оплачено.
        """
        return int(self.trades.sum())

    @property
    def risk_off_share(self) -> float:
        """Доля ребалансировок в защитном режиме — для сравнения версий (ТЗ 7.1)."""
        total = self.n_rebalances
        return len(self.risk_off_dates) / total if total else 0.0

    @property
    def returns_net(self) -> pd.Series:
        return self.equity_net.pct_change().dropna()

    @property
    def returns_gross(self) -> pd.Series:
        return self.equity_gross.pct_change().dropna()

    @property
    def annual_turnover(self) -> float:
        """Оборот в долях капитала за год (ТЗ 10)."""
        return float(self.turnover.mean() * 12) if len(self.turnover) else 0.0

    @property
    def n_rebalances(self) -> int:
        return len(self.weights)

    @property
    def average_holding_months(self) -> float:
        """Средний срок удержания в месяцах (ТЗ 10).

        Считается из оборота: при обороте t за ребалансировку средняя позиция
        живёт 1/t месяцев.
        """
        mean_turnover = float(self.turnover.mean()) if len(self.turnover) else 0.0
        return float("inf") if mean_turnover <= 0 else 1.0 / mean_turnover


def run_backtest(
    panel: PricePanel,
    securities: pd.DataFrame,
    *,
    score_fn,
    universe_rules: UniverseRules,
    portfolio_rules: PortfolioRules,
    cost_model: CostModel,
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    delisting_returns: pd.Series | None = None,
    regime: RegimeFilter | None = None,
    stop_rules: StopLossRules | None = None,
    limits: PositionLimits | None = None,
    throttle: TradeThrottle | None = None,
) -> BacktestResult:
    """Прогоняет стратегию на панели цен.

    Args:
        score_fn: `(panel, as_of, universe) -> pd.Series` — итоговый балл по
            бумагам вселенной, уже нормализованный. Нормализация живёт в стратегии,
            а не здесь: value усредняет z-оценки четырёх компонентов по отдельности
            (ТЗ 6.3), и одним вызовом на выходе это не выражается. Движок отвечает
            за отбор, исполнение и учёт, но не за то, что считать баллом.
        delisting_returns: доходность ушедших бумаг. Пропуск означает −100%
            для всех (ТЗ 4.1).
        regime: режимный фильтр (ТЗ 7.1). None означает прогон без фильтра —
            именно с ним ТЗ требует сравнивать версию с фильтром.
        stop_rules: стоп-лосс. Вне ТЗ, по умолчанию выключен; результат обязан
            сравниваться с версией без него.
        limits: лимит на долю одной бумаги.
        throttle: ограничитель числа стоп-выходов за неделю.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if panel.trading_days.empty:
        raise ValueError("Панель цен пуста: бэктест не на чем прогонять.")

    delisting_returns = (
        pd.Series(dtype="float64") if delisting_returns is None else delisting_returns
    )
    stop_rules = stop_rules or StopLossRules()
    limits = limits or PositionLimits()
    throttle = throttle or TradeThrottle()
    closeadj = panel.closeadj.ffill()
    last_alive = panel.closeadj.apply(lambda col: col.last_valid_index())

    rebalance_dates = [d for d in panel.month_end_dates() if start <= d <= end]
    if not rebalance_dates:
        raise ValueError(f"Между {start.date()} и {end.date()} нет дат ребалансировки.")

    schedule: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_day in rebalance_dates:
        execution_day = panel.next_trading_day(signal_day)
        # Последняя ребалансировка периода исполняться некогда — сигнал без
        # следующего дня не сделка, а намерение.
        if execution_day is not None and execution_day <= end:
            schedule[execution_day] = signal_day

    if not schedule:
        raise ValueError("Ни одну ребалансировку не удалось исполнить внутри периода.")

    days = panel.trading_days[
        (panel.trading_days >= min(schedule)) & (panel.trading_days <= end)
    ]

    positions: dict[int, float] = {}
    cash = 1.0
    cost_factor = 1.0

    equity_net: list[float] = []
    equity_gross: list[float] = []
    weights_log: dict[pd.Timestamp, pd.Series] = {}
    turnover_log: dict[pd.Timestamp, float] = {}
    costs_log: dict[pd.Timestamp, float] = {}
    universe_log: dict[pd.Timestamp, int] = {}
    universe_members: dict[pd.Timestamp, pd.Index] = {}
    trades_log: dict[pd.Timestamp, int] = {}
    delisted_hits = 0
    risk_off_dates: list[pd.Timestamp] = []

    # Состояние стоп-лосса. entries — цена входа по каждой открытой позиции,
    # pending — то, что пробило стоп вчера и продаётся сегодня по открытию.
    entries: dict[int, float] = {}
    pending_stops: set[int] = set()
    quarantine: dict[int, pd.Timestamp] = {}
    stop_history: list[pd.Timestamp] = []
    stops_log: list[tuple[pd.Timestamp, int]] = []
    throttled = 0

    previous_day: pd.Timestamp | None = None
    recorded: list[pd.Timestamp] = []
    invested = False

    for today in days:
        # Цена открытия нужна в двух случаях: ребалансировка и исполнение стопов,
        # пробитых вчера. В остальные дни хватает переоценки от закрытия к закрытию.
        priced_at_open = today in schedule or bool(pending_stops)

        if previous_day is not None:
            to_prices = panel.openadj if priced_at_open else closeadj
            positions = _revalue(positions, closeadj, to_prices, previous_day, today)

            positions, cash, hits = _settle_delistings(
                positions, cash, last_alive, today, delisting_returns
            )
            delisted_hits += hits

            if pending_stops:
                positions, cash, executed, deferred = _execute_stops(
                    positions, cash, pending_stops, throttle, stop_history, today,
                )
                for permaticker in executed:
                    entries.pop(permaticker, None)
                    quarantine[permaticker] = stop_rules.quarantine_until(today)
                    stop_history.append(today)
                    stops_log.append((today, permaticker))
                throttled += len(deferred)
                pending_stops = deferred

        if today in schedule:
            signal_day = schedule[today]
            # Фильтр смотрит на рынок в день сигнала, а не исполнения (ТЗ 7.1).
            risk_off = regime is not None and regime.is_risk_off(signal_day)

            target: pd.Series | None = None
            if risk_off:
                # Вселенная в защитном режиме не считается вовсе. Записать сюда
                # ноль значило бы сказать «к покупке ничего не было доступно» —
                # неправда, и она портит медиану размера вселенной в отчёте.
                risk_off_dates.append(signal_day)
                target = regime.risk_off_weights(today)
                target = _tradable_at_execution_weights(target, last_alive, today)
            else:
                universe = build_universe(panel, securities, signal_day, universe_rules)
                universe_log[signal_day] = len(universe)
                universe_members[signal_day] = universe.index
                if len(universe) == 0:
                    # До первой покупки это разогрев, а не сбой: истории цен нет.
                    report = log.info if not invested else log.warning
                    report("%s: вселенная пуста, портфель не меняется", signal_day.date())
                else:
                    scores = score_fn(panel, signal_day, universe)
                    # Выбывшие по стопу не рассматриваются, пока идёт карантин.
                    # Убираются до отбора, а не после: иначе портфель просто
                    # недосчитается позиций вместо того, чтобы взять следующих.
                    #
                    # Срок сверяется с днём сигнала, а не исполнения. Разница в
                    # один день решает всё: стоп, сработавший в начале месяца,
                    # истекает ровно перед следующей ребалансировкой, и бумага
                    # возвращается в портфель — та самая карусель, ради отказа от
                    # которой карантин и введён.
                    blocked = [p for p, until in quarantine.items() if signal_day < until]
                    scores = scores.drop(index=blocked, errors="ignore")
                    selected = select_portfolio(
                        scores, universe["sector"], portfolio_rules,
                        held=pd.Index(list(positions)),
                    )
                    selected = _tradable_at_execution(selected, last_alive, today)
                    target = limits.apply(equal_weights(selected))

            if target is not None:
                equity = sum(positions.values()) + cash
                if equity <= 0:
                    raise RuntimeError(f"Капитал обнулился к {today.date()}.")

                current = pd.Series(positions, dtype="float64") / equity
                # Оборот считается по всем затронутым бумагам, включая продаваемые,
                # а они могли выпасть из вселенной — их ставку издержек надо знать.
                liquidity = _average_dollar_volume(
                    panel, signal_day, universe_rules.dollar_volume_window
                )
                cost = rebalance_cost(current, target, liquidity, cost_model)
                turnover_log[today] = turnover(current, target)
                trades_log[today] = _count_trades(current, target)
                costs_log[today] = cost
                weights_log[today] = target

                equity_after = equity * (1.0 - cost)
                cost_factor *= 1.0 - cost
                opened = {int(p): float(w * equity_after) for p, w in target.items()}
                cash = equity_after * float(1.0 - target.sum()) if len(target) else equity_after
                entries = _update_entries(entries, opened, panel.openadj.loc[today])
                positions = opened
                invested = invested or bool(positions) or risk_off

        if priced_at_open:
            # От открытия к закрытию — уже итоговым составом дня.
            positions = _revalue(positions, panel.openadj, closeadj, today, today)

        # Вечером проверяем стопы: условие по закрытию, сделка завтра по открытию.
        # Внутридневного минимума у нас нет и быть не должно (ТЗ 2).
        if stop_rules.enabled and positions:
            live = {p: e for p, e in entries.items() if p in positions}
            pending_stops |= stop_rules.triggered(live, closeadj.loc[today])

        if invested:
            value = sum(positions.values()) + cash
            equity_net.append(value)
            equity_gross.append(value / cost_factor if cost_factor > 0 else np.nan)
            recorded.append(today)
        previous_day = today

    if not recorded:
        raise ValueError(
            "Ни одной позиции за весь период: вселенная ни разу не набралась. "
            "Проверьте пороги ТЗ 5 и глубину истории цен."
        )

    index = pd.DatetimeIndex(recorded, name="date")
    return BacktestResult(
        equity_net=pd.Series(equity_net, index=index, name="equity_net"),
        equity_gross=pd.Series(equity_gross, index=index, name="equity_gross"),
        weights=weights_log,
        turnover=pd.Series(turnover_log, name="turnover").sort_index(),
        costs=pd.Series(costs_log, name="costs").sort_index(),
        universe_size=pd.Series(universe_log, name="universe_size").sort_index(),
        delisted_hits=delisted_hits,
        risk_off_dates=risk_off_dates,
        universe_members=universe_members,
        trades=pd.Series(trades_log, name="trades", dtype="int64").sort_index(),
        stops=stops_log,
        throttled=throttled,
    )


def _tradable_at_execution(
    selected: pd.Index, last_alive: pd.Series, execution_day: pd.Timestamp
) -> pd.Index:
    """Отсеивает бумаги, которые к моменту сделки уже не торгуются.

    Вселенная считается на день сигнала, сделка проходит на следующий день (ТЗ 7).
    За эту ночь бумага может уйти с биржи. Купить её нельзя — а без проверки движок
    покупал, и делистинг списывался с неё дважды: до ребалансировки и после.

    Та же проверка на живых деньгах — сверка с `/v2/assets` перед отправкой заявок
    (ТЗ 4.6.1). Здесь она обязана быть по той же причине.
    """
    alive = [
        p for p in selected
        if pd.notna(last_alive.get(p)) and last_alive.get(p) >= execution_day
    ]
    if len(alive) < len(selected):
        log.debug(
            "%s: к исполнению не дожили %d бумаг", execution_day.date(),
            len(selected) - len(alive),
        )
    return pd.Index(alive, name="permaticker")


def _update_entries(
    entries: dict[int, float], positions: dict[int, float], open_prices: pd.Series
) -> dict[int, float]:
    """Цены входа после ребалансировки.

    У новой позиции вход — открытие дня исполнения. У продолжающейся цена входа
    сохраняется: изменение веса сделкой по входу не является, и обнулять счётчик
    просадки при доведении доли было бы неправдой.
    """
    updated = {}
    for permaticker in positions:
        if permaticker in entries:
            updated[permaticker] = entries[permaticker]
        else:
            price = open_prices.get(permaticker)
            if price is not None and pd.notna(price) and price > 0:
                updated[permaticker] = float(price)
    return updated


def _execute_stops(
    positions: dict[int, float],
    cash: float,
    pending: set[int],
    throttle: TradeThrottle,
    history: list[pd.Timestamp],
    today: pd.Timestamp,
) -> tuple[dict[int, float], float, list[int], set[int]]:
    """Продаёт по открытию позиции, пробившие стоп вчера.

    Порядок исполнения при работающем ограничителе — по возрастанию permaticker.
    Выбор произволен, но он обязан быть детерминированным: иначе результат
    прогона зависел бы от порядка обхода словаря, и повторить его было бы нельзя.
    """
    wanted = sorted(p for p in pending if p in positions)
    allowed = throttle.allowed(history, today)
    executing, deferred = wanted[:allowed], set(wanted[allowed:])

    if deferred:
        log.warning(
            "%s: ограничитель отложил %d стоп-выходов из %d (не больше %s за неделю)",
            today.date(), len(deferred), len(wanted), throttle.max_trades_per_week,
        )

    for permaticker in executing:
        cash += positions.pop(permaticker, 0.0)

    # Позиции, исчезнувшие по другой причине (делистинг), из очереди убираются.
    return positions, cash, executing, deferred


def _count_trades(before: pd.Series, after: pd.Series) -> int:
    """Сколько бумаг сменили вес на этой ребалансировке.

    Порог в одну сотую процентного пункта отсекает численный шум: доля 1/30
    после переоценки редко получается точно такой же, но сделкой это не является.
    """
    names = before.index.union(after.index)
    changes = after.reindex(names).fillna(0.0) - before.reindex(names).fillna(0.0)
    return int((changes.abs() > 1e-6).sum())


def _tradable_at_execution_weights(
    target: pd.Series, last_alive: pd.Series, execution_day: pd.Timestamp
) -> pd.Series:
    """То же, что `_tradable_at_execution`, но для готовых весов.

    Защитный актив тоже надо проверять: делистинг ETF редок, но подставить в
    портфель бумагу, которой на эту дату нет, — способ получить плоскую кривую
    вместо честного результата.
    """
    alive = _tradable_at_execution(pd.Index(target.index), last_alive, execution_day)
    if len(alive) == len(target):
        return target
    kept = target.reindex(alive)
    total = kept.sum()
    return kept / total if total > 0 else pd.Series(dtype="float64", name="weight")


def _average_dollar_volume(
    panel: PricePanel, as_of: pd.Timestamp, window: int
) -> pd.Series:
    """Средний дневной оборот по всем бумагам панели — для ставки издержек (ТЗ 8)."""
    day = panel.trading_days.get_loc(as_of)
    return panel.dollar_volume.iloc[max(0, day - window + 1): day + 1].mean(skipna=True)


def _revalue(
    positions: dict[int, float],
    from_prices: pd.DataFrame,
    to_prices: pd.DataFrame,
    from_day: pd.Timestamp,
    to_day: pd.Timestamp,
) -> dict[int, float]:
    """Переоценка позиций между двумя ценовыми срезами.

    Отсутствующая цена оставляет позицию как есть: остановка торгов не доход и не
    убыток. Уход бумаги навсегда обрабатывается отдельно, в `_settle_delistings`.
    """
    if not positions:
        return positions

    start = from_prices.loc[from_day]
    finish = to_prices.loc[to_day]
    out: dict[int, float] = {}
    for permaticker, value in positions.items():
        p0 = start.get(permaticker, np.nan)
        p1 = finish.get(permaticker, np.nan)
        ratio = p1 / p0 if (p0 and p0 > 0 and np.isfinite(p0) and np.isfinite(p1)) else 1.0
        out[permaticker] = value * float(ratio)
    return out


def _settle_delistings(
    positions: dict[int, float],
    cash: float,
    last_alive: pd.Series,
    today: pd.Timestamp,
    delisting_returns: pd.Series,
) -> tuple[dict[int, float], float, int]:
    """Закрывает позиции по бумагам, которые перестали торговаться (ТЗ 4.1).

    Именно здесь банкротство превращается в −100%, а не в исчезновение строки.
    Остаток уходит в деньги и ждёт ближайшей ребалансировки.
    """
    survivors: dict[int, float] = {}
    hits = 0
    for permaticker, value in positions.items():
        last_day = last_alive.get(permaticker)
        if last_day is not None and pd.notna(last_day) and today > last_day:
            ret = float(delisting_returns.get(permaticker, DEFAULT_DELISTING_RETURN))
            cash += value * (1.0 + ret)
            hits += 1
        else:
            survivors[permaticker] = value
    return survivors, cash, hits
