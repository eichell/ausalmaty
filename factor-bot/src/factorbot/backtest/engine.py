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
from factorbot.normalize import normalize_within_sector
from factorbot.portfolio import PortfolioRules, equal_weights, select_portfolio
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
) -> BacktestResult:
    """Прогоняет стратегию на панели цен.

    Args:
        score_fn: `(panel, as_of, universe) -> pd.Series` — сырой балл по бумагам
            вселенной. Нормализацию и отбор делает движок, чтобы этап 2 и этап 4
            отличались только этой функцией.
        delisting_returns: доходность ушедших бумаг. Пропуск означает −100%
            для всех (ТЗ 4.1).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if panel.trading_days.empty:
        raise ValueError("Панель цен пуста: бэктест не на чем прогонять.")

    delisting_returns = (
        pd.Series(dtype="float64") if delisting_returns is None else delisting_returns
    )
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
    delisted_hits = 0

    previous_day: pd.Timestamp | None = None
    recorded: list[pd.Timestamp] = []
    invested = False

    for today in days:
        if previous_day is not None:
            if today in schedule:
                positions = _revalue(positions, closeadj, panel.openadj, previous_day, today)
            else:
                positions = _revalue(positions, closeadj, closeadj, previous_day, today)

            positions, cash, hits = _settle_delistings(
                positions, cash, last_alive, today, delisting_returns
            )
            delisted_hits += hits

        if today in schedule:
            signal_day = schedule[today]
            universe = build_universe(panel, securities, signal_day, universe_rules)
            universe_log[signal_day] = len(universe)

            if len(universe) == 0:
                # До первой покупки это разогрев, а не сбой: истории цен ещё нет.
                report = log.info if not invested else log.warning
                report("%s: вселенная пуста, портфель не меняется", signal_day.date())
            else:
                raw = score_fn(panel, signal_day, universe)
                z = normalize_within_sector(raw, universe["sector"])
                selected = select_portfolio(
                    z, universe["sector"], portfolio_rules,
                    held=pd.Index(list(positions)),
                )
                selected = _tradable_at_execution(selected, last_alive, today)
                target = equal_weights(selected)

                equity = sum(positions.values()) + cash
                if equity <= 0:
                    raise RuntimeError(f"Капитал обнулился к {today.date()}.")

                current = pd.Series(positions, dtype="float64") / equity
                cost = rebalance_cost(
                    current, target, universe["dollar_volume"], cost_model
                )
                turnover_log[today] = turnover(current, target)
                costs_log[today] = cost
                weights_log[today] = target

                equity_after = equity * (1.0 - cost)
                cost_factor *= 1.0 - cost
                positions = {int(p): float(w * equity_after) for p, w in target.items()}
                cash = 0.0
                invested = invested or bool(positions)

            # Шаг 3: от открытия к закрытию уже новым составом.
            positions = _revalue(positions, panel.openadj, closeadj, today, today)

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
