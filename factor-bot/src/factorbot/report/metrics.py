"""Метрики отчёта (ТЗ 10).

Обязательный набор задан ТЗ. Одна цифра в нём выделена особо — максимальная
длительность отставания от SPY в месяцах. Она важнее Sharpe по практической
причине: просадку в 40% инвестор переживает, если рынок падает вместе с ним, а
пять лет отставания от индекса при растущем рынке не выдерживает почти никто, и
стратегию бросают на дне относительной кривой.

Безрисковая ставка принята нулевой. В ТЗ её нет; на горизонте 1999–2012 это
завышает Sharpe примерно на 0.2. Цифра сравнивается с самой собой между версиями,
поэтому смещение общее для всех прогонов, но помнить о нём надо.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    """Обязательные метрики ТЗ 10."""

    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_months: float
    max_underperformance_months: float
    annual_turnover: float
    n_rebalances: int
    average_holding_months: float
    total_return: float
    years: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def cagr(equity: pd.Series) -> float:
    """Среднегодовой темп роста."""
    years = _years(equity)
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    sigma = excess.std(ddof=1)
    if not sigma or not np.isfinite(sigma):
        return float("nan")
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Как Sharpe, но в знаменателе только отклонения вниз."""
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess.loc[excess < 0]
    if downside.empty:
        return float("inf")
    sigma = np.sqrt((downside**2).mean())
    if not sigma:
        return float("nan")
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS_PER_YEAR))


def drawdown(equity: pd.Series) -> pd.Series:
    """Просадка от предыдущего максимума, в долях."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def longest_underwater_months(equity: pd.Series) -> float:
    """Самый длинный период под предыдущим максимумом, в месяцах.

    Отрезок, не успевший восстановиться к концу выборки, тоже считается: инвестор
    в нём находится прямо сейчас, и делать вид, что он не считается, нечестно.
    """
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    underwater = equity < peaks

    longest = pd.Timedelta(0)
    current_start: pd.Timestamp | None = None
    for day, is_under in underwater.items():
        if is_under and current_start is None:
            current_start = day
        elif not is_under and current_start is not None:
            longest = max(longest, day - current_start)
            current_start = None
    if current_start is not None:
        longest = max(longest, equity.index[-1] - current_start)

    return float(longest.days / 30.44)


def relative_equity(equity: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Кривая относительно бенчмарка, нормированная на начало общего отрезка."""
    common = equity.index.intersection(benchmark.index)
    if common.empty:
        return pd.Series(dtype="float64")
    ratio = (equity.loc[common] / equity.loc[common].iloc[0]) / (
        benchmark.loc[common] / benchmark.loc[common].iloc[0]
    )
    return ratio


def yearly_returns(equity: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Доходность по календарным годам, рядом с бенчмарком (ТЗ 10)."""
    by_year = equity.resample("YE").last() / equity.resample("YE").first() - 1
    out = pd.DataFrame({"strategy": by_year})
    if benchmark is not None and not benchmark.empty:
        common = equity.index.intersection(benchmark.index)
        bench = benchmark.loc[common]
        out["benchmark"] = bench.resample("YE").last() / bench.resample("YE").first() - 1
        out["excess"] = out["strategy"] - out["benchmark"]
    out.index = out.index.year
    out.index.name = "year"
    return out


def summarize(result, benchmark: pd.Series | None = None, *, gross: bool = False) -> Metrics:
    """Полный набор ТЗ 10 по результату прогона."""
    equity = result.equity_gross if gross else result.equity_net
    returns = equity.pct_change().dropna()

    underperformance = 0.0
    if benchmark is not None and not benchmark.empty:
        relative = relative_equity(equity, benchmark)
        underperformance = longest_underwater_months(relative)

    return Metrics(
        cagr=cagr(equity),
        volatility=volatility(returns),
        sharpe=sharpe(returns),
        sortino=sortino(returns),
        max_drawdown=max_drawdown(equity),
        max_drawdown_months=longest_underwater_months(equity),
        max_underperformance_months=underperformance,
        annual_turnover=result.annual_turnover,
        n_rebalances=result.n_rebalances,
        average_holding_months=result.average_holding_months,
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1),
        years=_years(equity),
    )


def sanity_warnings(metrics: Metrics) -> list[str]:
    """Признаки, что что-то не так (ТЗ 9.3).

    Не проверка корректности, а список поводов искать ошибку у себя. Реалистичный
    ориентир для честно построенной версии: CAGR на несколько процентных пунктов
    выше S&P 500, Sharpe 0.6–0.9, максимальная просадка 35–50%.
    """
    flags: list[str] = []
    if metrics.sharpe > 1.5:
        flags.append(
            f"Sharpe {metrics.sharpe:.2f} > 1.5 на месячной ребалансировке акций — "
            "для честно построенной версии это слишком хорошо (ТЗ 9.3)."
        )
    if metrics.max_drawdown > -0.25:
        flags.append(
            f"Максимальная просадка {metrics.max_drawdown:.1%} мельче 25% — "
            "проверьте, попали ли в выборку 2000-е и 2008 год (ТЗ 9.3)."
        )
    return flags


def _years(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float((equity.index[-1] - equity.index[0]).days / 365.25)
