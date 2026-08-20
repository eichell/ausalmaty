"""Deflated Sharpe Ratio (ТЗ 9.2.5, Bailey & López de Prado).

Обычный Sharpe отвечает на вопрос «отличается ли результат от нуля». Это не тот
вопрос. После двадцати прогонов с разными параметрами лучший из них будет
выглядеть прилично даже на чистом шуме: максимум двадцати случайных величин
сдвинут вправо просто по устройству максимума.

DSR отвечает на правильный вопрос: какова вероятность, что найденный результат не
объясняется перебором. Поправка состоит из трёх частей:

*   **Число испытаний.** Чем больше прогонов, тем выше планка. Число берётся из
    журнала `experiments.log` (ТЗ 9.2.1) — вот зачем его ведут.
*   **Разброс результатов между испытаниями.** Если все прогоны дают похожий
    Sharpe, максимум сдвинут слабо; если разброс велик — сильно. Оценивается по
    карте чувствительности (ТЗ 9.2.2).
*   **Форма распределения доходностей.** Отрицательная асимметрия и толстые
    хвосты делают тот же Sharpe менее надёжным. Стратегия, которая долго
    зарабатывает по чуть-чуть и изредка теряет много, обычная в акциях, и
    обычный Sharpe её переоценивает.

Sharpe везде **не годовой**: формула работает в тех единицах, в которых считаются
доходности, и число наблюдений входит в неё отдельно.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

log = logging.getLogger(__name__)

#: Постоянная Эйлера — Маскерони. Входит в оценку матожидания максимума выборки.
EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DeflatedSharpe:
    """Результат поправки. `probability` — то, ради чего всё считалось."""

    sharpe: float
    expected_max_sharpe: float
    probability: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float

    @property
    def survives(self) -> bool:
        """Принято считать результат состоятельным при DSR выше 0.95."""
        return self.probability > 0.95

    def summary(self) -> str:
        verdict = "проходит" if self.survives else "НЕ ПРОХОДИТ"
        return (
            f"DSR = {self.probability:.3f} ({verdict}) при {self.n_trials} испытаниях; "
            f"Sharpe {self.sharpe:.3f} против порога перебора {self.expected_max_sharpe:.3f}"
        )


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Матожидание максимального Sharpe из `n_trials` испытаний при нулевом сигнале.

    Это и есть планка, которую надо перейти: столько «зарабатывает» чистый перебор.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    sigma = math.sqrt(sharpe_variance)
    left = norm.ppf(1.0 - 1.0 / n_trials)
    right = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * left + EULER_MASCHERONI * right))


def deflated_sharpe_ratio(
    returns: pd.Series, *, n_trials: int, sharpe_variance: float | None = None
) -> DeflatedSharpe:
    """Вероятность того, что Sharpe не объясняется перебором.

    Args:
        returns: доходности стратегии в своей исходной частоте (дневные).
        n_trials: число проведённых испытаний — из журнала ТЗ 9.2.1.
        sharpe_variance: разброс Sharpe между испытаниями. Если не задан, берётся
            консервативная оценка 1/(T−1) — дисперсия Sharpe при нулевом сигнале.

    Raises:
        ValueError: если наблюдений меньше трёх — считать нечего.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3:
        raise ValueError(f"Для DSR нужно хотя бы три наблюдения, получено {n}.")

    sigma = clean.std(ddof=1)
    sharpe = float(clean.mean() / sigma) if sigma > 0 else 0.0
    g3 = float(skew(clean, bias=False))
    # Полный (не избыточный) четвёртый момент: формула Bailey ждёт именно его.
    g4 = float(kurtosis(clean, fisher=False, bias=False))

    if sharpe_variance is None:
        sharpe_variance = 1.0 / (n - 1)
        log.debug("Разброс Sharpe между испытаниями не задан, взята оценка 1/(T−1)")

    threshold = expected_max_sharpe(sharpe_variance, n_trials)

    denominator = 1.0 - g3 * sharpe + (g4 - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        # Бывает при сильной асимметрии и высоком Sharpe: формула перестаёт быть
        # определённой. Честнее отказаться, чем выдать красивое число.
        log.warning("Знаменатель DSR неположителен (%.4f): поправка не определена", denominator)
        probability = float("nan")
    else:
        statistic = (sharpe - threshold) * math.sqrt(n - 1) / math.sqrt(denominator)
        probability = float(norm.cdf(statistic))

    return DeflatedSharpe(
        sharpe=sharpe, expected_max_sharpe=threshold, probability=probability,
        n_trials=n_trials, n_observations=n, skewness=g3, kurtosis=g4,
    )


def count_experiments(path: str | Path = "experiments.log") -> int:
    """Число испытаний из журнала (ТЗ 9.2.1).

    Считаются строки прогонов; комментарии и пустые строки пропускаются. Ноль
    означает, что журнал пуст, — для DSR это трактуется как одно испытание.
    """
    path = Path(path)
    if not path.exists():
        log.warning("Журнал испытаний %s не найден: число испытаний принято за 1", path)
        return 1

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return max(len(lines), 1)


def sharpe_variance_from_trials(sharpes: pd.Series | np.ndarray) -> float:
    """Разброс Sharpe между испытаниями — по карте чувствительности (ТЗ 9.2.2)."""
    values = pd.Series(sharpes).dropna()
    if len(values) < 2:
        return 0.0
    return float(values.var(ddof=1))
