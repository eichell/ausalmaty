"""Карты чувствительности и тесты на переподгонку (ТЗ 9.2).

Три инструмента, отвечающие на три разных вопроса.

**Карта чувствительности (ТЗ 9.2.2).** Требовать плато, а не пик. Прогон с
окном momentum 9/12/15 месяцев, портфелем 20/30/40/50 бумаг, порогом SMA
150/200/250. Если результат хорош только при одном значении и резко падает у
соседних — это подгонка, а не эффект. Настоящий фактор работает на диапазоне,
потому что рынок не знает, что вы выбрали именно 252 дня.

**Тест на перемешанных данных (ТЗ 9.2.4).** Сигналы случайно перетасовываются
между бумагами, всё остальное остаётся как есть. Если стратегия «работает» и
так, в коде утечка будущего.

Одна оговорка к формулировке ТЗ, и она важна на практике. Буквально «результат
должен быть неотличим от нуля» на перемешанных сигналах не выполняется никогда:
случайный портфель из тридцати акций на растущем рынке зарабатывает — это
рыночная бета, а не сигнал. Более того, для long-only портфеля акций Sharpe
перемешанного прогона почти всегда близок к настоящему, потому что обоими движет
одна и та же общая компонента.

Поэтому читать надо не уровни, а две другие величины:

*   **p-значение** — доля перемешанных прогонов, не уступивших настоящему. Это
    ответ на вопрос «есть ли у фактора преимущество над случайным отбором».
*   **Перемешанный результат против бенчмарка.** Случайный отбор из вселенной —
    это примерно равновзвешенный индекс, и обгонять рынок с большим отрывом он
    не должен. Если обгоняет, дело не в факторе, и вот это уже повод искать
    утечку. Без бенчмарка вопрос не решается, и тест честно говорит, что не знает.

**Deflated Sharpe (ТЗ 9.2.5).** Число испытаний берётся из журнала, разброс
Sharpe между испытаниями — из карты чувствительности. Обе цифры нужны формуле, и
обе появляются здесь естественным образом.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from factorbot.config import with_overrides
from factorbot.report import metrics as M
from factorbot.report.deflated_sharpe import (
    count_experiments,
    deflated_sharpe_ratio,
    sharpe_variance_from_trials,
)

log = logging.getLogger(__name__)

#: Насколько соседние значения параметра могут уступать лучшему, чтобы это
#: считалось плато. Порог грубый и намеренно мягкий: задача — ловить одинокие
#: пики, а не назначать точную границу.
PLATEAU_TOLERANCE = 0.6

#: На сколько перемешанный прогон должен обойти бенчмарк, чтобы это выглядело
#: подозрительно. Правило большого пальца, а не закон: случайный отбор из
#: вселенной обычно чуть лучше индекса за счёт крена в малую капитализацию,
#: но не на половину единицы Sharpe.
LEAK_MARGIN = 0.5


@dataclass(frozen=True)
class SweepSpec:
    """Одна ось карты чувствительности."""

    name: str
    path: str
    values: Sequence[Any]
    #: Параметры, которые обязаны меняться вместе с основным.
    linked: dict[str, Callable[[Any], Any]] = field(default_factory=dict)

    def overrides(self, value: Any) -> dict[str, Any]:
        out = {self.path: value}
        out.update({path: fn(value) for path, fn in self.linked.items()})
        return out


#: Сетка из ТЗ 9.2.2. Окна momentum заданы в торговых днях: 9/12/15 месяцев по
#: 21 дню. Буфер привязан к размеру портфеля: в ТЗ 7 это 45 при 30 бумагах, то
#: есть полтора состава. Оставить его постоянным нельзя — при портфеле в 50 бумаг
#: буферная зона оказалась бы меньше самого портфеля.
DEFAULT_SWEEPS: tuple[SweepSpec, ...] = (
    SweepSpec("окно momentum, мес.", "factors.momentum.lookback_days", [189, 252, 315]),
    SweepSpec("размер портфеля", "portfolio.top_n", [20, 30, 40, 50],
              linked={"portfolio.buffer_rank": lambda n: int(round(1.5 * n))}),
    SweepSpec("порог SMA", "regime_filter.sma_window", [150, 200, 250]),
)


@dataclass(frozen=True)
class SweepVerdict:
    """Плато или пик — и почему так решено."""

    spec_name: str
    table: pd.DataFrame
    best_value: Any
    best_metric: float
    neighbour_ratio: float
    is_plateau: bool

    def summary(self) -> str:
        if not np.isfinite(self.best_metric) or self.best_metric <= 0:
            return f"{self.spec_name}: сигнала нет ни при одном значении"
        verdict = "плато" if self.is_plateau else "ПИК — похоже на подгонку (ТЗ 9.2.2)"
        return (
            f"{self.spec_name}: лучшее {self.best_value} ({self.best_metric:.2f}), "
            f"соседи держат {self.neighbour_ratio:.0%} от него — {verdict}"
        )


def run_sweep(
    runner: Callable[[dict[str, Any]], pd.Series],
    spec: SweepSpec,
    *,
    metric: str = "sharpe",
) -> SweepVerdict:
    """Прогоняет стратегию по всем значениям одного параметра.

    Args:
        runner: `overrides -> Series метрик`. Возвращать должен как минимум
            колонку `metric`.
    """
    rows = {}
    for value in spec.values:
        log.info("%s = %s", spec.name, value)
        rows[value] = runner(spec.overrides(value))

    table = pd.DataFrame(rows).T
    table.index.name = spec.path
    return _verdict(spec, table, metric)


def _verdict(spec: SweepSpec, table: pd.DataFrame, metric: str) -> SweepVerdict:
    series = table[metric].astype("float64")
    best_position = int(series.to_numpy().argmax())
    best_value = series.index[best_position]
    best = float(series.iloc[best_position])

    neighbours = [
        series.iloc[i] for i in (best_position - 1, best_position + 1)
        if 0 <= i < len(series)
    ]
    if not neighbours or best <= 0:
        ratio = 0.0
    else:
        ratio = float(np.mean(neighbours) / best)

    return SweepVerdict(
        spec_name=spec.name, table=table, best_value=best_value, best_metric=best,
        neighbour_ratio=ratio, is_plateau=bool(best > 0 and ratio >= PLATEAU_TOLERANCE),
    )


# --------------------------------------------------------------------------- #
# Перемешанные данные (ТЗ 9.2.4)
# --------------------------------------------------------------------------- #


def shuffle_scores(rng: np.random.Generator) -> Callable:
    """Обёртка, перемешивающая баллы между бумагами на каждой дате.

    Распределение баллов сохраняется полностью, рушится только соответствие
    «балл — бумага». Значит всё, что останется в результате, идёт от механики
    портфеля и от рынка, но не от фактора.
    """

    def wrapper(score_fn):
        def shuffled(panel, as_of, universe):
            scores = score_fn(panel, as_of, universe)
            values = scores.to_numpy(dtype="float64", copy=True)
            rng.shuffle(values)
            return pd.Series(values, index=scores.index, name="shuffled")

        return shuffled

    return wrapper


@dataclass(frozen=True)
class ShuffleTest:
    """Результат перестановочного теста."""

    real: float
    shuffled: list[float]
    metric: str
    #: Та же метрика у бенчмарка. Без неё вопрос об утечке не решается.
    baseline: float | None = None

    @property
    def shuffled_median(self) -> float:
        return float(np.median(self.shuffled)) if self.shuffled else float("nan")

    @property
    def p_value(self) -> float:
        """Доля перемешанных прогонов, не уступивших настоящему.

        Единица в числителе и знаменателе — стандартная поправка: настоящий
        прогон сам по себе является одной из перестановок.
        """
        beaten = sum(1 for value in self.shuffled if value >= self.real)
        return (1 + beaten) / (1 + len(self.shuffled))

    @property
    def has_edge(self) -> bool:
        """Есть ли у фактора преимущество над случайным отбором."""
        return self.p_value <= 0.05

    @property
    def leak_suspected(self) -> bool | None:
        """Обгоняет ли случайный отбор рынок с неправдоподобным отрывом.

        Именно это, а не близость к настоящему прогону, является признаком
        утечки: для long-only портфеля акций перемешанный Sharpe и так близок к
        настоящему, потому что обоими движет рыночная бета.

        Returns:
            None, если бенчмарк не задан — вопрос остаётся без ответа, и делать
            вид, что ответ есть, нельзя.
        """
        if self.baseline is None or not self.shuffled:
            return None
        return bool(self.shuffled_median > self.baseline + LEAK_MARGIN)

    def summary(self) -> str:
        parts = [
            f"{self.metric}: настоящий {self.real:.2f}, перемешанные (медиана) "
            f"{self.shuffled_median:.2f}, p = {self.p_value:.3f}"
        ]
        if not self.has_edge:
            parts.append(
                "\n  Преимущества над случайным отбором не видно: фактор не "
                "отличим от случайного выбора бумаг."
            )
        leak = self.leak_suspected
        if leak is None:
            parts.append(
                "\n  Бенчмарк не задан — обгоняет ли случайный отбор рынок, "
                "проверить нечем (ТЗ 9.2.4)."
            )
        elif leak:
            parts.append(
                f"\n  ВОЗМОЖНА УТЕЧКА БУДУЩЕГО: случайный отбор обгоняет бенчмарк "
                f"({self.baseline:.2f}) с большим отрывом (ТЗ 9.2.4)."
            )
        return "".join(parts)


def shuffle_test(
    runner: Callable[[Callable | None], pd.Series],
    *,
    n_shuffles: int = 20,
    metric: str = "sharpe",
    seed: int = 20240101,
    baseline: float | None = None,
) -> ShuffleTest:
    """Сравнивает настоящий прогон с прогонами на перемешанных сигналах."""
    real = float(runner(None)[metric])
    rng = np.random.default_rng(seed)
    shuffled = [float(runner(shuffle_scores(rng))[metric]) for _ in range(n_shuffles)]
    return ShuffleTest(
        real=real, shuffled=shuffled, metric=metric, baseline=baseline
    )


# --------------------------------------------------------------------------- #
# Сборка отчёта ТЗ 9.2
# --------------------------------------------------------------------------- #


def _benchmark_metric(benchmark: pd.Series, metric: str) -> float | None:
    """Та же метрика у купил-и-держи бенчмарка — планка для перемешанных прогонов."""
    if benchmark is None or benchmark.empty:
        return None
    returns = benchmark.pct_change().dropna()
    if len(returns) < 3:
        return None
    return {
        "sharpe": lambda: M.sharpe(returns),
        "cagr": lambda: M.cagr(benchmark),
        "max_drawdown": lambda: M.max_drawdown(benchmark),
        "turnover": lambda: 0.0,
    }[metric]()


def metrics_row(result, benchmark: pd.Series | None = None) -> pd.Series:
    """Строка метрик одного прогона — то, из чего складываются карты."""
    net = M.summarize(result, benchmark)
    return pd.Series({
        "cagr": net.cagr,
        "sharpe": net.sharpe,
        "max_drawdown": net.max_drawdown,
        "turnover": net.annual_turnover,
    })


def overfitting_report(
    verdicts: Sequence[SweepVerdict],
    shuffle: ShuffleTest | None,
    returns: pd.Series,
    *,
    experiments_log: str = "experiments.log",
) -> str:
    """Сводка раздела 9.2 одним текстом."""
    lines = ["=== защита от переподгонки (ТЗ 9.2) ==="]

    lines.append("\nКарты чувствительности (ТЗ 9.2.2):")
    for verdict in verdicts:
        lines.append("  " + verdict.summary())

    if shuffle is not None:
        lines.append("\nПеремешанные данные (ТЗ 9.2.4):")
        lines.append("  " + shuffle.summary())

    all_sharpes = pd.concat(
        [v.table["sharpe"] for v in verdicts]
    ) if verdicts else pd.Series(dtype="float64")
    n_trials = count_experiments(experiments_log)
    variance = sharpe_variance_from_trials(all_sharpes)

    lines.append("\nDeflated Sharpe (ТЗ 9.2.5):")
    try:
        dsr = deflated_sharpe_ratio(
            returns, n_trials=n_trials,
            sharpe_variance=variance if variance > 0 else None,
        )
        lines.append("  " + dsr.summary())
        if not dsr.survives:
            lines.append(
                "  Результат объясним перебором. Это не приговор стратегии, "
                "но принимать её как есть нельзя."
            )
    except ValueError as exc:
        lines.append(f"  посчитать не удалось: {exc}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """python -m factorbot.sensitivity --strategy composite --period in_sample"""
    import argparse

    from factorbot.config import load_config, load_dotenv
    from factorbot.data.periods import PERIODS
    from factorbot.run import execute, open_run_context

    parser = argparse.ArgumentParser(description="Защита от переподгонки (ТЗ 9.2)")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--strategy", default="composite")
    parser.add_argument("--period", default="in_sample", choices=sorted(PERIODS))
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "cagr", "max_drawdown", "turnover"])
    parser.add_argument("--shuffles", type=int, default=20,
                        help="сколько прогонов на перемешанных сигналах (ТЗ 9.2.4)")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    load_dotenv()
    base_cfg = load_config(args.config)
    context = open_run_context(base_cfg, args.period)

    try:
        def run_with(overrides: dict[str, Any]) -> pd.Series:
            context.cfg = with_overrides(base_cfg, overrides)
            result = execute(context, args.strategy)
            return metrics_row(result, context.benchmark)

        verdicts = [run_sweep(run_with, spec, metric=args.metric)
                    for spec in DEFAULT_SWEEPS]

        context.cfg = base_cfg
        baseline = execute(context, args.strategy)

        shuffle = None
        if not args.no_shuffle:
            def run_shuffled(wrapper) -> pd.Series:
                context.cfg = base_cfg
                result = execute(context, args.strategy, score_wrapper=wrapper)
                return metrics_row(result, context.benchmark)

            shuffle = shuffle_test(
                run_shuffled, n_shuffles=args.shuffles, metric=args.metric,
                baseline=_benchmark_metric(context.benchmark, args.metric),
            )
    finally:
        context.close()

    for verdict in verdicts:
        print(f"\n--- {verdict.spec_name} ---")
        print(verdict.table.to_string(float_format=lambda v: f"{v:8.3f}"))

    print()
    print(overfitting_report(verdicts, shuffle, baseline.returns_net))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
