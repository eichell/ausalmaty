"""Графики отчёта (ТЗ 10). Кривая эквити — в логарифмическом масштабе."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def equity_curve(
    equity_net: pd.Series,
    equity_gross: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    *,
    path: str | Path = "equity.png",
    title: str = "Кривая эквити",
) -> Path:
    """Сохраняет кривую эквити в логарифмическом масштабе.

    Логарифм обязателен: в линейном масштабе последние годы визуально съедают
    первые, и просадка 2008 года на графике за четырнадцать лет выглядит рябью.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    normalized = equity_net / equity_net.iloc[0]
    ax.plot(normalized.index, normalized.to_numpy(), label="после издержек", linewidth=1.4)

    if equity_gross is not None and not equity_gross.empty:
        gross = equity_gross / equity_gross.iloc[0]
        ax.plot(gross.index, gross.to_numpy(), label="до издержек",
                linewidth=1.0, alpha=0.7)

    if benchmark is not None and not benchmark.empty:
        common = normalized.index.intersection(benchmark.index)
        if len(common):
            bench = benchmark.loc[common] / benchmark.loc[common].iloc[0]
            ax.plot(bench.index, bench.to_numpy(), label="SPY",
                    linewidth=1.0, linestyle="--")

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("рост капитала, логарифмическая шкала")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def drawdown_curve(equity: pd.Series, *, path: str | Path = "drawdown.png") -> Path:
    """Просадка во времени: длительность видна лучше, чем по одной цифре."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(series.index, series.to_numpy(), 0.0, alpha=0.5)
    ax.set_title("Просадка")
    ax.set_ylabel("доля от максимума")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
