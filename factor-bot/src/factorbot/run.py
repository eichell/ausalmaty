"""Прогон бэктеста (ТЗ 13, этап 2).

    python -m factorbot.run --period in_sample --note "базовый momentum"

Доступны momentum и value по отдельности, их композит и режимный фильтр.
Фильтр отключаемый, и `--regime both` прогоняет обе версии рядом: ТЗ 7.1 требует
обязательно сравнить результат с версией без фильтра.

Каждый прогон дописывается в `experiments.log` (ТЗ 9.2.1). Это не отчётность:
число испытаний входит в поправку Deflated Sharpe Ratio, без которой лучший
результат неотличим от результата перебора.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from factorbot.backtest.costs import CostModel
from factorbot.backtest.delisting import build_delisting_returns
from factorbot.backtest.engine import run_backtest
from factorbot.config import load_config, load_dotenv
from factorbot.data.panel import load_price_panel
from factorbot.data.periods import PERIODS, open_period
from factorbot.factors.composite import CompositeRules, combine
from factorbot.factors.momentum import momentum
from factorbot.factors.value import ValueRules, compute_yields, value_score
from factorbot.normalize import normalize_within_sector
from factorbot.portfolio import PortfolioRules
from factorbot.regime import RegimeRules, build_regime_filter
from factorbot.report import metrics as M
from factorbot.universe import UniverseRules

log = logging.getLogger("factorbot.run")

EXPERIMENTS_LOG = Path("experiments.log")


def _momentum_z(cfg):
    """z-оценка импульса. Общая для этапа 2 и композита — расчёт должен быть один."""

    def z(panel, as_of, universe):
        values = momentum(
            panel, as_of,
            lookback_days=int(cfg.factors.momentum.lookback_days),
            skip_days=int(cfg.factors.momentum.skip_days),
        )
        return normalize_within_sector(values.reindex(universe.index), universe["sector"])

    return z


def _value_z(cfg, conn):
    """z-оценка value. Фундаментал читается на каждой дате заново (ТЗ 4.8)."""
    rules = ValueRules.from_config(cfg.factors, cfg.universe)

    def z(panel, as_of, universe):
        yields = compute_yields(conn, panel, as_of, list(universe.index))
        return value_score(yields, universe["sector"], rules).reindex(universe.index)

    return z


def momentum_only(cfg, conn):
    """Балл этапа 2: чистый импульс, без value и без фильтра (ТЗ 13.2)."""
    return _momentum_z(cfg)


def value_only(cfg, conn):
    """Балл этапа 3: композит доходностей, без momentum и без фильтра (ТЗ 13.3)."""
    return _value_z(cfg, conn)


def composite(cfg, conn):
    """Балл этапа 4: 0.5 * z_momentum + 0.5 * z_value (ТЗ 6.3, 13.4).

    Веса берутся из конфига и не подбираются по результату (ТЗ 9.2.3).
    """
    rules = CompositeRules.from_config(cfg.factors)
    momentum_z, value_z = _momentum_z(cfg), _value_z(cfg, conn)

    def score(panel, as_of, universe):
        return combine({
            "momentum": momentum_z(panel, as_of, universe),
            "value": value_z(panel, as_of, universe),
        }, rules)

    return score


STRATEGIES = {"momentum": momentum_only, "value": value_only, "composite": composite}


def load_benchmark(conn, ticker: str) -> pd.Series:
    """Дневная кривая бенчмарка из тех же цен. Пусто, если бумаги нет в базе."""
    df = conn.execute(
        """
        SELECT p.date, p.closeadj
        FROM prices p JOIN securities s USING (permaticker)
        WHERE upper(s.ticker) = upper($ticker) AND p.closeadj IS NOT NULL
        ORDER BY p.date
        """,
        {"ticker": ticker},
    ).df()
    if df.empty:
        log.warning("Бенчмарк %s в базе не найден: сравнение с ним пропущено.", ticker)
        return pd.Series(dtype="float64")
    return df.set_index(pd.to_datetime(df["date"]))["closeadj"].rename(ticker)


def append_experiment(note: str, strategy: str, period: str, result_line: str) -> None:
    """Дописывает строку в журнал испытаний (ТЗ 9.2.1)."""
    line = (
        f"{datetime.now():%Y-%m-%d %H:%M} | {strategy}/{period} | "
        f"{note or 'без комментария'} | {result_line}\n"
    )
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    log.info("Записано в %s", EXPERIMENTS_LOG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогон бэктеста factorbot")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--strategy", default="momentum", choices=sorted(STRATEGIES))
    parser.add_argument("--period", default="in_sample", choices=sorted(PERIODS))
    parser.add_argument("--note", default="", help="что изменено в этом прогоне (ТЗ 9.2.1)")
    parser.add_argument(
        "--regime", default="config", choices=["config", "on", "off", "both"],
        help="режимный фильтр ТЗ 7.1; both прогоняет обе версии рядом",
    )
    parser.add_argument("--plots", action="store_true", help="сохранить графики")
    parser.add_argument("--out", default="data/processed/report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    load_dotenv()
    cfg = load_config(args.config)
    period = PERIODS[args.period]

    # Соединение живёт весь прогон: value читает фундаментал на каждой дате
    # ребалансировки, и закрывать базу до конца бэктеста нельзя.
    conn = open_period(args.period, cfg.data.processed_dir)
    try:
        panel = load_price_panel(conn, end=period.end)
        securities = conn.execute("SELECT * FROM securities").df()
        corp_actions = conn.execute("SELECT * FROM corp_actions").df()
        benchmark = load_benchmark(conn, cfg.reporting.benchmark)

        last_prices = panel.closeadj.ffill().iloc[-1] if len(panel.closeadj) else pd.Series()
        delisting = build_delisting_returns(securities, corp_actions, last_prices)

        base = RegimeRules.from_config(cfg.regime_filter)
        wanted = {
            "config": [base.enabled], "on": [True], "off": [False], "both": [False, True],
        }[args.regime]

        results = {}
        for enabled in wanted:
            rules = replace(base, enabled=enabled)
            results[enabled] = run_backtest(
                panel, securities,
                score_fn=STRATEGIES[args.strategy](cfg, conn),
                universe_rules=UniverseRules.from_config(cfg.universe),
                portfolio_rules=PortfolioRules.from_config(cfg.portfolio),
                cost_model=CostModel.from_config(cfg.costs),
                start=period.start,
                end=min(period.end, date.today()),
                delisting_returns=delisting,
                regime=build_regime_filter(panel, securities, rules),
            )
    finally:
        conn.close()

    primary = results[wanted[-1]]
    result = primary
    net = M.summarize(primary, benchmark)
    gross = M.summarize(primary, benchmark, gross=True)
    _print_report(args, primary, net, gross, benchmark, filtered=wanted[-1])

    if len(results) > 1:
        _print_regime_comparison(results, benchmark)

    append_experiment(
        args.note, f"{args.strategy}{'+filter' if wanted[-1] else ''}", args.period,
        f"CAGR {net.cagr:.2%} (до издержек {gross.cagr:.2%}), Sharpe {net.sharpe:.2f}, "
        f"maxDD {net.max_drawdown:.1%}, оборот {net.annual_turnover:.0%}/год",
    )

    if args.plots:
        from factorbot.report import plots

        out = Path(args.out)
        plots.equity_curve(
            result.equity_net, result.equity_gross, benchmark,
            path=out / f"equity_{args.strategy}_{args.period}.png",
            title=f"{args.strategy} / {args.period}",
        )
        plots.drawdown_curve(
            result.equity_net, path=out / f"drawdown_{args.strategy}_{args.period}.png"
        )
        log.info("Графики: %s", out)

    return 0


def _print_regime_comparison(results: dict[bool, object], benchmark) -> None:
    """ТЗ 7.1: результаты с фильтром обязательно сравнить с версией без него."""
    off, on = M.summarize(results[False], benchmark), M.summarize(results[True], benchmark)
    print("\n=== режимный фильтр: с ним и без него (ТЗ 7.1) ===")
    print(f"{'':28}{'без фильтра':>14}{'с фильтром':>14}")
    for label, attr, fmt in [
        ("CAGR", "cagr", "{:.2%}"),
        ("Волатильность", "volatility", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Макс. просадка", "max_drawdown", "{:.1%}"),
        ("Просадка, мес.", "max_drawdown_months", "{:.0f}"),
        ("Отставание от SPY, мес.", "max_underperformance_months", "{:.0f}"),
        ("Оборот в год", "annual_turnover", "{:.0%}"),
    ]:
        print(f"{label:<28}{fmt.format(getattr(off, attr)):>14}"
              f"{fmt.format(getattr(on, attr)):>14}")
    share = results[True].risk_off_share
    print(f"{'Ребалансировок в защите':<28}{'—':>14}{share:>13.0%}")


def _print_report(args, result, net: M.Metrics, gross: M.Metrics, benchmark,
                  *, filtered: bool = False) -> None:
    suffix = " + режимный фильтр" if filtered else ""
    print(f"\n=== {args.strategy}{suffix} / {args.period} ===")
    print(f"Период:                  {result.equity_net.index[0].date()} — "
          f"{result.equity_net.index[-1].date()}  ({net.years:.1f} лет)")
    print(f"Ребалансировок:          {net.n_rebalances}")
    print(f"Вселенная (медиана):     {int(result.universe_size.median())} бумаг")
    print(f"Делистингов в портфеле:  {result.delisted_hits}")
    if filtered:
        print(f"Ребалансировок в защите: {len(result.risk_off_dates)} "
              f"({result.risk_off_share:.0%})")
    print()
    print(f"{'':24}{'после издержек':>16}{'до издержек':>16}")
    for label, attr, fmt in [
        ("CAGR", "cagr", "{:.2%}"),
        ("Волатильность", "volatility", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Макс. просадка", "max_drawdown", "{:.1%}"),
        ("Просадка, мес.", "max_drawdown_months", "{:.0f}"),
    ]:
        print(f"{label:<24}{fmt.format(getattr(net, attr)):>16}"
              f"{fmt.format(getattr(gross, attr)):>16}")
    print()
    if not benchmark.empty:
        print(f"Отставание от {benchmark.name}, мес.: {net.max_underperformance_months:.0f}")
    print(f"Оборот:                  {net.annual_turnover:.0%} в год")
    print(f"Средний срок удержания:  {net.average_holding_months:.1f} мес.")

    yearly = M.yearly_returns(result.equity_net, benchmark)
    print("\nПо годам:")
    print(yearly.to_string(float_format=lambda v: f"{v:7.2%}"))

    for flag in M.sanity_warnings(net):
        print(f"\n[!] {flag}")


if __name__ == "__main__":
    sys.exit(main())
