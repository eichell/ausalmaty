"""Стоп-лосс, лимит на бумагу, ограничитель сделок. Всё это вне ТЗ."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_panel, make_securities, trading_days

from factorbot.backtest.costs import CostModel
from factorbot.backtest.engine import run_backtest
from factorbot.portfolio import PortfolioRules
from factorbot.risk import PositionLimits, StopLossRules, TradeThrottle
from factorbot.universe import UniverseRules

# --------------------------------------------------------------------------- #
# Правила стопа
# --------------------------------------------------------------------------- #


def test_stop_price_is_measured_from_the_entry():
    rules = StopLossRules(enabled=True, threshold=0.30)
    assert rules.stop_price(100.0) == pytest.approx(70.0)


def test_position_at_the_threshold_is_stopped():
    rules = StopLossRules(enabled=True, threshold=0.30)
    hit = rules.triggered({1: 100.0}, pd.Series({1: 70.0}))
    assert hit == {1}


def test_position_just_above_the_threshold_survives():
    rules = StopLossRules(enabled=True, threshold=0.30)
    assert rules.triggered({1: 100.0}, pd.Series({1: 70.01})) == set()


def test_disabled_rules_never_trigger():
    rules = StopLossRules(enabled=False, threshold=0.30)
    assert rules.triggered({1: 100.0}, pd.Series({1: 1.0})) == set()


def test_missing_price_does_not_trigger_a_stop():
    """Остановка торгов — не повод считать позицию проданной."""
    rules = StopLossRules(enabled=True, threshold=0.30)
    assert rules.triggered({1: 100.0}, pd.Series({1: np.nan})) == set()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_impossible_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="между 0 и 1"):
        StopLossRules(enabled=True, threshold=bad)


def test_narrow_stop_is_warned_about(caplog):
    """Стоп в 10% на месячной стратегии режет обычную волатильность и работает
    против value, который по построению покупает падавшее."""
    with caplog.at_level("WARNING"):
        StopLossRules(enabled=True, threshold=0.10)
    assert any("против value" in r.getMessage() for r in caplog.records)


def test_wide_stop_raises_no_warning(caplog):
    with caplog.at_level("WARNING"):
        StopLossRules(enabled=True, threshold=0.30)
    assert not caplog.records


def test_quarantine_runs_for_the_configured_months():
    rules = StopLossRules(enabled=True, quarantine_months=1)
    assert rules.quarantine_until(pd.Timestamp("2005-06-15")) == pd.Timestamp("2005-07-15")


# --------------------------------------------------------------------------- #
# Лимит на бумагу
# --------------------------------------------------------------------------- #


def test_limit_does_not_bind_on_a_full_portfolio():
    """При 30 бумагах доля каждой 3.33% — лимит в 5% ничего не меняет."""
    weights = pd.Series({i: 1 / 30 for i in range(30)})
    capped = PositionLimits(max_position_weight=0.05).apply(weights)
    assert capped.equals(weights)


def test_thin_portfolio_is_capped_and_the_rest_stays_in_cash():
    weights = pd.Series({i: 0.2 for i in range(5)})
    capped = PositionLimits(max_position_weight=0.05).apply(weights)
    assert (capped == 0.05).all()
    assert capped.sum() == pytest.approx(0.25), "остаток обязан уйти в деньги"


def test_remainder_is_not_redistributed_to_the_others():
    """Раздать остаток значило бы нарушить тот же лимит у соседей."""
    weights = pd.Series({1: 0.5, 2: 0.5})
    capped = PositionLimits(max_position_weight=0.10).apply(weights)
    assert capped.max() == pytest.approx(0.10)


def test_no_limit_configured_changes_nothing():
    weights = pd.Series({1: 0.9, 2: 0.1})
    assert PositionLimits(None).apply(weights).equals(weights)


# --------------------------------------------------------------------------- #
# Ограничитель сделок
# --------------------------------------------------------------------------- #


def test_throttle_counts_only_the_last_seven_days():
    throttle = TradeThrottle(max_trades_per_week=3)
    today = pd.Timestamp("2005-06-15")
    recent = [pd.Timestamp("2005-06-14"), pd.Timestamp("2005-06-01")]
    assert throttle.allowed(recent, today) == 2


def test_throttle_blocks_when_the_week_is_used_up():
    throttle = TradeThrottle(max_trades_per_week=2)
    today = pd.Timestamp("2005-06-15")
    recent = [pd.Timestamp("2005-06-14"), pd.Timestamp("2005-06-13")]
    assert throttle.allowed(recent, today) == 0


def test_no_throttle_configured_allows_everything():
    assert TradeThrottle(None).allowed([pd.Timestamp("2005-06-14")] * 100,
                                       pd.Timestamp("2005-06-15")) > 1000


# --------------------------------------------------------------------------- #
# Через движок
# --------------------------------------------------------------------------- #


DAYS = trading_days(600, start="2004-01-01")
START, END = pd.Timestamp("2005-06-01"), DAYS[-1]
NO_COSTS = CostModel(0.0, 0.0, 0.0)
RULES = PortfolioRules(top_n=2, buffer_rank=3, max_sector_weight=1.0)


def _crash_panel(crash_day: int, level: float):
    """Бумага 2 обваливается до `level` от цены входа, бумага 1 стоит."""
    flat = [50.0] * len(DAYS)
    crashing = [50.0] * crash_day + [50.0 * level] * (len(DAYS) - crash_day)
    return make_panel({1: flat, 2: crashing}, DAYS)


def _run(panel, *, stop=None, throttle=None, limits=None, rules=RULES):
    return run_backtest(
        panel, make_securities([1, 2]),
        score_fn=lambda p, d, u: pd.Series({1: 2.0, 2: 1.0}).reindex(u.index),
        universe_rules=UniverseRules(), portfolio_rules=rules,
        cost_model=NO_COSTS, start=START, end=END,
        stop_rules=stop, throttle=throttle, limits=limits,
    )


def test_crash_below_the_threshold_closes_the_position():
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    result = _run(_crash_panel(crash, 0.5), stop=StopLossRules(enabled=True, threshold=0.30))

    assert result.n_stops == 1
    assert result.stops[0][1] == 2


def test_shallow_drop_does_not_trigger_the_stop():
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    result = _run(_crash_panel(crash, 0.85), stop=StopLossRules(enabled=True, threshold=0.30))
    assert result.n_stops == 0


def test_stop_executes_at_the_next_open_not_the_trigger_close():
    """Внутридневных данных нет (ТЗ 2): условие по закрытию, сделка завтра."""
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    result = _run(_crash_panel(crash, 0.5), stop=StopLossRules(enabled=True, threshold=0.30))
    executed = result.stops[0][0]
    assert executed > DAYS[crash], "продажа не может состояться в день срабатывания"


def test_disabled_stop_changes_nothing():
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = _crash_panel(crash, 0.5)
    without = _run(panel)
    disabled = _run(panel, stop=StopLossRules(enabled=False, threshold=0.30))
    assert disabled.equity_net.iloc[-1] == pytest.approx(without.equity_net.iloc[-1])
    assert disabled.n_stops == 0


def test_gap_down_is_not_saved_by_the_stop():
    """Стоп не спасает от гэпа: бумага, открывшаяся на 80% ниже, будет продана
    по этой цене, а не по уровню стопа. Ровно так это работает и в жизни."""
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = _crash_panel(crash, 0.2)
    without = _run(panel)
    with_stop = _run(panel, stop=StopLossRules(enabled=True, threshold=0.30))
    assert with_stop.equity_net.iloc[-1] == pytest.approx(
        without.equity_net.iloc[-1], rel=0.02
    )


def test_stop_cuts_a_gradual_decline():
    """А вот от медленного сползания спасает: выход около порога избавляет от
    остатка падения."""
    start = DAYS.get_loc(pd.Timestamp("2005-08-01"))
    prices = [50.0] * start
    prices += list(np.linspace(50.0, 5.0, len(DAYS) - start))
    panel = make_panel({1: [50.0] * len(DAYS), 2: prices}, DAYS)

    without = _run(panel)
    with_stop = _run(panel, stop=StopLossRules(enabled=True, threshold=0.30))

    # Стопов больше одного: во вселенной всего две бумаги при портфеле из двух,
    # поэтому после карантина падающую обязаны купить снова.
    assert with_stop.n_stops >= 1
    assert with_stop.equity_net.iloc[-1] > without.equity_net.iloc[-1] * 1.1


def test_quarantine_keeps_the_name_out_of_the_next_rebalance():
    """Без карантина стоп превращается в карусель: продали, купили обратно,
    заплатили дважды за то, чтобы остаться при своих."""
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    # После обвала бумага восстанавливается — без карантина её бы откупили.
    prices = [50.0] * crash + [20.0] * 5 + [50.0] * (len(DAYS) - crash - 5)
    panel = make_panel({1: [50.0] * len(DAYS), 2: prices}, DAYS)

    result = _run(panel, stop=StopLossRules(enabled=True, threshold=0.30,
                                            quarantine_months=1))
    stopped_on = result.stops[0][0]
    after = [d for d in result.weights if d > stopped_on][:1]
    assert after, "должна быть хотя бы одна ребалансировка после стопа"
    assert 2 not in result.weights[after[0]].index


def test_throttle_defers_extra_exits():
    crash = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = make_panel(
        {p: ([50.0] * crash + [10.0] * (len(DAYS) - crash)) for p in (1, 2)}, DAYS
    )
    result = run_backtest(
        panel, make_securities([1, 2]),
        score_fn=lambda p, d, u: pd.Series({1: 2.0, 2: 1.0}).reindex(u.index),
        universe_rules=UniverseRules(), portfolio_rules=RULES,
        cost_model=NO_COSTS, start=START, end=END,
        stop_rules=StopLossRules(enabled=True, threshold=0.30),
        throttle=TradeThrottle(max_trades_per_week=1),
    )
    assert result.throttled > 0, "второй выход обязан быть отложен"
    assert result.n_stops == 2, "но в итоге исполниться оба"


def test_position_limit_leaves_cash_when_the_portfolio_is_thin():
    panel = make_panel({1: [50.0] * len(DAYS), 2: [50.0] * len(DAYS)}, DAYS)
    result = _run(panel, limits=PositionLimits(max_position_weight=0.20))
    weights = list(result.weights.values())[0]
    assert weights.sum() == pytest.approx(0.40)
    assert (weights == 0.20).all()


def test_capped_portfolio_keeps_the_rest_in_cash_not_in_stocks():
    """Капитал не должен молча раствориться: доля в деньгах — часть эквити."""
    panel = make_panel({1: [50.0] * len(DAYS), 2: [50.0] * len(DAYS)}, DAYS)
    result = _run(panel, limits=PositionLimits(max_position_weight=0.20))
    assert result.equity_net.iloc[-1] == pytest.approx(1.0), "цены плоские"
