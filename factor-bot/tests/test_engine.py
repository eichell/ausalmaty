"""Движок бэктеста (ТЗ 7, 8) и обязательные тесты ТЗ 12."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import first_execution_day, make_panel, make_securities, trading_days

from factorbot.backtest.costs import CostModel
from factorbot.backtest.engine import run_backtest
from factorbot.portfolio import PortfolioRules
from factorbot.universe import UniverseRules

DAYS = trading_days(600, start="2004-01-01")
START, END = pd.Timestamp("2005-06-01"), DAYS[-1]

UNIVERSE = UniverseRules()
NO_COSTS = CostModel(commission_bps=0.0, slippage_bps_liquid=0.0, slippage_bps_illiquid=0.0)


def fixed_scores(ranking: dict[int, float]):
    """Балл задан вручную: движок проверяется отдельно от фактора."""

    def score(panel, as_of, universe):
        return pd.Series(ranking).reindex(universe.index)

    return score


def run(panel, securities, *, ranking, rules=None, costs=NO_COSTS, delisting=None):
    return run_backtest(
        panel, securities,
        score_fn=fixed_scores(ranking),
        universe_rules=UNIVERSE,
        portfolio_rules=rules or PortfolioRules(top_n=2, buffer_rank=3, max_sector_weight=1.0),
        cost_model=costs,
        start=START, end=END,
        delisting_returns=delisting,
    )


# --------------------------------------------------------------------------- #
# Базовая механика
# --------------------------------------------------------------------------- #


def test_weights_sum_to_one_at_every_rebalance():
    """ТЗ 12."""
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2, 3, 4)}, DAYS)
    result = run(panel, make_securities([1, 2, 3, 4]), ranking={1: 4, 2: 3, 3: 2, 4: 1})

    assert result.n_rebalances > 5
    for weights in result.weights.values():
        assert weights.sum() == pytest.approx(1.0)


def test_flat_prices_keep_the_capital_flat():
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2)}, DAYS)
    result = run(panel, make_securities([1, 2]), ranking={1: 2, 2: 1})
    assert result.equity_net.iloc[-1] == pytest.approx(1.0)


def test_a_doubling_stock_doubles_a_single_name_portfolio():
    prices = np.linspace(50.0, 100.0, len(DAYS))
    panel = make_panel({1: list(prices), 2: [50.0] * len(DAYS)}, DAYS)
    result = run(
        panel, make_securities([1, 2]), ranking={1: 2, 2: 1},
        rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
    )
    # Вход по открытию дня исполнения, а оно в этой панели равно вчерашнему закрытию.
    entry = DAYS.get_loc(first_execution_day(DAYS, START)) - 1
    assert result.equity_net.iloc[-1] == pytest.approx(prices[-1] / prices[entry], rel=1e-6)


# --------------------------------------------------------------------------- #
# Исполнение по открытию следующего дня (ТЗ 7)
# --------------------------------------------------------------------------- #


def test_overnight_jump_before_execution_is_not_captured():
    """Сигнал считается по закрытию, сделка идёт по открытию следующего дня.

    Если бумага улетает вверх на открытии, портфель покупает её уже дорого. Иначе
    бэктест зарабатывает на движении, которое произошло до входа в позицию, —
    ровно тот класс ошибки, ради которого ТЗ 7 требует исполнения по открытию.
    """
    closes = [50.0] * len(DAYS)
    opens = [50.0] * len(DAYS)

    first_exec = DAYS.get_loc(first_execution_day(DAYS, START))
    opens[first_exec] = 75.0                      # гэп вверх на 50%
    for i in range(first_exec, len(DAYS)):
        closes[i] = 75.0

    panel = make_panel({1: closes}, DAYS, opens={1: opens})
    result = run(
        panel, make_securities([1]), ranking={1: 1.0},
        rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
    )
    # Капитал не изменился: вошли по 75 и закрылись по 75.
    assert result.equity_net.loc[DAYS[first_exec]] == pytest.approx(1.0)


def test_move_after_the_open_is_captured():
    """Обратная сторона: то, что произошло уже в позиции, в результат попадает."""
    closes = [50.0] * len(DAYS)
    opens = [50.0] * len(DAYS)

    first_exec = DAYS.get_loc(first_execution_day(DAYS, START))
    for i in range(first_exec, len(DAYS)):
        closes[i] = 75.0                          # рост внутри дня исполнения

    panel = make_panel({1: closes}, DAYS, opens={1: opens})
    result = run(
        panel, make_securities([1]), ranking={1: 1.0},
        rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
    )
    assert result.equity_net.loc[DAYS[first_exec]] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Делистинг (ТЗ 4.1, тест ТЗ 12)
# --------------------------------------------------------------------------- #


def _panel_with_a_death(death_index: int):
    alive = [50.0] * len(DAYS)
    dying = [50.0] * death_index + [np.nan] * (len(DAYS) - death_index)
    return make_panel({1: alive, 2: dying}, DAYS)


def test_delisted_name_gives_minus_one_hundred_percent_not_a_disappearance():
    """ТЗ 12: делистингованная бумага даёт −100%, а не выпадает из расчёта.

    Половина портфеля обнуляется — капитал обязан упасть примерно вдвое. Если бы
    позиция просто исчезала, кривая осталась бы плоской, и бэктест никогда не
    показал бы убыток, которого в жизни было не избежать.
    """
    death = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = _panel_with_a_death(death)
    securities = make_securities([1, 2], delisted={2})

    result = run(panel, securities, ranking={1: 2.0, 2: 1.0})

    assert result.delisted_hits == 1
    assert result.equity_net.iloc[-1] == pytest.approx(0.5, rel=0.02)


def test_known_recovery_value_is_used_instead_of_a_total_loss():
    """Поглощение — не банкротство: если известно, сколько получил держатель,
    считается оно, а не −100%."""
    death = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = _panel_with_a_death(death)
    securities = make_securities([1, 2], delisted={2})

    result = run(
        panel, securities, ranking={1: 2.0, 2: 1.0},
        delisting=pd.Series({2: -0.20}),
    )
    assert result.equity_net.iloc[-1] == pytest.approx(0.9, rel=0.02)


def test_capital_freed_by_a_delisting_is_reinvested_at_the_next_rebalance():
    death = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    panel = _panel_with_a_death(death)
    securities = make_securities([1, 2], delisted={2})

    result = run(panel, securities, ranking={1: 2.0, 2: 1.0},
                 delisting=pd.Series({2: 0.0}))
    # Потери нет, деньги вернулись в бумагу 1 — капитал не должен просесть.
    assert result.equity_net.iloc[-1] == pytest.approx(1.0, rel=1e-6)
    last_weights = list(result.weights.values())[-1]
    assert list(last_weights.index) == [1]


# --------------------------------------------------------------------------- #
# Издержки (ТЗ 8)
# --------------------------------------------------------------------------- #


def test_costs_only_ever_reduce_the_result():
    prices = {1: list(np.linspace(50, 80, len(DAYS))),
              2: list(np.linspace(50, 60, len(DAYS))),
              3: list(np.linspace(50, 40, len(DAYS)))}
    panel = make_panel(prices, DAYS)
    securities = make_securities([1, 2, 3])

    ranking = {1: 3.0, 2: 2.0, 3: 1.0}
    free = run(panel, securities, ranking=ranking)
    charged = run(panel, securities, ranking=ranking, costs=CostModel())

    assert charged.equity_net.iloc[-1] <= free.equity_net.iloc[-1]


def test_gross_equity_is_exactly_net_without_the_cost_drag():
    panel = make_panel({p: list(np.linspace(50, 50 + p, len(DAYS))) for p in (1, 2, 3)}, DAYS)
    result = run(panel, make_securities([1, 2, 3]), ranking={1: 3.0, 2: 2.0, 3: 1.0},
                 costs=CostModel())

    drag = float((1 - result.costs).prod())
    assert result.equity_net.iloc[-1] == pytest.approx(
        result.equity_gross.iloc[-1] * drag, rel=1e-9
    )
    assert result.equity_gross.iloc[-1] > result.equity_net.iloc[-1]


def test_turnover_is_reported_per_year():
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2, 3, 4)}, DAYS)
    result = run(panel, make_securities([1, 2, 3, 4]), ranking={1: 4, 2: 3, 3: 2, 4: 1})
    # Состав не меняется: после первой покупки оборота быть не должно.
    assert result.turnover.iloc[1:].sum() == pytest.approx(0.0)
    assert result.annual_turnover == pytest.approx(result.turnover.mean() * 12)


# --------------------------------------------------------------------------- #
# Отказы
# --------------------------------------------------------------------------- #


def test_empty_panel_is_rejected():
    empty = make_panel({}, DAYS)
    with pytest.raises(ValueError, match="пуста"):
        run_backtest(
            empty, make_securities([]), score_fn=fixed_scores({}),
            universe_rules=UNIVERSE, portfolio_rules=PortfolioRules(),
            cost_model=NO_COSTS, start=START, end=END,
        )


def test_period_without_rebalance_dates_is_rejected():
    panel = make_panel({1: [50.0] * len(DAYS)}, DAYS)
    with pytest.raises(ValueError, match="ребалансировки"):
        run_backtest(
            panel, make_securities([1]), score_fn=fixed_scores({1: 1.0}),
            universe_rules=UNIVERSE, portfolio_rules=PortfolioRules(),
            cost_model=NO_COSTS,
            start=pd.Timestamp("2005-06-06"), end=pd.Timestamp("2005-06-10"),
        )


def test_delisting_warning_actually_renders(caplog):
    """Литеральный процент в шаблоне лога ломает форматирование, и предупреждение
    молча превращается в traceback логгера — то есть исчезает."""
    import logging

    from factorbot.backtest.delisting import build_delisting_returns

    securities = make_securities([1, 2], delisted={2})
    with caplog.at_level(logging.WARNING):
        build_delisting_returns(securities, corp_actions=None)

    messages = [r.getMessage() for r in caplog.records]
    assert any("−100%" in m and "ТЗ 4.1" in m for m in messages)


# --------------------------------------------------------------------------- #
# Учёт для отчёта (ТЗ 10, 4.7)
# --------------------------------------------------------------------------- #


def test_first_rebalance_counts_every_purchase_as_a_trade():
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2)}, DAYS)
    result = run(panel, make_securities([1, 2]), ranking={1: 2, 2: 1})
    first = result.trades.index[0]
    assert result.trades.loc[first] == 2


def test_unchanged_portfolio_makes_no_trades():
    """Иначе число сделок распухнет на численном шуме переоценки весов."""
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2)}, DAYS)
    result = run(panel, make_securities([1, 2]), ranking={1: 2, 2: 1})
    assert result.trades.iloc[1:].sum() == 0
    assert result.n_trades == 2


def test_swapping_a_name_counts_as_two_trades():
    """Продать одну и купить другую — две сделки, и обе оплачены (ТЗ 8)."""
    switch = DAYS.get_loc(pd.Timestamp("2005-09-01"))
    rising = [50.0] * switch + [50.0 + i for i in range(len(DAYS) - switch)]
    panel = make_panel({1: [50.0] * len(DAYS), 2: rising}, DAYS)

    result = run_backtest(
        panel, make_securities([1, 2]),
        score_fn=lambda p, d, u: panel.closeadj.loc[d].reindex(u.index),
        universe_rules=UNIVERSE,
        portfolio_rules=PortfolioRules(top_n=1, buffer_rank=1, max_sector_weight=1.0),
        cost_model=NO_COSTS, start=START, end=END,
    )
    assert result.n_trades > 2


def test_universe_membership_is_recorded_for_every_signal_date():
    """Без состава вселенной нельзя ответить, почему бумага не куплена: балл
    низкий или её вообще не было в отборе (ТЗ 4.7)."""
    panel = make_panel({p: [50.0] * len(DAYS) for p in (1, 2, 3)}, DAYS)
    result = run(panel, make_securities([1, 2, 3]), ranking={1: 3, 2: 2, 3: 1})

    assert set(result.universe_members) == set(result.universe_size.index)
    for day, members in result.universe_members.items():
        assert len(members) == result.universe_size.loc[day]
        assert set(members) <= {1, 2, 3}
