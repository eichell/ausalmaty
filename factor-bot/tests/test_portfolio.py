"""Формирование портфеля (ТЗ 7) и обязательные тесты ТЗ 12."""

from __future__ import annotations

import pandas as pd
import pytest

from factorbot.portfolio import (
    PortfolioRules,
    equal_weights,
    sector_weights,
    select_portfolio,
)

RULES = PortfolioRules(top_n=10, buffer_rank=15, max_sector_weight=0.30)


def _scores(n: int) -> pd.Series:
    """Балл убывает с номером: у бумаги 0 он самый высокий."""
    return pd.Series({i: float(n - i) for i in range(n)})


def _sectors(n: int, groups: int = 5) -> pd.Series:
    return pd.Series({i: f"S{i % groups}" for i in range(n)})


def test_weights_sum_to_one(): 
    """ТЗ 12: сумма весов портфеля равна 1 на каждую дату."""
    selected = select_portfolio(_scores(60), _sectors(60), RULES)
    assert equal_weights(selected).sum() == pytest.approx(1.0)


def test_weights_sum_to_one_even_when_the_universe_is_thin():
    scores = _scores(4)
    selected = select_portfolio(scores, _sectors(4, groups=4), RULES)
    weights = equal_weights(selected)
    assert len(weights) == 4
    assert weights.sum() == pytest.approx(1.0)


def test_sector_limit_is_respected():
    """ТЗ 12: соблюдение лимита по сектору. Все бумаги одного сектора — в портфель
    должно попасть не больше лимита, даже если лучших по баллу больше."""
    n = 60
    single_sector = pd.Series({i: "Technology" for i in range(n)})
    selected = select_portfolio(_scores(n), single_sector, RULES)

    assert len(selected) == RULES.max_names_per_sector == 3
    weights = equal_weights(selected)
    assert sector_weights(weights, single_sector).max() == pytest.approx(1.0)


def test_sector_cap_pushes_out_the_worst_of_the_sector_not_the_best():
    n = 30
    sectors = pd.Series({i: ("Technology" if i < 10 else f"S{i}") for i in range(n)})
    selected = select_portfolio(_scores(n), sectors, RULES)
    tech = [p for p in selected if sectors[p] == "Technology"]
    assert tech == [0, 1, 2], "остаться должны три лучших по баллу, а не любые три"


def test_sector_share_never_exceeds_the_limit():
    """Лимит ТЗ 7 задан в весах, поэтому проверяется на весах, а не на числе бумаг."""
    n = 60
    sectors = pd.Series({i: f"S{i % 11}" for i in range(n)})     # 11 секторов GICS
    selected = select_portfolio(_scores(n), sectors, RULES)
    shares = sector_weights(equal_weights(selected), sectors)
    assert shares.max() <= RULES.max_sector_weight + 1e-12


def test_thin_universe_does_not_smuggle_a_sector_over_the_limit():
    """Портфель не набрался до top_n: вес каждой бумаги вырос, и лимит в штуках
    от top_n перестал соответствовать лимиту в весах."""
    n = 8                                        # кандидатов меньше, чем top_n = 10
    sectors = pd.Series({i: ("A" if i < 3 else f"S{i}") for i in range(n)})
    selected = select_portfolio(_scores(n), sectors, RULES)

    # По лимиту от top_n трём бумагам сектора A место есть, по фактическим
    # весам — нет: 3 из 8 это 37.5%.
    assert 2 not in selected, "убрать должны худшую по баллу бумагу сектора"
    assert len(selected) == 7
    shares = sector_weights(equal_weights(selected), sectors)
    assert shares.max() <= RULES.max_sector_weight + 1e-12


def test_unreachable_limit_is_reported_not_silently_violated(caplog):
    """Три сектора и порог 30%: любое распределение даёт кому-то 34%. Убирать
    бумаги бессмысленно — веса оставшихся только вырастут."""
    n = 60
    sectors = pd.Series({i: f"S{i % 3}" for i in range(n)})
    with caplog.at_level("WARNING"):
        selected = select_portfolio(_scores(n), sectors, RULES)
    assert len(selected) > 0
    assert any("недостижим" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Буферизация (ТЗ 7)
# --------------------------------------------------------------------------- #


def test_incumbent_inside_the_buffer_is_kept_over_a_slightly_better_challenger():
    n = 40
    sectors = pd.Series({i: f"S{i}" for i in range(n)})
    held = pd.Index([12])                       # 13-е место: вне топ-10, но в топ-15
    selected = select_portfolio(_scores(n), sectors, RULES, held=held)
    assert 12 in selected
    assert len(selected) == RULES.top_n


def test_incumbent_below_the_buffer_is_dropped():
    n = 40
    sectors = pd.Series({i: f"S{i}" for i in range(n)})
    selected = select_portfolio(_scores(n), sectors, RULES, held=pd.Index([30]))
    assert 30 not in selected


def test_buffer_reduces_churn_between_two_similar_dates():
    n = 40
    sectors = pd.Series({i: f"S{i}" for i in range(n)})
    first = select_portfolio(_scores(n), sectors, RULES)

    # Небольшая перетасовка около границы отсечки.
    shuffled = _scores(n).copy()
    shuffled[9], shuffled[10] = shuffled[10], shuffled[9]

    without_buffer = select_portfolio(shuffled, sectors, PortfolioRules(10, 10, 0.30))
    with_buffer = select_portfolio(shuffled, sectors, RULES, held=first)

    churn_without = len(set(first) - set(without_buffer))
    churn_with = len(set(first) - set(with_buffer))
    assert churn_with <= churn_without


def test_a_name_that_left_the_universe_cannot_be_held():
    n = 20
    sectors = pd.Series({i: f"S{i}" for i in range(n)})
    selected = select_portfolio(_scores(n), sectors, RULES, held=pd.Index([999]))
    assert 999 not in selected


def test_missing_scores_are_excluded_not_ranked_last():
    scores = pd.Series({1: 1.0, 2: float("nan"), 3: 3.0})
    sectors = pd.Series({1: "A", 2: "B", 3: "C"})
    assert list(select_portfolio(scores, sectors, RULES)) == [3, 1]


def test_empty_scores_give_an_empty_portfolio():
    selected = select_portfolio(pd.Series(dtype="float64"), pd.Series(dtype="string"), RULES)
    assert len(selected) == 0
    assert equal_weights(selected).empty
