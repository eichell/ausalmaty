"""Доходность при делистинге (ТЗ 4.1).

Модуль переписан после того, как первая версия приняла колонку `value` события
`delisted` за выплату на акцию. На самом деле это капитализация компании в
миллионах долларов, и бэктест получал делистинги с доходностью в сотни раз:
CAGR 1878%, положительный 2008 год. Ни один тест этого не поймал, потому что
арифметику выплаты не проверял никто — движку доходности подавались руками.

Поэтому здесь проверяется именно семантика полей ACTIONS, а не только формула.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from factorbot.backtest.delisting import (
    DEFAULT_DELISTING_RETURN,
    build_delisting_returns,
)

SECURITIES = pd.DataFrame([
    {"permaticker": 1, "ticker": "CASH", "is_delisted": True},
    {"permaticker": 2, "ticker": "STOK", "is_delisted": True},
    {"permaticker": 3, "ticker": "MIXD", "is_delisted": True},
    {"permaticker": 4, "ticker": "BANK", "is_delisted": True},
    {"permaticker": 5, "ticker": "MUTE", "is_delisted": True},
    {"permaticker": 6, "ticker": "LIVE", "is_delisted": False},
])

#: Последние нескорректированные цены: именно в этих долларах указана выплата.
LAST_PRICES = pd.Series({1: 20.0, 2: 15.0, 3: 35.5, 4: 2.0, 5: 7.0, 6: 50.0})


def actions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["permaticker", "date", "action", "value"])


def test_cash_acquisition_pays_the_stated_price_per_share():
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 1, "date": "2005-06-01",
                  "action": "acquisitioncash", "value": 22.0}]),
        LAST_PRICES,
    )
    assert out[1] == pytest.approx(0.10)


def test_market_capitalisation_of_a_delisting_is_never_read_as_a_payout():
    """Тот самый дефект: $11.4 млрд капитализации против цены акции $2.55.

    Событие `delisted` не несёт сведений об оплате, и бумага обязана остаться на
    значении по умолчанию, а не получить доходность в четыре тысячи раз.
    """
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 1, "date": "2006-01-01",
                  "action": "delisted", "value": 11429.1}]),
        LAST_PRICES,
    )
    assert out[1] == DEFAULT_DELISTING_RETURN


@pytest.mark.parametrize("action", [
    "acquisitionby", "acquisitionof", "voluntarydelisting",
    "regulatorydelisting", "mergerfrom",
])
def test_other_capitalisation_fields_are_not_payouts_either(action):
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 1, "date": "2006-01-01",
                  "action": action, "value": 4200.0}]),
        LAST_PRICES,
    )
    assert out[1] <= 0.0


def test_payment_in_shares_of_the_acquirer_is_not_a_loss():
    """Держатель получает бумаги примерно на ту же сумму: это 0%, а не −100%."""
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 2, "date": "2005-06-01",
                  "action": "acquisitionstock", "value": 0.71}]),
        LAST_PRICES,
    )
    assert out[2] == pytest.approx(0.0)


def test_exchange_ratio_is_not_mistaken_for_a_price():
    """0.71 акции покупателя за акцию цели — это не $0.71 выплаты."""
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 2, "date": "2005-06-01",
                  "action": "acquisitionstock", "value": 0.71}]),
        LAST_PRICES,
    )
    assert out[2] > -0.5


def test_mixed_consideration_counts_as_a_stock_deal():
    """Денежная часть смешанной сделки — доплата, а не вся цена.

    $7.07 деньгами при цене $35.50 дали бы −80% на удачном поглощении.
    """
    out = build_delisting_returns(
        SECURITIES,
        actions([
            {"permaticker": 3, "date": "2007-12-31",
             "action": "acquisitioncash", "value": 7.07},
            {"permaticker": 3, "date": "2007-12-31",
             "action": "acquisitionstock", "value": 0.62},
        ]),
        LAST_PRICES,
    )
    assert out[3] == pytest.approx(0.0)


def test_liquidation_is_a_total_loss():
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 4, "date": "2003-04-01",
                  "action": "bankruptcyliquidation", "value": 2.2}]),
        LAST_PRICES,
    )
    assert out[4] == pytest.approx(DEFAULT_DELISTING_RETURN)


def test_security_without_any_event_stays_at_minus_one_hundred_percent():
    out = build_delisting_returns(SECURITIES, actions([]), LAST_PRICES)
    assert (out == DEFAULT_DELISTING_RETURN).all()


def test_still_listed_security_is_absent_from_the_result():
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 6, "date": "2005-06-01",
                  "action": "acquisitioncash", "value": 55.0}]),
        LAST_PRICES,
    )
    assert 6 not in out.index


def test_payout_outside_the_trust_band_is_refused_and_reported(caplog):
    """Выплата в двадцать цен — признак прочитанного не того поля."""
    with caplog.at_level(logging.WARNING):
        out = build_delisting_returns(
            SECURITIES,
            actions([{"permaticker": 1, "date": "2005-06-01",
                      "action": "acquisitioncash", "value": 400.0}]),
            LAST_PRICES,
        )
    assert out[1] == DEFAULT_DELISTING_RETURN
    assert "полосы доверия" in caplog.text


def test_last_of_several_cash_events_wins():
    out = build_delisting_returns(
        SECURITIES,
        actions([
            {"permaticker": 1, "date": "2005-05-01",
             "action": "acquisitioncash", "value": 21.0},
            {"permaticker": 1, "date": "2005-06-01",
             "action": "acquisitioncash", "value": 24.0},
        ]),
        LAST_PRICES,
    )
    assert out[1] == pytest.approx(0.20)


def test_zero_reference_price_does_not_produce_infinity():
    out = build_delisting_returns(
        SECURITIES,
        actions([{"permaticker": 5, "date": "2005-06-01",
                  "action": "acquisitioncash", "value": 3.0}]),
        pd.Series({5: 0.0}),
    )
    assert out[5] == DEFAULT_DELISTING_RETURN


def test_counts_are_reported_by_category(caplog):
    with caplog.at_level(logging.INFO):
        build_delisting_returns(
            SECURITIES,
            actions([
                {"permaticker": 1, "date": "2005-06-01",
                 "action": "acquisitioncash", "value": 22.0},
                {"permaticker": 2, "date": "2005-06-01",
                 "action": "acquisitionstock", "value": 0.71},
                {"permaticker": 4, "date": "2003-04-01",
                 "action": "bankruptcyliquidation", "value": 2.2},
            ]),
            LAST_PRICES,
        )
    assert "деньгами 1" in caplog.text
    assert "акциями 1" in caplog.text
    assert "без данных 2" in caplog.text


# --------------------------------------------------------------------------- #
# Разбор делистингов в результате прогона (ТЗ 4.1 в отчёте)
# --------------------------------------------------------------------------- #


def test_breakdown_separates_total_losses_from_acquisitions():
    """Отчёт обязан показывать, сколько результата держится на допущении −100%."""
    from factorbot.backtest.engine import BacktestResult

    result = BacktestResult(
        equity_net=pd.Series([1.0, 1.1]),
        equity_gross=pd.Series([1.0, 1.1]),
        delistings=[
            (pd.Timestamp("2005-01-03"), 1, -1.0, 0.033),
            (pd.Timestamp("2006-01-03"), 2, -1.0, 0.033),
            (pd.Timestamp("2007-01-03"), 3, 0.01, 0.033),
            (pd.Timestamp("2008-01-03"), 4, 0.35, 0.033),
        ],
    )
    out = result.delisting_breakdown()
    assert int(out.loc["полная потеря (−100%)", "событий"]) == 2
    assert out.loc["полная потеря (−100%)", "вклад, пп"] == pytest.approx(-6.6, abs=0.01)
    assert int(out.loc["около нуля", "событий"]) == 1
    assert int(out.loc["прибыль", "событий"]) == 1


def test_breakdown_is_empty_without_delistings():
    from factorbot.backtest.engine import BacktestResult

    result = BacktestResult(
        equity_net=pd.Series([1.0]), equity_gross=pd.Series([1.0])
    )
    assert result.delisting_breakdown().empty


def test_engine_records_every_delisting_with_its_weight():
    """Вклад считается от капитала на момент события, а не от единицы."""
    from factorbot.backtest.engine import _settle_delistings

    positions = {1: 0.30, 2: 0.20}
    last_alive = pd.Series({1: pd.Timestamp("2005-01-03"), 2: pd.Timestamp("2010-01-01")})
    survivors, cash, events = _settle_delistings(
        positions, 0.5, last_alive, pd.Timestamp("2005-01-04"),
        pd.Series({1: -0.40}),
    )
    assert list(survivors) == [2]
    assert events == [(1, 0.30, -0.40)]
    assert cash == pytest.approx(0.5 + 0.30 * 0.60)
