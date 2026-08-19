"""Режимный фильтр по широкому рынку (ТЗ 7.1).

    if SPY.close < SPY.sma_200:
        портфель = 100% SHY (короткие Treasuries)
    else:
        портфель = топ-30 по composite

Фильтр применяется к портфелю целиком, а не к отдельным бумагам, и реализован
отключаемым флагом: ТЗ 7.1 требует обязательно сравнить результат с версией без
него. Защитный оверлей, который никто не сравнивал с его отсутствием, — это
лишний параметр, а не защита.

Три решения, которые пришлось принять явно.

**Дата сигнала.** Скользящая средняя считается на последний торговый день месяца,
сделка проходит по открытию следующего (ТЗ 7). Заглянуть в цену дня исполнения
здесь особенно соблазнительно и особенно вредно: фильтр срабатывает на резких
движениях, и один день разницы систематически улучшает результат на всех разворотах.

**Чем считать среднюю.** ТЗ пишет `SPY.close`. Здесь берётся скорректированный
ряд: на нескорректированном сплит создаёт мгновенный разрыв, средняя ломается, и
фильтр выдаёт ложное пересечение на пустом месте. Дивидендный дрейф на окне в 200
дней — величина второго порядка по сравнению с этим риском.

**Когда SHY ещё не существует.** ETF начал торговаться в июле 2002, а in-sample
по ТЗ 9.1 начинается в 1999 году. Первые три года уходить в защитный актив
физически некуда, и капитал остаётся в деньгах под нулевую ставку. Это занижает
результат риск-офф периодов 2000–2002, то есть работает против фильтра, а не в
его пользу — из двух ошибок это безопасная.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from factorbot.data.panel import PricePanel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeRules:
    """Параметры ТЗ 7.1."""

    enabled: bool = True
    benchmark: str = "SPY"
    sma_window: int = 200
    risk_off_asset: str = "SHY"

    @classmethod
    def from_config(cls, cfg) -> RegimeRules:
        return cls(
            enabled=bool(cfg.enabled),
            benchmark=str(cfg.benchmark),
            sma_window=int(cfg.sma_window),
            risk_off_asset=str(cfg.risk_off_asset),
        )


class RegimeFilter:
    """Решает, включён ли риск, на каждую дату ребалансировки."""

    def __init__(
        self,
        benchmark_prices: pd.Series,
        *,
        sma_window: int = 200,
        risk_off_permaticker: int | None = None,
        risk_off_prices: pd.Series | None = None,
    ) -> None:
        self.prices = benchmark_prices.dropna().sort_index()
        self.sma_window = sma_window
        self.risk_off_permaticker = risk_off_permaticker
        self.risk_off_prices = risk_off_prices
        self._warned_no_asset = False

    def is_risk_off(self, as_of: pd.Timestamp) -> bool:
        """Ниже ли рынок своей скользящей средней на дату сигнала.

        Пока истории на полное окно не набралось, режим считается рискованным:
        отсутствие сигнала — не повод сидеть в деньгах, а фильтр здесь защитный
        оверлей, а не разрешение на вход.
        """
        window = self.prices.loc[self.prices.index <= as_of]
        if len(window) < self.sma_window:
            return False
        return bool(window.iloc[-1] < window.iloc[-self.sma_window:].mean())

    def risk_off_weights(self, as_of: pd.Timestamp) -> pd.Series:
        """Портфель в защитном режиме: 100% в короткие Treasuries, если они есть.

        Если защитного актива на эту дату ещё не существует, возвращается пустой
        портфель — капитал остаётся в деньгах. Молча подставлять что-то другое
        нельзя: это меняло бы стратегию незаметно для отчёта.
        """
        if self.risk_off_permaticker is None or not self._asset_trades_on(as_of):
            if not self._warned_no_asset:
                log.warning(
                    "Защитный актив недоступен на %s: риск-офф проводится в деньгах "
                    "под нулевую ставку (ТЗ 7.1). Результат таких периодов занижен.",
                    as_of.date(),
                )
                self._warned_no_asset = True
            return pd.Series(dtype="float64", name="weight")

        return pd.Series(
            {self.risk_off_permaticker: 1.0}, name="weight", dtype="float64"
        )

    def _asset_trades_on(self, as_of: pd.Timestamp) -> bool:
        if self.risk_off_prices is None:
            return False
        available = self.risk_off_prices.dropna()
        return bool(len(available) and available.index[0] <= as_of)


def build_regime_filter(
    panel: PricePanel, securities: pd.DataFrame, rules: RegimeRules
) -> RegimeFilter | None:
    """Собирает фильтр из панели цен. Возвращает None, если фильтр отключён.

    Raises:
        ValueError: если бенчмарка нет в базе. Молча выключить фильтр нельзя —
            прогон назывался бы «с фильтром» и шёл бы без него.
    """
    if not rules.enabled:
        return None

    benchmark_id = _resolve(securities, rules.benchmark)
    if benchmark_id is None or benchmark_id not in panel.tickers:
        raise ValueError(
            f"Бенчмарк {rules.benchmark!r} для режимного фильтра не найден в базе цен. "
            "Он не входит во вселенную (ТЗ 5), но загружаться обязан."
        )

    risk_off_id = _resolve(securities, rules.risk_off_asset)
    risk_off_prices = None
    if risk_off_id is not None and risk_off_id in panel.tickers:
        risk_off_prices = panel.closeadj[risk_off_id]
    else:
        log.warning(
            "Защитный актив %r в базе цен отсутствует: риск-офф будет в деньгах.",
            rules.risk_off_asset,
        )
        risk_off_id = None

    return RegimeFilter(
        panel.closeadj[benchmark_id],
        sma_window=rules.sma_window,
        risk_off_permaticker=risk_off_id,
        risk_off_prices=risk_off_prices,
    )


def _resolve(securities: pd.DataFrame, ticker: str) -> int | None:
    match = securities.loc[
        securities["ticker"].astype("string").str.upper() == ticker.upper()
    ]
    if match.empty:
        return None
    return int(match["permaticker"].iloc[0])
