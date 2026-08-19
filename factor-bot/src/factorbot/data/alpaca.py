"""Alpaca: торгуемость и сверка цен (ТЗ 4.6).

В исследовательской части Alpaca не участвует — глубина истории с 2016 года
несовместима с протоколом ТЗ 9. Здесь только две разрешённые задачи:

1.  Проверка торгуемости перед формированием заявок (`/v2/assets`).
2.  Сверка `SEP.closeadj` с барами Alpaca на пересекающемся отрезке.

Отправки ордеров в этом модуле нет и не будет до прохождения всех проверок
раздела 9 (ТЗ 2). Это не техническое ограничение, а условие задачи.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import requests

from factorbot.data.provider import ExecutionVenue

log = logging.getLogger(__name__)

TRADING_API = "https://api.alpaca.markets"
DATA_API = "https://data.alpaca.markets"

#: Порог расхождения цен из ТЗ 4.6.3. Больше — искать ошибку в обработке
#: корпоративных действий, у себя или у поставщика.
PRICE_TOLERANCE = 0.005


class AlpacaError(RuntimeError):
    """Ошибка обращения к Alpaca."""


@dataclass
class AlpacaVenue(ExecutionVenue):
    """Клиент Alpaca только на чтение справочника и баров."""

    key_id: str | None = None
    secret_key: str | None = None
    paper: bool = True
    name: str = "alpaca"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.key_id = self.key_id or os.environ.get("ALPACA_API_KEY_ID")
        self.secret_key = self.secret_key or os.environ.get("ALPACA_API_SECRET_KEY")
        if not (self.key_id and self.secret_key):
            log.warning(
                "Ключи Alpaca не заданы (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY). "
                "Проверка торгуемости и сверка цен будут недоступны."
            )
        self._session.headers.update({
            "APCA-API-KEY-ID": self.key_id or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
        })

    @property
    def _trading_root(self) -> str:
        return "https://paper-api.alpaca.markets" if self.paper else TRADING_API

    def tradable_assets(self) -> pd.DataFrame:
        """Справочник инструментов с полями tradable/status/fractionable (ТЗ 4.6.1)."""
        r = self._session.get(
            f"{self._trading_root}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
            timeout=120,
        )
        if r.status_code in (401, 403):
            raise AlpacaError("Alpaca отклонила ключи: проверьте ALPACA_API_KEY_ID/SECRET.")
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            raise AlpacaError("Alpaca вернула пустой список инструментов.")
        keep = ["symbol", "name", "exchange", "status", "tradable", "fractionable",
                "shortable", "easy_to_borrow"]
        return df[[c for c in keep if c in df.columns]]

    def daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Дневные бары с полной корректировкой — для сверки с `SEP.closeadj`."""
        rows: list[dict] = []
        page_token: str | None = None
        # Alpaca принимает символы пачками; 200 на запрос — компромисс между
        # числом обращений и длиной URL.
        for chunk_start in range(0, len(symbols), 200):
            chunk = symbols[chunk_start:chunk_start + 200]
            page_token = None
            while True:
                params = {
                    "symbols": ",".join(chunk),
                    "timeframe": "1Day",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "adjustment": "all",
                    "limit": 10000,
                }
                if page_token:
                    params["page_token"] = page_token
                r = self._session.get(f"{DATA_API}/v2/stocks/bars", params=params, timeout=120)
                r.raise_for_status()
                payload = r.json()
                for symbol, bars in (payload.get("bars") or {}).items():
                    for b in bars:
                        rows.append({"symbol": symbol, "date": b["t"][:10], "close": b["c"]})
                page_token = payload.get("next_page_token")
                if not page_token:
                    break

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df


# --------------------------------------------------------------------------- #
# Чистые функции — проверяемы без сети
# --------------------------------------------------------------------------- #


def build_alpaca_map(
    securities: pd.DataFrame, assets: pd.DataFrame, *, checked_at: date
) -> pd.DataFrame:
    """Сопоставляет вселенную Sharadar с инструментами Alpaca (ТЗ 4.6.1).

    Стыковка идёт по символу — единственному общему полю двух поставщиков. Внутри
    проекта символ по-прежнему не является ключом (ТЗ 4.5): карта хранит
    permaticker и дату сверки, и переспрашивается заново, а не наследуется.

    Бумаги, которых на Alpaca нет (OTC, отдельные классы акций), остаются в карте
    со значением tradable=False — их надо отсеивать осознанно и видеть в логе,
    а не терять на молчаливом inner join.
    """
    left = securities[["permaticker", "ticker"]].copy()
    left["ticker"] = left["ticker"].astype("string").str.upper()

    right = assets.copy()
    right["symbol"] = right["symbol"].astype("string").str.upper()
    right = right.drop_duplicates("symbol", keep="last")

    merged = left.merge(right, how="left", left_on="ticker", right_on="symbol")
    tradable = merged.get("tradable")
    status_ok = merged["status"].eq("active") if "status" in merged else True

    out = pd.DataFrame({
        "permaticker": merged["permaticker"].astype("int64"),
        "alpaca_symbol": merged["symbol"],
        "tradable": (tradable.fillna(False).astype(bool) & status_ok)
        if tradable is not None else False,
        "fractionable": merged["fractionable"].fillna(False).astype(bool)
        if "fractionable" in merged else False,
        "checked_at": checked_at,
    })

    missing = int((~out["tradable"]).sum())
    if missing:
        log.info("Недоступно на Alpaca: %d бумаг из %d (ТЗ 4.6.1)", missing, len(out))
    return out.reset_index(drop=True)


def reconcile_prices(
    sep_prices: pd.DataFrame, alpaca_bars: pd.DataFrame, alpaca_map: pd.DataFrame,
    *, tolerance: float = PRICE_TOLERANCE,
) -> pd.DataFrame:
    """Сравнивает `SEP.closeadj` с барами Alpaca на пересекающемся отрезке (ТЗ 4.6.3).

    Возвращает только расхождения больше порога. Пустой кадр — это и есть
    ожидаемый результат; непустой означает ошибку обработки корпоративных действий
    с чьей-то стороны, и разбирать её нужно до того, как momentum примет её за сигнал.
    """
    bars = alpaca_bars.merge(
        alpaca_map[["permaticker", "alpaca_symbol"]],
        left_on="symbol", right_on="alpaca_symbol", how="inner",
    )
    merged = sep_prices.merge(
        bars[["permaticker", "date", "close"]].rename(columns={"close": "alpaca_close"}),
        on=["permaticker", "date"], how="inner",
    )
    if merged.empty:
        return merged.assign(rel_diff=pd.Series(dtype="float64"))

    # Сравниваются относительные ряды, а не уровни: обе стороны корректируют
    # дивиденды от своей базовой даты, поэтому абсолютные цены расходятся законно.
    merged = merged.sort_values(["permaticker", "date"])
    grp = merged.groupby("permaticker", sort=False)
    ret_sep = grp["closeadj"].pct_change()
    ret_alp = grp["alpaca_close"].pct_change()
    merged["rel_diff"] = (ret_sep - ret_alp).abs()

    bad = merged.loc[merged["rel_diff"] > tolerance]
    if not bad.empty:
        log.warning(
            "Расхождение цен больше %.2f%%: %d дней по %d бумагам (ТЗ 4.6.3)",
            100 * tolerance, len(bad), bad["permaticker"].nunique(),
        )
    return bad.reset_index(drop=True)
