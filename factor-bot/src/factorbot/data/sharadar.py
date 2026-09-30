"""Загрузка и нормализация Sharadar (ТЗ 4.2–4.5).

Два независимых слоя:

*   `SharadarProvider` — сеть. Bulk-выгрузка через прямой API Sharadar, кэш
    сырых архивов на диске. Тянуть таблицы поштучно ТЗ 4.2 запрещает.
*   `normalize_*` — чистые функции над кадрами. Ни сети, ни файлов, ни базы,
    поэтому проверяемы на синтетике и не требуют ключа API.

Соединения только по `permaticker` (ТЗ 4.5). Тикеры переиспользуются, поэтому
сопоставление ticker → permaticker всегда идёт с интервалом дат, а не по равенству
строк: свободный символ достаётся другой компании, и join по `ticker` даст тихую
склейку двух разных историй.

Про адрес. ТЗ 4.2 называет эндпоинт Nasdaq Data Link, но Sharadar с тех пор
продаёт данные напрямую, и ключ с sharadar.com в системе Nasdaq не работает —
их документация предупреждает об этом отдельной строкой. Поэтому здесь прямой
API. Имена таблиц у него свои (`fundamentals`, `stocks`), а прежние коды `SF1` и
`SEP` остались псевдонимами; в коде используются современные имена, чтобы
сообщения об ошибках совпадали с документацией поставщика.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from factorbot.data.provider import DataProvider

log = logging.getLogger(__name__)

API_ROOT = "https://api.sharadar.com/v1.0"

#: Таблицы ТЗ 4.2 в именах прямого API. В скобках прежние коды: fundamentals =
#: SF1, stocks = SEP. `daily` — только контрольный источник (ТЗ 4.4).
TABLES: tuple[str, ...] = ("tickers", "stocks", "fundamentals", "actions", "daily")

#: Без этих трёх не собирается ничего: справочник задаёт вселенную и карту
#: permaticker, stocks даёт momentum, fundamentals — value.
REQUIRED_TABLES: tuple[str, ...] = ("tickers", "stocks", "fundamentals")

#: `actions` нужна для корректных delisting returns (ТЗ 4.1), `daily` — только
#: для сверки (ТЗ 4.4). Их отсутствие на урезанном тарифе не должно ронять
#: загрузку, но обязано быть видно: без actions результат завышен систематически.
OPTIONAL_TABLES: tuple[str, ...] = ("actions", "daily")

#: Сколько строк читать за раз у больших таблиц. Два миллиона строк цен — это
#: порядка гигабайта в pandas, что оставляет запас даже в тесном контейнере.
DEFAULT_CHUNK_ROWS = 2_000_000

#: Таблицы, которые заведомо не помещаются в память целиком: одна строка на
#: бумагу на торговый день за тридцать лет.
STREAMED_TABLES: frozenset[str] = frozenset({"stocks", "daily"})

#: Глубина выгрузки. Протокол ТЗ 9.1 требует истории с 1998 года, поэтому здесь
#: только "full"; значения "5" и "10" оставлены на случай урезанной подписки —
#: с ними раздел 9 придётся переписывать, и это должно быть видно в конфиге.
DEFAULT_YEARS = "full"

#: Измерения SF1, которые вообще попадают в базу (ТЗ 4.3). MR* отбрасываются
#: на загрузке, а не на выборке: то, чего нет в файле, нельзя прочитать по ошибке.
FLOW_FIELDS = {"revenue": "revenue_ttm", "netinc": "netinc_ttm",
               "ncfo": "opcf_ttm", "capex": "capex_ttm"}
STOCK_FIELDS = {"equity": "equity", "debt": "debt", "cashneq": "cash",
                "assets": "assets", "sharesbas": "sharesbas"}


class SharadarError(RuntimeError):
    """Ошибка загрузки, при которой продолжать бессмысленно."""


class SubscriptionError(SharadarError):
    """Таблица не входит в тариф ключа. Отличается от сбоя сети: повтор не поможет."""


class RateLimitError(SharadarError):
    """Превышена частота запросов. В отличие от тарифа, проходит само — но не сразу.

    Долбить поставщика после такого ответа бессмысленно и вредно: счётчик
    продлевается, и разбираться потом приходится часами.
    """


class AuthError(SharadarError):
    """Ключ не принят. Повтор не поможет, нужен другой ключ."""


@dataclass(frozen=True)
class TableAccess:
    """Результат проверки доступа к одной таблице."""

    table: str
    ok: bool
    http_status: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        mark = "доступна" if self.ok else "НЕТ ДОСТУПА"
        tail = f" ({self.detail})" if self.detail else ""
        return f"{self.table:<8} {mark}{tail}"


# --------------------------------------------------------------------------- #
# Сеть
# --------------------------------------------------------------------------- #


@dataclass
class SharadarProvider(DataProvider):
    """Bulk-выгрузка таблиц Sharadar с кэшем на диске.

    Args:
        api_key: ключ с sharadar.com. По умолчанию из SHARADAR_API_KEY; ради
            совместимости принимается и прежнее имя NASDAQ_DATA_LINK_API_KEY.
        cache_dir: куда складывать сырые архивы.
        years: глубина выгрузки, "full" по умолчанию (ТЗ 9.1).
    """

    api_key: str | None = None
    cache_dir: Path = Path("data/raw")
    years: str = DEFAULT_YEARS
    #: Полная история цен — сотни мегабайт, и скачивание идёт минутами.
    timeout_s: int = 1800
    #: Пауза между проверками таблиц: пять запросов подряд поставщику не нравятся.
    probe_interval_s: float = 2.0
    name: str = "sharadar"

    def __post_init__(self) -> None:
        self.api_key = (
            self.api_key
            or os.environ.get("SHARADAR_API_KEY")
            or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
        )
        self.cache_dir = Path(self.cache_dir)

    @property
    def _headers(self) -> dict[str, str]:
        """Ключ уходит заголовком, а не в строке запроса.

        В URL он попал бы в логи прокси, в историю команд и в текст исключений
        requests. Заголовок этого не делает.
        """
        return {"x-api-key": self.api_key or ""}

    def available_tables(self) -> tuple[str, ...]:
        return TABLES

    def fetch_table(self, table: str, *, force: bool = False) -> pd.DataFrame:
        """Отдаёт сырую таблицу целиком, из кэша или из сети (ТЗ 4.2).

        Годится для справочника, отчётности и корпоративных действий. Для цен и
        дневных метрик пользуйтесь `iter_table`: они не помещаются в память.
        """
        return _read_csv_zip(self._ensure_archive(table, force=force))

    def iter_table(
        self, table: str, *, chunksize: int = DEFAULT_CHUNK_ROWS, force: bool = False
    ):
        """Отдаёт сырую таблицу порциями строк.

        Полная история цен — 3.2 ГБ в CSV, и в память одним кадром она
        разворачивается примерно в четырнадцать: pandas хранит строковые колонки
        объектами. Контейнер такого не переживает, и однажды уже не пережил.

        Порции решают это без потери логики: нормализация у нас построчная, а
        дубликаты между порциями снимает первичный ключ таблицы.

        Yields:
            Кадры не более `chunksize` строк каждый.
        """
        archive = self._ensure_archive(table, force=force)
        with zipfile.ZipFile(archive) as z:
            name = _single_csv(z)
            with z.open(name) as fh:
                reader = pd.read_csv(
                    io.TextIOWrapper(fh, encoding="utf-8"),
                    low_memory=False,
                    chunksize=chunksize,
                )
                for i, chunk in enumerate(reader, start=1):
                    log.info("%s: порция %d, строк %d", table, i, len(chunk))
                    yield chunk

    def _ensure_archive(self, table: str, *, force: bool = False) -> Path:
        """Путь к архиву таблицы: из кэша или после скачивания."""
        if table not in TABLES:
            raise ValueError(f"Таблица {table!r} не входит в ТЗ 4.2: {TABLES}")

        cached = self._cached_path(table)
        if cached is not None and not force:
            log.info("%s: беру из кэша %s", table, cached)
            return cached

        if not self.api_key:
            raise SharadarError(
                f"Нет ключа Sharadar и нет кэша для {table}. Задайте "
                f"SHARADAR_API_KEY или положите архив в {self.cache_dir / table}."
            )

        target = self.cache_dir / table / f"{date.today().isoformat()}.zip"
        self._download_bulk(table, target)
        log.info("%s: сохранено в %s (%.1f МБ)", table, target,
                 target.stat().st_size / 1e6)
        return target

    def _cached_path(self, table: str) -> Path | None:
        d = self.cache_dir / table
        if not d.is_dir():
            return None
        archives = sorted(d.glob("*.zip"))
        return archives[-1] if archives else None

    def probe_table(self, table: str) -> TableAccess:
        """Одна строка из таблицы — дёшево и достаточно, чтобы узнать про тариф.

        Проверять это до запуска многочасовой загрузки полезнее, чем узнавать
        на четвёртой таблице.
        """
        if not self.api_key:
            return TableAccess(table, False, None, "не задан SHARADAR_API_KEY")
        try:
            r = requests.get(
                f"{API_ROOT}/data/{table}",
                params={"limit": 1},
                headers=self._headers,
                timeout=60,
            )
        except requests.RequestException as exc:
            return TableAccess(table, False, None, f"сеть: {exc.__class__.__name__}")

        if r.ok:
            return TableAccess(table, True, r.status_code)

        detail = _api_error(r)
        if r.status_code == 429:
            raise RateLimitError(detail)
        if r.status_code == 401:
            raise AuthError(detail)
        return TableAccess(table, False, r.status_code, detail)

    def check_access(self) -> dict[str, TableAccess]:
        """Проверяет тариф ключа по всем таблицам ТЗ 4.2.

        Между запросами стоит пауза, а при отказе по частоте или по ключу
        проверка прекращается: продолжать бессмысленно.

        Raises:
            RateLimitError: поставщик временно отказывает по частоте.
            AuthError: ключ не принят.
        """
        access: dict[str, TableAccess] = {}
        for i, table in enumerate(TABLES):
            if i and self.probe_interval_s:
                time.sleep(self.probe_interval_s)
            access[table] = self.probe_table(table)
        return access

    def _download_bulk(self, table: str, target: Path) -> None:
        """Скачивает снимок таблицы целиком (ТЗ 4.2).

        Запрос с `years` отвечает редиректом на подписанную ссылку в хранилище
        поставщика; `requests` идёт по нему сам. Файл пишется потоком во
        временный и переименовывается только после успеха: оборванная закачка не
        должна остаться в кэше под видом готового архива.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")

        with requests.get(
            f"{API_ROOT}/data/{table}",
            params={"years": self.years},
            headers=self._headers,
            timeout=self.timeout_s,
            stream=True,
        ) as r:
            self._raise_for_status(r, table)
            written = 0
            with partial.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    if written % (64 << 20) < (1 << 20):
                        log.info("%s: скачано %.0f МБ", table, written / 1e6)

        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise SharadarError(f"{table}: поставщик вернул пустой файл.")
        partial.replace(target)

    def _raise_for_status(self, r: requests.Response, table: str) -> None:
        if r.ok:
            return
        detail = _api_error(r)
        if r.status_code == 401:
            raise AuthError(f"{table}: ключ не принят. {detail}")
        if r.status_code == 429:
            raise RateLimitError(f"{table}: {detail}")
        if r.status_code == 403:
            raise SubscriptionError(
                f"{table}: нет доступа по подписке. {detail} "
                "Проверьте тариф на https://sharadar.com/account"
            )
        r.raise_for_status()


def _api_error(response: requests.Response) -> str:
    """Текст ошибки поставщика; при неразборчивом ответе — код и начало тела."""
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"

    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return f"HTTP {response.status_code}: {str(payload)[:200]}"


def _single_csv(archive: zipfile.ZipFile) -> str:
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise SharadarError(f"В архиве ожидался один CSV, найдено: {names}")
    return names[0]


def _read_csv_zip(archive: Path) -> pd.DataFrame:
    """Bulk-выгрузка приходит zip-архивом с одним CSV внутри."""
    with zipfile.ZipFile(archive) as z:
        with z.open(_single_csv(z)) as fh:
            return pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), low_memory=False)


# --------------------------------------------------------------------------- #
# Сопоставление тикеров (ТЗ 4.5)
# --------------------------------------------------------------------------- #


#: Прежние коды таблиц и новые имена указывают на одни и те же данные.
TABLE_ALIASES: dict[str, set[str]] = {
    "stocks": {"stocks", "sep"},
    "fundamentals": {"fundamentals", "sf1"},
    "actions": {"actions"},
    "daily": {"daily"},
    "tickers": {"tickers"},
}


def _table_aliases(table: str) -> set[str]:
    key = table.lower()
    for name, aliases in TABLE_ALIASES.items():
        if key == name or key in aliases:
            return aliases
    return {key}


def build_ticker_map(tickers_raw: pd.DataFrame, source_table: str) -> pd.DataFrame:
    """Интервальная карта ticker → permaticker для одной исходной таблицы.

    TICKERS содержит по строке на пару (таблица, бумага) с окном, в котором символ
    принадлежал именно этой компании. Для цен окно задаётся первой и последней
    датой котировки, для отчётности — первым и последним кварталом.

    Returns:
        Кадр с колонками ticker, permaticker, valid_from, valid_to.
    """
    wanted = _table_aliases(source_table)
    df = tickers_raw.loc[tickers_raw["table"].str.lower().isin(wanted)].copy()
    if df.empty:
        raise SharadarError(f"В TICKERS нет строк для таблицы {source_table!r}.")

    # Для отчётности окно владения символом задаётся кварталами, для цен —
    # датами котировок. Прежние коды SF1/SEP принимаются наравне с новыми
    # именами: в документации поставщика они остались первоклассными.
    if source_table.lower() in ("fundamentals", "sf1"):
        lo, hi = "firstquarter", "lastquarter"
    else:
        lo, hi = "firstpricedate", "lastpricedate"

    out = pd.DataFrame({
        "ticker": df["ticker"].astype("string"),
        "permaticker": pd.to_numeric(df["permaticker"], errors="coerce").astype("Int64"),
        "valid_from": pd.to_datetime(df[lo], errors="coerce"),
        "valid_to": pd.to_datetime(df[hi], errors="coerce"),
    })
    out = out.dropna(subset=["ticker", "permaticker"])

    # Открытые интервалы. Бумага, которая всё ещё торгуется, не имеет lastpricedate;
    # незаполненный нижний край — дефект справочника, такие строки бесполезны.
    out["valid_to"] = out["valid_to"].fillna(pd.Timestamp("2262-04-10"))
    out = out.dropna(subset=["valid_from"])
    return out.reset_index(drop=True)


def attach_permaticker(
    df: pd.DataFrame, tmap: pd.DataFrame, *, date_col: str, slack_days: int = 0
) -> pd.DataFrame:
    """Проставляет permaticker по паре (ticker, дата) внутри интервала владения.

    Строки, для которых символ на эту дату не принадлежал никому, отбрасываются с
    записью в лог: молча терять историю нельзя, но и падать на всей выгрузке из-за
    сотни служебных тикеров бессмысленно.

    Args:
        slack_days: допуск на краях интервала. Нужен для SF1, где `datekey` может
            выходить за границу последнего квартала.

    Raises:
        SharadarError: если пара (ticker, дата) попала сразу в два интервала —
            это дефект справочника, и любая склейка после него будет тихой ошибкой.
    """
    left = df.copy()
    left[date_col] = pd.to_datetime(left[date_col], errors="coerce")
    left = left.dropna(subset=[date_col])
    left["ticker"] = left["ticker"].astype("string")
    left["_row"] = range(len(left))

    slack = pd.Timedelta(days=slack_days)
    joined = left.merge(tmap, on="ticker", how="left", suffixes=("", "_map"))
    hit = joined["permaticker"].notna() & (
        (joined[date_col] >= joined["valid_from"] - slack)
        & (joined[date_col] <= joined["valid_to"] + slack)
    )
    joined = joined.loc[hit]

    dupes = joined["_row"].duplicated()
    if dupes.any():
        bad = joined.loc[joined["_row"].isin(joined.loc[dupes, "_row"]), ["ticker", date_col]]
        raise SharadarError(
            "Символ принадлежит двум permaticker одновременно — справочник "
            f"противоречив: {bad.head(10).to_dict('records')}"
        )

    dropped = len(left) - len(joined)
    if dropped:
        log.warning(
            "Не сопоставлено с permaticker: %d строк из %d (%.2f%%)",
            dropped, len(left), 100 * dropped / max(len(left), 1),
        )

    return joined.drop(columns=["_row", "valid_from", "valid_to"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Нормализация к схемам ТЗ 4.7
# --------------------------------------------------------------------------- #


def normalize_tickers(tickers_raw: pd.DataFrame) -> pd.DataFrame:
    """`tickers` → securities. Одна строка на permaticker."""
    df = tickers_raw.loc[
        tickers_raw["table"].str.lower().isin(_table_aliases("stocks"))
    ].copy()
    out = pd.DataFrame({
        "permaticker": pd.to_numeric(df["permaticker"], errors="coerce").astype("Int64"),
        "ticker": df["ticker"].astype("string"),
        "name": df.get("name"),
        "exchange": df.get("exchange"),
        "sector": df.get("sector"),
        "industry": df.get("industry"),
        "siccode": df.get("siccode").astype("string") if "siccode" in df else None,
        "category": df.get("category"),
        "is_delisted": df["isdelisted"].astype("string").str.upper().eq("Y")
        if "isdelisted" in df else False,
        "first_price_date": pd.to_datetime(df.get("firstpricedate"), errors="coerce"),
        "last_price_date": pd.to_datetime(df.get("lastpricedate"), errors="coerce"),
    })
    out = out.dropna(subset=["permaticker"])
    # Справочник изредка отдаёт две строки на бумагу; берём последнюю по истории цен.
    out = out.sort_values("last_price_date").drop_duplicates("permaticker", keep="last")
    return out.reset_index(drop=True)


def normalize_sep(sep_raw: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    """SEP → prices. Момент истины для momentum — `closeadj` (ТЗ 4.1)."""
    df = attach_permaticker(sep_raw, tmap, date_col="date")
    unadj = df["closeunadj"] if "closeunadj" in df else df["close"]
    out = pd.DataFrame({
        "permaticker": df["permaticker"].astype("int64"),
        "ticker": df["ticker"],
        "date": df["date"].dt.date,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "closeadj": pd.to_numeric(df["closeadj"], errors="coerce"),
        "close_unadj": pd.to_numeric(unadj, errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    })
    out["dollar_volume"] = out["close_unadj"] * out["volume"]
    return out.drop_duplicates(subset=["permaticker", "date"], keep="last").reset_index(drop=True)


def normalize_sf1(sf1_raw: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    """SF1 → кадр под схему фундаментала (ТЗ 4.3, 4.7).

    Разделение измерений жёсткое: ART несёт только потоковые величины, ARQ — только
    балансовые. Так одна строка не может смешать TTM-прибыль с балансом другого
    квартала, а факторный код не обязан помнить, откуда какое поле.

    `available_from` = `date` (прежний `datekey`) — дата подачи формы в SEC.
    `lastupdated` не участвует (ТЗ 4.3): это момент правки записи у поставщика,
    а не публичного раскрытия.

    Знак `capex_ttm` сохраняется как у поставщика — отток отрицателен. Формула FCF
    в ТЗ 6.2 записана для положительного capex; см. README, раздел отклонений.
    """
    df = sf1_raw.copy()
    df["dimension"] = df["dimension"].astype("string").str.upper()
    df = df.loc[df["dimension"].isin(["ART", "ARQ"])]
    if df.empty:
        raise SharadarError("В SF1 не осталось строк с измерениями ART/ARQ (ТЗ 4.3).")

    df = attach_permaticker(df, tmap, date_col="reportperiod", slack_days=0)

    out = pd.DataFrame({
        "permaticker": df["permaticker"].astype("int64"),
        "ticker": df["ticker"],
        "dimension": df["dimension"],
        "reportperiod": df["reportperiod"].dt.date,
        "calendardate": pd.to_datetime(df["calendardate"], errors="coerce").dt.date,
        # В прямом API поле называется `date`, и это прежний `datekey`: словарь
        # полей поставщика определяет его как «SEC filing date for AR dimensions».
        "available_from": pd.to_datetime(df["date"], errors="coerce").dt.date,
    })

    is_flow = out["dimension"] == "ART"
    for src, dst in FLOW_FIELDS.items():
        col = pd.to_numeric(df.get(src), errors="coerce") if src in df else pd.NA
        out[dst] = pd.Series(col, index=out.index).where(is_flow)
    for src, dst in STOCK_FIELDS.items():
        col = pd.to_numeric(df.get(src), errors="coerce") if src in df else pd.NA
        out[dst] = pd.Series(col, index=out.index).where(~is_flow)

    # Строка без даты раскрытия непригодна для PIT: неизвестно, когда её было
    # можно увидеть, а значит нельзя использовать никогда.
    missing_key = out["available_from"].isna()
    if missing_key.any():
        log.warning("Отброшено строк отчётности без даты подачи: %d", int(missing_key.sum()))
        out = out.loc[~missing_key]

    return out.reset_index(drop=True)


def normalize_actions(actions_raw: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    """ACTIONS → corp_actions. Источник правды для delisting returns (ТЗ 4.1)."""
    df = attach_permaticker(actions_raw, tmap, date_col="date")
    out = pd.DataFrame({
        "permaticker": df["permaticker"].astype("int64"),
        "date": df["date"].dt.date,
        "action": df["action"].astype("string"),
        "value": pd.to_numeric(df.get("value"), errors="coerce"),
    })
    return out.drop_duplicates(
        subset=["permaticker", "date", "action"], keep="last"
    ).reset_index(drop=True)


def normalize_daily(daily_raw: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    """DAILY → daily_control. Только сверка, не сигналы (ТЗ 4.4)."""
    df = attach_permaticker(daily_raw, tmap, date_col="date")
    out = pd.DataFrame({
        "permaticker": df["permaticker"].astype("int64"),
        "date": df["date"].dt.date,
        "marketcap": pd.to_numeric(df.get("marketcap"), errors="coerce"),
        "ev": pd.to_numeric(df.get("ev"), errors="coerce"),
        "pe": pd.to_numeric(df.get("pe"), errors="coerce"),
        "pb": pd.to_numeric(df.get("pb"), errors="coerce"),
        "ps": pd.to_numeric(df.get("ps"), errors="coerce"),
    })
    return out.drop_duplicates(subset=["permaticker", "date"], keep="last").reset_index(drop=True)


def snapshot_timestamp() -> datetime:
    """Момент загрузки — попадает в лог и в отчёт о воспроизводимости."""
    return datetime.now()
