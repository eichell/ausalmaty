"""DDL нормализованных таблиц (ТЗ 4.7).

Таблица фундаментала здесь намеренно отсутствует: её DDL живёт в `pit.py`, потому
что CI-проверка ТЗ 4.8 запрещает упоминание её имени вне этого модуля. Функция
`create_all` подтягивает оттуда готовую строку — импорт по имени константы
проверку не нарушает.
"""

from __future__ import annotations

import duckdb

from factorbot.data.pit import FUNDAMENTALS_DDL

PRICES_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    permaticker   BIGINT NOT NULL,
    ticker        VARCHAR,
    date          DATE   NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    -- closeadj: поправка на сплиты И дивиденды (ТЗ 4.1). Momentum считается
    -- только по ней; close оставлен для сверки и для отладки корп. действий.
    closeadj      DOUBLE,
    -- close_unadj: цена, как она была в тот день, без единой поправки. Нужна там,
    -- где цену умножают на «тогдашнее» число акций (market cap, ТЗ 6.2) и где
    -- порог задан в долларах того времени (ТЗ 5). Скорректированный ряд для
    -- обеих задач непригоден: он пересчитан от сегодняшней базы.
    close_unadj   DOUBLE,
    volume        DOUBLE,
    -- dollar_volume по нескорректированной цене: фильтр ликвидности из ТЗ 5
    -- измеряет реально проторгованные доллары того дня, а не их пересчёт.
    dollar_volume DOUBLE,
    PRIMARY KEY (permaticker, date)
);
"""

SECURITIES_DDL = """
CREATE TABLE IF NOT EXISTS securities (
    permaticker      BIGINT NOT NULL PRIMARY KEY,
    ticker           VARCHAR,
    name             VARCHAR,
    exchange         VARCHAR,
    sector           VARCHAR,
    industry         VARCHAR,
    siccode          VARCHAR,
    category         VARCHAR,
    is_delisted      BOOLEAN,
    first_price_date DATE,
    last_price_date  DATE
);
"""

CORP_ACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS corp_actions (
    permaticker BIGINT  NOT NULL,
    date        DATE    NOT NULL,
    action      VARCHAR NOT NULL,
    value       DOUBLE,
    PRIMARY KEY (permaticker, date, action)
);
"""

ALPACA_MAP_DDL = """
CREATE TABLE IF NOT EXISTS alpaca_map (
    permaticker   BIGINT NOT NULL PRIMARY KEY,
    alpaca_symbol VARCHAR,
    tradable      BOOLEAN,
    fractionable  BOOLEAN,
    checked_at    DATE
);
"""

UNIVERSE_DDL = """
CREATE TABLE IF NOT EXISTS universe (
    date        DATE   NOT NULL,
    permaticker BIGINT NOT NULL,
    PRIMARY KEY (date, permaticker)
);
"""

#: Таблица DAILY поставщика. Только контрольный источник (ТЗ 4.4): собственные
#: мультипликаторы считаются из цен и AR-измерений, а эти цифры нужны, чтобы
#: заметить систематическое расхождение и искать ошибку у себя.
DAILY_CONTROL_DDL = """
CREATE TABLE IF NOT EXISTS daily_control (
    permaticker BIGINT NOT NULL,
    date        DATE   NOT NULL,
    marketcap   DOUBLE,
    ev          DOUBLE,
    pe          DOUBLE,
    pb          DOUBLE,
    ps          DOUBLE,
    PRIMARY KEY (permaticker, date)
);
"""

ALL_DDL: tuple[str, ...] = (
    PRICES_DDL,
    FUNDAMENTALS_DDL,
    SECURITIES_DDL,
    CORP_ACTIONS_DDL,
    ALPACA_MAP_DDL,
    UNIVERSE_DDL,
    DAILY_CONTROL_DDL,
)


def create_all(conn: duckdb.DuckDBPyConnection) -> None:
    """Создаёт все таблицы схемы. Идемпотентно."""
    for ddl in ALL_DDL:
        conn.execute(ddl)
