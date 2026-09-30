"""Сверка с опубликованными результатами по фактору (ТЗ 13.2).

ТЗ требует не просто посчитать momentum, а сверить результат с известными
результатами по фактору. Без этой сверки любая цифра прогона — проверка проводки,
а не результат: тесты на синтетике ловят ошибки знака, окна и таймингов, но не
отвечают на вопрос, воспроизводит ли реализация то, что публикуют другие на тех
же данных.

Эталон — библиотека Кеннета Френча, портфели `10_Portfolios_Prior_12_2`: десять
портфелей, отсортированных по доходности с t−12 по t−2, месячная ребалансировка,
в двух вариантах взвешивания. Определение сигнала совпадает с ТЗ 6.1
(`lookback_days=252`, `skip_days=21`) — это те же двенадцать месяцев с пропуском
последнего.

Что сравнивать честно, а что нет.

**Ни один вариант эталона не подходит целиком, поэтому сравниваются оба.** Это
не осторожность, а следствие двух разных несовпадений, которые тянут в разные
стороны:

*   *По взвешиванию внутри портфеля* ближе равновзвешенный дециль: наш портфель
    равновзвешенный (ТЗ 7).
*   *По вселенной* ближе взвешенный по капитализации. У Френча это вся CRSP,
    включая микрокапы, и в равновзвешенном дециле их тысячи — доходность такого
    портфеля определяют бумаги, которых в нашей вселенной нет вовсе: ТЗ 5 требует
    цену выше $5 и оборот выше $5 млн, и остаётся около 1650 ликвидных имён.
    Экспозиция value-weighted дециля к крупным и средним компаниям гораздо ближе
    к нашей.

Выбрать один и назвать его «эталоном» значило бы спрятать эту неопределённость.
Поэтому печатаются оба, и вывод делается по тому, попадает ли наш результат
между ними.

**Совпадения уровней ждать нельзя ни с одним.** Френч берёт верхние 10% вселенной,
у нас тридцать бумаг из примерно 1650 — верхние 1.8%. Концентрация в пять раз
выше, значит выше и волатильность, и разброс по годам. Совпадать обязаны знаки,
порядок величин и форма кривой, а не проценты.

**Издержки у эталона нулевые.** Френч публикует доходности портфелей без
транзакционных издержек, поэтому сравнение идёт с нашей доходностью до издержек
(ТЗ 8 добавляет 10–25 bps на сторону, и при обороте 400% в год это порядка двух
процентных пунктов).

Главная проверка — корреляция месячных доходностей. Она не зависит от уровня и
почти не зависит от концентрации: если реализация считает тот же сигнал на тех же
данных, месяцы обязаны ходить вместе. Расхождение по знаку в отдельные месяцы
нормально, расхождение по корреляции — нет.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "10_Portfolios_Prior_12_2_CSV.zip"
)

#: Куда складывать эталон. Скачивается один раз: файл обновляется раз в месяц, а
#: сверка повторяется на каждой правке фактора.
DEFAULT_CACHE = Path("data/raw/reference/10_Portfolios_Prior_12_2_CSV.zip")

#: Пропуски в файле Френча помечены так.
MISSING_MARKERS = (-99.99, -999.0)

#: Названия блоков внутри файла: сначала value-weighted, потом equal-weighted.
VALUE_WEIGHTED_HEADER = "Value Weight Returns -- Monthly"
EQUAL_WEIGHTED_HEADER = "Average Equal Weighted Returns -- Monthly"

#: Колонка верхнего дециля.
TOP_DECILE = "Hi PRIOR"

#: Ниже этой корреляции месячных доходностей сверку нельзя считать пройденной:
#: тот же сигнал на тех же данных так расходиться не может.
MIN_MONTHLY_CORRELATION = 0.80


class ReferenceUnavailable(RuntimeError):
    """Эталон не скачан и не лежит в кэше."""


@dataclass(frozen=True)
class Comparison:
    """Результат сверки с одним вариантом эталона."""

    weighting: str
    monthly_correlation: float
    months: int
    ours_cagr: float
    reference_cagr: float
    ours_volatility: float
    reference_volatility: float
    yearly: pd.DataFrame
    sign_agreement: float

    @property
    def passed(self) -> bool:
        return self.monthly_correlation >= MIN_MONTHLY_CORRELATION

    def report(self) -> str:
        verdict = "пройдена" if self.passed else "НЕ ПРОЙДЕНА"
        lines = [
            f"--- эталон: верхний дециль, {self.weighting} — сверка {verdict} ---",
            f"Месяцев в сравнении:         {self.months}",
            f"Корреляция месячных:         {self.monthly_correlation:.3f} "
            f"(порог {MIN_MONTHLY_CORRELATION})",
            f"Совпадение знака по месяцам: {self.sign_agreement:.0%}",
            "",
            f"{'':24}{'наш (до издержек)':>20}{'эталон':>12}",
            f"{'CAGR':<24}{self.ours_cagr:>19.2%}{self.reference_cagr:>12.2%}",
            f"{'Волатильность':<24}{self.ours_volatility:>19.2%}"
            f"{self.reference_volatility:>12.2%}",
            "",
            "По годам (%); diff не считается для неполных лет:",
            self.yearly.to_string(float_format=lambda v: f"{v:7.2f}",
                                  na_rep="   —"),
        ]
        return "\n".join(lines)


def fetch_reference(
    cache: str | Path = DEFAULT_CACHE, *, force: bool = False, timeout_s: int = 120
) -> Path:
    """Скачивает архив Френча в кэш и возвращает путь."""
    cache = Path(cache)
    if cache.exists() and not force:
        log.info("Эталон из кэша: %s", cache)
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(FRENCH_URL, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ReferenceUnavailable(
            f"Не удалось скачать эталон: {exc}. Если это отказ прокси, домен "
            "mba.tuck.dartmouth.edu не разрешён сетевой политикой окружения."
        ) from exc

    cache.write_bytes(response.content)
    log.info("Эталон сохранён: %s (%.0f КБ)", cache, len(response.content) / 1024)
    return cache


def load_decile_returns(
    archive: str | Path, *, equal_weighted: bool = True
) -> pd.Series:
    """Месячные доходности верхнего дециля из файла Френча, в долях.

    Файл содержит несколько блоков подряд, разделённых строками-заголовками. Брать
    первую таблицу нельзя: она value-weighted, а наш портфель равновзвешенный.
    """
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ReferenceUnavailable(f"В архиве ожидался один CSV, найдено: {names}")
        text = z.read(names[0]).decode("utf-8", "replace")

    wanted = EQUAL_WEIGHTED_HEADER if equal_weighted else VALUE_WEIGHTED_HEADER
    block = _extract_block(text, wanted)
    frame = pd.read_csv(io.StringIO(block), index_col=0)
    frame.columns = [c.strip() for c in frame.columns]
    if TOP_DECILE not in frame.columns:
        raise ReferenceUnavailable(
            f"В эталоне нет колонки {TOP_DECILE!r}: {list(frame.columns)}"
        )

    series = pd.to_numeric(frame[TOP_DECILE], errors="coerce")
    series = series.loc[[str(i).strip().isdigit() and len(str(i).strip()) == 6
                         for i in series.index]]
    series.index = pd.PeriodIndex(
        [str(i).strip() for i in series.index], freq="M"
    )
    for marker in MISSING_MARKERS:
        series = series.loc[series != marker]
    return (series.dropna() / 100.0).rename("reference")


def _extract_block(text: str, header: str) -> str:
    """Строки одной таблицы файла: от её заголовка до первой непохожей строки."""
    lines = [line.rstrip("\r") for line in text.split("\n")]
    try:
        start = next(i for i, line in enumerate(lines) if header in line)
    except StopIteration as exc:
        raise ReferenceUnavailable(f"В эталоне нет блока {header!r}") from exc

    # Следующая строка — шапка с названиями портфелей, дальше идут годы-месяцы.
    out = [lines[start + 1]]
    for line in lines[start + 2:]:
        first = line.split(",")[0].strip()
        if not first.isdigit():
            break
        out.append(line)
    return "\n".join(out)


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Месячные доходности из дневной кривой эквити.

    По последнему значению месяца, а не по среднему: эквити — это уровень, и
    доходность месяца считается от закрытия к закрытию.
    """
    monthly = equity.resample("ME").last().dropna()
    out = monthly.pct_change().dropna()
    out.index = out.index.to_period("M")
    return out.rename("ours")


def compare(
    equity: pd.Series, reference: pd.Series, *, weighting: str = "?"
) -> Comparison:
    """Сравнивает нашу кривую с эталонным децилем.

    Сравнение идёт напрямую по доходностям, без вычета рыночной беты. Вычитать её
    было бы отдельным решением: у long-only momentum бета около единицы, и после
    вычета сравнивались бы уже не стратегии, а остатки.
    """
    ours = monthly_returns(equity)
    common = ours.index.intersection(reference.index)
    if len(common) < 24:
        raise ValueError(
            f"Пересечение всего {len(common)} месяцев: сравнивать нечего. "
            "Проверьте период прогона и глубину эталона."
        )
    a, b = ours.loc[common], reference.loc[common]

    yearly = pd.DataFrame({
        "ours": _annual(a) * 100,
        "reference": _annual(b) * 100,
    })
    yearly["diff"] = yearly["ours"] - yearly["reference"]
    # Неполный год обязан быть помечен. Первый месяц прогона в месячный ряд не
    # попадает (доходность считается от предыдущего закрытия), и год начала
    # выборки короче календарного. Без пометки такая строка читается как
    # отставание стратегии от эталона, хотя это разная длина периода — и именно
    # так я сам однажды прочитал 1999 год.
    months = pd.Series(1, index=a.index).groupby(a.index.year).sum()
    yearly["мес."] = months.reindex(yearly.index)
    yearly.loc[yearly["мес."] < 12, "diff"] = float("nan")

    return Comparison(
        weighting=weighting,
        monthly_correlation=float(a.corr(b)),
        months=len(common),
        ours_cagr=_cagr(a),
        reference_cagr=_cagr(b),
        ours_volatility=float(a.std(ddof=1) * np.sqrt(12)),
        reference_volatility=float(b.std(ddof=1) * np.sqrt(12)),
        yearly=yearly,
        sign_agreement=float((np.sign(a) == np.sign(b)).mean()),
    )


def _annual(monthly: pd.Series) -> pd.Series:
    """Годовая доходность из месячных, сложением по правилу сложного процента."""
    by_year = (1.0 + monthly).groupby(monthly.index.year).prod() - 1.0
    by_year.index.name = "year"
    return by_year


def _cagr(monthly: pd.Series) -> float:
    years = len(monthly) / 12.0
    if years <= 0:
        return float("nan")
    return float((1.0 + monthly).prod() ** (1 / years) - 1)


WEIGHTING_LABELS = {True: "равновзвешенный", False: "по капитализации"}


def compare_both(
    equity: pd.Series, archive: str | Path | None = None, *, force: bool = False
) -> dict[str, Comparison]:
    """Сверка с обоими вариантами эталона. Ключ — подпись взвешивания."""
    archive = fetch_reference(archive or DEFAULT_CACHE, force=force)
    out: dict[str, Comparison] = {}
    for equal_weighted, label in WEIGHTING_LABELS.items():
        reference = load_decile_returns(archive, equal_weighted=equal_weighted)
        out[label] = compare(equity, reference, weighting=label)
    return out


def verdict(comparisons: dict[str, Comparison]) -> str:
    """Общий вывод по обоим эталонам (ТЗ 13.2).

    Проверка считается пройденной, если корреляция месячных доходностей выше
    порога хотя бы с одним из вариантов: варианты различаются вселенной, и
    совпадение с тем, чья вселенная ближе к нашей, — это и есть сверка. Уровень
    доходности отдельно не проверяется порогом, но выводится: наш результат обязан
    лежать между двумя эталонами, а не вне их обоих.
    """
    best = max(comparisons.values(), key=lambda c: c.monthly_correlation)
    lines = ["=== Сверка с библиотекой Френча (ТЗ 13.2) ==="]
    for c in comparisons.values():
        lines.append("")
        lines.append(c.report())

    lines.append("")
    if best.passed:
        lines.append(
            f"ВЫВОД: сверка пройдена. Лучшая корреляция месячных {best.monthly_correlation:.3f} "
            f"с эталоном «{best.weighting}» — тот же сигнал на тех же данных."
        )
    else:
        lines.append(
            f"ВЫВОД: СВЕРКА НЕ ПРОЙДЕНА. Лучшая корреляция месячных всего "
            f"{best.monthly_correlation:.3f} при пороге {MIN_MONTHLY_CORRELATION}. "
            "Реализация считает не тот сигнал, не на тех данных или не в те даты. "
            "Цифры прогона принимать нельзя (ТЗ 13.2)."
        )

    levels = sorted(c.reference_cagr for c in comparisons.values())
    ours = next(iter(comparisons.values())).ours_cagr
    if levels[0] <= ours <= levels[-1]:
        lines.append(
            f"Уровень: CAGR {ours:.2%} лежит между эталонами "
            f"({levels[0]:.2%} — {levels[-1]:.2%}), как и должен."
        )
    else:
        side = "ниже" if ours < levels[0] else "выше"
        lines.append(
            f"Уровень: CAGR {ours:.2%} {side} обоих эталонов "
            f"({levels[0]:.2%} — {levels[-1]:.2%}). Само по себе это не дефект — "
            "вселенная и концентрация у нас другие, — но объяснить разницу надо."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Сверка momentum с библиотекой Френча (ТЗ 13.2)"
    )
    parser.add_argument(
        "--equity", default="data/processed/report/equity_momentum_in_sample.csv",
        help="CSV кривой эквити, сохранённый прогоном",
    )
    parser.add_argument(
        "--column", default="equity_gross",
        help="какую кривую сравнивать; у эталона издержек нет, поэтому до издержек",
    )
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--force", action="store_true", help="перескачать эталон")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    path = Path(args.equity)
    if not path.exists():
        log.error(
            "Нет файла кривой %s. Сначала прогон: "
            "python -m factorbot.run --period in_sample --strategy momentum", path,
        )
        return 2

    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if args.column not in frame.columns:
        log.error("В %s нет колонки %s: %s", path, args.column, list(frame.columns))
        return 2

    comparisons = compare_both(frame[args.column].dropna(), args.cache, force=args.force)
    print(verdict(comparisons))
    best = max(c.monthly_correlation for c in comparisons.values())
    return 0 if best >= MIN_MONTHLY_CORRELATION else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
