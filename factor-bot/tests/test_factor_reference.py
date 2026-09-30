"""Сверка с библиотекой Френча (ТЗ 13.2). Сети нет — архив собирается в тесте.

Разбор файла проверяется отдельно и подробно, потому что ошибиться в нём легко и
незаметно: файл содержит два блока подряд, и взять первый вместо нужного значит
сверяться с другой стратегией, получить расхождение и пойти искать дефект там,
где его нет.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from factorbot.report import factor_reference as FR

HEAD = ",Lo PRIOR,PRIOR 2,PRIOR 3,PRIOR 4,PRIOR 5,PRIOR 6,PRIOR 7,PRIOR 8,PRIOR 9,Hi PRIOR"


def csv_block(header: str, rows: list[tuple[str, float]]) -> str:
    body = [f"{ym}," + ",".join(["1.00"] * 9) + f",{hi:.2f}" for ym, hi in rows]
    return "\n".join([f"  {header}", HEAD, *body])


def archive(tmp_path, vw_rows, ew_rows, *, tail: str = "") -> str:
    text = "\n".join([
        "This file was created using the 202608 CRSP database.",
        "Missing data are indicated by -99.99 or -999.",
        "",
        csv_block(FR.VALUE_WEIGHTED_HEADER, vw_rows),
        "",
        csv_block(FR.EQUAL_WEIGHTED_HEADER, ew_rows),
        tail,
    ])
    path = tmp_path / "ref.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("10_Portfolios_Prior_12_2.csv", text)
    return str(path)


MONTHS = [f"20{y:02d}{m:02d}" for y in range(10, 13) for m in range(1, 13)]


def test_equal_weighted_block_is_not_the_first_one(tmp_path):
    """Взять первый блок — значит сверяться с портфелем по капитализации."""
    path = archive(
        tmp_path,
        vw_rows=[(m, 1.0) for m in MONTHS],
        ew_rows=[(m, 5.0) for m in MONTHS],
    )
    ew = FR.load_decile_returns(path, equal_weighted=True)
    vw = FR.load_decile_returns(path, equal_weighted=False)
    assert ew.iloc[0] == pytest.approx(0.05)
    assert vw.iloc[0] == pytest.approx(0.01)


def test_percent_is_converted_to_fractions(tmp_path):
    path = archive(tmp_path, [(m, 2.5) for m in MONTHS], [(m, 2.5) for m in MONTHS])
    assert FR.load_decile_returns(path).iloc[0] == pytest.approx(0.025)


@pytest.mark.parametrize("marker", FR.MISSING_MARKERS)
def test_missing_values_are_dropped_not_read_as_returns(tmp_path, marker):
    """−99.99 как доходность −9999% обнулила бы кривую целиком."""
    rows = [(m, 1.0) for m in MONTHS]
    rows[5] = (MONTHS[5], marker)
    path = archive(tmp_path, rows, rows)
    out = FR.load_decile_returns(path)
    assert len(out) == len(MONTHS) - 1
    assert out.min() == pytest.approx(0.01)


def test_annual_block_after_the_monthly_one_is_not_swallowed(tmp_path):
    """В настоящем файле за месячными блоками идут годовые: там четыре цифры года."""
    path = archive(
        tmp_path,
        [(m, 1.0) for m in MONTHS],
        [(m, 1.0) for m in MONTHS],
        tail="\n".join(["  Annual Returns", HEAD, "2010," + ",".join(["9.0"] * 10)]),
    )
    out = FR.load_decile_returns(path)
    assert len(out) == len(MONTHS)
    assert out.max() == pytest.approx(0.01)


def test_missing_block_is_an_error(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.csv", "nothing here\n")
    with pytest.raises(FR.ReferenceUnavailable):
        FR.load_decile_returns(path)


# --------------------------------------------------------------------------- #
# Сравнение
# --------------------------------------------------------------------------- #


def daily_equity(monthly: pd.Series) -> pd.Series:
    """Дневная кривая, дающая ровно эти месячные доходности."""
    days = pd.date_range("2009-12-31", "2012-12-31", freq="B")
    equity = pd.Series(1.0, index=days)
    level = 1.0
    for period, ret in monthly.items():
        level *= 1.0 + ret
        month_end = period.to_timestamp(how="end").normalize()
        equity.loc[equity.index > month_end - pd.offsets.MonthBegin(1)] = level
    return equity


def test_identical_series_give_correlation_one():
    rng = np.random.default_rng(0)
    idx = pd.PeriodIndex(MONTHS, freq="M")
    reference = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    out = FR.compare(daily_equity(reference), reference)
    assert out.monthly_correlation == pytest.approx(1.0, abs=1e-6)
    assert out.passed


def test_unrelated_series_fail_the_check():
    rng = np.random.default_rng(1)
    idx = pd.PeriodIndex(MONTHS, freq="M")
    reference = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    noise = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    out = FR.compare(daily_equity(noise), reference)
    assert not out.passed


def test_too_short_overlap_is_an_error():
    idx = pd.PeriodIndex(["201001", "201002"], freq="M")
    reference = pd.Series([0.01, 0.02], index=idx)
    with pytest.raises(ValueError, match="месяцев"):
        FR.compare(daily_equity(reference), reference)


def test_verdict_names_the_better_reference():
    idx = pd.PeriodIndex(MONTHS, freq="M")
    rng = np.random.default_rng(2)
    good = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    bad = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    equity = daily_equity(good)
    comparisons = {
        "равновзвешенный": FR.compare(equity, bad, weighting="равновзвешенный"),
        "по капитализации": FR.compare(equity, good, weighting="по капитализации"),
    }
    text = FR.verdict(comparisons)
    assert "пройдена" in text
    assert "по капитализации" in text.split("ВЫВОД")[1]


def test_verdict_says_plainly_when_the_check_fails():
    idx = pd.PeriodIndex(MONTHS, freq="M")
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    b = pd.Series(rng.normal(0.01, 0.05, len(idx)), index=idx)
    comparisons = {"равновзвешенный": FR.compare(daily_equity(a), b)}
    text = FR.verdict(comparisons)
    assert "НЕ ПРОЙДЕНА" in text
    assert "принимать нельзя" in text


def test_no_key_no_cache_message_mentions_the_domain(tmp_path, monkeypatch):
    import requests

    def boom(*a, **kw):
        raise requests.RequestException("proxy said no")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(FR.ReferenceUnavailable, match="mba.tuck.dartmouth.edu"):
        FR.fetch_reference(tmp_path / "absent.zip")


def test_monthly_returns_are_taken_from_month_ends():
    days = pd.date_range("2010-01-01", "2010-03-31", freq="B")
    equity = pd.Series(np.linspace(1.0, 1.3, len(days)), index=days)
    out = FR.monthly_returns(equity)
    assert list(out.index.astype(str)) == ["2010-02", "2010-03"]
