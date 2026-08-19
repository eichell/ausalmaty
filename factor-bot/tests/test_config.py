"""Конфиг стратегии (ТЗ 11: никаких констант в коде)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from factorbot.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "strategy.yaml"


def test_shipped_config_matches_the_spec():
    cfg = load_config(CONFIG_PATH)
    assert cfg.portfolio.top_n == 30                     # ТЗ 7
    assert cfg.portfolio.buffer_rank == 45               # ТЗ 7
    assert cfg.portfolio.max_sector_weight == 0.30       # ТЗ 7
    assert cfg.factors.momentum.lookback_days == 252     # ТЗ 6.1
    assert cfg.factors.momentum.skip_days == 21          # ТЗ 6.1
    assert cfg.regime_filter.sma_window == 200           # ТЗ 7.1
    assert cfg.costs.slippage_bps_liquid == 10.0         # ТЗ 8
    assert cfg.costs.slippage_bps_illiquid == 25.0       # ТЗ 8


def test_factor_weights_are_fifty_fifty_by_construction():
    """ТЗ 9.2.3: веса — априорный выбор, не результат подбора."""
    cfg = load_config(CONFIG_PATH)
    assert cfg.factors.weights.momentum == 0.5
    assert cfg.factors.weights.value == 0.5


def test_unknown_parameter_fails_loudly(tmp_path):
    cfg = load_config(CONFIG_PATH)
    with pytest.raises(AttributeError, match="нет параметра"):
        _ = cfg.portfolio.top_nn


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path):
    bad = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "    momentum: 0.5\n    value: 0.5", "    momentum: 0.7\n    value: 0.5"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="1.0"):
        load_config(path)


def test_missing_section_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="секций"):
        load_config(path)


# --------------------------------------------------------------------------- #
# Ключи API: .env, не репозиторий
# --------------------------------------------------------------------------- #


def test_dotenv_values_reach_the_environment(tmp_path, monkeypatch):
    from factorbot.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("NASDAQ_DATA_LINK_API_KEY=abc123\n", encoding="utf-8")
    monkeypatch.delenv("NASDAQ_DATA_LINK_API_KEY", raising=False)

    assert load_dotenv(env) == ["NASDAQ_DATA_LINK_API_KEY"]
    assert os.environ["NASDAQ_DATA_LINK_API_KEY"] == "abc123"


def test_existing_environment_wins_over_the_file(tmp_path, monkeypatch):
    """В CI ключи приходят из секретов, и файл не должен их перебивать."""
    from factorbot.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("NASDAQ_DATA_LINK_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "from_ci")

    assert load_dotenv(env) == []
    assert os.environ["NASDAQ_DATA_LINK_API_KEY"] == "from_ci"


def test_comments_quotes_and_export_prefix_are_handled(tmp_path, monkeypatch):
    from factorbot.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        '# комментарий\n\nexport A="one"\nB=\'two\'\nC=\nмусор без равенства\n',
        encoding="utf-8",
    )
    for name in "ABC":
        monkeypatch.delenv(name, raising=False)

    assert load_dotenv(env) == ["A", "B", "C"]
    assert os.environ["A"] == "one"
    assert os.environ["B"] == "two"
    assert os.environ["C"] == ""


def test_missing_dotenv_is_not_an_error(tmp_path):
    from factorbot.config import load_dotenv

    assert load_dotenv(tmp_path / "нет-такого") == []


def test_repository_ships_an_example_but_never_a_real_key():
    root = CONFIG_PATH.parent.parent
    assert (root / ".env.example").exists()
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored, "ключи не должны попадать в git"
