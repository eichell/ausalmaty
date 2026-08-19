"""Загрузка config/strategy.yaml (ТЗ 11: никаких констант в коде) и ключей из .env.

Доступ через точку, потому что `cfg.portfolio.top_n` читается, а
`cfg["portfolio"]["top_n"]` — расшифровывается. Проверка обязательных секций на
входе: опечатка в ключе должна падать при старте, а не выдавать None в середине
бэктеста.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/strategy.yaml")
DEFAULT_ENV_PATH = Path(".env")

REQUIRED_SECTIONS = (
    "data", "periods", "universe", "factors", "portfolio", "regime_filter",
    "costs", "reporting",
)


def load_dotenv(path: str | Path = DEFAULT_ENV_PATH, *, override: bool = False) -> list[str]:
    """Переносит ключи из .env в окружение процесса.

    Ключи API в репозиторий не коммитятся: репозиторий на GitHub, а
    опубликованный ключ находят сканеры. В git идёт только .env.example.

    Уже заданная переменная окружения по умолчанию сильнее файла: в CI ключи
    приходят из секретов, и файл не должен их перебивать.

    Returns:
        Имена переменных, которые были установлены из файла.
    """
    path = Path(path)
    if not path.exists():
        # Поиск вверх: скрипты запускают и из корня проекта, и из подкаталогов.
        for parent in Path.cwd().resolve().parents[:3]:
            candidate = parent / DEFAULT_ENV_PATH.name
            if candidate.exists():
                path = candidate
                break
        else:
            return []

    applied: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


class Section(Mapping):
    """Словарь с доступом через точку. Неизвестный ключ — ошибка, не None."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {
            k: Section(v) if isinstance(v, Mapping) else v for k, v in data.items()
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"В конфиге нет параметра {name!r}. Доступны: {sorted(self._data)}"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Section({sorted(self._data)})"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Section:
    """Читает и проверяет конфиг стратегии.

    Raises:
        FileNotFoundError: файла нет.
        ValueError: отсутствует обязательная секция или веса факторов не дают 1.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден конфиг стратегии: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"В конфиге нет обязательных секций: {missing}")

    weights = raw["factors"].get("weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Веса факторов должны давать 1.0, получено {total} ({weights}).")

    return Section(raw)
