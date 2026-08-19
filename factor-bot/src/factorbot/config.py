"""Загрузка config/strategy.yaml (ТЗ 11: никаких констант в коде).

Доступ через точку, потому что `cfg.portfolio.top_n` читается, а
`cfg["portfolio"]["top_n"]` — расшифровывается. Проверка обязательных секций на
входе: опечатка в ключе должна падать при старте, а не выдавать None в середине
бэктеста.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/strategy.yaml")

REQUIRED_SECTIONS = (
    "data", "periods", "universe", "factors", "portfolio", "regime_filter",
    "costs", "reporting",
)


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
