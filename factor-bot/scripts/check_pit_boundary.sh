#!/usr/bin/env bash
# ТЗ 4.8: прямых обращений к таблице `fundamentals` вне pit.py быть не должно.
# Ставится в CI сейчас, пока нарушать нечего. Позже это будет разбор десятков
# «легитимных» исключений, и правило умрёт.
set -uo pipefail

# Скрипт вызывается и из CI, и руками из любого каталога.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

VIOLATIONS=$(
  grep -rn --include='*.py' -E '\bfundamentals\b' src/ \
    | grep -v '^src/factorbot/data/pit\.py:' \
    || true
)

if [[ -n "$VIOLATIONS" ]]; then
  echo "Нарушение границы PIT-доступа (ТЗ 4.8):" >&2
  echo "$VIOLATIONS" >&2
  echo >&2
  echo "Используйте factorbot.data.pit.get_fundamentals()." >&2
  exit 1
fi

echo "OK: обращения к fundamentals только из pit.py"
