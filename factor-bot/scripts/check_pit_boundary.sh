#!/usr/bin/env bash
# ТЗ 4.8: обращений к таблице базы `fundamentals` вне pit.py быть не должно.
# Ставится в CI сейчас, пока нарушать нечего. Позже это будет разбор десятков
# «легитимных» исключений, и правило умрёт.
#
# Правило ищет обращение к таблице, а не упоминание слова. Разница появилась,
# когда загрузчик переписали под прямой API Sharadar: там bulk-таблица отчётности
# тоже называется `fundamentals` (прежний код SF1), и поиск по слову начал ловить
# имя таблицы поставщика в модуле, у которого нет ни одного соединения с базой.
# Ослаблять правило из-за этого нельзя, поэтому проверок стало две:
#
#   1. SQL-обращение к таблице: FROM/INTO/JOIN/UPDATE/TABLE/DELETE FROM. Это и
#      есть запрещённый доступ, в каком бы файле он ни стоял.
#   2. Имя таблицы, собранное в строку рядом с вызовом соединения — так запрет
#      обходят, не написав SQL буквально.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

ALLOWED='^src/factorbot/data/pit\.py:'

sql_access=$(
  grep -rniE --include='*.py' \
    '(from|into|join|update|table|truncate)[[:space:]]+fundamentals\b' src/ \
    | grep -vE "$ALLOWED" || true
)

# Обход через переменную: имя таблицы в строке и тут же execute/register.
smuggled=$(
  grep -rnE --include='*.py' \
    '(execute|register|sql|query)[^#]*["'"'"']fundamentals["'"'"']' src/ \
    | grep -vE "$ALLOWED" || true
)

# Имя таблицы литералом. Разрешено только в pit.py и в загрузчике: у поставщика
# bulk-таблица зовётся так же, но у загрузчика нет соединения с базой — это
# проверяется ниже отдельно. Все прочие модули получают имя через константу
# sharadar.FUNDAMENTALS_TABLE, поэтому литерал у них означает обход правила:
# имя в переменной и SQL, собранный f-строкой.
literal=$(
  grep -rnE --include='*.py' '["'"'"']fundamentals["'"'"']' src/     | grep -vE "$ALLOWED"     | grep -vE '^src/factorbot/data/sharadar\.py:' || true
)

# Загрузчик не имеет права стать исключением незаметно: если в нём когда-нибудь
# появится соединение с базой, поблажка выше перестанет быть безопасной.
loader_db=$(
  grep -nE '^[[:space:]]*(import|from)[[:space:]]+duckdb'     src/factorbot/data/sharadar.py || true
)
if [[ -n "$loader_db" ]]; then
  loader_db="src/factorbot/data/sharadar.py: появился duckdb — поблажка на имя таблицы больше не действует"
fi

VIOLATIONS=$(printf '%s\n%s\n%s\n%s' "$sql_access" "$smuggled" "$literal" "$loader_db" \
  | grep -v '^$' || true)

if [[ -n "$VIOLATIONS" ]]; then
  echo "Нарушение границы PIT-доступа (ТЗ 4.8):" >&2
  echo "$VIOLATIONS" >&2
  echo >&2
  echo "Используйте factorbot.data.pit.get_fundamentals()." >&2
  exit 1
fi

echo "OK: к таблице fundamentals обращается только pit.py"
