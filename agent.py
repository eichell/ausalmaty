import json
from datetime import datetime
import pytz
import anthropic
import database as db
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, BOT_TIMEZONE
from tools.flight_search import search_flights
from tools.hotel_search import search_hotels
from tools.web_search import web_search, news_search
from tools.weather import get_weather
from tools.calendar_tool import add_calendar_event, get_calendar_events, delete_calendar_event

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

TOOLS = [
    {
        "name": "search_flights",
        "description": "Поиск авиабилетов. Возвращает доступные рейсы с ценами.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "IATA код аэропорта отправления (например: ALA, MOW, JFK)",
                },
                "destination": {
                    "type": "string",
                    "description": "IATA код аэропорта назначения (например: DXB, IST, LHR)",
                },
                "departure_date": {
                    "type": "string",
                    "description": "Дата вылета в формате YYYY-MM-DD",
                },
                "return_date": {
                    "type": "string",
                    "description": "Дата возвращения YYYY-MM-DD (опционально, для туда-обратно)",
                },
                "adults": {
                    "type": "integer",
                    "description": "Количество взрослых пассажиров (по умолчанию 1)",
                },
                "cabin_class": {
                    "type": "string",
                    "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
                    "description": "Класс обслуживания",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Максимальное количество результатов (1-10)",
                },
            },
            "required": ["origin", "destination", "departure_date"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Поиск и бронирование отелей в указанном городе.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city_code": {
                    "type": "string",
                    "description": "IATA код города (например: ALA для Алматы, LON для Лондона, PAR для Парижа)",
                },
                "check_in": {
                    "type": "string",
                    "description": "Дата заезда YYYY-MM-DD",
                },
                "check_out": {
                    "type": "string",
                    "description": "Дата выезда YYYY-MM-DD",
                },
                "adults": {
                    "type": "integer",
                    "description": "Количество гостей",
                },
                "ratings": {
                    "type": "string",
                    "description": "Минимальный рейтинг звёзд через запятую (например: '4,5' для 4 и 5 звёзд)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Максимальное количество отелей (1-10)",
                },
            },
            "required": ["city_code", "check_in", "check_out"],
        },
    },
    {
        "name": "web_search",
        "description": "Поиск информации в интернете по любому запросу.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Количество результатов (1-10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "news_search",
        "description": "Поиск последних новостей по теме.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Тема для поиска новостей",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Количество новостей",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Получить прогноз погоды для города.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города (на английском или русском)",
                },
                "days": {
                    "type": "integer",
                    "description": "Количество дней прогноза (1-5)",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "add_calendar_event",
        "description": "Добавить событие, встречу, задачу или напоминание в календарь.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Название события",
                },
                "start_datetime": {
                    "type": "string",
                    "description": "Дата и время начала в формате YYYY-MM-DD HH:MM или YYYY-MM-DD",
                },
                "description": {
                    "type": "string",
                    "description": "Описание события (опционально)",
                },
                "end_datetime": {
                    "type": "string",
                    "description": "Дата и время окончания (опционально)",
                },
                "location": {
                    "type": "string",
                    "description": "Место проведения (опционально)",
                },
                "category": {
                    "type": "string",
                    "enum": ["meeting", "travel", "personal", "work", "health", "general"],
                    "description": "Категория события",
                },
            },
            "required": ["title", "start_datetime"],
        },
    },
    {
        "name": "get_calendar_events",
        "description": "Получить список событий из календаря на указанный период.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["today", "week", "month", "year"],
                    "description": "Период: today (сегодня), week (неделя), month (месяц), year (год)",
                },
                "from_date": {
                    "type": "string",
                    "description": "Начало диапазона YYYY-MM-DD (опционально, если не указан period)",
                },
                "to_date": {
                    "type": "string",
                    "description": "Конец диапазона YYYY-MM-DD (опционально)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "delete_calendar_event",
        "description": "Удалить событие из календаря по ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "integer",
                    "description": "ID события для удаления",
                },
            },
            "required": ["event_id"],
        },
    },
]


def _get_system_prompt() -> str:
    now = datetime.now(pytz.timezone(BOT_TIMEZONE))
    return f"""Ты — Goldee, умная и профессиональная секретарь-ассистент. Сегодня: {now.strftime('%d %B %Y, %H:%M')} ({BOT_TIMEZONE}).

Твои возможности:
🛫 Поиск авиабилетов — ищешь рейсы по всему миру через Amadeus
🏨 Поиск отелей — находишь лучшие варианты размещения
🔍 Поиск в интернете — ищешь любую информацию в сети
📰 Новости — находишь последние новости по любой теме
🌤 Погода — прогноз погоды для любого города
📅 Календарь — планируешь день, неделю, месяц и год

Правила работы:
- Всегда отвечай на том языке, на котором написал пользователь (русский, казахский, английский)
- Будь дружелюбной, профессиональной и инициативной
- При поиске рейсов уточняй даты и аэропорты, если не указаны
- IATA коды аэропортов: Алматы=ALA, Астана=NQZ, Москва=SVO/DME, Дубай=DXB, Стамбул=IST, Лондон=LHR, Париж=CDG, Нью-Йорк=JFK, Пекин=PEK, Токио=NRT
- Для городских кодов отелей: Алматы=ALA, Лондон=LON, Париж=PAR, Нью-Йорк=NYC, Дубай=DXB
- Форматируй результаты красиво с эмодзи и структурированным текстом
- При добавлении событий в календарь используй формат даты YYYY-MM-DD HH:MM
- Если пользователь просит спланировать путешествие — предлагай полный пакет: рейсы + отель + погода
- Когда показываешь события календаря — форматируй их по датам наглядно"""


async def _execute_tool(tool_name: str, tool_input: dict, user_id: int) -> str:
    if tool_name == "search_flights":
        result = await search_flights(**tool_input)
    elif tool_name == "search_hotels":
        result = await search_hotels(**tool_input)
    elif tool_name == "web_search":
        result = await web_search(**tool_input)
    elif tool_name == "news_search":
        result = await news_search(**tool_input)
    elif tool_name == "get_weather":
        result = await get_weather(**tool_input)
    elif tool_name == "add_calendar_event":
        result = add_calendar_event(user_id=user_id, **tool_input)
    elif tool_name == "get_calendar_events":
        result = get_calendar_events(user_id=user_id, **tool_input)
    elif tool_name == "delete_calendar_event":
        result = delete_calendar_event(user_id=user_id, **tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)


async def process_message(user_id: int, user_message: str) -> str:
    """Process user message through Claude AI agent with tools."""
    db.save_message(user_id, "user", user_message)
    history = db.get_history(user_id, limit=20)

    messages = history[:-1]  # exclude the last (just saved) message, re-add it cleanly
    messages = [{"role": m["role"], "content": m["content"]} for m in history[:-1]]
    messages.append({"role": "user", "content": user_message})

    max_iterations = 10
    for _ in range(max_iterations):
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_get_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            db.save_message(user_id, "assistant", text)
            return text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_output = await _execute_tool(block.name, block.input, user_id)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return "Извините, произошла ошибка при обработке запроса. Попробуйте снова."
