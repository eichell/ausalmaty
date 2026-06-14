import asyncio
import logging
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode

import database as db
import agent
from config import TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я — *Аяша*, ваш персональный секретарь-ассистент.\n\n"
        "Что я умею:\n"
        "🛫 Поиск авиабилетов\n"
        "🏨 Поиск и бронирование отелей\n"
        "🔍 Поиск в интернете\n"
        "📰 Последние новости\n"
        "🌤 Прогноз погоды\n"
        "📅 Планирование: день, неделя, месяц, год\n\n"
        "Просто напишите мне, что вам нужно!\n\n"
        "_Например: «Найди билеты из Алматы в Дубай на 25 декабря»_",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    await update.message.reply_text(
        "*Команды:*\n\n"
        "/start — Начать работу\n"
        "/help — Справка\n"
        "/clear — Очистить историю диалога\n"
        "/calendar — Мой календарь на неделю\n"
        "/today — События на сегодня\n"
        "/month — События на месяц\n"
        "/year — Планы на год\n"
        "/bookings — Мои сохранённые бронирования\n\n"
        "*Примеры запросов:*\n"
        "• «Найди билеты Алматы → Дубай 15 января туда-обратно»\n"
        "• «Отели в Лондоне с 10 по 15 марта 4 звезды»\n"
        "• «Погода в Барселоне на 5 дней»\n"
        "• «Добавь встречу с командой в пятницу в 15:00»\n"
        "• «Что нового про Казахстан?»\n"
        "• «Спланируй мне поездку в Токио на неделю в апреле»",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    db.clear_history(update.effective_user.id)
    await update.message.reply_text("🗑 История диалога очищена.")


async def cmd_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text("📅 Загружаю события на неделю...")
    await _show_events(update, "week")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await _show_events(update, "today")


async def cmd_month(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await _show_events(update, "month")


async def cmd_year(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await _show_events(update, "year")


async def _show_events(update: Update, period: str):
    from tools.calendar_tool import get_calendar_events
    user_id = update.effective_user.id
    result = get_calendar_events(user_id=user_id, period=period)

    period_names = {
        "today": "сегодня",
        "week": "эту неделю",
        "month": "этот месяц",
        "year": "этот год",
    }

    events = result.get("events", [])
    if not events:
        await update.message.reply_text(
            f"📭 На {period_names.get(period, period)} событий нет.\n\n"
            "Напишите мне, чтобы добавить событие!"
        )
        return

    cat_icons = {
        "meeting": "🤝",
        "travel": "✈️",
        "personal": "👤",
        "work": "💼",
        "health": "🏥",
        "general": "📌",
    }

    lines = [f"📅 *События на {period_names.get(period, period)}:*\n"]
    current_date = ""
    for e in events:
        date_part = e["start_datetime"][:10]
        if date_part != current_date:
            current_date = date_part
            lines.append(f"\n*{date_part}*")

        icon = cat_icons.get(e.get("category", "general"), "📌")
        time_part = e["start_datetime"][11:16] if len(e["start_datetime"]) > 10 else ""
        time_str = f" {time_part}" if time_part else ""
        lines.append(f"{icon} [{e['id']}]{time_str} {e['title']}")
        if e.get("location"):
            lines.append(f"   📍 {e['location']}")
        if e.get("description"):
            lines.append(f"   💬 {e['description']}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_bookings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    bookings = db.get_bookings(user_id)

    if not bookings:
        await update.message.reply_text("📭 У вас нет сохранённых бронирований.")
        return

    lines = ["📋 *Ваши бронирования:*\n"]
    for b in bookings:
        icon = "✈️" if b["type"] == "flight" else "🏨"
        lines.append(f"{icon} *{b['type'].upper()}* #{b['id']}")
        lines.append(f"   Статус: {b['status']}")
        lines.append(f"   Дата: {b['created_at'][:10]}")
        lines.append("")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    user_text = update.message.text
    if not user_text:
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        response = await agent.process_message(user.id, user_text)
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                await update.message.reply_text(
                    response[i:i+4096],
                    parse_mode=ParseMode.MARKDOWN,
                )
        else:
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error processing message for user {user.id}: {e}", exc_info=True)
        await update.message.reply_text(
            "😔 Произошла ошибка. Попробуйте ещё раз или используйте /clear для сброса диалога."
        )


async def post_init(application: Application):
    commands = [
        BotCommand("start", "Начать работу"),
        BotCommand("help", "Справка и примеры"),
        BotCommand("calendar", "Мой календарь на неделю"),
        BotCommand("today", "События на сегодня"),
        BotCommand("month", "События на месяц"),
        BotCommand("year", "Планы на год"),
        BotCommand("bookings", "Мои бронирования"),
        BotCommand("clear", "Очистить историю диалога"),
    ]
    await application.bot.set_my_commands(commands)


def main():
    db.init_db()

    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set! Copy .env.example to .env and fill in values.")
        return

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set!")
        return

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("calendar", cmd_calendar))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("month", cmd_month))
    app.add_handler(CommandHandler("year", cmd_year))
    app.add_handler(CommandHandler("bookings", cmd_bookings))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Аяша запущена! Ожидаю сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    from config import ANTHROPIC_API_KEY
    main()
