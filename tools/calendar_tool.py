from datetime import datetime, timedelta
import pytz
import database as db
from config import BOT_TIMEZONE


def _tz():
    return pytz.timezone(BOT_TIMEZONE)


def add_calendar_event(
    user_id: int,
    title: str,
    start_datetime: str,
    description: str = "",
    end_datetime: str = "",
    location: str = "",
    category: str = "general",
) -> dict:
    """Add event to user's calendar."""
    try:
        event_id = db.add_event(
            user_id=user_id,
            title=title,
            start_datetime=start_datetime,
            description=description,
            end_datetime=end_datetime,
            location=location,
            category=category,
        )
        return {
            "success": True,
            "event_id": event_id,
            "message": f"Событие '{title}' добавлено на {start_datetime}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_calendar_events(
    user_id: int,
    period: str = "week",
    from_date: str = "",
    to_date: str = "",
) -> dict:
    """Get calendar events for specified period."""
    try:
        now = datetime.now(_tz())

        if from_date and to_date:
            events = db.get_events(user_id, from_date, to_date)
        elif period == "today":
            start = now.strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d") + " 23:59:59"
            events = db.get_events(user_id, start, end)
        elif period == "week":
            start = now.strftime("%Y-%m-%d")
            end = (now + timedelta(days=7)).strftime("%Y-%m-%d")
            events = db.get_events(user_id, start, end)
        elif period == "month":
            start = now.strftime("%Y-%m-%d")
            end = (now + timedelta(days=30)).strftime("%Y-%m-%d")
            events = db.get_events(user_id, start, end)
        elif period == "year":
            start = now.strftime("%Y-%m-%d")
            end = (now + timedelta(days=365)).strftime("%Y-%m-%d")
            events = db.get_events(user_id, start, end)
        else:
            events = db.get_events(user_id, now.strftime("%Y-%m-%d"))

        return {
            "events": events,
            "count": len(events),
            "period": period,
        }
    except Exception as e:
        return {"error": str(e), "events": []}


def delete_calendar_event(user_id: int, event_id: int) -> dict:
    """Delete a calendar event by ID."""
    try:
        success = db.delete_event(user_id, event_id)
        if success:
            return {"success": True, "message": f"Событие #{event_id} удалено"}
        return {"success": False, "message": f"Событие #{event_id} не найдено"}
    except Exception as e:
        return {"success": False, "error": str(e)}
