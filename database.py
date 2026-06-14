import sqlite3
import json
from datetime import datetime
from config import DB_PATH
import os

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                start_datetime TEXT NOT NULL,
                end_datetime TEXT,
                location TEXT,
                category TEXT DEFAULT 'general',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                details TEXT NOT NULL,
                status TEXT DEFAULT 'saved',
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)


def add_event(user_id: int, title: str, start_datetime: str, description: str = "",
              end_datetime: str = "", location: str = "", category: str = "general") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calendar_events (user_id, title, description, start_datetime, end_datetime, location, category) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, title, description, start_datetime, end_datetime, location, category)
        )
        return cur.lastrowid


def get_events(user_id: int, from_date: str = "", to_date: str = "") -> list:
    with get_conn() as conn:
        if from_date and to_date:
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE user_id=? AND start_datetime BETWEEN ? AND ? ORDER BY start_datetime",
                (user_id, from_date, to_date)
            ).fetchall()
        elif from_date:
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE user_id=? AND start_datetime >= ? ORDER BY start_datetime",
                (user_id, from_date)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE user_id=? ORDER BY start_datetime",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def delete_event(user_id: int, event_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM calendar_events WHERE id=? AND user_id=?", (event_id, user_id)
        )
        return cur.rowcount > 0


def save_message(user_id: int, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )


def get_history(user_id: int, limit: int = 20) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_history(user_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))


def save_booking(user_id: int, booking_type: str, details: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO bookings (user_id, type, details) VALUES (?, ?, ?)",
            (user_id, booking_type, json.dumps(details, ensure_ascii=False))
        )
        return cur.lastrowid


def get_bookings(user_id: int) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bookings WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d["details"])
            result.append(d)
        return result
