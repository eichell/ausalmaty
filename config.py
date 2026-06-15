import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "Asia/Almaty")

_allowed = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = [int(x.strip()) for x in _allowed.split(",") if x.strip()] if _allowed else []

CLAUDE_MODEL = "claude-sonnet-4-6"
DB_PATH = "data/ausalmaty.db"
