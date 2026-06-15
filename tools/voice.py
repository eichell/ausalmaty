import httpx
from config import GROQ_API_KEY, OPENAI_API_KEY

# Prompt helps Whisper recognize Russian/Kazakh travel terms correctly
_PROMPT = (
    "Разговорный русский или казахский язык. "
    "Тематика: путешествия, авиабилеты, отели, планирование, календарь, встречи, погода."
)


async def transcribe_audio(audio_bytes: bytes, language: str = "") -> str:
    """Transcribe audio using Groq Whisper (free) or OpenAI Whisper."""
    if GROQ_API_KEY:
        return await _transcribe_groq(audio_bytes, language)
    elif OPENAI_API_KEY:
        return await _transcribe_openai(audio_bytes, language)
    return ""


async def _transcribe_groq(audio_bytes: bytes, language: str) -> str:
    """Transcribe using Groq Whisper large-v3 (accurate, free tier)."""
    data = {
        "model": "whisper-large-v3",
        "prompt": _PROMPT,
        "response_format": "text",
        "temperature": "0",
    }
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg; codecs=opus")},
            data=data,
        )
        resp.raise_for_status()
        result = resp.text.strip() if resp.headers.get("content-type", "").startswith("text") else resp.json().get("text", "")
        return result


async def _transcribe_openai(audio_bytes: bytes, language: str) -> str:
    """Transcribe using OpenAI Whisper-1."""
    data = {
        "model": "whisper-1",
        "prompt": _PROMPT,
        "response_format": "text",
        "temperature": "0",
    }
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg; codecs=opus")},
            data=data,
        )
        resp.raise_for_status()
        result = resp.text.strip() if resp.headers.get("content-type", "").startswith("text") else resp.json().get("text", "")
        return result
