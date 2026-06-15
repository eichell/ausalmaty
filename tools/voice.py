import io
import httpx
from config import GROQ_API_KEY, OPENAI_API_KEY


async def transcribe_audio(audio_bytes: bytes, language: str = "ru") -> str:
    """Transcribe audio using Groq Whisper (free) or OpenAI Whisper."""
    if GROQ_API_KEY:
        return await _transcribe_groq(audio_bytes, language)
    elif OPENAI_API_KEY:
        return await _transcribe_openai(audio_bytes, language)
    else:
        return ""


async def _transcribe_groq(audio_bytes: bytes, language: str) -> str:
    """Transcribe using Groq's free Whisper API."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg")},
            data={"model": "whisper-large-v3-turbo", "language": language},
        )
        resp.raise_for_status()
        return resp.json().get("text", "")


async def _transcribe_openai(audio_bytes: bytes, language: str) -> str:
    """Transcribe using OpenAI Whisper API."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg")},
            data={"model": "whisper-1", "language": language},
        )
        resp.raise_for_status()
        return resp.json().get("text", "")
