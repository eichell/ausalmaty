import httpx
from config import OPENWEATHER_API_KEY


async def get_weather(city: str, days: int = 3) -> dict:
    """Get weather forecast for a city (1-5 days)."""
    if not OPENWEATHER_API_KEY:
        return {"error": "OpenWeatherMap API key not configured. Set OPENWEATHER_API_KEY in .env"}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric",
                    "lang": "ru",
                    "cnt": min(days * 8, 40),
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        city_info = data["city"]
        forecasts = {}
        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            if date not in forecasts:
                forecasts[date] = []
            forecasts[date].append({
                "time": item["dt_txt"].split(" ")[1][:5],
                "temp": round(item["main"]["temp"]),
                "feels_like": round(item["main"]["feels_like"]),
                "description": item["weather"][0]["description"],
                "humidity": item["main"]["humidity"],
                "wind_speed": round(item["wind"]["speed"]),
            })

        daily_summary = []
        for date, entries in list(forecasts.items())[:days]:
            temps = [e["temp"] for e in entries]
            daily_summary.append({
                "date": date,
                "temp_min": min(temps),
                "temp_max": max(temps),
                "description": entries[len(entries) // 2]["description"],
                "humidity": entries[len(entries) // 2]["humidity"],
                "wind_speed": entries[len(entries) // 2]["wind_speed"],
            })

        return {
            "city": city_info["name"],
            "country": city_info["country"],
            "forecast": daily_summary,
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Город '{city}' не найден"}
        return {"error": f"Ошибка OpenWeatherMap API: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Ошибка получения погоды: {str(e)}"}
