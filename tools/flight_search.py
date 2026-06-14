import httpx
from config import AMADEUS_API_KEY, AMADEUS_API_SECRET

_token_cache: dict = {}


async def _get_amadeus_token() -> str:
    if _token_cache.get("token") and _token_cache.get("expires_at", 0) > __import__("time").time():
        return _token_cache["token"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://test.api.amadeus.com/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": AMADEUS_API_KEY,
                "client_secret": AMADEUS_API_SECRET,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = __import__("time").time() + data["expires_in"] - 60
        return _token_cache["token"]


async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = "",
    adults: int = 1,
    max_results: int = 5,
    cabin_class: str = "ECONOMY",
) -> dict:
    """Search for available flights using Amadeus API."""
    if not AMADEUS_API_KEY:
        return {"error": "Amadeus API key not configured. Please set AMADEUS_API_KEY in .env"}

    try:
        token = await _get_amadeus_token()
        params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
            "travelClass": cabin_class,
        }
        if return_date:
            params["returnDate"] = return_date

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://test.api.amadeus.com/v2/shopping/flight-offers",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

        offers = data.get("data", [])
        if not offers:
            return {"message": "Рейсы не найдены по вашему запросу.", "flights": []}

        results = []
        for offer in offers[:max_results]:
            price = offer["price"]
            itineraries = offer["itineraries"]

            segments_info = []
            for itin in itineraries:
                for seg in itin["segments"]:
                    segments_info.append({
                        "from": seg["departure"]["iataCode"],
                        "to": seg["arrival"]["iataCode"],
                        "departure": seg["departure"]["at"],
                        "arrival": seg["arrival"]["at"],
                        "flight": f"{seg['carrierCode']}{seg['number']}",
                        "duration": seg.get("duration", ""),
                    })

            results.append({
                "id": offer["id"],
                "price": f"{price['total']} {price['currency']}",
                "segments": segments_info,
                "seats_available": offer.get("numberOfBookableSeats", "?"),
            })

        return {"flights": results, "count": len(results)}

    except httpx.HTTPStatusError as e:
        return {"error": f"Ошибка API Amadeus: {e.response.status_code} - {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Ошибка поиска рейсов: {str(e)}"}


async def get_flight_offers(origin: str, destination: str, date: str) -> dict:
    """Alias for search_flights with simpler signature."""
    return await search_flights(origin, destination, date)
