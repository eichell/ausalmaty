import httpx
from config import AMADEUS_API_KEY, AMADEUS_API_SECRET
from tools.flight_search import _get_amadeus_token


async def search_hotels(
    city_code: str,
    check_in: str,
    check_out: str,
    adults: int = 1,
    max_results: int = 5,
    ratings: str = "3,4,5",
) -> dict:
    """Search for hotels using Amadeus Hotel Search API."""
    if not AMADEUS_API_KEY:
        return {"error": "Amadeus API key not configured. Please set AMADEUS_API_KEY in .env"}

    try:
        token = await _get_amadeus_token()

        async with httpx.AsyncClient() as client:
            # Step 1: Get hotel list for city
            hotel_resp = await client.get(
                "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city",
                params={
                    "cityCode": city_code.upper(),
                    "hotelSource": "ALL",
                    "ratings": ratings,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            hotel_resp.raise_for_status()
            hotels_data = hotel_resp.json()
            hotel_ids = [h["hotelId"] for h in hotels_data.get("data", [])[:20]]

            if not hotel_ids:
                return {"message": f"Отели в {city_code} не найдены.", "hotels": []}

            # Step 2: Get offers for found hotels
            offers_resp = await client.get(
                "https://test.api.amadeus.com/v3/shopping/hotel-offers",
                params={
                    "hotelIds": ",".join(hotel_ids),
                    "checkInDate": check_in,
                    "checkOutDate": check_out,
                    "adults": adults,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            offers_resp.raise_for_status()
            offers_data = offers_resp.json()

        results = []
        for item in offers_data.get("data", [])[:max_results]:
            hotel = item.get("hotel", {})
            offers = item.get("offers", [{}])
            offer = offers[0] if offers else {}
            price = offer.get("price", {})

            results.append({
                "hotel_id": hotel.get("hotelId", ""),
                "name": hotel.get("name", "Unknown"),
                "rating": hotel.get("rating", "?"),
                "city": hotel.get("cityCode", city_code),
                "price_per_night": f"{price.get('total', '?')} {price.get('currency', 'USD')}",
                "room_type": offer.get("room", {}).get("typeEstimated", {}).get("category", "Standard"),
                "check_in": check_in,
                "check_out": check_out,
            })

        return {"hotels": results, "count": len(results)}

    except httpx.HTTPStatusError as e:
        return {"error": f"Ошибка API Amadeus Hotels: {e.response.status_code} - {e.response.text[:300]}"}
    except Exception as e:
        return {"error": f"Ошибка поиска отелей: {str(e)}"}
