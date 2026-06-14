from duckduckgo_search import DDGS


async def web_search(query: str, max_results: int = 5, region: str = "ru-ru") -> dict:
    """Search the web using DuckDuckGo (no API key required)."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": f"Ошибка поиска: {str(e)}", "query": query}


async def news_search(query: str, max_results: int = 5) -> dict:
    """Search for recent news using DuckDuckGo."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "date": r.get("date", ""),
                    "source": r.get("source", ""),
                })
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": f"Ошибка поиска новостей: {str(e)}", "query": query}
