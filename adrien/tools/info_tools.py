"""Information lookup: weather, news, web search, time (spec 7.6).

Every tool here has a **keyless default**, because a voice assistant that
cannot answer "what's the weather" until you sign up for an API is a broken
voice assistant:

* weather - Open-Meteo (no key, no signup). OpenWeather is used instead when
  `OPENWEATHER_API_KEY` is set.
* news    - Google News RSS (no key). NewsAPI when `NEWSAPI_KEY` is set.
* search  - DuckDuckGo's Instant Answer API (no key). Brave Search when
  `BRAVE_SEARCH_API_KEY` is set, which gives much better results.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any
from xml.etree import ElementTree

from adrien.config import env_str
from adrien.core.http import get_client
from adrien.logging_setup import get_logger
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)

# WMO weather codes, as spoken words rather than numbers.
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


async def _geocode(place: str) -> tuple[float, float, str] | None:
    """Place name -> (lat, lon, label) using Open-Meteo's free geocoder."""
    try:
        response = await get_client().get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        results = (response.json() or {}).get("results") or []
    except Exception as exc:
        log.warning("geocoding failed: %s", type(exc).__name__)
        return None
    if not results:
        return None
    hit = results[0]
    label = ", ".join(part for part in (hit.get("name"), hit.get("country")) if part)
    return float(hit["latitude"]), float(hit["longitude"]), label


@tool(category="info", timeout=20.0)
async def get_weather(location: str = "", units: str = "celsius") -> ToolResult:
    """Current weather and today's outlook for a place.

    Args:
        location: City or place name. Defaults to the Mac's own location.
        units: celsius or fahrenheit.
    """
    place = location or env_str("ADRIEN_DEFAULT_LOCATION", "")
    if not place:
        return ToolResult.failure(
            "no location given, and no default is set - say which place you mean"
        )

    coordinates = await _geocode(place)
    if coordinates is None:
        return ToolResult.failure(f"could not find a place called {place}")
    latitude, longitude, label = coordinates

    fahrenheit = units.lower().startswith("f")
    try:
        response = await get_client().get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit": "fahrenheit" if fahrenheit else "celsius",
                "wind_speed_unit": "mph" if fahrenheit else "kmh",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=15,
        )
        body = response.json() or {}
    except Exception as exc:
        return ToolResult.failure(f"could not reach the weather service ({type(exc).__name__})")

    current = body.get("current") or {}
    daily = body.get("daily") or {}
    if not current:
        return ToolResult.failure("the weather service returned nothing usable")

    symbol = "F" if fahrenheit else "C"
    conditions = _WMO.get(int(current.get("weather_code", -1)), "unclear")
    temperature = round(current.get("temperature_2m", 0))
    high = round((daily.get("temperature_2m_max") or [0])[0])
    low = round((daily.get("temperature_2m_min") or [0])[0])
    rain_chance = (daily.get("precipitation_probability_max") or [0])[0]

    speak = f"{label}: {temperature} degrees and {conditions}, high of {high}"
    if rain_chance and rain_chance >= 30:
        speak += f", {rain_chance} percent chance of rain"

    return ToolResult.success(
        {
            "location": label,
            "temperature": temperature,
            "feels_like": round(current.get("apparent_temperature", temperature)),
            "conditions": conditions,
            "humidity": current.get("relative_humidity_2m"),
            "wind": current.get("wind_speed_10m"),
            "high": high,
            "low": low,
            "rain_chance": rain_chance,
            "units": symbol,
        },
        speak=speak,
    )


@tool(category="info", timeout=20.0)
async def get_news_headlines(topic: str = "", limit: int = 5) -> ToolResult:
    """Recent news headlines, optionally about a specific topic.

    Args:
        topic: What to look for. Leave empty for top stories.
        limit: How many headlines to return.
    """
    limit = max(1, min(limit, 10))
    api_key = env_str("NEWSAPI_KEY")

    if api_key:
        try:
            response = await get_client().get(
                "https://newsapi.org/v2/top-headlines" if not topic
                else "https://newsapi.org/v2/everything",
                params={"q": topic, "pageSize": limit, "language": "en",
                        "sortBy": "publishedAt"} if topic
                else {"pageSize": limit, "language": "en"},
                headers={"x-api-key": api_key},
                timeout=15,
            )
            articles = (response.json() or {}).get("articles") or []
            if articles:
                items = [{"title": a.get("title"), "source": (a.get("source") or {}).get("name")}
                         for a in articles[:limit]]
                return ToolResult.success(
                    {"topic": topic or "top stories", "headlines": items},
                    speak="; ".join(item["title"] for item in items[:3]),
                )
        except Exception as exc:
            log.warning("NewsAPI failed, falling back to RSS: %s", type(exc).__name__)

    # Keyless fallback: Google News RSS.
    url = "https://news.google.com/rss"
    params = {"hl": "en", "gl": "US", "ceid": "US:en"}
    if topic:
        url = "https://news.google.com/rss/search"
        params["q"] = topic

    try:
        response = await get_client().get(url, params=params, timeout=15)
        root = ElementTree.fromstring(response.text)
    except Exception as exc:
        return ToolResult.failure(f"could not fetch the news ({type(exc).__name__})")

    items: list[dict[str, Any]] = []
    for item in list(root.iterfind(".//item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        # Google appends " - Publisher" to every title.
        headline, _, source = title.rpartition(" - ")
        items.append({"title": headline or title, "source": source})

    if not items:
        return ToolResult.success({"headlines": []}, speak="no headlines found")
    return ToolResult.success(
        {"topic": topic or "top stories", "headlines": items},
        speak="; ".join(item["title"] for item in items[:3]),
    )


@tool(category="info", timeout=40.0)
async def web_search(query: str, answer: bool = True) -> ToolResult:
    """Search the web and give a short spoken answer.

    Args:
        query: What to search for.
        answer: Summarise the results into one spoken answer rather than
            listing them.
    """
    results = await _search(query)
    if isinstance(results, str):
        return ToolResult.failure(results)
    if not results:
        return ToolResult.success({"results": []}, speak=f"nothing useful for {query}")

    if not answer:
        return ToolResult.success(
            {"query": query, "results": results},
            speak=f"top result: {results[0]['title']}",
        )

    from adrien.core.llm_router import LLMRouter

    context = "\n\n".join(
        f"{item['title']}\n{item['snippet']}" for item in results[:5] if item.get("snippet")
    )
    try:
        synthesis = await LLMRouter().complete(
            f"Answer this out loud in one or two sentences, using only what is below. "
            f"If it does not answer the question, say so.\n\n"
            f"Question: {query}\n\nSearch results:\n{context[:5000]}",
            tier="smart",
            max_tokens=220,
        )
    except Exception as exc:
        # The search itself worked; only the summary failed. Hand back the raw
        # results rather than losing the answer entirely.
        log.warning("search synthesis failed: %s", exc)
        return ToolResult.success(
            {"query": query, "results": results},
            speak=f"top result: {results[0]['title']}",
        )

    return ToolResult.success({"query": query, "answer": synthesis, "sources": results[:3]},
                              speak=synthesis)


async def _search(query: str) -> list[dict[str, str]] | str:
    """Brave when a key is configured, DuckDuckGo otherwise."""
    brave_key = env_str("BRAVE_SEARCH_API_KEY")
    client = get_client()

    if brave_key:
        try:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 6},
                headers={"x-subscription-token": brave_key, "accept": "application/json"},
                timeout=15,
            )
            if response.status_code == 200:
                web = (response.json() or {}).get("web") or {}
                return [
                    {"title": item.get("title", ""),
                     "snippet": _strip_tags(item.get("description", "")),
                     "url": item.get("url", "")}
                    for item in (web.get("results") or [])
                ]
            log.warning("Brave search returned %d, falling back", response.status_code)
        except Exception as exc:
            log.warning("Brave search failed, falling back: %s", type(exc).__name__)

    try:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=15,
        )
        body = response.json() or {}
    except Exception as exc:
        return f"could not reach a search engine ({type(exc).__name__})"

    results: list[dict[str, str]] = []
    if body.get("AbstractText"):
        results.append({
            "title": body.get("Heading", query),
            "snippet": body["AbstractText"],
            "url": body.get("AbstractURL", ""),
        })
    for topic in (body.get("RelatedTopics") or [])[:6]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic["Text"][:80],
                "snippet": topic["Text"],
                "url": topic.get("FirstURL", ""),
            })
    return results


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


@tool(category="info")
def get_time(location: str = "") -> ToolResult:
    """The current time and date, here or in another place.

    Args:
        location: A city or IANA timezone name. Defaults to local time.
    """
    if not location:
        now = dt.datetime.now().astimezone()
        return ToolResult.success(
            {"time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y"),
             "timezone": str(now.tzinfo)},
            speak=f"it's {now.strftime('%H:%M')} on {now.strftime('%A the %d')}",
        )

    from zoneinfo import ZoneInfo, available_timezones

    wanted = location.strip().replace(" ", "_").lower()
    zone_name = next(
        (zone for zone in available_timezones()
         if zone.lower() == wanted or zone.lower().split("/")[-1] == wanted),
        None,
    )
    if zone_name is None:
        return ToolResult.failure(f"could not find a timezone for {location}")

    now = dt.datetime.now(ZoneInfo(zone_name))
    return ToolResult.success(
        {"time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y"), "timezone": zone_name},
        speak=f"it's {now.strftime('%H:%M')} in {zone_name.split('/')[-1].replace('_', ' ')}",
    )
