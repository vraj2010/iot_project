"""Weather for the get_weather tool, from Open-Meteo (free, no key).

Much faster and more exact than a web search: one lookup for the place (cached)
and one for the forecast, usually well under a second. The answer is kept for
WEATHER_CACHE_S seconds, and the home city's is refreshed in the background, so
asking "what's the weather" is normally instant.

    python weather.py Gandhinagar      check it works from this laptop
"""

from __future__ import annotations

import copy
import threading
import time

import requests

import config
from httpsession import new_session

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes, in words Friday can say.
_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

_session = new_session()
_places: dict[str, dict] = {}
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _get(url: str, params: dict) -> requests.Response:
    """GET with a short timeout. A connection that has gone dead (the line was quiet for a while)
    is swapped for a fresh one and the request tried once more, instead of waiting for it."""
    global _session
    timeout = (3.05, config.WEATHER_TIMEOUT)
    try:
        return _session.get(url, params=params, timeout=timeout)
    except (requests.Timeout, requests.ConnectionError):
        _session = new_session()
        return _session.get(url, params=params, timeout=timeout)


def _place(name: str) -> dict:
    key = name.strip().lower()
    with _lock:
        if key in _places:
            return _places[key]
    r = _get(GEO_URL, {"name": name.split(",")[0].strip(), "count": 5, "language": "en", "format": "json"})
    r.raise_for_status()
    results = (r.json() or {}).get("results") or []
    if not results:
        raise LookupError(f"no place called {name}")
    # "Gandhinagar, Gujarat" -> prefer the result in that state/country; else the biggest place.
    hint = ",".join(name.split(",")[1:]).strip().lower()
    best = None
    if hint:
        best = next((p for p in results if hint in f"{p.get('admin1', '')} {p.get('country', '')}".lower()), None)
    best = best or max(results, key=lambda p: p.get("population") or 0)
    with _lock:
        _places[key] = best
    return best


def _fetch(place: str) -> dict:
    p = _place(place)
    r = _get(FORECAST_URL, {
        "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
    })
    r.raise_for_status()
    d = r.json()
    cur, day = d.get("current", {}), d.get("daily", {})

    def daily(i: int) -> dict:
        return {"sky": _CODES.get((day.get("weather_code") or [None] * 3)[i], "unknown"),
                "high_c": (day.get("temperature_2m_max") or [None] * 3)[i],
                "low_c": (day.get("temperature_2m_min") or [None] * 3)[i],
                "rain_chance_percent": (day.get("precipitation_probability_max") or [None] * 3)[i],
                "sunrise": ((day.get("sunrise") or [""] * 3)[i] or "")[-5:],
                "sunset": ((day.get("sunset") or [""] * 3)[i] or "")[-5:]}

    where = ", ".join(x for x in (p.get("name"), p.get("admin1"), p.get("country")) if x)
    return {
        "place": where,
        "now": {"sky": _CODES.get(cur.get("weather_code"), "unknown"), "temp_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"), "humidity_percent": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"), "rain_mm": cur.get("precipitation"),
                "local_time": (cur.get("time") or "").replace("T", " ")},
        "today": daily(0),
        "tomorrow": daily(1),
        "source": "Open-Meteo",
    }


def _cached(key: str) -> dict | None:
    with _lock:
        item = _cache.get(key)
    if item and time.time() - item[0] < config.WEATHER_CACHE_S:
        return copy.deepcopy(item[1])
    return None


def is_cached(place: str) -> bool:
    """True when asking for this place's weather needs no trip to the internet."""
    return config.WEATHER_CACHE_S > 0 and _cached(place.strip().lower()) is not None


def get_weather(place: str, force: bool = False) -> dict:
    """Now, today and tomorrow for `place`. A lookup is reused for WEATHER_CACHE_S seconds;
    force=True always asks Open-Meteo."""
    key = place.strip().lower()
    if not force and config.WEATHER_CACHE_S > 0:
        hit = _cached(key)
        if hit is not None:
            return hit
    out = _fetch(place)
    if config.WEATHER_CACHE_S > 0:
        with _lock:
            _cache[key] = (time.time(), out)
    return copy.deepcopy(out)


def start_prefetch(get_place, interval: float | None = None) -> threading.Event | None:
    """Keep the weather for get_place() fresh in the background (it runs every 80% of
    WEATHER_CACHE_S). Returns an Event that stops it, or None if it is switched off."""
    if config.WEATHER_CACHE_S <= 0 or not config.WEATHER_PREFETCH:
        return None
    stop = threading.Event()
    wait = interval if interval is not None else max(60.0, config.WEATHER_CACHE_S * 0.8)

    def loop() -> None:
        warned = False
        while not stop.is_set():
            try:
                place = (get_place() or "").strip()
                if place:
                    get_weather(place, force=True)
                    warned = False
            except Exception as exc:       # offline, or the city isn't known: try again next time
                if not warned:
                    print(f"[weather] background refresh failed: {exc}")
                    warned = True
            stop.wait(wait)

    threading.Thread(target=loop, name="weather-prefetch", daemon=True).start()
    return stop


if __name__ == "__main__":
    import json
    import sys
    import time

    t0 = time.time()
    print(json.dumps(get_weather(" ".join(sys.argv[1:]) or "Gandhinagar"), indent=2))
    print(f"{(time.time() - t0) * 1000:.0f} ms")
