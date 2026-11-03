import time
import unittest
from unittest import mock

import _path  # noqa: F401
import requests

import config
import weather
from fakes import FakeResponse, FakeSession

GEO = {"results": [{"name": "Rajkot", "admin1": "Gujarat", "country": "India", "latitude": 22.3, "longitude": 70.8,
                    "population": 1000000}]}
FORECAST = {"current": {"time": "2026-10-19T12:00", "temperature_2m": 31.0, "weather_code": 1,
                        "apparent_temperature": 33.0, "relative_humidity_2m": 40, "wind_speed_10m": 9.0,
                        "precipitation": 0.0},
            "daily": {"weather_code": [1, 2, 3], "temperature_2m_max": [34, 33, 32], "temperature_2m_min": [22, 21, 20],
                      "precipitation_probability_max": [0, 10, 20], "sunrise": ["2026-10-19T06:40"] * 3,
                      "sunset": ["2026-10-19T18:10"] * 3}}


def lookup():
    """The two replies one fresh weather lookup needs."""
    return [FakeResponse(json_data=GEO), FakeResponse(json_data=FORECAST)]


class WeatherCache(unittest.TestCase):
    def setUp(self):
        weather._cache.clear()
        weather._places.clear()
        for name, value in (("WEATHER_CACHE_S", 600), ("WEATHER_PREFETCH", True), ("WEATHER_TIMEOUT", 4.0)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def use(self, *responses):
        session = FakeSession(*responses)
        p = mock.patch.object(weather, "_session", session)
        p.start()
        self.addCleanup(p.stop)
        return session

    def test_a_lookup_is_reused_without_asking_again(self):
        session = self.use(*lookup())
        first = weather.get_weather("Rajkot")
        second = weather.get_weather("rajkot ")
        self.assertEqual(first, second)
        self.assertEqual(len(session.calls), 2)              # place + forecast, once
        self.assertEqual(first["now"]["temp_c"], 31.0)
        self.assertEqual(first["place"], "Rajkot, Gujarat, India")

    def test_it_expires(self):
        session = self.use(*lookup(), FakeResponse(json_data=FORECAST))
        weather.get_weather("Rajkot")
        weather._cache["rajkot"] = (time.time() - 601, weather._cache["rajkot"][1])
        weather.get_weather("Rajkot")
        self.assertEqual(len(session.calls), 3)              # the place is remembered; only the forecast again

    def test_force_always_asks(self):
        session = self.use(*lookup(), FakeResponse(json_data=FORECAST))
        weather.get_weather("Rajkot")
        weather.get_weather("Rajkot", force=True)
        self.assertEqual(len(session.calls), 3)

    def test_cache_off(self):
        with mock.patch.object(config, "WEATHER_CACHE_S", 0):
            session = self.use(*lookup(), FakeResponse(json_data=FORECAST))
            weather.get_weather("Rajkot")
            weather.get_weather("Rajkot")
            self.assertEqual(len(session.calls), 3)
            self.assertFalse(weather.is_cached("Rajkot"))

    def test_callers_get_copies(self):
        self.use(*lookup())
        weather.get_weather("Rajkot")["now"]["temp_c"] = -99
        self.assertEqual(weather.get_weather("Rajkot")["now"]["temp_c"], 31.0)

    def test_is_cached(self):
        self.use(*lookup())
        self.assertFalse(weather.is_cached("Rajkot"))
        weather.get_weather("Rajkot")
        self.assertTrue(weather.is_cached("RAJKOT"))

    def test_a_dead_connection_is_replaced_and_the_request_repeated(self):
        dead = FakeSession(requests.ConnectionError("reset"))
        fresh = FakeSession(*lookup())
        with mock.patch.object(weather, "_session", dead), mock.patch.object(weather, "new_session", lambda: fresh):
            out = weather.get_weather("Rajkot")
        self.assertEqual(out["now"]["temp_c"], 31.0)

    def test_an_unknown_place_is_an_error_not_a_cache_entry(self):
        self.use(FakeResponse(json_data={"results": []}))
        with self.assertRaises(LookupError):
            weather.get_weather("Nowhereville")
        self.assertFalse(weather.is_cached("Nowhereville"))


class Prefetch(unittest.TestCase):
    def setUp(self):
        weather._cache.clear()
        weather._places.clear()
        for name, value in (("WEATHER_CACHE_S", 600), ("WEATHER_PREFETCH", True), ("WEATHER_TIMEOUT", 4.0)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_the_home_city_is_fetched_in_the_background(self):
        session = FakeSession(*lookup(), FakeResponse(json_data=FORECAST), FakeResponse(json_data=FORECAST))
        with mock.patch.object(weather, "_session", session):
            stop = weather.start_prefetch(lambda: "Rajkot", interval=0.05)
            try:
                for _ in range(100):
                    if weather.is_cached("Rajkot"):
                        break
                    time.sleep(0.02)
                self.assertTrue(weather.is_cached("Rajkot"))
            finally:
                stop.set()

    def test_no_city_no_request(self):
        session = FakeSession()
        with mock.patch.object(weather, "_session", session):
            stop = weather.start_prefetch(lambda: "", interval=0.02)
            time.sleep(0.1)
            stop.set()
        self.assertEqual(session.calls, [])

    def test_switched_off(self):
        with mock.patch.object(config, "WEATHER_PREFETCH", False):
            self.assertIsNone(weather.start_prefetch(lambda: "Rajkot"))


if __name__ == "__main__":
    unittest.main()
