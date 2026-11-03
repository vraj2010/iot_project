import unittest
from unittest import mock

import _path  # noqa: F401
import requests

import config
import gemini
from fakes import FakeResponse, FakeSession


class Warm(unittest.TestCase):
    def setUp(self):
        for name, value in (("GEMINI_API_KEY", "k"), ("GEMINI_BASE", "http://gemini.test/v1beta")):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_any_answer_from_google_counts(self):
        with mock.patch.object(gemini, "_session", FakeSession(FakeResponse(status=404))):
            self.assertTrue(gemini.warm())

    def test_a_dead_connection_is_replaced(self):
        dead, fresh = FakeSession(requests.ConnectionError("reset")), FakeSession()
        with mock.patch.object(gemini, "_session", dead), mock.patch.object(gemini, "_new_session", lambda: fresh):
            self.assertFalse(gemini.warm())
            self.assertIs(gemini._session, fresh)


if __name__ == "__main__":
    unittest.main()
