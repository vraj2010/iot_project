import socket
import unittest

import _path  # noqa: F401

import httpsession


class KeepAlive(unittest.TestCase):
    def test_options_turn_keepalive_on(self):
        options = httpsession.socket_options()
        self.assertIn((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), options)

    def test_every_option_is_accepted_by_the_os(self):
        s = socket.socket()
        try:
            for level, option, value in httpsession.socket_options():
                s.setsockopt(level, option, value)
        finally:
            s.close()

    def test_sessions_carry_the_options_to_their_connections(self):
        adapter = httpsession.new_session().get_adapter("https://example.test")
        self.assertEqual(adapter.poolmanager.connection_pool_kw["socket_options"], httpsession.socket_options())


if __name__ == "__main__":
    unittest.main()
