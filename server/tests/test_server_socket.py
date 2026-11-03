import http.client
import socket
import sys
import threading
import unittest

import _path  # noqa: F401
import server


class Probe(server.Handler):
    """The real handler, with a route that reports how its socket is set up."""

    seen: list = []

    def do_GET(self):
        Probe.seen.append(self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
        self._json({"ok": True})


class SpeakerSockets(unittest.TestCase):
    def test_small_writes_to_the_speaker_go_out_at_once(self):
        """With Nagle's algorithm a small write waits until the previous one has been acknowledged, and the
        speaker acknowledges late (up to a quarter of a second): that wait could land on every chunk of a
        streamed reply. Python's http.server leaves Nagle on unless told otherwise."""
        Probe.seen.clear()
        httpd = server.Server(("127.0.0.1", 0), Probe)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/probe")
        self.assertEqual(conn.getresponse().status, 200)
        self.assertEqual(len(Probe.seen), 1)
        self.assertTrue(Probe.seen[0], "TCP_NODELAY is off on the connection to the speaker")


class OneFridayAtATime(unittest.TestCase):
    """On Windows, SO_REUSEADDR let a second server.py bind port 8765 as well, silently: the speaker then
    kept talking to an old copy of Friday (one started before an update) while the new one sat idle."""

    def test_a_second_server_on_the_same_port_fails_loudly(self):
        first = server.Server(("127.0.0.1", 0), server.Handler)
        self.addCleanup(first.server_close)
        with self.assertRaises(OSError):
            second = server.Server(("127.0.0.1", first.server_address[1]), server.Handler)
            second.server_close()

    def test_windows_asks_for_the_port_exclusively(self):
        from unittest import mock

        with mock.patch.object(server, "_exclusive") as exclusive:
            httpd = server.Server(("127.0.0.1", 0), server.Handler)
            httpd.server_close()
        exclusive.assert_called_once()
        with mock.patch.object(server.sys, "platform", "win32"), \
                mock.patch.object(server.socket, "SO_EXCLUSIVEADDRUSE", 0x7FFF, create=True):
            sock = mock.Mock()
            server._exclusive(sock)
            sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, 0x7FFF, 1)
        self.assertFalse(server.Server.allow_reuse_address if sys.platform == "win32" else False)


if __name__ == "__main__":
    unittest.main()
