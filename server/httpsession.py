"""HTTP sessions that notice a dead connection instead of hanging on it.

A router (or a phone hotspot) quietly forgets a connection that has been idle for a
minute or so. The next request on it then waits 15-20 s for the operating system to
give up, which is how a one-second Gemini call turns into a twenty second one.
TCP keep-alive probes keep the connection known to the router and tell us when it
is gone, so the next request opens a fresh one straight away.
"""

from __future__ import annotations

import socket

import requests
from requests.adapters import HTTPAdapter


def socket_options() -> list[tuple[int, int, int]]:
    """TCP_NODELAY plus keep-alive probes (first one after 30 s idle, then every 10 s).
    Only options this operating system accepts are returned."""
    wanted = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
        if hasattr(socket, name):
            wanted.append((socket.IPPROTO_TCP, getattr(socket, name), value))
    usable = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        for level, option, value in wanted:
            try:
                probe.setsockopt(level, option, value)
                usable.append((level, option, value))
            except OSError:
                pass                              # an older Windows without that option: skip it
    finally:
        probe.close()
    return usable


class _KeepAliveAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs) -> None:
        self._socket_opts = socket_options()     # before super().__init__, which builds the pool
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = self._socket_opts
        super().init_poolmanager(*args, **kwargs)


def new_session(pool_size: int = 8) -> requests.Session:
    """A requests.Session whose connections send TCP keep-alive probes."""
    session = requests.Session()
    adapter = _KeepAliveAdapter(pool_connections=4, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
