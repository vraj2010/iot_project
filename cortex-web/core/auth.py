"""Password login and the signed session cookie (HMAC-SHA256 over the expiry time)."""
import hashlib
import hmac
import re
import time

# __Host- prefix: browsers only accept it with Secure, Path=/ and no Domain (no subdomain can set it).
COOKIE_NAME = "__Host-cortex_session"
_TOKEN = re.compile(r"v1\.([0-9]{1,12})\.([0-9a-f]{64})")
SESSION_TTL_S = 12 * 3600
_VERSION = "v1"


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(secret: str, now: float | None = None, ttl_s: int = SESSION_TTL_S) -> str:
    exp = int((time.time() if now is None else now) + ttl_s)
    payload = f"{_VERSION}.{exp}"
    return f"{payload}.{_sign(secret, payload)}"


def token_expiry(token: str | None, secret: str, now: float | None = None) -> int | None:
    """The token's expiry (unix seconds) if it is genuine and not expired, else None."""
    if not token or not secret or not isinstance(token, str):
        return None
    m = _TOKEN.fullmatch(token)  # also rejects non-ASCII before compare_digest sees it
    if m is None:
        return None
    payload = f"{_VERSION}.{m.group(1)}"
    if not hmac.compare_digest(_sign(secret, payload), m.group(2)):
        return None
    exp = int(m.group(1))
    if exp <= (time.time() if now is None else now):
        return None
    return exp


def check_password(given: str, expected: str) -> bool:
    """Constant-time comparison (hashing first makes both sides the same length)."""
    if not expected:
        return False
    a = hashlib.sha256((given or "").encode()).digest()
    b = hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(a, b)
