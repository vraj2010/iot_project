"""Session cookie: valid / forged / expired; login, logout, and every endpoint needs it."""
import time

from conftest import PASSWORD, SECRET

from core import auth


def test_token_roundtrip_forged_and_expired():
    now = 1_800_000_000
    tok = auth.make_token(SECRET, now=now)
    assert auth.token_expiry(tok, SECRET, now=now + 10) == now + auth.SESSION_TTL_S
    assert auth.token_expiry(tok, SECRET, now=now + auth.SESSION_TTL_S + 1) is None    # expired
    assert auth.token_expiry(tok, "another-secret", now=now) is None                     # wrong key
    v, exp, sig = tok.split(".")
    assert auth.token_expiry(f"{v}.{int(exp) + 999999}.{sig}", SECRET, now=now) is None  # extended expiry
    assert auth.token_expiry(f"{v}.{exp}.{'0' * len(sig)}", SECRET, now=now) is None      # forged signature
    for junk in ("", "abc", "v1.x.y", None, "v2." + exp + "." + sig, f"v1.{exp}.{sig.upper()}",
                 f"v1.{exp}.{sig[:-1]}\u00e9", f"v1.\u0661\u0662.{sig}", f"v1.{exp}.{sig}\n"):
        assert auth.token_expiry(junk, SECRET, now=now) is None


def test_check_password():
    assert auth.check_password("pw", "pw")
    assert not auth.check_password("pw ", "pw")
    assert not auth.check_password("", "")
    assert not auth.check_password(None, "pw")


def test_login_sets_secure_cookie_and_logout(client, monkeypatch):
    from api import index

    monkeypatch.setattr(index, "LOGIN_FAIL_DELAY_S", 0.8)
    r = client.get("/api/session")
    assert r.json()["authenticated"] is False
    t0 = time.monotonic()
    r = client.post("/api/login", json={"password": "wrong"})
    assert r.status_code == 401 and time.monotonic() - t0 >= 0.7       # small delay on failure
    assert auth.COOKIE_NAME not in client.cookies

    r = client.post("/api/login", json={"password": PASSWORD})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth.COOKIE_NAME}=")
    low = cookie.lower()
    assert "httponly" in low and "secure" in low and "samesite=lax" in low and "max-age=43200" in low
    assert auth.COOKIE_NAME == "__Host-cortex_session" and "path=/" in low and "domain" not in low
    assert client.get("/api/session").json()["authenticated"] is True
    assert client.get("/api/documents").status_code == 200

    client.post("/api/logout")
    assert client.get("/api/documents").status_code == 401


def test_endpoints_need_the_cookie(client):
    checks = [("get", "/api/documents", None), ("post", "/api/upload-url", {"filename": "a.pdf", "size": 1}),
              ("post", "/api/ingest", {"document_id": 1}), ("delete", "/api/documents/1", None),
              ("post", "/api/search", {"query": "x"}), ("get", "/api/answers", None), ("get", "/api/file/1", None)]
    for method, path, body in checks:
        kwargs = {"json": body} if body is not None else {}
        assert getattr(client, method)(path, **kwargs).status_code == 401, path
    client.cookies.set(auth.COOKIE_NAME, auth.make_token("forged-secret"))
    assert client.get("/api/documents").status_code == 401
    client.cookies.set(auth.COOKIE_NAME, auth.make_token(SECRET, now=time.time() - auth.SESSION_TTL_S - 5))
    assert client.get("/api/documents").status_code == 401
    client.cookies.set(auth.COOKIE_NAME, auth.make_token(SECRET))
    assert client.get("/api/documents").status_code == 200


def test_missing_config_is_a_clear_500(client, monkeypatch):
    monkeypatch.delenv("SESSION_SECRET")
    r = client.get("/api/session")
    assert r.status_code == 500 and "SESSION_SECRET" in r.json()["detail"]


def test_non_ascii_cookie_is_401_not_500(client):
    # raw UTF-8 bytes in the Cookie header (used to reach hmac.compare_digest and raise TypeError)
    bad = ("v1.9999999999." + "\u00e9" * 64).encode("utf-8")
    r = client.get("/api/documents", headers={"Cookie": auth.COOKIE_NAME.encode() + b"=" + bad})
    assert r.status_code == 401


def test_failed_logins_are_throttled(client, clean_db):
    for _ in range(10):
        assert client.post("/api/login", json={"password": "guess"}).status_code == 401
    r = client.post("/api/login", json={"password": "guess"})
    assert r.status_code == 429 and 1 <= int(r.headers["retry-after"]) <= 900
    r = client.post("/api/login", json={"password": PASSWORD})          # locked for everyone
    assert r.status_code == 429
    import psycopg
    with psycopg.connect(clean_db, autocommit=True) as conn:               # window passes
        conn.execute("update cortex.login_failures set at = at - interval '16 minutes'")
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 200


def test_short_password_refuses_to_serve(client, monkeypatch, caplog):
    monkeypatch.setenv("UPLOAD_PASSWORD", "short-pw")
    with caplog.at_level("ERROR"):
        r = client.post("/api/login", json={"password": "short-pw"})
    assert r.status_code == 500 and "at least 12 characters" in r.json()["detail"]
    assert client.get("/api/session").status_code == 500
    assert any("REFUSING TO SERVE" in rec.getMessage() for rec in caplog.records)
