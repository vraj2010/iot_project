"""Cortex upload/search API: one FastAPI app = one Vercel function (fewer cold starts).

vercel.json rewrites /api/* here. Routes keep their full /api/... paths; the rewrite also passes
the original path as ?__route= so routing works whichever path the runtime hands the app.
The work lives in core/ (auth, parse, chunk, embed, storage, db, ingest).
"""
import asyncio
import logging
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path, PurePath
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse, RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core import auth, config, ingest as ingest_mod, queries  # noqa: E402
from core.config import ConfigError  # noqa: E402
from core.embed import EmbedError, Gemini  # noqa: E402
from core.parse import CONTENT_TYPES, SUPPORTED_EXTENSIONS, extension  # noqa: E402
from core.storage import StorageError, SupabaseStorage  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cortex.api")

# --- services (created on first use, reused while the instance is warm; tests replace them) ---
_services: dict = {}


def get_storage():
    if "storage" not in _services:
        _services["storage"] = SupabaseStorage(config.supabase_url(), config.supabase_secret_key(),
                                               config.storage_bucket())
    return _services["storage"]


def get_embedder():
    if "embedder" not in _services:
        _services["embedder"] = Gemini(config.gemini_api_key(), config.gemini_embed_model())
    return _services["embedder"]


def set_services(storage=None, embedder=None) -> None:
    _services.clear()
    if storage is not None:
        _services["storage"] = storage
    if embedder is not None:
        _services["embedder"] = embedder


# --- app -----------------------------------------------------------------------------------
class RouteFromQuery:
    """ASGI middleware: if the runtime hands us the rewrite target (/api/index), restore the
    original path from ?__route=, and drop that parameter either way."""

    def __init__(self, app):
        self.inner = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            qs = scope.get("query_string", b"").decode("latin-1")
            if "__route=" in qs:
                pairs = parse_qsl(qs, keep_blank_values=True)
                route = next((v for k, v in pairs if k == "__route"), None)
                rest = [(k, v) for k, v in pairs if k != "__route"]
                scope = dict(scope)
                scope["query_string"] = urlencode(rest).encode("latin-1")
                if route is not None and scope.get("path", "").rstrip("/") in ("/api/index", "/api/index.py", "/api"):
                    path = "/api/" + route.lstrip("/")
                    scope["path"] = path
                    scope["raw_path"] = path.encode()
        await self.inner(scope, receive, send)


app = FastAPI(title="Cortex", docs_url=None, redoc_url=None, openapi_url=None)


def auth_config_problem() -> str | None:
    """The API refuses to run without a session secret and a long enough upload password."""
    try:
        config.session_secret()
        config.upload_password()
    except ConfigError as exc:
        return str(exc)
    return None


_problem = auth_config_problem()
if _problem:
    log.error("REFUSING TO SERVE: %s", _problem)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        problem = auth_config_problem()
        if problem:
            log.error("REFUSING TO SERVE: %s", problem)
            return JSONResponse({"detail": problem}, status_code=500, headers={"Cache-Control": "no-store"})
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(ConfigError)
async def config_error(_: Request, exc: ConfigError):
    log.error("%s", exc)
    return JSONResponse({"detail": str(exc)}, status_code=500)


app.add_middleware(RouteFromQuery)  # added last = outermost: runs before routing


# --- auth ----------------------------------------------------------------------------------
def require_auth(request: Request) -> int:
    exp = auth.token_expiry(request.cookies.get(auth.COOKIE_NAME), config.session_secret())
    if exp is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return exp


class LoginIn(BaseModel):
    password: str = Field(max_length=500)


LOGIN_FAIL_DELAY_S = 0.8


@app.post("/api/login")
async def login(body: LoginIn, response: Response):
    secret = config.session_secret()
    expected = config.upload_password()
    wait = await asyncio.to_thread(queries.login_locked_for)
    if wait:
        raise HTTPException(status_code=429, detail="Too many failed logins. Try again later.",
                            headers={"Retry-After": str(wait)})
    if not auth.check_password(body.password, expected):
        await asyncio.to_thread(queries.record_login_failure)
        await asyncio.sleep(LOGIN_FAIL_DELAY_S)  # slows down guessing
        raise HTTPException(status_code=401, detail="Wrong password")
    token = auth.make_token(secret)
    # __Host- cookie: Secure, Path=/, no Domain
    response.set_cookie(auth.COOKIE_NAME, token, max_age=auth.SESSION_TTL_S, httponly=True,
                        secure=True, samesite="lax", path="/")
    return {"ok": True, "expires_at": auth.token_expiry(token, secret)}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path="/", secure=True, httponly=True, samesite="lax")
    return {"ok": True}


@app.get("/api/session")
def session(request: Request):
    exp = auth.token_expiry(request.cookies.get(auth.COOKIE_NAME), config.session_secret())
    return {"authenticated": exp is not None, "expires_at": exp,
            "max_upload_mb": round(config.max_upload_bytes() / 1024 / 1024, 1),
            "extensions": list(SUPPORTED_EXTENSIONS)}


# --- upload ----------------------------------------------------------------------------------
def display_name(filename: str) -> str:
    """The name people see and Gemini gets as title: last path part, no control characters."""
    name = PurePath(filename.replace("\\", "/")).name
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    return name[:200]


def safe_key_name(filename: str) -> str:
    """ASCII-only object name for the storage key docs/<uuid>/<safe-filename>."""
    ext = extension(filename)
    stem = filename[: -len(ext)] if ext else filename
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:100] or "file"
    return stem + ext


class UploadUrlIn(BaseModel):
    filename: str = Field(min_length=1, max_length=300)
    size: int
    subject: str | None = Field(default=None, max_length=200)


@app.post("/api/upload-url", dependencies=[Depends(require_auth)])
def upload_url(body: UploadUrlIn):
    name = display_name(body.filename)
    ext = extension(name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are accepted")
    limit = config.max_upload_bytes()
    if body.size <= 0:
        raise HTTPException(status_code=400, detail="The file is empty")
    if body.size > limit:
        raise HTTPException(status_code=413, detail=f"The file is larger than {limit // (1024 * 1024)} MB")
    subject = (body.subject or "").strip() or None
    path = f"docs/{uuid.uuid4()}/{safe_key_name(name)}"
    doc = queries.create_document(name, subject, path, body.size)
    try:
        signed = get_storage().create_upload_url(path)
    except Exception as exc:
        queries.delete_document(doc["id"])
        log.exception("signed upload URL failed")
        raise HTTPException(status_code=502, detail=f"Storage refused the upload: {exc}") from exc
    return {"document_id": doc["id"], "upload_url": signed["url"], "token": signed["token"],
            "path": path, "content_type": CONTENT_TYPES[ext], "filename": name}


class IngestIn(BaseModel):
    document_id: int


@app.post("/api/ingest", dependencies=[Depends(require_auth)])
def ingest(body: IngestIn):
    out = ingest_mod.ingest(body.document_id, get_storage(), get_embedder(), budget_s=config.ingest_budget_s())
    if out.get("status") == "missing":
        raise HTTPException(status_code=404, detail="No such document")
    return out


# --- documents -------------------------------------------------------------------------------
@app.get("/api/documents", dependencies=[Depends(require_auth)])
def documents():
    rows = queries.list_documents()
    for r in rows:
        total = r.get("chunk_total")
        r["progress"] = (min(1.0, r["chunk_count"] / total) if total else
                         (1.0 if r["status"] == "ready" else 0.0))
        r.pop("storage_path", None)
    return {"documents": rows}


@app.delete("/api/documents/{document_id}", dependencies=[Depends(require_auth)])
def delete_document(document_id: int):
    gone = queries.delete_document(document_id)
    if gone is None:
        raise HTTPException(status_code=404, detail="No such document")
    ingest_mod.clear_cache_key(gone["storage_path"])
    storage_deleted = True
    try:
        get_storage().delete(gone["storage_path"])
    except StorageError as exc:
        storage_deleted = False
        log.warning("could not delete %s from storage: %s", gone["storage_path"], exc)
    return {"ok": True, "id": document_id, "storage_deleted": storage_deleted}


# --- search ----------------------------------------------------------------------------------
class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: int | None = None                         # exactly this document
    document: str | None = Field(default=None, max_length=300)  # or: filename contains this
    limit: int = Field(default=4, ge=1, le=20)


@app.post("/api/search", dependencies=[Depends(require_auth)])
def search(body: SearchIn):
    t0 = time.perf_counter()
    filename_filter = (body.document or "").strip() or None
    want = body.limit
    if body.document_id is not None:
        filename_filter = queries.ready_filename(body.document_id)
        if filename_filter is None:  # no such ready document
            return {"query": body.query, "passages": [], "timings": {"embed_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0}}
        want = min(50, body.limit * 4)  # other files whose names contain this one are filtered out below
    try:
        vec = get_embedder().embed_query(body.query.strip())
    except EmbedError as exc:
        raise HTTPException(status_code=503 if exc.transient else 502, detail=str(exc)) from exc
    t1 = time.perf_counter()
    rows = queries.search(vec, body.query, want, filename_filter)
    t2 = time.perf_counter()
    if body.document_id is not None:
        rows = [r for r in rows if r["document_id"] == body.document_id]
    rows = rows[:body.limit]
    passages = [{"n": i + 1, **r} for i, r in enumerate(rows)]
    return {"query": body.query, "passages": passages,
            "timings": {"embed_ms": round((t1 - t0) * 1000, 1), "search_ms": round((t2 - t1) * 1000, 1),
                        "total_ms": round((t2 - t0) * 1000, 1)}}


@app.get("/api/answers", dependencies=[Depends(require_auth)])
def answers(limit: int = Query(default=20, ge=1, le=100)):
    return {"answers": queries.recent_answers(limit)}


@app.get("/api/file/{document_id}", dependencies=[Depends(require_auth)])
def file(document_id: int, page: int | None = Query(default=None, ge=1, le=100000)):
    doc = queries.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="No such document")
    try:
        url = get_storage().signed_download_url(doc["storage_path"], expires_in=300)
    except StorageError as exc:
        raise HTTPException(status_code=502, detail=f"Storage: {exc}") from exc
    if page and extension(doc["filename"]) == ".pdf":
        url += f"#page={page}"
    return RedirectResponse(url, status_code=302, headers={"Referrer-Policy": "no-referrer"})


# --- local development only: serve the page from the same app (Vercel serves public/ itself) ---
if __import__("os").environ.get("CORTEX_SERVE_PUBLIC") == "1":
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "public"), html=True))
