# cortex-web: Friday's document upload page

The hosted half of **Cortex**, Friday's document memory (contract: [`docs/cortex.md`](../docs/cortex.md)).
A password-protected page where you drop PDFs, Word files and text files. Each file is parsed,
cut into small page-exact passages, embedded with Gemini and stored in Supabase (pgvector). Friday on
the laptop searches those passages and cites them; her answers show up here under "Recent answers",
with links that open the PDF at the cited page.

```
browser ── /api/upload-url ──▶ signed upload URL ──▶ PUT file straight to Supabase Storage (private bucket)
        └─ /api/ingest {id} (repeated until ready): download → parse → chunk → Gemini embed → cortex.chunks
```

- **One Vercel function** (`api/index.py`, FastAPI) behind `/api/*`, region `bom1` (Mumbai, next to the
  Supabase database), `maxDuration` 300 s. The page is two static files, `public/index.html` + `public/app.js`.
- The file itself never passes through the function (Vercel caps request bodies at 4.5 MB): the browser
  uploads straight to Storage with a one-time signed URL. The browser never sees a Supabase key.
- Ingest is **resumable**: each `/api/ingest` call works for at most ~240 s, then returns
  `{status: "processing", done, total}` and the page calls again. A Gemini 429 backs off and also returns
  `processing` (with `retry_after`). Chunks already stored are skipped, so nothing is embedded twice.
- A document is **never searched half-done**: chunks are stored with `searchable = false` and all of them
  become searchable in the same transaction that marks the document `ready`.
- One call at a time per document: each call takes a lease (`documents.ingest_lease_until`, 290 s); a
  second tab or retry gets `{busy: true, retry_after: 10}` and does nothing.
- Gemini 429: the call backs off briefly and halves its batch (min 10); after 3 in a row it hands back
  with `retry_after`. A used-up **daily** quota returns `paused: "quota"` and the page stops ("Paused:
  Gemini quota used up, try later"). Batches are also capped at 60k characters (non-Latin text).
- Storage/Gemini 5xx, timeouts and a dropped DB connection return `processing` + `retry_after`; only
  real problems (bad file, no text, Gemini refusing the request) mark the document `failed`.
- `chunker_version` is stored per document; a half-finished document from another chunker version is
  wiped and redone.

## Layout

| Path | What |
|---|---|
| `api/index.py` | The FastAPI app (thin): routes, auth check, request validation |
| `core/auth.py` | Password check, HMAC-signed session cookie |
| `core/parse.py` | PDF (PyMuPDF), DOCX (paragraphs + tables), TXT → text per page |
| `core/chunk.py` | Page-exact chunking (≤1,200 chars, sentence overlap, tiny-chunk rule) |
| `core/embed.py` | Gemini `gemini-embedding-001` over REST, 768 dims, L2-normalised |
| `core/storage.py` | Supabase Storage REST: signed upload/download URLs, download, delete |
| `core/db.py`, `core/queries.py` | One reused psycopg connection (transaction pooler), the SQL |
| `core/ingest.py` | The resumable ingest loop |
| `public/index.html`, `public/app.js` | The page (vanilla JS/CSS, no build step, no inline script) |
| `sql/schema.sql`, `sql/storage.sql` | Database schema and the storage bucket (applied to Supabase separately) |
| `tests/` | pytest suite (real Postgres + pgvector, fake Gemini and Storage) |

## API

All routes except `login`, `logout` and `session` need the session cookie.

| Method & path | Body / query | Returns |
|---|---|---|
| `POST /api/login` | `{password}` | sets `__Host-cortex_session` (HttpOnly, Secure, SameSite=Lax, Path=/, 12 h). At most 10 failed logins per 15 min (all clients, `cortex.login_failures`), then 429 + `Retry-After` |
| `POST /api/logout` | | clears the cookie |
| `GET /api/session` | | `{authenticated, expires_at, max_upload_mb, extensions}` |
| `POST /api/upload-url` | `{filename, size, subject?}` | `{document_id, upload_url, token, path, content_type, filename}` |
| `POST /api/ingest` | `{document_id}` | `{status, done, total, duplicate_of, retry_after?, busy?, paused?, note?, error?, elapsed_ms}` |
| `GET /api/documents` | | `{documents: [...]}` newest first, with `progress` 0..1 |
| `DELETE /api/documents/{id}` | | deletes the row (chunks cascade) and the stored file |
| `POST /api/search` | `{query, document_id?, document?, limit?}` (`document_id` = exactly that document; `document` = filename contains) | `{passages: [{n, filename, page, similarity, content, ...}], timings: {embed_ms, search_ms, total_ms}}` |
| `GET /api/answers?limit=20` | | Friday's recent document answers (`cortex.answers`) |
| `GET /api/file/{id}?page=N` | | 302 to a 5-minute signed download URL (`#page=N` for PDFs) |

## Environment variables

| Name | Required | What |
|---|---|---|
| `DATABASE_URL` | yes | Supabase **transaction pooler** URL (port 6543), user `postgres.<project-ref>`. SSL is forced (`sslmode=require`) unless the URL sets `sslmode`. |
| `SUPABASE_URL` | yes | `https://<project-ref>.supabase.co` |
| `SUPABASE_SECRET_KEY` | yes | A secret API key (`sb_secret_...`; a legacy `service_role` JWT also works). Server side only. |
| `GEMINI_API_KEY` | yes | Google AI Studio key, used for embeddings |
| `UPLOAD_PASSWORD` | yes | The page's password, **at least 12 characters** (otherwise every `/api` call returns 500 and the log says why) |
| `SESSION_SECRET` | yes | Long random string that signs the cookie (`python -c "import secrets; print(secrets.token_urlsafe(48))"`). Changing it signs everyone out. |
| `GEMINI_EMBED_MODEL` | no | default `gemini-embedding-001` (must match Friday's) |
| `STORAGE_BUCKET` | no | default `cortex-docs` |
| `MAX_UPLOAD_MB` | no | default `50` (the bucket's own limit is also 50 MB) |
| `INGEST_BUDGET_S` | no | work time per ingest call, default `240` (Hobby functions stop at 300 s) |

## Deploy (Vercel)

1. Database: apply `sql/schema.sql` and then `sql/storage.sql` to the Supabase project (SQL editor).
2. Vercel → **Add New Project** → import this repository.
   - **Root Directory**: `cortex-web`
   - **Framework Preset**: **Other** (`vercel.json` also sets `"framework": null`). Leave the build and
     output settings empty: `public/` is served as static files, `api/index.py` becomes the function.
3. Add the environment variables above (Production, and Preview if you use it).
4. Deploy. Python 3.12 (`.python-version`); dependencies from `requirements.txt`.
5. Open the site, sign in, drop a PDF, wait for "Ready", then try a search.

`vercel.json` pins the function to `bom1`, gives it `maxDuration: 300`, rewrites `/api/*` to the single
function and adds security headers. The Content-Security-Policy allows browser uploads only to
`https://*.supabase.co`; if the Supabase project uses a custom domain, add it to `connect-src`.

## Run locally

```bash
cd cortex-web
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                               # fill it in (never commit .env)
vercel dev                                         # serves public/ and /api like production
```

Without the Vercel CLI: `CORTEX_SERVE_PUBLIC=1 uvicorn api.index:app --env-file .env` serves the page and the
API together on http://localhost:8000 (that flag is for local use only; on Vercel `public/` is served by the
CDN). The cookie is `Secure`; browsers accept that on `http://localhost`.

## Tests

```bash
pip install -r requirements-dev.txt
# a local PostgreSQL with pgvector; the tests rebuild schema "cortex" in this database from sql/schema.sql
export CORTEX_TEST_DATABASE_URL="host=localhost port=5432 user=postgres password=... dbname=cortex_web_test sslmode=disable"
python -m pytest -q tests
```

Gemini and Supabase Storage are faked (no network). Database tests are skipped when the test database
cannot be reached; they refuse to run unless the database is local and its name contains `test`
(the fixture drops schema `cortex`). Covered: chunking rules, searchability until ready, crowd-out by
unfinished chunks, duplicate passages, the ingest lease, 429 halving and daily-quota pause, login throttling, the Gemini request shape, normalisation and 429 handling,
the Storage REST calls, the session cookie (valid/forged/expired), upload validation, resumable ingest
(one call vs. several calls under a time budget give identical rows), failure marking and retry,
duplicate detection, search with citations, delete cascade, answers, file links, and connection reuse.
