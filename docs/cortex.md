# Cortex: Friday's document memory (contract)

Both halves are built against this file. Change it first if a decision changes.

```
UPLOAD (hosted, Vercel bom1 + Supabase ap-south-1)            ASK (laptop, server/)
browser ─ /api/upload-url ─▶ signed URL                        question ─▶ Brain._begin
   └─ PUT file ─▶ Storage bucket cortex-docs (private)           ├─ gate says "about my docs"? ─▶ retrieve now, inject passages → ONE Gemini round
   └─ /api/ingest {id} (repeat until ready)                      ├─ otherwise: prefetch retrieval in the background, offer search_documents tool
        parse → chunk → Gemini embed → cortex.chunks             └─ reply with [n] markers → markers stripped before speech,
                                                                      citations returned + logged to cortex.answers (background)
```

## Database (Supabase project `i2sonic`, schema `cortex`)
Source of truth: `cortex-web/sql/schema.sql` (+ `storage.sql` for the bucket). Tables `cortex.documents`,
`cortex.chunks`, `cortex.answers`; function `cortex.search(query_embedding, query_text, match_count,
filename_filter, min_similarity, rrf_k)` = hybrid vector + keyword search, one round trip, ~5-20 ms in the DB.
Not exposed through Supabase's REST API; only server code connects, as `postgres`:
- Vercel: `DATABASE_URL` = transaction pooler (port 6543) → psycopg with `prepare_threshold=None`.
- Friday: `CORTEX_DATABASE_URL` = session pooler (port 5432) → one long-lived psycopg connection pool.

Document status: `uploaded` → `processing` → `ready` | `failed` (reason in `error`). Only `ready` is searched:
chunks are inserted with `searchable = false` and flipped to `true` in the same transaction that marks the
document `ready` (a partial HNSW index covers searchable chunks only, so unfinished chunks can never crowd
ready ones out of a scan). The same passage stored twice is returned once.
Storage object key: `docs/<uuid>/<safe-filename>`, bucket `cortex-docs` (private, 50 MB max, pdf/docx/txt).

## Embeddings (must match on both sides, or search silently returns garbage)
- Model `gemini-embedding-001`, `outputDimensionality: 768`, then **L2-normalise** (shortened vectors aren't unit length).
- Documents: `taskType: RETRIEVAL_DOCUMENT`, `title: <filename>`; batched with `:batchEmbedContents` (≤100 per call).
- Questions: `taskType: RETRIEVAL_QUERY`; single `:embedContent`.
- Store as `extensions.vector(768)`; cosine distance (`<=>`), similarity = 1 - distance.

## Chunking (ingest)
- Text per page (PDF via PyMuPDF; DOCX paragraphs + tables and TXT have no pages → `page = null`).
- A chunk never crosses a page, so every citation has an exact page.
- Pack whole sentences/paragraphs up to ~1,200 characters (≈300 tokens); overlap the last sentence (≤200 chars)
  into the next chunk; a single over-long sentence is split hard at 1,200. Drop chunks under 40 characters
  unless the page has nothing else. Small chunks = precise citations and a small prompt (latency).
- `chunk_index` is 0..n-1 in reading order, deterministic for the same file (ingest is resumable).

## Ingest is resumable (Vercel Hobby: 300 s per call, Gemini free-tier rate limits)
`POST /api/ingest {document_id}` parses, chunks, skips chunk indexes already stored, embeds and inserts the
rest until ~240 s have passed, then returns `{status: "processing", done, total}`; the page calls again until
`status` is `ready`. On a Gemini 429 it backs off and returns `processing` (the page retries); any other
error marks the document `failed` with the reason.

## Retrieval on the laptop (latency budget is the point)
Target: a document question costs at most **~0.4 s more than an ordinary question** to first audio.
- Query embedding ~150-350 ms (Gemini) — cached (LRU) and fetched on a kept-alive connection.
- `cortex.search` ~20-60 ms incl. the Bengaluru→Mumbai round trip — pooled connection, warmed at start, pinged when idle.
- **Inject path (one Gemini round):** a cheap local gate (phrases like "my notes/document/file/pdf/manual",
  "according to", "in the …", or a word from a document's name or subject) → retrieve before the Gemini call and
  put the passages in the prompt. Saves the whole second Gemini round (~0.8-1.5 s) on obvious document questions.
- **Tool path:** otherwise retrieval for the raw question starts in the background at turn start (prefetch); if
  Gemini calls `search_documents`, the prefetched result is returned at once when ready (else it searches).
- Document list (names, subjects) for the prompt: cached in RAM, refreshed in the background, never awaited.
- Passages handed to Gemini: top 4, each ≤1,200 chars, numbered `[1]..[4]`.
- If the DB or Gemini embedding fails or takes > `CORTEX_TIMEOUT` (default 2.5 s), the turn carries on without
  documents and Friday says she couldn't reach them — a document problem must never break a normal answer.
  A call that times out must not keep holding a connection (cancel/close it), so one dead network path can't
  make the next turns wait too.
- The prefetched result (raw question) is reused for the tool call only when Gemini's query is about the same
  thing (word overlap); otherwise Gemini's wording is searched (follow-ups like "tell me more about that one").

## Citations
- Gemini is told to put `[n]` after claims taken from passage n and to say the source once in words
  ("your biology notes, page 3, say …"). `[n]` markers are stripped before speech (spoken and streamed paths).
- The turn result gets `citations: [{n, document_id, filename, page, chunk_id, excerpt, similarity}]`:
  - inject path: only passages actually marked; no markers = the answer didn't use the documents (gate false
    positive) → no citations, not logged as a document answer;
  - tool path (Gemini chose to search): the marked passages, or all returned passages if none were marked.
- Where citations show: `/chat` JSON (`citations`), the `X-Citations` header on non-streamed `/say`/`/ask`
  replies, the console, and `cortex.answers` → the hosted page's "Recent answers" (with links that open the PDF
  at the cited page). Streamed replies (the default) send their headers before the answer exists, so for them
  the hosted page and the console are the citation display; the speaker says the source in words.
- A document answer (≥1 citation, no error) is written to `cortex.answers` on its own background worker
  (never delays the reply, never blocks memory learning); writes are not retried after an ambiguous failure.

## Safety
- Passages are reference text, never instructions; the prompt says so.
- The browser never sees a Supabase key: uploads use one-time signed upload URLs made by `/api/upload-url`.
- The hosted page is behind `UPLOAD_PASSWORD` (HttpOnly signed cookie).
