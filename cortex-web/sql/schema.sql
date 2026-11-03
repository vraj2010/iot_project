-- Cortex: Friday's document memory. Source of truth for the database side.
-- Applied to Supabase (project i2sonic). Idempotent: safe to run again.
--
-- Everything lives in its own schema `cortex`, which Supabase's auto-generated REST/GraphQL API
-- does not expose. Only server-side code touches it: the Vercel ingest function (transaction
-- pooler) and Friday on the laptop (session pooler), both as the postgres role.

create extension if not exists vector with schema extensions;
create schema if not exists cortex;
revoke all on schema cortex from public;
do $$ begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    execute 'revoke all on schema cortex from anon, authenticated';
  end if;
end $$;

-- One row per uploaded file.
create table if not exists cortex.documents (
  id            bigint generated always as identity primary key,
  filename      text        not null,
  subject       text,
  storage_path  text        not null unique,          -- object key in the private bucket "cortex-docs"
  bytes         bigint,
  sha256        text,                                  -- to spot the same file uploaded twice
  pages         integer,                               -- PDFs only
  chunk_total   integer,                               -- known once parsed; progress = chunk_count / chunk_total
  chunk_count   integer     not null default 0,
  status        text        not null default 'uploaded'
                check (status in ('uploaded', 'processing', 'ready', 'failed')),
  error         text,
  uploaded_at   timestamptz not null default now(),
  ready_at      timestamptz
);
create index if not exists documents_status_idx on cortex.documents (status);
-- Ingest bookkeeping (cortex-web): a lease so two /api/ingest calls never work on one document at
-- once (no advisory locks: they don't survive the transaction pooler), and the chunker version the
-- stored chunks were made with (a half-finished document from another version is redone).
alter table cortex.documents add column if not exists ingest_lease_until timestamptz;
alter table cortex.documents add column if not exists chunker_version integer;

-- Passages. embedding = gemini-embedding-001, 768 dims, task RETRIEVAL_DOCUMENT, L2-normalised.
create table if not exists cortex.chunks (
  id           bigint generated always as identity primary key,
  document_id  bigint  not null references cortex.documents (id) on delete cascade,
  chunk_index  integer not null,
  page         integer,                                -- 1-based; null for DOCX/TXT
  content      text    not null,
  embedding    extensions.vector(768) not null,
  fts          tsvector generated always as (to_tsvector('english', content)) stored,
  -- false while the document is being ingested (or failed); set true for all of a document's
  -- chunks in the same transaction that marks it 'ready'. Only searchable chunks are indexed.
  searchable   boolean not null default false,
  unique (document_id, chunk_index)
);
alter table cortex.chunks add column if not exists searchable boolean not null default false;
-- Partial index: chunks of unfinished documents are not in it, so they can never crowd ready
-- ones out of a nearest-neighbour scan (an HNSW scan that filters rows after finding them can
-- come back EMPTY when many near-identical unfinished chunks sit closest to the question).
drop index if exists cortex.chunks_embedding_hnsw_idx;
create index if not exists chunks_embedding_ready_hnsw_idx on cortex.chunks
  using hnsw (embedding extensions.vector_cosine_ops) where searchable;
create index if not exists chunks_fts_idx on cortex.chunks using gin (fts);
create index if not exists chunks_document_idx on cortex.chunks (document_id);

-- What Friday answered from the documents, with citations: the "Recent answers" panel.
create table if not exists cortex.answers (
  id          bigint generated always as identity primary key,
  question    text        not null,
  answer      text        not null,
  citations   jsonb       not null default '[]'::jsonb,  -- [{n, document_id, filename, page, chunk_id, excerpt, similarity}]
  mode        text,                                       -- inject | tool | prefetch
  latency     jsonb,                                      -- {"embed_ms":..,"search_ms":..,"first_audio_ms":..}
  created_at  timestamptz not null default now()
);
create index if not exists answers_created_idx on cortex.answers (created_at desc);

-- Failed logins on the upload page: cortex-web allows at most 10 per 15 minutes (all clients).
create table if not exists cortex.login_failures (
  id  bigint generated always as identity primary key,
  at  timestamptz not null default now()
);
create index if not exists login_failures_at_idx on cortex.login_failures (at);

alter table cortex.documents enable row level security;
alter table cortex.chunks    enable row level security;
alter table cortex.answers   enable row level security;
alter table cortex.login_failures enable row level security;

-- Hybrid search in one round trip: vector similarity + keyword match (helps with names and
-- misheard words), fused by reciprocal rank. Only documents whose status is 'ready'.
-- similarity = cosine similarity of the question to the chunk (1 = same meaning).
-- Rows below min_similarity are dropped, so "nothing relevant" comes back as no rows.
--
-- Speed: without a filename filter, the nearest chunks come straight from the partial HNSW index
-- (searchable chunks only), with no filter inside the scan; a join, IN (subquery) or array filter
-- there makes the planner fall back to scanning every vector (~100 ms at 20k chunks) or lets
-- unfinished chunks crowd the results out. With a filename filter the search is exact over just
-- those documents' chunks, which are few. The keyword query is computed once so the GIN index is
-- used. The same passage stored twice (a file uploaded twice) is returned once.
drop function if exists cortex.search(extensions.vector, text, integer, text, float8, integer);
create or replace function cortex.search(
  query_embedding extensions.vector(768),
  query_text      text,
  match_count     integer default 4,
  filename_filter text    default null,
  min_similarity  float8  default 0.0,
  rrf_k           integer default 50
) returns table (
  chunk_id bigint, document_id bigint, filename text, subject text, page integer,
  chunk_index integer, content text, similarity float8, keyword_hit boolean, score float8
)
language plpgsql stable
set search_path = cortex, extensions, pg_catalog
set hnsw.iterative_scan = relaxed_order
as $$
declare
  pool    integer := greatest(match_count * 5, 20);
  filt    text    := nullif(btrim(coalesce(filename_filter, '')), '');
  doc_ids bigint[];
  q       tsquery;
begin
  select array_agg(d.id) into doc_ids
  from cortex.documents d
  where d.status = 'ready'
    and (filt is null
         or d.filename ilike '%' || replace(replace(replace(filt, '\', '\\'), '%', '\%'), '_', '\_') || '%');
  if doc_ids is null then
    return;
  end if;

  -- any of the question's words (OR), not all of them
  select to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) into q
  from unnest(to_tsvector('english', coalesce(query_text, '')));

  return query
  with semantic as (
    select s.id, 1 - s.dist as sim, row_number() over (order by s.dist) as rnk
    from (
      select c.id, c.document_id, c.embedding <=> query_embedding as dist
      from cortex.chunks c
      where filt is not null and c.searchable and c.document_id = any (doc_ids)
      union all
      select n.id, n.document_id, n.dist
      from (select c.id, c.document_id, c.embedding <=> query_embedding as dist
            from cortex.chunks c
            where c.searchable
            order by c.embedding <=> query_embedding
            limit pool) n
      where filt is null and n.document_id = any (doc_ids)
    ) s
    order by s.dist
    limit pool
  ),
  keyword as (
    select c.id, row_number() over (order by ts_rank_cd(c.fts, q) desc) as rnk
    from cortex.chunks c
    where q is not null and c.searchable and c.fts @@ q and c.document_id = any (doc_ids)
    order by ts_rank_cd(c.fts, q) desc
    limit pool
  ),
  fused as (
    select coalesce(s.id, k.id) as id, s.sim, (k.id is not null) as kw,
           coalesce(1.0 / (rrf_k + s.rnk), 0) + coalesce(1.0 / (rrf_k + k.rnk), 0) as fscore
    from semantic s full outer join keyword k on k.id = s.id
  )
  select r.id, r.document_id, r.filename, r.subject, r.page, r.chunk_index, r.content, r.sim, r.kw, r.fscore
  from (
    select c.id, c.document_id, d.filename, d.subject, c.page, c.chunk_index, c.content,
           coalesce(f.sim, 1 - (c.embedding <=> query_embedding))::float8 as sim, f.kw, f.fscore::float8 as fscore,
           row_number() over (partition by md5(c.content) order by f.fscore desc, c.id) as copy_no
    from fused f
    join cortex.chunks c on c.id = f.id
    join cortex.documents d on d.id = c.document_id
    where coalesce(f.sim, 1 - (c.embedding <=> query_embedding)) >= min_similarity
  ) r
  where r.copy_no = 1
  order by r.fscore desc
  limit match_count;
end;
$$;
revoke all on function cortex.search(extensions.vector, text, integer, text, float8, integer) from public;
