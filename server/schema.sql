-- NEXA memory schema. server.py runs this automatically at start-up;
-- it is safe to run again (everything is "if not exists").
-- Plain PostgreSQL: embeddings are stored as real[] and compared in Python,
-- so the pgvector extension is not needed.

create or replace function nexa_touch() returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Always in the prompt: who the user is. Edit freely, e.g. {"name": "Deep", "city": "Bengaluru"}.
create table if not exists profile (
  user_key   text primary key,
  content    jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- Long-term facts. Add rows by hand too: leave embedding empty and the server fills it in.
create table if not exists facts (
  id           bigserial primary key,
  user_key     text not null default 'household',
  content      text not null,
  category     text,
  source       text not null default 'auto' check (source in ('auto', 'explicit', 'manual')),
  embedding    real[],
  is_active    boolean not null default true,
  last_used_at timestamptz,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);
create index if not exists facts_user_active_idx on facts (user_key, is_active);

-- One row per conversation, with a short summary once it ends.
create table if not exists episodes (
  id         bigserial primary key,
  user_key   text not null default 'household',
  started_at timestamptz not null default now(),
  ended_at   timestamptz,
  summary    text,
  embedding  real[]
);
create index if not exists episodes_user_started_idx on episodes (user_key, started_at desc);

-- Every exchange, word for word (the rolling context and your log).
create table if not exists turns (
  id         bigserial primary key,
  episode_id bigint not null references episodes (id) on delete cascade,
  role       text not null check (role in ('user', 'assistant')),
  content    text not null,
  tools      jsonb,
  ms         integer,
  created_at timestamptz not null default now()
);
create index if not exists turns_episode_idx on turns (episode_id, created_at);

-- "Store anything": notes, lists, recipes... any shape of data in `data`.
create table if not exists items (
  id         bigserial primary key,
  user_key   text not null default 'household',
  kind       text not null,
  title      text,
  data       jsonb not null default '{}'::jsonb,
  tags       text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists items_kind_idx on items (user_key, kind);

-- Music library: music.py indexes it, jukebox.py plays it.
create table if not exists music_tracks (
  id         bigserial primary key,
  title      text not null,
  artist     text,
  album      text,
  genre      text,
  tags       text[] not null default '{}',
  path       text,          -- file on this laptop
  stream_url text,          -- or an internet stream
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Columns added for the music library (music.py). Safe to run again.
alter table music_tracks add column if not exists source_type     text not null default 'file';  -- file | radio
alter table music_tracks add column if not exists file_name       text;
alter table music_tracks add column if not exists file_size       bigint;
alter table music_tracks add column if not exists file_mtime      double precision;
alter table music_tracks add column if not exists year            integer;
alter table music_tracks add column if not exists duration_s      integer;
alter table music_tracks add column if not exists cover_url       text;
alter table music_tracks add column if not exists itunes_track_id bigint;
alter table music_tracks add column if not exists itunes_url      text;
alter table music_tracks add column if not exists explicit        boolean;
alter table music_tracks add column if not exists match_source    text;      -- itunes | tags | filename | manual
alter table music_tracks add column if not exists match_score     real;
alter table music_tracks add column if not exists needs_review    boolean not null default false;
alter table music_tracks add column if not exists play_count      integer not null default 0;
alter table music_tracks add column if not exists last_played_at  timestamptz;
create unique index if not exists music_tracks_path_idx on music_tracks (path);
-- Where the file lives (local = MUSIC_DIR today; a synced store later) and a
-- fingerprint of its bytes, so a renamed or moved song keeps its metadata.
alter table music_tracks add column if not exists storage         text not null default 'local';
alter table music_tracks add column if not exists content_hash    text;
create index if not exists music_tracks_hash_idx on music_tracks (content_hash);

-- Settings you can change without restarting, e.g.
--   insert into settings values ('extra_instructions', '"Call me Deep."');
create table if not exists settings (
  key         text primary key,
  value       jsonb not null,
  description text,
  updated_at  timestamptz not null default now()
);

drop trigger if exists profile_touch on profile;
create trigger profile_touch before update on profile for each row execute function nexa_touch();
drop trigger if exists facts_touch on facts;
create trigger facts_touch before update on facts for each row execute function nexa_touch();
drop trigger if exists items_touch on items;
create trigger items_touch before update on items for each row execute function nexa_touch();
drop trigger if exists music_touch on music_tracks;
create trigger music_touch before update on music_tracks for each row execute function nexa_touch();
drop trigger if exists settings_touch on settings;
create trigger settings_touch before update on settings for each row execute function nexa_touch();

insert into settings (key, value, description) values
  ('extra_instructions', '""', 'Extra instructions added to Nexa''s system prompt')
on conflict (key) do nothing;
