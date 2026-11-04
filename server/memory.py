"""NEXA's memory, stored in PostgreSQL on this laptop.

  profile    always in the prompt (name, city, likes...)
  facts      long-term memories, found by meaning (Gemini embeddings, compared here)
  episodes   one row per conversation, summarised when it ends
  turns      every exchange word for word; the recent ones are the rolling context

Writes that need Gemini (embeddings, learning facts, summaries) never block a
reply: they run after the answer has been spoken, on a background thread.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import gemini

HERE = Path(__file__).resolve().parent

_WORD = re.compile(r"[a-z0-9']+")
_STOP = set("""a an the is are was were be been am i me my mine you your yours it its this that
these those to of in on at for with and or but not no do does did have has had what when where
who whom which how why can could would should will shall may might about tell said say please
nexa Nexa hey ok okay so just than then there here from by as into""".split())


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _keyword_score(query: str, text: str) -> float:
    q = _words(query)
    if not q:
        return 0.0
    return len(q & _words(text)) / len(q)


class Memory:
    def __init__(self) -> None:
        import psycopg  # imported here so --help works without it

        self._psycopg = psycopg
        self._lock = threading.RLock()
        self._conn = None
        self._cache_key: tuple | None = None
        self._facts: list[dict] = []            # active facts for USER_KEY
        self._matrix: np.ndarray | None = None  # one unit-length row per fact that has an embedding
        self._matrix_ids: list[int] = []
        self._embed_failed: set[int] = set()    # don't retry a broken embedding on every turn
        self._connect()
        self._ensure_schema()
        print(f"[memory] PostgreSQL ready ({self._safe_url()}), {self.count_facts()} facts")

    # ------------------------------------------------------------ connection

    def _safe_url(self) -> str:
        return re.sub(r"://([^:@/]+):[^@/]*@", r"://\1:***@", config.DATABASE_URL)

    def _connect(self) -> None:
        psycopg = self._psycopg
        try:
            self._conn = psycopg.connect(config.DATABASE_URL, autocommit=True, connect_timeout=5)
            return
        except psycopg.OperationalError as exc:
            if "does not exist" not in str(exc):
                raise RuntimeError(
                    f"can't connect to PostgreSQL at {self._safe_url()}: {exc}\n"
                    "  Is the Docker database running (docker compose ps in the db folder), and is the password in DATABASE_URL right?") from None
        # The server is up but the database isn't there yet: create it.
        from psycopg import conninfo, sql

        info = conninfo.conninfo_to_dict(config.DATABASE_URL)
        dbname = info.get("dbname") or "nexa"
        info["dbname"] = "postgres"
        with psycopg.connect(conninfo.make_conninfo(**info), autocommit=True, connect_timeout=5) as admin:
            admin.execute(sql.SQL("create database {}").format(sql.Identifier(dbname)))
        print(f"[memory] created database '{dbname}'")
        self._conn = psycopg.connect(config.DATABASE_URL, autocommit=True, connect_timeout=5)

    def _sql(self, query: str, params: tuple | dict | None = None, fetch: str = "") -> list | tuple | None:
        """Run one statement; reconnect once if the connection dropped."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    cur = self._conn.execute(query, params)
                    if fetch == "one":
                        return cur.fetchone()
                    if fetch == "all":
                        return cur.fetchall()
                    return None
                except self._psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    self._connect()
        return None

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute((HERE / "schema.sql").read_text(encoding="utf-8"))
            self._sql("insert into profile (user_key) values (%s) on conflict do nothing", (config.USER_KEY,))

    # ------------------------------------------------------------ profile + settings

    def profile(self) -> dict:
        row = self._sql("select content from profile where user_key = %s", (config.USER_KEY,), "one")
        return row[0] if row and isinstance(row[0], dict) else {}

    def set_profile(self, key: str, value: str) -> None:
        key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")[:40] or "note"
        self._sql(
            "update profile set content = content || jsonb_build_object(%s::text, %s::text) where user_key = %s",
            (key, value.strip()[:300], config.USER_KEY))

    def setting(self, key: str, default=None):
        row = self._sql("select value from settings where key = %s", (key,), "one")
        return row[0] if row else default

    # ------------------------------------------------------------ facts cache

    def count_facts(self) -> int:
        row = self._sql("select count(*) from facts where user_key = %s and is_active", (config.USER_KEY,), "one")
        return int(row[0]) if row else 0

    def _refresh(self) -> None:
        """Reload facts when the table changed (including edits made by hand in pgAdmin)."""
        key = self._sql(
            "select count(*), max(updated_at), count(*) filter (where embedding is null) "
            "from facts where user_key = %s and is_active", (config.USER_KEY,), "one")
        if key == self._cache_key and self._matrix is not None:
            return
        rows = self._sql(
            "select id, content, category, embedding from facts "
            "where user_key = %s and is_active order by id", (config.USER_KEY,), "all") or []

        missing = [r for r in rows if r[3] is None and r[0] not in self._embed_failed]
        for fid, content, _cat, _emb in missing[:20]:          # fill embeddings for hand-added rows
            try:
                vec = gemini.embed(content)
                self._sql("update facts set embedding = %s where id = %s", (vec, fid))
            except gemini.GeminiError as exc:
                self._embed_failed.add(fid)
                print(f"[memory] couldn't embed fact {fid}: {exc}")
        if missing:
            rows = self._sql(
                "select id, content, category, embedding from facts "
                "where user_key = %s and is_active order by id", (config.USER_KEY,), "all") or []

        self._facts = [{"id": r[0], "content": r[1], "category": r[2]} for r in rows]
        with_vec = [r for r in rows if r[3] and len(r[3]) == config.EMBED_DIMS]
        self._matrix_ids = [r[0] for r in with_vec]
        if with_vec:
            m = np.array([r[3] for r in with_vec], dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self._matrix = m / np.maximum(norms, 1e-9)
        else:
            self._matrix = np.zeros((0, config.EMBED_DIMS), dtype=np.float32)
        self._cache_key = self._sql(
            "select count(*), max(updated_at), count(*) filter (where embedding is null) "
            "from facts where user_key = %s and is_active", (config.USER_KEY,), "one")

    def _search_facts(self, query: str, k: int, min_sim: float,
                      query_vec: list[float] | None = None) -> list[tuple[dict, float]]:
        """Best matching active facts as (fact, score). Meaning first, keywords as a backup."""
        with self._lock:
            self._refresh()
            facts_by_id = {f["id"]: f for f in self._facts}
            scored: dict[int, float] = {}
            if query_vec is None and len(self._matrix_ids):
                try:
                    query_vec = gemini.embed(query)
                except gemini.GeminiError as exc:
                    print(f"[memory] embedding failed, using keywords: {exc}")
            if query_vec is not None and len(self._matrix_ids):
                sims = self._matrix @ np.asarray(query_vec, dtype=np.float32)
                for fid, s in zip(self._matrix_ids, sims.tolist()):
                    if s >= min_sim:
                        scored[fid] = s
            for f in self._facts:                       # keywords catch names and rare words
                kw = _keyword_score(query, f["content"])
                if kw >= 0.5:
                    scored[f["id"]] = max(scored.get(f["id"], 0.0), min_sim + 0.1 * kw)
            best = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]
            return [(facts_by_id[fid], s) for fid, s in best if fid in facts_by_id]

    def relevant_facts(self, text: str, k: int | None = None) -> list[str]:
        if not text:
            return []
        with self._lock:
            self._refresh()
            if len(self._facts) <= config.MEMORY_ALL_FACTS_MAX:   # few memories: all of them, no search
                return [f["content"] for f in self._facts]
        hits = self._search_facts(text, k or config.MEMORY_TOP_K, config.MEMORY_MIN_SIMILARITY)
        return [f["content"] for f, _s in hits]

    # ------------------------------------------------------------ facts: tools

    def save_fact(self, content: str, category: str | None = None, source: str = "explicit") -> str:
        """Add a fact, or replace a near-duplicate. Returns what happened, for the model."""
        content = re.sub(r"\s+", " ", content or "").strip()[:400]
        if not content:
            return "nothing to save"
        if re.search(r"\b(password|passcode|pin|otp|cvv|card number|account number)\b", content, re.I):
            return "refused: secrets like passwords and card numbers are never stored"
        try:
            vec = gemini.embed(content)
        except gemini.GeminiError:
            vec = None
        with self._lock:
            dup = self._search_facts(content, 1, 0.92, query_vec=vec) if vec else []
            if dup:
                fid = dup[0][0]["id"]
                self._sql("update facts set content = %s, embedding = %s, source = %s where id = %s",
                          (content, vec, source, fid))
                return f"updated memory {fid}: {content}"
            row = self._sql(
                "insert into facts (user_key, content, category, source, embedding) "
                "values (%s, %s, %s, %s, %s) returning id",
                (config.USER_KEY, content, category, source, vec), "one")
            return f"saved memory {row[0]}: {content}"

    def recall(self, query: str) -> dict:
        facts = self._search_facts(query, 8, config.MEMORY_MIN_SIMILARITY - 0.1)
        episodes = self._search_episodes(query, 3)
        return {
            "memories": [f["content"] for f, _s in facts],
            "past_conversations": episodes,
            "note": "nothing stored about that" if not facts and not episodes else "",
        }

    def forget(self, query: str) -> str:
        hits = self._search_facts(query, 3, 0.62)
        if not hits:
            return "no matching memory found"
        fact, _score = hits[0]
        self._sql("update facts set is_active = false where id = %s", (fact["id"],))
        return f"forgot: {fact['content']}"

    def apply_ops(self, ops: list, allowed_ids: set[int]) -> list[str]:
        """Changes proposed by the background learner. Only touches facts it was shown."""
        done = []
        for op in ops[:5]:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).lower()
            text = str(op.get("fact", "")).strip()
            fid = op.get("id")
            if kind == "add" and text:
                done.append(self.save_fact(text, source="auto"))
            elif kind == "update" and text and isinstance(fid, int) and fid in allowed_ids:
                try:
                    vec = gemini.embed(text)
                except gemini.GeminiError:
                    vec = None
                self._sql("update facts set content = %s, embedding = %s where id = %s", (text[:400], vec, fid))
                done.append(f"updated {fid}: {text}")
            elif kind == "delete" and isinstance(fid, int) and fid in allowed_ids:
                self._sql("update facts set is_active = false where id = %s", (fid,))
                done.append(f"deactivated {fid}")
        return done

    # ------------------------------------------------------------ conversations

    def begin_turn(self) -> dict:
        """Find (or start) the current conversation. Returns its id, the recent
        exchanges, and a summary of the previous conversation for context."""
        row = self._sql(
            "select e.id, coalesce(max(t.created_at), e.started_at) "
            "from episodes e left join turns t on t.episode_id = e.id "
            "where e.user_key = %s group by e.id order by e.id desc limit 1",
            (config.USER_KEY,), "one")
        gap_s = config.SESSION_GAP_MIN * 60
        if row and (datetime.now(row[1].tzinfo) - row[1]).total_seconds() < gap_s:
            episode_id, new_session, previous_id = row[0], False, None
        else:
            previous_id = row[0] if row else None
            new = self._sql("insert into episodes (user_key) values (%s) returning id", (config.USER_KEY,), "one")
            episode_id, new_session = new[0], True
            if previous_id:
                self._sql("update episodes set ended_at = now() where id = %s and ended_at is null", (previous_id,))

        rows = self._sql(
            "select role, content from turns where episode_id = %s order by id desc limit %s",
            (episode_id, config.HISTORY_TURNS * 2), "all") or []
        history = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

        last = self._sql(
            "select summary, started_at from episodes where user_key = %s and summary is not null "
            "and id <> %s order by id desc limit 1", (config.USER_KEY, episode_id), "one")
        last_summary = f"{last[1]:%d %b %Y}: {last[0]}" if last else ""
        return {"episode_id": episode_id, "history": history, "new_session": new_session,
                "previous_id": previous_id, "last_summary": last_summary}

    def log_turn(self, episode_id: int, heard: str, reply: str, tools: list, ms: int) -> None:
        self._sql("insert into turns (episode_id, role, content) values (%s, 'user', %s)", (episode_id, heard))
        self._sql("insert into turns (episode_id, role, content, tools, ms) values (%s, 'assistant', %s, %s, %s)",
                  (episode_id, reply, json.dumps(tools) if tools else None, ms))

    def _search_episodes(self, query: str, k: int) -> list[str]:
        rows = self._sql(
            "select summary, embedding, started_at from episodes "
            "where user_key = %s and summary is not null order by id desc limit 200",
            (config.USER_KEY,), "all") or []
        if not rows:
            return []
        try:
            q = np.asarray(gemini.embed(query), dtype=np.float32)
        except gemini.GeminiError:
            q = None
        scored = []
        for summary, emb, started in rows:
            s = 0.0
            if q is not None and emb and len(emb) == len(q):
                s = float(np.dot(q, np.asarray(emb, dtype=np.float32)))
            kw = _keyword_score(query, summary)
            if kw >= 0.5:
                s = max(s, config.MEMORY_MIN_SIMILARITY + 0.1 * kw)
            if s >= config.MEMORY_MIN_SIMILARITY - 0.05:
                scored.append((s, f"{started:%d %b %Y}: {summary}"))
        return [t for _s, t in sorted(scored, reverse=True)[:k]]

    # ------------------------------------------------------------ background learning

    def learn_from_turn(self, heard: str, reply: str) -> None:
        """Ask Gemini which lasting facts the user just shared, and store them."""
        if not config.AUTO_MEMORY or len(_words(heard)) < 3:
            return
        related = self._search_facts(heard, 6, 0.45)
        shown = {f["id"]: f["content"] for f, _s in related}
        existing = "\n".join(f"{fid}: {text}" for fid, text in shown.items()) or "(none)"
        prompt = (
            "You maintain the long-term memory of a home voice assistant. From the USER's message only, "
            "pick out lasting facts about the user, their household, preferences, routines, plans, people "
            "and belongings. Ignore questions, small talk, requests to the assistant, facts about the assistant, "
            "general knowledge, and temporary states (hungry, tired right now). Never keep passwords, PINs, "
            "card or account numbers, or health details.\n\n"
            f"Existing related memories (id: text):\n{existing}\n\n"
            f"USER said: {json.dumps(heard)}\nASSISTANT replied: {json.dumps(reply)}\n\n"
            'Answer with JSON only: {"ops": [...]} where each op is {"op": "add", "fact": "..."}, '
            '{"op": "update", "id": <id>, "fact": "..."} when the user changed an existing memory, or '
            '{"op": "delete", "id": <id>} when the user said it is no longer true. '
            'Facts are short third-person sentences such as "The user\'s favourite colour is blue." '
            'Return {"ops": []} when nothing is worth keeping, which is most of the time.'
        )
        try:
            data = gemini.chat_json(prompt, model=config.GEMINI_MEMORY_MODEL)
        except gemini.GeminiError as exc:
            print(f"[memory] learning skipped: {exc}")
            return
        ops = data.get("ops") if isinstance(data, dict) else None
        if isinstance(ops, list) and ops:
            for line in self.apply_ops(ops, set(shown)):
                print(f"[memory] learned: {line}")

    def summarise_episode(self, episode_id: int) -> None:
        rows = self._sql("select role, content from turns where episode_id = %s order by id",
                         (episode_id,), "all") or []
        if len(rows) < 2:
            return
        talk = "\n".join(f"{r[0]}: {r[1]}" for r in rows[-40:])
        prompt = ("Summarise this conversation between a user and their voice assistant in one or two "
                  "sentences, keeping names, decisions and anything worth remembering later. "
                  'Answer with JSON only: {"summary": "..."}\n\n' + talk)
        try:
            data = gemini.chat_json(prompt, model=config.GEMINI_MEMORY_MODEL)
            summary = str((data or {}).get("summary", "")).strip()
            if not summary:
                return
            vec = gemini.embed(summary)
        except gemini.GeminiError as exc:
            print(f"[memory] summary skipped: {exc}")
            return
        self._sql("update episodes set summary = %s, embedding = %s where id = %s", (summary[:600], vec, episode_id))
        print(f"[memory] conversation {episode_id} summarised: {summary}")


class Background:
    """One worker thread for slow memory jobs, so replies never wait for them."""

    def __init__(self) -> None:
        self._jobs: list = []
        self._cv = threading.Condition()
        threading.Thread(target=self._run, name="memory-jobs", daemon=True).start()

    def submit(self, fn, *args) -> None:
        with self._cv:
            self._jobs.append((fn, args))
            self._cv.notify()

    def idle(self, timeout: float = 30.0) -> bool:
        """Wait until the queue is empty (used by tests)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._jobs and not getattr(self, "_busy", False):
                    return True
            time.sleep(0.05)
        return False

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                fn, args = self._jobs.pop(0)
                self._busy = True
            try:
                fn(*args)
            except Exception as exc:  # a failed memory job must never stop the server
                print(f"[memory] background job failed: {exc}")
            finally:
                self._busy = False
