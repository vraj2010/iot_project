"""Friday's jukebox: finds songs in the library, keeps the play queue, and
prepares the audio the speaker streams.

The speaker does the playing; the laptop decides what plays. A reply can carry
an action for the speaker (play track 12, pause, volume 60), and the speaker
reports what it's doing in an X-Music-State header with every question, so
Friday always knows what's on.

    GET /music/<id>?t=<seconds>   a song (from t seconds in), 16 kHz mono MP3
    GET /music/next?after=<id>    whatever comes next in the queue (204 = nothing)
"""

from __future__ import annotations

import difflib
import random
import re
import threading
import time
from pathlib import Path

import config
import music

# Words people add that aren't part of a song's name: "play the song Kesariya".
_FILLER = re.compile(r"\b(?:play|the|a|song|songs|track|tracks|music|some|by|from|movie|film|album|please)\b", re.I)
_SAMPLE_RECENT = 20                                  # autoplay avoids the last N songs played


def _norm(s: str) -> str:
    return music._norm(s or "")


def _similar(query: str, field: str) -> float:
    """0..1: how well a spoken name matches a title / artist / album."""
    q, f = _norm(query), _norm(field)
    if not q or not f:
        return 0.0
    if q == f:
        return 1.0
    if len(q) >= 3 and f" {q} " in f" {f} ":         # "arijit singh" in "arijit singh, pritam"
        return 0.92
    if len(f) >= 4 and f" {f} " in f" {q} ":         # "kesariya" in "kesariya brahmastra"
        return 0.85
    qw, fw = q.split(), f.split()
    overlap = len(set(qw) & set(fw)) / len(set(qw))
    # Spelling slips are fine ("kesaria" ~ "kesariya"), but every word said must
    # resemble a word in the name: "baby" must not match "Bad Boy".
    words = sum(max(difflib.SequenceMatcher(None, w, x).ratio() for x in fw) for w in qw) / len(qw)
    chars = difflib.SequenceMatcher(None, q, f).ratio()
    return max(min(chars, words), overlap * 0.9)


def spoken(result) -> str:
    """What Friday says after a successful play / pause / skip / volume result ("Here's Kesariya.", "Paused."),
    or "" when it needs Gemini to word it (an error, or anything else)."""
    if not isinstance(result, dict) or "error" in result:
        return ""
    if "playing" in result:
        return "From the top." if result.get("note") == "from the start" else f"Here's {result['playing']}."
    if "queued" in result:
        what = "those" if "songs (" in str(result["queued"]) else str(result["queued"])
        return f"Okay, {what} up next." if result.get("position") == "next" else f"Added {what} to the queue."
    if "done" in result:
        done = str(result["done"])
        return {"paused": "Paused.", "resumed": "Back on.", "stopped": "Stopped.",
                "already playing": "It's already playing.",
                "nothing was playing": "Nothing was playing."}.get(done, done[0].upper() + done[1:] + ".")
    return ""


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


class Jukebox:
    def __init__(self, memory) -> None:
        self.mem = memory
        self.lock = threading.RLock()
        self._tracks: dict[int, dict] = {}
        self._loaded_at = 0.0
        self._sig = None
        self.queue: list[int] = []                   # what plays after the current song
        self.history: list[int] = []                 # played, oldest first
        self.device: dict = {}                       # last X-Music-State from the speaker
        self._scan_thread: threading.Thread | None = None
        self.ffmpeg = music.ffmpeg_path()
        self.reload()
        print(f"[music] {len(self._tracks)} songs in the library"
              + ("" if self.ffmpeg else " (ffmpeg not found: MP3s only, sent as they are)"))
        if config.MUSIC_SCAN_MIN > 0:
            threading.Thread(target=self._watch, name="music-watch", daemon=True).start()

    # ------------------------------------------------------------ library

    def reload(self, force: bool = True) -> None:
        """Re-read the library if it changed (cheap check), or always with force."""
        with self.lock:
            if not force and time.time() - self._loaded_at < 10:
                return
            sig = self.mem._sql("select count(*), max(updated_at) from music_tracks where source_type = 'file'",
                                fetch="one")
            self._loaded_at = time.time()
            if not force and sig == self._sig:
                return
            self._sig = sig
            rows = self.mem._sql(
                "select id, title, artist, album, genre, year, duration_s, path, file_name, content_hash, "
                "file_size, file_mtime from music_tracks where source_type = 'file'", fetch="all") or []
            keys = ("id", "title", "artist", "album", "genre", "year", "duration_s", "path", "file_name",
                    "content_hash", "file_size", "file_mtime")
            self._tracks = {r[0]: dict(zip(keys, r)) for r in rows}

    def track(self, track_id: int) -> dict | None:
        self.reload(force=False)
        with self.lock:
            return self._tracks.get(track_id)

    def count(self) -> int:
        with self.lock:
            return len(self._tracks)

    @staticmethod
    def label(t: dict | None) -> str:
        if not t:
            return "nothing"
        return f"{t['title']} by {t['artist']}" if t.get("artist") else t["title"]

    def find(self, query: str, kind: str = "any", artist: str = "") -> tuple[list[dict], str]:
        """Songs for a request. Returns (songs, what matched: song | artist | album | genre | none)."""
        self.reload(force=False)
        with self.lock:
            tracks = list(self._tracks.values())
        q = _FILLER.sub(" ", query or "").strip()
        if not q and artist:
            q, kind = artist, "artist"
        if not q:
            return [], "none"

        def song_score(t: dict) -> float:
            s = max(_similar(q, t["title"]), 0.9 * _similar(q, music.clean_name(t["file_name"] or "")["pretty"]))
            if artist:
                s += 0.15 if _similar(artist, t["artist"] or "") >= 0.7 else -0.25
            return s

        def group(field: str, cutoff: float = 0.75) -> list[dict]:
            scored = [(_similar(q, t[field] or ""), t) for t in tracks]
            best = max((s for s, _ in scored), default=0.0)
            if best < cutoff:
                return []
            return [t for s, t in scored if s >= max(cutoff, best - 0.08)]

        if kind in ("artist", "album", "genre"):
            hits = group(kind)
            return hits, kind if hits else "none"

        ranked = sorted(((song_score(t), t) for t in tracks), key=lambda x: -x[0])
        best_song = ranked[0][0] if ranked else 0.0
        if kind == "song":
            return ([ranked[0][1]], "song") if best_song >= 0.6 else ([], "none")
        # "any": the best of song, artist, album
        artist_hits, album_hits = group("artist"), group("album", 0.8)
        artist_best = max((_similar(q, t["artist"] or "") for t in artist_hits), default=0.0)
        album_best = max((_similar(q, t["album"] or "") for t in album_hits), default=0.0)
        if best_song >= 0.6 and best_song >= artist_best and best_song >= album_best:
            return [ranked[0][1]], "song"
        if artist_hits and artist_best >= album_best:
            return artist_hits, "artist"
        if album_hits:
            return album_hits, "album"
        if best_song >= 0.6:
            return [ranked[0][1]], "song"
        return [], "none"

    def closest(self, query: str, n: int = 3) -> list[str]:
        """A few near-misses, so Friday can say what she does have."""
        with self.lock:
            tracks = list(self._tracks.values())
        q = _FILLER.sub(" ", query or "")
        ranked = sorted(tracks, key=lambda t: -max(_similar(q, t["title"]), _similar(q, t["artist"] or "")))
        return [self.label(t) for t in ranked[:n]]

    # ------------------------------------------------------------ what the speaker is doing

    def observe(self, header: str | None) -> bool:
        """Read the speaker's X-Music-State, e.g. "state=playing;track=12;pos=45.2;vol=70".
        Returns False when the speaker sent none (old firmware, or the cloud Worker's path)."""
        if not header:
            self.device = {}
            return False
        state = {}
        for part in header.split(";"):
            k, _, v = part.partition("=")
            state[k.strip()] = v.strip()
        out = {"state": state.get("state", "stopped")}
        for key, cast in (("track", int), ("pos", float), ("vol", int)):
            try:
                out[key] = cast(state[key])
            except (KeyError, ValueError):
                pass
        self.device = out
        return True

    def current(self) -> dict | None:
        if self.device.get("state") in ("playing", "paused") and self.device.get("track"):
            return self.track(self.device["track"])
        return None

    def describe(self) -> str:
        """One line for Friday's prompt."""
        d = self.device
        t = self.current()
        n = self.count()
        if not t:
            return f"Music: nothing playing. Your library has {n} songs."
        where = f"{_fmt_time(d.get('pos'))} of {_fmt_time(t.get('duration_s'))}"
        line = f"Music: {d['state']} \"{t['title']}\"" + (f" by {t['artist']}" if t.get("artist") else "") + f", {where}."
        if "vol" in d:
            line += f" Volume {d['vol']} percent."
        with self.lock:
            nxt = self._tracks.get(self.queue[0]) if self.queue else None
        if nxt:
            line += f" Up next: \"{nxt['title']}\"."
        return line

    # ------------------------------------------------------------ queue

    def _others(self, exclude: set[int], prefer_artist: str = "") -> list[int]:
        """Everything else, shuffled, songs by `prefer_artist` first, recent plays last."""
        with self.lock:
            ids = [i for i in self._tracks if i not in exclude]
            recent = set(self.history[-_SAMPLE_RECENT:])
            main_artist = _norm(prefer_artist.split(",")[0]) if prefer_artist else ""
            same = [i for i in ids if main_artist and main_artist in _norm(self._tracks[i]["artist"] or "")]
        rest = [i for i in ids if i not in same]
        random.shuffle(same)
        random.shuffle(rest)
        rest.sort(key=lambda i: i in recent)           # stable: fresh songs first, still shuffled
        return same + rest

    def _start(self, track_id: int) -> dict:
        t = self.track(track_id)
        return {"music": "play", "track": track_id, "title": self.label(t)[:80]}

    def next_after(self, after: int | None) -> int | None:
        """The song after `after`, for the speaker's auto-advance and "skip"."""
        with self.lock:
            while self.queue:
                nxt = self.queue.pop(0)
                if nxt in self._tracks:
                    return nxt
            if not config.MUSIC_AUTOPLAY or not self._tracks:
                return None
            cur = self._tracks.get(after) if after else None
            more = self._others({after} if after else set(), cur["artist"] if cur else "")
            if not more:
                return None
            self.queue = more[1:50]
            return more[0]

    def played(self, track_id: int) -> None:
        """A song started from the beginning: remember it (queue, play count)."""
        with self.lock:
            self.history.append(track_id)
            del self.history[:-200]
        try:
            self.mem._sql("update music_tracks set play_count = play_count + 1, last_played_at = now() "
                          "where id = %s", (track_id,))
        except Exception as exc:
            print(f"[music] couldn't update the play count: {exc}")

    # ------------------------------------------------------------ tools for Gemini

    def play(self, query: str = "", kind: str = "any", artist: str = "", when: str = "now",
             shuffle: bool = False) -> tuple[dict, dict | None]:
        """Returns (result for Gemini, action for the speaker)."""
        if not self.count():
            return {"error": "the music library is empty; run python music.py scan on the laptop"}, None
        kind = kind if kind in ("song", "artist", "album", "genre", "any") else "any"

        if not (query or "").strip() and not artist:   # "play some music"
            order = self._others(set())
            with self.lock:
                self.queue = order[1:50]
            return {"playing": self.label(self.track(order[0])), "mode": "shuffle, whole library"}, \
                self._start(order[0])

        songs, matched = self.find(query, kind, artist)
        if not songs:
            return {"error": f"no song matching '{query or artist}' in the library",
                    "closest": self.closest(query or artist)}, None

        ids = [t["id"] for t in songs]
        if matched == "song":
            first, rest = ids[0], []
        else:
            if shuffle or matched in ("artist", "genre"):
                random.shuffle(ids)
            else:                                      # albums in a stable order
                ids.sort(key=lambda i: (self._tracks[i]["title"] or "").lower())
            first, rest = ids[0], ids[1:]

        if when in ("next", "later") and self.current():
            with self.lock:
                if when == "next":
                    self.queue[:0] = [first] + rest
                else:
                    self.queue += [first] + rest
            what = self.label(self.track(first)) if matched == "song" else f"{len(ids)} songs ({matched}: {query or artist})"
            return {"queued": what, "position": when}, None

        cur = self.track(first)
        with self.lock:
            tail = self._others(set(ids), cur.get("artist") or "") if config.MUSIC_AUTOPLAY else []
            self.queue = (rest + tail)[:60]
        result = {"playing": self.label(cur)}
        if matched != "song":
            result["matched"] = f"{matched} '{query or artist}', {len(ids)} songs"
        return result, self._start(first)

    def control(self, action: str, level: int | None = None) -> tuple[dict, dict | None]:
        d = self.device
        playing = d.get("state") in ("playing", "paused")
        vol = d.get("vol", 70)
        action = (action or "").lower().replace(" ", "_")

        if action == "pause":
            return ({"done": "paused"}, {"music": "pause"}) if playing else ({"error": "nothing is playing"}, None)
        if action == "resume":
            if d.get("state") == "paused":
                return {"done": "resumed"}, {"music": "resume"}
            if d.get("state") == "playing":
                return {"done": "already playing"}, None
            return {"error": "nothing to resume; ask for a song"}, None
        if action == "stop":
            with self.lock:
                self.queue.clear()
            return ({"done": "stopped"}, {"music": "stop"}) if playing else ({"done": "nothing was playing"}, None)
        if action in ("next", "skip"):
            nxt = self.next_after(d.get("track"))
            if nxt is None:
                return {"error": "nothing else in the queue"}, None
            return {"playing": self.label(self.track(nxt))}, self._start(nxt)
        if action in ("previous", "back", "restart"):
            cur = d.get("track")
            with self.lock:
                past = [i for i in self.history if i != cur]
            if action == "restart" or (cur and d.get("pos", 0) > 8) or not past:
                if not cur:
                    return {"error": "nothing is playing"}, None
                return {"playing": self.label(self.track(cur)), "note": "from the start"}, self._start(cur)
            prev = past[-1]
            with self.lock:
                if cur:
                    self.queue.insert(0, cur)
                if prev in self.history:
                    self.history.remove(prev)
            return {"playing": self.label(self.track(prev))}, self._start(prev)
        if action in ("volume_up", "louder"):
            new = min(100, vol + 15)
        elif action in ("volume_down", "quieter", "softer"):
            new = max(5, vol - 15)
        elif action in ("set_volume", "volume"):
            if level is None:
                return {"error": "say what volume, 0 to 100"}, None
            new = max(0, min(100, int(level)))
        else:
            return {"error": f"unknown action {action}"}, None
        return {"done": f"volume {new} percent"}, {"volume": new}

    # ------------------------------------------------------------ new songs

    def check_new(self) -> dict:
        """Quick look for added or removed songs; starts indexing them in the background."""
        try:
            changes = music.pending_changes(self.mem)
        except FileNotFoundError as exc:
            return {"error": str(exc)}
        todo = changes["new"] + changes["changed"]
        if self._scan_thread and self._scan_thread.is_alive():
            return {**changes, "status": "already adding songs, give it a few minutes"}
        if todo or changes["removed"]:
            self._scan_thread = threading.Thread(target=self._scan, name="music-scan", daemon=True)
            self._scan_thread.start()
            minutes = max(1, round(todo * config.ITUNES_GAP_S * 1.4 / 60))
            return {**changes, "status": f"adding them now, about {minutes} minute(s)" if todo else "updating"}
        return {**changes, "status": "nothing new"}

    def _scan(self) -> None:
        try:
            print("[music] indexing new songs...")
            music.scan(mem=self.mem)
        except BaseException as exc:                   # scan() uses SystemExit for a missing folder
            print(f"[music] scan failed: {exc}")
        self.reload()
        print(f"[music] library now has {self.count()} songs")

    def _watch(self) -> None:
        """Every MUSIC_SCAN_MIN minutes: index songs dropped into the folder."""
        while True:
            time.sleep(config.MUSIC_SCAN_MIN * 60)
            try:
                c = music.pending_changes(self.mem)
                if c["new"] or c["changed"] or c["removed"]:
                    print(f"[music] folder changed ({c['new']} new, {c['removed']} removed)")
                    self.check_new()
            except Exception as exc:
                print(f"[music] folder check failed: {exc}")

    # ------------------------------------------------------------ audio for the speaker

    def source_file(self, t: dict) -> Path | None:
        """The song's file, only if it really is an indexed audio file inside MUSIC_DIR."""
        try:
            p = Path(t["path"]).resolve()
            root = Path(config.MUSIC_DIR).resolve()
        except (OSError, TypeError):
            return None
        if p.suffix.lower() not in music.AUDIO_EXT or not p.is_file() or not p.is_relative_to(root):
            return None
        return p

    def cache_file(self, t: dict) -> Path:
        return music.cache_path(t["id"], music._fingerprint(t["content_hash"], t["file_size"], t["file_mtime"]))
