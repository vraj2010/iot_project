"""Friday's music library: your song files + clean metadata from Apple's iTunes Search API.

    python music.py scan              index MUSIC_DIR (new or changed files only)
    python music.py scan --rescan     look every song up again
    python music.py scan --dry-run    show what would be matched, store nothing
    python music.py list              everything in the library
    python music.py list --review     only songs that need a human look
    python music.py convert           only convert songs for the speaker (needs ffmpeg)

How a song is identified:
  1. Read the file's own tags (title, artist, length) with mutagen.
  2. Clean the filename: drop "(256k)", "PagalWorld.com", "Official Video", track numbers...
  3. Search iTunes (India store first) and score each result on title, artist and length.
  4. Confident match -> official title, artists, album, year, genre, artwork.
     Unsure -> stored too, but needs_review = true, so you can check it in Adminer.
     No match (ringtones, BGMs) -> cleaned filename is kept as the title, needs_review = true.

The iTunes Search API is free, needs no key, and allows roughly 20 requests a minute,
so a first scan of 150 songs takes about 10 minutes. Later scans only look up new files,
and a renamed file is recognised by its fingerprint (content_hash), so it keeps its metadata.

After indexing, each song is converted once with ffmpeg into MUSIC_CACHE_DIR in the
speaker's format (16 kHz mono MP3), so playback starts instantly.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests

import config

AUDIO_EXT = {".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg"}

# Junk that download sites and video rips add to names and tags.
_JUNK = [
    r"\(?\b\d{2,3}\s*k(?:bps)?\b\)?",            # (128k) 320Kbps 128 Kbps
    r"\b(?:128|190|256|320)\b(?=\s*$|\s*-)",      # trailing bitrate numbers
    r"\bkbps\b",
    r"\(?\bwww\.[^\s)]+\)?", r"\(?\b[\w.]*pagalworld[\w.]*\b\)?", r"\braag\.me\b",
    r"\[?mymp3bhojpuri\.in\]?", r"\bmuzmo\s*ru\b", r"\bgeet\s*mp3\b", r"\bgk\.digital\b",
    r"\b(?:full\s+)?(?:official\s+)?(?:music\s+)?video(?:\s+song)?\b", r"\bfull\s+song\b", r"\bsong\s+video\b",
    r"\bofficial(?:\s+song|\s+audio)?\b", r"\blyric(?:al|s)?(?:\s+video)?\b", r"\bwith\s+lyrics\b",
    r"\bwebrip\b", r"\bhd\b", r"\bt-series\b", r"\bspeed\s+records\b", r"\bbass\s+boosted\b",
    r"\blatest(?:\s+\w+){0,3}\s+songs?(?:\s+20\d\d)?\b", r"\bnew(?:\s+\w+){0,2}\s+songs?(?:\s+20\d\d)?\b",
    r"\b20[012]\d\b", r"\by!\s*", r"-\s*copy\b", r"\bquot\b", r"\bfree\s+download\b",
    r"\[\d+\]", r"\bmp3\b",
]
_JUNK_RE = re.compile("|".join(_JUNK), re.I)


def _norm(s: str) -> str:
    """Lowercase letters and digits only, for comparing names."""
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)          # "(From "Kabir Singh")", "[Remix]"
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def clean_name(filename: str) -> dict:
    """Turn a messy filename into a search query plus its parts.
    Returns {"query", "parts", "film", "pretty"}."""
    name = filename
    while Path(name).suffix.lower() in AUDIO_EXT | {".mp4"}:   # "song.mp3.mp3"
        name = name[: -len(Path(name).suffix)]
    name = name.replace("u0026", "&").replace("&quot;", "").replace("quot;", "").replace("–", "-")
    if " " not in name and name.count("-") >= 3:                # "Rang-Bhini-Radha-Folk-Box-..."
        name = name.replace("-", " ")
    name = name.replace("´", "'").replace("’", "'").replace("{", "(").replace("}", ")")

    film = ""
    m = re.search(r"\(From[_\s]+_?(.+?)_?\)", name, re.I)    # "(From__Ae_Dil_Hai_Mushkil_)"
    if m:
        film = m.group(1).replace("_", " ").strip(" '\"")
        name = name[: m.start()] + name[m.end():]

    name = re.sub(r"_{2,}", " | ", name)            # "__" separates title / film / cast in video rips
    name = name.replace("_", " ")
    name = re.sub(r"^\s*(?:\d{1,3}\s*[-.]\s*|\d{1,2}\s+(?=[A-Za-z]))", "", name)   # "07 ", "03 - ", "128-"
    name = _JUNK_RE.sub(" ", _JUNK_RE.sub(" ", name))           # twice: removing one bit can expose another
    name = re.sub(r"\(\s*\)|\[\s*\]", " ", name)
    name = re.sub(r"[#\"]|(?<!\w)'|'(?!\w)", " ", name)        # keep the ' in "Let's"
    name = name.strip(" &")

    parts = [p.strip(" -|,.") for p in re.split(r"\s+\|\s+|\s+-\s+", name)]
    parts = [re.sub(r"\s{2,}", " ", p) for p in parts if len(_norm(p)) >= 2]
    if not parts:
        parts = [Path(filename).stem]
    words = " ".join(parts[:2]).split()
    query = re.sub(r"[()\[\]]", " ", " ".join(words[:9]))
    query = re.sub(r"\s{2,}", " ", query)
    if film and film.lower() not in query.lower():
        query = f"{query} {film}"
    query = re.sub(r"\s{2,}", " ", query)
    pretty = parts[0].strip()
    return {"query": query.strip(), "parts": parts, "film": film, "pretty": pretty[:120]}


def _clean_tag(value: str | None) -> str:
    if not value:
        return ""
    v = _JUNK_RE.sub(" ", value)
    v = re.sub(r"^\s*\d{1,3}\s*[-.]?\s*", "", v)
    v = re.sub(r"\s{2,}", " ", v).strip(" -|,.")
    return "" if len(_norm(v)) < 2 else v


def read_tags(path: Path) -> dict:
    """Title, artist, album and length from the file itself (mutagen)."""
    out = {"title": "", "artist": "", "album": "", "duration": None}
    try:
        import mutagen

        f = mutagen.File(str(path), easy=True)
        if f is None:
            return out
        if getattr(f, "info", None) and getattr(f.info, "length", None):
            out["duration"] = float(f.info.length)
        tags = f.tags or {}
        for key in ("title", "artist", "album"):
            val = tags.get(key)
            if isinstance(val, list):
                val = val[0] if val else ""
            out[key] = _clean_tag(str(val)) if val else ""
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- iTunes

_last_call = 0.0


def itunes_search(term: str, country: str) -> list[dict]:
    """Songs matching `term`. Paces itself to stay under Apple's ~20 requests/minute."""
    global _last_call
    wait = config.ITUNES_GAP_S - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    for attempt in (1, 2):
        _last_call = time.time()
        r = requests.get(f"{config.ITUNES_BASE}/search", timeout=15, params={
            "term": term, "media": "music", "entity": "song", "country": country, "limit": 10})
        if r.status_code in (403, 429) and attempt == 1:          # rate limited: back off once
            print(f"    (iTunes says slow down; waiting {config.ITUNES_BACKOFF_S:.0f} s)")
            time.sleep(config.ITUNES_BACKOFF_S)
            continue
        r.raise_for_status()
        return (r.json() or {}).get("results", [])
    return []


def _score(hit: dict, info: dict, tags: dict) -> float:
    """0..1+ : how well an iTunes result fits this file."""
    track = _norm(hit.get("trackName", ""))
    if not track:
        return 0.0
    names = [p for p in info["parts"][:2]] + [tags["title"]]
    sims = []
    for n in filter(None, names):
        nn = _norm(n)
        if not nn:
            continue
        if track and (track in nn or nn in track) and min(len(track), len(nn)) >= 4:
            sims.append(0.95)
        sims.append(difflib.SequenceMatcher(None, track, nn).ratio())
    score = max(sims) if sims else 0.0

    hay = set(_norm(" ".join(info["parts"] + [tags["artist"], info["film"]])).split())
    artist_words = {w for w in _norm(hit.get("artistName", "")).split() if len(w) > 3}
    album_words = {w for w in _norm(hit.get("collectionName", "")).split() if len(w) > 3}
    if artist_words & hay:
        score += 0.12
    if album_words & hay:
        score += 0.08

    if tags["duration"] and hit.get("trackTimeMillis"):
        diff = abs(hit["trackTimeMillis"] / 1000 - tags["duration"])
        if diff <= 8:
            score += 0.15
        elif diff > 90:                        # video rips run long, so only punish big gaps
            score -= 0.2
    return score


def identify(path: Path) -> dict:
    """Best metadata for one file. Never raises for a bad lookup; falls back to the filename."""
    tags = read_tags(path)
    info = clean_name(path.name)
    queries = []
    if tags["title"] and tags["artist"]:
        queries.append(f"{tags['title']} {tags['artist'].split(',')[0]}")
    queries.append(info["query"])
    if info["parts"][0].lower() != info["query"].lower():
        queries.append(info["parts"][0])
    seen, best, best_score, used = set(), None, 0.0, ""
    for q in queries:
        q = q.strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        for country in dict.fromkeys([config.ITUNES_COUNTRY, "US"]):
            try:
                hits = itunes_search(q, country)
            except requests.RequestException as exc:
                print(f"    iTunes lookup failed: {exc}")
                hits = []
            for h in hits:
                s = _score(h, info, tags)
                if s > best_score:
                    best, best_score, used = h, s, q
            if best_score >= config.MATCH_ACCEPT or hits:
                break                          # found results in this store; don't ask the next one
        if best_score >= config.MATCH_ACCEPT:
            break

    base = {"duration_s": int(tags["duration"]) if tags["duration"] else None, "query": used or info["query"]}
    if best and best_score >= config.MATCH_REVIEW:
        art = (best.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
        year = (best.get("releaseDate") or "")[:4]
        return {**base,
                "title": best.get("trackName", "")[:200], "artist": best.get("artistName", "")[:200],
                "album": best.get("collectionName", "")[:200], "genre": best.get("primaryGenreName"),
                "year": int(year) if year.isdigit() else None, "cover_url": art or None,
                "itunes_track_id": best.get("trackId"), "itunes_url": best.get("trackViewUrl"),
                "explicit": best.get("trackExplicitness") == "explicit",
                "match_source": "itunes", "match_score": round(best_score, 3),
                "needs_review": best_score < config.MATCH_ACCEPT}
    return {**base, "title": tags["title"] or info["pretty"], "artist": tags["artist"] or None,
            "album": tags["album"] or (info["film"] or None), "genre": None, "year": None, "cover_url": None,
            "itunes_track_id": None, "itunes_url": None, "explicit": None,
            "match_source": "tags" if tags["title"] else "filename",
            "match_score": round(best_score, 3), "needs_review": True}


# ---------------------------------------------------------------- files for the speaker

def file_hash(path: Path) -> str:
    """SHA-1 of the file's bytes: the same song keeps the same fingerprint when renamed or moved."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ffmpeg_path() -> str | None:
    """ffmpeg's location, or None if it isn't installed."""
    p = config.FFMPEG
    if p and os.path.isfile(p):
        return p
    return shutil.which(p or "ffmpeg")


def _id3_size(head: bytes) -> int:
    """Length of an ID3v2 tag at the start of a file (0 if there isn't one)."""
    if len(head) < 10 or head[:3] != b"ID3":
        return 0
    size = (head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14 | (head[8] & 0x7F) << 7 | (head[9] & 0x7F)
    return 10 + size + (10 if head[5] & 0x10 else 0)


def mp3_span(path: Path) -> tuple[int, int] | None:
    """(first, end) byte range of the MP3 frames, skipping ID3 tags and cover art.
    None when the file isn't really an MP3 (video rips are often AAC named .mp3)."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        start = _id3_size(f.read(10))
        f.seek(start)
        sync = f.read(2)
        if len(sync) < 2 or sync[0] != 0xFF or (sync[1] & 0xE0) != 0xE0:
            return None
        end = size
        if size >= 128:
            f.seek(size - 128)
            if f.read(3) == b"TAG":                    # ID3v1 at the end
                end = size - 128
    return start, end


def _settings_tag() -> str:
    return f"{config.MUSIC_RATE}-{config.MUSIC_BITRATE}-{int(config.MUSIC_LOUDNORM)}"


def cache_path(track_id: int, fingerprint: str) -> Path:
    """Where the speaker-ready copy of a song is kept. The name changes when the
    file or the conversion settings change, so a stale copy is never used."""
    key = hashlib.sha1(f"{fingerprint}|{_settings_tag()}".encode()).hexdigest()[:10]
    return Path(config.MUSIC_CACHE_DIR) / f"{track_id}-{key}.mp3"


def ffmpeg_cmd(src: Path, start_s: float = 0.0, out: str = "pipe:1") -> list[str]:
    """Song -> 16 kHz mono constant-bitrate MP3 with no tags, as the speaker wants it."""
    cmd = [ffmpeg_path() or "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.2f}"]
    cmd += ["-i", str(src), "-vn", "-sn", "-map_metadata", "-1", "-ac", "1"]
    if config.MUSIC_LOUDNORM:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-ar", str(config.MUSIC_RATE), "-c:a", "libmp3lame", "-b:a", config.MUSIC_BITRATE,
            "-write_xing", "0", "-id3v2_version", "0", "-f", "mp3", out]
    return cmd


def convert(src: Path, dest: Path) -> bool:
    """Write the speaker-ready copy of one song. False if ffmpeg failed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
    try:
        r = subprocess.run(ffmpeg_cmd(src, out=str(tmp)), capture_output=True, timeout=600)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            print(f"    ffmpeg failed on {src.name}: {r.stderr.decode(errors='replace').strip()[:200]}")
            return False
        os.replace(tmp, dest)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"    ffmpeg failed on {src.name}: {exc}")
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------- database

def _db():
    from memory import Memory

    return Memory()


def _folder_files() -> list[Path]:
    root = Path(config.MUSIC_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"MUSIC_DIR not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def _known(mem) -> dict:
    """path -> {id, size, mtime, hash} for every indexed local file."""
    rows = mem._sql("select id, path, file_size, file_mtime, content_hash from music_tracks "
                    "where source_type = 'file' and storage = 'local'", fetch="all") or []
    return {r[1]: {"id": r[0], "size": r[2], "mtime": r[3], "hash": r[4]} for r in rows}


def _changed(p: Path, k: dict | None) -> bool:
    st = p.stat()
    return not k or k["size"] != st.st_size or abs((k["mtime"] or 0) - st.st_mtime) > 1


def pending_changes(mem) -> dict:
    """Quick look (no internet): how many songs were added, changed or removed since the last scan."""
    files = _folder_files()
    known = _known(mem)
    present = {str(p) for p in files}
    new = sum(1 for p in files if str(p) not in known)
    changed = sum(1 for p in files if str(p) in known and _changed(p, known[str(p)]))
    removed = sum(1 for k in known if k not in present)
    return {"new": new, "changed": changed, "removed": removed, "total": len(files)}


def scan(rescan: bool = False, dry_run: bool = False, limit: int | None = None, mem=None,
         prepare: bool = True) -> dict:
    """Index MUSIC_DIR. Returns counts. Safe to run while the server is playing music."""
    try:
        files = _folder_files()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    root = Path(config.MUSIC_DIR)
    if mem is None and not dry_run:
        mem = _db()
    known = _known(mem) if mem else {}
    present = {str(p) for p in files}
    gone = {path: k for path, k in known.items() if path not in present}

    todo = [p for p in files if rescan or _changed(p, known.get(str(p)))]
    if limit:
        todo = todo[:limit]
    print(f"{len(files)} audio files in {root}, {len(todo)} to look up "
          f"(~{len(todo) * config.ITUNES_GAP_S * 1.4 / 60:.0f} min at Apple's pace)\n")

    counts = {"matched": 0, "review": 0, "unmatched": 0, "renamed": 0, "removed": 0}
    for i, p in enumerate(todo, 1):
        st = p.stat()
        digest = file_hash(p)
        if mem and not rescan:
            # Same bytes as a song that vanished from its old name: a rename. Keep its metadata.
            moved = next((path for path, k in gone.items() if k["hash"] == digest), None)
            if moved:
                mem._sql("update music_tracks set path = %s, file_name = %s, file_size = %s, file_mtime = %s "
                         "where id = %s", (str(p), p.name, st.st_size, st.st_mtime, gone[moved]["id"]))
                gone.pop(moved)
                counts["renamed"] += 1
                print(f"[{i:3}/{len(todo)}] >> {p.name[:60]:60} renamed from {Path(moved).name[:40]}")
                continue
        meta = identify(p)
        kind = ("matched" if not meta["needs_review"] else
                "review" if meta["match_source"] == "itunes" else "unmatched")
        counts[kind] += 1
        mark = {"matched": "OK ", "review": "?? ", "unmatched": "-- "}[kind]
        print(f"[{i:3}/{len(todo)}] {mark}{p.name[:60]:60} -> {meta['title']} | {meta['artist'] or ''} "
              f"({meta['match_score']})")
        if dry_run:
            continue
        mem._sql("""
            insert into music_tracks (source_type, storage, path, file_name, file_size, file_mtime, content_hash,
                title, artist, album, genre, year, duration_s, cover_url, itunes_track_id, itunes_url, explicit,
                match_source, match_score, needs_review)
            values ('file', 'local', %(path)s, %(file_name)s, %(size)s, %(mtime)s, %(hash)s, %(title)s,
                %(artist)s, %(album)s, %(genre)s, %(year)s, %(duration_s)s, %(cover_url)s, %(itunes_track_id)s,
                %(itunes_url)s, %(explicit)s, %(match_source)s, %(match_score)s, %(needs_review)s)
            on conflict (path) do update set file_name = excluded.file_name, file_size = excluded.file_size,
                file_mtime = excluded.file_mtime, content_hash = excluded.content_hash, title = excluded.title,
                artist = excluded.artist, album = excluded.album, genre = excluded.genre, year = excluded.year,
                duration_s = excluded.duration_s, cover_url = excluded.cover_url,
                itunes_track_id = excluded.itunes_track_id, itunes_url = excluded.itunes_url,
                explicit = excluded.explicit, match_source = excluded.match_source,
                match_score = excluded.match_score, needs_review = excluded.needs_review
            """, {**meta, "path": str(p), "file_name": p.name, "size": st.st_size, "mtime": st.st_mtime,
                  "hash": digest})

    if mem and not dry_run:
        for path in gone:                              # deleted from the folder
            mem._sql("delete from music_tracks where path = %s and source_type = 'file'", (path,))
        counts["removed"] = len(gone)
        if gone:
            print(f"\nremoved {len(gone)} songs that are no longer in the folder")
        # Fingerprint songs indexed before fingerprints existed (fast, local only).
        for path, k in known.items():
            if not k["hash"] and path in present:
                try:
                    mem._sql("update music_tracks set content_hash = %s where id = %s",
                             (file_hash(Path(path)), k["id"]))
                except OSError:
                    pass
    print(f"\nmatched {counts['matched']}, needs review {counts['review']}, not found {counts['unmatched']}"
          + (f", renamed {counts['renamed']}" if counts["renamed"] else ""))
    if counts["review"] or counts["unmatched"]:
        print("Check the unsure ones with:  python music.py list --review  (or in Adminer: table music_tracks)")
    if mem and prepare and not dry_run:
        prepare_all(mem)
    return counts


def _fingerprint(row_hash: str | None, size, mtime) -> str:
    return row_hash or f"{size}-{mtime}"


def prepare_all(mem) -> None:
    """Convert every song for the speaker ahead of time, so the first play starts instantly."""
    if not ffmpeg_path():
        print("\nffmpeg not found: songs will be sent as they are and M4A files won't play. "
              "Install it with:  winget install Gyan.FFmpeg")
        return
    rows = mem._sql("select id, path, content_hash, file_size, file_mtime from music_tracks "
                    "where source_type = 'file' order by id", fetch="all") or []
    todo = [(r[0], Path(r[1]), cache_path(r[0], _fingerprint(r[2], r[3], r[4]))) for r in rows]
    todo = [t for t in todo if not t[2].exists() and t[1].exists()]
    if not todo:
        return
    print(f"\nconverting {len(todo)} songs for the speaker (once per song)...")
    t0 = time.time()
    ok = 0
    for i, (tid, src, dest) in enumerate(todo, 1):
        if convert(src, dest):
            ok += 1
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f} s)")
    _prune_cache({cache_path(r[0], _fingerprint(r[2], r[3], r[4])).name for r in rows})
    print(f"converted {ok} of {len(todo)}")


def _prune_cache(keep: set[str]) -> None:
    """Delete converted copies of songs that changed or were removed."""
    d = Path(config.MUSIC_CACHE_DIR)
    if not d.is_dir():
        return
    for f in d.glob("*.mp3"):
        if f.name not in keep:
            try:
                f.unlink()
            except OSError:
                pass


def list_tracks(review_only: bool) -> None:
    mem = _db()
    rows = mem._sql(
        "select id, title, artist, album, year, match_source, match_score, needs_review, file_name "
        "from music_tracks where source_type = 'file' " + ("and needs_review " if review_only else "")
        + "order by needs_review desc, artist nulls last, title", fetch="all") or []
    for r in rows:
        flag = "??" if r[7] else "  "
        print(f"{flag} {r[0]:4} {r[1][:40]:40} | {(r[2] or '')[:30]:30} | {(r[3] or '')[:25]:25} | {r[4] or '':4} "
              f"| {r[5]} {r[6]} | {r[8][:40]}")
    print(f"\n{len(rows)} songs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Friday's music library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="index new or changed songs, then convert them for the speaker")
    s.add_argument("--rescan", action="store_true", help="look every song up again")
    s.add_argument("--dry-run", action="store_true", help="show matches, store nothing")
    s.add_argument("--limit", type=int)
    s.add_argument("--no-convert", action="store_true", help="skip the ffmpeg step")
    l = sub.add_parser("list")
    l.add_argument("--review", action="store_true")
    sub.add_parser("convert", help="only convert songs for the speaker (ffmpeg)")
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.rescan, args.dry_run, args.limit, prepare=not args.no_convert)
    elif args.cmd == "convert":
        prepare_all(_db())
    else:
        list_tracks(args.review)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
