"""Split parsed pages into small, citable chunks (rules: docs/cortex.md, "Chunking").

- A chunk never crosses a page, so every citation has an exact page.
- Whole sentences/paragraphs are packed up to MAX_CHARS; the last sentence of a chunk (if it is
  at most OVERLAP_MAX chars) is repeated at the start of the next one.
- A sentence longer than MAX_CHARS is split at its line breaks, then hard at MAX_CHARS
  (at the last space in the second half of the window when there is one).
- Chunks under MIN_CHARS are dropped unless the page has nothing else.
- Pure and deterministic: the same pages always give the same chunks (ingest is resumable).
"""
import re
from dataclasses import dataclass

from .parse import Page

# Bump whenever parsing or chunking output changes: ingest wipes and redoes a half-finished
# document whose stored chunks came from another version (chunk_index must mean the same thing).
CHUNKER_VERSION = 1

MAX_CHARS = 1200
OVERLAP_MAX = 200
MIN_CHARS = 40

_PARAGRAPHS = re.compile(r"\n[ \t\r\f\v]*\n")
# After . ! ? or an ellipsis, optionally followed by a closing quote/bracket, then whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…][\"'”’)\]])\s+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    page: int | None
    content: str


def _squash(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def _hard_split(text: str) -> list[str]:
    """Pieces of at most MAX_CHARS; cut at a space when one falls in the second half."""
    pieces: list[str] = []
    rest = text
    while len(rest) > MAX_CHARS:
        cut = rest.rfind(" ", MAX_CHARS // 2, MAX_CHARS + 1)
        if cut <= 0:
            cut = MAX_CHARS
        head, rest = rest[:cut].strip(), rest[cut:].strip()
        if head:
            pieces.append(head)
    if rest:
        pieces.append(rest)
    return pieces


def _split_long(sentence: str) -> list[str]:
    """An over-long 'sentence' (tables, slides, lists): pack its lines, then hard split."""
    pieces: list[str] = []
    current = ""
    for line in sentence.split("\n"):
        line = _squash(line)
        if not line:
            continue
        if len(line) > MAX_CHARS:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(line))
            continue
        if current and len(current) + 1 + len(line) > MAX_CHARS:
            pieces.append(current)
            current = line
        else:
            current = f"{current} {line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def _units(text: str) -> list[tuple[str, bool]]:
    """(unit text, starts a new paragraph) in reading order; each unit is <= MAX_CHARS."""
    units: list[tuple[str, bool]] = []
    for para in _PARAGRAPHS.split(text):
        first = True
        for sentence in _SENTENCE_END.split(para.strip()):
            if not sentence.strip():
                continue
            flat = _squash(sentence)
            pieces = [flat] if len(flat) <= MAX_CHARS else _split_long(sentence)
            for piece in pieces:
                units.append((piece, first))
                first = False
    return units


def _pack(units: list[tuple[str, bool]]) -> list[str]:
    chunks: list[str] = []
    parts: list[str] = []   # unit texts in the current chunk
    body = ""               # the current chunk's text
    for text, new_para in units:
        if parts and len(body) + 1 + len(text) > MAX_CHARS:
            chunks.append(body)
            last = parts[-1]
            if len(parts) > 1 and len(last) <= OVERLAP_MAX and len(last) + 1 + len(text) <= MAX_CHARS:
                parts, body = [last], last
            else:
                parts, body = [], ""
        if parts:
            body += ("\n" if new_para else " ") + text
        else:
            body = text
        parts.append(text)
    if parts:
        chunks.append(body)
    return chunks


def chunk_page_text(text: str) -> list[str]:
    """The chunks for one page's text, tiny-chunk rule applied."""
    packed = [c.strip() for c in _pack(_units(text)) if c.strip()]
    kept = [c for c in packed if len(c) >= MIN_CHARS]
    return kept or packed  # a page with only a short line keeps it


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        for content in chunk_page_text(p.text):
            out.append(Chunk(chunk_index=len(out), page=p.page, content=content))
    return out
