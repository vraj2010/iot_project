"""Extract text page by page from PDF, DOCX and TXT (same rules as Cortex/backend/app/services/parser.py).

PyMuPDF and python-docx are imported inside the functions: only /api/ingest pays for them.
"""
import io
from dataclasses import dataclass
from pathlib import PurePath

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    page: int | None  # 1-based; None when the format has no real pages (DOCX, TXT)
    text: str


@dataclass(frozen=True)
class Parsed:
    pages: list[Page]        # only pages with text
    page_count: int | None   # PDFs: every page, including empty ones; else None


def extension(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def _clean(text: str) -> str:
    # Postgres TEXT cannot store NUL bytes, which some PDFs contain.
    return text.replace("\x00", "").strip()


def parse_pdf(data: bytes) -> Parsed:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [Page(page=i + 1, text=_clean(p.get_text())) for i, p in enumerate(doc)]
        count = doc.page_count
    return Parsed(pages=[p for p in pages if p.text], page_count=count)


def parse_docx(data: bytes) -> Parsed:
    """Paragraphs and tables in reading order. Blocks are separated by blank lines so a heading
    never runs into the next sentence; each table row becomes one 'a | b | c' line."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    blocks: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, d).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            rows = []
            for row in Table(child, d).rows:
                cells, seen = [], set()
                for cell in row.cells:  # merged cells repeat: keep each once
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    cells.append(" ".join(cell.text.split()))
                line = " | ".join(cells).strip(" |")
                if line:
                    rows.append(line)
            if rows:
                blocks.append("\n".join(rows))
    text = _clean("\n\n".join(blocks))
    return Parsed(pages=[Page(page=None, text=text)] if text else [], page_count=None)


def parse_txt(data: bytes) -> Parsed:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = _clean(text)
    return Parsed(pages=[Page(page=None, text=text)] if text else [], page_count=None)


def parse(filename: str, data: bytes) -> Parsed:
    ext = extension(filename)
    if ext == ".pdf":
        return parse_pdf(data)
    if ext == ".docx":
        return parse_docx(data)
    if ext == ".txt":
        return parse_txt(data)
    raise UnsupportedFileType(f"Unsupported file type: {ext or 'none'}")
