"""Chunking rules from docs/cortex.md."""
import re

from conftest import page_text

from core.chunk import MAX_CHARS, MIN_CHARS, OVERLAP_MAX, chunk_page_text, chunk_pages
from core.parse import Page, parse


def _pages(n=3, sentences=40):
    return [Page(page=p, text=page_text(p, sentences)) for p in range(1, n + 1)]


def test_never_crosses_a_page_and_indexes_are_sequential():
    pages = _pages(4)
    chunks = chunk_pages(pages)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    by_page = {p.page: re.sub(r"\s+", " ", p.text) for p in pages}
    for c in chunks:
        # every sentence of the chunk comes from the page it is labelled with
        assert f"page {c.page} " in c.content or c.content.startswith("Chapter")
        for sentence in re.split(r"(?<=\.)\s+", c.content):
            assert sentence in by_page[c.page]
    # reading order: pages never go backwards
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)


def test_max_length_and_whole_sentences():
    chunks = chunk_pages(_pages(2, 60))
    assert len(chunks) > 4
    for c in chunks:
        assert len(c.content) <= MAX_CHARS
        assert c.content.endswith(".")          # packed whole sentences
        assert c.content[0].isupper()


def test_overlap_repeats_last_sentence():
    chunks = [c for c in chunk_pages(_pages(1, 60))]
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:]):
        last = re.split(r"(?<=\.)\s+", a.content)[-1]
        assert len(last) <= OVERLAP_MAX
        assert b.content.startswith(last)


def test_no_overlap_when_last_sentence_is_long():
    long_sentence = "Long " + "word " * 60 + "end."          # ~305 chars > OVERLAP_MAX
    text = " ".join([long_sentence] * 6)
    out = chunk_page_text(text)
    assert len(out) >= 2
    for a, b in zip(out, out[1:]):
        assert not b.startswith(a[-50:])
        assert len(a) <= MAX_CHARS


def test_overlong_sentence_is_split_hard():
    words = " ".join(f"w{i}" for i in range(1500))            # ~8 KB, no full stop
    out = chunk_page_text(words)
    assert len(out) >= 6
    assert all(len(c) <= MAX_CHARS for c in out)
    assert " ".join(out).split() == words.split()              # nothing lost, nothing added
    blob = "x" * 5000                                          # no spaces at all
    out = chunk_page_text(blob)
    assert [len(c) for c in out] == [1200, 1200, 1200, 1200, 200]


def test_long_table_is_split_at_lines():
    rows = "\n".join(f"Row {i} | value {i * 3} | note about item {i}" for i in range(200))
    out = chunk_page_text(rows)
    assert all(len(c) <= MAX_CHARS for c in out)
    assert all(c.startswith("Row ") for c in out)              # cut between lines, not inside one


def test_tiny_chunks_dropped_unless_page_has_nothing_else():
    big = "A" + "b" * 1196 + "."                                # 1198 chars: next unit won't fit
    out = chunk_page_text(big + " Ok fine.")
    assert out == [big]                                         # 'Ok fine.' (< 40) dropped
    assert chunk_page_text("Ok fine.") == ["Ok fine."]          # alone on its page: kept
    assert all(len(c.content) >= MIN_CHARS for c in chunk_pages(_pages(3)))


def test_deterministic():
    pages = _pages(5)
    assert chunk_pages(pages) == chunk_pages(list(pages))


def test_paragraphs_are_kept_apart():
    text = "First paragraph has one sentence here.\n\nSecond paragraph starts here. It goes on."
    assert chunk_page_text(text) == ["First paragraph has one sentence here.\nSecond paragraph starts here. It goes on."]


def test_pdf_pages_txt_and_docx_have_no_pages():
    from conftest import make_pdf

    parsed = parse("notes.pdf", make_pdf(3, 10))
    assert parsed.page_count == 3
    assert [p.page for p in parsed.pages] == [1, 2, 3]
    chunks = chunk_pages(parsed.pages)
    assert {c.page for c in chunks} == {1, 2, 3}

    t = parse("a.txt", "Hello there, this is a plain text note about osmosis.\x00".encode())
    assert t.page_count is None and t.pages[0].page is None and "\x00" not in t.pages[0].text

    import io

    import docx

    d = docx.Document()
    d.add_paragraph("Docx paragraph about enzymes and their active sites.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Cell A", "Cell B"
    buf = io.BytesIO()
    d.save(buf)
    p = parse("b.docx", buf.getvalue())
    assert p.pages[0].page is None
    assert "Cell A | Cell B" in p.pages[0].text
