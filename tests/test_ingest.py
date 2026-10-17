from pathlib import Path

from src.ingest import load_corpus, split_markdown
from src.language import detect_language

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_language_detection():
    assert detect_language("How much does a bike ride cost after 30 minutes?") == "en"
    assert detect_language("Wie lange kann ich ein Buch ausleihen?") == "de"
    assert detect_language("1234 !!") == "unknown"


def test_split_markdown_one_chunk_per_section():
    md = "# Title\n\n## First\nAlpha text.\n\n## Second\nBeta text."
    chunks = split_markdown("doc", md)
    assert [c.section for c in chunks] == ["First", "Second"]
    assert chunks[0].id == "doc#first" and chunks[0].title == "Title"


def test_long_section_is_split_on_sentences():
    body = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = split_markdown("doc", f"# T\n## Long\n{body}", max_chars=300)
    assert len(chunks) > 1 and all(len(c.text) <= 330 for c in chunks)
    assert chunks[0].id == "doc#long-1"


def test_corpus_has_both_languages():
    chunks = load_corpus(DOCS)
    langs = {c.lang for c in chunks}
    assert langs == {"en", "de"}
    assert len(chunks) == 16
