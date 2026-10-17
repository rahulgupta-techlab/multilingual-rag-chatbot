"""Load markdown documents and split them into section-level chunks.

Chunks follow the document's own structure (one chunk per `##` section) so
every retrieved passage is a self-contained answer unit and can be cited by
document and section title.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.language import detect_language


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc>#<section-slug>"
    doc: str         # file stem
    title: str       # document title (# heading)
    section: str     # section heading (## heading)
    text: str
    lang: str        # "en" | "de" | "unknown"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def split_markdown(doc: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    title_m = re.search(r"^#\s+(.+)$", markdown, re.M)
    title = title_m.group(1).strip() if title_m else doc
    parts = re.split(r"^##\s+(.+)$", markdown, flags=re.M)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    chunks: list[Chunk] = []
    for heading, body in zip(parts[1::2], parts[2::2]):
        body = " ".join(body.split())
        if not body:
            continue
        # very long sections are split on sentence boundaries
        pieces, buf = [], ""
        for sent in re.split(r"(?<=[.!?])\s+", body):
            if buf and len(buf) + len(sent) > max_chars:
                pieces.append(buf); buf = ""
            buf = f"{buf} {sent}".strip()
        pieces.append(buf)
        for i, piece in enumerate(pieces):
            suffix = f"-{i + 1}" if len(pieces) > 1 else ""
            chunks.append(Chunk(f"{doc}#{_slug(heading)}{suffix}", doc, title, heading.strip(), piece,
                                detect_language(f"{heading} {piece}")))
    return chunks


def load_corpus(folder: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(split_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks
