"""Tiny, dependency-free language identification for English vs German.

Good enough to route answers into the user's language; swap in fastText
lid.176 or `langdetect` when more languages are added.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "en": {"the", "and", "is", "are", "of", "to", "in", "for", "you", "can", "what", "how", "my", "do",
           "does", "with", "on", "it", "be", "this", "after", "when", "per", "your", "at", "from"},
    "de": {"der", "die", "das", "und", "ist", "sind", "von", "zu", "im", "für", "ich", "kann", "wie", "was",
           "mein", "meine", "nicht", "mit", "auf", "es", "wird", "nach", "wann", "pro", "bei", "ein", "eine"},
}
_GERMAN_CHARS = re.compile(r"[äöüß]", re.I)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    if not words:
        return "unknown"
    scores = {lang: sum(w in sw for w in words) for lang, sw in _STOPWORDS.items()}
    scores["de"] += 2 * len(_GERMAN_CHARS.findall(text))
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"
