"""Lexical (BM25) and hybrid retrieval.

Dense multilingual embeddings handle paraphrases and cross-lingual questions,
but they are weak on exact tokens such as network names, product codes or
numbers. BM25 is the opposite. HybridRetriever runs both and merges the two
rankings with Reciprocal Rank Fusion (RRF), which only uses ranks, so the
very different score scales of BM25 and cosine similarity never need to be
calibrated against each other.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from src.embeddings import VectorIndex

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})

# Small EN/DE stopword list: enough to stop function words from dominating BM25.
STOPWORDS = frozenset("""
a an and are as at be by can do does for from how i in is it of on or the to
what when where which who why with you your
der die das den dem des ein eine einen einem einer und oder ist sind wie was
wann wo wer mit von zu im in am an auf fuer bei nach ich man kann muss es
""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().translate(_UMLAUTS))
    return [w for w in words if w not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over pre-tokenised documents.

    k1 controls term-frequency saturation, b controls document-length
    normalisation. The IDF uses the "+1" variant so it is never negative.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        for doc_id, text in zip(ids, texts):
            tokens = tokenize(text)
            self.ids.append(doc_id)
            self.docs.append(Counter(tokens))
            self.lengths.append(len(tokens))
            self.df.update(set(tokens))

    def _idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        avg_len = sum(self.lengths) / len(self.lengths) or 1.0
        terms = tokenize(query)
        scores = []
        for tf, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / avg_len)
                    s += self._idf(t) * f * (self.k1 + 1) / norm
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.ids[i], scores[i]) for i in order[:top_k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self, vector_index: VectorIndex, bm25: BM25Index | None = None, rrf_k: int = 60,
                 candidates: int = 20):
        self.vector_index = vector_index
        self.bm25 = bm25 or BM25Index()
        self.rrf_k = rrf_k
        self.candidates = candidates

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        self.vector_index.add(ids, texts)
        self.bm25.add(ids, texts)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        dense = [doc_id for doc_id, _ in self.vector_index.search(query, self.candidates)]
        lexical = [doc_id for doc_id, _ in self.bm25.search(query, self.candidates)]
        return reciprocal_rank_fusion([dense, lexical], k=self.rrf_k)[:top_k]
