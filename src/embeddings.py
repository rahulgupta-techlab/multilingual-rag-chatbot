"""Embedding backends and an in-memory vector index.

- HashingEmbedder: deterministic, offline, no model download. Used in tests and
  CI. It is lexical, so it cannot match across languages.
- MultilingualEmbedder: sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2`,
  which maps EN and DE (and 50+ other languages) into one vector space. This is
  what makes cross-lingual retrieval work.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-zäöüß0-9]+", text.lower())
        grams = [w[i:i + 3] for w in words for i in range(max(len(w) - 2, 1))]
        return words + grams

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                m[row, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(m)


class MultilingualEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class VectorIndex:
    """Exact cosine-similarity search over normalised vectors.

    Fine for thousands of chunks; the same interface is later backed by Qdrant.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.ids: list[str] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        vecs = self.embedder.embed(texts)
        self.matrix = vecs if not self.ids else np.vstack([self.matrix, vecs])
        self.ids.extend(ids)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]
