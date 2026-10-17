from pathlib import Path

import numpy as np

from src.embeddings import HashingEmbedder, VectorIndex
from src.ingest import load_corpus

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["bike rental prices"]), e.embed(["bike rental prices"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_empty_index_returns_nothing():
    assert VectorIndex(HashingEmbedder()).search("anything") == []


def test_same_language_retrieval_finds_right_section():
    chunks = load_corpus(DOCS)
    index = VectorIndex(HashingEmbedder())
    index.add([c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks])
    top_en = index.search("how much does a bike ride cost after the free minutes", top_k=1)[0][0]
    top_de = index.search("Wie hoch ist die Gebühr bei Überschreitung der Leihfrist?", top_k=1)[0][0]
    assert top_en == "bike_rental_en#prices"
    assert top_de == "stadtbibliothek_de#gebuehren"
