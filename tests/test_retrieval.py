from pathlib import Path

from src.embeddings import HashingEmbedder, VectorIndex
from src.ingest import load_corpus
from src.retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion, tokenize

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def _texts(chunks):
    return [c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks]


def test_tokenize_folds_umlauts_and_drops_stopwords():
    assert tokenize("Die Gebühr für das Ausleihen") == ["gebuehr", "ausleihen"]


def test_bm25_finds_exact_identifier():
    ids, texts = _texts(load_corpus(DOCS))
    bm25 = BM25Index()
    bm25.add(ids, texts)
    assert bm25.search("CoWork-Members", top_k=1)[0][0] == "coworking_en#internet"


def test_bm25_returns_nothing_for_unknown_terms():
    bm25 = BM25Index()
    bm25.add(["a"], ["bike rental prices"])
    assert bm25.search("quantum chromodynamics") == []


def test_rrf_rewards_documents_ranked_by_both_lists():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert max(fused, key=fused.get) == "b"
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert fused["d"] < fused["a"]


def test_hybrid_combines_lexical_and_dense_hits():
    ids, texts = _texts(load_corpus(DOCS))
    hybrid = HybridRetriever(VectorIndex(HashingEmbedder()))
    hybrid.add(ids, texts)
    assert hybrid.search("Wi-Fi network CoWork-Members password", top_k=1)[0][0] == "coworking_en#internet"
    assert hybrid.search("Mindestlaufzeit der Mitgliedschaft", top_k=1)[0][0] == "fitnessstudio_de#vertragslaufzeit"
