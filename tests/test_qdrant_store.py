import pytest

from src.embeddings import HashingEmbedder, VectorIndex
from src.evaluation import DOCS, load_questions
from src.ingest import load_corpus
from src.qdrant_store import QdrantIndex, point_id
from src.retrieval import HybridRetriever

qdrant_client = pytest.importorskip("qdrant_client")

CHUNKS = load_corpus(DOCS)
IDS, TEXTS = [c.id for c in CHUNKS], [f"{c.title} {c.section} {c.text}" for c in CHUNKS]


def _qdrant_index(**kwargs):
    return QdrantIndex(HashingEmbedder(), qdrant_client.QdrantClient(":memory:"), **kwargs)


def test_point_ids_are_stable_uuids():
    assert point_id("coworking_en#internet") == point_id("coworking_en#internet")
    assert point_id("coworking_en#internet") != point_id("coworking_en#access")


def test_qdrant_ranks_like_the_in_memory_index():
    memory, qdrant = VectorIndex(HashingEmbedder()), _qdrant_index()
    memory.add(IDS, TEXTS)
    qdrant.add(IDS, TEXTS)
    assert len(qdrant) == len(IDS)
    for q in load_questions():
        expected, got = memory.search(q.question, top_k=5), qdrant.search(q.question, top_k=5)
        assert [cid for cid, _ in got] == [cid for cid, _ in expected], q.id
        assert [s for _, s in got] == pytest.approx([s for _, s in expected], abs=1e-4)


def test_empty_index_returns_nothing():
    assert _qdrant_index().search("Wi-Fi password") == []


def test_rebuild_replaces_the_old_collection():
    client = qdrant_client.QdrantClient(":memory:")
    first = QdrantIndex(HashingEmbedder(), client)
    first.add(IDS, TEXTS)
    second = QdrantIndex(HashingEmbedder(), client)
    second.add(IDS[:3], TEXTS[:3])
    assert len(second) == 3


def test_hybrid_retriever_accepts_the_qdrant_index():
    hybrid = HybridRetriever(_qdrant_index())
    hybrid.add(IDS, TEXTS)
    assert hybrid.search("CoWork-Members Wi-Fi", top_k=1)[0][0] == "coworking_en#internet"
