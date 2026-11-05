import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import build_retriever, build_service, create_app  # noqa: E402
from src.embeddings import VectorIndex  # noqa: E402
from src.qdrant_store import QdrantIndex  # noqa: E402
from src.retrieval import BM25Index, HybridRetriever  # noqa: E402

from tests.test_generation import ScriptedLLM, chat_server  # noqa: E402,F401


def _client(env=None, llm=None):
    service = build_service(env or {"RAG_RETRIEVER": "hybrid-hashing"})
    if llm is not None:
        service.pipeline.llm = llm
    return TestClient(create_app(service))


def test_health_reports_the_configuration():
    with _client() as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "chunks": 16, "retriever": "hybrid-hashing", "vector_store": "memory",
                    "llm": None}


def test_chat_answers_extractively_with_cited_sources():
    with _client() as client:
        r = client.post("/chat", json={"question": "What is the Wi-Fi password?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive" and body["fallback_reason"] == "no LLM configured"
    assert body["lang"] == "en" and body["citations"] == ["coworking_en#internet"]
    assert '"CoWork-Members"' in body["answer"]
    (source,) = body["sources"]
    assert source["id"] == "coworking_en#internet" and source["lang"] == "en" and "CoWork-Members" in source["text"]
    assert len(body["context"]) == 3 and body["latency_ms"] >= 0


def test_chat_answers_german_question_in_german_mode():
    with _client() as client:
        body = client.post("/chat", json={"question": "Kann ich die Mitgliedschaft pausieren, wenn ich krank bin?",
                                          "top_k": 1}).json()
    assert body["lang"] == "de" and body["citations"] == ["fitnessstudio_de#pausieren"]
    assert body["context"] == ["fitnessstudio_de#pausieren"]


def test_chat_uses_the_llm_and_drops_invented_citations():
    llm = ScriptedLLM("Das Passwort steht auf den Bildschirmen in der Küche [coworking_en#internet] [made#up].")
    with _client(llm=llm) as client:
        body = client.post("/chat", json={"question": "Wie lautet das WLAN-Passwort?"}).json()
    assert body["mode"] == "llm" and body["lang"] == "de" and body["fallback_reason"] is None
    assert body["citations"] == ["coworking_en#internet"]
    assert "Always answer in German" in llm.calls[0][0]["content"]


def test_chat_falls_back_when_the_llm_is_unreachable():
    env = {"RAG_RETRIEVER": "bm25", "RAG_LLM_BASE_URL": "http://127.0.0.1:9/v1", "RAG_LLM_MODEL": "qwen"}
    with _client(env) as client:
        assert client.get("/health").json()["llm"] == "qwen"
        body = client.post("/chat", json={"question": "What is the Wi-Fi password?"}).json()
    assert body["mode"] == "extractive" and body["fallback_reason"].startswith("LLM call failed: URLError")
    assert body["citations"] == ["coworking_en#internet"]


def test_chat_calls_an_openai_compatible_server(chat_server):  # noqa: F811
    with _client({"RAG_RETRIEVER": "bm25", "RAG_LLM_BASE_URL": chat_server}) as client:
        body = client.post("/chat", json={"question": "How much does a ride cost?"}).json()
    assert body["mode"] == "llm" and body["answer"] == "2 Euro [bike_rental_en#prices]"
    assert [s["id"] for s in body["sources"]] == ["bike_rental_en#prices"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 2001},
                                     {"question": "Preis?", "top_k": 0}, {"question": "Preis?", "top_k": 11}])
def test_invalid_requests_are_rejected(payload):
    with _client() as client:
        assert client.post("/chat", json=payload).status_code == 422


def test_build_retriever_without_qdrant_uses_the_in_memory_index():
    assert isinstance(build_retriever("bm25"), BM25Index)
    assert isinstance(build_retriever("hybrid-hashing").vector_index, VectorIndex)
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("tfidf")
    with pytest.raises(ValueError, match="no vectors"):
        build_retriever("bm25", qdrant_url="http://localhost:6333")


def test_service_can_store_vectors_in_qdrant(monkeypatch):
    qdrant_client = pytest.importorskip("qdrant_client")
    local = qdrant_client.QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: local)
    service = build_service({"RAG_RETRIEVER": "hybrid-hashing", "RAG_QDRANT_URL": "http://qdrant:6333"})
    assert service.vector_store == "qdrant"
    assert isinstance(service.pipeline.retriever, HybridRetriever)
    assert isinstance(service.pipeline.retriever.vector_index, QdrantIndex)
    assert local.count("chunks-hashing").count == 16
    with TestClient(create_app(service)) as client:
        body = client.post("/chat", json={"question": "What is the Wi-Fi password?"}).json()
    assert body["citations"] == ["coworking_en#internet"]
