"""FastAPI chat endpoint over the RAG pipeline.

    uvicorn src.api:app --port 8080
    curl -X POST localhost:8080/chat -H 'content-type: application/json' \\
        -d '{"question": "Wie lange darf ich einen Besprechungsraum buchen?"}'

Configuration (environment variables, all optional):

- RAG_RETRIEVER: bm25 | hashing | hybrid-hashing (default) | multilingual | hybrid-multilingual
- RAG_QDRANT_URL: e.g. http://qdrant:6333. If set, the dense part of the
  retriever is stored in and searched from Qdrant instead of in memory.
- RAG_TOP_K: chunks given to the answerer (default 3)
- RAG_DOCS: folder with the markdown corpus (default data/docs)
- RAG_LLM_BASE_URL, RAG_LLM_MODEL, RAG_LLM_API_KEY: OpenAI-compatible LLM
  (see src.generation). Without it, answers are extractive.

The index is built once at startup from the corpus.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from src.embeddings import HashingEmbedder
from src.evaluation import DOCS, RETRIEVERS
from src.generation import OpenAICompatibleClient, RAGPipeline
from src.ingest import load_corpus
from src.retrieval import HybridRetriever


def _multilingual_embedder():
    from src.embeddings import MultilingualEmbedder  # needs sentence-transformers
    return MultilingualEmbedder()


EMBEDDERS = {"hashing": HashingEmbedder, "multilingual": _multilingual_embedder}


def build_retriever(name: str, qdrant_url: str | None = None, connect_timeout: float = 30.0):
    """A retriever from evaluation.RETRIEVERS, with its vectors in Qdrant if qdrant_url is set."""
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}, choose from {sorted(RETRIEVERS)}")
    if not qdrant_url:
        return RETRIEVERS[name]()
    base = name.removeprefix("hybrid-")
    if base not in EMBEDDERS:
        raise ValueError(f"{name!r} has no vectors to store in Qdrant")
    from qdrant_client import QdrantClient

    from src.qdrant_store import QdrantIndex

    client = QdrantClient(url=qdrant_url)
    deadline = time.monotonic() + connect_timeout
    while True:  # the Qdrant container may still be starting
        try:
            client.get_collections()
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)
    index = QdrantIndex(EMBEDDERS[base](), client, collection=f"chunks-{base}")
    return HybridRetriever(index) if name.startswith("hybrid-") else index


@dataclass
class Service:
    pipeline: RAGPipeline
    retriever: str
    vector_store: str        # "qdrant" | "memory" | "none" (bm25)
    llm_model: str | None


def build_service(env: dict[str, str] | None = None) -> Service:
    env = dict(os.environ if env is None else env)
    name = env.get("RAG_RETRIEVER", "hybrid-hashing")
    qdrant_url = env.get("RAG_QDRANT_URL") or None
    chunks = load_corpus(env.get("RAG_DOCS", str(DOCS)))
    retriever = build_retriever(name, qdrant_url)
    retriever.add([c.id for c in chunks], [f"{c.title} {c.section} {c.text}" for c in chunks])
    llm = None
    if env.get("RAG_LLM_BASE_URL"):
        llm = OpenAICompatibleClient(env["RAG_LLM_BASE_URL"], env.get("RAG_LLM_MODEL", "default"),
                                     env.get("RAG_LLM_API_KEY"))
    store = "none" if name == "bm25" else ("qdrant" if qdrant_url else "memory")
    pipeline = RAGPipeline(retriever, chunks, llm, top_k=int(env.get("RAG_TOP_K", "3")))
    return Service(pipeline, name, store, llm.model if llm else None)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question is blank")
        return v.strip()


class Source(BaseModel):
    id: str
    title: str
    section: str
    lang: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    lang: str
    mode: str
    citations: list[str]
    sources: list[Source]          # the cited chunks, in citation order
    context: list[str]             # every chunk id the answerer was given
    fallback_reason: str | None
    latency_ms: float


def create_app(service: Service | None = None) -> FastAPI:
    """With service=None the service is built from the environment at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service()
        yield

    app = FastAPI(title="Multilingual RAG Chatbot", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        svc: Service = request.app.state.service
        return {"status": "ok", "chunks": len(svc.pipeline.chunks), "retriever": svc.retriever,
                "vector_store": svc.vector_store, "llm": svc.llm_model}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        svc: Service = request.app.state.service
        start = time.perf_counter()
        result = svc.pipeline.ask(body.question, top_k=body.top_k)
        chunks = svc.pipeline.chunks
        sources = [Source(id=c.id, title=c.title, section=c.section, lang=c.lang, text=c.text)
                   for c in (chunks[cid] for cid in result.citations)]
        return ChatResponse(answer=result.text, lang=result.lang, mode=result.mode, citations=result.citations,
                            sources=sources, context=result.context, fallback_reason=result.fallback_reason,
                            latency_ms=round((time.perf_counter() - start) * 1000, 2))

    return app


app = create_app()
