"""Qdrant-backed vector index with the same interface as VectorIndex.

VectorIndex keeps every vector in one numpy matrix, which is fine for this
corpus but lives and dies with the process. QdrantIndex stores the vectors in
a Qdrant collection instead (cosine distance), so the API can run against a
real vector database. It takes the same embedders and plugs into
HybridRetriever unchanged.

Point ids must be integers or UUIDs, so each chunk id is mapped to a UUID5
and the chunk id itself is kept in the payload.
"""
from __future__ import annotations

import uuid
from typing import Sequence

from src.embeddings import Embedder

_NAMESPACE = uuid.UUID("6f9c1d2e-4b7a-4c55-9e3f-2a8d0b1c7e61")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantIndex:
    """Cosine search over a Qdrant collection.

    `client` is a qdrant_client.QdrantClient: QdrantClient(url="http://qdrant:6333")
    for a server, QdrantClient(":memory:") for tests. With recreate=True (the
    default) the collection is dropped and rebuilt on the first add, so the
    index always matches the corpus it was built from.
    """

    def __init__(self, embedder: Embedder, client, collection: str = "chunks", recreate: bool = True,
                 batch_size: int = 64):
        self.embedder, self.client, self.collection = embedder, client, collection
        self.recreate, self.batch_size = recreate, batch_size
        self._ready = False

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and self.recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        self._ready = True

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        from qdrant_client import models

        if not ids:
            return
        vecs = self.embedder.embed(texts)
        if not self._ready:
            self._ensure_collection(vecs.shape[1])
        for start in range(0, len(ids), self.batch_size):
            points = [models.PointStruct(id=point_id(cid), vector=vec.tolist(), payload={"chunk_id": cid})
                      for cid, vec in zip(ids[start:start + self.batch_size], vecs[start:start + self.batch_size])]
            self.client.upsert(self.collection, points=points, wait=True)

    def __len__(self) -> int:
        if not self._ready:
            return 0
        return self.client.count(self.collection, exact=True).count

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._ready:
            return []
        q = self.embedder.embed([query])[0]
        hits = self.client.query_points(self.collection, query=q.tolist(), limit=top_k, with_payload=True).points
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]
