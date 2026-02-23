from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import requests

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "persona_docs"


def ollama_embed(text: str) -> List[float]:
    # 兼容不同 Ollama 版本：优先 /api/embed，不行再 /api/embeddings
    r = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=120,
    )
    if r.status_code == 404:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
    r.raise_for_status()
    data = r.json()
    if "embedding" in data:
        return data["embedding"]
    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    raise RuntimeError(f"Unexpected Ollama embed response: {data}")


@dataclass
class QdrantStore:
    url: str = "http://localhost:6333"

    def __post_init__(self):
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self, dim: int):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing:
            return
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def reset_collection(self, dim: int):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing:
            self.client.delete_collection(collection_name=COLLECTION)
        self.ensure_collection(dim)

    def upsert_chunks(self, points: List[qm.PointStruct]):
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ✅ 这里开始：你要改的“返回带 id 的 search()”
    def search(self, query: str, top_k: int = 8) -> List[Tuple[int, str, Dict[str, Any], float]]:
        qv = ollama_embed(query)

        # 新版 / 兼容版：query_points
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=COLLECTION,
                query=qv,             # 向量
                limit=top_k,
                with_payload=True,
            )
            hits = res.points
            out = []
            for h in hits:
                payload = h.payload or {}
                text = payload.get("text", "")
                meta = payload.get("meta", {})
                score = float(h.score)  # cosine 相似度，越大越相关
                out.append((int(h.id), text, meta, score))
            return out

        # 老版：search（如果你的 client 有这个方法）
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=COLLECTION,
                query_vector=qv,
                limit=top_k,
                with_payload=True,
            )
            out = []
            for h in hits:
                payload = h.payload or {}
                text = payload.get("text", "")
                meta = payload.get("meta", {})
                score = float(h.score)
                out.append((int(h.id), text, meta, score))
            return out

        raise RuntimeError("Your qdrant-client has neither query_points nor search. Please upgrade qdrant-client.")
