from __future__ import annotations
from typing import List
from sentence_transformers import CrossEncoder

# 这是非常常用的中文/多语 reranker（HF 上下载）
# 如果你 HF 网络有问题：export HF_ENDPOINT=https://hf-mirror.com
MODEL_NAME = "BAAI/bge-reranker-base"

_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker

def rerank(query: str, passages: List[str]) -> List[float]:
    ce = get_reranker()
    pairs = [(query, p) for p in passages]
    scores = ce.predict(pairs)
    return [float(s) for s in scores]
