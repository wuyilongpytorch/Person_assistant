import pickle
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

def tokenize_zh(s: str) -> List[str]:
    return [w.strip() for w in jieba.lcut(s) if w.strip()]

class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.ids: List[int] = []
        self.texts: List[str] = []
        self.metas: List[Dict] = []

    def build(self, ids: List[int], texts: List[str], metas: List[Dict]):
        self.ids, self.texts, self.metas = ids, texts, metas
        corpus = [tokenize_zh(t) for t in texts]
        self.bm25 = BM25Okapi(corpus)

    def query(self, q: str, top_k: int = 30) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_zh(q))
        pairs = list(zip(self.ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
