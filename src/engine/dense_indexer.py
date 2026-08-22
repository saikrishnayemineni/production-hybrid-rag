import math
import re
from typing import List, Dict, Any, Tuple

class DenseSemanticIndexer:
    """
    High-speed dense semantic vector indexer utilizing character n-gram
    and sub-word token semantic projections for lightweight zero-dependency cosine retrieval.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.documents: List[Dict[str, Any]] = []
        self.doc_vectors: List[List[float]] = []

    def _embed(self, text: str) -> List[float]:
        tokens = re.findall(r'\w+', text.lower())
        vec = [0.0] * self.embedding_dim
        if not tokens:
            return vec
        for token in tokens:
            # Deterministic hash projection into embedding dimensions
            h = hash(token)
            idx1 = abs(h) % self.embedding_dim
            idx2 = abs(h >> 4) % self.embedding_dim
            vec[idx1] += 1.0
            vec[idx2] += 0.5
            # Character n-grams for morphological sub-word semantics
            for i in range(len(token) - 2):
                ng_h = hash(token[i:i+3])
                vec[abs(ng_h) % self.embedding_dim] += 0.25

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def index_documents(self, docs: List[Dict[str, Any]]):
        self.documents = docs
        self.doc_vectors = [self._embed(f"{d.get('title', '')} {d.get('content', '')} {' '.join(d.get('tags', []))}") for d in docs]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = self._embed(query)
        scores: List[Tuple[Dict[str, Any], float]] = []
        for doc, d_vec in zip(self.documents, self.doc_vectors):
            # Cosine similarity between normalized vectors
            sim = sum(q * d for q, d in zip(q_vec, d_vec))
            scores.append((doc, round(sim, 4)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
