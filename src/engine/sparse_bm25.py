import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

class SparseBM25Indexer:
    """
    Production-grade Okapi BM25 inverted index with inverse document frequency (IDF)
    and document length normalization, optimized for technical acronyms and exact keyword matching.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_term_counts: List[Counter] = []
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r'\b[\w\-]+\b', text)]

    def index_documents(self, docs: List[Dict[str, Any]]):
        self.documents = docs
        self.num_docs = len(docs)
        self.doc_term_counts = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        total_length = 0
        for doc in docs:
            full_text = f"{doc.get('title', '')} {doc.get('content', '')} {' '.join(doc.get('tags', []))}"
            tokens = self._tokenize(full_text)
            term_cnt = Counter(tokens)
            self.doc_term_counts.append(term_cnt)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            for term in term_cnt.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_length / self.num_docs if self.num_docs > 0 else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        # Standard probabilistic BM25 IDF formula
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = self._tokenize(query)
        scores: List[Tuple[Dict[str, Any], float]] = []

        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            term_cnt = self.doc_term_counts[idx]

            for term in q_tokens:
                if term not in term_cnt:
                    continue
                tf = term_cnt[term]
                idf = self._idf(term)
                # BM25 term weighting formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            scores.append((doc, round(score, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
