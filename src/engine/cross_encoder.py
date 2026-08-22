import re
from typing import List, Dict, Any, Tuple

class CrossEncoderReranker:
    """
    Simulates cross-attention deep reranking scoring the joint interaction
    between the input query and candidate context chunks.
    """
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float, Dict[str, Any]]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))

        for doc, rrf_score, rank_meta in candidates:
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            doc_tokens = set(re.findall(r'\b\w+\b', text))
            
            # Compute token overlap & n-gram cohesion
            overlap = len(q_tokens.intersection(doc_tokens))
            jaccard = overlap / len(q_tokens.union(doc_tokens)) if q_tokens.union(doc_tokens) else 0.0
            
            # Phrase proximity bonus
            proximity_bonus = 0.2 if any(qt in text for qt in q_tokens if len(qt) > 4) else 0.0
            
            # Joint cross-encoder relevance score [0.0 - 1.0]
            cross_score = min(1.0, round((jaccard * 1.5) + (rrf_score * 35.0) + proximity_bonus, 4))
            
            reranked.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "tags": doc["tags"],
                "cross_encoder_score": cross_score,
                "rrf_score": rrf_score,
                "dense_rank": rank_meta.get("dense_rank"),
                "sparse_rank": rank_meta.get("sparse_rank")
            })

        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked[:top_n]
