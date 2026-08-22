from typing import List, Dict, Any, Tuple

class ReciprocalRankFusion:
    """
    Mathematical Reciprocal Rank Fusion (RRF) aggregating ranked lists
    from heterogeneous search channels (Dense Vector + Sparse BM25).
    """
    def __init__(self, k_constant: int = 60, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        self.k_constant = k_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        rank_details: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks
        for rank, (doc, raw_score) in enumerate(dense_results, start=1):
            doc_id = doc["doc_id"]
            doc_map[doc_id] = doc
            rrf_val = self.dense_weight * (1.0 / (self.k_constant + rank))
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_val
            rank_details[doc_id] = {
                "dense_rank": rank,
                "dense_raw": raw_score,
                "sparse_rank": None,
                "sparse_raw": 0.0
            }

        # Process Sparse Ranks
        for rank, (doc, raw_score) in enumerate(sparse_results, start=1):
            doc_id = doc["doc_id"]
            doc_map[doc_id] = doc
            rrf_val = self.sparse_weight * (1.0 / (self.k_constant + rank))
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_val
            if doc_id not in rank_details:
                rank_details[doc_id] = {
                    "dense_rank": None,
                    "dense_raw": 0.0,
                    "sparse_rank": rank,
                    "sparse_raw": raw_score
                }
            else:
                rank_details[doc_id]["sparse_rank"] = rank
                rank_details[doc_id]["sparse_raw"] = raw_score

        # Sort aggregated results
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_output = []
        for doc_id, final_score in sorted_docs[:top_k]:
            fused_output.append((doc_map[doc_id], round(final_score, 6), rank_details[doc_id]))

        return fused_output
