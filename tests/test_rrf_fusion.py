import pytest
from src.engine.rrf_fusion import ReciprocalRankFusion

def test_rrf_fusion_logic():
    rrf = ReciprocalRankFusion(k_constant=60, dense_weight=0.5, sparse_weight=0.5)
    
    dense_hits = [({"doc_id": "D1"}, 0.95), ({"doc_id": "D2"}, 0.70)]
    sparse_hits = [({"doc_id": "D2"}, 4.2), ({"doc_id": "D1"}, 1.1)]
    
    fused = rrf.fuse(dense_hits, sparse_hits, top_k=2)
    assert len(fused) == 2
    # Both documents should have positive RRF scores
    assert fused[0][1] > 0
    assert fused[1][1] > 0
