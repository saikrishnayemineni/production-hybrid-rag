import pytest
from src.engine.cross_encoder import CrossEncoderReranker
from src.engine.query_corrector import CorrectiveRAGQueryTransformer

def test_cross_encoder_scoring():
    reranker = CrossEncoderReranker()
    candidates = [
        ({"doc_id": "D1", "title": "STEMI Protocol", "category": "Cardiology", "content": "Door-to-balloon PCI within 90 min", "tags": ["STEMI"]}, 0.016, {"dense_rank": 1, "sparse_rank": 1})
    ]
    reranked = reranker.rerank("STEMI PCI door to balloon", candidates, top_n=1)
    assert len(reranked) == 1
    assert reranked[0]["cross_encoder_score"] > 0.50

def test_crag_query_transformation():
    corrector = CorrectiveRAGQueryTransformer()
    res = corrector.evaluate_and_transform("stemi time", top_confidence=0.40, threshold=0.70)
    assert res["was_transformed"] is True
    assert "door-to-balloon" in res["transformed_query"]
