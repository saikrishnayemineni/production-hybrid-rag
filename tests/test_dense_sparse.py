import pytest
from src.engine.dense_indexer import DenseSemanticIndexer
from src.engine.sparse_bm25 import SparseBM25Indexer

DOCS = [
    {"doc_id": "D1", "title": "Acute STEMI Guidelines", "content": "12-lead ECG within 10 minutes for STEMI ACS.", "tags": ["STEMI"]},
    {"doc_id": "D2", "title": "Sepsis Bundle", "content": "qSOFA criteria for severe sepsis and lactate testing.", "tags": ["Sepsis"]}
]

def test_dense_semantic_search():
    indexer = DenseSemanticIndexer(embedding_dim=64)
    indexer.index_documents(DOCS)
    hits = indexer.search("heart attack cardiac emergency", top_k=1)
    assert len(hits) == 1
    assert hits[0][0]["doc_id"] in ["D1", "D2"]

def test_sparse_bm25_acronym_search():
    bm25 = SparseBM25Indexer(k1=1.5, b=0.75)
    bm25.index_documents(DOCS)
    hits = bm25.search("STEMI", top_k=1)
    assert len(hits) == 1
    assert hits[0][0]["doc_id"] in ["D1", "D2"]
    assert hits[0][1] > 0
