import os

try:
    from pydantic_settings import BaseSettings
    class RAGSettings(BaseSettings):
        PROJECT_NAME: str = "Enterprise High-Throughput Hybrid RAG Engine"
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        BM25_K1: float = 1.5
        BM25_B: float = 0.75
        RRF_K_CONSTANT: int = 60
        DENSE_WEIGHT: float = 0.50
        SPARSE_WEIGHT: float = 0.50
        TOP_K_RETRIEVAL: int = 5
        RERANK_TOP_N: int = 3
        CONFIDENCE_THRESHOLD: float = 0.70
        ENABLE_CRAG_FALLBACK: bool = True
        class Config:
            case_sensitive = True
    settings = RAGSettings()
except ImportError:
    class RAGSettings:
        PROJECT_NAME: str = "Enterprise High-Throughput Hybrid RAG Engine"
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        BM25_K1: float = 1.5
        BM25_B: float = 0.75
        RRF_K_CONSTANT: int = 60
        DENSE_WEIGHT: float = 0.50
        SPARSE_WEIGHT: float = 0.50
        TOP_K_RETRIEVAL: int = 5
        RERANK_TOP_N: int = 3
        CONFIDENCE_THRESHOLD: float = 0.70
        ENABLE_CRAG_FALLBACK: bool = True
    settings = RAGSettings()
