from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IngestDocument(BaseModel):
    doc_id: str = Field(..., example="DOC-001")
    title: str = Field(..., example="Clinical Protocols for Heart Failure")
    category: str = Field(..., example="Cardiology")
    content: str = Field(..., example="Detailed clinical guidance...")
    tags: List[str] = Field(default_factory=list, example=["Cardiology", "Guidelines"])

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the door-to-balloon PCI time for STEMI patients?")
    top_k: int = Field(default=5, ge=1, le=20)
    top_n_rerank: int = Field(default=3, ge=1, le=10)

class CitationModel(BaseModel):
    citation_index: int
    doc_id: str
    title: str
    relevance_score: float
    category: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    retrieved_chunks: List[Dict[str, Any]]
    crag_correction: Dict[str, Any]
    telemetry: Dict[str, float]
