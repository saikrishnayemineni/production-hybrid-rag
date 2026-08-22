from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.engine.hybrid_pipeline import hybrid_rag_pipeline
from src.api.models import QueryRequest, QueryResponse, IngestDocument

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production High-Throughput Hybrid RAG Engine with BM25, Dense Vectors, RRF, and Cross-Encoder Reranking.",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "indexed_documents": len(hybrid_rag_pipeline.docs)
    }

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Hybrid RAG"])
async def execute_hybrid_query(request: QueryRequest):
    result = hybrid_rag_pipeline.query(
        query_text=request.query,
        top_k=request.top_k,
        top_n_rerank=request.top_n_rerank
    )
    return result

@app.post("/api/v1/ingest", tags=["Document Ingestion"])
async def ingest_documents(documents: list[IngestDocument]):
    current_docs = list(hybrid_rag_pipeline.docs)
    for doc in documents:
        current_docs.append(doc.dict())
    hybrid_rag_pipeline.index(current_docs)
    return {
        "status": "success",
        "added_count": len(documents),
        "total_documents": len(hybrid_rag_pipeline.docs)
    }
