import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.core.config import settings
from src.core.telemetry import LatencyProfiler
from src.engine.dense_indexer import DenseSemanticIndexer
from src.engine.sparse_bm25 import SparseBM25Indexer
from src.engine.rrf_fusion import ReciprocalRankFusion
from src.engine.cross_encoder import CrossEncoderReranker
from src.engine.query_corrector import CorrectiveRAGQueryTransformer
from src.engine.compressor import ContextualCompressor

class ProductionHybridRAGPipeline:
    """
    Master Hybrid RAG pipeline orchestrating Dense + Sparse BM25 retrieval,
    RRF fusion, Cross-Encoder reranking, and Self-Corrective CRAG synthesis.
    """
    def __init__(self, kb_path: str = "data/enterprise_kb.json"):
        self.dense_indexer = DenseSemanticIndexer(embedding_dim=128)
        self.sparse_indexer = SparseBM25Indexer(k1=settings.BM25_K1, b=settings.BM25_B)
        self.rrf_engine = ReciprocalRankFusion(
            k_constant=settings.RRF_K_CONSTANT,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.SPARSE_WEIGHT
        )
        self.reranker = CrossEncoderReranker()
        self.query_corrector = CorrectiveRAGQueryTransformer()
        self.compressor = ContextualCompressor()

        # Load Knowledge Base
        path = Path(kb_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                docs = json.load(f)
                self.index(docs)
        else:
            self.docs = []

    def index(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.dense_indexer.index_documents(docs)
        self.sparse_indexer.index_documents(docs)

    def query(self, query_text: str, top_k: int = 5, top_n_rerank: int = 3) -> Dict[str, Any]:
        profiler = LatencyProfiler()
        profiler.start("total_pipeline")

        # Step 1: Parallel Dense + Sparse Retrieval
        profiler.start("dense_retrieval")
        dense_hits = self.dense_indexer.search(query_text, top_k=top_k)
        profiler.stop("dense_retrieval")

        profiler.start("sparse_bm25")
        sparse_hits = self.sparse_indexer.search(query_text, top_k=top_k)
        profiler.stop("sparse_bm25")

        # Step 2: Reciprocal Rank Fusion
        profiler.start("rrf_fusion")
        fused_candidates = self.rrf_engine.fuse(dense_hits, sparse_hits, top_k=top_k)
        profiler.stop("rrf_fusion")

        # Step 3: Cross-Encoder Context Reranking
        profiler.start("cross_encoder_rerank")
        reranked_chunks = self.reranker.rerank(query_text, fused_candidates, top_n=top_n_rerank)
        profiler.stop("cross_encoder_rerank")

        # Step 4: Self-Correction Evaluation (CRAG)
        top_conf = reranked_chunks[0]["cross_encoder_score"] if reranked_chunks else 0.0
        correction_result = self.query_corrector.evaluate_and_transform(
            query_text,
            top_confidence=top_conf,
            threshold=settings.CONFIDENCE_THRESHOLD
        )

        if correction_result["was_transformed"] and settings.ENABLE_CRAG_FALLBACK:
            # Re-execute with transformed query
            d_hits2 = self.dense_indexer.search(correction_result["transformed_query"], top_k=top_k)
            s_hits2 = self.sparse_indexer.search(correction_result["transformed_query"], top_k=top_k)
            f_cand2 = self.rrf_engine.fuse(d_hits2, s_hits2, top_k=top_k)
            reranked_chunks = self.reranker.rerank(correction_result["transformed_query"], f_cand2, top_n=top_n_rerank)

        # Step 5: Grounded Answer Synthesis & Citations
        profiler.start("synthesis")
        context_blocks = []
        citations = []
        for idx, chunk in enumerate(reranked_chunks, start=1):
            compressed_content = self.compressor.compress(chunk["content"])
            context_blocks.append(f"[{idx}] {chunk['title']}: {compressed_content}")
            citations.append({
                "citation_index": idx,
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "relevance_score": chunk["cross_encoder_score"],
                "category": chunk["category"]
            })

        synthesized_answer = self._synthesize_grounded_response(query_text, reranked_chunks)
        profiler.stop("synthesis")

        profiler.stop("total_pipeline")

        return {
            "query": query_text,
            "answer": synthesized_answer,
            "citations": citations,
            "retrieved_chunks": reranked_chunks,
            "crag_correction": correction_result,
            "telemetry": profiler.get_summary()
        }

    def _synthesize_grounded_response(self, query: str, top_chunks: List[Dict[str, Any]]) -> str:
        if not top_chunks:
            return "No relevant enterprise documentation found matching the query."
        primary = top_chunks[0]
        conf = int(primary["cross_encoder_score"] * 100)
        title = primary["title"]
        content = primary["content"]
        return f"Based on grounded retrieval from **{title}** (Confidence: {conf}%):\n\n{content}"

# Global singleton pipeline
hybrid_rag_pipeline = ProductionHybridRAGPipeline()
