import sys
import os
from pathlib import Path

# Ensure project root is in sys.path for both local and Docker execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import json
from pathlib import Path
from src.engine.hybrid_pipeline import hybrid_rag_pipeline

st.set_page_config(
    page_title="Production Hybrid RAG Visualizer & Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #090d16; }
    .stMetric { background-color: #111827; border-radius: 10px; padding: 12px; border: 1px solid #1f2937; }
    .rag-box { background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 18px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# Sample benchmark questions
qa_path = Path("data/gold_standard_qa.json")
if qa_path.exists():
    with open(qa_path, "r", encoding="utf-8") as f:
        SAMPLE_QA = json.load(f)
else:
    SAMPLE_QA = []

st.sidebar.title("⚡ Hybrid RAG Controls")
st.sidebar.caption("Dense + Sparse BM25 + RRF + Cross-Encoder")

sample_queries = [f"{q['query_type']}: {q['query']}" for q in SAMPLE_QA]
selected_idx = st.sidebar.selectbox("📂 Load Benchmark Query", range(len(sample_queries)), format_func=lambda i: sample_queries[i] if sample_queries else "None")

selected_query_text = SAMPLE_QA[selected_idx]["query"] if SAMPLE_QA else "What are the door-to-balloon PCI time requirements for STEMI?"

st.sidebar.divider()
st.sidebar.markdown("##### ⚙️ Algorithmic Hyperparameters")
st.sidebar.slider("BM25 k1 (Term Saturation)", 1.0, 2.0, 1.5, 0.1)
st.sidebar.slider("BM25 b (Length Normalization)", 0.0, 1.0, 0.75, 0.05)
st.sidebar.number_input("RRF Constant (k)", value=60)

# Header
st.title("⚡ Enterprise Hybrid RAG Engine Visualizer")
st.caption("Sub-50ms Hybrid Vector & BM25 Search • Reciprocal Rank Fusion (RRF) • Cross-Attention Reranking • Self-Corrective CRAG")

query_input = st.text_input("🔍 Enter Natural Language Query or Technical Keyword:", value=selected_query_text)
execute_btn = st.button("🚀 Execute Hybrid Search & Reranking", type="primary")

if execute_btn or query_input:
    res = hybrid_rag_pipeline.query(query_input)
    t = res["telemetry"]
    
    # Telemetry Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dense Vector Time", f"{t.get('dense_retrieval', 0.0)} ms")
    c2.metric("Sparse BM25 Time", f"{t.get('sparse_bm25', 0.0)} ms")
    c3.metric("RRF Fusion Time", f"{t.get('rrf_fusion', 0.0)} ms")
    c4.metric("Reranker Time", f"{t.get('cross_encoder_rerank', 0.0)} ms")
    c5.metric("Total Pipeline SLA", f"{t.get('total_pipeline', 0.0)} ms", delta="Sub-50ms")

    st.divider()

    col_ans, col_chunks = st.columns([1.2, 1])

    with col_ans:
        st.subheader("💡 Grounded Synthesis & Citations")
        st.markdown(res["answer"])
        
        st.markdown("##### 📌 Grounded Context Citations")
        for cit in res["citations"]:
            st.info(f"**[{cit['citation_index']}] {cit['title']}** (`{cit['doc_id']}`) — Relevance: **{int(cit['relevance_score'] * 100)}%**")

        if res["crag_correction"]["was_transformed"]:
            st.warning(f"⚠️ **CRAG Self-Correction Triggered**: {', '.join(res['crag_correction']['reasons'])}")

    with col_chunks:
        st.subheader("📊 Retrieval Channels & Cross-Encoder Scores")
        for chunk in res["retrieved_chunks"]:
            with st.expander(f"{chunk['title']} — Match: {int(chunk['cross_encoder_score'] * 100)}%", expanded=True):
                st.progress(chunk["cross_encoder_score"])
                st.markdown(f"- **Dense Rank**: {chunk['dense_rank']} | **BM25 Sparse Rank**: {chunk['sparse_rank']}")
                st.markdown(f"- **RRF Merged Score**: `{chunk['rrf_score']}`")
                st.caption(chunk["content"][:200] + "...")
