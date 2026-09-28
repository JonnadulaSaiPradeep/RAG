"""
KnowFlow AI — Enterprise Knowledge Intelligence Platform 2026
Streamlit-based Multi-Document RAG Application
Author: JonnadulaSaiPradeep (231fa18264@gmail.com)
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="KnowFlow AI — Enterprise Knowledge Intelligence 2026",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Styling
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 14px;
        line-height: 1.4;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-primary {
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.15);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    
    /* Stat cards */
    .stat-card {
        background: #131d31;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Citation Box */
    .citation-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .relevance-score {
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Import RAG Engine modules
from core.rag_engine import KnowFlowRAGEngine

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = KnowFlowRAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# Pre-populate sample documents if vector store is empty
sample_dir = os.path.join(os.path.dirname(__file__), "sample_documents")


# ==========================================
# SIDEBAR: Configuration & Knowledge Base
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ KnowFlow Control Center")
    st.caption("Enterprise RAG Configuration & Document Hub")
    
    # 1. API Configuration
    st.markdown("#### 🔑 GenAI Model Provider")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enter your Google Gemini API Key from AI Studio. Offline synthesis is used if left blank."
    )
    if gemini_key:
        st.session_state.rag_engine.set_api_key(gemini_key)
        st.success("Gemini 1.5 Flash Connected", icon="✅")
    else:
        st.info("Operating in Offline Synthesis Mode (Enter Gemini Key for full LLM generative synthesis)", icon="💡")

    st.markdown("---")

    # 2. Document Ingestion Section
    st.markdown("#### 📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Enterprise Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown"
    )

    if uploaded_files:
        process_btn = st.button("🚀 Process & Ingest Files", use_container_width=True, type="primary")
        if process_btn:
            with st.spinner("Extracting text, chunking, and computing embeddings..."):
                total_chunks = 0
                for f in uploaded_files:
                    file_bytes = f.read()
                    res = st.session_state.rag_engine.ingest_document(file_bytes, f.name)
                    if res["status"] == "success":
                        st.session_state.ingested_files.add(f.name)
                        total_chunks += res["chunks_created"]
                st.success(f"Ingested {len(uploaded_files)} files ({total_chunks} chunks indexed in ChromaDB)!", icon="🎉")

    # Quick Ingest Sample Documents
    if os.path.exists(sample_dir):
        if st.button("📥 Load Built-in Sample Documents", use_container_width=True):
            with st.spinner("Ingesting sample enterprise documentation..."):
                sample_count = 0
                total_chunks = 0
                for fname in os.listdir(sample_dir):
                    if fname.endswith((".txt", ".pdf", ".md")):
                        fpath = os.path.join(sample_dir, fname)
                        with open(fpath, "rb") as sf:
                            res = st.session_state.rag_engine.ingest_document(sf.read(), fname)
                            if res["status"] == "success":
                                st.session_state.ingested_files.add(fname)
                                total_chunks += res["chunks_created"]
                                sample_count += 1
                st.success(f"Loaded {sample_count} sample files ({total_chunks} chunks indexed)!", icon="📚")

    st.markdown("---")

    # 3. Retrieval Hyperparameters
    st.markdown("#### 🎛️ RAG Parameters")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=4, step=1)
    sim_threshold = st.slider("Min Similarity Threshold", min_value=0.0, max_value=0.8, value=0.25, step=0.05)

    st.markdown("---")

    # 4. Vector Store Stats
    doc_summary = st.session_state.rag_engine.vector_store.get_document_summary()
    total_chunks_stored = st.session_state.rag_engine.vector_store.count()
    
    st.markdown("#### 📊 Vector Store Metrics")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Documents", len(doc_summary))
    with col_sb2:
        st.metric("Indexed Chunks", total_chunks_stored)

    if total_chunks_stored > 0:
        if st.button("🗑️ Clear Vector Database", use_container_width=True):
            st.session_state.rag_engine.vector_store.clear()
            st.session_state.ingested_files.clear()
            st.session_state.chat_history.clear()
            st.rerun()

    st.caption("Engine: Sentence Transformers + ChromaDB + Gemini")


# ==========================================
# MAIN PANEL: Hero Header & Navigation
# ==========================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ KnowFlow AI</div>
    <div class="hero-subtitle">
        Enterprise Knowledge Intelligence Platform 2026 • Multi-Document Grounded RAG Application
    </div>
    <div class="tag-container">
        <span class="badge badge-primary">Streamlit Core</span>
        <span class="badge badge-purple">Google Gemini GenAI</span>
        <span class="badge badge-primary">Sentence Transformers (all-MiniLM-L6-v2)</span>
        <span class="badge badge-green">ChromaDB Vector Store</span>
        <span class="badge badge-primary">Developer: JonnadulaSaiPradeep</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metrics Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(doc_summary)}</div>
        <div class="stat-label">Active Knowledge Docs</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{total_chunks_stored}</div>
        <div class="stat-label">ChromaDB Vector Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">384-D</div>
        <div class="stat-label">Embedding Dimensions</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    mode_label = "Gemini Flash" if gemini_key else "Offline Grounded"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{mode_label}</div>
        <div class="stat-label">Synthesis Engine</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Interface Tabs
tab_chat, tab_docs, tab_arch = st.tabs([
    "💬 Conversational RAG Assistant",
    "📚 Document & Vector Inspector",
    "⚙️ System Architecture & Provenance"
])


# ==========================================
# TAB 1: Conversational RAG Assistant
# ==========================================
with tab_chat:
    # Quick Starter Query Pills
    st.markdown("##### 💡 Suggested Enterprise Inquiries:")
    quick_queries = [
        "What are the Tier-1 data encryption mandates in 2026?",
        "How does KnowFlow AI ensure hallucination-free generation?",
        "Explain the recursive chunking parameters and overlap strategy.",
        "What are the key responsibilities under Zero Trust cloud governance?"
    ]
    
    q_cols = st.columns(len(quick_queries))
    selected_quick_query = None
    for i, q in enumerate(quick_queries):
        with q_cols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                selected_quick_query = q

    st.markdown("---")

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "⚡"):
            st.markdown(message["content"])
            # If sources exist, render source citation accordion
            if "sources" in message and message["sources"]:
                with st.expander(f"📑 Grounded Source References ({len(message['sources'])} chunks retrieved)", expanded=False):
                    for s_idx, src in enumerate(message["sources"]):
                        sim_pct = src['similarity_score'] * 100
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>📄 [{s_idx+1}] {src['source']} — Page {src['page']}</span>
                                <span class="relevance-score">Cosine Similarity: {sim_pct:.1f}%</span>
                            </div>
                            <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4;">
                                "{src['text']}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    if "latency" in message:
                        st.caption(f"⚡ Retrieval & Synthesis Latency: {message['latency']:.3f}s")

    # Handle User Input
    chat_prompt = st.chat_input("Ask any question grounded in your uploaded documents...")
    effective_prompt = selected_quick_query or chat_prompt

    if effective_prompt:
        if total_chunks_stored == 0:
            st.warning("⚠️ The knowledge base is currently empty. Please upload documents or click 'Load Built-in Sample Documents' in the sidebar first!", icon="⚠️")
        else:
            # Append User Message
            st.session_state.chat_history.append({"role": "user", "content": effective_prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(effective_prompt)

            # Process Query with RAG Engine
            with st.chat_message("assistant", avatar="⚡"):
                with st.spinner("Searching ChromaDB semantic index & synthesizing grounded answer..."):
                    result = st.session_state.rag_engine.query(
                        user_query=effective_prompt,
                        top_k=top_k,
                        similarity_threshold=sim_threshold,
                        chat_history=st.session_state.chat_history[:-1]
                    )

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander(f"📑 Grounded Source References ({len(result['sources'])} chunks retrieved)", expanded=True):
                        for s_idx, src in enumerate(result["sources"]):
                            sim_pct = src['similarity_score'] * 100
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span>📄 [{s_idx+1}] {src['source']} — Page {src['page']}</span>
                                    <span class="relevance-score">Cosine Similarity: {sim_pct:.1f}%</span>
                                </div>
                                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4;">
                                    "{src['text']}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.caption(f"⚡ Retrieval & Synthesis Latency: {result['latency_seconds']:.3f}s | Chunks: {result['chunks_retrieved']}")

            # Save Assistant Response to History
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "latency": result["latency_seconds"]
            })


# ==========================================
# TAB 2: Document & Vector Inspector
# ==========================================
with tab_docs:
    st.markdown("### 📚 Enterprise Document Ingestion & Chunk Directory")
    st.caption("Verify document provenance, chunk breakdown, and vector index health.")

    if not doc_summary:
        st.info("No documents have been ingested yet. Upload files or load sample documents via the sidebar.", icon="ℹ️")
    else:
        st.markdown("#### Ingested Documents Overview")
        doc_table = []
        for d in doc_summary:
            doc_table.append({
                "Document Name": d["source"],
                "Indexed Chunks": d["chunks"],
                "Total Pages": d["max_page"],
                "Vector Store Status": "✅ Indexed & Persistent"
            })
        st.dataframe(doc_table, use_container_width=True)

        st.markdown("#### 🔍 Real-Time Semantic Similarity Probe")
        probe_query = st.text_input("Test query to inspect nearest neighbor chunks:", "Zero trust encryption")
        if probe_query:
            p_emb = st.session_state.rag_engine.embedding_mgr.embed_query(probe_query)
            matches = st.session_state.rag_engine.vector_store.similarity_search(p_emb, top_k=3)
            if matches:
                for idx, m in enumerate(matches):
                    st.markdown(f"**Match #{idx+1} — {m['source']} (Page {m['page']})** • *Relevance: {m['similarity_score']*100:.1f}%*")
                    st.code(m['text'], language="text")
            else:
                st.write("No matching chunks found.")


# ==========================================
# TAB 3: Architecture & Provenance
# ==========================================
with tab_arch:
    st.markdown("### ⚙️ KnowFlow AI Architectural Specifications")
    
    st.markdown("""
    #### 1. End-to-End Multi-Document RAG Pipeline
    ```
    ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
    │ Enterprise Documents │ ───> │   Document Ingestion    │ ───> │   Recursive Chunking    │
    │ (PDF, TXT, Markdown) │      │ (Page & Provenance Map) │      │  (600 chars, 120 ovlp)  │
    └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                                │
                                                                                ▼
    ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
    │  Google Gemini 1.5   │ <─── │ Context-Grounded Prompt │ <─── │   ChromaDB Vector DB    │
    │  Generative Synthesis│      │ + In-line Attributions  │      │(Sentence Transformers)  │
    └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
    ```

    #### 2. Project Metadata & Configuration
    - **Platform Name:** KnowFlow AI — Enterprise Knowledge Intelligence Platform 2026
    - **Developer:** `JonnadulaSaiPradeep`
    - **Git Email:** `231fa18264@gmail.com`
    - **Frameworks:** Streamlit, Python 3.11, ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), Google Gemini
    - **Persistence:** Local Persistent ChromaDB vector database (`./chroma_db`)
    """)
