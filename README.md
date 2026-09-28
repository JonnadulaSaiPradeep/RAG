# ⚡ KnowFlow AI — Enterprise Knowledge Intelligence Platform (2026)

> **GenAI Multi-Document RAG Application**  
> **Author:** JonnadulaSaiPradeep (`231fa18264@gmail.com`)  
> **Repository:** `knowflow-ai`

---

## 🌟 Executive Summary

**KnowFlow AI** is a state-of-the-art enterprise knowledge intelligence platform engineered to transform unstructured organizational documentation into actionable, grounded intelligence. Powered by a modern multi-document Retrieval-Augmented Generation (RAG) architecture, KnowFlow AI eliminates LLM hallucinations by enforcing strict factual grounding against enterprise repositories with auditable, page-level source attribution.

---

## 🚀 Core Features

- **Multi-Document Ingestion:** Automated extraction and ingestion across PDF, TXT, and Markdown documents while preserving page-level provenance.
- **Recursive Semantic Chunking:** Intelligent document segmenting with configurable chunk sizes (600 chars) and overlap windows (120 chars) to maintain semantic context across boundaries.
- **Dense Vector Search:** High-performance similarity search powered by Sentence Transformers (`all-MiniLM-L6-v2`) and **ChromaDB** vector database.
- **Gemini GenAI Integration:** Native support for Google Gemini (Gemini 1.5 Flash / Pro) with low-temperature prompt templates optimized for strict document grounding.
- **Transparent Source Attribution:** Every synthesized response provides direct inline citations `[Document, Page X]` and an expandable citation inspector displaying exact text excerpts and cosine similarity scores.
- **Interactive Enterprise Dashboard:** Streamlit UI featuring dark glassmorphism styling, real-time vector metrics, dynamic parameter sliders, and quick-load sample documents.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────┐
│ Enterprise Documents (PDF, TXT) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Document Processing Pipeline  │
│  (Page & Provenance Extraction) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Recursive Text Chunker      │
│     (600 chars, 120 overlap)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Sentence Transformers Embed   │
│   (384-dimensional vectors)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   ChromaDB Persistent Store     │
│   (Cosine similarity index)     │
└────────────────┬────────────────┘
                 │ Top-K Chunks
                 ▼
┌─────────────────────────────────┐
│     Google Gemini 1.5 Flash     │
│  Context-Grounded Generation    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Streamlit Conversational UI     │
│ With Inline Source Citations    │
└─────────────────────────────────┘
```

---

## 🛠️ Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.11+**
- **Git**

### 2. Virtual Environment Setup
A dedicated virtual environment has been created in `.venv/`:

```bash
# Navigate to project directory
cd knowflow-ai

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
```

### 3. Dependencies Installation
```bash
pip install -r requirements.txt
```

### 4. Git Configuration
Verified commit author details:
```bash
git config user.name "JonnadulaSaiPradeep"
git config user.email "231fa18264@gmail.com"
```

### 5. Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Or simply double-click `start.bat` on Windows.

---

## 📂 Project Directory Structure

```
knowflow-ai/
├── .streamlit/
│   └── config.toml                  # Streamlit dark enterprise theme configuration
├── .venv/                            # Python 3.11 Virtual Environment
├── core/
│   ├── __init__.py                  # Core package init
│   ├── chunking.py                  # Recursive text chunker with overlap & provenance
│   ├── document_processor.py        # PDF & text ingestion with page extraction
│   ├── embeddings.py                # Sentence Transformers & dense vector embedding manager
│   ├── vector_store.py              # ChromaDB vector store with cosine similarity index
│   └── rag_engine.py                # RAG orchestrator with Gemini API & citation synthesis
├── sample_documents/
│   ├── Enterprise_Cloud_Governance_2026.txt
│   ├── KnowFlow_System_Architecture_Guide.txt
│   ├── KnowFlow_AI_Overview_2026.pdf
│   └── create_sample_pdf.py
├── .env.example                     # Environment template (GEMINI_API_KEY)
├── .gitignore                       # Git ignore rules for venv, cache & chroma_db
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # Project dependencies
├── start.bat                        # Windows 1-click launcher
└── test_pipeline.py                 # Automated verification test script
```

---

## 🧪 Testing & Validation

Run the end-to-end RAG verification test:
```bash
python test_pipeline.py
```

This verifies document ingestion, recursive chunking, embedding generation, vector indexing, and grounded question answering with provenance citations.

---

## 📄 License
Enterprise MIT License — © 2026 JonnadulaSaiPradeep.
