"""
KnowFlow RAG Engine Orchestrator
Coordinates document processing, vector search, Gemini generation, and source attribution.
"""

import os
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from .document_processor import DocumentProcessor
from .chunking import RecursiveChunker
from .embeddings import EmbeddingManager
from .vector_store import ChromaVectorStore


class KnowFlowRAGEngine:
    """Enterprise RAG pipeline orchestrating vector retrieval and grounded Gemini generation."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 600,
        chunk_overlap: int = 120
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.doc_processor = DocumentProcessor()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_mgr = EmbeddingManager(model_name=embedding_model)
        self.vector_store = ChromaVectorStore(persist_directory=persist_dir)
        self._gemini_client = None
        self._setup_gemini()

    def set_api_key(self, api_key: str):
        """Update Gemini API Key dynamically."""
        self.gemini_api_key = api_key.strip()
        self._setup_gemini()

    def _setup_gemini(self):
        """Configures google.generativeai if key is present."""
        if not self.gemini_api_key:
            self._gemini_client = None
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            # Try to initialize gemini-1.5-flash as default fast model
            self._gemini_client = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                }
            )
        except Exception:
            self._gemini_client = None

    def ingest_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Full ingestion pipeline:
        1. Extract text and pages
        2. Chunk text with provenance
        3. Compute embeddings
        4. Store into ChromaDB
        """
        start_time = time.time()
        # 1. Extract
        pages = self.doc_processor.process_file(file_bytes, filename)
        if not pages:
            return {"status": "error", "message": f"No extractable text found in {filename}."}

        # 2. Chunk
        chunks = self.chunker.chunk_documents(pages)
        if not chunks:
            return {"status": "error", "message": f"Text could not be chunked for {filename}."}

        # 3. Embed
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_mgr.embed_texts(chunk_texts)

        # 4. Store
        added_count = self.vector_store.add_chunks(chunks, embeddings)
        elapsed = time.time() - start_time

        return {
            "status": "success",
            "filename": filename,
            "pages_processed": len(pages),
            "chunks_created": added_count,
            "elapsed_seconds": round(elapsed, 2)
        }

    def query(
        self,
        user_query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.25,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes document-grounded question answering:
        1. Embeds user query
        2. Retrieves top relevant chunks from ChromaDB
        3. Constructs context-aware grounded prompt
        4. Generates synthesized answer with source references
        """
        start_time = time.time()
        if not user_query.strip():
            return {
                "answer": "Please provide a valid question to query the knowledge base.",
                "sources": [],
                "latency_seconds": 0.0,
                "chunks_retrieved": 0
            }

        # 1. Embed query
        query_embedding = self.embedding_mgr.embed_query(user_query)

        # 2. Retrieve chunks
        retrieved_chunks = self.vector_store.similarity_search(query_embedding, top_k=top_k)

        # Filter by threshold
        valid_chunks = [c for c in retrieved_chunks if c["similarity_score"] >= similarity_threshold]

        if not valid_chunks:
            elapsed = time.time() - start_time
            return {
                "answer": (
                    "I searched the uploaded knowledge base but could not find relevant content "
                    "matching your query. Please try rephrasing your question or verify that the relevant "
                    "documents have been ingested into KnowFlow AI."
                ),
                "sources": [],
                "latency_seconds": round(elapsed, 3),
                "chunks_retrieved": 0
            }

        # 3. Generate response
        answer = self._generate_grounded_answer(user_query, valid_chunks, chat_history)
        elapsed = time.time() - start_time

        return {
            "answer": answer,
            "sources": valid_chunks,
            "latency_seconds": round(elapsed, 3),
            "chunks_retrieved": len(valid_chunks)
        }

    def _generate_grounded_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Constructs prompt and queries Gemini API (or intelligent grounded fallback)."""
        # Format context with explicit tags
        context_blocks = []
        for i, c in enumerate(chunks):
            source_tag = f"Source [{i+1}]: {c['source']} (Page {c['page']})"
            context_blocks.append(f"{source_tag}\n{c['text']}\n")

        context_str = "\n".join(context_blocks)

        system_instruction = (
            "You are KnowFlow AI, an enterprise knowledge intelligence assistant. "
            "Your objective is to provide precise, professional, and factually grounded answers "
            "strictly using the provided Document Context.\n\n"
            "Guidelines:\n"
            "1. Answer the user query thoroughly using the context facts.\n"
            "2. Cite your sources in the text using bracketed citations, e.g. [Document.pdf, Page 1].\n"
            "3. If different sources present complementary or differing information, synthesize them clearly.\n"
            "4. If the provided context does not contain enough information to answer fully, explicitly note what is missing.\n"
            "5. Do NOT hallucinate facts outside the provided document excerpts."
        )

        history_str = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            formatted_turns = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_turns]
            history_str = "\nRecent Conversation History:\n" + "\n".join(formatted_turns) + "\n"

        prompt = (
            f"{system_instruction}\n\n"
            f"=== DOCUMENT CONTEXT ===\n{context_str}\n"
            f"========================\n"
            f"{history_str}\n"
            f"User Question: {query}\n\n"
            f"Enterprise Synthesized Answer (with source citations):"
        )

        # Call Gemini if client is active
        if self._gemini_client is not None and self.gemini_api_key:
            try:
                response = self._gemini_client.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                # If Gemini API returns rate limit or quota issue, fall through to synthesizer
                pass

        # Intelligent offline grounded synthesizer
        return self._offline_synthesizer(query, chunks)

    def _offline_synthesizer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Robust offline extractive-abstractive synthesis when Gemini API key is not configured.
        Extracts key sentences relevant to query keywords and formats with proper enterprise citations.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {"what", "is", "the", "how", "are", "and", "or", "in", "to", "for", "with", "a", "an", "of", "about"}
        keywords = query_words - stopwords

        top_sentences = []
        citations_used = set()

        for c in chunks:
            text = c["text"]
            source = c["source"]
            page = c["page"]
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 25:
                    continue
                s_lower = s_clean.lower()
                # Keyword overlap match
                overlap = sum(1 for kw in keywords if kw in s_lower)
                if overlap > 0:
                    top_sentences.append({
                        "sentence": s_clean,
                        "overlap": overlap,
                        "source": source,
                        "page": page,
                        "sim": c["similarity_score"]
                    })
                    citations_used.add(f"[{source}, Page {page}]")

        if not top_sentences:
            # Fall back to using top chunk excerpts directly
            first = chunks[0]
            return (
                f"Based on the most relevant match from **{first['source']}** (Page {first['page']}, "
                f"Relevance: {first['similarity_score'] * 100:.1f}%):\n\n"
                f"> \"{first['text'][:400]}...\"\n\n"
                f"*Source: [{first['source']}, Page {first['page']}]*\n\n"
                f"*(Note: Connect your Gemini API Key in the sidebar for complete generative synthesis)*"
            )

        # Sort sentences by keyword overlap and similarity
        top_sentences.sort(key=lambda x: (x["overlap"], x["sim"]), reverse=True)
        selected = top_sentences[:4]

        # Group by source
        grouped = {}
        for item in selected:
            key = f"{item['source']} (Page {item['page']})"
            if key not in grouped:
                grouped[key] = []
            if item["sentence"] not in grouped[key]:
                grouped[key].append(item["sentence"])

        output_lines = [
            f"### Document Findings for \"{query}\"\n",
            "According to the ingested enterprise documentation, here are the key findings:\n"
        ]

        for src_label, s_list in grouped.items():
            output_lines.append(f"**From {src_label}:**")
            for s in s_list:
                output_lines.append(f"- {s}")
            output_lines.append("")

        sources_tag = ", ".join(sorted(citations_used))
        output_lines.append(f"**Verified Sources:** {sources_tag}")
        output_lines.append("\n*(Note: For conversational AI synthesis, enter your Google Gemini API Key in the sidebar)*")

        return "\n".join(output_lines)
