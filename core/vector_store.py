"""
ChromaDB Vector Store Module
Manages vector indexing, persistent storage, and semantic similarity search.
"""

import os
import shutil
import numpy as np
from typing import List, Dict, Any, Optional


class ChromaVectorStore:
    """Enterprise ChromaDB vector store wrapper with persistence and fallback resilience."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "knowflow_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.is_chroma_active = False

        # In-memory storage fallback
        self._fallback_chunks: List[Dict[str, Any]] = []
        self._fallback_embeddings: List[List[float]] = []

        self._initialize_store()

    def _initialize_store(self):
        """Initializes ChromaDB persistent client; falls back to in-memory store if needed."""
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.is_chroma_active = True
        except Exception:
            self.client = None
            self.collection = None
            self.is_chroma_active = False

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Adds text chunks and their corresponding embeddings to the vector store.
        Returns the number of added chunks.
        """
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": str(c.get("source", "unknown")),
                "page": int(c.get("page", 1)),
                "chunk_index": int(c.get("chunk_index", 0)),
                "char_count": int(c.get("char_count", len(c["text"])))
            }
            for c in chunks
        ]

        if self.is_chroma_active and self.collection is not None:
            try:
                # Upsert to ChromaDB in batches of 100
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    self.collection.upsert(
                        ids=ids[i:i + batch_size],
                        embeddings=embeddings[i:i + batch_size],
                        documents=documents[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size]
                    )
                return len(ids)
            except Exception:
                # Fall through to fallback
                pass

        # Fallback storage
        for c, emb in zip(chunks, embeddings):
            # Check for duplicate ID
            existing_idx = next((i for i, fc in enumerate(self._fallback_chunks) if fc["id"] == c["id"]), None)
            if existing_idx is not None:
                self._fallback_chunks[existing_idx] = c
                self._fallback_embeddings[existing_idx] = emb
            else:
                self._fallback_chunks.append(c)
                self._fallback_embeddings.append(emb)

        return len(ids)

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar document chunks given a query embedding.
        Returns items with text, metadata, and calculated similarity_score (0.0 to 1.0).
        """
        if self.count() == 0:
            return []

        if self.is_chroma_active and self.collection is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.count()),
                    include=["documents", "metadatas", "distances"]
                )

                retrieved = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                ids = results.get("ids", [[]])[0]

                for doc_text, meta, dist, cid in zip(docs, metas, dists, ids):
                    # For cosine distance, similarity = 1 - distance
                    sim = max(0.0, min(1.0, 1.0 - float(dist)))
                    retrieved.append({
                        "id": cid,
                        "text": doc_text,
                        "source": meta.get("source", "unknown"),
                        "page": meta.get("page", 1),
                        "chunk_index": meta.get("chunk_index", 0),
                        "similarity_score": round(sim, 4),
                        "distance": round(float(dist), 4)
                    })

                # Sort by similarity score descending
                retrieved.sort(key=lambda x: x["similarity_score"], reverse=True)
                return retrieved
            except Exception:
                pass

        # Fallback cosine similarity
        if not self._fallback_embeddings:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec = q_vec / q_norm

        scores = []
        for idx, emb in enumerate(self._fallback_embeddings):
            e_vec = np.array(emb, dtype=np.float32)
            e_norm = np.linalg.norm(e_vec)
            if e_norm > 1e-6:
                e_vec = e_vec / e_norm
            score = float(np.dot(q_vec, e_vec))
            sim = max(0.0, min(1.0, (score + 1.0) / 2.0))
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_k]

        retrieved = []
        for idx, sim in top_matches:
            c = self._fallback_chunks[idx]
            retrieved.append({
                "id": c["id"],
                "text": c["text"],
                "source": c.get("source", "unknown"),
                "page": c.get("page", 1),
                "chunk_index": c.get("chunk_index", 0),
                "similarity_score": round(sim, 4),
                "distance": round(1.0 - sim, 4)
            })

        return retrieved

    def count(self) -> int:
        """Returns the total number of chunks currently stored."""
        if self.is_chroma_active and self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._fallback_chunks)

    def get_document_summary(self) -> List[Dict[str, Any]]:
        """Returns list of unique documents and their chunk counts."""
        if self.is_chroma_active and self.collection is not None:
            try:
                data = self.collection.get(include=["metadatas"])
                metas = data.get("metadatas", [])
                doc_map = {}
                for m in metas:
                    s = m.get("source", "unknown")
                    p = m.get("page", 1)
                    if s not in doc_map:
                        doc_map[s] = {"source": s, "chunks": 0, "max_page": 1}
                    doc_map[s]["chunks"] += 1
                    doc_map[s]["max_page"] = max(doc_map[s]["max_page"], p)
                return list(doc_map.values())
            except Exception:
                pass

        doc_map = {}
        for c in self._fallback_chunks:
            s = c.get("source", "unknown")
            p = c.get("page", 1)
            if s not in doc_map:
                doc_map[s] = {"source": s, "chunks": 0, "max_page": 1}
            doc_map[s]["chunks"] += 1
            doc_map[s]["max_page"] = max(doc_map[s]["max_page"], p)
        return list(doc_map.values())

    def clear(self):
        """Clears all records in the vector database."""
        if self.is_chroma_active and self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
        self._fallback_chunks = []
        self._fallback_embeddings = []
