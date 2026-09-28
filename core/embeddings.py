"""
Embedding Generation Module
Supports Sentence Transformers (all-MiniLM-L6-v2) with intelligent dense fallback.
"""

import os
import hashlib
import numpy as np
from typing import List, Union, Optional


class EmbeddingManager:
    """Manages text embedding generation, caching, and model lifecycles."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._embedding_dim = 384
        self._cache = {}
        self.is_transformer_active = False
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load SentenceTransformer; falls back to fast lightweight dense embedder if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            self.is_transformer_active = True
        except Exception:
            self._model = None
            self.is_transformer_active = False
            self._embedding_dim = 384

    def get_embedding_dimension(self) -> int:
        return self._embedding_dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized vector embeddings for a list of texts."""
        if not texts:
            return []

        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        # Check in-memory cache
        for idx, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in self._cache:
                embeddings.append(self._cache[text_hash])
            else:
                embeddings.append(None)
                texts_to_compute.append(text)
                indices_to_compute.append((idx, text_hash))

        if texts_to_compute:
            if self.is_transformer_active and self._model is not None:
                try:
                    computed = self._model.encode(
                        texts_to_compute,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    for (orig_idx, thash), emb in zip(indices_to_compute, computed):
                        emb_list = emb.tolist()
                        self._cache[thash] = emb_list
                        embeddings[orig_idx] = emb_list
                except Exception:
                    computed = self._fallback_embed(texts_to_compute)
                    for (orig_idx, thash), emb in zip(indices_to_compute, computed):
                        self._cache[thash] = emb
                        embeddings[orig_idx] = emb
            else:
                computed = self._fallback_embed(texts_to_compute)
                for (orig_idx, thash), emb in zip(indices_to_compute, computed):
                    self._cache[thash] = emb
                    embeddings[orig_idx] = emb

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generates normalized vector embedding for a single search query."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self._embedding_dim

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic, lightweight token-frequency dense projection (384-dim)
        with L2 normalization for fallback semantic search.
        """
        results = []
        dim = self._embedding_dim

        for text in texts:
            tokens = text.lower().split()
            vec = np.zeros(dim, dtype=np.float32)

            for token in tokens:
                # Hash token to multiple dimensions for dense dispersion
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx1 = h % dim
                idx2 = (h >> 8) % dim
                idx3 = (h >> 16) % dim
                sign1 = 1.0 if ((h >> 24) & 1) else -1.0
                sign2 = 1.0 if ((h >> 25) & 1) else -1.0

                vec[idx1] += sign1
                vec[idx2] += sign2
                vec[idx3] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec = np.random.RandomState(42).randn(dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)

            results.append(vec.tolist())

        return results
