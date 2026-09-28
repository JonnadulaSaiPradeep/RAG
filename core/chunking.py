"""
Text Chunking Module
Implements Recursive Character Chunking with overlap and metadata retention.
"""

import hashlib
from typing import List, Dict, Any


class RecursiveChunker:
    """
    Splits text recursively using prioritized separators:
    Paragraphs ("\\n\\n") -> Lines ("\\n") -> Sentences (". ") -> Words (" ") -> Characters.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of document pages/sections and returns granular chunks with metadata.
        """
        all_chunks = []
        global_index = 0

        for doc in documents:
            source = doc.get("source", "unknown")
            page = doc.get("page", 1)
            text = doc.get("text", "")

            if not text.strip():
                continue

            raw_chunks = self._split_text(text, self.separators)

            for i, chunk_text in enumerate(raw_chunks):
                cleaned_chunk = chunk_text.strip()
                if len(cleaned_chunk) < 20:  # Skip trivial fragments
                    continue

                chunk_hash = hashlib.md5(f"{source}_{page}_{i}_{cleaned_chunk[:40]}".encode()).hexdigest()[:10]
                chunk_id = f"chunk_{source}_{page}_{i}_{chunk_hash}"

                all_chunks.append({
                    "id": chunk_id,
                    "text": cleaned_chunk,
                    "source": source,
                    "page": page,
                    "chunk_index": i,
                    "global_index": global_index,
                    "char_count": len(cleaned_chunk),
                    "estimated_tokens": max(1, len(cleaned_chunk) // 4)
                })
                global_index += 1

        return all_chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting algorithm respecting separator hierarchy."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if separator and s != splits[-1]:
                s = s + separator

            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    recursive_subsplits = self._split_text(s, new_separators)
                    final_chunks.extend(recursive_subsplits)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Combines small splits into chunks up to chunk_size with overlap."""
        chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current_chunk:
                combined = "".join(current_chunk).strip()
                if combined:
                    chunks.append(combined)

                # Keep overlap from end of current chunk
                overlap_chars = 0
                overlap_splits = []
                for s in reversed(current_chunk):
                    overlap_splits.insert(0, s)
                    overlap_chars += len(s)
                    if overlap_chars >= self.chunk_overlap:
                        break

                current_chunk = list(overlap_splits)
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(split)
            current_len += split_len

        if current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

        return chunks
