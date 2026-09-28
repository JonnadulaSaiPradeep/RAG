"""
Document Processor Module
Handles ingestion of PDF, TXT, and Markdown files with metadata extraction.
"""

import os
import io
import re
from typing import List, Dict, Any, Union


class DocumentProcessor:
    """Extracts structured text and metadata from uploaded enterprise documents."""

    def __init__(self):
        pass

    def process_file(
        self,
        file_bytes: Union[bytes, bytearray],
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Process a file given its raw bytes and filename.
        Returns a list of page/section dicts:
        [{'text': ..., 'source': filename, 'page': page_num, 'char_count': ...}, ...]
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_bytes, filename)
        elif ext in [".txt", ".md", ".csv", ".json", ".log"]:
            return self._process_text(file_bytes, filename)
        else:
            # Fallback text extraction
            return self._process_text(file_bytes, filename)

    def _process_pdf(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned = self._clean_text(raw_text)
                if cleaned:
                    pages.append({
                        "text": cleaned,
                        "source": filename,
                        "page": idx + 1,
                        "total_pages": total_pages,
                        "char_count": len(cleaned)
                    })
        except ImportError:
            # If pypdf is not yet installed, use a regex-based stream fallback
            text = self._fallback_pdf_text_extraction(file_bytes)
            pages.append({
                "text": text,
                "source": filename,
                "page": 1,
                "total_pages": 1,
                "char_count": len(text)
            })
        except Exception as e:
            # General fallback
            text = self._clean_text(file_bytes.decode("utf-8", errors="ignore"))
            pages.append({
                "text": text,
                "source": filename,
                "page": 1,
                "total_pages": 1,
                "char_count": len(text),
                "error": str(e)
            })

        return pages

    def _process_text(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        text = file_bytes.decode("utf-8", errors="ignore")
        cleaned = self._clean_text(text)
        return [{
            "text": cleaned,
            "source": filename,
            "page": 1,
            "total_pages": 1,
            "char_count": len(cleaned)
        }]

    def _clean_text(self, text: str) -> str:
        """Normalizes excessive whitespace, controls, and formatting artifacts."""
        if not text:
            return ""
        # Replace multiple horizontal spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize excessive linebreaks (keep paragraph breaks)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    def _fallback_pdf_text_extraction(self, file_bytes: bytes) -> str:
        """Primitive text extraction from PDF stream when pypdf is unavailable."""
        text_parts = []
        # Find stream blocks or plain text patterns
        matches = re.findall(b"\(([^\(\)\\\\]{4,})\)", file_bytes)
        for m in matches:
            try:
                decoded = m.decode("latin-1", errors="ignore").strip()
                if len(decoded) > 3 and not decoded.startswith("/"):
                    text_parts.append(decoded)
            except Exception:
                continue
        res = " ".join(text_parts)
        return self._clean_text(res) if res else "PDF binary loaded (install pypdf for full text extraction)."
