"""
rag/retriever.py
Handles chunking, embedding generation, FAISS vector store, and retrieval.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup cost
_embedding_model = None
_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_by_lines(text: str, max_lines: int = 60, overlap: int = 10) -> List[str]:
    """Split text into overlapping line-based chunks."""
    lines = text.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += max_lines - overlap
    return chunks


def _chunk_python(text: str) -> List[str]:
    """Split Python files at class/function boundaries."""
    pattern = r"(?=^(?:class |def |async def ))"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    chunks = []
    buffer = ""
    for part in parts:
        if len(buffer) + len(part) < 3000:
            buffer += part
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = part
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks if chunks else _chunk_by_lines(text)


def chunk_file(file_info: Dict) -> List[Dict]:
    """Chunk a file into smaller pieces for embedding."""
    content = file_info["content"]
    path = file_info["path"]
    ext = file_info.get("extension", "")

    if ext == ".py":
        raw_chunks = _chunk_python(content)
    elif ext in {".md", ".rst", ".txt"}:
        # Split by headings or paragraphs
        raw_chunks = re.split(r"\n#{1,4} |\n\n", content)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        # Merge tiny chunks
        merged, buf = [], ""
        for c in raw_chunks:
            if len(buf) + len(c) < 1500:
                buf += "\n\n" + c
            else:
                if buf.strip():
                    merged.append(buf.strip())
                buf = c
        if buf.strip():
            merged.append(buf.strip())
        raw_chunks = merged if merged else _chunk_by_lines(content)
    else:
        raw_chunks = _chunk_by_lines(content)

    return [
        {"text": chunk, "source": path, "chunk_index": i}
        for i, chunk in enumerate(raw_chunks)
        if len(chunk.strip()) > 30
    ]


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """FAISS-backed vector store for RAG retrieval."""

    def __init__(self):
        self.chunks: List[Dict] = []
        self.index = None
        self.dimension: Optional[int] = None

    def build(self, all_chunks: List[Dict], progress_callback=None) -> None:
        """Embed all chunks and build the FAISS index."""
        if not all_chunks:
            raise ValueError("No chunks to embed.")

        faiss = _get_faiss()
        model = _get_embedding_model()

        texts = [c["text"] for c in all_chunks]

        # Batch embed
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embeddings)
            if progress_callback:
                progress_callback(min(i + batch_size, len(texts)), len(texts))

        embeddings_matrix = np.vstack(all_embeddings).astype("float32")
        self.dimension = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner-product (cosine on normalized)
        self.index.add(embeddings_matrix)
        self.chunks = all_chunks

        logger.info(f"Built FAISS index with {len(all_chunks)} chunks, dim={self.dimension}")

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict]:
        """Retrieve top-k chunks relevant to query."""
        if self.index is None or not self.chunks:
            return []

        model = _get_embedding_model()
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_emb, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def retrieve_for_file(self, file_path: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks relevant to a specific file."""
        return self.retrieve(f"documentation for {file_path}", top_k=top_k)