"""Embedding generation for RAG."""

import os
from typing import Any


def embed_chunks(chunks: list[dict[str, Any]]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """
    Given a list of chunks, return their vector embeddings and the original chunks.
    Uses all-MiniLM-L6-v2 on CPU.
    """
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Force CPU device for sentence transformers to preserve VRAM for generation
    # Upgraded to BAAI/bge-small-en-v1.5 for state-of-the-art retrieval accuracy
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    
    texts = [c["text"] for c in chunks]
    if not texts:
        return [], []
        
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist(), chunks
