"""FAISS Vector Store for RAG."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Any

class VectorStore:
    def __init__(self, dimension: int = 384): # 384 is size for all-MiniLM-L6-v2
        import faiss
        # L2 distance index
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[dict[str, Any]] = []
        
    def add(self, embeddings: list[list[float]], chunks: list[dict[str, Any]]):
        if not embeddings:
            return
            
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        
    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top k chunks for a given query embedding."""
        if not self.chunks:
            return []
            
        k = min(k, len(self.chunks))
        vector = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(vector, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
                
        return results
        
    def save(self, path: Any):
        """Persist FAISS index and chunk metadata to disk."""
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)
            
    @classmethod
    def load(cls, path: Any) -> "VectorStore":
        import faiss
        path = Path(path)
        
        index_path = path / "index.faiss"
        chunks_path = path / "chunks.json"
        
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
            
        index = faiss.read_index(str(index_path))
        
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
            
        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks
        return store
