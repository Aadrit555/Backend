"""Document chunking for RAG."""

from typing import Any

def chunk_document(doc: dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> list[dict[str, Any]]:
    """Split a document dictionary (text + metadata) into sliding window chunks."""
    text = doc["text"]
    metadata = doc["metadata"]
    
    # Simple word-based chunking for MVP
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "metadata": metadata.copy()
        })
        
        # If we reached the end, stop
        if i + chunk_size >= len(words):
            break
            
    return chunks

def chunk_documents(docs: list[dict[str, Any]], chunk_size: int = 500, overlap: int = 50) -> list[dict[str, Any]]:
    """Process a list of document dicts into a flat list of chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
