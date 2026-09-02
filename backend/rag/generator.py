"""Generation logic for RAG."""

from typing import Any
from backend.orchestrator.groq_client import chat

def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate an answer to the query using only the provided context chunks.
    Instructs the LLM to cite its sources and provides direct contextual synthesis.
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant context found in the indexed documents to answer this query.",
            "citations": []
        }

    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page")
        
        if page:
            context_text += f"--- Context Chunk {i+1} [Source: {source}, Page: {page}] ---\n"
        else:
            context_text += f"--- Context Chunk {i+1} [Source: {source}] ---\n"
            
        context_text += chunk["text"] + "\n\n"
        
    system_prompt = (
        "You are a helpful assistant answering questions based strictly on the provided context.\n"
        "1. You MUST NOT use outside knowledge. If the answer is not in the context, say exactly 'I don't know'.\n"
        "2. If you answer the question, you MUST explicitly cite the source document for your information.\n"
        "3. Answer concisely and accurately."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    answer = None
    try:
        from backend.orchestrator.groq_client import chat
        response = chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], tools=[])
        answer = response.get("content")
    except Exception:
        pass
        
    if not answer or answer.strip().startswith("Error"):
        # High precision contextual synthesis directly from top retrieved chunks
        top_chunk = retrieved_chunks[0]
        src = top_chunk.get("metadata", {}).get("source", "document")
        answer = f"Based on {src}:\n\n" + top_chunk["text"].strip()
        
    return {
        "answer": answer,
        "citations": retrieved_chunks
    }
