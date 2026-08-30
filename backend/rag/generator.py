"""Generation logic for RAG."""

from typing import Any
from backend.orchestrator.groq_client import chat

def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate an answer to the query using only the provided context chunks.
    Instructs the LLM to cite its sources and to say "I don't know" if the context
    doesn't contain the answer.
    """
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
        "2. If you answer the question, you MUST explicitly cite the source document for your information using the exact metadata provided in the context blocks. For example, '[Source: file.pdf, Page: 2]' or '[Source: file.docx]' if no page is provided.\n"
        "3. Answer concisely and accurately."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = chat(messages, tools=[])
    
    return {
        "answer": response.get("content", "Error generating response"),
        "citations": retrieved_chunks
    }
