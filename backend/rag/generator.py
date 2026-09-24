from typing import Any
import os
from groq import Groq
from dotenv import load_dotenv

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        # Load from the root .env where the API keys actually live
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        load_dotenv(dotenv_path=env_path)
        
        api_key = os.environ.get("GROQ_API_KEY_1")
        if not api_key:
            raise ValueError("GROQ_API_KEY_1 is missing from environment variables.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate an answer to the query using only the provided context chunks.
    Uses Groq for lightning fast, conversational answers.
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
        "You are an expert, helpful AI assistant analyzing a provided document.\n"
        "Your task is to answer the user's question accurately, directly, and comprehensively based on the context:\n"
        "- If asked where someone studies, their university, or education: state their university (e.g. SRM University AP), degree, and coursework clearly.\n"
        "- If asked about projects: list their projects and key technical implementations with concise bullet points.\n"
        "- If asked about skills, experience, or background: summarize the relevant details directly.\n"
        "- Format your answer with clean markdown bullet points and bold highlights.\n"
        "- Answer concisely without dumping unrelated text."
    )
    user_prompt = f"Document Context:\n\"\"\"\n{context_text}\n\"\"\"\n\nUser Question: {query}"
    
    answer = None
    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
    except Exception as e:
        print(f"[RAG] Groq API error: {e}")
        
    if not answer:
        top_chunk = retrieved_chunks[0]
        src = top_chunk.get("metadata", {}).get("source", "document")
        answer = f"**[Source: {src}]**\n\n" + top_chunk["text"].strip()
        
    return {
        "answer": answer,
        "citations": retrieved_chunks
    }
