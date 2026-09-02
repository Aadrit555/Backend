_rag_llm = None
_rag_tokenizer = None

def get_rag_llm():
    """Load local Llama-3.2-1B model on GPU for genuine RAG generation."""
    global _rag_llm, _rag_tokenizer
    if _rag_llm is None:
        print("[RAG] Initializing local Llama-3.2-1B on GPU for real-time generative RAG...")
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            max_seq_length=1024,
            load_in_4bit=True,
            device_map="cuda"
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        _rag_llm = model
        _rag_tokenizer = tokenizer
    return _rag_llm, _rag_tokenizer

def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate an answer to the query using only the provided context chunks.
    Uses local Llama-3.2 on GPU for genuine, conversational answers.
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
        "You are an intelligent, factual AI assistant. Answer the user's question directly, concisely, and accurately based ONLY on the provided context.\n"
        "1. Answer ONLY what was asked using clean formatting, bullet points, and concise language.\n"
        "2. Do not dump the entire raw context or irrelevant sections.\n"
        "3. If the context does not contain the answer, say 'The provided document does not contain this information.'"
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    answer = None
    try:
        llm, tok = get_rag_llm()
        inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = llm.generate(input_ids=inputs, max_new_tokens=256, temperature=0.1, use_cache=True, pad_token_id=tok.eos_token_id)
        answer = tok.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    except Exception as e:
        print(f"[RAG] Local LLM error: {e}")
        
    if not answer:
        top_chunk = retrieved_chunks[0]
        src = top_chunk.get("metadata", {}).get("source", "document")
        answer = f"**[Source: {src}]**\n\n" + top_chunk["text"].strip()
        
    return {
        "answer": answer,
        "citations": retrieved_chunks
    }
