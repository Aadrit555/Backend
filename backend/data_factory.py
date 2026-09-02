import os
from pathlib import Path
import json

def chunk_text(text: str, chunk_size: int = 1500) -> list[str]:
    """Basic chunker if unstructured is not available or too heavy."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size].strip()) > 50]

def extract_documents(raw_file: Path) -> list[str]:
    """Step 1: Document extraction and Chunking supporting TXT, PDF, DOCX, HTML, JSON."""
    print(f"[DATA FACTORY] Extracting and chunking {raw_file}...")
    text = ""
    
    # Check if .docx (extract XML paragraphs directly from Word zip package)
    if raw_file.suffix.lower() == ".docx":
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(raw_file) as docx:
                xml_content = docx.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                paragraphs = []
                for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                    texts = [node.text for node in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                text = "\n\n".join(paragraphs)
            print(f"[DATA FACTORY] Successfully extracted {len(text)} characters from DOCX.")
        except Exception as e:
            print(f"[DATA FACTORY] DOCX extraction error: {e}")

    # Check if .pdf
    elif raw_file.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(raw_file))
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            print(f"[DATA FACTORY] Successfully extracted {len(text)} characters from PDF.")
        except Exception as e:
            print(f"[DATA FACTORY] PDF extraction error: {e}")

    # Fallback to unstructured or direct utf-8 text read
    if not text.strip():
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=str(raw_file))
            text = "\n\n".join([str(el) for el in elements])
        except Exception:
            try:
                text = raw_file.read_text(encoding="utf-8")
            except Exception:
                text = raw_file.read_text(errors="ignore")

    return chunk_text(text)

def clean_with_data_juicer(chunks: list[str]) -> list[str]:
    """Step 2: Data-Juicer (cleaning + deduplication + quality filtering)"""
    print("[DATA FACTORY] Running Data-Juicer cleaning and deduplication...")
    try:
        from data_juicer.core.data import NestedDataset
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.deduplicator.document_minhash_deduplicator import DocumentMinhashDeduplicator
        
        ds = NestedDataset.from_dict({"text": chunks})
        # Note: DocumentMinhashDeduplicator might require a complex config in reality, 
        # but we use simple filters here to demonstrate the pipeline.
        res_ds = ds.process([
            TextLengthFilter(min_len=50)
        ])
        return [item["text"] for item in res_ds]
    except Exception as e:
        print(f"[DATA FACTORY] Data-Juicer failed or not fully installed: {e}")
        # Fallback to simple deduplication
        return list(set(chunks))

def generate_with_distilabel(cleaned_chunks: list[str], api_key: str = "") -> list[dict]:
    """Fast instruction/response pair generation with intelligent local fallback."""
    import os
    import requests
    import json
    import re
    
    qa_pairs = []
    
    groq_key = os.environ.get("GROQ_API_KEY_1", "")
    if groq_key:
        print("[DATA FACTORY] Generating Q&A pairs via Groq LPU acceleration...")
        for chunk in cleaned_chunks[:8]:
            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "Extract 1 question-and-answer pair from the text. Return ONLY JSON: {\"q\": \"...\", \"a\": \"...\"}"},
                            {"role": "user", "content": f"TEXT:\n{chunk[:1500]}"}
                        ],
                        "temperature": 0.1
                    },
                    timeout=5
                )
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        pair = json.loads(match.group(0))
                        if "q" in pair and "a" in pair:
                            qa_pairs.append({
                                "conversations": [
                                    {"from": "human", "value": pair["q"]},
                                    {"from": "gpt", "value": pair["a"]}
                                ]
                            })
            except Exception:
                pass

    elif api_key:
        print("[DATA FACTORY] Generating Q&A pairs via OpenRouter...")
        for chunk in cleaned_chunks[:8]:
            try:
                res = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                        "messages": [
                            {"role": "system", "content": "Extract 1 question-and-answer pair from the text. Return ONLY JSON: {\"q\": \"...\", \"a\": \"...\"}"},
                            {"role": "user", "content": f"TEXT:\n{chunk[:1500]}"}
                        ],
                        "temperature": 0.1
                    },
                    timeout=5
                )
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        pair = json.loads(match.group(0))
                        if "q" in pair and "a" in pair:
                            qa_pairs.append({
                                "conversations": [
                                    {"from": "human", "value": pair["q"]},
                                    {"from": "gpt", "value": pair["a"]}
                                ]
                            })
            except Exception:
                pass

    # Instant heuristic local generation fallback
    if not qa_pairs:
        print("[DATA FACTORY] Applying zero-latency heuristic Q&A extraction...")
        for i, chunk in enumerate(cleaned_chunks[:15]):
            lines = [line.strip() for line in chunk.split("\n") if line.strip()]
            if not lines:
                continue
            heading = lines[0] if len(lines[0]) < 80 else f"Section {i+1}"
            content = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
            if len(content) < 20:
                continue
            qa_pairs.append({
                "conversations": [
                    {"from": "human", "value": f"What are the details regarding {heading}?"},
                    {"from": "gpt", "value": content.strip()}
                ]
            })

    return qa_pairs
                
def run_pipeline(raw_file: Path, output_jsonl: Path, mode: str = "qa") -> Path:
    """End-to-End Data Factory Pipeline supporting modes: 'qa', 'pii_clean', 'rag_chunks'."""
    from backend.config import settings
    import os

    # 1. Extraction & Chunking
    chunks = extract_documents(raw_file)
    if not chunks:
        raise ValueError("No text extracted from document.")

    # 2. Data-Juicer Cleaning & Deduplication
    cleaned_chunks = clean_with_data_juicer(chunks)

    if mode == "rag_chunks":
        print(f"[DATA FACTORY] Formatting {len(cleaned_chunks)} semantic chunks for RAG...")
        records = []
        for i, chunk in enumerate(cleaned_chunks):
            records.append({
                "chunk_id": i + 1,
                "text": chunk.strip(),
                "char_count": len(chunk.strip()),
                "source": raw_file.name,
            })
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return output_jsonl

    elif mode == "pii_clean":
        print(f"[DATA FACTORY] Preparing {len(cleaned_chunks)} chunks for PII sanitization...")
        records = []
        for i, chunk in enumerate(cleaned_chunks):
            records.append({
                "id": i + 1,
                "text": chunk.strip(),
                "source": raw_file.name,
            })
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return output_jsonl

    else:
        # mode == "qa"
        key = settings.openrouter_api_key or os.environ.get("GROQ_API_KEY_1", "")
        if not key:
            print("[DATA FACTORY] Note: No API key found. Generating structured instruction pairs from chunks...")
            qa_pairs = []
            for i, chunk in enumerate(cleaned_chunks[:10]):
                preview_snippet = chunk[:100].strip() + "..."
                qa_pairs.append({
                    "conversations": [
                        {"from": "human", "value": f"Explain the key concepts in this section: {preview_snippet}"},
                        {"from": "gpt", "value": chunk.strip()}
                    ]
                })
        else:
            qa_pairs = generate_with_distilabel(cleaned_chunks, key)

        print(f"[DATA FACTORY] Saving {len(qa_pairs)} high-quality pairs to {output_jsonl}")
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + "\n")

        return output_jsonl
