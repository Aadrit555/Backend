"""End-to-End Test for World 4 (RAG) — Phase 6."""

from __future__ import annotations

import time
import pytest
from pathlib import Path

from backend.db import SessionLocal, Project, Experiment, Evaluation
from backend.ingestion.engine import ingest_files
from backend.understanding.engine import analyze_project
from backend.orchestrator.groq_client import run_orchestrator_loop
from backend.orchestrator.validation_gate import validate_and_execute
from backend.status import _read_status
from backend.config import settings

def test_rag_e2e(tmp_path: Path) -> None:
    """Run the entire RAG pipeline from a script-level call."""
    from backend.db import init_db
    init_db()
    
    # 0. Setup a mock project
    db = SessionLocal()
    proj = Project(name="E2E RAG Test")
    db.add(proj)
    db.commit()
    db.refresh(proj)
    project_id = proj.id
    
    # Create sample docs
    doc1 = tmp_path / "company_policy.txt"
    doc1.write_text(
        "Company Policy 2026:\\n"
        "All employees must work from the office on Tuesdays and Thursdays.\\n"
        "The standard vacation allowance is 25 days per year.\\n"
        "Expenses for remote work setups are capped at $500."
    )
    
    doc2 = tmp_path / "product_roadmap.txt"
    doc2.write_text(
        "Product Roadmap Q3:\\n"
        "We are launching the new Unified AI Platform in September.\\n"
        "Key features include RAG, Tabular AutoML, and Agentic Data Preparation.\\n"
        "The main target audience is data scientists and enterprise developers."
    )
    
    try:
        # 1. Ingest files
        print("\\n[1] Ingesting files...")
        ingest_manifest = ingest_files([str(doc1), str(doc2)], project_id)
        assert len(ingest_manifest) == 2
        
        # 2. Data Understanding
        print("[2] Analyzing project...")
        report = analyze_project(project_id)
        assert report["file_type_counts"]["txt"] == 2
        
        # 3. AI Orchestrator Loop
        print("[3] Running orchestrator loop...")
        # Give a specific prompt so it selects rag_default
        loop_result = run_orchestrator_loop(project_id, "answer questions about these text documents using RAG")
        
        print(f"    Orchestrator loop result: {loop_result}")
        assert loop_result.get("status") == "success"
        experiment_id = loop_result["experiment"]["experiment_id"]
        model_chosen = loop_result["experiment"]["model_name"]
        print(f"    Orchestrator chose model: {model_chosen}")
        print(f"    Experiment ID: {experiment_id}")
        
        # Usually it should choose rag_default or something similar, depending on how smart the LLM is.
        # It might choose a different model if the prompt isn't perfect, but since we asked for RAG, 
        # and rag_default has task="rag", it should pick it.
        
        # 4. Start Training (Indexing)
        print("[4] Starting indexing (training step)...")
        start_result = validate_and_execute("start_training", {"experiment_id": experiment_id})
        assert start_result.get("status") == "started"
        
        # 5. Poll Status
        print("[5] Polling status...")
        max_polls = 60
        polls = 0
        last_pct = -1
        while polls < max_polls:
            status = _read_status(experiment_id)
            current_pct = status.get("pct", 0)
            print(f"    Status: {status.get('stage')} - {current_pct}%: {status.get('message')}")
            assert current_pct >= last_pct, "Percentage should increase monotonically"
            last_pct = current_pct
            
            if status.get("stage") == "completed":
                break
            if status.get("stage") == "failed":
                pytest.fail(f"Indexing failed: {status.get('message')}")
            time.sleep(2)
            polls += 1
            
        assert polls < max_polls, "Indexing timed out."
        
        # 6. Evaluate Model (Retrieval check)
        print("[6] Evaluating retrieval...")
        eval_result = validate_and_execute("evaluate_model", {"experiment_id": experiment_id})
        assert eval_result.get("status") == "success"
        print("    Metrics:", eval_result["metrics"])
        
        # 7. Real Generation Query
        print("[7] Testing Live RAG Generation...")
        from backend.rag.vector_store import VectorStore
        from backend.rag.embeddings import embed_chunks
        from backend.rag.generator import generate_answer
        
        # Load the saved index from the exported artifact directory
        model_path = settings.experiments_dir / experiment_id / "export"
        store = VectorStore.load(model_path)
        
        query = "How many vacation days do employees get?"
        q_emb, _ = embed_chunks([{"text": query}])
        retrieved = store.retrieve(q_emb[0], k=3)
        
        print(f"    Query: {query}")
        print(f"    Retrieved Chunks: {len(retrieved)}")
        assert len(retrieved) > 0
        
        answer_payload = generate_answer(query, retrieved)
        print(f"    LLM Answer (Happy Path): {answer_payload['answer']}")
        
        # Verify the citation is in the context
        assert "company_policy.txt" in answer_payload["answer"] or "company_policy" in str(answer_payload["citations"])

        # 8. Adversarial Grounding Test
        print("[8] Testing Adversarial Grounding...")
        adv_query = "What is the secret recipe for the Krabby Patty?"
        adv_emb, _ = embed_chunks([{"text": adv_query}])
        adv_retrieved = store.retrieve(adv_emb[0], k=3)
        
        print(f"    Query: {adv_query}")
        adv_answer = generate_answer(adv_query, adv_retrieved)
        print(f"    LLM Answer (Adversarial): {adv_answer['answer']}")
        
        lower_ans = adv_answer['answer'].lower()
        assert "i don't know" in lower_ans or "not in the provided context" in lower_ans or "not provided" in lower_ans or "cannot answer" in lower_ans, "Model hallucinated instead of declining!"
        refusal_phrases = [
            "i don't know", "not in the provided context", "not provided",
            "cannot answer", "no mention", "not mentioned", "does not contain",
            "not contain", "does not mention", "no information", "unavailable",
            "unknown", "not state", "not stated", "cannot find"
        ]
        assert any(phrase in lower_ans for phrase in refusal_phrases), f"Model hallucinated instead of declining: {adv_answer['answer']}"
        
        print("\\n=== RAG E2E TEST PASSED ===")
        print("\n=== RAG E2E TEST PASSED ===")
        
    finally:
        db.close()
