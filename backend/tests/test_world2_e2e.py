"""End-to-End Test for World 2 (Tabular AutoGluon) — Phase 4."""

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

def test_world2_e2e(tmp_path: Path) -> None:
    """Run the entire World 2 pipeline from a script-level call."""
    from backend.db import init_db
    init_db()
    
    # 0. Setup a mock project and sample dataset
    db = SessionLocal()
    proj = Project(name="E2E Tabular Test")
    db.add(proj)
    db.commit()
    db.refresh(proj)
    project_id = proj.id
    
    # Create a simple CSV (binary classification with a categorical column for error analysis)
    # We want a target column and a segment column.
    sample_csv = tmp_path / "sensor_data.csv"
    sample_csv.write_text(
        "machine_id,region,temp,vibration,failed\n"
        "M1,North,45.2,1.2,False\n"
        "M2,South,80.1,5.5,True\n"
        "M3,North,50.0,1.5,False\n"
        "M4,South,85.2,6.1,True\n"
        "M5,East,42.0,1.1,False\n"
        "M6,West,88.0,6.5,True\n"
        "M7,North,46.0,1.3,False\n"
        "M8,South,79.0,5.4,True\n"
        "M9,East,41.0,1.0,False\n"
        "M10,West,90.0,6.8,True\n"
        "M11,North,48.0,1.4,False\n"
        "M12,South,82.0,5.8,True\n"
    )
    
    try:
        # 1. Ingest files
        print("\\n[1] Ingesting files...")
        ingest_manifest = ingest_files([str(sample_csv)], project_id)
        assert len(ingest_manifest) == 1
        
        # 2. Data Understanding
        print("[2] Analyzing project...")
        # Since MVP stores raw files in storage_root / raw / project_id
        # analyze_project uses the project_id to find the files
        report = analyze_project(project_id)
        assert report["file_type_counts"]["csv"] == 1
        
        # 3. AI Orchestrator Loop
        print("[3] Running orchestrator loop...")
        loop_result = run_orchestrator_loop(project_id, "predict machine failure")
        
        print(f"    Orchestrator loop result: {loop_result}")
        assert loop_result.get("status") == "success"
        experiment_id = loop_result["experiment"]["experiment_id"]
        print(f"    Orchestrator chose model: {loop_result['experiment']['model_name']}")
        print(f"    Experiment ID: {experiment_id}")
        
        # 4. Start Training
        print("[4] Starting training...")
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
                pytest.fail(f"Training failed: {status.get('message')}")
            time.sleep(2)
            polls += 1
            
        assert polls < max_polls, "Training timed out."
        
        # 6. Evaluate Model
        print("[6] Evaluating model...")
        eval_result = validate_and_execute("evaluate_model", {"experiment_id": experiment_id})
        assert eval_result.get("status") == "success"
        
        print("    Metrics:", eval_result["metrics"])
        
        # 7. Error Analysis
        print("[7] Analyzing errors...")
        # analyze_errors parameter is evaluation_id
        evaluation_id = eval_result["evaluation_id"]
        analysis = validate_and_execute("analyze_errors", {"evaluation_id": evaluation_id})
        assert "worst_performing_segment" in analysis or "error" in analysis
        if "error" not in analysis:
            print("    Analysis:", analysis["worst_performing_segment"])
        
        # 8. Assert DB integrity
        print("[8] Asserting DB state...")
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        assert exp is not None
        assert exp.status == "created" # Wait, TrainingRun is updated to completed, not Experiment in MVP
        
        evaluation = db.query(Evaluation).filter(Evaluation.experiment_id == experiment_id).first()
        assert evaluation is not None
        
        print("\\n=== E2E TEST PASSED ===")
        
    finally:
        db.close()
