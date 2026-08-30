"""Tests for the full Orchestrator Loop — Phase 3."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.orchestrator.groq_client import run_orchestrator_loop


@pytest.fixture
def mock_project_dir(tmp_storage_dir: Path) -> str:
    """Create a project with a sample CSV for prediction."""
    import uuid
    project_id = f"test_loop_{uuid.uuid4().hex[:8]}"
    from backend.config import settings
    raw_dir = settings.raw_dir / project_id
    raw_dir.mkdir(parents=True)
    
    from backend.db import init_db, SessionLocal, Project
    init_db()
    db = SessionLocal()
    proj = Project(id=project_id, name="Test Loop")
    db.add(proj)
    db.commit()
    db.close()
    
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    sensor_csv = temp_dir / "sensor_data.csv"
    sensor_csv.write_text(
        "machine_id,temp,vibration,failed\n"
        "M1,45.2,1.2,0\n"
        "M2,80.1,5.5,1\n"
        "M3,50.0,1.5,0\n"
        "M4,85.2,6.1,1\n"
    )
    
    from backend.ingestion.engine import ingest_files
    ingest_files([str(sensor_csv)], project_id)
    
    return project_id


def test_orchestrator_loop_end_to_end(mock_project_dir: str) -> None:
    """Test the full planning loop with a real Groq call.
    
    Scenario: tabular sample CSV + goal 'predict machine failure'.
    Asserts loop runs through steps and outputs an Experiment.
    """
    goal = "predict machine failure based on sensor data"
    
    result = run_orchestrator_loop(mock_project_dir, goal)
    
    assert result["status"] == "success"
    assert "experiment" in result
    
    exp = result["experiment"]
    assert exp["status"] == "created"
    assert exp["project_id"] == mock_project_dir
    assert "model_name" in exp
    assert "training_method" in exp
