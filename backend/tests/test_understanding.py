"""Tests for the Data Understanding Engine — Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.understanding.engine import analyze_file, analyze_project, detect_relationships


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """A CSV with obvious ID and target columns."""
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "customer_id,name,age,churned,country\n"
        "C001,Alice,30,1,USA\n"
        "C002,Bob,25,0,UK\n"
        "C003,Charlie,35,0,USA\n"
        "C004,David,,1,CA\n"  # null age
        "C005,Eve,28,0,UK\n"
    )
    return csv_path


@pytest.fixture
def mock_project_dir(tmp_storage_dir: Path, sample_csv: Path) -> str:
    """Create a project with multiple files to test relationships."""
    from backend.db import init_db, SessionLocal, Project
    from backend.ingestion.engine import ingest_files
    import uuid
    
    init_db()
    project_id = f"test_rel_{uuid.uuid4().hex[:8]}"
    
    db = SessionLocal()
    proj = Project(id=project_id, name="Test Rel")
    db.add(proj)
    db.commit()
    db.close()
    
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    # File 1: Customers (from fixture)
    customers_csv = temp_dir / "customers.csv"
    import shutil
    shutil.copy2(sample_csv, customers_csv)
    
    # File 2: Machines
    machines_csv = temp_dir / "machines.csv"
    machines_csv.write_text(
        "machine_id,customer_id,os\n"
        "M001,C001,Linux\n"
        "M002,C001,Windows\n"
        "M003,C002,macOS\n"
    )
    
    # File 3: Logs
    logs_csv = temp_dir / "logs.csv"
    logs_csv.write_text(
        "log_id,machine_id,event\n"
        "L1,M001,login\n"
        "L2,M001,logout\n"
        "L3,M003,login\n"
    )
    
    # File 4: Unrelated JSON
    json_path = temp_dir / "unrelated.json"
    import json
    json_path.write_text(json.dumps([{"x": 1}, {"x": 2}]))
    
    ingest_files([str(customers_csv), str(machines_csv), str(logs_csv), str(json_path)], project_id)
    
    return project_id


def test_analyze_file_csv_heuristics(sample_csv: Path) -> None:
    """Test column type inference, ID, and target heuristics."""
    report = analyze_file(sample_csv, "csv")
    
    assert report["type"] == "tabular"
    assert report["row_count"] == 5
    assert len(report["columns"]) == 5
    
    # customer_id: cardinality 5, no nulls -> likely ID
    assert report["likely_id_column"] == "customer_id"
    
    # churned: cardinality 2 (values 0,1), no nulls -> likely Target
    # country: cardinality 3, no nulls -> could be target, but churned has lower cardinality
    assert report["likely_target_column"] == "churned"
    
    # Check null inference on 'age'
    age_col = next(c for c in report["columns"] if c["name"] == "age")
    assert age_col["null_count"] == 1


def test_analyze_file_json(tmp_path: Path) -> None:
    """Test JSON structure and schema inference."""
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([
        {"id": 1, "name": "Alice", "active": True},
        {"id": 2, "name": "Bob"}
    ]))
    
    report = analyze_file(json_path, "json")
    
    assert report["type"] == "json"
    assert report["structure"] == "list"
    assert "id" in report["inferred_schema"]
    assert report["inferred_schema"]["id"] == "int"
    assert report["inferred_schema"]["name"] == "str"
    assert report["inferred_schema"]["active"] == "bool"


def test_analyze_project_aggregation(mock_project_dir: str) -> None:
    """Test analyze_project processes all files."""
    report = analyze_project(mock_project_dir)
    
    assert report["project_id"] == mock_project_dir
    assert report["file_type_counts"]["csv"] == 3
    assert report["file_type_counts"]["json"] == 1
    
    filenames = [s["filename"] for s in report["sources"]]
    assert "customers.csv" in filenames
    assert "machines.csv" in filenames
    assert "unrelated.json" in filenames


def test_detect_relationships(mock_project_dir: str) -> None:
    """Test detection of shared columns across tabular files."""
    graph = detect_relationships(mock_project_dir)
    
    assert len(graph["nodes"]) == 3 # 3 CSVs
    
    # Edges should exist between customers <-> machines (customer_id)
    # and machines <-> logs (machine_id)
    edges = graph["edges"]
    
    cust_mach = [e for e in edges if e["shared_column"] == "customer_id"]
    assert len(cust_mach) == 1
    assert set([cust_mach[0]["from"], cust_mach[0]["to"]]) == {"customers.csv", "machines.csv"}
    
    mach_logs = [e for e in edges if e["shared_column"] == "machine_id"]
    assert len(mach_logs) == 1
    assert set([mach_logs[0]["from"], mach_logs[0]["to"]]) == {"machines.csv", "logs.csv"}
