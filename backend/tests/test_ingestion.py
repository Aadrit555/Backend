"""Tests for the Data Ingestion Engine — Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ingestion.engine import detect_file_type, ingest_files


@pytest.fixture
def sample_mixed_files(tmp_path: Path) -> dict[str, Path]:
    """Create a temporary directory with mixed file types."""
    files = {}
    
    # 1. Real CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name,age\n1,Alice,30\n2,Bob,25")
    files["csv"] = csv_path
    
    # 2. Fake CSV (actually text)
    fake_csv_path = tmp_path / "fake.csv"
    fake_csv_path.write_text("This is just some random prose. Not tabular at all.")
    files["fake_csv"] = fake_csv_path
    
    # 3. JSON list
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    files["json"] = json_path
    
    # 4. JSONL
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text('{"a": 1}\n{"a": 2}')
    files["jsonl"] = jsonl_path
    
    # 5. Fake JSON (invalid syntax)
    fake_json_path = tmp_path / "fake.json"
    fake_json_path.write_text("{this is not valid json")
    files["fake_json"] = fake_json_path
    
    # 6. TXT
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Hello world")
    files["txt"] = txt_path
    
    # 7. Tiny PNG
    png_path = tmp_path / "image.png"
    # A valid 1x1 transparent PNG
    png_path.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\x0dIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\x0bIDAT\\x08\\xd7c\\xfa\\xcf\\x00\\x00\\x00\\x02\\x00\\x01\\xe2!\\xbc3\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82")
    files["png"] = png_path
    
    return files


def test_detect_file_type(sample_mixed_files: dict[str, Path]) -> None:
    """Test detect_file_type accurately identifies files using sniffing."""
    assert detect_file_type(sample_mixed_files["csv"]) == "csv"
    assert detect_file_type(sample_mixed_files["fake_csv"]) == "txt" # sniff fails, falls back to txt
    assert detect_file_type(sample_mixed_files["json"]) == "json"
    assert detect_file_type(sample_mixed_files["jsonl"]) == "json"
    assert detect_file_type(sample_mixed_files["fake_json"]) == "txt" # sniff fails, falls back to txt
    assert detect_file_type(sample_mixed_files["txt"]) == "txt"
    assert detect_file_type(sample_mixed_files["png"]) == "image"


def test_ingest_files(sample_mixed_files: dict[str, Path], tmp_storage_dir: Path) -> None:
    """Test ingest_files copies files and returns a manifest."""
    source_paths = [str(p) for p in sample_mixed_files.values()]
    project_id = "test_proj_123"
    
    manifest = ingest_files(source_paths, project_id)
    
    assert len(manifest) == len(source_paths)
    
    # Verify files landed in storage/raw/ untouched
    from backend.config import settings
    raw_dir = settings.raw_dir / project_id
    assert raw_dir.exists()
    
    from backend.db import SessionLocal, DataSource, DataFile
    db = SessionLocal()
    try:
        for entry in manifest:
            filename = entry["filename"]
            dest_path = raw_dir / filename
            assert dest_path.exists()
            
            # Verify DB row
            ds = db.query(DataSource).filter_by(id=entry["id"]).first()
            assert ds is not None
            assert ds.project_id == project_id
            assert ds.original_filename == filename
            assert ds.file_type == entry["file_type"]
            
            df = db.query(DataFile).filter_by(datasource_id=ds.id).first()
            assert df is not None
            assert Path(df.path).name == filename
            
            # Verify sizes match
            source_path = [p for p in sample_mixed_files.values() if p.name == filename][0]
            assert dest_path.stat().st_size == source_path.stat().st_size
            assert entry["size_bytes"] == source_path.stat().st_size
            
            # Check some expected types from manifest
            if filename == "data.csv":
                assert entry["file_type"] == "csv"
            elif filename == "fake.csv":
                assert entry["file_type"] == "txt"
            
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64
    finally:
        db.close()
