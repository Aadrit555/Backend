"""Shared test fixtures.

Provides:
  - tmp_storage_dir: a temporary storage root for tests
  - sample_csv_path: a small CSV file for adapter integration tests
  - db_session: a clean SQLite in-memory session
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base


@pytest.fixture
def tmp_storage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary storage directory tree and patch settings."""
    from backend.config import settings
    monkeypatch.setattr(settings, "storage_root", tmp_path)
    for sub in ("raw", "processed", "models", "experiments", "logs"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Generate a small sample CSV for tabular adapter tests."""
    csv_path = tmp_path / "sample.csv"
    rows = [
        {"feature_a": 1.0, "feature_b": "cat", "target": 0},
        {"feature_a": 2.5, "feature_b": "dog", "target": 1},
        {"feature_a": 3.1, "feature_b": "cat", "target": 0},
        {"feature_a": 0.8, "feature_b": "dog", "target": 1},
        {"feature_a": 4.2, "feature_b": "cat", "target": 0},
        {"feature_a": 1.9, "feature_b": "dog", "target": 1},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_a", "feature_b", "target"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def db_session():
    """Create a clean in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
