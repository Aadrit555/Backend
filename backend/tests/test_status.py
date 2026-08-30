"""Tests for the status polling endpoint — Phase 4."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.run_training import _write_status

client = TestClient(app)


def test_status_polling(tmp_storage_dir: Path) -> None:
    """Simulate training and poll the status endpoint."""
    experiment_id = "test_status_exp"
    
    # Pre-create the directory so _write_status doesn't fail
    # tmp_storage_dir fixture overrides settings.storage_root
    from backend.config import settings
    exp_dir = settings.experiments_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initial status before file exists
    response = client.get(f"/api/experiments/{experiment_id}/status")
    assert response.status_code == 200
    assert response.json()["stage"] == "unknown"
    
    # 2. Write a status
    _write_status(experiment_id, "starting", 0, "Loading config...")
    
    response = client.get(f"/api/experiments/{experiment_id}/status")
    assert response.status_code == 200
    assert response.json()["stage"] == "starting"
    assert response.json()["pct"] == 0
    
    # 3. Write another status
    _write_status(experiment_id, "training", 50, "Training model...")
    
    response = client.get(f"/api/experiments/{experiment_id}/status")
    assert response.status_code == 200
    assert response.json()["stage"] == "training"
    assert response.json()["pct"] == 50
    
    # 4. Write final status
    _write_status(experiment_id, "completed", 100, "Done.")
    
    response = client.get(f"/api/experiments/{experiment_id}/status")
    assert response.status_code == 200
    assert response.json()["stage"] == "completed"
    assert response.json()["pct"] == 100
