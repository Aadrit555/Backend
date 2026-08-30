"""AutoGluon adapter integration test — BIBLE §19 (World 2: Tabular).

Tests the full prepare → train → evaluate → export cycle against a
small sample CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.adapters.autogluon import AutoGluonAdapter

class TestAutoGluonAdapter:
    """Integration tests for the AutoGluon adapter."""

    def test_capabilities_returns_expected_keys(self) -> None:
        adapter = AutoGluonAdapter()
        caps = adapter.capabilities()
        assert "supported_tasks" in caps
        assert "classification" in caps["supported_tasks"]
        assert "supported_export_formats" in caps

    def test_prepare_creates_dataset(self, sample_csv_path: Path, tmp_storage_dir: Path) -> None:
        adapter = AutoGluonAdapter()
        prepared_dir = adapter.prepare(sample_csv_path, {"target_column": "target"})
        assert prepared_dir.exists()
        assert prepared_dir.is_dir()
        assert (prepared_dir / "train.csv").exists()
        assert (prepared_dir / "val.csv").exists()
        assert (prepared_dir / "config.json").exists()

    def test_train_produces_artifact(self, sample_csv_path: Path, tmp_storage_dir: Path) -> None:
        adapter = AutoGluonAdapter()
        prepared_dir = adapter.prepare(sample_csv_path, {"target_column": "target"})
        
        result = adapter.train(prepared_dir, {"target_column": "target", "time_limit": 10})
        
        assert result.artifact_path.exists()
        assert (result.artifact_path / "predictor.pkl").exists()
        assert "best_val_score" in result.metrics
        assert "best_model" in result.metrics
        
        # Check that status.json was written
        experiments_dir = tmp_storage_dir / "experiments"
        exp_dirs = list(experiments_dir.glob("exp_*"))
        assert len(exp_dirs) > 0
        status_file = exp_dirs[0] / "status.json"
        assert status_file.exists()

    def test_evaluate_returns_metrics(self, sample_csv_path: Path, tmp_storage_dir: Path) -> None:
        adapter = AutoGluonAdapter()
        prepared_dir = adapter.prepare(sample_csv_path, {"target_column": "target"})
        train_result = adapter.train(prepared_dir, {"target_column": "target", "time_limit": 10})
        
        eval_result = adapter.evaluate(train_result.artifact_path, prepared_dir, {"target_column": "target"})
        assert isinstance(eval_result.metrics["accuracy"], float)
        assert eval_result.metrics["accuracy"] >= 0.0

    def test_export_produces_file(self, sample_csv_path: Path, tmp_storage_dir: Path) -> None:
        adapter = AutoGluonAdapter()
        prepared_dir = adapter.prepare(sample_csv_path, {"target_column": "target"})
        train_result = adapter.train(prepared_dir, {"target_column": "target", "time_limit": 10})
        
        output_path = tmp_storage_dir / "exported_model"
        result_path = adapter.export(train_result.artifact_path, "predictor_dir", output_path)
        
        assert result_path.exists()
        assert result_path.is_dir()
        assert (result_path / "predictor.pkl").exists()
