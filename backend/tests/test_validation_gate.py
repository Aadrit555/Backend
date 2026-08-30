"""Tests for the Validation Gate — Phase 3."""

from __future__ import annotations

import pytest

from backend.orchestrator.validation_gate import validate_and_execute


def test_validation_gate_missing_param() -> None:
    """Test schema validation catches missing required parameters."""
    # create_experiment requires project_id, model_name, training_method
    args = {"project_id": "proj_123"}
    
    result = validate_and_execute("create_experiment", args)
    
    assert "error" in result
    assert result["error"] == "schema_error"
    assert "is a required property" in result["detail"]


def test_validation_gate_infeasible_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test VRAM gate blocks oversized models."""
    # Mock GPU probe to report 8000 VRAM
    import backend.orchestrator.validation_gate
    monkeypatch.setattr(backend.orchestrator.validation_gate, "get_max_free_vram_mb", lambda: 8000)
    
    # We need a capability registry that has a model needing > 8000 MB
    import backend.registry.loader
    mock_registry = {
        "models": {
            "massive-model": {
                "id": "massive-model",
                "backends": ["test_backend"],
                "training_methods": ["fine_tuning"],
                "vram_estimates": {
                    "fine_tuning": {"min_mb": 16000, "recommended_mb": 24000}
                }
            }
        }
    }
    import backend.orchestrator.validation_gate
    monkeypatch.setattr(backend.orchestrator.validation_gate, "get_registry", lambda: mock_registry)
    
    args = {
        "project_id": "proj_123",
        "dataset_version_id": "ds_1",
        "model_name": "massive-model",
        "backend": "test_backend",
        "training_method": "fine_tuning"
    }
    
    result = validate_and_execute("create_experiment", args)
    
    assert "error" in result
    assert result["error"] == "vram_insufficient"
    assert result["required_mb"] == 16000
    assert result["available_mb"] == 8000
    assert result["suggestions"] == []
