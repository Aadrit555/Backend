"""Model registry (dynamic, DB-backed) — BIBLE §31, ARCHITECTURE.md §2.

Queries the SQLite database for trained ModelArtifacts, Experiments,
and Deployments — i.e. "what HAS been built".

This is deliberately separate from registry/loader.py (static
capabilities.yaml = "what CAN we do").
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from backend.db import Experiment, ModelArtifact, Deployment


def list_trained_models(db: Session, project_id: str | None = None) -> list[dict[str, Any]]:
    """Return all trained model artifacts, optionally filtered by project."""
    # TODO (Phase 1): query ModelArtifact joined with TrainingRun + Experiment
    return []


def get_experiment_detail(db: Session, experiment_id: str) -> dict[str, Any] | None:
    """Return full detail for an experiment including metrics."""
    # TODO (Phase 1)
    return None


def list_deployments(db: Session, project_id: str | None = None) -> list[dict[str, Any]]:
    """Return active deployments."""
    # TODO (Phase 1)
    return []
