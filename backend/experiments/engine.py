"""Experiment Engine — BIBLE §21-22, ARCHITECTURE.md §1.

Manages experiments: each experiment is a specific
(model, dataset_version, config, backend) combination.

Responsibilities:
  - CRUD for experiments
  - Launch training as a background subprocess (no Redis — just subprocess.Popen)
  - Compare metrics across experiments in the same project
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from sqlalchemy.orm import Session


def create_experiment(
    db: Session,
    project_id: str,
    dataset_version_id: str,
    model_name: str,
    backend: str,
    training_method: str,
    config_json: str | None = None,
) -> dict[str, Any]:
    """Register a new experiment.

    Backs the create_experiment tool (ARCHITECTURE.md §4).
    """
    # TODO (Phase 1): insert Experiment row, return experiment_id
    raise NotImplementedError


def start_training(db: Session, experiment_id: str) -> dict[str, Any]:
    """Launch a training run as a background subprocess.

    Spawns: python -m backend.run_training <experiment_id>
    Updates the Experiment status and stores the PID.
    Backs the start_training tool.
    """
    # TODO (Phase 1):
    #   proc = subprocess.Popen(
    #       [sys.executable, "-m", "backend.run_training", experiment_id],
    #   )
    #   Store proc.pid, set status = "running"
    raise NotImplementedError


def get_training_status(experiment_id: str) -> dict[str, Any]:
    """Read the status.json for a training run.

    Backs the get_training_status tool.
    """
    # TODO (Phase 1): delegate to status.py._read_status
    raise NotImplementedError


def compare_models(db: Session, experiment_ids: list[str]) -> dict[str, Any]:
    """Compare metrics across multiple experiments.

    Backs the compare_models tool (BIBLE §22).
    """
    # TODO (Phase 1)
    raise NotImplementedError
