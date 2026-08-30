"""Deployment Engine — BIBLE §30, ARCHITECTURE.md §1.

Local-only deployment for MVP (no cloud — locked decision):
  - chat:        Launch a chat interface (e.g. Gradio) for LLMs
  - rest_api:    Spin up a FastAPI prediction endpoint
  - export_file: Export the model artifact for download
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


def deploy_model(
    db: Session,
    experiment_id: str,
    deploy_type: str,
) -> dict[str, Any]:
    """Deploy a trained model locally.

    Backs the deploy_model tool (ARCHITECTURE.md §4).

    Parameters
    ----------
    deploy_type : str
        One of: "chat", "rest_api", "export_file".
    """
    # TODO (Phase 1)
    raise NotImplementedError


def stop_deployment(db: Session, deployment_id: str) -> dict[str, Any]:
    """Stop a running deployment."""
    # TODO (Phase 1)
    raise NotImplementedError
