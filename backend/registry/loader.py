"""Capability registry loader — BIBLE §16, ARCHITECTURE.md §5.

Loads and queries the static capabilities.yaml at startup.
This is "what CAN we do" — distinct from model_registry.py which is
"what HAVE we built" (DB-backed).

Public API (used by orchestrator/tools.py dispatch and validation_gate.py):
  - load_registry()          -> full dict
  - get_model_capabilities() -> single model entry or KeyError
  - list_models_for_task()   -> filtered list
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REGISTRY_PATH = Path(__file__).resolve().parent / "capabilities.yaml"
_CACHE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, Any]:
    """Load capabilities.yaml and return the full registry dict (cached).

    Returns a dict with top-level keys "models" and "backends".
    """
    global _CACHE
    if _CACHE is None:
        with open(_REGISTRY_PATH) as f:
            _CACHE = yaml.safe_load(f) or {}
    return _CACHE


# Keep the old name as an alias so existing imports don't break.
get_registry = load_registry


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_model_capabilities(model_id: str) -> dict[str, Any]:
    """Return the full capability entry for a single model.

    Raises KeyError if the model_id is not in the registry.
    This is what the orchestrator's get_model_capabilities tool calls.
    """
    registry = load_registry()
    models = registry.get("models", {})
    if model_id not in models:
        raise KeyError(
            f"Model '{model_id}' not found in capability registry. "
            f"Available: {list(models.keys())}"
        )
    return {"id": model_id, **models[model_id]}


def list_models_for_task(task: str) -> list[dict[str, Any]]:
    """Return all models that support the given task type.

    Example tasks: "classification", "regression", "fine_tuning",
    "object_detection", "text_generation", etc.
    """
    registry = load_registry()
    results = []
    for name, info in registry.get("models", {}).items():
        if task in info.get("tasks", []):
            results.append({"id": name, **info})
    return results


def list_models(
    task_type: str | None = None,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Return models from the registry, optionally filtered by task and/or modality."""
    registry = load_registry()
    results = []
    for name, info in registry.get("models", {}).items():
        if task_type and task_type not in info.get("tasks", []):
            continue
        if modality and modality not in info.get("modalities", []):
            continue
        results.append({"id": name, **info})
    return results


def get_model_info(model_name: str) -> dict[str, Any] | None:
    """Return capability info for a single model, or None.

    Soft version of get_model_capabilities (returns None instead of raising).
    """
    try:
        return get_model_capabilities(model_name)
    except KeyError:
        return None


def get_backend_info(backend_name: str) -> dict[str, Any] | None:
    """Return info for a backend adapter, or None."""
    return load_registry().get("backends", {}).get(backend_name)


def reload() -> None:
    """Force-reload the YAML (useful for tests)."""
    global _CACHE
    _CACHE = None
