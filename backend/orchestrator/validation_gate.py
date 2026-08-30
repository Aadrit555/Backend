"""Validation gate — BIBLE §40/§58, ARCHITECTURE.md §5.

Every tool call from the GPT OSS 120B orchestrator passes through this
gate BEFORE any side-effect executes (Llama 3.3 70B deprecated).  The gate 
is deterministic Python code — never an LLM.

Pipeline:
  1. Schema validate  — does the JSON match the tool's JSON Schema?
  2. Entity resolve   — do referenced IDs exist in SQLite?
  3. Capability check — is the (model, backend, task) combo supported?
  4. VRAM gate        — does estimated VRAM fit detected local GPU?
  5. Execute          — call the tool implementation.

On failure at any stage: return a structured error dict to the LLM so it
can retry.  Never silently proceed.  Max retries per tool call: 3.
"""

from __future__ import annotations

import json
import jsonschema  # type: ignore[import-untyped]
from typing import Any

from backend.gpu_probe import get_max_free_vram_mb
from backend.orchestrator.tools import TOOL_DEFINITIONS, dispatch_tool
from backend.registry.loader import get_registry

MAX_RETRIES = 3

# Build a lookup: tool_name → JSON Schema for its parameters
_SCHEMAS: dict[str, dict] = {
    defn["function"]["name"]: defn["function"]["parameters"]
    for defn in TOOL_DEFINITIONS
}

# Tools that involve training/deployment and therefore need VRAM checks
_VRAM_GATED_TOOLS = {"start_training", "deploy_model", "create_experiment"}

# Tools whose arguments reference entity IDs we should resolve
_ENTITY_ID_FIELDS = {
    "project_id", "datasource_id", "dataset_version_id",
    "experiment_id", "evaluation_id", "model_artifact_id",
}


def _schema_validate(name: str, arguments: dict[str, Any]) -> dict | None:
    """Return an error dict if arguments don't match the tool's schema, else None."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        return {"error": "unknown_tool", "detail": f"No tool named '{name}'."}
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return {"error": "schema_error", "detail": str(exc.message)}
    return None


def _entity_resolve(arguments: dict[str, Any]) -> dict | None:
    """Return an error dict if any referenced entity ID doesn't exist in DB, else None."""
    # TODO (Phase 1): query SQLite for each ID field present in arguments.
    # For now, pass through — entities aren't created yet.
    return None


def _capability_check(name: str, arguments: dict[str, Any]) -> dict | None:
    """Return an error dict if the requested combo is unsupported, else None."""
    if name not in ("create_experiment", "estimate_vram", "start_training"):
        return None

    registry = get_registry()
    model_name = arguments.get("model_name", "")
    backend = arguments.get("backend", "")
    training_method = arguments.get("training_method", "")

    if model_name and model_name not in registry.get("models", {}):
        available = list(registry.get("models", {}).keys())
        return {
            "error": "capability_mismatch",
            "detail": f"Model '{model_name}' is not in the capability registry.",
            "supported_alternatives": available,
        }

    if model_name and backend:
        model_info = registry.get("models", {}).get(model_name, {})
        if backend not in model_info.get("backends", []):
            return {
                "error": "capability_mismatch",
                "detail": f"Backend '{backend}' does not support model '{model_name}'.",
                "supported_alternatives": model_info.get("backends", []),
            }

    if model_name and training_method:
        model_info = registry.get("models", {}).get(model_name, {})
        if training_method not in model_info.get("training_methods", []):
            return {
                "error": "capability_mismatch",
                "detail": f"Training method '{training_method}' not supported for '{model_name}'.",
                "supported_alternatives": model_info.get("training_methods", []),
            }

    return None


def _vram_check(name: str, arguments: dict[str, Any]) -> dict | None:
    """Return an error dict if estimated VRAM exceeds local GPU, else None."""
    if name not in _VRAM_GATED_TOOLS:
        return None

    registry = get_registry()
    model_name = arguments.get("model_name", "")
    training_method = arguments.get("training_method", "")

    model_info = registry.get("models", {}).get(model_name, {})
    vram_estimates = model_info.get("vram_estimates", {})
    method_estimate = vram_estimates.get(training_method, {})
    required_mb = method_estimate.get("min_mb", 0)

    if required_mb == 0:
        return None  # no estimate available — allow through

    available_mb = get_max_free_vram_mb()
    if available_mb == 0:
        return None  # no GPU detected — will fail at training, not here

    if required_mb > available_mb:
        suggestions = []
        for method, est in vram_estimates.items():
            if est.get("min_mb", 0) <= available_mb:
                suggestions.append(method)
        return {
            "error": "vram_insufficient",
            "required_mb": required_mb,
            "available_mb": available_mb,
            "suggestions": suggestions,
        }

    return None


def validate_and_execute(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Full validation gate → execute pipeline.

    Returns a result dict (success or structured error).
    Called by the orchestrator loop after parsing a tool call from Groq.
    """
    # 1. Schema
    err = _schema_validate(name, arguments)
    if err:
        return err

    # 2. Entity resolution
    err = _entity_resolve(arguments)
    if err:
        return err

    # 3. Capability check
    err = _capability_check(name, arguments)
    if err:
        return err

    # 4. VRAM gate
    err = _vram_check(name, arguments)
    if err:
        return err

    # 5. Execute
    return dispatch_tool(name, arguments)
