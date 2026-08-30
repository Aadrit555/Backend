"""Orchestrator tool definitions + dispatch — BIBLE §41, ARCHITECTURE.md §4.

Contains:
  1. TOOL_DEFINITIONS — the canonical JSON-Schema list sent to Groq.
  2. TOOL_DISPATCH    — maps tool name → Python implementation function.

All schemas are Groq-safe: flat parameters, no nested objects, no arrays.
Comma-separated strings are used where arrays would be natural; they are
parsed server-side.
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# 1.  Canonical tool definitions  (ARCHITECTURE.md §4)
# ---------------------------------------------------------------------------

STAGE_TOOLS: dict[str, list[str]] = {
    "DISCOVERY": ["inspect_files", "inspect_dataset", "analyze_schema", "find_relationships"],
    "FORMULATION": ["formulate_problem"], # Formulate problem is added dynamically in loop, but we track it here
    "DATASET": ["create_dataset", "clean_dataset", "improve_dataset"],
    "CAPABILITY": ["list_models", "get_model_capabilities", "estimate_vram", "estimate_cost"],
    "EXECUTION": ["create_experiment", "start_training", "get_training_status"],
    "EVALUATION": ["evaluate_model", "analyze_errors", "compare_models"],
    "DEPLOYMENT": ["deploy_model"],
}

TRANSITION_TOOL = {
    "type": "function",
    "function": {
        "name": "transition_stage",
        "description": "Move to a different pipeline stage when you have completed all tasks in the current stage, or need tools from another stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "next_stage": {
                    "type": "string",
                    "enum": ["DISCOVERY", "FORMULATION", "DATASET", "CAPABILITY", "EXECUTION", "EVALUATION", "DEPLOYMENT"]
                },
                "reason": {"type": "string", "description": "Why you are transitioning to this stage."}
            },
            "required": ["next_stage", "reason"],
        },
    },
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_files",
            "description": "List and identify the types of all files in a data source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datasource_id": {"type": "string", "description": "ID of the uploaded data source."},
                },
                "required": ["datasource_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "Return schema, stats, and sample rows for a dataset version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_version_id": {"type": "string", "description": "ID of the dataset version."},
                },
                "required": ["dataset_version_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_schema",
            "description": "Detect column types, missing values, cardinality, and potential targets in a tabular source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datasource_id": {"type": "string", "description": "ID of the data source."},
                },
                "required": ["datasource_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_relationships",
            "description": "Discover join keys or semantic relationships across data sources in a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID."},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_dataset",
            "description": "Construct a training dataset from one or more data sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID."},
                    "datasource_ids": {"type": "string", "description": "Comma-separated data source IDs."},
                    "task_type": {"type": "string", "description": "ML task: classification, regression, fine_tuning, object_detection, etc."},
                    "target_column": {"type": "string", "description": "Target/label column name (if applicable)."},
                    "instructions": {"type": "string", "description": "Free-text instructions for dataset construction."},
                },
                "required": ["project_id", "datasource_ids", "task_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clean_dataset",
            "description": "Run automated cleaning: dedup, handle missing values, fix encodings, detect outliers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_version_id": {"type": "string", "description": "Dataset version ID to clean."},
                    "operations": {"type": "string", "description": "Comma-separated ops: dedup, drop_missing, fix_encoding, remove_outliers, rebalance."},
                },
                "required": ["dataset_version_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List available base models, optionally filtered by task type or modality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "Filter: classification, regression, fine_tuning, object_detection, etc."},
                    "modality": {"type": "string", "description": "Filter: text, image, tabular, audio."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_capabilities",
            "description": "Return detailed capability info for a specific model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier, e.g. 'qwen2.5-7b'."},
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_vram",
            "description": "Check whether a model + training config fits in local GPU VRAM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."},
                    "training_method": {"type": "string", "description": "sft, lora, qlora, full, autogluon, etc."},
                    "dataset_size": {"type": "integer", "description": "Number of samples."},
                    "batch_size": {"type": "integer", "description": "Proposed batch size."},
                },
                "required": ["model_name", "training_method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_cost",
            "description": "Estimate wall-clock time for a training run (local only in MVP).",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."},
                    "training_method": {"type": "string", "description": "Training method."},
                    "dataset_size": {"type": "integer", "description": "Number of samples."},
                    "epochs": {"type": "integer", "description": "Number of epochs."},
                },
                "required": ["model_name", "training_method", "dataset_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_experiment",
            "description": "Register a new experiment: (model, dataset_version, config, backend).",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID."},
                    "dataset_version_id": {"type": "string", "description": "Dataset version ID."},
                    "model_name": {"type": "string", "description": "Base model identifier."},
                    "backend": {"type": "string", "description": "Backend adapter: llama_factory, unsloth, autogluon, ultralytics."},
                    "training_method": {"type": "string", "description": "sft, lora, qlora, full, autogluon, etc."},
                    "config_json": {"type": "object", "description": "JSON object of additional hyperparameters."},
                },
                "required": ["project_id", "dataset_version_id", "model_name", "backend", "training_method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_training",
            "description": "Launch a training run as a local background subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment ID."},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_status",
            "description": "Poll the status of a training run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment ID."},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Run evaluation on a trained model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment ID."},
                    "dataset_version_id": {"type": "string", "description": "Dataset version to evaluate against."},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_errors",
            "description": "Break down model failures by class/category to find dominant error sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluation_id": {"type": "string", "description": "Evaluation ID."},
                },
                "required": ["evaluation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_models",
            "description": "Compare metrics across multiple experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_ids": {"type": "string", "description": "Comma-separated experiment IDs."},
                },
                "required": ["experiment_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "improve_dataset",
            "description": "Apply a targeted improvement to a dataset version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_version_id": {"type": "string", "description": "Dataset version to improve."},
                    "strategy": {"type": "string", "description": "Strategy: augment, rebalance, denoise, synthetic, relabel."},
                    "instructions": {"type": "string", "description": "Free-text guidance."},
                },
                "required": ["dataset_version_id", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deploy_model",
            "description": "Deploy a trained model locally for inference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment whose artifact to deploy."},
                    "deploy_type": {"type": "string", "description": "Deployment type: chat, rest_api, export_file."},
                },
                "required": ["experiment_id", "deploy_type"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2.  Tool dispatch  (name → implementation)
# ---------------------------------------------------------------------------

def _not_implemented(**kwargs: Any) -> dict:
    """Placeholder for tools not yet implemented."""
    return {"error": "not_implemented", "detail": "This tool is not yet implemented."}


# --- Real implementations for Phase 1 tools ---

def _list_models(task_type: str = "", modality: str = "", **_: Any) -> dict:
    """Real implementation: queries the capability registry."""
    from backend.registry.loader import list_models
    results = list_models(
        task_type=task_type or None,
        modality=modality or None,
    )
    return {"models": results}


def _get_model_capabilities(model_name: str, **_: Any) -> dict:
    """Real implementation: queries the capability registry."""
    from backend.registry.loader import get_model_capabilities
    try:
        return get_model_capabilities(model_name)
    except KeyError as exc:
        return {"error": "not_found", "detail": str(exc)}


def _estimate_vram(
    model_name: str,
    training_method: str,
    dataset_size: int = 0,
    batch_size: int = 0,
    **_: Any,
) -> dict:
    """Real implementation: checks registry VRAM estimates against local GPU."""
    from backend.registry.loader import get_model_capabilities
    from backend.gpu_probe import get_max_free_vram_mb

    try:
        info = get_model_capabilities(model_name)
    except KeyError as exc:
        return {"error": "not_found", "detail": str(exc)}

    vram_estimates = info.get("vram_estimates", {})
    method_est = vram_estimates.get(training_method, {})
    required_mb = method_est.get("min_mb", 0)
    recommended_mb = method_est.get("recommended_mb", 0)
    available_mb = get_max_free_vram_mb()

    fits = required_mb <= available_mb if available_mb > 0 else required_mb == 0

    return {
        "model_name": model_name,
        "training_method": training_method,
        "required_mb": required_mb,
        "available_mb": available_mb,
        "fits": fits,
    }


def _inspect_files(datasource_id: str, **_: Any) -> dict:
    from backend.db import SessionLocal, DataSource
    from backend.understanding.engine import analyze_project
    db = SessionLocal()
    ds = db.query(DataSource).filter_by(id=datasource_id).first()
    db.close()
    if ds:
        return analyze_project(ds.project_id)
    return {"error": "not_found", "detail": f"DataSource {datasource_id} not found."}


def _inspect_dataset(dataset_version_id: str, **_: Any) -> dict:
    from backend.db import SessionLocal, DatasetVersion
    from backend.understanding.engine import analyze_project
    db = SessionLocal()
    dv = db.query(DatasetVersion).filter_by(id=dataset_version_id).first()
    if dv:
        project_id = dv.dataset.project_id
        db.close()
        return analyze_project(project_id)
    db.close()
    return {"error": "not_found", "detail": f"DatasetVersion {dataset_version_id} not found."}


def _analyze_schema(datasource_id: str, **_: Any) -> dict:
    return _inspect_files(datasource_id)


def _find_relationships(project_id: str, **_: Any) -> dict:
    from backend.understanding.engine import detect_relationships
    return detect_relationships(project_id)


def _create_experiment(project_id: str, model_name: str, backend: str, training_method: str, dataset_version_id: str, **kwargs: Any) -> dict:
    """Create an experiment and training run row in the database."""
    from backend.db import SessionLocal, Experiment, TrainingRun
    import json
    
    db = SessionLocal()
    try:
        config_kw = kwargs.get("config_json", {})
        if isinstance(config_kw, str):
            config_kw = json.loads(config_kw)
        if not isinstance(config_kw, dict):
            config_kw = {}
        config_json = json.dumps({"training_method": training_method, **config_kw})
        # In a real system, dataset_version_id would link to DatasetVersion. 
        # For our MVP, since we don't have datasets fully implemented, we'll just store it 
        # as the dataset_id in the Experiment table for now.
        exp = Experiment(
            project_id=project_id,
            dataset_id=dataset_version_id,
            model_name=model_name,
            backend=backend,
            config_json=config_json
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        
        run = TrainingRun(
            experiment_id=exp.id,
            backend=backend,
            config_json=config_json
        )
        db.add(run)
        db.commit()
        
        return {
            "status": "created",
            "experiment_id": exp.id,
            "project_id": project_id,
            "model_name": model_name,
            "training_method": training_method,
        }
    except Exception as e:
        db.rollback()
        return {"error": "db_error", "detail": str(e)}
    finally:
        db.close()


def _start_training(experiment_id: str, **kwargs: Any) -> dict:
    """Launch the training subprocess."""
    import subprocess
    import sys
    
    try:
        # Launch run_training.py as a non-blocking background process
        # Using sys.executable ensures we use the same venv
        subprocess.Popen(
            [sys.executable, "-m", "backend.run_training", experiment_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return {
            "status": "started",
            "experiment_id": experiment_id,
            "message": "Training launched in the background."
        }
    except Exception as e:
        return {"error": "launch_failed", "detail": str(e)}


def _evaluate_model(experiment_id: str, **kwargs: Any) -> dict:
    from backend.evaluation.engine import evaluate_model
    from backend.db import SessionLocal
    db = SessionLocal()
    try:
        return evaluate_model(db, experiment_id)
    finally:
        db.close()


def _analyze_errors(evaluation_id: str, **kwargs: Any) -> dict:
    from backend.evaluation.engine import analyze_errors
    from backend.db import SessionLocal
    db = SessionLocal()
    try:
        return analyze_errors(db, evaluation_id)
    finally:
        db.close()


# Populated during Phase 1.  Each value is a callable(**kwargs) → dict.
TOOL_DISPATCH: dict[str, Callable[..., dict]] = {
    defn["function"]["name"]: _not_implemented
    for defn in TOOL_DEFINITIONS
}

# Wire Phase 1 real implementations
TOOL_DISPATCH["list_models"] = _list_models
TOOL_DISPATCH["get_model_capabilities"] = _get_model_capabilities
TOOL_DISPATCH["estimate_vram"] = _estimate_vram

# Wire Phase 2 real implementations
TOOL_DISPATCH["inspect_files"] = _inspect_files
TOOL_DISPATCH["inspect_dataset"] = _inspect_dataset
TOOL_DISPATCH["analyze_schema"] = _analyze_schema
TOOL_DISPATCH["find_relationships"] = _find_relationships

def _create_dataset(**kwargs: Any) -> dict:
    from backend.dataset.builder import create_dataset as real_create_dataset
    from backend.db import SessionLocal
    db = SessionLocal()
    try:
        return real_create_dataset(db=db, **kwargs)
    finally:
        db.close()

TOOL_DISPATCH["create_dataset"] = _create_dataset

# Wire Phase 3 & 4 real implementations
TOOL_DISPATCH["create_experiment"] = _create_experiment
TOOL_DISPATCH["start_training"] = _start_training
TOOL_DISPATCH["evaluate_model"] = _evaluate_model
TOOL_DISPATCH["analyze_errors"] = _analyze_errors


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up and call a tool by name.

    Called by the validation gate AFTER schema/capability/VRAM checks pass.
    """
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": "unknown_tool", "detail": f"No tool named '{name}'."}
    return fn(**arguments)
