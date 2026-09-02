"""Hugging Face Hub Integration — Search and Direct Import.

Provides functions to search models on Hugging Face Hub, retrieve metadata,
and import models directly into the Unified AI Platform registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from backend.config import settings

_IMPORTED_MODELS_PATH = settings.storage_root / "imported_models.json"


def _get_api() -> HfApi:
    return HfApi()


def search_hf_models(
    query: str,
    pipeline_tag: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search Hugging Face models by query string and optional task/pipeline tag.

    Returns a list of clean dictionaries sorted by downloads.
    """
    if not query.strip():
        return []

    api = _get_api()
    results = []

    try:
        # If pipeline_tag is specified and not empty, use filter
        filter_tag = pipeline_tag if pipeline_tag and pipeline_tag != "all" else None
        
        models = api.list_models(
            search=query.strip(),
            filter=filter_tag,
            limit=limit,
            sort="downloads",
            full=True,
        )

        for m in models:
            model_id = getattr(m, "id", None) or getattr(m, "modelId", None)
            if not model_id:
                continue

            author = model_id.split("/")[0] if "/" in model_id else "community"
            name = model_id.split("/")[-1]
            downloads = getattr(m, "downloads", 0) or 0
            likes = getattr(m, "likes", 0) or 0
            tag = getattr(m, "pipeline_tag", None) or "unknown"
            tags = getattr(m, "tags", []) or []
            private = getattr(m, "private", False) or False

            # Infer modality
            modality = "text"
            if tag in ["image-classification", "object-detection", "image-segmentation", "zero-shot-image-classification"] or any(t in tags for t in ["vision", "image"]):
                modality = "image"
            elif tag in ["tabular-classification", "tabular-regression"]:
                modality = "tabular"

            results.append({
                "id": model_id,
                "name": name,
                "author": author,
                "downloads": downloads,
                "likes": likes,
                "pipeline_tag": tag,
                "modality": modality,
                "tags": tags[:8],
                "private": private,
            })
    except Exception as e:
        print(f"[HF Hub] Search error for query '{query}': {e}")

    return results


def list_imported_models() -> list[dict[str, Any]]:
    """Return the list of all models imported from Hugging Face."""
    if not _IMPORTED_MODELS_PATH.exists():
        return []
    try:
        data = json.loads(_IMPORTED_MODELS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"[HF Hub] Failed to read imported models: {e}")
    return []


def import_hf_model(
    model_id: str,
    pipeline_type: str | None = None,
) -> dict[str, Any]:
    """Import a model from Hugging Face Hub into the platform registry.

    Validates model existence on HF, infers appropriate backend adapter
    (unsloth, ultralytics, autogluon), stores metadata, and registers it.
    """
    api = _get_api()
    model_id = model_id.strip()

    try:
        info = api.model_info(model_id)
    except Exception as e:
        raise ValueError(f"Could not find model '{model_id}' on Hugging Face: {str(e)}")

    tag = getattr(info, "pipeline_tag", None) or ""
    tags = getattr(info, "tags", []) or []
    id_lower = model_id.lower()

    # Determine pipeline type and backend adapter
    if not pipeline_type or pipeline_type == "auto":
        if tag in ["image-classification", "object-detection", "image-segmentation"] or any(k in id_lower for k in ["yolo", "vit", "resnet", "detr"]):
            pipeline_type = "vision"
            backend = "ultralytics"
            modality = "image"
            training_method = "full"
            task = tag or "object_detection"
        elif tag in ["tabular-classification", "tabular-regression"]:
            pipeline_type = "tabular"
            backend = "autogluon"
            modality = "tabular"
            training_method = "ensemble"
            task = tag or "classification"
        else:
            # Default to LLM / text
            pipeline_type = "llm"
            backend = "unsloth"
            modality = "text"
            training_method = "lora"
            task = "fine_tuning"
    else:
        if pipeline_type == "vision":
            backend = "ultralytics"
            modality = "image"
            training_method = "full"
            task = tag or "object_detection"
        elif pipeline_type == "tabular":
            backend = "autogluon"
            modality = "tabular"
            training_method = "ensemble"
            task = tag or "classification"
        elif pipeline_type == "rag":
            backend = "rag"
            modality = "text"
            training_method = "faiss_index"
            task = "rag"
        else:
            backend = "unsloth"
            modality = "text"
            training_method = "lora"
            task = "fine_tuning"

    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0

    record = {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "author": model_id.split("/")[0] if "/" in model_id else "community",
        "pipeline_type": pipeline_type,
        "pipeline_tag": tag or pipeline_type,
        "backend": backend,
        "modalities": [modality],
        "tasks": [task],
        "training_methods": [training_method],
        "vram_estimates": {
            training_method: {"min_mb": 1024, "recommended_mb": 2048}
        },
        "downloads": downloads,
        "likes": likes,
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save to imported_models.json
    _IMPORTED_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = list_imported_models()
    # Filter out duplicate if re-importing
    updated = [m for m in existing if m["id"] != model_id]
    updated.insert(0, record)

    _IMPORTED_MODELS_PATH.write_text(json.dumps(updated, indent=2), encoding="utf-8")

    # Invalidate registry cache so loader picks up newly imported model
    from backend.registry import loader
    loader.reload()

    return record
