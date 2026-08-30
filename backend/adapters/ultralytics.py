"""Ultralytics adapter stub — BIBLE §19 (World 3: Vision).

Wraps Ultralytics YOLO for object detection / image classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.adapters.base import (
    BackendAdapter,
    EvaluationResult,
    ResourceEstimate,
    TrainingResult,
)


class UltralyticsAdapter(BackendAdapter):
    """BIBLE §19 — Computer vision via Ultralytics (YOLOv8+)."""

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_tasks": ["object_detection", "classification", "segmentation"],
            "supported_models": ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
            "supported_training_methods": ["full"],
            "supported_export_formats": ["pt", "onnx", "torchscript"],
        }

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        raise NotImplementedError  # Phase 1

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        raise NotImplementedError  # Phase 1

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError  # Phase 1

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        raise NotImplementedError  # Phase 1

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        raise NotImplementedError  # Phase 1

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError  # Phase 1
