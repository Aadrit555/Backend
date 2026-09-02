"""Hugging Face AutoTrain Advanced adapter stub — BIBLE §19 (World 3: Vision).

Wraps autotrain-advanced for Image Classification and Object Detection.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
import shutil

from backend.adapters.base import (
    BackendAdapter,
    EvaluationResult,
    ResourceEstimate,
    TrainingResult,
)


class AutoTrainAdapter(BackendAdapter):
    """BIBLE §19 — Computer vision via Hugging Face AutoTrain."""

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_tasks": ["image_classification", "object_detection"],
            "supported_models": ["autotrain_vision"],
            "supported_training_methods": ["full", "peft"],
            "supported_export_formats": ["pytorch"],
        }

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        """
        AutoTrain expects a specific folder structure for images (e.g. train/ class1/, train/ class2/).
        For this MVP, we assume the user's zip/folder is already in the right format,
        or we just return the raw dataset_path.
        """
        prepared_dir = Path(config.get("prepared_dir", str(dataset_path.parent / "prepared")))
        prepared_dir.mkdir(parents=True, exist_ok=True)
        
        # In a real scenario, you'd extract zip or map csv -> folder structure here.
        # MVP: just symlink or copy to prepared_dir
        if dataset_path.is_file() and dataset_path.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(prepared_dir)
        else:
            # If it's already a directory, copy it over or just use it
            if dataset_path.is_dir() and dataset_path != prepared_dir:
                shutil.copytree(dataset_path, prepared_dir, dirs_exist_ok=True)
            else:
                return dataset_path
                
        return prepared_dir

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        """
        Executes `autotrain image-classification --data-path <dataset_path> ...`
        """
        import os
        project_name = config.get("project_name", "autotrain_vision_project")
        task = config.get("task", "image-classification") # or object-detection
        
        # Default fallback model for AutoTrain vision
        model = config.get("base_model", "google/vit-base-patch16-224")
        if model == "autotrain_vision":
            model = "google/vit-base-patch16-224"
            
        output_dir = dataset_path.parent / project_name
        
        # Construct CLI command
        # Note: autotrain requires python environment where it's installed.
        cmd = [
            sys.executable, "-m", "autotrain.cli.autotrain",
            task,
            "--data-path", str(dataset_path),
            "--project-name", str(output_dir),
            "--model", model,
            "--epochs", str(config.get("epochs", 3)),
            "--batch-size", str(config.get("batch_size", 4)),
        ]
        
        try:
            print(f"[AutoTrain] Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"AutoTrain failed: {e}")
            raise e
            
        return TrainingResult(
            artifact_path=output_dir,
            metrics={"status": "completed_by_autotrain"}, 
        )

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        # AutoTrain typically evaluates internally. For MVP, just return success.
        return EvaluationResult(
            metrics={"status": "evaluated_by_autotrain"}
        )

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        return model_path

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        return ResourceEstimate(
            vram_required_mb=8000,
            ram_required_mb=16000,
            disk_required_mb=5000,
            estimated_training_seconds=600
        )

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
