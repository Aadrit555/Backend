"""AutoGluon adapter — BIBLE §19 (World 2: Tabular).

Wraps AutoGluon for traditional ML + multimodal AutoML.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.adapters.base import (
    BackendAdapter,
    EvaluationResult,
    ResourceEstimate,
    TrainingResult,
)
from backend.config import settings
from backend.registry.loader import get_model_capabilities


class AutoGluonAdapter(BackendAdapter):
    """BIBLE §19 — Traditional + multimodal AutoML via AutoGluon."""

    def capabilities(self) -> dict[str, Any]:
        """Returns the matching registry entry from Part A."""
        info = get_model_capabilities("autogluon_best")
        return {
            "supported_tasks": info.get("tasks", []),
            "supported_models": ["autogluon_best"] + info.get("sub_models", []),
            "supported_training_methods": info.get("training_methods", []),
            "supported_export_formats": ["predictor_dir"],
        }

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(dataset_path)
        print(f"DEBUG prepare: dataset_path={dataset_path}, df_shape={df.shape}")
        
        # basic train/val split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        out_dir_str = config.get("prepared_dir")
        if out_dir_str:
            out_dir = Path(out_dir_str)
        else:
            dataset_id = f"ds_{int(time.time())}"
            out_dir = settings.processed_dir / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        # Save config so train() knows what the target is
        (out_dir / "config.json").write_text(json.dumps(config))
        
        return out_dir

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        from autogluon.tabular import TabularPredictor

        target = config.get("target_column")
        if not target:
            cfg_file = dataset_path / "config.json"
            if cfg_file.exists():
                ds_config = json.loads(cfg_file.read_text())
                target = ds_config.get("target_column")
        if not target:
            raise ValueError("target_column is required for AutoGluon training")

        train_path = dataset_path / "train.csv"
        
        run_id = f"exp_{int(time.time())}"
        model_out_dir = settings.models_dir / run_id
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        def update_status(stage: str, pct: int, msg: str):
            status_path = settings.experiments_dir / run_id / "status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stage": stage,
                "pct": pct,
                "message": msg,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            status_path.write_text(json.dumps(payload))

        update_status("initializing", 10, "Initializing TabularPredictor")
        
        predictor = TabularPredictor(label=target, path=str(model_out_dir))
        
        update_status("training", 50, "Fitting AutoGluon ensemble")
        
        time_limit = config.get("time_limit", 60)
        predictor.fit(
            train_data=str(train_path),
            time_limit=time_limit,
            presets="medium_quality"
        )
        
        update_status("completed", 100, "Training complete")
        
        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best if hasattr(predictor, "model_best") else leaderboard.iloc[0]["model"]
        metrics = {"best_model": best_model}
        if not leaderboard.empty and "score_val" in leaderboard.columns:
            metrics["best_val_score"] = float(leaderboard.iloc[0]["score_val"])
            
        return TrainingResult(
            artifact_path=model_out_dir,
            metrics=metrics,
        )

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        from autogluon.tabular import TabularPredictor
        
        predictor = TabularPredictor.load(str(model_path))
        
        val_path = dataset_path / "val.csv"
        if not val_path.exists():
            val_path = dataset_path
            
        metrics = predictor.evaluate(str(val_path))
        
        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best if hasattr(predictor, "model_best") else (leaderboard.iloc[0]["model"] if not leaderboard.empty else "")
        
        leaderboard_data = []
        for _, row in leaderboard.iterrows():
            m_name = row["model"]
            leaderboard_data.append({
                "model_name": m_name,
                "score": float(row.get("score_val", row.get("score_test", 0.0))),
                "fit_time": float(row.get("fit_time", 0.0)),
                "pred_time": float(row.get("pred_time_val", row.get("pred_time_test", 0.0))),
                "is_best": bool(m_name == best_model)
            })
            
        metrics["leaderboard"] = leaderboard_data
        
        return EvaluationResult(
            metrics=metrics,
            error_analysis=None
        )

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        if export_format not in ["predictor_dir"]:
            raise NotImplementedError(f"Format {export_format} not supported by AutoGluon adapter")
            
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(model_path, output_path)
        
        return output_path

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        # AutoGluon is CPU-friendly. Give a conservative estimate based on dataset_size.
        # Assuming each row is roughly 1KB.
        ram_mb = max(512, (dataset_size * 1024) // (1024 * 1024) * 10)
        disk_mb = max(100, ram_mb * 2)
        
        return ResourceEstimate(
            vram_required_mb=0,
            ram_required_mb=ram_mb,
            disk_required_mb=disk_mb,
            estimated_training_seconds=config.get("time_limit", 60),
            estimated_cost_usd=0.0
        )

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        """
        Deploy simply returns a callable prediction function wrapped in a dict.
        For MVP local-first architecture, returning an in-memory callable is the simplest 
        way to support immediate inference without complex ASGI lifecycle management.
        """
        from autogluon.tabular import TabularPredictor
        
        predictor = TabularPredictor.load(str(model_path))
        
        def predict_fn(data: dict | list) -> Any:
            import pandas as pd
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            return predictor.predict(data).tolist()
            
        return {"predict_fn": predict_fn, "status": "deployed", "type": "callable"}
