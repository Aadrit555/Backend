"""Evaluation Engine — BIBLE §24-25, ARCHITECTURE.md §1.

Runs evaluation suites and error analysis on trained models.

Metrics selected automatically by task type (BIBLE §24):
  - Classification: accuracy, precision, recall, F1, confusion matrix
  - Regression: MAE, MSE, RMSE, R²
  - Vision: mAP, precision, recall, IoU
  - LLM: task accuracy, instruction following, hallucination, latency
  - Time series: MAE, RMSE, forecasting accuracy

Error analysis (BIBLE §25): breaks down failures by class/category
to identify dominant error sources.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


def evaluate_model(
    db: Session,
    experiment_id: str,
    dataset_version_id: str | None = None,
) -> dict[str, Any]:
    """Run the evaluation suite on a trained model."""
    from backend.db import Experiment, TrainingRun, Evaluation
    from backend.config import settings
    import json
    
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        return {"error": "experiment_not_found"}
        
    run = db.query(TrainingRun).filter(TrainingRun.experiment_id == experiment_id).first()
    if not run or run.status != "completed":
        return {"error": "training_not_completed"}
        
    config = json.loads(exp.config_json)
    
    if exp.backend == "autogluon":
        from backend.adapters.autogluon import AutoGluonAdapter
        adapter = AutoGluonAdapter()
    elif exp.backend == "unsloth":
        from backend.adapters.unsloth import UnslothAdapter
        adapter = UnslothAdapter()
    elif exp.backend == "rag":
        from backend.adapters.rag import RagAdapter
        adapter = RagAdapter()
    else:
        return {"error": "unsupported_backend"}
    
    run_dir = settings.experiments_dir / experiment_id
    prepared_dir = run_dir / "prepared"
    
    # We need the artifact path from model_artifacts or just assume it's in run_dir/export
    # Actually, train returns an artifact path but we exported it to run_dir/export
    artifact_path = str(run_dir / "export")
    
    eval_result = adapter.evaluate(artifact_path, prepared_dir, config)
    metrics = eval_result.metrics
    
    # Compute confusion_matrix and other metrics explicitly if it's classification
    try:
        from backend.adapters.autogluon import _get_tabular_predictor_cls
        TabularPredictor = _get_tabular_predictor_cls()
        import pandas as pd
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
        
        predictor = TabularPredictor.load(artifact_path)
        val_path = prepared_dir / "val.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            target_column = config.get("target_column")
            if target_column and target_column in val_df.columns:
                y_true = val_df[target_column]
                y_pred = predictor.predict(val_df)
                
                # Check if classification
                if predictor.problem_type in ["binary", "multiclass"]:
                    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                    
                    if predictor.problem_type == "binary":
                        # For binary, need to know pos_label, or just use average='binary'
                        metrics["precision"] = float(precision_score(y_true, y_pred, average="macro"))
                        metrics["recall"] = float(recall_score(y_true, y_pred, average="macro"))
                        metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
                    else:
                        metrics["precision"] = float(precision_score(y_true, y_pred, average="macro"))
                        metrics["recall"] = float(recall_score(y_true, y_pred, average="macro"))
                        metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
                    
                    cm = confusion_matrix(y_true, y_pred)
                    metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:
        print(f"Error computing extended metrics: {e}")
        pass
    
    evaluation = Evaluation(
        experiment_id=experiment_id,
        metrics_json=json.dumps(metrics)
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    
    return {
        "status": "success",
        "evaluation_id": evaluation.id,
        "metrics": metrics
    }


def analyze_errors(db: Session, evaluation_id: str) -> dict[str, Any]:
    """Break down model failures by class/category/segment."""
    from backend.db import Experiment, Evaluation
    from backend.config import settings
    import json
    import pandas as pd
    import numpy as np
    
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if not evaluation:
        return {"error": "evaluation_not_found"}
        
    exp = db.query(Experiment).filter(Experiment.id == evaluation.experiment_id).first()
    if not exp:
        return {"error": "experiment_not_found"}
        
    metrics = json.loads(evaluation.metrics_json)
    config = json.loads(exp.config_json)
    target_column = config.get("target_column")
    
    if exp.backend != "autogluon":
        return {"error": "unsupported_backend"}
        
    from backend.adapters.autogluon import _get_tabular_predictor_cls
    TabularPredictor = _get_tabular_predictor_cls()
    run_dir = settings.experiments_dir / exp.id
    artifact_path = run_dir / "export"
    
    try:
        predictor = TabularPredictor.load(str(artifact_path))
        # Load test data
        prepared_dir = run_dir / "prepared"
        test_data_path = prepared_dir / "test.csv"
        
        if not test_data_path.exists():
            return {"error": "test_data_not_found"}
            
        test_df = pd.read_csv(test_data_path)
        if target_column not in test_df.columns:
            return {"error": "target_column_not_in_test_data"}
            
        y_true = test_df[target_column]
        y_pred = predictor.predict(test_df)
        
        test_df["_correct"] = (y_true == y_pred)
        
        # Find categorical columns (other than target)
        cat_cols = [c for c in test_df.columns if c != target_column and c != "_correct" and test_df[c].dtype == "object"]
        
        worst_segment = None
        worst_accuracy = 1.0
        analysis_report = {}
        
        if cat_cols:
            # Pick the one with fewest unique values just for simplicity
            segment_col = min(cat_cols, key=lambda c: test_df[c].nunique())
            
            segment_acc = test_df.groupby(segment_col)["_correct"].mean().to_dict()
            analysis_report["segment_column"] = segment_col
            analysis_report["segment_accuracy"] = segment_acc
            
            worst_val = min(segment_acc, key=segment_acc.get)
            worst_segment = f"{segment_col}={worst_val}"
            worst_accuracy = segment_acc[worst_val]
        else:
            # Fallback to per-class accuracy
            class_acc = test_df.groupby(target_column)["_correct"].mean().to_dict()
            # Convert keys to str for JSON
            class_acc_str = {str(k): v for k, v in class_acc.items()}
            analysis_report["class_accuracy"] = class_acc_str
            
            if class_acc:
                worst_val = min(class_acc, key=class_acc.get)
                worst_segment = f"class={worst_val}"
                worst_accuracy = class_acc[worst_val]
                
        analysis_report["worst_performing_segment"] = {
            "segment": worst_segment,
            "accuracy": float(worst_accuracy),
            "recommendation": f"Collect more training examples for {worst_segment}"
        }
        
        evaluation.error_analysis_json = json.dumps(analysis_report)
        db.commit()
        
        return analysis_report
        
    except Exception as e:
        return {"error": "analysis_failed", "detail": str(e)}
