import os
import sys
import uvicorn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from backend.db import SessionLocal, init_db, Project, Experiment, TrainingRun, ModelArtifact

def get_latest_lora_model() -> Path:
    models_dir = Path("backend/storage/models")
    experiments = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for exp in experiments:
        lora_path = exp / "lora_model"
        if lora_path.exists() and (lora_path / "adapter_model.safetensors").exists():
            return lora_path
    raise FileNotFoundError("No valid lora_model artifact found in any experiment.")

def serve_mock_api():
    try:
        model_path = get_latest_lora_model()
    except Exception as e:
        print(f"FAILED TO FIND LORA ARTIFACT: {e}")
        sys.exit(1)
        
    init_db()
    db = SessionLocal()
    
    # We want a clean slate for testing so the UI hits this experiment specifically.
    exp_id = "exp_frontend_test_unsloth"
    proj_id = "proj_frontend_test"
    
    try:
        if not db.query(Project).filter_by(id=proj_id).first():
            proj = Project(id=proj_id, name="Frontend Deploy Test")
            db.add(proj)
            
        if not db.query(Experiment).filter_by(id=exp_id).first():
            exp = Experiment(id=exp_id, project_id=proj_id, dataset_id="dummy", model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", backend="unsloth")
            db.add(exp)
            
        run_id = f"run_{exp_id}"
        if not db.query(TrainingRun).filter_by(id=run_id).first():
            run = TrainingRun(id=run_id, experiment_id=exp_id, backend="unsloth", status="completed")
            db.add(run)
            
        art_id = f"art_{exp_id}"
        existing = db.query(ModelArtifact).filter_by(training_run_id=run_id).first()
        if existing:
            existing.path = str(model_path.absolute())
        else:
            art = ModelArtifact(id=art_id, training_run_id=run_id, path=str(model_path.absolute()))
            db.add(art)
            
        db.commit()
        print(f"Injected mock experiment {exp_id} for frontend testing.")
    finally:
        db.close()
        
    print("\nStarting FastAPI Server on 0.0.0.0:8000...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    serve_mock_api()
