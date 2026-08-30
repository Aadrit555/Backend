import os
import sys
import time
import requests
import subprocess
import shutil
from pathlib import Path
import json

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

def create_mock_experiment(db, exp_id: str, model_path_str: str):
    proj_id = "test_project_deploy"
    if not db.query(Project).filter_by(id=proj_id).first():
        proj = Project(id=proj_id, name="Deploy Test")
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
        existing.path = model_path_str
    else:
        art = ModelArtifact(id=art_id, training_run_id=run_id, path=model_path_str)
        db.add(art)
    db.commit()

def run_verification():
    original_model_path = get_latest_lora_model()
    
    # Create a duplicate physical copy to force a path change in the API cache
    copy_model_path = original_model_path.parent / "lora_model_copy"
    if copy_model_path.exists():
        shutil.rmtree(copy_model_path)
    shutil.copytree(original_model_path, copy_model_path)
    
    init_db()
    db = SessionLocal()
    try:
        # Experiment 1
        exp_id_1 = "exp_deploy_test_1"
        create_mock_experiment(db, exp_id_1, str(original_model_path.absolute()))
        
        # Experiment 2 (pointing to duplicate physical copy)
        exp_id_2 = "exp_deploy_test_2"
        create_mock_experiment(db, exp_id_2, str(copy_model_path.absolute()))
    finally:
        db.close()
        
    print("\nStarting FastAPI Server...")
    server = subprocess.Popen(
        ["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(5)
    
    try:
        poem_prompt = "Write a short 2-line poem about coding."
        
        # TEST 1
        print(f"\n=== TEST 1: Original Experiment ({exp_id_1}) ===")
        payload = {"prompt": poem_prompt, "max_tokens": 64}
        response = requests.post(f"http://127.0.0.1:8000/api/models/{exp_id_1}/chat", json=payload, timeout=60)
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
            
        # TEST 2: Cache Swap
        print(f"\n=== TEST 2: Second Experiment ({exp_id_2}) Cache Swap ===")
        response2 = requests.post(f"http://127.0.0.1:8000/api/models/{exp_id_2}/chat", json=payload, timeout=60)
        try:
            print(json.dumps(response2.json(), indent=2))
        except Exception:
            print(response2.text)
            
        # TEST 3: Swap Back
        print(f"\n=== TEST 3: Original Experiment ({exp_id_1}) Cache Swap Back ===")
        response3 = requests.post(f"http://127.0.0.1:8000/api/models/{exp_id_1}/chat", json=payload, timeout=60)
        try:
            print(json.dumps(response3.json(), indent=2))
        except Exception:
            print(response3.text)

    finally:
        print("\nShutting down FastAPI Server...")
        server.terminate()
        server.wait()

if __name__ == "__main__":
    run_verification()
