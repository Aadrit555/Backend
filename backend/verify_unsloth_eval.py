import os
import sys
from pathlib import Path
from backend.adapters.unsloth import UnslothAdapter

def get_latest_lora_model() -> Path:
    models_dir = Path("backend/storage/models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory does not exist.")
        
    # Get all subdirectories starting with exp_
    experiments = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    if not experiments:
        raise FileNotFoundError("No experiments found.")
        
    # Sort by modification time to get latest
    experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for exp in experiments:
        lora_path = exp / "lora_model"
        if lora_path.exists() and (lora_path / "adapter_model.safetensors").exists():
            return lora_path
            
    raise FileNotFoundError("No valid lora_model artifact found in any experiment.")

def verify_eval():
    try:
        model_path = get_latest_lora_model()
    except Exception as e:
        print(f"FAILED TO FIND LORA ARTIFACT: {e}")
        sys.exit(1)
        
    print(f"Found latest LoRA adapter at: {model_path}")
    print("Initiating Unsloth Adapter Evaluation...\n")
    
    adapter = UnslothAdapter()
    
    config = {
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "prompts": [
            "What is the capital of France?",
            "Write a short 2-line poem about coding."
        ],
        "max_seq_length": 512,
    }
    
    # We pass a dummy dataset_path because it's not used in this specific test
    dataset_path = Path("dummy.jsonl")
    
    try:
        adapter.evaluate(model_path, dataset_path, config)
    except Exception as e:
        print(f"EVALUATION FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_eval()
