import os
import sys
from pathlib import Path
from backend.adapters.unsloth import UnslothAdapter

def verify_training():
    scratch_dir = Path("backend/scratch")
    dataset_path = scratch_dir / "sharegpt_graph_dataset.jsonl"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Please run verify_chat_graph.py first.")
        sys.exit(1)
        
    print(f"Found dataset at {dataset_path}.")
    print("Initiating Unsloth Adapter Training...")
    
    adapter = UnslothAdapter()
    
    config = {
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "max_seq_length": 512, # smaller context for verification
        "per_device_train_batch_size": 1,
        "epochs": 1
    }
    
    # This will output directly to stdout so we can capture the training logs
    result = adapter.train(dataset_path, config)
    
    print("\n=== VERIFICATION ARTIFACT CHECKS ===")
    artifact_path = result.artifact_path
    
    if artifact_path.exists():
        print(f"SUCCESS: Artifact directory created at {artifact_path}")
        
        # equivalent of ls -la
        total_size = 0
        for p in artifact_path.glob("*"):
            size = p.stat().st_size
            total_size += size
            print(f" - {p.name} ({size / 1024 / 1024:.2f} MB)")
            
        print(f"Total Adapter Size: {total_size / 1024 / 1024:.2f} MB")
        if total_size > 1: # A real LoRA adapter is at least several MBs
            print("SUCCESS: File sizes indicate a physical LoRA model was written.")
        else:
            print("ERROR: Adapter size is impossibly small.")
    else:
        print("ERROR: Artifact directory was not created.")

if __name__ == "__main__":
    verify_training()
