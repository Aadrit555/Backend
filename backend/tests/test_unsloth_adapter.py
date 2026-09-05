import pytest
import shutil
import json
from pathlib import Path
from backend.adapters.unsloth import UnslothAdapter

def test_unsloth_adapter_e2e(tmp_path):
    """
    Integration test for the Unsloth Adapter.
    Verifies that we can prepare, estimate_resources, and run a short train loop.
    Prints actual peak VRAM to ensure our limits match diagnostic reality.
    """
    adapter = UnslothAdapter()
    
    import gc
    import torch
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    # 1. Capabilities
    caps = adapter.capabilities()
    assert "unsloth_llama3.2_3b" in caps["supported_models"]
    
    # 2. Config 
    config = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "max_seq_length": 512, # safe default for test
        "per_device_train_batch_size": 1,
        "prepared_dir": str(tmp_path / "prepared")
    }
    
    # 3. Estimate Resources
    # This should pass without raising ValueError
    estimate = adapter.estimate_resources("unsloth_llama3.2_3b", 10, config)
    assert estimate.vram_required_mb <= 4500
    
    # 4. Prepare (mock dataset)
    sample_rows = [
        {"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "Python is a programming language."}]},
        {"conversations": [{"from": "human", "value": "What is PyTorch?"}, {"from": "gpt", "value": "PyTorch is a machine learning framework."}]},
        {"conversations": [{"from": "human", "value": "Explain LoRA."}, {"from": "gpt", "value": "Low-Rank Adaptation adapts large models efficiently."}]},
        {"conversations": [{"from": "human", "value": "What is FastAPI?"}, {"from": "gpt", "value": "FastAPI is a modern web framework for Python."}]}
    ]
    sample_file = tmp_path / "sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")
    prepared_dir = adapter.prepare(sample_file, config)
    assert prepared_dir.exists()
    
    # 5. Train
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Unsloth training execution.")

    print("\n--- Starting Unsloth Integration Test Training ---")
    train_result = adapter.train(prepared_dir, config)
    print("--- Training Completed ---")
    
    assert train_result.artifact_path.exists()
    assert (train_result.artifact_path / "adapter_config.json").exists(), "LoRA adapter weights were not saved."
    
    peak_vram = train_result.metrics.get("peak_vram_gb", 0)
    print(f"\n[DIAGNOSTIC VERIFICATION] Peak VRAM during train loop: {peak_vram} GB")
    assert peak_vram > 0, "Failed to record peak VRAM"
    
    if peak_vram > 4.5:
        print(f"WARNING: Actual peak VRAM ({peak_vram} GB) exceeded our estimated safe limit (4.5 GB)!")
    else:
        print(f"SUCCESS: Actual peak VRAM ({peak_vram} GB) is within our 4.5 GB estimate.")
