import os
import sys
from pprint import pprint
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.adapters.llama_factory import LlamaFactoryAdapter
from backend.adapters.unsloth import UnslothAdapter

def verify():
    print("=== LLaMA-Factory Adapter ===")
    lf = LlamaFactoryAdapter()
    print("\nCapabilities:")
    pprint(lf.capabilities())
    
    print("\nResource Estimate (qwen2.5-7b, QLoRA, Batch 4, Seq 2048):")
    res_lf = lf.estimate_resources("qwen2.5-7b", 10000, {
        "training_method": "qlora",
        "per_device_train_batch_size": 4,
        "max_seq_length": 2048,
        "lora_rank": 16,
        "epochs": 1
    })
    print(res_lf)
    
    print("\nTesting Anti-Simulation train()...")
    try:
        lf.train(Path("/dummy"), {})
        print("FAIL: Did not raise NotImplementedError!")
    except NotImplementedError as e:
        print(f"SUCCESS: Raised expected error -> {e}")

    print("\n=== Unsloth Adapter ===")
    us = UnslothAdapter()
    print("\nCapabilities:")
    pprint(us.capabilities())
    
    print("\nResource Estimate (unsloth_llama3.2_3b, QLoRA, Batch 4, Seq 2048):")
    res_us = us.estimate_resources("unsloth_llama3.2_3b", 10000, {
        "training_method": "qlora",
        "per_device_train_batch_size": 4,
        "max_seq_length": 2048,
        "lora_rank": 16,
        "epochs": 1
    })
    print(res_us)
    
    print("\nTesting Anti-Simulation train()...")
    try:
        us.train(Path("/dummy"), {})
        print("FAIL: Did not raise NotImplementedError!")
    except NotImplementedError as e:
        print(f"SUCCESS: Raised expected error -> {e}")
        
    print("\nImports verified successfully.")

if __name__ == "__main__":
    verify()
