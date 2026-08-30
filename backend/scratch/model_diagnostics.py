import time
import torch
import gc
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODELS_TO_TEST = [
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/gemma-2-2b-it-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
]

def get_vram_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved

def test_model(model_name):
    print(f"\n=================================")
    print(f"Testing model: {model_name}")
    print(f"=================================")
    
    # clean up before starting
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_vram_alloc, start_vram_res = get_vram_usage()
    print(f"Starting VRAM (Allocated/Reserved): {start_vram_alloc:.2f} GB / {start_vram_res:.2f} GB")
    
    try:
        t0 = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        t1 = time.time()
        load_time = t1 - t0
        print(f"Load time: {load_time:.2f}s")
        
        load_alloc, load_res = get_vram_usage()
        print(f"Post-load VRAM (Allocated/Reserved): {load_alloc:.2f} GB / {load_res:.2f} GB")
        
        # apply PEFT
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Create a dummy dataset mimicking 1024 token length sequence
        dummy_text = "This is a dummy prompt to test the sequence length. " * 200 # roughly 1000+ words
        dataset = Dataset.from_dict({"text": [dummy_text] * 4})
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            dataset_num_proc=1,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=2,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none"
            ),
        )
        
        trainer_stats = trainer.train()
        
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Peak Training VRAM Reserved: {peak_vram:.2f} GB")
        
        del model
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "load_time": load_time,
            "load_vram": load_res,
            "peak_vram": peak_vram
        }
    except Exception as e:
        print(f"Failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

results = {}
for m in MODELS_TO_TEST:
    results[m] = test_model(m)

print("\n\n=== FINAL RESULTS ===")
for m, res in results.items():
    print(f"\nModel: {m}")
    if res["status"] == "success":
        print(f"Load Time: {res['load_time']:.2f}s")
        print(f"Load VRAM (Reserved): {res['load_vram']:.2f} GB")
        print(f"Peak Training VRAM: {res['peak_vram']:.2f} GB")
    else:
        print(f"Status: FAILED")
        print(f"Error: {res['error']}")
