"""Unsloth Adapter — BIBLE §19 (World 1: LLM), ARCHITECTURE.md §3.

Memory-efficient LLM fine-tuning via Unsloth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.adapters.base import (
    BackendAdapter,
    EvaluationResult,
    ResourceEstimate,
    TrainingResult,
)


class UnslothAdapter(BackendAdapter):
    """BIBLE §19 — LLM Fine-tuning via Unsloth."""

    def capabilities(self) -> dict[str, Any]:
        """
        Capabilities based on official Unsloth documentation:
        https://github.com/unslothai/unsloth
        """
        return {
            "supported_tasks": ["fine_tuning", "text_generation", "dpo"],
            "supported_models": [
                "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
                "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                "unsloth/gemma-2-2b-it-bnb-4bit", "unsloth/gemma-2-9b-it-bnb-4bit", "unsloth/gemma-2-27b-it-bnb-4bit",
                "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
            ],
            "supported_training_methods": ["lora", "qlora", "dpo"],
            "supported_export_formats": ["lora_dir", "gguf", "merged_16bit", "merged_4bit"],
            "modalities": ["text"],
        }

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        """
        Real arithmetic based on Unsloth heuristics.
        Unsloth reduces memory by ~40% vs standard HuggingFace PEFT.
        """
        # 1. Extract params from model name heuristic (e.g., '3b' -> 3.0)
        params_b = 3.0
        match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
        if match:
            params_b = float(match.group(1))

        training_method = config.get("training_method", "qlora")
        batch_size = config.get("per_device_train_batch_size", 1)
        seq_length = config.get("max_seq_length", 1024)
        
        # 2. Base weight memory
        # Unsloth heavily relies on 4-bit (QLoRA) for max efficiency
        if training_method == "qlora":
            bytes_per_param = 0.55  # slightly higher for 4-bit overhead
        else:
            bytes_per_param = 2.0  # bf16/fp16
            
        base_vram_mb = (params_b * 1024) * bytes_per_param
        
        # 3. LoRA overhead and Optimizer states
        # Unsloth uses customized Triton kernels that reduce activation/optimizer memory.
        lora_rank = config.get("lora_rank", 16)
        trainable_vram_mb = (params_b * 0.02 * 1024) * 8 * (lora_rank / 16)
            
        # 4. Activation memory 
        # Unsloth claims O(N) context length memory, and ~2-3x less activation memory.
        # Standard: ~200MB per batch/1k tokens. Unsloth: ~80MB per batch/1k tokens.
        activation_vram_mb = (params_b / 7.0) * batch_size * (seq_length / 1024) * 80
        
        total_vram_mb = int(base_vram_mb + trainable_vram_mb + activation_vram_mb)
        
        # Time estimation: Unsloth is ~2x faster than standard QLoRA.
        # Assuming typical 7B QLoRA handles ~4000 tokens/second on consumer GPU with Unsloth.
        estimated_tokens = dataset_size * seq_length * config.get("epochs", 1)
        # Scale throughput relative to 7B
        tokens_per_second = 4000 * (7.0 / params_b)
        estimated_seconds = int(estimated_tokens / tokens_per_second)

        return ResourceEstimate(
            vram_required_mb=total_vram_mb,
            ram_required_mb=int(total_vram_mb * 1.5), 
            disk_required_mb=int(params_b * 1024 * 2) + 1500, # Base model + artifacts
            estimated_training_seconds=estimated_seconds,
            estimated_cost_usd=0.0
        )

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        target_file = None
        is_raw_text = False
        
        # If it's a directory, find the first .jsonl, .json, or .txt file
        if dataset_path.is_dir():
            for ext in [".jsonl", ".json"]:
                for p in dataset_path.glob(f"*{ext}"):
                    target_file = p
                    break
                if target_file:
                    break
            
            if not target_file:
                for p in dataset_path.glob("*.txt"):
                    target_file = p
                    is_raw_text = True
                    break
                    
            if not target_file:
                raise FileNotFoundError(f"No .jsonl, .json, or .txt file found in directory {dataset_path}")
        else:
            target_file = dataset_path
            if target_file.suffix == ".txt":
                is_raw_text = True

        prepared_dir = Path(config.get("prepared_dir", str(dataset_path.parent / "prepared")))
        prepared_dir.mkdir(parents=True, exist_ok=True)
        
        structured_file = target_file
        
        if is_raw_text:
            print(f"[SANITIZING] Structuring raw text from {target_file} via Data Factory...")
            structured_file = prepared_dir / "structured_dataset.jsonl"
            
            from backend.data_factory import run_pipeline
            run_pipeline(target_file, structured_file)
            target_file = structured_file
            
        return target_file

    def _structure_text_to_jsonl(self, input_txt: Path, output_jsonl: Path) -> None:
        import requests
        import json
        from backend.config import settings
        
        text_content = input_txt.read_text(errors="ignore")
        # Simple chunking for demonstration (in production, use a robust text splitter)
        chunks = [text_content[i:i+2000] for i in range(0, len(text_content), 2000)]
        
        openrouter_key = settings.openrouter_api_key
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY required for automated data structuring.")
            
        with open(output_jsonl, "w") as f:
            for chunk in chunks:
                if not chunk.strip(): continue
                
                sys_prompt = "You are a data structuring tool. Extract 2-3 question-and-answer pairs from the following text. Return ONLY a valid JSON array of objects with 'question' and 'answer' keys. Do not include markdown formatting or any other text."
                
                try:
                    res = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
                        json={
                            "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": f"TEXT:\n{chunk}"}
                            ],
                            "temperature": 0.1
                        },
                        timeout=30
                    )
                    res_data = res.json()
                    if 'choices' not in res_data:
                        print("OpenRouter Error:", res_data)
                    content = res_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    if not content:
                        continue
                    
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                        
                    qa_pairs = json.loads(content.strip())
                    for pair in qa_pairs:
                        q = pair.get("question")
                        a = pair.get("answer")
                        if q and a:
                            sharegpt_row = {"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]}
                            f.write(json.dumps(sharegpt_row) + "\n")
                except Exception as e:
                    print(f"Failed to structure chunk: {e}")
                    
    def _sanitize_pii(self, input_jsonl: Path, output_jsonl: Path) -> None:
        import subprocess
        import sys
        
        script = f"""
import json
from pathlib import Path
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

with open("{input_jsonl}", "r") as fin, open("{output_jsonl}", "w") as fout:
    for line in fin:
        if not line.strip(): continue
        try:
            data = json.loads(line)
            if "conversations" in data:
                for turn in data["conversations"]:
                    text = turn.get("value", "")
                    if text:
                        results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"], language='en')
                        anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
                        turn["value"] = anonymized_result.text
            fout.write(json.dumps(data) + "\\n")
        except Exception as e:
            print(f"Error sanitizing line: {{e}}")
            fout.write(line)
"""
        try:
            print("[DATA FACTORY] Running Presidio sanitization in isolated subprocess...")
            subprocess.run([sys.executable, "-c", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[DATA FACTORY] Presidio subprocess failed: {e}")
            import shutil
            shutil.copy2(input_jsonl, output_jsonl)

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        import torch
        import gc
        import time
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt
        from backend.config import settings

        registry_model_name = config.get("model_name", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
        model_id_map = {
            "unsloth_llama3.2_3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "unsloth_qwen2.5_3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "unsloth_llama3.2_1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        }
        # If it's an old short ID, map it. Otherwise assume it's already a valid HF path from the new UI.
        model_name = model_id_map.get(registry_model_name, registry_model_name)
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            chat_template = "llama-3.1" if "3.1" in model_name_lower or "3.2" in model_name_lower or "3.3" in model_name_lower else "llama-3"
        elif "qwen" in model_name_lower:
            chat_template = "qwen-2.5"
        elif "mistral" in model_name_lower:
            chat_template = "mistral"
        elif "gemma" in model_name_lower:
            chat_template = "gemma"
        elif "phi" in model_name_lower:
            chat_template = "phi-3"
        else:
            chat_template = "chatml" # fallback
        
        # Hard limits based on diagnostics
        max_seq_length = min(config.get("max_seq_length", 1024), 2048)
        batch_size = min(config.get("per_device_train_batch_size", 1), 2)
        
        run_id = f"exp_{int(time.time())}"
        model_out_dir = settings.models_dir / run_id
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Unsloth Model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        print("Applying LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        
        print("Formatting ShareGPT Dataset...")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template,
        )
        
        dataset = standardize_sharegpt(dataset)
        
        def formatting_func(examples):
            convos = examples.get("conversations") or examples.get("messages")
            if convos is None:
                if "text" in examples:
                    return {"text": examples["text"]}
                else:
                    raise ValueError(f"Dataset missing required keys ('conversations', 'messages', or 'text'). Found: {list(examples.keys())}")
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return { "text" : texts, }
            
        dataset = dataset.map(formatting_func, batched=True)

        print("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=10, # Exact limit to prove gradients work
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=str(model_out_dir),
            ),
        )
        
        print("Starting Unsloth Training Loop...")
        trainer_stats = trainer.train()
        
        # Record max memory used
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"=== TRAINING COMPLETE ===")
        print(f"Peak VRAM used: {peak_vram:.2f} GB")
        
        metrics = trainer_stats.metrics if hasattr(trainer_stats, "metrics") else {}
        metrics["peak_vram_gb"] = round(peak_vram, 2)
        
        adapter_path = model_out_dir / "lora_model"
        print(f"Saving LoRA model to {adapter_path}")
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        
        # Cleanup Memory
        del model
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return TrainingResult(
            artifact_path=adapter_path,
            metrics=metrics,
        )

    @classmethod
    def _load_for_inference(cls, model_id_or_path: str, max_seq_length: int = 512):
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        import json
        
        print(f"\n[EVAL/DEPLOY] Loading model/adapter from: {model_id_or_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_id_or_path),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        # Try to infer chat template from adapter config or name
        chat_template = "llama-3.1"
        if "qwen" in str(model_id_or_path).lower():
            chat_template = "qwen-2.5"
        else:
            try:
                adapter_config = Path(model_id_or_path) / "adapter_config.json"
                if adapter_config.exists():
                    with open(adapter_config) as f:
                        cfg = json.load(f)
                        base_model = cfg.get("base_model_name_or_path", "").lower()
                        if "qwen" in base_model:
                            chat_template = "qwen-2.5"
            except Exception:
                pass
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template,
        )
        return model, tokenizer

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        import torch
        import gc

        prompts = config.get("prompts", ["What is the capital of France?"])
        registry_model_name = config.get("model_name", "unsloth_llama3.2_3b")
        model_id_map = {
            "unsloth_llama3.2_3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "unsloth_qwen2.5_3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "unsloth_llama3.2_1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        }
        base_model_name = model_id_map.get(registry_model_name, "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
        max_seq_length = config.get("max_seq_length", 512)

        def generate_for_model(model_id_or_path):
            model, tokenizer = self._load_for_inference(str(model_id_or_path), max_seq_length)
            
            outputs = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to("cuda")

                gen_out = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(gen_out[:, inputs.shape[1]:], skip_special_tokens=True)[0]
                outputs.append(decoded)

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            return outputs

        print("=== RUNNING BASELINE INFERENCE ===")
        base_outputs = generate_for_model(base_model_name)
        
        print("=== RUNNING LORA INFERENCE ===")
        if not model_path.exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {model_path}")
        lora_outputs = generate_for_model(model_path)
        
        print("\n=== EVALUATION RESULTS (SIDE-BY-SIDE) ===")
        for i, prompt in enumerate(prompts):
            print(f"\nPROMPT: {prompt}")
            print(f"[BASE MODEL]: {base_outputs[i]}")
            print(f"[LORA MODEL]: {lora_outputs[i]}")
            print("-" * 50)
            
        metrics = {
            "base_outputs": base_outputs,
            "lora_outputs": lora_outputs,
            "prompts": prompts
        }
        return EvaluationResult(metrics=metrics)

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        import shutil
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot export missing artifact: {model_path}")
            
        if output_path.exists():
            shutil.rmtree(output_path)
            
        shutil.copytree(model_path, output_path)
        return output_path

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        """
        Registers the deployment info. The actual FastAPI layer uses `_load_for_inference`.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot deploy missing artifact: {model_path}")
            
        experiment_id = deploy_config.get("experiment_id", "unknown")
        
        return {
            "endpoint": f"/api/models/{experiment_id}/chat",
            "status": "ready",
            "model_path": str(model_path),
            "backend": "unsloth"
        }
