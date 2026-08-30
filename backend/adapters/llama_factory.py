"""LLaMA-Factory adapter stub — BIBLE §19 (World 1: LLM).

Wraps LLaMA-Factory for LLM/VLM fine-tuning (SFT, LoRA, QLoRA).
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


class LlamaFactoryAdapter(BackendAdapter):
    """BIBLE §19 — LLMs / VLMs via LLaMA-Factory."""

    def capabilities(self) -> dict[str, Any]:
        """
        Capabilities based on official LLaMA-Factory documentation:
        https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models
        """
        return {
            "supported_tasks": ["fine_tuning", "text_generation", "reward_modeling"],
            "supported_models": [
                "qwen2.5-7b", "qwen2.5-3b", "llama-3.1-8b", "gemma-2-9b",
                "mistral-7b", "phi-3", "baichuan2"
            ],
            "supported_training_methods": ["sft", "lora", "qlora", "ppo", "dpo", "orpo", "kto", "rm"],
            "supported_export_formats": ["safetensors", "gguf", "vllm", "ollama"],
            "modalities": ["text", "vision"], # LLaMA-Factory added VLM support recently (e.g. Llama-3-Vision, Qwen2-VL)
        }

    def estimate_resources(self, model_name: str, dataset_size: int, config: dict[str, Any]) -> ResourceEstimate:
        """
        Real arithmetic based on HuggingFace and LLaMA-Factory heuristics.
        Memory ≈ (Parameters × Bytes_per_param) + Optimizer_States + Activations + LoRA_weights
        """
        # 1. Extract params from model name heuristic (e.g., '7b' -> 7.0)
        params_b = 7.0
        match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
        if match:
            params_b = float(match.group(1))

        training_method = config.get("training_method", "lora")
        batch_size = config.get("per_device_train_batch_size", 1)
        seq_length = config.get("max_seq_length", 1024)
        
        # 2. Base weight memory
        if training_method == "qlora":
            bytes_per_param = 0.5  # 4-bit quantization
        elif training_method in ["lora", "sft"]:
            bytes_per_param = 2.0  # bf16/fp16
        else:
            bytes_per_param = 2.0
            
        base_vram_mb = (params_b * 1024) * bytes_per_param
        
        # 3. LoRA overhead and Optimizer states
        # LoRA usually targets ~1-2% of parameters. AdamW uses 8 bytes per trainable param.
        if training_method in ["lora", "qlora"]:
            lora_rank = config.get("lora_rank", 8)
            # Roughly (rank / 8) * 100MB for typical 7B models
            trainable_vram_mb = (params_b * 0.02 * 1024) * 8 * (lora_rank / 8)
        else: # Full SFT
            trainable_vram_mb = (params_b * 1024) * 8 # AdamW on all params
            
        # 4. Activation memory (heuristic: ~200MB per batch per 1024 seq length for 7B)
        # Scales linearly with params, batch, seq_len.
        activation_vram_mb = (params_b / 7.0) * batch_size * (seq_length / 1024) * 200
        
        total_vram_mb = int(base_vram_mb + trainable_vram_mb + activation_vram_mb)
        # Add 10% safety margin for PyTorch context
        total_vram_mb = int(total_vram_mb * 1.1)
        
        # Time estimation: Roughly (Dataset size * seq_len) / (GPU throughput)
        # Assuming typical 7B QLoRA on consumer GPU handles ~2000 tokens/second
        estimated_tokens = dataset_size * seq_length * config.get("epochs", 1)
        tokens_per_second = 2000 if training_method == "qlora" else 1500
        estimated_seconds = int(estimated_tokens / tokens_per_second)

        return ResourceEstimate(
            vram_required_mb=total_vram_mb,
            ram_required_mb=int(total_vram_mb * 1.5), # System RAM is usually 1.5x-2x VRAM needed for loading
            disk_required_mb=int(params_b * 1024 * 2) + 2000, # Base model + LoRA artifacts
            estimated_training_seconds=estimated_seconds,
            estimated_cost_usd=0.0
        )

    def prepare(self, dataset_path: Path, config: dict[str, Any]) -> Path:
        raise NotImplementedError("Phase 1: LLaMA-Factory prepare() not implemented. Anti-simulation enforced.")

    def train(self, dataset_path: Path, config: dict[str, Any]) -> TrainingResult:
        raise NotImplementedError("Phase 1: LLaMA-Factory train() not implemented. Anti-simulation enforced.")

    def evaluate(self, model_path: Path, dataset_path: Path, config: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError("Phase 1: LLaMA-Factory evaluate() not implemented. Anti-simulation enforced.")

    def export(self, model_path: Path, export_format: str, output_path: Path) -> Path:
        raise NotImplementedError("Phase 1: LLaMA-Factory export() not implemented. Anti-simulation enforced.")

    def deploy(self, model_path: Path, deploy_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Phase 1: LLaMA-Factory deploy() not implemented. Anti-simulation enforced.")
