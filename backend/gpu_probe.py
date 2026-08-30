"""GPU Probe — ARCHITECTURE.md §5 (Validation Gate, VRAM gate).

Detects local NVIDIA GPU(s) and available VRAM via nvidia-smi.
Called at startup and by the validation gate before any training
proposal is accepted.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """One physical GPU."""

    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


def probe_gpus() -> list[GPUInfo]:
    """Detect local NVIDIA GPUs and their VRAM.

    Returns an empty list if nvidia-smi is not available or no GPU is found.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(parts[2]),
                    free_mb=int(parts[3]),
                    used_mb=int(parts[4]),
                )
            )
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_max_free_vram_mb() -> int:
    """Return the largest free VRAM across all detected GPUs, or 0."""
    gpus = probe_gpus()
    if not gpus:
        return 0
    return max(g.free_mb for g in gpus)


if __name__ == "__main__":
    gpus = probe_gpus()
    if not gpus:
        print("No NVIDIA GPU detected.")
    else:
        for g in gpus:
            print(f"GPU {g.index}: {g.name} — {g.free_mb}/{g.total_mb} MB free")
