"""BackendAdapter abstract base class — BIBLE §17-18, ARCHITECTURE.md §3.

Every training backend (LLaMA-Factory, Unsloth, AutoGluon, Ultralytics, …)
is accessed exclusively through a subclass of BackendAdapter.
The rest of the platform never imports a framework directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResourceEstimate:
    """Returned by estimate_resources().  All fields are best-effort."""

    vram_required_mb: int
    ram_required_mb: int
    disk_required_mb: int
    estimated_training_seconds: int | None = None
    estimated_cost_usd: float | None = None


@dataclass
class TrainingResult:
    """Returned by train()."""

    artifact_path: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    logs_path: Path | None = None


@dataclass
class EvaluationResult:
    """Returned by evaluate()."""

    metrics: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] | None = None


class BackendAdapter(ABC):
    """Abstract interface that every training backend must implement.

    See ARCHITECTURE.md §3 for the full contract.
    """

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return a structured description of what this backend supports.

        Must include at minimum:
          - supported_tasks:            list[str]
          - supported_models:           list[str]
          - supported_training_methods: list[str]
          - supported_export_formats:   list[str]
        """
        ...

    @abstractmethod
    def prepare(
        self,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> Path:
        """Convert / validate the dataset into the format this backend expects.

        Returns the path to the prepared dataset directory.
        """
        ...

    @abstractmethod
    def train(
        self,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> TrainingResult:
        """Launch training synchronously (called inside the background subprocess).

        Returns a TrainingResult with the artifact path and metrics.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        model_path: Path,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> EvaluationResult:
        """Run evaluation on a trained model."""
        ...

    @abstractmethod
    def export(
        self,
        model_path: Path,
        export_format: str,
        output_path: Path,
    ) -> Path:
        """Export / convert a trained model to the requested format."""
        ...

    @abstractmethod
    def estimate_resources(
        self,
        model_name: str,
        dataset_size: int,
        config: dict[str, Any],
    ) -> ResourceEstimate:
        """Estimate VRAM, RAM, disk, time, and cost BEFORE training starts.

        Called by the validation gate to check feasibility against the
        user's local GPU before the orchestrator's proposal is accepted.
        """
        ...

    @abstractmethod
    def deploy(
        self,
        model_path: Path,
        deploy_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Deploy a trained model locally.  Returns connection info."""
        ...
