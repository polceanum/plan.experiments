"""Interfaces for current and future latent planning modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PlanningModule(Protocol):
    """A module that maps task context to latent planning state."""

    def encode_task(self, prompt: str) -> torch.Tensor:
        ...

    def generate_latent(self, task_embedding: torch.Tensor) -> torch.Tensor:
        ...

    def decode_to_cache_vector(self, latent: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class FineTuningPlanConfig:
    """Configuration placeholder for later planning-aware LLM adaptation."""

    base_model_id: str
    planner_checkpoint: str
    train_base_lm: bool = False
    train_planner: bool = True
    adapter_type: str = "latent_tokens"
    loss_weights: dict[str, float] | None = None


class FrozenLLMInjectionAdapter:
    """Marker adapter for the current phase: planner output is injected into a frozen LLM."""

    mode = "frozen_llm_injection"


class JointPlanningAdapter:
    """Marker adapter for the later phase: LLM learns to cooperate with generated states."""

    mode = "joint_planning_finetune"

