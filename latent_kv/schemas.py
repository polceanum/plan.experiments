"""Serializable records used across collection, compression, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskExample:
    benchmark: str
    task_id: str
    prompt: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetadata:
    model_id: str
    tokenizer_id: str
    dtype: str
    device: str
    layers: int
    selected_layers: list[int]
    selected_heads: list[int] | None
    token_count: int
    cache_path: str
    benchmark: str | None = None
    task_id: str | None = None
    prompt_baseline: str | None = None
    prompt_protocol: str | None = None
    target: str | None = None
    parsed_answer: str | None = None
    correct: bool | None = None
    generation_error: str | None = None


@dataclass
class TrajectoryRecord:
    run_id: str
    benchmark: str
    task_id: str
    model_id: str
    seed: int
    attempt_id: int
    prompt: str
    target: str
    output_text: str
    parsed_answer: str | None
    correct: bool
    retry_index: int
    cache_path: str | None = None
    hidden_path: str | None = None
    latency_s: float | None = None
    generated_tokens: int | None = None
    prompt_tokens: int | None = None
    memory_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    run_id: str
    method: str
    latent_dim: int
    records: int
    reconstruction_mse: float | None
    latent_path: str | None
    artifact_path: str | None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BaselineMetric:
    baseline: str
    benchmark: str
    examples: int
    accuracy: float
    ci95: float
    retry_efficiency: float
    error_recovery_rate: float
    mean_reasoning_length: float
    mean_latency_s: float | None
    memory_bytes: float | None
    reconstruction_mse: float | None = None
    logit_similarity: float | None = None


def to_json_line(record: Any) -> str:
    return json.dumps(asdict(record), sort_keys=True) + "\n"


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(to_json_line(record))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

