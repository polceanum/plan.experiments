"""Experiment configuration and model-profile resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

from .cache import select_layers
from .schemas import write_json


DTYPE_BYTES = {
    "float32": 4,
    "torch.float32": 4,
    "bfloat16": 2,
    "torch.bfloat16": 2,
    "float16": 2,
    "torch.float16": 2,
}


@dataclass(frozen=True)
class ModelProfile:
    profile_name: str
    model_id: str
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    max_position_embeddings: int | None
    grouped_query_attention: bool
    kv_values_per_token_all_layers: int
    bytes_per_token_all_layers: int


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    name: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    prompt: dict[str, Any]
    cache: dict[str, Any]
    codec: dict[str, Any]
    training: dict[str, Any]
    scale: dict[str, Any]
    model_profile: ModelProfile
    selected_layers: list[int]
    selected_kv_values_per_token: int
    selected_bytes_per_token: int
    estimated_cache_bytes: int | None = None
    source_config: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_profile"] = asdict(self.model_profile)
        return payload


def _read_hf_config(model_id: str) -> dict[str, Any]:
    model_path = Path(model_id).expanduser()
    config_path = model_path / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    return config.to_dict()


def _dtype_bytes(dtype: str) -> int:
    return DTYPE_BYTES.get(str(dtype), 4)


def resolve_model_profile(model_id: str, profile_name: str | None = None) -> ModelProfile:
    config = _read_hf_config(model_id)
    layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    attention_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or attention_heads)
    head_dim = int(config.get("head_dim") or hidden_size // attention_heads)
    dtype = str(config.get("torch_dtype") or "float32")
    kv_values = 2 * layers * kv_heads * head_dim
    return ModelProfile(
        profile_name=profile_name or str(config.get("model_type") or Path(model_id).name),
        model_id=model_id,
        layers=layers,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        max_position_embeddings=config.get("max_position_embeddings"),
        grouped_query_attention=kv_heads != attention_heads,
        kv_values_per_token_all_layers=kv_values,
        bytes_per_token_all_layers=kv_values * _dtype_bytes(dtype),
    )


def load_experiment_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def resolve_experiment_config(path: Path) -> ResolvedExperimentConfig:
    raw = load_experiment_config(path)
    model = dict(raw.get("model") or {})
    dataset = dict(raw.get("dataset") or {})
    prompt = dict(raw.get("prompt") or {})
    cache = dict(raw.get("cache") or {})
    codec = dict(raw.get("codec") or {})
    training = dict(raw.get("training") or {})
    scale = dict(raw.get("scale") or {})
    model_id = str(model.get("model_id") or "")
    if not model_id:
        raise ValueError("Experiment config requires model.model_id")
    profile = resolve_model_profile(model_id, profile_name=model.get("profile"))

    layer_mode = str(cache.get("layer_mode") or "all")
    dummy_cache = tuple((None, None) for _ in range(profile.layers))
    _, selected_layers = select_layers(dummy_cache, layer_mode)
    values_per_token = 2 * len(selected_layers) * profile.kv_heads * profile.head_dim
    storage_dtype = str(cache.get("storage_dtype") or profile.dtype)
    bytes_per_token = values_per_token * _dtype_bytes(storage_dtype)
    token_count = scale.get("estimated_tokens_per_cache")
    example_count = scale.get("examples") or dataset.get("limit")
    estimated_cache_bytes = None
    if token_count is not None and example_count is not None:
        estimated_cache_bytes = int(token_count) * int(example_count) * bytes_per_token

    return ResolvedExperimentConfig(
        name=str(raw.get("name") or path.stem),
        model=model,
        dataset=dataset,
        prompt=prompt,
        cache=cache,
        codec=codec,
        training=training,
        scale=scale,
        model_profile=profile,
        selected_layers=selected_layers,
        selected_kv_values_per_token=values_per_token,
        selected_bytes_per_token=bytes_per_token,
        estimated_cache_bytes=estimated_cache_bytes,
        source_config=str(path),
    )


def write_resolved_config(run_dir: Path, resolved: ResolvedExperimentConfig) -> Path:
    path = run_dir / "resolved_config.json"
    write_json(path, resolved.to_dict())
    return path
