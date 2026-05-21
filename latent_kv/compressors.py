"""Compression baselines for flattened KV cache vectors."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import torch
from torch import nn

from .cache import cache_shapes, cache_to_device, choose_device, flatten_cache, load_cache_bundle, load_model_and_tokenizer
from .injection import _forward_parameters
from .schemas import CompressionResult, read_jsonl, write_json


@dataclass(frozen=True)
class CacheMatrix:
    matrix: torch.Tensor
    mask: torch.Tensor
    paths: list[str]
    lengths: list[int]
    shapes: list[Any]
    aligned_shapes: list[Any]
    labels: list[dict[str, Any]]


def _record_label(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "benchmark": record.get("benchmark"),
        "task_id": record.get("task_id"),
        "attempt_id": record.get("attempt_id"),
        "correct": bool(record.get("correct")),
        "target": record.get("target"),
        "parsed_answer": record.get("parsed_answer"),
        "output_text": record.get("output_text"),
        "prompt_baseline": metadata.get("prompt_baseline"),
        "prompt_protocol": metadata.get("prompt_protocol"),
        "source": metadata.get("source"),
        "generation_error": metadata.get("generation_error"),
        "cache_path": record.get("cache_path"),
    }


def _aligned_cache_shapes(shapes: list[Any]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    first = shapes[0]
    aligned = []
    for layer_idx in range(len(first)):
        aligned_layer = []
        for kv_idx in range(2):
            base = list(first[layer_idx][kv_idx])
            max_tokens = max(int(shape[layer_idx][kv_idx][-2]) for shape in shapes)
            base[-2] = max_tokens
            aligned_layer.append(tuple(base))
        aligned.append((aligned_layer[0], aligned_layer[1]))
    return aligned


def _flatten_to_shapes(cache: Any, target_shapes: list[Any]) -> torch.Tensor:
    parts = []
    for layer_idx, (key, value) in enumerate(cache):
        for tensor, target_shape in [(key, target_shapes[layer_idx][0]), (value, target_shapes[layer_idx][1])]:
            target = torch.zeros(target_shape, dtype=torch.float32)
            slices = tuple(slice(0, size) for size in tensor.shape)
            target[slices] = tensor.detach().cpu().float()
            parts.append(target.reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _shape_mask(shapes: list[Any], target_shapes: list[Any]) -> torch.Tensor:
    parts = []
    for layer_idx, (key_shape, value_shape) in enumerate(shapes):
        for source_shape, target_shape in [(key_shape, target_shapes[layer_idx][0]), (value_shape, target_shapes[layer_idx][1])]:
            target = torch.zeros(target_shape, dtype=torch.bool)
            slices = tuple(slice(0, size) for size in source_shape)
            target[slices] = True
            parts.append(target.reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.bool)


def _aligned_to_compact(vector: torch.Tensor, shapes: list[Any], aligned_shapes: list[Any]) -> torch.Tensor:
    vector = vector.detach().cpu().float().reshape(-1)
    offset = 0
    parts = []
    for layer_idx, (key_shape, value_shape) in enumerate(shapes):
        for source_shape, aligned_shape in [(key_shape, aligned_shapes[layer_idx][0]), (value_shape, aligned_shapes[layer_idx][1])]:
            size = int(torch.tensor(aligned_shape).prod().item())
            tensor = vector[offset : offset + size].reshape(aligned_shape)
            offset += size
            slices = tuple(slice(0, dim) for dim in source_shape)
            parts.append(tensor[slices].reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _compact_reconstructions(reconstructed: torch.Tensor, shapes: list[Any], aligned_shapes: list[Any]) -> torch.Tensor:
    compact = [_aligned_to_compact(row, shape, aligned_shapes) for row, shape in zip(reconstructed, shapes)]
    max_len = max(int(row.numel()) for row in compact)
    padded = [torch.nn.functional.pad(row, (0, max_len - row.numel())) for row in compact]
    return torch.stack(padded).float()


def _temporal_token_count(aligned_shapes: list[Any]) -> int:
    return int(aligned_shapes[0][0][-2])


def _shape_without_token(shape: tuple[int, ...]) -> tuple[int, ...]:
    token_axis = len(shape) - 2
    return tuple(shape[:token_axis] + shape[token_axis + 1 :])


def _temporal_feature_dim(aligned_shapes: list[Any]) -> int:
    token_count = _temporal_token_count(aligned_shapes)
    return sum(int(torch.tensor(shape).prod().item()) // token_count for layer in aligned_shapes for shape in layer)


def _aligned_vector_to_temporal(vector: torch.Tensor, aligned_shapes: list[Any]) -> torch.Tensor:
    vector = vector.detach().cpu().float().reshape(-1)
    offset = 0
    token_parts = []
    for key_shape, value_shape in aligned_shapes:
        for aligned_shape in (key_shape, value_shape):
            size = int(torch.tensor(aligned_shape).prod().item())
            tensor = vector[offset : offset + size].reshape(aligned_shape)
            offset += size
            token_axis = len(aligned_shape) - 2
            token_first = tensor.movedim(token_axis, 0).reshape(int(aligned_shape[token_axis]), -1)
            token_parts.append(token_first)
    if offset != vector.numel():
        raise ValueError(f"Temporal conversion consumed {offset} values from vector with {vector.numel()} values")
    return torch.cat(token_parts, dim=-1) if token_parts else torch.empty(0, 0)


def _temporal_to_aligned_vector(sequence: torch.Tensor, aligned_shapes: list[Any]) -> torch.Tensor:
    sequence = sequence.detach().cpu().float()
    offset = 0
    parts = []
    token_count = int(sequence.shape[0])
    for key_shape, value_shape in aligned_shapes:
        for aligned_shape in (key_shape, value_shape):
            token_axis = len(aligned_shape) - 2
            feature_shape = _shape_without_token(tuple(aligned_shape))
            feature_size = int(torch.tensor(feature_shape).prod().item())
            token_first = sequence[:, offset : offset + feature_size].reshape((token_count, *feature_shape))
            offset += feature_size
            tensor = token_first.movedim(0, token_axis).reshape(aligned_shape)
            parts.append(tensor.reshape(-1))
    if offset != sequence.shape[-1]:
        raise ValueError(f"Temporal conversion consumed {offset} features from sequence with {sequence.shape[-1]} features")
    return torch.cat(parts) if parts else torch.empty(0)


def _temporal_to_aligned_vector_grad(sequence: torch.Tensor, aligned_shapes: list[Any]) -> torch.Tensor:
    offset = 0
    parts = []
    token_count = int(sequence.shape[0])
    for key_shape, value_shape in aligned_shapes:
        for aligned_shape in (key_shape, value_shape):
            token_axis = len(aligned_shape) - 2
            feature_shape = _shape_without_token(tuple(aligned_shape))
            feature_size = int(torch.tensor(feature_shape).prod().item())
            token_first = sequence[:, offset : offset + feature_size].reshape((token_count, *feature_shape))
            offset += feature_size
            parts.append(token_first.movedim(0, token_axis).reshape(aligned_shape).reshape(-1))
    if offset != sequence.shape[-1]:
        raise ValueError(f"Temporal conversion consumed {offset} features from sequence with {sequence.shape[-1]} features")
    return torch.cat(parts) if parts else sequence.new_empty(0)


def _aligned_to_compact_grad(vector: torch.Tensor, shapes: list[Any], aligned_shapes: list[Any]) -> torch.Tensor:
    vector = vector.reshape(-1)
    offset = 0
    parts = []
    for layer_idx, (key_shape, value_shape) in enumerate(shapes):
        for source_shape, aligned_shape in [(key_shape, aligned_shapes[layer_idx][0]), (value_shape, aligned_shapes[layer_idx][1])]:
            size = int(torch.tensor(aligned_shape).prod().item())
            tensor = vector[offset : offset + size].reshape(aligned_shape)
            offset += size
            slices = tuple(slice(0, dim) for dim in source_shape)
            parts.append(tensor[slices].reshape(-1))
    return torch.cat(parts) if parts else vector.new_empty(0)


def _unflatten_cache_grad(vector: torch.Tensor, shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    vector = vector.reshape(-1)
    offset = 0
    layers = []
    for key_shape, value_shape in shapes:
        key_size = int(torch.tensor(key_shape).prod().item())
        value_size = int(torch.tensor(value_shape).prod().item())
        key = vector[offset : offset + key_size].reshape(key_shape)
        offset += key_size
        value = vector[offset : offset + value_size].reshape(value_shape)
        offset += value_size
        layers.append((key, value))
    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} values but shapes consumed {offset}")
    return tuple(layers)


def _generation_token_ids(bundle: dict[str, Any], steps: int) -> torch.Tensor:
    generation_token_ids = bundle.get("generation_token_ids")
    if generation_token_ids is None or int(generation_token_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.long)
    return generation_token_ids.reshape(-1)[: max(0, int(steps))].long()


def _generation_tokens_for_replay(
    bundle: dict[str, Any],
    steps: int,
    tokenizer: Any | None = None,
    output_text: str | None = None,
) -> torch.Tensor:
    if int(steps) <= 0:
        return torch.empty(0, dtype=torch.long)
    token_ids = _generation_token_ids(bundle, steps)
    if int(token_ids.numel()) > 0:
        return token_ids
    if tokenizer is None or not output_text:
        return token_ids
    encoded = tokenizer.encode(str(output_text), add_special_tokens=False)
    return torch.tensor(encoded[: max(0, int(steps))], dtype=torch.long)


def _model_cache_dtype(model: Any) -> torch.dtype | None:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return None


def _cache_token_count(cache: Any) -> int:
    if not cache:
        return 0
    return int(cache[0][0].shape[-2])


def _slice_cache_tokens(cache: Any, token_count: int) -> Any:
    token_count = max(0, int(token_count))
    return tuple((key[..., :token_count, :], value[..., :token_count, :]) for key, value in cache)


def _teacher_forced_initial_cache_tokens(bundle: dict[str, Any], cache: Any) -> int:
    input_ids = bundle.get("input_ids")
    if input_ids is None:
        raise ValueError("Bundle must contain input_ids for teacher-forced replay loss")
    input_len = int(input_ids.shape[-1])
    cache_len = _cache_token_count(cache)
    generation_config = bundle.get("generation_config") or {}
    if generation_config.get("cache_mode") == "trajectory" and generation_config.get("prompt_tokens") is not None:
        return min(int(generation_config["prompt_tokens"]), cache_len)
    return min(input_len, cache_len)


def _teacher_forced_generation_logits(
    bundle: dict[str, Any],
    cache: Any,
    token_ids: torch.Tensor,
    model: Any,
    device: torch.device,
) -> list[torch.Tensor]:
    input_ids = bundle.get("input_ids")
    attention_mask = bundle.get("attention_mask")
    if input_ids is None:
        raise ValueError("Bundle must contain input_ids for teacher-forced replay loss")
    if token_ids.numel() == 0:
        return []
    input_ids = input_ids.to(device)
    cache_dtype = _model_cache_dtype(model)
    initial_cache_tokens = _teacher_forced_initial_cache_tokens(bundle, cache)
    if initial_cache_tokens < 1:
        raise ValueError("Teacher-forced replay loss requires at least one prefix token")
    prefix_tokens = input_ids[..., :initial_cache_tokens]
    current_mask = (
        attention_mask[..., :initial_cache_tokens].to(device)
        if attention_mask is not None
        else torch.ones((1, initial_cache_tokens), dtype=torch.long, device=device)
    )
    past = cache_to_device(_slice_cache_tokens(cache, initial_cache_tokens - 1), device, dtype=cache_dtype)
    forward_parameters = _forward_parameters(model)
    logits_by_step: list[torch.Tensor] = []

    def forward_one(token: torch.Tensor, position: int) -> Any:
        model_kwargs = {
            "input_ids": token,
            "attention_mask": current_mask,
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
        }
        replay_position = torch.tensor([[position]], dtype=torch.long, device=device)
        if "position_ids" in forward_parameters:
            model_kwargs["position_ids"] = replay_position
        if "cache_position" in forward_parameters:
            model_kwargs["cache_position"] = replay_position.reshape(-1)
        return model(**model_kwargs)

    outputs = forward_one(prefix_tokens[..., -1:], initial_cache_tokens - 1)
    past = outputs.past_key_values
    logits_by_step.append(outputs.logits[:, -1, :])

    for step_idx, token_id in enumerate(token_ids.reshape(-1).tolist()):
        if len(logits_by_step) >= int(token_ids.numel()):
            break
        current_mask = torch.cat(
            [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=device)],
            dim=-1,
        )
        next_token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        outputs = forward_one(next_token, initial_cache_tokens + step_idx)
        past = outputs.past_key_values
        logits_by_step.append(outputs.logits[:, -1, :])
    return logits_by_step


def _temporal_matrix(x: torch.Tensor, aligned_shapes: list[Any]) -> torch.Tensor:
    return torch.stack([_aligned_vector_to_temporal(row, aligned_shapes) for row in x]).float()


def _temporal_token_mask(shapes: list[Any], aligned_shapes: list[Any]) -> torch.Tensor:
    max_tokens = _temporal_token_count(aligned_shapes)
    rows = []
    for shape in shapes:
        token_count = int(shape[0][0][-2])
        row = torch.zeros(max_tokens, dtype=torch.bool)
        row[:token_count] = True
        rows.append(row)
    return torch.stack(rows)


def _temporal_reconstructions_to_aligned(reconstructed: torch.Tensor, aligned_shapes: list[Any]) -> torch.Tensor:
    return torch.stack([_temporal_to_aligned_vector(row, aligned_shapes) for row in reconstructed]).float()


def load_cache_matrix(run_dir: Path) -> CacheMatrix:
    records = read_jsonl(run_dir / "records.jsonl")
    caches = []
    paths = []
    shapes = []
    labels = []
    for record in records:
        cache_path = record.get("cache_path")
        if not cache_path:
            continue
        bundle = load_cache_bundle(Path(cache_path))
        cache = bundle["cache"]
        caches.append(cache)
        paths.append(cache_path)
        shapes.append(bundle.get("shapes") or cache_shapes(cache))
        labels.append(_record_label(record))
    if not caches:
        raise ValueError(f"No cache vectors found under {run_dir}")
    aligned_shapes = _aligned_cache_shapes(shapes)
    vectors = [_flatten_to_shapes(cache, aligned_shapes) for cache in caches]
    masks = [_shape_mask(shape, aligned_shapes) for shape in shapes]
    lengths = [int(flatten_cache(cache).numel()) for cache in caches]
    return CacheMatrix(
        matrix=torch.stack(vectors).float(),
        mask=torch.stack(masks),
        paths=paths,
        lengths=lengths,
        shapes=shapes,
        aligned_shapes=aligned_shapes,
        labels=labels,
    )


def reconstruction_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(torch.mean((original.float() - reconstructed.float()) ** 2).item())


def _append_training_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _current_memory_gb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 3))
    except Exception:
        return -1.0


def _empty_device_cache(device: torch.device | str) -> None:
    device_type = torch.device(device).type
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class RandomProjectionCompressor:
    input_dim: int
    latent_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        scale = 1.0 / max(self.latent_dim, 1) ** 0.5
        self.projection = torch.randn(self.input_dim, self.latent_dim, generator=gen) * scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.projection

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.projection.T


@dataclass
class SVDCompressor:
    mean: torch.Tensor
    components: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, latent_dim: int) -> "SVDCompressor":
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        _, _, v = torch.pca_lowrank(centered, q=min(latent_dim, min(centered.shape)))
        return cls(mean=mean, components=v[:, :latent_dim])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.components.T + self.mean


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        hidden_dim = min(hidden_dim, max(16, input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class TemporalLSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        max_tokens: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.token_to_hidden = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent_to_decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_positions = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.02)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, token_dim)

    def encode(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.token_to_hidden(sequence)
        encoded, (hidden, _) = self.encoder(projected)
        if token_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = token_mask.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        summary = torch.cat([hidden[-1], pooled], dim=-1)
        return self.to_latent(summary)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        hidden = self.latent_to_hidden(z).reshape(self.num_layers, batch, self.hidden_dim).contiguous()
        cell = torch.zeros_like(hidden)
        decoder_step = self.latent_to_decoder_input(z).unsqueeze(1)
        decoder_input = decoder_step + self.decoder_positions.unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output(decoded)

    def forward(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(sequence, token_mask=token_mask))


class TemporalPositionwiseAutoEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        max_tokens: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.token_to_hidden = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_positions = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def encode(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.token_to_hidden(sequence)
        encoded, (hidden, _) = self.encoder(projected)
        if token_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = token_mask.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        summary = torch.cat([hidden[-1], pooled], dim=-1)
        return self.to_latent(summary)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        latent = self.latent_to_decoder_input(z).unsqueeze(1).expand(batch, self.max_tokens, self.hidden_dim)
        positions = self.decoder_positions.unsqueeze(0).expand(batch, self.max_tokens, self.hidden_dim)
        return self.decoder(torch.cat([latent, positions], dim=-1))

    def forward(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(sequence, token_mask=token_mask))


class TemporalChunkedAutoEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        max_tokens: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_size = max(1, int(chunk_size))
        self.num_chunks = (max_tokens + self.chunk_size - 1) // self.chunk_size
        self.token_to_hidden = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.chunk_positions = nn.Parameter(torch.randn(self.num_chunks, hidden_dim) * 0.02)
        self.token_positions = nn.Parameter(torch.randn(self.chunk_size, hidden_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def _pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        padded_tokens = self.num_chunks * self.chunk_size
        if sequence.shape[1] == padded_tokens:
            return sequence
        pad_tokens = padded_tokens - sequence.shape[1]
        return torch.nn.functional.pad(sequence, (0, 0, 0, pad_tokens))

    def _pad_mask(self, token_mask: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        padded_tokens = self.num_chunks * self.chunk_size
        if token_mask is None:
            return torch.ones((batch, padded_tokens), dtype=torch.bool, device=device)
        mask = token_mask.to(device=device, dtype=torch.bool)
        if mask.shape[1] == padded_tokens:
            return mask
        return torch.nn.functional.pad(mask, (0, padded_tokens - mask.shape[1]))

    def encode(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = sequence.shape[0]
        padded = self._pad_sequence(sequence)
        mask = self._pad_mask(token_mask, batch, sequence.device)
        hidden = self.token_to_hidden(padded)
        hidden = hidden.reshape(batch, self.num_chunks, self.chunk_size, self.hidden_dim)
        chunk_mask = mask.reshape(batch, self.num_chunks, self.chunk_size).float().unsqueeze(-1)
        pooled = (hidden * chunk_mask).sum(dim=2) / chunk_mask.sum(dim=2).clamp_min(1.0)
        return self.to_latent(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(z.shape[0], self.num_chunks, z.shape[-1])
        if z.shape[1] != self.num_chunks:
            raise ValueError(f"Expected {self.num_chunks} latent chunks, got {z.shape[1]}")
        batch = z.shape[0]
        latent = self.latent_to_hidden(z).unsqueeze(2).expand(batch, self.num_chunks, self.chunk_size, self.hidden_dim)
        chunk_pos = self.chunk_positions.unsqueeze(0).unsqueeze(2).expand(
            batch, self.num_chunks, self.chunk_size, self.hidden_dim
        )
        token_pos = self.token_positions.unsqueeze(0).unsqueeze(0).expand(
            batch, self.num_chunks, self.chunk_size, self.hidden_dim
        )
        decoded = self.decoder(torch.cat([latent, chunk_pos, token_pos], dim=-1))
        return decoded.reshape(batch, self.num_chunks * self.chunk_size, self.token_dim)[:, : self.max_tokens, :]

    def forward(self, sequence: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(sequence, token_mask=token_mask))


def _temporal_model_class(codec_kind: str) -> type[nn.Module]:
    kind = codec_kind.lower()
    if kind in {"lstm", "temporal_lstm_rae"}:
        return TemporalLSTMAutoEncoder
    if kind in {"positionwise", "mlp", "temporal_positionwise_rae"}:
        return TemporalPositionwiseAutoEncoder
    if kind in {"chunked", "temporal_chunked_rae"}:
        return TemporalChunkedAutoEncoder
    raise ValueError("temporal codec kind must be lstm, positionwise, or chunked")


def train_autoencoder(
    x: torch.Tensor,
    latent_dim: int,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    progress_path: Path | None = None,
    log_every: int = 1,
) -> tuple[AutoEncoder, float, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = AutoEncoder(input_dim=x.shape[1], latent_dim=latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        reconstructed = model(x)
        loss = torch.mean((reconstructed - x) ** 2)
        loss.backward()
        opt.step()
        event = {
            "epoch": epoch,
            "epochs": epochs,
            "method": "autoencoder",
            "loss": float(loss.detach().item()),
            "elapsed_s": time.perf_counter() - start,
        }
        history.append(event)
        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            _append_training_event(progress_path, event)
            print(
                f"[autoencoder epoch {epoch}/{epochs}] loss={event['loss']:.6g} elapsed={event['elapsed_s']:.1f}s",
                flush=True,
            )
    with torch.no_grad():
        mse = reconstruction_mse(x, model(x))
    return model, mse, history


def train_temporal_lstm_autoencoder(
    x: torch.Tensor,
    shapes: list[Any],
    aligned_shapes: list[Any],
    latent_dim: int,
    cache_paths: list[str] | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    weight_decay: float = 1e-2,
    hidden_dim: int = 128,
    num_layers: int = 1,
    llm_model_id: str | None = None,
    llm_device_name: str = "auto",
    llm_loss_weight: float = 0.0,
    llm_steps: int = 1,
    replay_loss_weight: float = 0.0,
    replay_loss_steps: int = 0,
    source_labels: list[dict[str, Any]] | None = None,
    progress_path: Path | None = None,
    log_every: int = 1,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    heartbeat_every_batches: int = 100,
    train_batch_size: int = 0,
    resume_checkpoint_path: Path | None = None,
    grad_clip_norm: float = 0.0,
    mps_empty_cache_every_batches: int = 0,
    replay_loss_every_n_batches: int = 1,
    temporal_codec_kind: str = "chunked",
    temporal_chunk_size: int = 1,
    checkpoint_stem: str = "rae_temporal",
) -> tuple[nn.Module, torch.Tensor, float, dict[str, Any], list[dict[str, Any]]]:
    print("[rae_temporal] Starting training: seeding, preparing data...", flush=True)
    torch.manual_seed(seed)
    llm_loss_weight = float(llm_loss_weight)
    replay_loss_weight = float(replay_loss_weight)
    if llm_loss_weight != 0.0:
        raise ValueError(
            "llm_loss_weight is deprecated because prompt-prefix KL can bias the codec away from "
            "trajectory reconstruction. Use replay_loss_weight for teacher-forced generated-token replay KL."
        )
    training_device = choose_device(llm_device_name)
    print(f"[rae_temporal] Device selected: {training_device}", flush=True)
    sequence = _temporal_matrix(x, aligned_shapes)
    print(f"[rae_temporal] Temporal matrix shape: {sequence.shape}", flush=True)
    token_mask = _temporal_token_mask(shapes, aligned_shapes)
    feature_mask = token_mask.unsqueeze(-1).float()
    denominator = (feature_mask.sum() * sequence.shape[-1]).clamp_min(1.0)
    counts = feature_mask.sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
    masked_sequence = sequence * feature_mask
    mean = masked_sequence.sum(dim=(0, 1), keepdim=True) / counts
    variance = (((sequence - mean) * feature_mask) ** 2).sum(dim=(0, 1), keepdim=True) / counts
    std = variance.sqrt().clamp_min(1e-6)
    normalized = ((sequence - mean) / std) * feature_mask
    print(f"[rae_temporal] Normalization complete. mean shape: {mean.shape}, std shape: {std.shape}", flush=True)
    mean_device = mean.to(training_device)
    std_device = std.to(training_device)
    model_class = _temporal_model_class(temporal_codec_kind)
    print(f"[rae_temporal] Constructing {temporal_codec_kind} model...", flush=True)
    model_kwargs = {
        "token_dim": sequence.shape[-1],
        "max_tokens": sequence.shape[1],
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    if temporal_codec_kind == "chunked":
        model_kwargs["chunk_size"] = max(1, int(temporal_chunk_size))
    model = model_class(**model_kwargs).to(training_device)
    print("[rae_temporal] Model constructed.", flush=True)
    resume_epoch = 0
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        checkpoint_seq_len = int(checkpoint.get("seq_len", sequence.shape[1]))
        checkpoint_token_dim = int(checkpoint.get("token_dim", sequence.shape[-1]))
        if checkpoint_seq_len != int(sequence.shape[1]) or checkpoint_token_dim != int(sequence.shape[-1]):
            raise ValueError(
                "Resume checkpoint shape does not match current temporal matrix: "
                f"checkpoint seq/token=({checkpoint_seq_len}, {checkpoint_token_dim}) "
                f"current=({sequence.shape[1]}, {sequence.shape[-1]})"
            )
        model.load_state_dict(checkpoint["state_dict"])
        resume_epoch = int(checkpoint.get("epoch") or 0)
        print(f"[rae_temporal] Resumed model weights from epoch {resume_epoch}: {resume_checkpoint_path}", flush=True)
    llm = None
    tokenizer = None
    llm_bundles: list[dict[str, Any]] = []
    replay_token_ids: list[torch.Tensor] = []
    target_replay_logits: list[list[torch.Tensor]] = []
    if replay_loss_weight > 0:
        print("[rae_temporal] Loading LLM for teacher-forced replay loss...", flush=True)
        if not cache_paths:
            raise ValueError("cache_paths are required when replay loss is enabled")
        first_bundle = load_cache_bundle(Path(cache_paths[0]))
        chosen_model_id = llm_model_id or str((first_bundle.get("metadata") or {}).get("model_id") or "")
        if not chosen_model_id:
            raise ValueError("llm_model_id is required when replay loss is enabled and cache metadata has no model_id")
        llm, tokenizer = load_model_and_tokenizer(chosen_model_id, training_device, local_files_only=True)
        expected_layers = int(getattr(llm.config, "num_hidden_layers", len(shapes[0])))
        if len(shapes[0]) != expected_layers:
            raise ValueError(
                f"Replay loss requires full-layer caches; got {len(shapes[0])} cache layers for model with {expected_layers} layers"
            )
        for parameter in llm.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            for path in cache_paths:
                bundle = load_cache_bundle(Path(path))
                original_cache = cache_to_device(bundle["cache"], training_device)
                output_text = None
                if source_labels is not None and len(source_labels) > len(llm_bundles):
                    output_text = source_labels[len(llm_bundles)].get("output_text")
                token_ids = _generation_tokens_for_replay(bundle, replay_loss_steps, tokenizer=tokenizer, output_text=output_text)
                replay_token_ids.append(token_ids.cpu())
                target_replay_logits.append(
                    [
                        logits.detach().cpu()
                        for logits in _teacher_forced_generation_logits(
                            bundle,
                            original_cache,
                            token_ids.to(training_device),
                            llm,
                            training_device,
                        )
                    ]
                )
                llm_bundles.append(
                    {
                        "input_ids": bundle.get("input_ids"),
                        "attention_mask": bundle.get("attention_mask"),
                        "generation_config": bundle.get("generation_config") or {},
                    }
                )
                del original_cache, token_ids
                if training_device.type == "mps":
                    _empty_device_cache(training_device)
        print("[rae_temporal] Teacher-forced replay loss targets loaded.", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    if checkpoint_every > 0 and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(train_batch_size) if train_batch_size and train_batch_size > 0 else int(normalized.shape[0])
    batch_size = max(1, min(batch_size, int(normalized.shape[0])))
    # Log startup event
    startup_event = {
        "event": "startup",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "device": str(training_device),
        "input_shape": list(x.shape),
        "temporal_matrix_shape": list(sequence.shape),
        "batch_size": batch_size,
        "epochs": epochs,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "deprecated_llm_loss_weight": llm_loss_weight,
        "deprecated_llm_steps": llm_steps,
        "replay_loss_weight": replay_loss_weight,
        "replay_loss_steps": replay_loss_steps,
        "log_every": log_every,
        "checkpoint_every": checkpoint_every,
        "heartbeat_every_batches": heartbeat_every_batches,
        "resume_checkpoint_path": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "resume_epoch": resume_epoch,
        "grad_clip_norm": float(grad_clip_norm),
        "mps_empty_cache_every_batches": int(mps_empty_cache_every_batches),
        "replay_loss_every_n_batches": int(replay_loss_every_n_batches),
        "temporal_codec_kind": temporal_codec_kind,
        "temporal_chunk_size": getattr(model, "chunk_size", None),
        "note": "Startup event before first epoch."
    }
    _append_training_event(progress_path, startup_event)
    print("[rae_temporal] Startup event logged. Beginning training loop...", flush=True)
    replay_loss_every_n_batches = max(1, int(replay_loss_every_n_batches))
    for epoch in range(resume_epoch + 1, epochs + 1):
        try:
            epoch_start = time.perf_counter()
            reconstruction_numerator = 0.0
            denominator_total = 0.0
            replay_loss_total = 0.0
            replay_loss_observations = 0
            total_batches = (int(normalized.shape[0]) + batch_size - 1) // batch_size
            for batch_index, batch_start in enumerate(range(0, int(normalized.shape[0]), batch_size), start=1):
                batch_stop = min(batch_start + batch_size, int(normalized.shape[0]))
                batch_normalized = normalized[batch_start:batch_stop].to(training_device)
                batch_token_mask = token_mask[batch_start:batch_stop].to(training_device)
                batch_feature_mask = feature_mask[batch_start:batch_stop].to(training_device)
                batch_denominator = (batch_feature_mask.sum() * sequence.shape[-1]).clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                reconstructed_norm = model(batch_normalized, token_mask=batch_token_mask)
                reconstruction_loss = (((reconstructed_norm - batch_normalized) ** 2) * batch_feature_mask).sum() / batch_denominator
                replay_loss = reconstructed_norm.new_tensor(0.0)
                replay_loss_active = (
                    llm is not None
                    and replay_loss_weight > 0
                    and ((batch_index - 1) % replay_loss_every_n_batches == 0)
                )
                if replay_loss_active:
                    reconstructed_sequence = ((reconstructed_norm * std_device) + mean_device) * batch_feature_mask
                    replay_losses = []
                    for local_idx, row in enumerate(reconstructed_sequence):
                        row_idx = batch_start + local_idx
                        aligned = _temporal_to_aligned_vector_grad(row, aligned_shapes)
                        compact = _aligned_to_compact_grad(aligned, shapes[row_idx], aligned_shapes)
                        reconstructed_cache = _unflatten_cache_grad(compact, shapes[row_idx])
                        predicted_replay_logits = _teacher_forced_generation_logits(
                            llm_bundles[row_idx],
                            reconstructed_cache,
                            replay_token_ids[row_idx].to(training_device),
                            llm,
                            training_device,
                        )
                        for predicted, target in zip(predicted_replay_logits, target_replay_logits[row_idx]):
                            target_probs = torch.softmax(target.detach().to(training_device).float(), dim=-1)
                            predicted_log_probs = torch.log_softmax(predicted.float(), dim=-1)
                            replay_losses.append(torch.nn.functional.kl_div(predicted_log_probs, target_probs, reduction="batchmean"))
                    if replay_losses:
                        replay_loss = torch.stack(replay_losses).mean()
                        replay_loss_observations += 1
                    del reconstructed_sequence, replay_losses
                loss = reconstruction_loss + (replay_loss_weight * replay_loss)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()
                reconstruction_numerator += float((reconstruction_loss.detach() * batch_denominator).item())
                denominator_total += float(batch_denominator.detach().item())
                replay_loss_total += float(replay_loss.detach().item())
                if (
                    heartbeat_every_batches > 0
                    and (
                        batch_index == 1
                        or batch_index == total_batches
                        or batch_index % heartbeat_every_batches == 0
                    )
                ):
                    partial_reconstruction_loss = reconstruction_numerator / max(denominator_total, 1.0)
                    partial_sampled_replay_loss = replay_loss_total / max(replay_loss_observations, 1)
                    partial_effective_replay_loss = replay_loss_total / max(batch_index, 1)
                    heartbeat = {
                        "event": "batch_heartbeat",
                        "elapsed_s": time.perf_counter() - start,
                        "epoch_elapsed_s": time.perf_counter() - epoch_start,
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_index,
                        "batches": total_batches,
                        "batch_start": batch_start,
                        "batch_stop": batch_stop,
                        "training_method": "rae_temporal",
                        "partial_loss": partial_reconstruction_loss + (replay_loss_weight * partial_effective_replay_loss),
                        "partial_loss_components": {
                            "masked_temporal_reconstruction_mse": partial_reconstruction_loss,
                            "teacher_forced_generation_replay_kl": partial_sampled_replay_loss,
                            "teacher_forced_generation_replay_kl_effective": partial_effective_replay_loss,
                        },
                        "replay_loss_every_n_batches": replay_loss_every_n_batches,
                        "replay_loss_observations": replay_loss_observations,
                        "memory_gb": _current_memory_gb(),
                    }
                    _append_training_event(progress_path, heartbeat)
                    print(
                        f"[rae_temporal epoch {epoch}/{epochs} batch {batch_index}/{total_batches}] "
                        f"partial_loss={heartbeat['partial_loss']:.6g} "
                        f"elapsed={heartbeat['elapsed_s']:.1f}s",
                        flush=True,
                    )
                del batch_normalized, batch_token_mask, batch_feature_mask, reconstructed_norm, reconstruction_loss, replay_loss, loss
                if mps_empty_cache_every_batches > 0 and batch_index % int(mps_empty_cache_every_batches) == 0:
                    gc.collect()
                    _empty_device_cache(training_device)
            mean_reconstruction_loss = reconstruction_numerator / max(denominator_total, 1.0)
            mean_sampled_replay_loss = replay_loss_total / max(replay_loss_observations, 1)
            mean_effective_replay_loss = replay_loss_total / max(total_batches, 1)
            mean_loss = mean_reconstruction_loss + (replay_loss_weight * mean_effective_replay_loss)
            mem_gb = _current_memory_gb()
            event = {
                "elapsed_s": time.perf_counter() - start,
                "epoch_elapsed_s": time.perf_counter() - epoch_start,
                "epoch": epoch,
                "epochs": epochs,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "loss": mean_loss,
                "loss_components": {
                    "masked_temporal_reconstruction_mse": mean_reconstruction_loss,
                    "teacher_forced_generation_replay_kl": mean_sampled_replay_loss,
                    "teacher_forced_generation_replay_kl_effective": mean_effective_replay_loss,
                },
                "method": "rae_temporal",
                "masked_loss": True,
                "objective": "masked_temporal_reconstruction_mse_plus_optional_teacher_forced_generation_replay_kl",
                "seq_len": model.max_tokens,
                "token_dim": model.token_dim,
                "valid_tokens": int(token_mask.sum().item()),
                "valid_values": int(feature_mask.sum().item() * sequence.shape[-1]),
                "deprecated_llm_loss_weight": llm_loss_weight,
                "deprecated_llm_steps": int(llm_steps),
                "replay_loss_weight": replay_loss_weight,
                "replay_loss_steps": int(replay_loss_steps),
                "replay_gradients": llm is not None and replay_loss_weight > 0,
                "replay_loss_every_n_batches": replay_loss_every_n_batches,
                "replay_loss_observations": replay_loss_observations,
                "grad_clip_norm": float(grad_clip_norm),
                "mps_empty_cache_every_batches": int(mps_empty_cache_every_batches),
                "weight_decay": weight_decay,
                "train_batch_size": batch_size,
                "memory_gb": mem_gb,
                "temporal_codec_kind": temporal_codec_kind,
                "temporal_chunk_size": getattr(model, "chunk_size", None),
            }
            history.append(event)
            if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                _append_training_event(progress_path, event)
                print(
                    f"[rae_temporal epoch {epoch}/{epochs}] loss={event['loss']:.6g} "
                    f"mse={event['loss_components']['masked_temporal_reconstruction_mse']:.6g} "
                    f"replay_kl={event['loss_components']['teacher_forced_generation_replay_kl']:.6g} "
                    f"seq_len={model.max_tokens} token_dim={model.token_dim} elapsed={event['elapsed_s']:.1f}s "
                    f"mem={mem_gb:.2f}GB",
                    flush=True,
                )
            if checkpoint_every > 0 and checkpoint_dir is not None and (epoch == 1 or epoch == epochs or epoch % checkpoint_every == 0):
                checkpoint = {
                    "method": "rae_temporal",
                    "epoch": epoch,
                    "epochs": epochs,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "normalization_mean": mean.detach().cpu(),
                    "normalization_std": std.detach().cpu(),
                    "latent_dim": latent_dim,
                    "hidden_dim": model.hidden_dim,
                    "num_layers": model.num_layers,
                    "seq_len": model.max_tokens,
                    "token_dim": model.token_dim,
                    "latest_event": event,
                    "deprecated_llm_loss_weight": llm_loss_weight,
                    "deprecated_llm_steps": int(llm_steps),
                    "replay_loss_weight": replay_loss_weight,
                    "replay_loss_steps": int(replay_loss_steps),
                    "replay_loss_every_n_batches": replay_loss_every_n_batches,
                    "grad_clip_norm": float(grad_clip_norm),
                    "mps_empty_cache_every_batches": int(mps_empty_cache_every_batches),
                    "weight_decay": weight_decay,
                    "temporal_codec_kind": temporal_codec_kind,
                    "chunk_size": getattr(model, "chunk_size", None),
                }
                checkpoint_path = checkpoint_dir / f"{checkpoint_stem}_epoch_{epoch:06d}.pt"
                _atomic_torch_save(checkpoint, checkpoint_path)
                _atomic_torch_save(checkpoint, checkpoint_dir / f"{checkpoint_stem}_latest.pt")
        except Exception as exc:
            _append_training_event(
                progress_path,
                {
                    "event": "training_error",
                    "elapsed_s": time.perf_counter() - start,
                    "epoch": epoch,
                    "epochs": epochs,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "method": "rae_temporal",
                    "memory_gb": _current_memory_gb(),
                },
            )
            print(f"[ERROR][epoch {epoch}] {type(exc).__name__}: {exc}", flush=True)
            raise
    with torch.no_grad():
        latent_batches = []
        reconstructed_batches = []
        for batch_start in range(0, int(normalized.shape[0]), batch_size):
            batch_stop = min(batch_start + batch_size, int(normalized.shape[0]))
            batch_token_mask = token_mask[batch_start:batch_stop].to(training_device)
            batch_feature_mask = feature_mask[batch_start:batch_stop].to(training_device)
            batch_z = model.encode(normalized[batch_start:batch_stop].to(training_device), token_mask=batch_token_mask).detach()
            batch_reconstructed = (model.decode(batch_z).detach() * std_device) + mean_device
            latent_batches.append(batch_z.cpu())
            reconstructed_batches.append((batch_reconstructed * batch_feature_mask).cpu())
        z = torch.cat(latent_batches, dim=0)
        reconstructed_sequence = torch.cat(reconstructed_batches, dim=0)
        reconstructed = _temporal_reconstructions_to_aligned(reconstructed_sequence.cpu(), aligned_shapes)
        mse = reconstruction_mse(x, reconstructed)
    model = model.cpu()
    stats = {
        "normalization_mean": mean.detach().cpu(),
        "normalization_std": std.detach().cpu(),
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "seq_len": model.max_tokens,
        "token_dim": model.token_dim,
        "weight_decay": weight_decay,
        "masked_loss": True,
        "objective": "masked_temporal_reconstruction_mse_plus_optional_teacher_forced_generation_replay_kl",
        "deprecated_llm_loss_weight": llm_loss_weight,
        "deprecated_llm_steps": int(llm_steps),
        "teacher_forced_generation_replay_kl_weight": replay_loss_weight,
        "teacher_forced_generation_replay_steps": int(replay_loss_steps),
        "teacher_forced_generation_replay_gradients": replay_loss_weight > 0,
        "replay_loss_every_n_batches": replay_loss_every_n_batches,
        "resume_checkpoint_path": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "resume_epoch": resume_epoch,
        "grad_clip_norm": float(grad_clip_norm),
        "mps_empty_cache_every_batches": int(mps_empty_cache_every_batches),
        "regularization": "adamw_weight_decay_only",
        "train_batch_size": batch_size,
        "decoder_conditioning": "chunk_latents_plus_learned_chunk_and_token_positions"
        if temporal_codec_kind == "chunked"
        else "latent_repeated_input_plus_learned_temporal_position",
        "temporal_codec_kind": temporal_codec_kind,
        "chunk_size": getattr(model, "chunk_size", None),
        "num_chunks": getattr(model, "num_chunks", None),
        "latent_summary": "per_chunk_masked_mean_encoded"
        if temporal_codec_kind == "chunked"
        else "last_hidden_plus_masked_mean_encoded",
        "token_projection": "linear_layernorm_gelu",
        "latent_encoding_input": "masked_normalized_temporal_token_cache",
        "input_representation": "temporal_full_cache_token_states",
        "vector_alignment": "per_layer_key_value_token_padding",
        "decoder_source": "latent_only",
        "uses_retrieval_residual": False,
        "uses_per_cache_residual": False,
    }
    return model, reconstructed, mse, stats, history


def nearest_neighbor_reconstruct(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    if x.shape[0] == 1:
        return x.clone(), 0.0
    distances = torch.cdist(x, x)
    distances.fill_diagonal_(float("inf"))
    nn_idx = distances.argmin(dim=1)
    reconstructed = x[nn_idx]
    return reconstructed, reconstruction_mse(x, reconstructed)


def run_compression(
    run_dir: Path,
    method: str,
    latent_dim: int,
    seed: int = 0,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_dim: int = 128,
    num_layers: int = 1,
    llm_model_id: str | None = None,
    llm_device_name: str = "auto",
    llm_loss_weight: float = 0.0,
    llm_steps: int = 1,
    replay_loss_weight: float = 0.0,
    replay_loss_steps: int = 0,
    log_every: int = 1,
    checkpoint_every: int = 0,
    heartbeat_every_batches: int = 100,
    train_batch_size: int = 0,
    resume_checkpoint_path: Path | None = None,
    grad_clip_norm: float = 0.0,
    mps_empty_cache_every_batches: int = 0,
    replay_loss_every_n_batches: int = 1,
    temporal_chunk_size: int = 1,
) -> CompressionResult:
    cache_matrix = load_cache_matrix(run_dir)
    x = cache_matrix.matrix
    artifact_dir = run_dir / "compressions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    method = method.lower()

    progress_path = artifact_dir / f"{method}_training.jsonl"
    if progress_path.exists() and resume_checkpoint_path is None:
        progress_path.write_text("", encoding="utf-8")
    temporal_methods = {
        "rae_temporal",
        "temporal_rae",
        "temporal_lstm",
        "rae_temporal_mlp",
        "temporal_positionwise",
        "rae_temporal_chunked",
        "temporal_chunked",
    }
    artifact: dict[str, Any] = {"method": method, "latent_dim": latent_dim}
    if method in {"random", "random_projection"}:
        compressor = RandomProjectionCompressor(x.shape[1], latent_dim, seed=seed)
        z = compressor.encode(x)
        reconstructed = compressor.decode(z)
        artifact.update({"projection": compressor.projection})
    elif method in {"pca", "svd", "pca_svd"}:
        compressor = SVDCompressor.fit(x, latent_dim)
        z = compressor.encode(x)
        reconstructed = compressor.decode(z)
        artifact.update({"mean": compressor.mean, "components": compressor.components})
    elif method in {"ae", "autoencoder"}:
        compressor, mse, history = train_autoencoder(
            x,
            latent_dim,
            epochs=epochs,
            lr=lr,
            seed=seed,
            progress_path=progress_path,
            log_every=log_every,
        )
        z = compressor.encode(x).detach()
        reconstructed = compressor.decode(z).detach()
        artifact.update({"state_dict": compressor.state_dict(), "train_mse": mse, "training_history": history})
    elif method in temporal_methods:
        if method in {"rae_temporal_mlp", "temporal_positionwise"}:
            temporal_codec_kind = "positionwise"
        elif method in {"rae_temporal_chunked", "temporal_chunked"}:
            temporal_codec_kind = "chunked"
        elif method in {"rae_temporal", "temporal_rae"}:
            temporal_codec_kind = "chunked"
        else:
            temporal_codec_kind = "lstm"
        compressor, reconstructed, mse, stats, history = train_temporal_lstm_autoencoder(
            x,
            cache_matrix.shapes,
            cache_matrix.aligned_shapes,
            latent_dim,
            cache_paths=cache_matrix.paths,
            epochs=epochs,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            llm_model_id=llm_model_id,
            llm_device_name=llm_device_name,
            llm_loss_weight=llm_loss_weight,
            llm_steps=llm_steps,
            replay_loss_weight=replay_loss_weight,
            replay_loss_steps=replay_loss_steps,
            source_labels=cache_matrix.labels,
            progress_path=progress_path,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=artifact_dir / f"{method}_checkpoints",
            heartbeat_every_batches=heartbeat_every_batches,
            train_batch_size=train_batch_size,
            resume_checkpoint_path=resume_checkpoint_path,
            grad_clip_norm=grad_clip_norm,
            mps_empty_cache_every_batches=mps_empty_cache_every_batches,
            replay_loss_every_n_batches=replay_loss_every_n_batches,
            temporal_codec_kind=temporal_codec_kind,
            temporal_chunk_size=temporal_chunk_size,
            checkpoint_stem=method,
        )
        mean = stats.pop("normalization_mean")
        std = stats.pop("normalization_std")
        temporal = _temporal_matrix(x, cache_matrix.aligned_shapes)
        token_mask = _temporal_token_mask(cache_matrix.shapes, cache_matrix.aligned_shapes)
        normalized_for_latents = ((temporal - mean) / std) * token_mask.unsqueeze(-1).float()
        with torch.no_grad():
            latent_batches = []
            batch_size = int(stats.get("train_batch_size") or normalized_for_latents.shape[0])
            batch_size = max(1, min(batch_size, int(normalized_for_latents.shape[0])))
            for batch_start in range(0, int(normalized_for_latents.shape[0]), batch_size):
                batch_stop = min(batch_start + batch_size, int(normalized_for_latents.shape[0]))
                latent_batches.append(
                    compressor.encode(
                        normalized_for_latents[batch_start:batch_stop],
                        token_mask=token_mask[batch_start:batch_stop],
                    ).detach()
                )
            z = torch.cat(latent_batches, dim=0)
        artifact.update(
            {
                "state_dict": compressor.state_dict(),
                "train_mse": mse,
                "normalization_mean": mean,
                "normalization_std": std,
                "codec_kind": f"temporal_{temporal_codec_kind}_rae"
                if temporal_codec_kind != "lstm"
                else "temporal_lstm_rae",
                "training_history": history,
                **stats,
            }
        )
    elif method in {"retrieval", "nearest_neighbor_cache"}:
        z = x
        reconstructed, _ = nearest_neighbor_reconstruct(x)
        artifact.update({"note": "nearest-neighbour cache reconstruction baseline"})
    else:
        raise ValueError(
            "method must be random, pca_svd, autoencoder, rae_temporal, rae_temporal_mlp, "
            "rae_temporal_chunked, or retrieval"
        )

    compact_reconstructed = _compact_reconstructions(reconstructed, cache_matrix.shapes, cache_matrix.aligned_shapes)
    compact_original = _compact_reconstructions(x, cache_matrix.shapes, cache_matrix.aligned_shapes)
    mse = reconstruction_mse(compact_original, compact_reconstructed)
    latent_path = artifact_dir / f"{method}_latents.pt"
    artifact_path = artifact_dir / f"{method}_artifact.pt"
    torch.save(
        {
            "latents": z.detach().cpu(),
            "reconstructed": compact_reconstructed.detach().cpu(),
            "cache_paths": cache_matrix.paths,
            "source_labels": cache_matrix.labels,
            "lengths": cache_matrix.lengths,
            "shapes": cache_matrix.shapes,
            "aligned_shapes": cache_matrix.aligned_shapes,
            "vector_alignment": "per_layer_key_value_token_padding",
            "method": method,
            "latent_dim": latent_dim,
            "codec_contract": {
                "point_codec": True,
                "geometry_only": False,
                "input_representation": "temporal_full_cache_token_states"
                if method in temporal_methods
                else "flattened_full_cache",
                "one_latent_per_cache": True,
                "structured_latent_per_cache": method in temporal_methods and artifact.get("temporal_codec_kind") == "chunked",
                "one_global_latent_vector_per_cache": not (
                    method in temporal_methods and artifact.get("temporal_codec_kind") == "chunked"
                ),
                "decodes_to": "flattened_cache_vector",
            },
            "training_log_path": str(progress_path) if progress_path.exists() else None,
        },
        latent_path,
    )
    torch.save(artifact, artifact_path)
    result = CompressionResult(
        run_id=run_dir.name,
        method=method,
        latent_dim=latent_dim,
        records=int(x.shape[0]),
        reconstruction_mse=mse,
        latent_path=str(latent_path),
        artifact_path=str(artifact_path),
        metrics={
            "input_dim": float(x.shape[1]),
            "point_codec": 1.0,
            "training_log_written": 1.0 if progress_path.exists() else 0.0,
            "deprecated_frozen_prompt_transition_gradients": 0.0,
            "teacher_forced_generation_replay_gradients": 1.0
            if method in temporal_methods and replay_loss_weight > 0
            else 0.0,
        },
    )
    write_json(artifact_dir / f"{method}_result.json", result.__dict__)
    return result
