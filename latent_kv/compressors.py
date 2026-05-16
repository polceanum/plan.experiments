"""Compression baselines for flattened KV cache vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
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


def _cache_prefix(cache: Any, token_count: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((key[..., :token_count, :], value[..., :token_count, :]) for key, value in cache)


def _prompt_state_transition_logits(
    bundle: dict[str, Any],
    cache: Any,
    model: Any,
    device: torch.device,
    steps: int,
) -> list[torch.Tensor]:
    input_ids = bundle.get("input_ids")
    attention_mask = bundle.get("attention_mask")
    if input_ids is None:
        raise ValueError("Bundle must contain input_ids for frozen-LLM temporal loss")
    input_ids = input_ids.to(device)
    prompt_len = int(input_ids.shape[-1])
    max_steps = min(max(0, int(steps)), max(0, prompt_len - 1))
    if max_steps == 0:
        return []
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    forward_parameters = _forward_parameters(model)
    logits_by_step: list[torch.Tensor] = []
    for step_idx in range(max_steps):
        prefix_len = step_idx + 1
        model_kwargs = {
            "input_ids": input_ids[:, prefix_len : prefix_len + 1],
            "attention_mask": attention_mask[:, : prefix_len + 1],
            "past_key_values": _cache_prefix(cache, prefix_len),
            "use_cache": True,
            "return_dict": True,
        }
        position = torch.tensor([[prefix_len]], dtype=torch.long, device=device)
        if "position_ids" in forward_parameters:
            model_kwargs["position_ids"] = position
        if "cache_position" in forward_parameters:
            model_kwargs["cache_position"] = position.reshape(-1)
        outputs = model(**model_kwargs)
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
    progress_path: Path | None = None,
    log_every: int = 1,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    train_batch_size: int = 0,
) -> tuple[TemporalLSTMAutoEncoder, torch.Tensor, float, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    llm_loss_weight = float(llm_loss_weight)
    training_device = choose_device(llm_device_name) if llm_loss_weight > 0 else torch.device("cpu")
    sequence = _temporal_matrix(x, aligned_shapes)
    token_mask = _temporal_token_mask(shapes, aligned_shapes)
    feature_mask = token_mask.unsqueeze(-1).float()
    denominator = (feature_mask.sum() * sequence.shape[-1]).clamp_min(1.0)
    counts = feature_mask.sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
    masked_sequence = sequence * feature_mask
    mean = masked_sequence.sum(dim=(0, 1), keepdim=True) / counts
    variance = (((sequence - mean) * feature_mask) ** 2).sum(dim=(0, 1), keepdim=True) / counts
    std = variance.sqrt().clamp_min(1e-6)
    normalized = ((sequence - mean) / std) * feature_mask
    mean_device = mean.to(training_device)
    std_device = std.to(training_device)
    model = TemporalLSTMAutoEncoder(
        token_dim=sequence.shape[-1],
        max_tokens=sequence.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(training_device)
    llm = None
    llm_bundles: list[dict[str, Any]] = []
    target_transition_logits: list[list[torch.Tensor]] = []
    if llm_loss_weight > 0:
        if not cache_paths:
            raise ValueError("cache_paths are required when llm_loss_weight > 0")
        first_bundle = load_cache_bundle(Path(cache_paths[0]))
        chosen_model_id = llm_model_id or str((first_bundle.get("metadata") or {}).get("model_id") or "")
        if not chosen_model_id:
            raise ValueError("llm_model_id is required when cache metadata has no model_id")
        llm, _ = load_model_and_tokenizer(chosen_model_id, training_device, local_files_only=True)
        expected_layers = int(getattr(llm.config, "num_hidden_layers", len(shapes[0])))
        if len(shapes[0]) != expected_layers:
            raise ValueError(
                f"Frozen-LLM loss requires full-layer caches; got {len(shapes[0])} cache layers for model with {expected_layers} layers"
            )
        for parameter in llm.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            for path in cache_paths:
                bundle = load_cache_bundle(Path(path))
                target_transition_logits.append(
                    [
                        logits.detach().cpu()
                        for logits in _prompt_state_transition_logits(
                            bundle,
                            cache_to_device(bundle["cache"], training_device),
                            llm,
                            training_device,
                            llm_steps,
                        )
                    ]
                )
                llm_bundles.append(
                    {
                        "input_ids": bundle.get("input_ids"),
                        "attention_mask": bundle.get("attention_mask"),
                    }
                )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    if checkpoint_every > 0 and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(train_batch_size) if train_batch_size and train_batch_size > 0 else int(normalized.shape[0])
    batch_size = max(1, min(batch_size, int(normalized.shape[0])))
    import traceback
    import sys
    def log_exception_to_file(exc, path):
        try:
            with open(str(path).replace(".jsonl", "_error.log"), "a", encoding="utf-8") as f:
                f.write(f"Exception at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                traceback.print_exc(file=f)
                f.flush()
        except Exception:
            pass

    for epoch in range(1, epochs + 1):
        try:
            import psutil
            reconstruction_numerator = 0.0
            denominator_total = 0.0
            llm_loss_total = 0.0
            llm_loss_batches = 0
            for batch_start in range(0, int(normalized.shape[0]), batch_size):
                batch_stop = min(batch_start + batch_size, int(normalized.shape[0]))
                batch_normalized = normalized[batch_start:batch_stop].to(training_device)
                batch_token_mask = token_mask[batch_start:batch_stop].to(training_device)
                batch_feature_mask = feature_mask[batch_start:batch_stop].to(training_device)
                batch_denominator = (batch_feature_mask.sum() * sequence.shape[-1]).clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                reconstructed_norm = model(batch_normalized, token_mask=batch_token_mask)
                reconstruction_loss = (((reconstructed_norm - batch_normalized) ** 2) * batch_feature_mask).sum() / batch_denominator
                llm_loss = reconstructed_norm.new_tensor(0.0)
                if llm is not None and llm_loss_weight > 0:
                    reconstructed_sequence = ((reconstructed_norm * std_device) + mean_device) * batch_feature_mask
                    transition_losses = []
                    for local_idx, row in enumerate(reconstructed_sequence):
                        row_idx = batch_start + local_idx
                        aligned = _temporal_to_aligned_vector_grad(row, aligned_shapes)
                        compact = _aligned_to_compact_grad(aligned, shapes[row_idx], aligned_shapes)
                        reconstructed_cache = _unflatten_cache_grad(compact, shapes[row_idx])
                        predicted_logits = _prompt_state_transition_logits(
                            llm_bundles[row_idx],
                            reconstructed_cache,
                            llm,
                            training_device,
                            llm_steps,
                        )
                        for predicted, target in zip(predicted_logits, target_transition_logits[row_idx]):
                            target_probs = torch.softmax(target.detach().to(training_device), dim=-1)
                            predicted_log_probs = torch.log_softmax(predicted.float(), dim=-1)
                            transition_losses.append(torch.nn.functional.kl_div(predicted_log_probs, target_probs, reduction="batchmean"))
                    if transition_losses:
                        llm_loss = torch.stack(transition_losses).mean()
                loss = reconstruction_loss + (llm_loss_weight * llm_loss)
                loss.backward()
                opt.step()
                reconstruction_numerator += float((reconstruction_loss.detach() * batch_denominator).item())
                denominator_total += float(batch_denominator.detach().item())
                llm_loss_total += float(llm_loss.detach().item())
                llm_loss_batches += 1
            mean_reconstruction_loss = reconstruction_numerator / max(denominator_total, 1.0)
            mean_llm_loss = llm_loss_total / max(llm_loss_batches, 1)
            mean_loss = mean_reconstruction_loss + (llm_loss_weight * mean_llm_loss)
            # Log memory usage
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_gb = mem_info.rss / (1024 ** 3)
            except Exception:
                mem_gb = -1
            event = {
                "elapsed_s": time.perf_counter() - start,
                "epoch": epoch,
                "epochs": epochs,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "loss": mean_loss,
                "loss_components": {
                    "masked_temporal_reconstruction_mse": mean_reconstruction_loss,
                    "frozen_llm_prompt_transition_kl": mean_llm_loss,
                },
                "method": "rae_temporal",
                "masked_loss": True,
                "objective": "masked_temporal_reconstruction_mse_plus_optional_frozen_llm_prompt_transition_kl",
                "seq_len": model.max_tokens,
                "token_dim": model.token_dim,
                "valid_tokens": int(token_mask.sum().item()),
                "valid_values": int(feature_mask.sum().item() * sequence.shape[-1]),
                "llm_loss_weight": llm_loss_weight,
                "llm_steps": int(llm_steps),
                "llm_gradients": llm is not None and llm_loss_weight > 0,
                "weight_decay": weight_decay,
                "train_batch_size": batch_size,
                "memory_gb": mem_gb,
            }
            history.append(event)
            if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                _append_training_event(progress_path, event)
                print(
                    f"[rae_temporal epoch {epoch}/{epochs}] loss={event['loss']:.6g} "
                    f"mse={event['loss_components']['masked_temporal_reconstruction_mse']:.6g} "
                    f"llm_kl={event['loss_components']['frozen_llm_prompt_transition_kl']:.6g} "
                    f"seq_len={model.max_tokens} token_dim={model.token_dim} elapsed={event['elapsed_s']:.1f}s "
                    f"mem={mem_gb:.2f}GB",
                    flush=True,
                )
                sys.stdout.flush()
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
                    "llm_loss_weight": llm_loss_weight,
                    "llm_steps": int(llm_steps),
                    "weight_decay": weight_decay,
                }
                checkpoint_path = checkpoint_dir / f"rae_temporal_epoch_{epoch:06d}.pt"
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, checkpoint_dir / "rae_temporal_latest.pt")
        except RuntimeError as exc:
            # Catch OOM and other runtime errors, log and re-raise
            log_exception_to_file(exc, progress_path)
            print(f"[ERROR][epoch {epoch}] RuntimeError: {exc}", flush=True)
            sys.stdout.flush()
            if 'out of memory' in str(exc).lower():
                print("[FATAL] Out of memory detected. Stopping training.", flush=True)
                sys.stdout.flush()
                break
            raise
        except Exception as exc:
            log_exception_to_file(exc, progress_path)
            print(f"[ERROR][epoch {epoch}] Exception: {exc}", flush=True)
            sys.stdout.flush()
            break
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
        "objective": "masked_temporal_reconstruction_mse_plus_optional_frozen_llm_prompt_transition_kl",
        "kl_loss_weight": llm_loss_weight,
        "frozen_llm_prompt_transition_kl_weight": llm_loss_weight,
        "frozen_llm_prompt_transition_steps": int(llm_steps),
        "frozen_llm_gradients": llm_loss_weight > 0,
        "regularization": "adamw_weight_decay_only",
        "train_batch_size": batch_size,
        "decoder_conditioning": "latent_repeated_input_plus_learned_temporal_position",
        "latent_summary": "last_hidden_plus_masked_mean_encoded",
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
    log_every: int = 1,
    checkpoint_every: int = 0,
    train_batch_size: int = 0,
) -> CompressionResult:
    cache_matrix = load_cache_matrix(run_dir)
    x = cache_matrix.matrix
    artifact_dir = run_dir / "compressions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    method = method.lower()

    progress_path = artifact_dir / f"{method}_training.jsonl"
    if progress_path.exists():
        progress_path.write_text("", encoding="utf-8")
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
    elif method in {"rae_temporal", "temporal_rae", "temporal_lstm"}:
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
            progress_path=progress_path,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=artifact_dir / f"{method}_checkpoints",
            train_batch_size=train_batch_size,
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
                "codec_kind": "temporal_lstm_rae",
                "training_history": history,
                **stats,
            }
        )
    elif method in {"retrieval", "nearest_neighbor_cache"}:
        z = x
        reconstructed, _ = nearest_neighbor_reconstruct(x)
        artifact.update({"note": "nearest-neighbour cache reconstruction baseline"})
    else:
        raise ValueError("method must be random, pca_svd, autoencoder, rae_temporal, or retrieval")

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
                if method in {"rae_temporal", "temporal_rae", "temporal_lstm"}
                else "flattened_full_cache",
                "one_latent_per_cache": True,
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
            "frozen_llm_gradients": 1.0 if method in {"rae_temporal", "temporal_rae", "temporal_lstm"} and llm_loss_weight > 0 else 0.0,
        },
    )
    write_json(artifact_dir / f"{method}_result.json", result.__dict__)
    return result
