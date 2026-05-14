"""Compression baselines for flattened KV cache vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from .cache import cache_shapes, flatten_cache, load_cache_bundle
from .schemas import CompressionResult, read_jsonl, write_json


@dataclass(frozen=True)
class CacheMatrix:
    matrix: torch.Tensor
    paths: list[str]
    lengths: list[int]
    shapes: list[Any]
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


def load_cache_matrix(run_dir: Path) -> CacheMatrix:
    records = read_jsonl(run_dir / "records.jsonl")
    vectors = []
    paths = []
    shapes = []
    labels = []
    for record in records:
        cache_path = record.get("cache_path")
        if not cache_path:
            continue
        bundle = load_cache_bundle(Path(cache_path))
        cache = bundle["cache"]
        vectors.append(flatten_cache(cache))
        paths.append(cache_path)
        shapes.append(bundle.get("shapes") or cache_shapes(cache))
        labels.append(_record_label(record))
    if not vectors:
        raise ValueError(f"No cache vectors found under {run_dir}")
    max_len = max(vector.numel() for vector in vectors)
    padded = []
    lengths = []
    for vector in vectors:
        lengths.append(int(vector.numel()))
        if vector.numel() < max_len:
            vector = torch.nn.functional.pad(vector, (0, max_len - vector.numel()))
        padded.append(vector)
    return CacheMatrix(
        matrix=torch.stack(padded).float(),
        paths=paths,
        lengths=lengths,
        shapes=shapes,
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


class ChunkedLSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        chunk_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.chunk_dim = max(1, min(chunk_dim, max(1, input_dim)))
        self.padded_dim = ((input_dim + self.chunk_dim - 1) // self.chunk_dim) * self.chunk_dim
        self.seq_len = self.padded_dim // self.chunk_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_to_hidden = nn.Sequential(
            nn.Linear(self.chunk_dim, hidden_dim),
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
        self.decoder_positions = nn.Parameter(torch.randn(self.seq_len, hidden_dim) * 0.02)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, self.chunk_dim)

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - x.shape[1]))
        return x[:, : self.padded_dim].reshape(x.shape[0], self.seq_len, self.chunk_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self._to_sequence(x)
        projected = self.chunk_to_hidden(sequence)
        encoded, (hidden, _) = self.encoder(projected)
        summary = torch.cat([hidden[-1], encoded.mean(dim=1)], dim=-1)
        return self.to_latent(summary)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        hidden = self.latent_to_hidden(z).reshape(self.num_layers, batch, self.hidden_dim).contiguous()
        cell = torch.zeros_like(hidden)
        decoder_step = self.latent_to_decoder_input(z).unsqueeze(1)
        decoder_input = decoder_step + self.decoder_positions.unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        chunks = self.output(decoded).reshape(batch, self.padded_dim)
        return chunks[:, : self.input_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


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


def train_lstm_seq2seq_autoencoder(
    x: torch.Tensor,
    lengths: list[int],
    latent_dim: int,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    weight_decay: float = 1e-2,
    chunk_dim: int = 4096,
    hidden_dim: int = 128,
    progress_path: Path | None = None,
    log_every: int = 1,
) -> tuple[ChunkedLSTMAutoEncoder, torch.Tensor, float, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    valid_mask = torch.zeros_like(x, dtype=torch.bool)
    for row_idx, length in enumerate(lengths):
        valid_mask[row_idx, : int(length)] = True
    counts = valid_mask.sum(dim=0, keepdim=True).clamp_min(1).float()
    masked_x = x.masked_fill(~valid_mask, 0.0)
    mean = masked_x.sum(dim=0, keepdim=True) / counts
    variance = (((x - mean).masked_fill(~valid_mask, 0.0)) ** 2).sum(dim=0, keepdim=True) / counts
    std = variance.sqrt().clamp_min(1e-6)
    normalized = (x - mean) / std
    normalized = normalized.masked_fill(~valid_mask, 0.0)
    loss_mask = valid_mask.float()
    loss_denominator = loss_mask.sum().clamp_min(1.0)
    model = ChunkedLSTMAutoEncoder(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        chunk_dim=chunk_dim,
        hidden_dim=hidden_dim,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        reconstructed = model(normalized)
        loss = (((reconstructed - normalized) ** 2) * loss_mask).sum() / loss_denominator
        loss.backward()
        opt.step()
        event = {
            "chunk_dim": model.chunk_dim,
            "elapsed_s": time.perf_counter() - start,
            "epoch": epoch,
            "epochs": epochs,
            "hidden_dim": model.hidden_dim,
            "loss": float(loss.detach().item()),
            "method": "rae_lstm",
            "masked_loss": True,
            "seq_len": model.seq_len,
            "valid_values": int(loss_mask.sum().item()),
            "weight_decay": weight_decay,
        }
        history.append(event)
        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            _append_training_event(progress_path, event)
            print(
                f"[rae_lstm epoch {epoch}/{epochs}] loss={event['loss']:.6g} "
                f"seq_len={model.seq_len} elapsed={event['elapsed_s']:.1f}s",
                flush=True,
            )
    with torch.no_grad():
        z = model.encode(normalized).detach()
        reconstructed = (model.decode(z).detach() * std) + mean
        mse = reconstruction_mse(x, reconstructed)
    stats = {
        "normalization_mean": mean.detach().cpu(),
        "normalization_std": std.detach().cpu(),
        "chunk_dim": model.chunk_dim,
        "hidden_dim": model.hidden_dim,
        "seq_len": model.seq_len,
        "weight_decay": weight_decay,
        "masked_loss": True,
        "decoder_conditioning": "latent_repeated_input_plus_learned_position",
        "latent_summary": "last_hidden_plus_mean_encoded",
        "chunk_projection": "linear_layernorm_gelu",
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
    chunk_dim: int = 4096,
    hidden_dim: int = 128,
    log_every: int = 1,
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
    elif method in {"rae_lstm", "lstm_seq2seq", "lstm_rae"}:
        compressor, reconstructed, mse, stats, history = train_lstm_seq2seq_autoencoder(
            x,
            cache_matrix.lengths,
            latent_dim,
            epochs=epochs,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            chunk_dim=chunk_dim,
            hidden_dim=hidden_dim,
            progress_path=progress_path,
            log_every=log_every,
        )
        mean = stats.pop("normalization_mean")
        std = stats.pop("normalization_std")
        with torch.no_grad():
            z = compressor.encode((x - mean) / std).detach()
        artifact.update(
            {
                "state_dict": compressor.state_dict(),
                "train_mse": mse,
                "normalization_mean": mean,
                "normalization_std": std,
                "codec_kind": "lstm_seq2seq_rae",
                "training_history": history,
                **stats,
            }
        )
    elif method in {"retrieval", "nearest_neighbor_cache"}:
        z = x
        reconstructed, _ = nearest_neighbor_reconstruct(x)
        artifact.update({"note": "nearest-neighbour cache reconstruction baseline"})
    else:
        raise ValueError("method must be random, pca_svd, autoencoder, or retrieval")

    mse = reconstruction_mse(x, reconstructed)
    latent_path = artifact_dir / f"{method}_latents.pt"
    artifact_path = artifact_dir / f"{method}_artifact.pt"
    torch.save(
        {
            "latents": z.detach().cpu(),
            "reconstructed": reconstructed.detach().cpu(),
            "cache_paths": cache_matrix.paths,
            "source_labels": cache_matrix.labels,
            "lengths": cache_matrix.lengths,
            "shapes": cache_matrix.shapes,
            "method": method,
            "latent_dim": latent_dim,
            "codec_contract": {
                "point_codec": True,
                "geometry_only": False,
                "input_representation": "chunked_flattened_full_cache"
                if method in {"rae_lstm", "lstm_seq2seq", "lstm_rae"}
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
        },
    )
    write_json(artifact_dir / f"{method}_result.json", result.__dict__)
    return result
