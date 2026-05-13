"""Compression baselines for flattened KV cache vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def load_cache_matrix(run_dir: Path) -> CacheMatrix:
    records = read_jsonl(run_dir / "records.jsonl")
    vectors = []
    paths = []
    shapes = []
    for record in records:
        cache_path = record.get("cache_path")
        if not cache_path:
            continue
        bundle = load_cache_bundle(Path(cache_path))
        cache = bundle["cache"]
        vectors.append(flatten_cache(cache))
        paths.append(cache_path)
        shapes.append(bundle.get("shapes") or cache_shapes(cache))
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
    )


def reconstruction_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(torch.mean((original.float() - reconstructed.float()) ** 2).item())


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


def train_autoencoder(
    x: torch.Tensor,
    latent_dim: int,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[AutoEncoder, float]:
    torch.manual_seed(seed)
    model = AutoEncoder(input_dim=x.shape[1], latent_dim=latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        reconstructed = model(x)
        loss = torch.mean((reconstructed - x) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = reconstruction_mse(x, model(x))
    return model, mse


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
) -> CompressionResult:
    cache_matrix = load_cache_matrix(run_dir)
    x = cache_matrix.matrix
    artifact_dir = run_dir / "compressions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    method = method.lower()

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
        compressor, mse = train_autoencoder(x, latent_dim, epochs=epochs, seed=seed)
        z = compressor.encode(x).detach()
        reconstructed = compressor.decode(z).detach()
        artifact.update({"state_dict": compressor.state_dict(), "train_mse": mse})
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
            "lengths": cache_matrix.lengths,
            "shapes": cache_matrix.shapes,
            "method": method,
            "latent_dim": latent_dim,
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
        metrics={"input_dim": float(x.shape[1])},
    )
    write_json(artifact_dir / f"{method}_result.json", result.__dict__)
    return result
