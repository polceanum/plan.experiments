"""Validation helpers for KV point-codec artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .cache import CacheTuple, load_cache_bundle, unflatten_cache
from .schemas import write_json


@dataclass(frozen=True)
class CacheValidationResult:
    valid: bool
    errors: list[str]
    layers: int
    token_count: int | None
    finite: bool
    shape_match: bool
    replay_metadata_present: bool


@dataclass(frozen=True)
class CodecArtifactValidation:
    method: str
    records: int
    latent_dim: int | None
    one_point_per_cache: bool
    valid_caches: int
    invalid_caches: int
    mean_reconstruction_mse: float | None
    cache_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_cache_against_bundle(cache: CacheTuple, bundle: dict[str, Any]) -> CacheValidationResult:
    original = bundle["cache"]
    errors: list[str] = []
    if len(cache) != len(original):
        errors.append(f"layer_count_mismatch:{len(cache)}!={len(original)}")

    shape_match = True
    finite = True
    for idx, ((key, value), (orig_key, orig_value)) in enumerate(zip(cache, original)):
        if tuple(key.shape) != tuple(orig_key.shape):
            shape_match = False
            errors.append(f"layer_{idx}_key_shape:{tuple(key.shape)}!={tuple(orig_key.shape)}")
        if tuple(value.shape) != tuple(orig_value.shape):
            shape_match = False
            errors.append(f"layer_{idx}_value_shape:{tuple(value.shape)}!={tuple(orig_value.shape)}")
        if not torch.isfinite(key).all().item():
            finite = False
            errors.append(f"layer_{idx}_key_nonfinite")
        if not torch.isfinite(value).all().item():
            finite = False
            errors.append(f"layer_{idx}_value_nonfinite")

    replay_metadata_present = bundle.get("input_ids") is not None and bundle.get("last_logits") is not None
    if not replay_metadata_present:
        errors.append("missing_replay_metadata")

    token_count = None
    if cache:
        token_count = int(cache[0][0].shape[-2])
    return CacheValidationResult(
        valid=not errors,
        errors=errors,
        layers=len(cache),
        token_count=token_count,
        finite=finite,
        shape_match=shape_match,
        replay_metadata_present=replay_metadata_present,
    )


def validate_reconstructed_artifact(run_dir: Path, method: str) -> CodecArtifactValidation:
    method = method.lower()
    latent_path = run_dir / "compressions" / f"{method}_latents.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing compression artifact: {latent_path}")
    payload = torch.load(latent_path, map_location="cpu")
    required = {"latents", "reconstructed", "cache_paths", "lengths", "shapes"}
    if not required <= set(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"{latent_path} is missing required codec fields: {missing}")

    latents = payload["latents"]
    reconstructed = payload["reconstructed"]
    cache_paths = list(payload["cache_paths"])
    lengths = list(payload["lengths"])
    shapes = list(payload["shapes"])
    one_point_per_cache = int(latents.shape[0]) == len(cache_paths) == len(lengths) == len(shapes)
    cache_results: list[dict[str, Any]] = []
    mses: list[float] = []
    valid_count = 0

    for idx, cache_path in enumerate(cache_paths):
        bundle = load_cache_bundle(Path(cache_path))
        length = int(lengths[idx])
        vector = reconstructed[idx, :length]
        cache = unflatten_cache(vector, shapes[idx])
        original_vector = torch.cat([tensor.reshape(-1).float() for layer in bundle["cache"] for tensor in layer])
        mse = float(torch.mean((vector.float() - original_vector) ** 2).item())
        mses.append(mse)
        result = validate_cache_against_bundle(cache, bundle)
        if result.valid:
            valid_count += 1
        cache_results.append(
            {
                "cache_path": cache_path,
                "mse": mse,
                "validation": asdict(result),
            }
        )

    latent_dim = None
    if hasattr(latents, "ndim") and latents.ndim >= 2:
        latent_dim = int(latents.shape[-1])
    validation = CodecArtifactValidation(
        method=method,
        records=len(cache_paths),
        latent_dim=latent_dim,
        one_point_per_cache=one_point_per_cache,
        valid_caches=valid_count,
        invalid_caches=len(cache_paths) - valid_count,
        mean_reconstruction_mse=(sum(mses) / len(mses)) if mses else None,
        cache_results=cache_results,
    )
    out_path = run_dir / "compressions" / f"{method}_validation.json"
    write_json(out_path, validation.to_dict())
    return validation
