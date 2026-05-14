"""Replay-sensitive diagnostics for reconstructed KV caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .cache import cache_to_device, choose_device, load_cache_bundle, load_model_and_tokenizer, unflatten_cache
from .injection import _apply_repetition_penalty, _forward_parameters
from .schemas import read_json, read_jsonl, write_json


@dataclass(frozen=True)
class ReplayFidelitySummary:
    method: str
    records: int
    mean_logit_cosine: float | None
    mean_logit_mse: float | None
    mean_kl_original_to_reconstructed: float | None
    top1_match_rate: float | None
    source_second_token_match_rate: float | None
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _logit_comparison(original_logits: torch.Tensor, reconstructed_logits: torch.Tensor) -> dict[str, float | bool | int]:
    original = original_logits.detach().float().reshape(-1).cpu()
    reconstructed = reconstructed_logits.detach().float().reshape(-1).cpu()
    original_probs = torch.softmax(original, dim=-1)
    original_log_probs = torch.log_softmax(original, dim=-1)
    reconstructed_log_probs = torch.log_softmax(reconstructed, dim=-1)
    return {
        "logit_cosine": float(torch.nn.functional.cosine_similarity(original, reconstructed, dim=0).item()),
        "logit_mse": float(torch.mean((original - reconstructed) ** 2).item()),
        "kl_original_to_reconstructed": float((original_probs * (original_log_probs - reconstructed_log_probs)).sum().item()),
        "original_top1": int(torch.argmax(original).item()),
        "reconstructed_top1": int(torch.argmax(reconstructed).item()),
        "top1_match": bool(torch.argmax(original).item() == torch.argmax(reconstructed).item()),
    }


def _token_rank(logits: torch.Tensor, token_id: int) -> int:
    flat = logits.detach().float().reshape(-1).cpu()
    target = flat[int(token_id)]
    return int((flat > target).sum().item() + 1)


def _first_replay_token_id(bundle: dict[str, Any], model: Any) -> int:
    generation_token_ids = bundle.get("generation_token_ids")
    if generation_token_ids is not None and int(generation_token_ids.numel()) > 0:
        return int(generation_token_ids.reshape(-1)[0].item())
    input_ids = bundle.get("input_ids")
    last_logits = bundle.get("last_logits")
    if input_ids is None or last_logits is None:
        raise ValueError("Bundle needs generation_token_ids or input_ids plus last_logits")
    penalty = float(getattr(getattr(model, "generation_config", None), "repetition_penalty", 1.0) or 1.0)
    scored = _apply_repetition_penalty(last_logits.float(), input_ids.long(), penalty)
    return int(torch.argmax(scored, dim=-1).reshape(-1)[0].item())


@torch.no_grad()
def _next_logits_after_cache_token(
    bundle: dict[str, Any],
    cache: Any,
    token_id: int,
    model: Any,
    device: torch.device,
) -> torch.Tensor:
    input_ids = bundle.get("input_ids")
    attention_mask = bundle.get("attention_mask")
    if input_ids is None:
        raise ValueError("Bundle must contain input_ids for replay-fidelity diagnostics")
    prompt_len = int(input_ids.shape[-1])
    current_mask = (
        attention_mask.to(device)
        if attention_mask is not None
        else torch.ones((1, prompt_len), dtype=torch.long, device=device)
    )
    current_mask = torch.cat(
        [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=device)],
        dim=-1,
    )
    next_token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    model_kwargs = {
        "input_ids": next_token,
        "attention_mask": current_mask,
        "past_key_values": cache_to_device(cache, device),
        "use_cache": True,
        "return_dict": True,
    }
    replay_position = torch.tensor([[prompt_len]], dtype=torch.long, device=device)
    forward_parameters = _forward_parameters(model)
    if "position_ids" in forward_parameters:
        model_kwargs["position_ids"] = replay_position
    if "cache_position" in forward_parameters:
        model_kwargs["cache_position"] = replay_position.reshape(-1)
    outputs = model(**model_kwargs)
    return outputs.logits[:, -1, :].detach().cpu()


def _load_reconstructed_cache_from_payload(payload: dict[str, Any], cache_path: str) -> Any:
    try:
        idx = list(payload["cache_paths"]).index(cache_path)
    except ValueError as exc:
        raise ValueError(f"{cache_path} is not present in reconstructed payload") from exc
    length = int(payload["lengths"][idx])
    vector = payload["reconstructed"][idx, :length]
    return unflatten_cache(vector, payload["shapes"][idx])


def score_replay_fidelity(
    run_dir: Path,
    method: str,
    model_id: str | None = None,
    device_name: str = "auto",
    limit: int | None = None,
) -> ReplayFidelitySummary:
    method = method.lower()
    latent_path = run_dir / "compressions" / f"{method}_latents.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing reconstructed payload: {latent_path}")
    payload = torch.load(latent_path, map_location="cpu")
    records = [row for row in read_jsonl(run_dir / "records.jsonl") if row.get("cache_path")]
    if limit is not None:
        records = records[: int(limit)]
    if not records:
        raise ValueError(f"No cache-backed records found in {run_dir}")

    chosen_model = model_id or str(records[0]["model_id"])
    device = choose_device(device_name)
    model, _tokenizer = load_model_and_tokenizer(chosen_model, device, local_files_only=True)
    rows: list[dict[str, Any]] = []

    for record in records:
        cache_path = str(record["cache_path"])
        bundle = load_cache_bundle(Path(cache_path))
        first_token = _first_replay_token_id(bundle, model)
        original_logits = _next_logits_after_cache_token(bundle, bundle["cache"], first_token, model, device)
        reconstructed_cache = _load_reconstructed_cache_from_payload(payload, cache_path)
        reconstructed_logits = _next_logits_after_cache_token(bundle, reconstructed_cache, first_token, model, device)
        comparison = _logit_comparison(original_logits, reconstructed_logits)
        generation_token_ids = bundle.get("generation_token_ids")
        source_second_token = None
        if generation_token_ids is not None and int(generation_token_ids.numel()) >= 2:
            source_second_token = int(generation_token_ids.reshape(-1)[1].item())
            comparison["source_second_token"] = source_second_token
            comparison["source_second_token_original_rank"] = _token_rank(original_logits, source_second_token)
            comparison["source_second_token_reconstructed_rank"] = _token_rank(reconstructed_logits, source_second_token)
            comparison["source_second_token_match"] = bool(
                int(comparison["reconstructed_top1"]) == source_second_token
            )
        rows.append(
            {
                "task_id": record.get("task_id"),
                "cache_path": cache_path,
                "first_replay_token": first_token,
                **comparison,
            }
        )

    def mean_float(key: str) -> float | None:
        values = [float(row[key]) for row in rows if key in row]
        return (sum(values) / len(values)) if values else None

    def mean_bool(key: str) -> float | None:
        values = [1.0 if row[key] else 0.0 for row in rows if key in row]
        return (sum(values) / len(values)) if values else None

    out_path = run_dir / "compressions" / f"{method}_replay_fidelity.json"
    summary = ReplayFidelitySummary(
        method=method,
        records=len(rows),
        mean_logit_cosine=mean_float("logit_cosine"),
        mean_logit_mse=mean_float("logit_mse"),
        mean_kl_original_to_reconstructed=mean_float("kl_original_to_reconstructed"),
        top1_match_rate=mean_bool("top1_match"),
        source_second_token_match_rate=mean_bool("source_second_token_match"),
        result_path=str(out_path),
    )
    write_json(out_path, {"summary": summary.to_dict(), "records": rows})

    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {"baselines": [], "extra": {}}
    extra = metrics.setdefault("extra", {})
    prefix = f"replay_fidelity_{method}"
    extra[f"{prefix}_records"] = len(rows)
    extra[f"{prefix}_path"] = str(out_path)
    extra[f"{prefix}_mean_logit_cosine"] = summary.mean_logit_cosine
    extra[f"{prefix}_mean_logit_mse"] = summary.mean_logit_mse
    extra[f"{prefix}_mean_kl_original_to_reconstructed"] = summary.mean_kl_original_to_reconstructed
    extra[f"{prefix}_top1_match_rate"] = summary.top1_match_rate
    extra[f"{prefix}_source_second_token_match_rate"] = summary.source_second_token_match_rate
    write_json(metrics_path, metrics)
    return summary