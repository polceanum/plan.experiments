"""Replay sensitivity to interpolated KV reconstruction error."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .benchmarks import verify_output
from .cache import choose_device, flatten_cache, load_cache_bundle, load_model_and_tokenizer, unflatten_cache
from .injection import greedy_continue_from_loaded_bundle
from .schemas import TaskExample, read_jsonl, write_json


@dataclass(frozen=True)
class CorruptionAlphaSummary:
    alpha: float
    records: int
    correct: int
    accuracy: float
    mean_mse_to_original: float
    mean_output_words: float
    wrong_task_ids: list[str]
    replay_errors: int


@dataclass(frozen=True)
class CorruptionSensitivitySummary:
    run_id: str
    method: str
    alphas: list[CorruptionAlphaSummary]
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alphas"] = [asdict(row) for row in self.alphas]
        return payload


def _record_to_example(record: dict[str, Any]) -> TaskExample:
    return TaskExample(
        benchmark=record["benchmark"],
        task_id=record["task_id"],
        prompt=record["prompt"],
        answer=record["target"],
        metadata=record.get("metadata", {}),
    )


def _load_reconstruction_payload(run_dir: Path, method: str) -> dict[str, Any]:
    latent_path = run_dir / "compressions" / f"{method}_latents.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing compression artifact: {latent_path}")
    payload = torch.load(latent_path, map_location="cpu")
    required = {"reconstructed", "cache_paths", "lengths", "shapes"}
    if not required <= set(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"{latent_path} is missing required fields: {missing}")
    return payload


def score_corruption_sensitivity(
    run_dir: Path,
    method: str,
    alphas: list[float],
    model_id: str | None = None,
    device_name: str = "auto",
    limit: int | None = None,
    max_new_tokens: int | None = None,
) -> CorruptionSensitivitySummary:
    records = [row for row in read_jsonl(run_dir / "records.jsonl") if row.get("cache_path")]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No cache-backed records found in {run_dir}")

    method = method.lower()
    payload = _load_reconstruction_payload(run_dir, method)
    cache_paths = list(payload["cache_paths"])
    lengths = list(payload["lengths"])
    shapes = list(payload["shapes"])
    reconstructed = payload["reconstructed"].float()

    chosen_model = model_id or records[0]["model_id"]
    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(chosen_model, device, local_files_only=True)

    summaries: list[CorruptionAlphaSummary] = []
    for alpha in alphas:
        correct_count = 0
        mse_values: list[float] = []
        output_lengths: list[int] = []
        wrong_task_ids: list[str] = []
        replay_errors = 0
        for record in records:
            cache_path = str(record["cache_path"])
            try:
                idx = cache_paths.index(cache_path)
            except ValueError as exc:
                raise ValueError(f"{cache_path} is not present in {run_dir / 'compressions' / f'{method}_latents.pt'}") from exc
            bundle = load_cache_bundle(Path(cache_path))
            original_vector = flatten_cache(bundle["cache"]).float()
            length = int(lengths[idx])
            reconstructed_vector = reconstructed[idx, :length].float()
            mixed_vector = original_vector + float(alpha) * (reconstructed_vector - original_vector)
            mse_values.append(float(torch.mean((mixed_vector - original_vector) ** 2).item()))
            mixed_cache = unflatten_cache(mixed_vector, shapes[idx])
            replay_error = False
            try:
                output = greedy_continue_from_loaded_bundle(
                    bundle=bundle,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=int(max_new_tokens or record.get("generated_tokens") or 32),
                    cache_override=mixed_cache,
                )
            except Exception:
                output = ""
                replay_error = True
                replay_errors += 1
            parsed, correct = verify_output(output, _record_to_example(record))
            del parsed
            is_correct = bool(correct) and not replay_error
            if is_correct:
                correct_count += 1
            else:
                wrong_task_ids.append(str(record["task_id"]))
            output_lengths.append(len(output.split()))
        summaries.append(
            CorruptionAlphaSummary(
                alpha=float(alpha),
                records=len(records),
                correct=correct_count,
                accuracy=correct_count / len(records),
                mean_mse_to_original=sum(mse_values) / len(mse_values),
                mean_output_words=sum(output_lengths) / len(output_lengths),
                wrong_task_ids=wrong_task_ids,
                replay_errors=replay_errors,
            )
        )

    result_path = run_dir / "compressions" / f"{method}_corruption_sensitivity.json"
    summary = CorruptionSensitivitySummary(
        run_id=run_dir.name,
        method=method,
        alphas=summaries,
        result_path=str(result_path),
    )
    write_json(result_path, summary.to_dict())
    return summary