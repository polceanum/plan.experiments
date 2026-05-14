"""Behavioural evaluation for original and reconstructed KV cache baselines."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from .benchmarks import verify_output
from .cache import (
    choose_device,
    cache_num_bytes,
    load_cache_bundle,
    load_model_and_tokenizer,
    unflatten_cache,
)
from .codec_validation import validate_cache_against_bundle
from .injection import greedy_continue_from_loaded_bundle
from .metrics import aggregate_records, load_metric_payload, metric_to_dict, render_report
from .schemas import TaskExample, TrajectoryRecord, read_jsonl, to_json_line, write_json


def _record_to_example(record: dict[str, Any]) -> TaskExample:
    return TaskExample(
        benchmark=record["benchmark"],
        task_id=record["task_id"],
        prompt=record["prompt"],
        answer=record["target"],
        metadata=record.get("metadata", {}),
    )


def _write_behavior_records(run_dir: Path, baseline: str, rows: list[TrajectoryRecord]) -> Path:
    path = run_dir / "behavior" / f"{baseline}_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(to_json_line(row))
    return path


def _merge_metrics(run_dir: Path, baseline: str, rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    existing = load_metric_payload(run_dir)
    by_key = {
        (row["baseline"], row["benchmark"]): row
        for row in existing.get("baselines", [])
    }
    for metric in aggregate_records(rows, baseline=baseline):
        by_key[(metric.baseline, metric.benchmark)] = metric_to_dict(metric)
    merged_extra = existing.get("extra", {}) | extra
    payload = {"baselines": list(by_key.values()), "extra": merged_extra}
    write_json(run_dir / "metrics.json", payload)
    original_records = read_jsonl(run_dir / "records.jsonl")
    report = render_report(run_dir, original_records, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def _load_reconstructed_cache(
    run_dir: Path,
    method: str,
    cache_path: str,
) -> tuple[Any, float | None]:
    latent_path = run_dir / "compressions" / f"{method}_latents.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing compression artifact: {latent_path}")
    payload = torch.load(latent_path, map_location="cpu")
    if not {"reconstructed", "cache_paths", "lengths", "shapes"} <= set(payload):
        raise ValueError(
            f"{latent_path} is an older artifact. Re-run `latent-kv compress --method {method}`."
        )
    try:
        idx = payload["cache_paths"].index(cache_path)
    except ValueError as exc:
        raise ValueError(f"{cache_path} is not present in {latent_path}") from exc
    length = int(payload["lengths"][idx])
    vector = payload["reconstructed"][idx, :length]
    cache = unflatten_cache(vector, payload["shapes"][idx])
    original = load_cache_bundle(Path(cache_path))["cache"]
    original_vector = payload.get("original")
    del original_vector
    mse = float(torch.mean((vector - torch.cat([t.reshape(-1).float() for layer in original for t in layer])) ** 2).item())
    return cache, mse


def run_cache_behavioral_baseline(
    run_dir: Path,
    baseline: str,
    max_new_tokens: int | None = None,
    device_name: str = "auto",
    model_id: str | None = None,
) -> dict[str, Any]:
    """Score cache replay baselines using only local model weights.

    Supported baselines:
    - original_cache
    - random
    - pca_svd
    - autoencoder
    - retrieval
    - rae_lstm
    """

    records = [row for row in read_jsonl(run_dir / "records.jsonl") if row.get("cache_path")]
    if not records:
        raise ValueError(f"No cache-backed records found in {run_dir}")

    baseline = baseline.lower()
    reconstructed_methods = {"random", "pca_svd", "autoencoder", "rae_lstm", "retrieval"}
    if baseline not in reconstructed_methods | {"original_cache"}:
        raise ValueError(f"Unsupported behavioural baseline: {baseline}")

    chosen_model = model_id or records[0]["model_id"]
    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(chosen_model, device, local_files_only=True)
    out_rows: list[TrajectoryRecord] = []
    reconstruction_mses: list[float] = []
    replay_token_budgets: list[int] = []

    for idx, record in enumerate(records):
        cache_path = str(record["cache_path"])
        bundle = load_cache_bundle(Path(cache_path))
        cache_override = None
        if baseline in reconstructed_methods:
            cache_override, mse = _load_reconstructed_cache(run_dir, baseline, cache_path)
            if mse is not None:
                reconstruction_mses.append(mse)
        validation_error = None
        if cache_override is not None:
            validation = validate_cache_against_bundle(cache_override, bundle)
            if not validation.valid:
                validation_error = ";".join(validation.errors)

        start = time.perf_counter()
        replay_max_new_tokens = int(max_new_tokens or record.get("generated_tokens") or 32)
        replay_token_budgets.append(replay_max_new_tokens)
        try:
            if validation_error is not None:
                raise ValueError(f"Decoded cache validation failed: {validation_error}")
            output = greedy_continue_from_loaded_bundle(
                bundle=bundle,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=replay_max_new_tokens,
                cache_override=cache_override,
            )
            error = None
        except Exception as exc:
            output = ""
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - start
        example = _record_to_example(record)
        parsed, correct = verify_output(output, example)
        out_rows.append(
            TrajectoryRecord(
                run_id=record["run_id"],
                benchmark=record["benchmark"],
                task_id=record["task_id"],
                model_id=chosen_model,
                seed=int(record.get("seed", 0)),
                attempt_id=idx,
                prompt=record["prompt"],
                target=record["target"],
                output_text=output,
                parsed_answer=parsed,
                correct=correct and error is None,
                retry_index=int(record.get("retry_index", 0)),
                cache_path=cache_path,
                hidden_path=record.get("hidden_path"),
                latency_s=latency,
                generated_tokens=len(output.split()),
                prompt_tokens=record.get("prompt_tokens"),
                memory_bytes=cache_num_bytes(cache_override or bundle["cache"]),
                metadata=(record.get("metadata") or {})
                | {
                    "behavioral_baseline": baseline,
                    "replay_error": error,
                    "replay_max_new_tokens": replay_max_new_tokens,
                    "source_generated_tokens": record.get("generated_tokens"),
                    "codec_validation_error": validation_error,
                    "local_files_only": True,
                },
            )
        )

    record_path = _write_behavior_records(run_dir, baseline, out_rows)
    row_dicts = [row.__dict__ for row in out_rows]
    extra = {
        f"behavior_{baseline}_records": str(record_path),
        f"behavior_{baseline}_local_model": chosen_model,
        f"behavior_{baseline}_max_new_tokens": max_new_tokens,
        f"behavior_{baseline}_replay_budget_source": "cli" if max_new_tokens is not None else "record_generated_tokens",
    }
    if replay_token_budgets:
        extra[f"behavior_{baseline}_mean_replay_max_new_tokens"] = sum(replay_token_budgets) / len(replay_token_budgets)
    if reconstruction_mses:
        extra[f"behavior_{baseline}_mean_reconstruction_mse"] = sum(reconstruction_mses) / len(reconstruction_mses)
    return _merge_metrics(run_dir, baseline, row_dicts, extra)

