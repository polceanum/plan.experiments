"""Latent interpolation sweeps for temporal RAE cache plans."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

from .benchmarks import verify_output
from .cache import cache_shapes, choose_device, load_cache_bundle, load_model_and_tokenizer, unflatten_cache
from .codec_validation import validate_cache_against_bundle
from .compressors import (
    _aligned_cache_shapes,
    _aligned_to_compact,
    _temporal_to_aligned_vector,
)
from .injection import greedy_continue_from_loaded_bundle
from .latent_analysis import _load_checkpoint_model, latest_complete_checkpoint
from .schemas import TaskExample, read_jsonl, write_json


DEFAULT_ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


@dataclass(frozen=True)
class InterpolationPair:
    pair_id: str
    pair_type: str
    a_index: int
    b_index: int
    a_task_id: str
    b_task_id: str
    a_primary_category: str
    b_primary_category: str
    distance: float


@dataclass(frozen=True)
class InterpolationSummary:
    run_dir: str
    analysis_dir: str
    output_dir: str
    checkpoint_path: str
    checkpoint_epoch: int | None
    pairs: int
    alphas: list[float]
    replay_rows: int
    replay_contexts: list[str]
    pair_type_counts: dict[str, int]
    replay_failures: int
    accuracy_by_context_alpha: dict[str, dict[str, float]]
    artifacts: dict[str, str]


def parse_alphas(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return list(DEFAULT_ALPHAS)
    alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required")
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    return alphas


def interpolate_latents(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return ((1.0 - float(alpha)) * a) + (float(alpha) * b)


def _candidate_pairs(latents: torch.Tensor, annotations: list[dict[str, Any]], same_category: bool) -> list[tuple[float, int, int]]:
    correct_indices = [idx for idx, row in enumerate(annotations) if bool(row.get("correct"))]
    normalized = torch.nn.functional.normalize(latents.float(), dim=-1)
    candidates = []
    for pos, a_idx in enumerate(correct_indices):
        a_category = str(annotations[a_idx].get("primary_category"))
        for b_idx in correct_indices[pos + 1 :]:
            b_category = str(annotations[b_idx].get("primary_category"))
            is_same = a_category == b_category
            if is_same != same_category:
                continue
            distance = float(1.0 - torch.dot(normalized[a_idx], normalized[b_idx]).item())
            candidates.append((distance, a_idx, b_idx))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def select_interpolation_pairs(
    latents: torch.Tensor,
    annotations: list[dict[str, Any]],
    pairs: int,
    pair_mode: str = "mixed",
) -> list[InterpolationPair]:
    pair_mode = pair_mode.lower()
    if pair_mode not in {"same_category", "cross_category", "mixed"}:
        raise ValueError("pair_mode must be same_category, cross_category, or mixed")
    if pairs <= 0:
        raise ValueError("pairs must be positive")

    modes: list[tuple[str, int]]
    if pair_mode == "same_category":
        modes = [("same_category", pairs)]
    elif pair_mode == "cross_category":
        modes = [("cross_category", pairs)]
    else:
        same_count = pairs // 2
        modes = [("same_category", same_count), ("cross_category", pairs - same_count)]

    selected: list[InterpolationPair] = []
    used: set[tuple[int, int]] = set()
    candidate_cache = {
        "same_category": _candidate_pairs(latents, annotations, same_category=True),
        "cross_category": _candidate_pairs(latents, annotations, same_category=False),
    }
    for mode, wanted in modes:
        for distance, a_idx, b_idx in candidate_cache[mode]:
            key = (min(a_idx, b_idx), max(a_idx, b_idx))
            if key in used:
                continue
            used.add(key)
            selected.append(
                InterpolationPair(
                    pair_id=f"pair_{len(selected):04d}",
                    pair_type=mode,
                    a_index=a_idx,
                    b_index=b_idx,
                    a_task_id=str(annotations[a_idx]["task_id"]),
                    b_task_id=str(annotations[b_idx]["task_id"]),
                    a_primary_category=str(annotations[a_idx]["primary_category"]),
                    b_primary_category=str(annotations[b_idx]["primary_category"]),
                    distance=distance,
                )
            )
            if sum(1 for pair in selected if pair.pair_type == mode) >= wanted:
                break
    if len(selected) < pairs:
        raise ValueError(f"Only found {len(selected)} pairs for mode {pair_mode}; requested {pairs}")
    return selected[:pairs]


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record_to_example(record: dict[str, Any]) -> TaskExample:
    return TaskExample(
        benchmark=str(record["benchmark"]),
        task_id=str(record["task_id"]),
        prompt=str(record["prompt"]),
        answer=str(record["target"]),
        metadata=record.get("metadata") or {},
    )


def _tensor_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return None


def _endpoint_metadata(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "prompt": record.get("prompt"),
        "target": record.get("target"),
        "parsed_answer": record.get("parsed_answer"),
        "correct": bool(record.get("correct")),
        "output_text": record.get("output_text"),
        "primary_category": annotation.get("primary_category"),
        "categories": annotation.get("categories"),
        "difficulty_proxy": annotation.get("difficulty_proxy"),
        "category_notes": annotation.get("category_notes"),
        "cache_path": record.get("cache_path"),
        "generated_tokens": record.get("generated_tokens"),
    }


def _decode_latent_to_cache(
    z: torch.Tensor,
    endpoint_shapes: list[Any],
    aligned_shapes: list[Any],
    model: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Any:
    with torch.no_grad():
        decoded = model.decode(z.reshape(1, -1)).squeeze(0).cpu()
    sequence = (decoded * std.squeeze(0).cpu()) + mean.squeeze(0).cpu()
    aligned_vector = _temporal_to_aligned_vector(sequence, aligned_shapes)
    compact_vector = _aligned_to_compact(aligned_vector, endpoint_shapes, aligned_shapes)
    return unflatten_cache(compact_vector, endpoint_shapes)


def _plot_interpolation_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> str | None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/latent_kv_matplotlib")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    by_key: dict[tuple[str, str, float], list[bool]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["pair_type"]), str(row["replay_context"]), float(row["alpha"]))].append(bool(row["decoded_correct_for_endpoint"]))
    if not by_key:
        return None
    path = output_dir / "interpolation_accuracy_by_alpha.png"
    plt.figure(figsize=(9, 5))
    for pair_type in sorted({key[0] for key in by_key}):
        for context in ["a", "b"]:
            points = []
            for alpha in sorted({key[2] for key in by_key if key[0] == pair_type and key[1] == context}):
                values = by_key[(pair_type, context, alpha)]
                points.append((alpha, sum(values) / len(values)))
            if points:
                xs, ys = zip(*points)
                plt.plot(xs, ys, marker="o", label=f"{pair_type}/{context}")
    plt.xlabel("alpha")
    plt.ylabel("endpoint-target correctness")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _render_sequences(pair_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> str:
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    chunks = ["# Latent Interpolation Sequences", ""]
    for pair in pair_rows:
        pair_id = str(pair["pair_id"])
        chunks.extend(
            [
                f"## {pair_id} ({pair['pair_type']}, distance={pair['distance']:.6g})",
                "",
                f"### Endpoint A: {pair['a']['task_id']} [{pair['a']['primary_category']}]",
                "",
                pair["a"]["prompt"],
                "",
                f"Target: `{pair['a']['target']}` | parsed: `{pair['a']['parsed_answer']}`",
                "",
                "Original output:",
                "",
                "```text",
                str(pair["a"]["output_text"] or "").strip(),
                "```",
                "",
                "A-context decoded path:",
                "",
            ]
        )
        a_rows = sorted([row for row in rows_by_pair[pair_id] if row["replay_context"] == "a"], key=lambda row: float(row["alpha"]))
        for row in a_rows:
            chunks.extend(
                [
                    f"- alpha={row['alpha']:.3f} correct={row['decoded_correct_for_endpoint']} parsed=`{row.get('decoded_parsed_answer')}`",
                    "",
                    "```text",
                    str(row.get("decoded_output") or "").strip(),
                    "```",
                    "",
                ]
            )
        chunks.extend(
            [
                f"### Endpoint B: {pair['b']['task_id']} [{pair['b']['primary_category']}]",
                "",
                pair["b"]["prompt"],
                "",
                f"Target: `{pair['b']['target']}` | parsed: `{pair['b']['parsed_answer']}`",
                "",
                "B-context decoded path:",
                "",
            ]
        )
        b_rows = sorted([row for row in rows_by_pair[pair_id] if row["replay_context"] == "b"], key=lambda row: float(row["alpha"]), reverse=True)
        for row in b_rows:
            chunks.extend(
                [
                    f"- alpha={row['alpha']:.3f} correct={row['decoded_correct_for_endpoint']} parsed=`{row.get('decoded_parsed_answer')}`",
                    "",
                    "```text",
                    str(row.get("decoded_output") or "").strip(),
                    "```",
                    "",
                ]
            )
        chunks.extend(
            [
                "B original output:",
                "",
                "```text",
                str(pair["b"]["output_text"] or "").strip(),
                "```",
                "",
            ]
        )
    return "\n".join(chunks)


def _accuracy_by_context_alpha(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        key = f"{row['pair_type']}|{row['replay_context']}|{float(row['alpha']):.6g}"
        grouped[key].append(bool(row["decoded_correct_for_endpoint"]))
    return {
        key: {"rows": len(values), "accuracy": sum(values) / len(values)}
        for key, values in sorted(grouped.items())
    }


def run_latent_interpolation(
    run_dir: Path,
    analysis_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    pairs: int = 50,
    alphas: list[float] | None = None,
    pair_mode: str = "mixed",
    latent_device_name: str = "cpu",
    replay_device_name: str = "auto",
    model_id: str | None = None,
    max_new_tokens: int | None = None,
    progress_every: int = 25,
) -> InterpolationSummary:
    analysis_dir = analysis_dir or run_dir / "analysis"
    latents_path = analysis_dir / "checkpoint_latents.pt"
    categories_path = analysis_dir / "task_categories.jsonl"
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing latent analysis artifact: {latents_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Missing category artifact: {categories_path}")

    latent_payload = torch.load(latents_path, map_location="cpu")
    latents = latent_payload["latents"].float()
    annotations = read_jsonl(categories_path)
    records = read_jsonl(run_dir / "records.jsonl")
    if len(records) != len(annotations) or len(records) != int(latents.shape[0]):
        raise ValueError("records, annotations, and latents must have matching row counts")

    checkpoint_path = checkpoint_path or latest_complete_checkpoint(run_dir, method="rae_temporal")
    checkpoint_meta = latent_payload.get("checkpoint_metadata") or {}
    if str(checkpoint_meta.get("checkpoint_path") or "") != str(checkpoint_path):
        raise ValueError(
            "checkpoint_latents.pt was produced with a different checkpoint. "
            "Run latent-analysis for the checkpoint you want to interpolate, or pass that exact checkpoint."
        )
    alphas = alphas or list(DEFAULT_ALPHAS)
    checkpoint_epoch = int(checkpoint_meta.get("checkpoint_epoch")) if checkpoint_meta.get("checkpoint_epoch") is not None else None
    output_dir = output_dir or analysis_dir / f"interpolations_epoch_{checkpoint_epoch or 'unknown'}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_selected = select_interpolation_pairs(latents, annotations, pairs=pairs, pair_mode=pair_mode)

    shapes = []
    bundles = []
    for record in records:
        bundle = load_cache_bundle(Path(str(record["cache_path"])))
        bundles.append(bundle)
        shapes.append(bundle.get("shapes") or cache_shapes(bundle["cache"]))
    aligned_shapes = _aligned_cache_shapes(shapes)

    latent_device = choose_device(latent_device_name)
    model, checkpoint = _load_checkpoint_model(checkpoint_path)
    model.to(latent_device)
    model.eval()
    mean = checkpoint["normalization_mean"].to(latent_device)
    std = checkpoint["normalization_std"].to(latent_device)

    chosen_model = model_id or str(records[0].get("model_id"))
    replay_device = choose_device(replay_device_name)
    replay_model, tokenizer = load_model_and_tokenizer(chosen_model, replay_device, local_files_only=True)

    pair_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    total_replays = len(pairs_selected) * len(alphas) * 2
    completed_replays = 0
    for pair_index, pair in enumerate(pairs_selected, start=1):
        a_record = records[pair.a_index]
        b_record = records[pair.b_index]
        a_annotation = annotations[pair.a_index]
        b_annotation = annotations[pair.b_index]
        pair_row = {
            **asdict(pair),
            "a": _endpoint_metadata(a_record, a_annotation),
            "b": _endpoint_metadata(b_record, b_annotation),
        }
        pair_rows.append(pair_row)
        for alpha in alphas:
            z = interpolate_latents(latents[pair.a_index], latents[pair.b_index], alpha).to(latent_device)
            for context, endpoint_index, endpoint_record, endpoint_annotation, other_record in [
                ("a", pair.a_index, a_record, a_annotation, b_record),
                ("b", pair.b_index, b_record, b_annotation, a_record),
            ]:
                bundle = bundles[endpoint_index]
                cache_override = None
                validation_payload = None
                error = None
                output = ""
                parsed = None
                correct = False
                try:
                    cache_override = _decode_latent_to_cache(
                        z=z,
                        endpoint_shapes=shapes[endpoint_index],
                        aligned_shapes=aligned_shapes,
                        model=model,
                        mean=mean,
                        std=std,
                    )
                    validation = validate_cache_against_bundle(cache_override, bundle)
                    validation_payload = asdict(validation)
                    if not validation.valid:
                        raise ValueError(";".join(validation.errors))
                    output = greedy_continue_from_loaded_bundle(
                        bundle=bundle,
                        model=replay_model,
                        tokenizer=tokenizer,
                        device=replay_device,
                        max_new_tokens=int(max_new_tokens or endpoint_record.get("generated_tokens") or 32),
                        cache_override=cache_override,
                    )
                    parsed, correct = verify_output(output, _record_to_example(endpoint_record))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                row = {
                    "pair_id": pair.pair_id,
                    "pair_type": pair.pair_type,
                    "alpha": float(alpha),
                    "replay_context": context,
                    "endpoint_task_id": endpoint_record.get("task_id"),
                    "other_task_id": other_record.get("task_id"),
                    "endpoint_prompt": endpoint_record.get("prompt"),
                    "endpoint_target": endpoint_record.get("target"),
                    "endpoint_original_output": endpoint_record.get("output_text"),
                    "endpoint_parsed_answer": endpoint_record.get("parsed_answer"),
                    "endpoint_correct": bool(endpoint_record.get("correct")),
                    "endpoint_primary_category": endpoint_annotation.get("primary_category"),
                    "endpoint_categories": endpoint_annotation.get("categories"),
                    "endpoint_generated_token_ids": _tensor_list(bundle.get("generation_token_ids")),
                    "decoded_output": output,
                    "decoded_parsed_answer": parsed,
                    "decoded_correct_for_endpoint": bool(correct) and error is None,
                    "cache_validation": validation_payload,
                    "replay_error": error,
                    "model_id": chosen_model,
                    "replay_max_new_tokens": int(max_new_tokens or endpoint_record.get("generated_tokens") or 32),
                    "a_task_id": pair.a_task_id,
                    "b_task_id": pair.b_task_id,
                    "a_primary_category": pair.a_primary_category,
                    "b_primary_category": pair.b_primary_category,
                    "latent_distance": pair.distance,
                    "pair_index": pair_index - 1,
                }
                replay_rows.append(row)
                completed_replays += 1
                if progress_every > 0 and (completed_replays == 1 or completed_replays == total_replays or completed_replays % progress_every == 0):
                    print(
                        f"[latent-interpolate] replay {completed_replays}/{total_replays} pair={pair.pair_id} alpha={alpha:.3f} context={context}",
                        file=sys.stderr,
                        flush=True,
                    )

    pairs_path = output_dir / "interpolation_pairs.jsonl"
    replays_path = output_dir / "interpolation_replays.jsonl"
    sequence_path = output_dir / "interpolation_sequences.md"
    _jsonl_write(pairs_path, pair_rows)
    _jsonl_write(replays_path, replay_rows)
    sequence_path.write_text(_render_sequences(pair_rows, replay_rows), encoding="utf-8")
    plot_path = _plot_interpolation_accuracy(replay_rows, output_dir)

    artifacts = {
        "interpolation_pairs.jsonl": str(pairs_path),
        "interpolation_replays.jsonl": str(replays_path),
        "interpolation_sequences.md": str(sequence_path),
    }
    if plot_path is not None:
        artifacts["interpolation_accuracy_by_alpha.png"] = plot_path
    summary = InterpolationSummary(
        run_dir=str(run_dir),
        analysis_dir=str(analysis_dir),
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        pairs=len(pair_rows),
        alphas=[float(alpha) for alpha in alphas],
        replay_rows=len(replay_rows),
        replay_contexts=["a", "b"],
        pair_type_counts=dict(sorted(Counter(row["pair_type"] for row in pair_rows).items())),
        replay_failures=sum(1 for row in replay_rows if row.get("replay_error")),
        accuracy_by_context_alpha=_accuracy_by_context_alpha(replay_rows),
        artifacts=artifacts,
    )
    write_json(output_dir / "interpolation_summary.json", asdict(summary))
    return summary
