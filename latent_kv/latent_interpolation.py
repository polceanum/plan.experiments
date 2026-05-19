"""Latent interpolation sweeps for temporal RAE cache plans."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch

from .benchmarks import verify_output
from .cache import cache_shapes, choose_device, load_cache_bundle, load_model_and_tokenizer, unflatten_cache
from .codec_validation import validate_cache_against_bundle
from .compressors import (
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
    candidate_quality_summary: dict[str, Any]
    artifacts: dict[str, str]
    reconstruction_scan_path: str | None = None
    eligible_endpoint_count: int | None = None
    require_convincing_reconstruction: bool = False


@dataclass(frozen=True)
class ReconstructionScanSummary:
    run_dir: str
    analysis_dir: str
    output_dir: str
    checkpoint_path: str
    checkpoint_epoch: int | None
    scanned: int
    solved_reconstructions: int
    convincing_reconstructions: int
    replay_failures: int
    max_new_tokens: int
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


def _candidate_pairs(
    latents: torch.Tensor,
    annotations: list[dict[str, Any]],
    same_category: bool,
    eligible_indices: set[int] | None = None,
) -> list[tuple[float, int, int]]:
    correct_indices = [idx for idx, row in enumerate(annotations) if bool(row.get("correct"))]
    if eligible_indices is not None:
        correct_indices = [idx for idx in correct_indices if idx in eligible_indices]
    if not correct_indices:
        return []
    index_tensor = torch.tensor(correct_indices, dtype=torch.long)
    normalized = torch.nn.functional.normalize(latents.float()[index_tensor], dim=-1)
    distances = 1.0 - (normalized @ normalized.T)
    candidates = []
    for pos, a_idx in enumerate(correct_indices):
        a_category = str(annotations[a_idx].get("primary_category"))
        for next_pos, b_idx in enumerate(correct_indices[pos + 1 :], start=pos + 1):
            b_category = str(annotations[b_idx].get("primary_category"))
            is_same = a_category == b_category
            if is_same != same_category:
                continue
            distance = float(distances[pos, next_pos].item())
            candidates.append((distance, a_idx, b_idx))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def _problem_text(prompt: str) -> str:
    marker = "Solve the math problem. Give the final numeric answer."
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _prompt_overlap(a_prompt: str, b_prompt: str) -> float:
    a_tokens = _token_set(_problem_text(a_prompt))
    b_tokens = _token_set(_problem_text(b_prompt))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _select_candidates(
    candidates: list[tuple[float, int, int]],
    annotations: list[dict[str, Any]],
    records: list[dict[str, Any]] | None,
    wanted: int,
    used: set[tuple[int, int]],
    selection: str,
    min_distance: float,
    max_distance: float | None,
    max_prompt_overlap: float,
) -> list[tuple[float, int, int]]:
    filtered = []
    for distance, a_idx, b_idx in candidates:
        key = (min(a_idx, b_idx), max(a_idx, b_idx))
        if key in used or distance < min_distance:
            continue
        if max_distance is not None and distance > max_distance:
            continue
        if records is not None:
            if str(records[a_idx].get("target")) == str(records[b_idx].get("target")):
                continue
            overlap = _prompt_overlap(str(records[a_idx].get("prompt") or ""), str(records[b_idx].get("prompt") or ""))
            if overlap > max_prompt_overlap:
                continue
        filtered.append((distance, a_idx, b_idx))
    if selection == "nearest":
        return filtered[:wanted]
    if selection != "spread":
        raise ValueError("selection must be nearest or spread")
    if len(filtered) <= wanted:
        return filtered
    positions = torch.linspace(0, len(filtered) - 1, steps=wanted).round().long().tolist()
    chosen = []
    seen_positions = set()
    for position in positions:
        if position in seen_positions:
            continue
        seen_positions.add(position)
        chosen.append(filtered[int(position)])
    cursor = 0
    while len(chosen) < wanted and cursor < len(filtered):
        if cursor not in seen_positions:
            chosen.append(filtered[cursor])
        cursor += 1
    chosen.sort(key=lambda item: (item[0], item[1], item[2]))
    return chosen[:wanted]


def select_interpolation_pairs(
    latents: torch.Tensor,
    annotations: list[dict[str, Any]],
    pairs: int,
    pair_mode: str = "mixed",
    records: list[dict[str, Any]] | None = None,
    selection: str = "nearest",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    max_prompt_overlap: float = 0.65,
    eligible_indices: set[int] | None = None,
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
        "same_category": _candidate_pairs(latents, annotations, same_category=True, eligible_indices=eligible_indices),
        "cross_category": _candidate_pairs(latents, annotations, same_category=False, eligible_indices=eligible_indices),
    }
    for mode, wanted in modes:
        selected_candidates = _select_candidates(
            candidate_cache[mode],
            annotations=annotations,
            records=records,
            wanted=wanted,
            used=used,
            selection=selection,
            min_distance=min_distance,
            max_distance=max_distance,
            max_prompt_overlap=max_prompt_overlap,
        )
        for distance, a_idx, b_idx in selected_candidates:
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


_FAITHFULNESS_STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "answer",
    "are",
    "assume",
    "based",
    "before",
    "calculate",
    "can",
    "case",
    "determine",
    "does",
    "each",
    "final",
    "find",
    "first",
    "for",
    "from",
    "get",
    "give",
    "given",
    "has",
    "have",
    "how",
    "into",
    "let",
    "math",
    "many",
    "more",
    "need",
    "next",
    "number",
    "out",
    "problem",
    "solve",
    "step",
    "than",
    "that",
    "the",
    "then",
    "there",
    "this",
    "total",
    "using",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _FAITHFULNESS_STOPWORDS
    }


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?(?:/\d+)?(?![a-z0-9])", text.lower()))


def reconstruction_faithfulness(prompt: str, decoded_output: str, decoded_correct: bool) -> dict[str, Any]:
    """Heuristic guard against numeric-answer-only reconstructions.

    This is deliberately conservative and local-only: a decoded continuation must be
    verifier-correct, reuse enough prompt-specific content words, and preserve some
    prompt numbers to be treated as an inspectable reconstructed plan.
    """

    problem = _problem_text(prompt)
    prompt_tokens = _content_tokens(problem)
    output_tokens = _content_tokens(decoded_output)
    prompt_numbers = _numeric_tokens(problem)
    output_numbers = _numeric_tokens(decoded_output)
    token_overlap = len(prompt_tokens & output_tokens)
    token_recall = token_overlap / max(len(prompt_tokens), 1)
    number_overlap = len(prompt_numbers & output_numbers)
    number_recall = 1.0 if not prompt_numbers else number_overlap / len(prompt_numbers)
    output_token_count = len(output_tokens)
    convincing = (
        bool(decoded_correct)
        and output_token_count >= 5
        and token_overlap >= 2
        and token_recall >= 0.18
        and number_recall >= 0.5
    )
    return {
        "convincing": convincing,
        "prompt_content_tokens": sorted(prompt_tokens),
        "decoded_content_tokens": sorted(output_tokens),
        "content_token_overlap": token_overlap,
        "content_token_recall": token_recall,
        "prompt_numbers": sorted(prompt_numbers),
        "decoded_numbers": sorted(output_numbers),
        "number_overlap": number_overlap,
        "number_recall": number_recall,
        "decoded_content_token_count": output_token_count,
        "rule": "decoded_correct && >=5 decoded content tokens && >=2 prompt-token overlap && >=0.18 token recall && >=0.5 number recall",
    }


def candidate_plan_quality(decoded_output: str, replay_error: str | None = None) -> dict[str, Any]:
    """Lightweight triage for interpolated continuations as generated plans.

    This intentionally does not compare against endpoint targets. Interpolation
    points may be valid generated reasoning traces even when they do not solve
    either endpoint task.
    """

    output = str(decoded_output or "")
    normalized = " ".join(output.split())
    lower = normalized.lower()
    word_count = len(normalized.split())
    has_number = bool(re.search(r"[-+]?\d+(?:\.\d+)?", normalized))
    has_equation = bool(re.search(r"\d+\s*[-+*/=]\s*\d+|\\frac|=", normalized))
    has_answer_marker = bool(re.search(r"\b(the answer is|final answer|therefore|boxed)\b", lower))
    has_reasoning_marker = bool(re.search(r"\b(step|first|second|then|so|because|therefore|let)\b", lower))
    appears_truncated = bool(re.search(r"\b(let'?s break|we need to determine|based on the information provided)\s*$", lower))
    has_placeholder_drift = bool(re.search(r"\bpens each person\b", lower))
    inspectable = replay_error is None and word_count >= 24 and (has_reasoning_marker or has_equation)
    potentially_solved = inspectable and has_number and has_answer_marker and not appears_truncated
    score = sum(
        [
            int(word_count >= 24),
            int(has_reasoning_marker),
            int(has_equation),
            int(has_number),
            int(has_answer_marker),
            int(not appears_truncated),
            int(not has_placeholder_drift),
        ]
    )
    return {
        "word_count": word_count,
        "has_number": has_number,
        "has_equation": has_equation,
        "has_answer_marker": has_answer_marker,
        "has_reasoning_marker": has_reasoning_marker,
        "appears_truncated": appears_truncated,
        "has_placeholder_drift": has_placeholder_drift,
        "inspectable": inspectable,
        "potentially_solved": potentially_solved,
        "score": score,
    }


def _load_reconstructed_correct_indices(path: Path, require_convincing: bool = False) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstruction scan artifact: {path}")
    rows = read_jsonl(path)
    indices = set()
    for row in rows:
        if not bool(row.get("decoded_correct")) or row.get("replay_error") not in {None, ""}:
            continue
        if require_convincing:
            convincing = row.get("decoded_convincing")
            if convincing is None:
                convincing = reconstruction_faithfulness(
                    prompt=str(row.get("prompt") or ""),
                    decoded_output=str(row.get("decoded_output") or ""),
                    decoded_correct=bool(row.get("decoded_correct")),
                )["convincing"]
            if not convincing:
                continue
        indices.add(int(row["index"]))
    if not indices:
        descriptor = "decoded-correct convincing" if require_convincing else "decoded-correct"
        raise ValueError(f"No {descriptor} reconstruction endpoints found in {path}")
    return indices


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


def _aligned_shapes_for_checkpoint(base_shapes: list[Any], seq_len: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    aligned = []
    for key_shape, value_shape in base_shapes:
        aligned_layer = []
        for shape in (key_shape, value_shape):
            shape_list = list(shape)
            shape_list[-2] = int(seq_len)
            aligned_layer.append(tuple(shape_list))
        aligned.append((aligned_layer[0], aligned_layer[1]))
    return aligned


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


def _excerpt(text: Any, limit: int = 420) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _pair_reconstructions_solve(pair: dict[str, Any], pair_replay_rows: list[dict[str, Any]]) -> bool:
    del pair
    a_rows = [row for row in pair_replay_rows if row["replay_context"] == "a"]
    b_rows = [row for row in pair_replay_rows if row["replay_context"] == "b"]
    if not a_rows or not b_rows:
        return False
    min_alpha = min(float(row["alpha"]) for row in a_rows)
    max_alpha = max(float(row["alpha"]) for row in b_rows)
    a_reconstruction = [row for row in a_rows if abs(float(row["alpha"]) - min_alpha) < 1e-9]
    b_reconstruction = [row for row in b_rows if abs(float(row["alpha"]) - max_alpha) < 1e-9]
    return bool(
        a_reconstruction
        and b_reconstruction
        and a_reconstruction[0].get("decoded_correct_for_endpoint")
        and b_reconstruction[0].get("decoded_correct_for_endpoint")
    )


def _render_inspection_report(
    pair_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    solved_reconstructions_only: bool = False,
) -> str:
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    chunks = [
        "# Latent Interpolation Inspection Report" if not solved_reconstructions_only else "# Solved-Reconstruction Interpolation Inspection Report",
        "",
        "This file is optimized for human inspection. Each table reads left-to-right as a bridge:",
        "",
        "Endpoint A solved plan -> decoded A reconstruction -> interpolation points -> decoded B reconstruction -> Endpoint B solved plan.",
        "",
        "The JSON blocks are stable machine-readable anchors.",
        "",
    ]
    rendered_pairs = 0
    for pair in pair_rows:
        pair_id = str(pair["pair_id"])
        pair_replay_rows = rows_by_pair[pair_id]
        if solved_reconstructions_only and not _pair_reconstructions_solve(pair, pair_replay_rows):
            continue
        rendered_pairs += 1
        metadata = {
            "pair_id": pair_id,
            "pair_type": pair["pair_type"],
            "distance": pair["distance"],
            "a_task_id": pair["a"]["task_id"],
            "b_task_id": pair["b"]["task_id"],
            "a_primary_category": pair["a"]["primary_category"],
            "b_primary_category": pair["b"]["primary_category"],
            "a_target": pair["a"]["target"],
            "b_target": pair["b"]["target"],
        }
        chunks.extend(
            [
                f"## {pair_id}: {pair['a']['task_id']} -> {pair['b']['task_id']}",
                "",
                "```json",
                json.dumps(metadata, sort_keys=True),
                "```",
                "",
                f"**A problem** ({pair['a']['primary_category']}, target `{pair['a']['target']}`): {_excerpt(_problem_text(pair['a']['prompt']), 520)}",
                "",
                f"**A original**: {_excerpt(pair['a']['output_text'], 520)}",
                "",
                f"**B problem** ({pair['b']['primary_category']}, target `{pair['b']['target']}`): {_excerpt(_problem_text(pair['b']['prompt']), 520)}",
                "",
                f"**B original**: {_excerpt(pair['b']['output_text'], 520)}",
                "",
                "| stage | alpha | replay context | target | parsed/correct | continuation |",
                "|---|---:|---|---|---|---|",
            ]
        )
        a_rows = {float(row["alpha"]): row for row in pair_replay_rows if row["replay_context"] == "a"}
        b_rows = {float(row["alpha"]): row for row in pair_replay_rows if row["replay_context"] == "b"}
        alphas = sorted(set(a_rows) | set(b_rows))

        def append_original_row(stage: str, endpoint: dict[str, Any], context: str) -> None:
            text = _excerpt(endpoint.get("output_text"), 300).replace("|", "\\|")
            chunks.append(
                f"| {stage} | - | {context} | `{endpoint.get('target')}` | parsed `{endpoint.get('parsed_answer')}` correct `{endpoint.get('correct')}` | {text} |"
            )

        def append_decoded_row(stage: str, row: dict[str, Any]) -> None:
            text = _excerpt(row.get("decoded_output"), 300).replace("|", "\\|")
            chunks.append(
                f"| {stage} | {float(row.get('alpha', 0.0)):.3f} | {row.get('replay_context')} | `{row.get('endpoint_target')}` | "
                f"parsed `{row.get('decoded_parsed_answer')}` correct `{row.get('decoded_correct_for_endpoint')}` | {text} |"
            )

        append_original_row("Endpoint A solved plan", pair["a"], "a/original")
        if alphas:
            first_alpha = alphas[0]
            last_alpha = alphas[-1]
            if first_alpha in a_rows:
                append_decoded_row("Decoded A reconstruction", a_rows[first_alpha])
            for alpha in alphas[1:-1]:
                if alpha in a_rows:
                    append_decoded_row("Interpolation from A prompt", a_rows[alpha])
                if alpha in b_rows:
                    append_decoded_row("Interpolation from B prompt", b_rows[alpha])
            if last_alpha in b_rows:
                append_decoded_row("Decoded B reconstruction", b_rows[last_alpha])
        append_original_row("Endpoint B solved plan", pair["b"], "b/original")
        chunks.append("")
    if solved_reconstructions_only and rendered_pairs == 0:
        chunks.extend(
            [
                "## No solved endpoint reconstructions found",
                "",
                "No pair in this sweep had both decoded endpoint reconstructions solve their original endpoint tasks.",
                "",
                "That means there is currently no trustworthy interpolation bridge to inspect first. Use the full inspection report for qualitative failure modes, or rerun this report on a later checkpoint after reconstruction/replay quality improves.",
                "",
            ]
        )
    return "\n".join(chunks)


def _render_candidate_plan_report(pair_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> str:
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    chunks = [
        "# Interpolated Candidate Plans",
        "",
        "This report treats interpolation rows as generated candidate reasoning traces.",
        "Endpoint correctness anchors the endpoints only; middle rows are judged by local coherence and completeness, not by whether they answer endpoint A or B.",
        "",
    ]
    for pair in pair_rows:
        pair_id = str(pair["pair_id"])
        chunks.extend(
            [
                f"## {pair_id}: {pair['a']['task_id']} -> {pair['b']['task_id']}",
                "",
                f"**A endpoint solved**: target `{pair['a']['target']}`, parsed `{pair['a']['parsed_answer']}`, category `{pair['a']['primary_category']}`",
                "",
                f"**A problem**: {_excerpt(_problem_text(pair['a']['prompt']), 420)}",
                "",
                f"**B endpoint solved**: target `{pair['b']['target']}`, parsed `{pair['b']['parsed_answer']}`, category `{pair['b']['primary_category']}`",
                "",
                f"**B problem**: {_excerpt(_problem_text(pair['b']['prompt']), 420)}",
                "",
                "| alpha | context | quality | endpoint target match | parsed | candidate continuation |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for row in sorted(rows_by_pair[pair_id], key=lambda item: (float(item["alpha"]), str(item["replay_context"]))):
            quality = row.get("candidate_plan_quality") or {}
            labels = []
            if quality.get("potentially_solved"):
                labels.append("potentially solved")
            if quality.get("inspectable"):
                labels.append("inspectable")
            if quality.get("appears_truncated"):
                labels.append("truncated")
            if quality.get("has_placeholder_drift"):
                labels.append("placeholder drift")
            if not labels:
                labels.append("weak")
            text = _excerpt(row.get("decoded_output"), 340).replace("|", "\\|")
            chunks.append(
                f"| {float(row.get('alpha', 0.0)):.3f} | {row.get('replay_context')} | "
                f"{', '.join(labels)}; score `{quality.get('score')}` | "
                f"`{row.get('decoded_correct_for_endpoint')}` | `{row.get('decoded_parsed_answer')}` | {text} |"
            )
        chunks.append("")
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


def _candidate_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualities = [row.get("candidate_plan_quality") or {} for row in rows]
    total = len(qualities)
    if total == 0:
        return {"rows": 0, "inspectable": 0, "potentially_solved": 0, "mean_score": 0.0}
    return {
        "rows": total,
        "inspectable": sum(1 for quality in qualities if quality.get("inspectable")),
        "potentially_solved": sum(1 for quality in qualities if quality.get("potentially_solved")),
        "truncated": sum(1 for quality in qualities if quality.get("appears_truncated")),
        "placeholder_drift": sum(1 for quality in qualities if quality.get("has_placeholder_drift")),
        "mean_score": sum(float(quality.get("score", 0.0)) for quality in qualities) / total,
    }


def _load_interpolation_inputs(
    run_dir: Path,
    analysis_dir: Path | None,
    checkpoint_path: Path | None,
) -> tuple[Path, Path, dict[str, Any], torch.Tensor, list[dict[str, Any]], list[dict[str, Any]], int | None]:
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
            "Run latent-analysis for the checkpoint you want to use, or pass that exact checkpoint."
        )
    checkpoint_epoch = int(checkpoint_meta.get("checkpoint_epoch")) if checkpoint_meta.get("checkpoint_epoch") is not None else None
    return analysis_dir, checkpoint_path, checkpoint_meta, latents, annotations, records, checkpoint_epoch


def run_reconstruction_scan(
    run_dir: Path,
    analysis_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    latent_device_name: str = "cpu",
    replay_device_name: str = "auto",
    model_id: str | None = None,
    max_new_tokens: int = 128,
    limit: int | None = None,
    progress_every: int = 25,
) -> ReconstructionScanSummary:
    analysis_dir, checkpoint_path, checkpoint_meta, latents, annotations, records, checkpoint_epoch = _load_interpolation_inputs(
        run_dir,
        analysis_dir,
        checkpoint_path,
    )
    output_dir = output_dir or analysis_dir / f"reconstruction_scan_epoch_{checkpoint_epoch or 'unknown'}"
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = [idx for idx, row in enumerate(annotations) if bool(row.get("correct"))]
    if limit is not None:
        indices = indices[: int(limit)]

    latent_device = choose_device(latent_device_name)
    model, checkpoint = _load_checkpoint_model(checkpoint_path)
    model.to(latent_device)
    model.eval()
    mean = checkpoint["normalization_mean"].to(latent_device)
    std = checkpoint["normalization_std"].to(latent_device)
    seq_len = int(checkpoint.get("seq_len") or checkpoint_meta.get("seq_len") or mean.shape[-2])

    chosen_model = model_id or str(records[0].get("model_id"))
    replay_device = choose_device(replay_device_name)
    replay_model, tokenizer = load_model_and_tokenizer(chosen_model, replay_device, local_files_only=True)

    rows: list[dict[str, Any]] = []
    total = len(indices)
    for scan_index, idx in enumerate(indices, start=1):
        record = records[idx]
        annotation = annotations[idx]
        bundle = load_cache_bundle(Path(str(record["cache_path"])))
        endpoint_shapes = bundle.get("shapes") or cache_shapes(bundle["cache"])
        aligned_shapes = _aligned_shapes_for_checkpoint(endpoint_shapes, seq_len)
        error = None
        validation_payload = None
        output = ""
        parsed = None
        correct = False
        faithfulness = None
        try:
            cache_override = _decode_latent_to_cache(
                z=latents[idx].to(latent_device),
                endpoint_shapes=endpoint_shapes,
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
                max_new_tokens=int(max_new_tokens),
                cache_override=cache_override,
            )
            parsed, correct = verify_output(output, _record_to_example(record))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        decoded_correct = bool(correct) and error is None
        faithfulness = reconstruction_faithfulness(
            prompt=str(record.get("prompt") or ""),
            decoded_output=output,
            decoded_correct=decoded_correct,
        )
        rows.append(
            {
                "index": idx,
                "task_id": record.get("task_id"),
                "prompt": record.get("prompt"),
                "target": record.get("target"),
                "original_output": record.get("output_text"),
                "original_parsed_answer": record.get("parsed_answer"),
                "original_correct": bool(record.get("correct")),
                "primary_category": annotation.get("primary_category"),
                "categories": annotation.get("categories"),
                "decoded_output": output,
                "decoded_parsed_answer": parsed,
                "decoded_correct": decoded_correct,
                "decoded_convincing": bool(faithfulness["convincing"]),
                "faithfulness": faithfulness,
                "cache_validation": validation_payload,
                "replay_error": error,
                "model_id": chosen_model,
                "max_new_tokens": int(max_new_tokens),
            }
        )
        if progress_every > 0 and (scan_index == 1 or scan_index == total or scan_index % progress_every == 0):
            print(
                f"[latent-reconstruction-scan] {scan_index}/{total} task={record.get('task_id')} correct={bool(correct) and error is None}",
                file=sys.stderr,
                flush=True,
            )

    rows_path = output_dir / "reconstruction_replays.jsonl"
    _jsonl_write(rows_path, rows)
    summary = ReconstructionScanSummary(
        run_dir=str(run_dir),
        analysis_dir=str(analysis_dir),
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        scanned=len(rows),
        solved_reconstructions=sum(1 for row in rows if row.get("decoded_correct")),
        convincing_reconstructions=sum(1 for row in rows if row.get("decoded_convincing")),
        replay_failures=sum(1 for row in rows if row.get("replay_error")),
        max_new_tokens=int(max_new_tokens),
        artifacts={"reconstruction_replays.jsonl": str(rows_path)},
    )
    write_json(output_dir / "reconstruction_scan_summary.json", asdict(summary))
    return summary


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
    selection: str = "nearest",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    max_prompt_overlap: float = 0.65,
    reconstruction_scan_path: Path | None = None,
    require_convincing_reconstruction: bool = False,
) -> InterpolationSummary:
    analysis_dir, checkpoint_path, checkpoint_meta, latents, annotations, records, checkpoint_epoch = _load_interpolation_inputs(
        run_dir,
        analysis_dir,
        checkpoint_path,
    )
    alphas = alphas or list(DEFAULT_ALPHAS)
    output_dir = output_dir or analysis_dir / f"interpolations_epoch_{checkpoint_epoch or 'unknown'}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_indices = (
        _load_reconstructed_correct_indices(
            reconstruction_scan_path,
            require_convincing=require_convincing_reconstruction,
        )
        if reconstruction_scan_path is not None
        else None
    )

    pairs_selected = select_interpolation_pairs(
        latents,
        annotations,
        pairs=pairs,
        pair_mode=pair_mode,
        records=records,
        selection=selection,
        min_distance=min_distance,
        max_distance=max_distance,
        max_prompt_overlap=max_prompt_overlap,
        eligible_indices=eligible_indices,
    )

    latent_device = choose_device(latent_device_name)
    model, checkpoint = _load_checkpoint_model(checkpoint_path)
    model.to(latent_device)
    model.eval()
    mean = checkpoint["normalization_mean"].to(latent_device)
    std = checkpoint["normalization_std"].to(latent_device)
    seq_len = int(checkpoint.get("seq_len") or checkpoint_meta.get("seq_len") or mean.shape[-2])

    selected_indices = sorted({idx for pair in pairs_selected for idx in (pair.a_index, pair.b_index)})
    shapes_by_index: dict[int, list[Any]] = {}
    for idx in selected_indices:
        bundle = load_cache_bundle(Path(str(records[idx]["cache_path"])))
        shapes_by_index[idx] = bundle.get("shapes") or cache_shapes(bundle["cache"])
    if not shapes_by_index:
        raise ValueError("No endpoint cache shapes found")
    aligned_shapes = _aligned_shapes_for_checkpoint(next(iter(shapes_by_index.values())), seq_len)

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
                bundle = load_cache_bundle(Path(str(endpoint_record["cache_path"])))
                cache_override = None
                validation_payload = None
                error = None
                output = ""
                parsed = None
                correct = False
                try:
                    cache_override = _decode_latent_to_cache(
                        z=z,
                        endpoint_shapes=shapes_by_index[endpoint_index],
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
                quality = candidate_plan_quality(output, replay_error=error)
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
                    "candidate_plan_quality": quality,
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
    inspection_path = output_dir / "interpolation_inspection.md"
    solved_inspection_path = output_dir / "interpolation_inspection_solved_reconstructions.md"
    candidate_plan_path = output_dir / "interpolation_candidate_plans.md"
    _jsonl_write(pairs_path, pair_rows)
    _jsonl_write(replays_path, replay_rows)
    sequence_path.write_text(_render_sequences(pair_rows, replay_rows), encoding="utf-8")
    inspection_path.write_text(_render_inspection_report(pair_rows, replay_rows), encoding="utf-8")
    solved_inspection_path.write_text(
        _render_inspection_report(pair_rows, replay_rows, solved_reconstructions_only=True),
        encoding="utf-8",
    )
    candidate_plan_path.write_text(_render_candidate_plan_report(pair_rows, replay_rows), encoding="utf-8")
    plot_path = _plot_interpolation_accuracy(replay_rows, output_dir)

    artifacts = {
        "interpolation_pairs.jsonl": str(pairs_path),
        "interpolation_replays.jsonl": str(replays_path),
        "interpolation_sequences.md": str(sequence_path),
        "interpolation_inspection.md": str(inspection_path),
        "interpolation_inspection_solved_reconstructions.md": str(solved_inspection_path),
        "interpolation_candidate_plans.md": str(candidate_plan_path),
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
        candidate_quality_summary=_candidate_quality_summary(replay_rows),
        artifacts=artifacts,
        reconstruction_scan_path=str(reconstruction_scan_path) if reconstruction_scan_path is not None else None,
        eligible_endpoint_count=len(eligible_indices) if eligible_indices is not None else None,
        require_convincing_reconstruction=bool(require_convincing_reconstruction),
    )
    write_json(output_dir / "interpolation_summary.json", asdict(summary))
    return summary
