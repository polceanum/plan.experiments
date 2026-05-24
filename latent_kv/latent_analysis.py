"""Latent trajectory annotation and PCA plotting utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch

from .cache import cache_shapes, load_cache_bundle
from .compressors import (
    TemporalLSTMAutoEncoder,
    _temporal_model_class,
    _aligned_cache_shapes,
    _flatten_to_shapes,
    _temporal_matrix,
    _temporal_token_mask,
)
from .schemas import read_jsonl, write_json


TAXONOMY = [
    "arithmetic_add_subtract",
    "arithmetic_multiply_divide",
    "fractions_ratios_percents",
    "linear_equation",
    "rate_time_work",
    "money_price_profit",
    "measurement_units",
    "counting_combinatorics",
    "comparison_relative",
    "multi_step_chain",
]

PRIMARY_ORDER = [
    "linear_equation",
    "rate_time_work",
    "money_price_profit",
    "fractions_ratios_percents",
    "arithmetic_multiply_divide",
    "arithmetic_add_subtract",
    "measurement_units",
    "comparison_relative",
    "counting_combinatorics",
    "multi_step_chain",
]


@dataclass(frozen=True)
class CategoryAnnotation:
    task_id: str
    correct: bool
    target: str | None
    parsed_answer: str | None
    prompt: str
    categories: list[str]
    primary_category: str
    difficulty_proxy: str
    category_notes: str


@dataclass(frozen=True)
class LatentAnalysisSummary:
    run_dir: str
    output_dir: str
    records: int
    checkpoint_path: str
    checkpoint_epoch: int | None
    latent_dim: int
    effective_latent_dim: int
    temporal_latent_tokens: int | None
    temporal_flatten_latent_tokens: bool
    temporal_decoder_memory_tokens: int | None
    pca_explained_variance_ratio: list[float]
    category_counts: dict[str, int]
    primary_category_counts: dict[str, int]
    correctness_counts: dict[str, int]
    artifacts: dict[str, str]


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _has_regex(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _number_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|half|quarter)\b", text))


def categorize_prompt(prompt: str) -> tuple[list[str], str, str, str]:
    """Assign exploratory multi-label GSM8K categories from prompt semantics."""

    text = prompt.lower()
    labels: set[str] = set()
    notes: list[str] = []

    if _has_any(text, ["total", "altogether", "sum", "left", "remain", "remaining", "difference", "more than", "less than", "gave away", "lost", "bought", "sold"]):
        labels.add("arithmetic_add_subtract")
        notes.append("contains total/remainder/difference language")
    if _has_any(text, ["each", "per ", "times", "twice", "triple", "dozen", "groups", "rows", "packs", "boxes", "share", "split", "divide", "equal"]):
        labels.add("arithmetic_multiply_divide")
        notes.append("contains repeated-group or per-unit language")
    if _has_any(text, ["half", "quarter", "third", "fraction", "percent", "%", "ratio", "proportion", "double", "twice as", "one-half"]):
        labels.add("fractions_ratios_percents")
        notes.append("contains fraction/ratio/percent language")
    if _has_regex(text, [r"\bhow many .* originally\b", r"\bhow many .* at first\b", r"\bunknown\b", r"\bif .* then .* how many\b", r"\bthe rest\b", r"\bremainder\b"]):
        labels.add("linear_equation")
        notes.append("asks for an implied unknown quantity")
    if _has_any(
        text,
        [
            "per day",
            "per hour",
            "per minute",
            " a day",
            " a week",
            " a month",
            " a year",
            "each day",
            "each week",
            "every day",
            "every week",
            "every year",
            "mph",
            "speed",
            "rate",
            "daily",
            "weekly",
            "work",
            "schedule",
            "minutes",
            "hours",
        ],
    ):
        labels.add("rate_time_work")
        notes.append("contains rate, time, schedule, or productivity terms")
    if _has_any(text, ["$", "dollar", "cent", "cost", "price", "paid", "profit", "revenue", "earn", "salary", "tax", "sale"]):
        labels.add("money_price_profit")
        notes.append("contains money, price, payment, or profit terms")
    if _has_any(text, ["mile", "meter", "feet", "foot", "inch", "pound", "ounce", "liter", "gallon", "calorie", "page", "minutes", "hours", "days", "weeks", "years"]):
        labels.add("measurement_units")
        notes.append("uses measurement or time units")
    if _has_any(text, ["ways", "arrange", "combination", "possible", "tickets", "students", "people", "children", "objects", "items", "books", "cards"]):
        labels.add("counting_combinatorics")
        notes.append("centers on counting discrete objects/events")
    if _has_any(text, ["more than", "less than", "how much more", "how many more", "fewer", "older", "younger", "twice", "half as", "as many", "compared", "difference"]):
        labels.add("comparison_relative")
        notes.append("contains comparative or relative quantity language")

    step_markers = len(re.findall(r"\b(then|after|before|next|each|per|total|left|remaining|more than|less than)\b", text))
    if step_markers >= 4 or _number_count(text) >= 5:
        labels.add("multi_step_chain")
        notes.append("has several numeric dependencies or sequencing markers")

    if not labels:
        labels.add("counting_combinatorics")
        notes.append("fallback: generic discrete word problem")

    if "multi_step_chain" in labels or step_markers >= 4 or _number_count(text) >= 5 or len(labels) >= 4:
        difficulty = "multi_step"
    elif step_markers >= 2 or _number_count(text) >= 3:
        difficulty = "two_step"
    else:
        difficulty = "single_step"

    primary = next(label for label in PRIMARY_ORDER if label in labels)
    ordered = [label for label in TAXONOMY if label in labels]
    return ordered, primary, difficulty, "; ".join(notes[:3])


def annotate_records(records: list[dict[str, Any]]) -> list[CategoryAnnotation]:
    annotations = []
    for record in records:
        categories, primary, difficulty, notes = categorize_prompt(str(record.get("prompt") or ""))
        annotations.append(
            CategoryAnnotation(
                task_id=str(record.get("task_id")),
                correct=bool(record.get("correct")),
                target=record.get("target"),
                parsed_answer=record.get("parsed_answer"),
                prompt=str(record.get("prompt") or ""),
                categories=categories,
                primary_category=primary,
                difficulty_proxy=difficulty,
                category_notes=notes,
            )
        )
    return annotations


def _write_annotations(path: Path, annotations: list[CategoryAnnotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(json.dumps(asdict(annotation), sort_keys=True) + "\n")


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def latest_complete_checkpoint(run_dir: Path, method: str = "rae_temporal") -> Path:
    checkpoint_dir = run_dir / "compressions" / f"{method}_checkpoints"
    candidates = sorted(checkpoint_dir.glob(f"{method}_epoch_*.pt"), key=lambda path: _checkpoint_epoch(path) or -1)
    if candidates:
        return candidates[-1]
    latest = checkpoint_dir / f"{method}_latest.pt"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No {method} checkpoint found under {checkpoint_dir}")


def _load_checkpoint_model(checkpoint_path: Path) -> tuple[TemporalLSTMAutoEncoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    codec_kind = str(checkpoint.get("temporal_codec_kind") or checkpoint.get("codec_kind") or "lstm")
    model_class = _temporal_model_class(codec_kind)
    model_kwargs = {
        "token_dim": int(checkpoint["token_dim"]),
        "max_tokens": int(checkpoint["seq_len"]),
        "latent_dim": int(checkpoint["latent_dim"]),
        "hidden_dim": int(checkpoint.get("hidden_dim", 128)),
        "num_layers": int(checkpoint.get("num_layers", 1)),
    }
    if str(checkpoint.get("temporal_codec_kind") or checkpoint.get("codec_kind") or "").lower() in {
        "chunked",
        "temporal_chunked_rae",
    }:
        model_kwargs["chunk_size"] = int(checkpoint.get("chunk_size") or 16)
    if str(checkpoint.get("temporal_codec_kind") or checkpoint.get("codec_kind") or "").lower() in {
        "transformer",
        "temporal_transformer_rae",
    }:
        model_kwargs["num_heads"] = int(checkpoint.get("temporal_num_heads") or checkpoint.get("num_heads") or 8)
        model_kwargs["latent_tokens"] = int(checkpoint.get("temporal_latent_tokens") or checkpoint.get("latent_tokens") or 1)
        model_kwargs["decoder_memory_tokens"] = int(
            checkpoint.get("temporal_decoder_memory_tokens") or checkpoint.get("decoder_memory_tokens") or 1
        )
        model_kwargs["flatten_latent_tokens"] = bool(checkpoint.get("temporal_flatten_latent_tokens"))
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def extract_checkpoint_latents(
    run_dir: Path,
    checkpoint_path: Path,
    annotations: list[dict[str, Any]],
    batch_size: int = 8,
    progress_every_batches: int = 50,
) -> tuple[torch.Tensor, dict[str, Any]]:
    records = read_jsonl(run_dir / "records.jsonl")
    if len(records) != len(annotations):
        raise ValueError(f"Record/category row mismatch: {len(records)} records vs {len(annotations)} annotations")

    shapes = []
    cache_paths = []
    for record in records:
        cache_path = record.get("cache_path")
        if not cache_path:
            raise ValueError(f"Record {record.get('task_id')} has no cache_path")
        bundle = load_cache_bundle(Path(cache_path))
        shapes.append(bundle.get("shapes") or cache_shapes(bundle["cache"]))
        cache_paths.append(str(cache_path))

    aligned_shapes = _aligned_cache_shapes(shapes)
    token_mask = _temporal_token_mask(shapes, aligned_shapes)

    model, checkpoint = _load_checkpoint_model(checkpoint_path)
    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]
    token_dim = int(checkpoint["token_dim"])
    if tuple(mean.shape[-2:]) != (1, token_dim) and tuple(mean.shape) != (1, 1, token_dim):
        raise ValueError(f"Checkpoint normalization mean shape {tuple(mean.shape)} does not match token_dim {token_dim}")

    latents = []
    batch_size = max(1, int(batch_size))
    total_batches = math.ceil(len(records) / batch_size)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
            stop = min(start + batch_size, len(records))
            batch_caches = [load_cache_bundle(Path(cache_path))["cache"] for cache_path in cache_paths[start:stop]]
            batch_vectors = torch.stack([_flatten_to_shapes(cache, aligned_shapes) for cache in batch_caches]).float()
            batch_temporal = _temporal_matrix(batch_vectors, aligned_shapes)
            batch_mask = token_mask[start:stop]
            normalized = ((batch_temporal - mean) / std) * batch_mask.unsqueeze(-1).float()
            batch_latents = model.encode(normalized, token_mask=batch_mask).cpu()
            latents.append(batch_latents)
            if progress_every_batches > 0 and (batch_index == 1 or batch_index == total_batches or batch_index % progress_every_batches == 0):
                print(
                    f"[latent-analysis] encoded batch {batch_index}/{total_batches} records={stop}/{len(records)}",
                    file=sys.stderr,
                    flush=True,
                )

    all_latents = torch.cat(latents, dim=0)
    temporal_latent_tokens = checkpoint.get("temporal_latent_tokens") or checkpoint.get("latent_tokens")
    temporal_decoder_memory_tokens = checkpoint.get("temporal_decoder_memory_tokens") or checkpoint.get("decoder_memory_tokens")
    flatten_latent_tokens = bool(checkpoint.get("temporal_flatten_latent_tokens"))
    effective_latent_dim = int(
        checkpoint.get("effective_latent_dim")
        or (int(checkpoint["latent_dim"]) * int(temporal_latent_tokens or 1) if flatten_latent_tokens else all_latents.reshape(all_latents.shape[0], -1).shape[-1])
    )
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch")) if checkpoint.get("epoch") is not None else _checkpoint_epoch(checkpoint_path),
        "latent_dim": int(checkpoint["latent_dim"]),
        "effective_latent_dim": effective_latent_dim,
        "hidden_dim": int(checkpoint.get("hidden_dim", model.hidden_dim)),
        "num_layers": int(checkpoint.get("num_layers", model.num_layers)),
        "seq_len": int(checkpoint["seq_len"]),
        "token_dim": int(checkpoint["token_dim"]),
        "temporal_latent_tokens": int(temporal_latent_tokens) if temporal_latent_tokens is not None else None,
        "temporal_flatten_latent_tokens": flatten_latent_tokens,
        "temporal_decoder_memory_tokens": int(temporal_decoder_memory_tokens) if temporal_decoder_memory_tokens is not None else None,
        "latent_shape": list(all_latents.shape),
        "cache_paths": cache_paths,
    }
    return all_latents, metadata


def pca_2d(latents: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    x = latents.detach().cpu().float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] == 1:
        return torch.zeros((1, 2), dtype=x.dtype), [0.0, 0.0]
    centered = x - x.mean(dim=0, keepdim=True)
    if centered.shape[0] < 1:
        raise ValueError("PCA requires at least one latent row")
    _, singular_values, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    variances = singular_values[:2].pow(2) / max(centered.shape[0] - 1, 1)
    total_variance = centered.var(dim=0, unbiased=True).sum().clamp_min(1e-12)
    ratio = (variances / total_variance).tolist()
    if not torch.isfinite(coords).all():
        raise ValueError("PCA produced non-finite coordinates")
    return coords, [float(value) for value in ratio]


def _write_pca_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "pc1",
        "pc2",
        "correct",
        "primary_category",
        "categories",
        "difficulty_proxy",
        "target",
        "parsed_answer",
        "checkpoint_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/latent_kv_matplotlib")
    import matplotlib.pyplot as plt

    return plt


def plot_pca(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    plt = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    correctness_path = output_dir / "latent_pca_by_correctness.png"
    category_path = output_dir / "latent_pca_by_primary_category.png"
    facets_path = output_dir / "latent_pca_facets.png"

    xs = [float(row["pc1"]) for row in rows]
    ys = [float(row["pc2"]) for row in rows]
    correct = [bool(row["correct"]) for row in rows]

    plt.figure(figsize=(8, 6))
    for wanted, color, label in [(True, "#2ca25f", "correct"), (False, "#de2d26", "incorrect")]:
        idx = [i for i, value in enumerate(correct) if value is wanted]
        plt.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=16, c=color, alpha=0.75, label=label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(correctness_path, dpi=180)
    plt.close()

    categories = sorted({str(row["primary_category"]) for row in rows})
    cmap = plt.get_cmap("tab20")
    color_by_category = {category: cmap(index % 20) for index, category in enumerate(categories)}
    plt.figure(figsize=(9, 7))
    for category in categories:
        cat_rows = [i for i, row in enumerate(rows) if row["primary_category"] == category]
        filled = [i for i in cat_rows if correct[i]]
        failed = [i for i in cat_rows if not correct[i]]
        color = color_by_category[category]
        if filled:
            plt.scatter([xs[i] for i in filled], [ys[i] for i in filled], s=14, c=[color], alpha=0.75, label=category)
        if failed:
            plt.scatter([xs[i] for i in failed], [ys[i] for i in failed], s=22, facecolors="none", edgecolors=[color], linewidths=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(category_path, dpi=180)
    plt.close()

    facet_categories = [category for category, _ in Counter(str(row["primary_category"]) for row in rows).most_common(9)]
    cols = 3
    rows_count = int(math.ceil(len(facet_categories) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(12, max(4, 3.4 * rows_count)), squeeze=False)
    for axis in axes.reshape(-1):
        axis.axis("off")
    for axis, category in zip(axes.reshape(-1), facet_categories):
        axis.axis("on")
        axis.scatter(xs, ys, s=7, c="#d9d9d9", alpha=0.45)
        idx = [i for i, row in enumerate(rows) if row["primary_category"] == category]
        axis.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=14, c=[color_by_category[category]], alpha=0.85)
        axis.set_title(category, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.savefig(facets_path, dpi=180)
    plt.close(fig)

    return {
        "latent_pca_by_correctness.png": str(correctness_path),
        "latent_pca_by_primary_category.png": str(category_path),
        "latent_pca_facets.png": str(facets_path),
    }


def run_latent_analysis(
    run_dir: Path,
    method: str = "rae_temporal",
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = 8,
    progress_every_batches: int = 50,
) -> LatentAnalysisSummary:
    records = read_jsonl(run_dir / "records.jsonl")
    if not records:
        raise ValueError(f"No records found under {run_dir}")

    output_dir = output_dir or run_dir / "analysis"
    checkpoint_path = checkpoint_path or latest_complete_checkpoint(run_dir, method=method)
    annotations = annotate_records(records)
    category_path = output_dir / "task_categories.jsonl"
    _write_annotations(category_path, annotations)
    annotation_rows = _read_annotations(category_path)

    latents, metadata = extract_checkpoint_latents(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        annotations=annotation_rows,
        batch_size=batch_size,
        progress_every_batches=progress_every_batches,
    )
    coords, variance_ratio = pca_2d(latents)
    checkpoint_epoch = metadata["checkpoint_epoch"]

    pca_rows: list[dict[str, Any]] = []
    for idx, (annotation, coord) in enumerate(zip(annotation_rows, coords, strict=True)):
        pca_rows.append(
            {
                "task_id": annotation["task_id"],
                "pc1": float(coord[0].item()),
                "pc2": float(coord[1].item()),
                "correct": bool(annotation["correct"]),
                "primary_category": annotation["primary_category"],
                "categories": "|".join(annotation["categories"]),
                "difficulty_proxy": annotation["difficulty_proxy"],
                "target": annotation.get("target"),
                "parsed_answer": annotation.get("parsed_answer"),
                "checkpoint_epoch": checkpoint_epoch,
            }
        )

    if len(records) != len(annotation_rows) or len(records) != int(latents.shape[0]) or len(records) != len(pca_rows):
        raise ValueError("Analysis row counts do not match")
    if not torch.isfinite(latents).all() or not torch.isfinite(coords).all():
        raise ValueError("Analysis contains non-finite latent/PCA values")
    for annotation in annotation_rows:
        if not annotation["categories"] or not annotation["primary_category"]:
            raise ValueError(f"Missing category assignment for {annotation['task_id']}")

    latent_path = output_dir / "checkpoint_latents.pt"
    torch.save(
        {
            "latents": latents,
            "annotations": annotation_rows,
            "checkpoint_metadata": metadata,
            "taxonomy": TAXONOMY,
        },
        latent_path,
    )
    pca_csv = output_dir / "latent_pca.csv"
    _write_pca_csv(pca_csv, pca_rows)
    plot_paths = plot_pca(pca_rows, output_dir)

    category_counts: Counter[str] = Counter()
    for annotation in annotation_rows:
        category_counts.update(annotation["categories"])
    primary_counts = Counter(str(annotation["primary_category"]) for annotation in annotation_rows)
    correctness_counts = Counter("correct" if bool(row["correct"]) else "incorrect" for row in annotation_rows)
    artifacts = {
        "task_categories.jsonl": str(category_path),
        "checkpoint_latents.pt": str(latent_path),
        "latent_pca.csv": str(pca_csv),
        **plot_paths,
    }
    summary = LatentAnalysisSummary(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        records=len(records),
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch,
        latent_dim=int(metadata["latent_dim"]),
        effective_latent_dim=int(metadata["effective_latent_dim"]),
        temporal_latent_tokens=metadata["temporal_latent_tokens"],
        temporal_flatten_latent_tokens=bool(metadata["temporal_flatten_latent_tokens"]),
        temporal_decoder_memory_tokens=metadata["temporal_decoder_memory_tokens"],
        pca_explained_variance_ratio=variance_ratio,
        category_counts=dict(sorted(category_counts.items())),
        primary_category_counts=dict(sorted(primary_counts.items())),
        correctness_counts=dict(sorted(correctness_counts.items())),
        artifacts=artifacts,
    )
    write_json(output_dir / "analysis_summary.json", asdict(summary))
    return summary
