"""Prompt-decoder dataset helpers for latent planning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .schemas import read_jsonl, write_json


@dataclass(frozen=True)
class PromptDecoderDatasetSummary:
    run_dir: str
    analysis_dir: str
    output_dir: str
    rows: int
    latent_dim: int
    artifacts: dict[str, str]


def export_prompt_decoder_dataset(
    run_dir: Path,
    analysis_dir: Path,
    output_dir: Path | None = None,
) -> PromptDecoderDatasetSummary:
    """Pair latent points with their source problem prompts.

    This is the first artifact needed for training a latent -> problem prompt
    decoder head. It deliberately saves prompt text rather than tokenizer-
    specific ids so different prompt decoders can choose their own tokenizer and
    sequence-length policy.
    """

    records = read_jsonl(run_dir / "records.jsonl")
    latent_payload = torch.load(analysis_dir / "checkpoint_latents.pt", map_location="cpu")
    latents = latent_payload["latents"].detach().cpu().float()
    annotations = latent_payload.get("annotations") or []
    if len(records) != int(latents.shape[0]):
        raise ValueError(f"Record/latent row mismatch: {len(records)} records vs {int(latents.shape[0])} latents")
    if annotations and len(annotations) != len(records):
        raise ValueError(f"Annotation/record row mismatch: {len(annotations)} annotations vs {len(records)} records")

    output_dir = output_dir or analysis_dir / "prompt_decoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        annotation = annotations[idx] if annotations else {}
        rows.append(
            {
                "index": idx,
                "task_id": record.get("task_id"),
                "prompt": record.get("prompt"),
                "target": record.get("target"),
                "correct": bool(record.get("correct")),
                "primary_category": annotation.get("primary_category"),
                "categories": annotation.get("categories"),
                "prompt_tokens": record.get("prompt_tokens"),
                "generated_tokens": record.get("generated_tokens"),
                "cache_mode": (record.get("metadata") or {}).get("cache_mode", "prompt"),
            }
        )

    jsonl_path = output_dir / "prompt_decoder_rows.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    pt_path = output_dir / "prompt_decoder_dataset.pt"
    torch.save(
        {
            "latents": latents,
            "rows": rows,
            "checkpoint_metadata": latent_payload.get("checkpoint_metadata"),
            "note": "Dataset for latent -> problem prompt decoder experiments.",
        },
        pt_path,
    )
    summary = PromptDecoderDatasetSummary(
        run_dir=str(run_dir),
        analysis_dir=str(analysis_dir),
        output_dir=str(output_dir),
        rows=len(rows),
        latent_dim=int(latents.shape[-1]) if latents.ndim == 2 else 0,
        artifacts={
            "prompt_decoder_rows.jsonl": str(jsonl_path),
            "prompt_decoder_dataset.pt": str(pt_path),
        },
    )
    write_json(output_dir / "prompt_decoder_summary.json", asdict(summary))
    return summary
