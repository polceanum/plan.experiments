"""Prompt-decoder dataset helpers for latent planning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cache import load_cache_bundle
from .schemas import read_jsonl, write_json


@dataclass(frozen=True)
class PromptDecoderDatasetSummary:
    run_dir: str
    analysis_dir: str
    output_dir: str
    rows: int
    latent_dim: int
    latent_shape: list[int]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class PromptDecoderTrainSummary:
    dataset_path: str
    output_dir: str
    rows: int
    epochs: int
    latent_shape: list[int]
    vocab_size: int
    original_token_vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_prompt_tokens: int
    max_latent_chunks: int | None
    progress_every_batches: int
    final_loss: float
    final_token_accuracy: float
    exact_token_match: float
    artifacts: dict[str, str]


@dataclass(frozen=True)
class PromptInterpolationDecodeSummary:
    prompt_decoder_checkpoint: str
    interpolation_dir: str
    output_dir: str
    rows: int
    checkpoint_epoch: int | None
    artifacts: dict[str, str]


class LatentPromptTokenDecoder(nn.Module):
    """Decode structured latents into prompt token IDs.

    This mirrors the structured cache decoder style: latent chunks are projected
    as memory, learned prompt-position queries attend to that memory, and a
    token classifier predicts the original prompt sequence.
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        max_prompt_tokens: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_prompt_tokens = max_prompt_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_positions = nn.Parameter(torch.randn(max_prompt_tokens, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, latents: torch.Tensor, prompt_length: int) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        if latents.ndim != 3:
            raise ValueError(f"Expected [batch, chunks, latent_dim] or [batch, latent_dim], got {tuple(latents.shape)}")
        if prompt_length > self.max_prompt_tokens:
            raise ValueError(f"prompt_length {prompt_length} exceeds max_prompt_tokens {self.max_prompt_tokens}")
        memory = self.latent_projection(latents)
        queries = self.prompt_positions[:prompt_length].unsqueeze(0).expand(latents.shape[0], prompt_length, self.hidden_dim)
        hidden = self.decoder(tgt=queries, memory=memory)
        return self.output(hidden)


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
        prompt_token_ids = None
        cache_path = record.get("cache_path")
        if cache_path:
            bundle = load_cache_bundle(Path(str(cache_path)))
            input_ids = bundle.get("input_ids")
            generation_config = bundle.get("generation_config") or {}
            prompt_tokens = generation_config.get("prompt_tokens") or record.get("prompt_tokens")
            if input_ids is not None and prompt_tokens is not None:
                prompt_token_ids = input_ids.reshape(-1)[: int(prompt_tokens)].tolist()
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
                "prompt_token_ids": prompt_token_ids,
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
        latent_dim=int(latents.shape[-1]),
        latent_shape=list(latents.shape),
        artifacts={
            "prompt_decoder_rows.jsonl": str(jsonl_path),
            "prompt_decoder_dataset.pt": str(pt_path),
        },
    )
    write_json(output_dir / "prompt_decoder_summary.json", asdict(summary))
    return summary


def _prompt_token_matrix(rows: list[dict[str, Any]], max_prompt_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = [row.get("prompt_token_ids") for row in rows]
    if any(ids is None for ids in prompt_ids):
        raise ValueError("Prompt decoder training requires prompt_token_ids. Re-export the dataset from trajectory cache bundles.")
    lengths = [min(len(ids), int(max_prompt_tokens) if max_prompt_tokens else len(ids)) for ids in prompt_ids]
    width = max(lengths) if max_prompt_tokens is None else min(max(lengths), int(max_prompt_tokens))
    if width < 1:
        raise ValueError("Prompt decoder training requires at least one prompt token")
    tokens = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for row_idx, ids in enumerate(prompt_ids):
        clipped = [int(token_id) for token_id in ids[:width]]
        tokens[row_idx, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
        mask[row_idx, : len(clipped)] = True
    return tokens, mask


def _pool_latent_chunks(latents: torch.Tensor, max_latent_chunks: int | None) -> torch.Tensor:
    if latents.ndim == 2 or max_latent_chunks is None:
        return latents
    if latents.ndim != 3:
        raise ValueError(f"Expected latent tensor [records, chunks, dim] or [records, dim], got {tuple(latents.shape)}")
    if latents.shape[1] <= int(max_latent_chunks):
        return latents
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        latents.transpose(1, 2),
        output_size=int(max_latent_chunks),
    ).transpose(1, 2)
    return pooled.contiguous()


def _checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _save_prompt_token_decoder_checkpoint(
    path: Path,
    model: LatentPromptTokenDecoder,
    *,
    token_vocab: list[int],
    latent_dim: int,
    vocab_size: int,
    prompt_length: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    max_latent_chunks: int | None,
    checkpoint_metadata: Any,
    epoch: int,
    loss: float | None,
) -> None:
    torch.save(
        {
            "state_dict": _checkpoint_state_dict(model),
            "config": {
                "latent_dim": int(latent_dim),
                "vocab_size": int(vocab_size),
                "token_id_vocab": token_vocab,
                "max_prompt_tokens": int(prompt_length),
                "hidden_dim": int(hidden_dim),
                "num_layers": int(num_layers),
                "num_heads": int(num_heads),
                "max_latent_chunks": int(max_latent_chunks) if max_latent_chunks is not None else None,
            },
            "checkpoint_metadata": checkpoint_metadata,
            "prompt_decoder_epoch": int(epoch),
            "prompt_decoder_loss": float(loss) if loss is not None else None,
        },
        path,
    )


def train_prompt_decoder(
    dataset_path: Path,
    output_dir: Path | None = None,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    num_layers: int = 2,
    num_heads: int = 8,
    max_prompt_tokens: int | None = None,
    max_latent_chunks: int | None = 128,
    device_name: str = "cpu",
    log_every: int = 25,
    progress_every_batches: int = 25,
    checkpoint_every: int = 10,
) -> PromptDecoderTrainSummary:
    payload = torch.load(dataset_path, map_location="cpu")
    latents = payload["latents"].detach().cpu().float()
    rows = list(payload["rows"])
    if len(rows) != int(latents.shape[0]):
        raise ValueError(f"Row/latent mismatch: {len(rows)} rows vs {int(latents.shape[0])} latents")
    latents = _pool_latent_chunks(latents, max_latent_chunks=max_latent_chunks)
    original_targets, target_mask = _prompt_token_matrix(rows, max_prompt_tokens=max_prompt_tokens)
    token_vocab = sorted({int(token_id) for token_id in original_targets[target_mask].tolist()})
    token_to_class = {token_id: idx for idx, token_id in enumerate(token_vocab)}
    targets = torch.full_like(original_targets, -100)
    for token_id, class_id in token_to_class.items():
        targets[original_targets == token_id] = int(class_id)
    vocab_size = len(token_vocab)

    output_dir = output_dir or dataset_path.parent / "prompt_decoder_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    latents = latents.to(device)
    targets = targets.to(device)
    target_mask = target_mask.to(device)
    prompt_length = int(targets.shape[1])
    model = LatentPromptTokenDecoder(
        latent_dim=int(latents.shape[-1]),
        vocab_size=vocab_size,
        max_prompt_tokens=prompt_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history: list[dict[str, float]] = []
    batch_size = max(1, int(batch_size))
    total_batches = (int(latents.shape[0]) + batch_size - 1) // batch_size
    print(
        "[latent-prompt-decoder-train] "
        f"rows={len(rows)} latent_shape={list(payload['latents'].shape)} "
        f"pooled_latent_shape={list(latents.shape)} prompt_tokens={prompt_length} "
        f"compact_vocab={vocab_size} hidden_dim={hidden_dim} layers={num_layers} heads={num_heads}",
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(int(latents.shape[0]), device=device)
        total_loss = 0.0
        total_tokens = 0
        for batch_index, start in enumerate(range(0, int(latents.shape[0]), batch_size), start=1):
            idx = permutation[start : start + batch_size]
            logits = model(latents[idx], prompt_length=prompt_length)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets[idx].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            valid_tokens = int(target_mask[idx].sum().item())
            total_loss += float(loss.item()) * max(1, valid_tokens)
            total_tokens += max(1, valid_tokens)
            if progress_every_batches > 0 and (
                batch_index == 1 or batch_index == total_batches or batch_index % int(progress_every_batches) == 0
            ):
                partial_loss = total_loss / max(1, total_tokens)
                print(
                    f"[latent-prompt-decoder-train epoch {epoch}/{epochs} batch {batch_index}/{total_batches}] "
                    f"partial_loss={partial_loss:.6g}",
                    flush=True,
                )
        mean_loss = total_loss / max(1, total_tokens)
        history.append({"epoch": float(epoch), "loss": mean_loss})
        if log_every > 0 and (epoch == 1 or epoch == int(epochs) or epoch % int(log_every) == 0):
            print(f"[latent-prompt-decoder-train] epoch {epoch}/{epochs} loss={mean_loss:.6g}", flush=True)
        if checkpoint_every > 0 and (epoch == 1 or epoch == int(epochs) or epoch % int(checkpoint_every) == 0):
            _save_prompt_token_decoder_checkpoint(
                output_dir / "prompt_token_decoder_latest.pt",
                model,
                token_vocab=token_vocab,
                latent_dim=int(latents.shape[-1]),
                vocab_size=vocab_size,
                prompt_length=prompt_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_latent_chunks=max_latent_chunks,
                checkpoint_metadata=payload.get("checkpoint_metadata"),
                epoch=epoch,
                loss=mean_loss,
            )
            if epoch == 1 or epoch % int(checkpoint_every) == 0:
                _save_prompt_token_decoder_checkpoint(
                    output_dir / f"prompt_token_decoder_epoch_{epoch:06d}.pt",
                    model,
                    token_vocab=token_vocab,
                    latent_dim=int(latents.shape[-1]),
                    vocab_size=vocab_size,
                    prompt_length=prompt_length,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    max_latent_chunks=max_latent_chunks,
                    checkpoint_metadata=payload.get("checkpoint_metadata"),
                    epoch=epoch,
                    loss=mean_loss,
                )

    with torch.no_grad():
        logits = model(latents, prompt_length=prompt_length)
        predictions = torch.argmax(logits, dim=-1)
        correct_tokens = ((predictions == targets) & target_mask).sum().item()
        total_valid = max(1, int(target_mask.sum().item()))
        token_accuracy = float(correct_tokens / total_valid)
        exact_match = float((((predictions == targets) | ~target_mask).all(dim=1)).float().mean().item())

    decoded_rows = []
    for idx, row in enumerate(rows):
        length = int(target_mask[idx].sum().item())
        decoded_rows.append(
            {
                "index": row.get("index", idx),
                "task_id": row.get("task_id"),
                "target_prompt_token_ids": original_targets[idx, :length].detach().cpu().tolist(),
                "decoded_prompt_token_ids": [
                    token_vocab[int(class_id)] for class_id in predictions[idx, :length].detach().cpu().tolist()
                ],
                "exact_token_match": bool(torch.all(predictions[idx, :length] == targets[idx, :length]).item()),
            }
        )

    checkpoint_path = output_dir / "prompt_token_decoder.pt"
    _save_prompt_token_decoder_checkpoint(
        checkpoint_path,
        model,
        token_vocab=token_vocab,
        latent_dim=int(latents.shape[-1]),
        vocab_size=vocab_size,
        prompt_length=prompt_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_latent_chunks=max_latent_chunks,
        checkpoint_metadata=payload.get("checkpoint_metadata"),
        epoch=int(epochs),
        loss=float(history[-1]["loss"]) if history else None,
    )
    rows_path = output_dir / "decoded_prompt_tokens.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in decoded_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    history_path = output_dir / "training_history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = PromptDecoderTrainSummary(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        rows=len(rows),
        epochs=int(epochs),
        latent_shape=list(payload["latents"].shape),
        vocab_size=int(vocab_size),
        original_token_vocab_size=int(max(token_vocab) + 1) if token_vocab else 0,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        max_prompt_tokens=prompt_length,
        max_latent_chunks=int(max_latent_chunks) if max_latent_chunks is not None else None,
        progress_every_batches=int(progress_every_batches),
        final_loss=float(history[-1]["loss"]) if history else float("nan"),
        final_token_accuracy=token_accuracy,
        exact_token_match=exact_match,
        artifacts={
            "prompt_token_decoder.pt": str(checkpoint_path),
            "decoded_prompt_tokens.jsonl": str(rows_path),
            "training_history.jsonl": str(history_path),
            "prompt_token_decoder_latest.pt": str(output_dir / "prompt_token_decoder_latest.pt"),
        },
    )
    write_json(output_dir / "prompt_decoder_train_summary.json", asdict(summary))
    return summary


def load_prompt_token_decoder(checkpoint_path: Path, device_name: str = "cpu") -> tuple[LatentPromptTokenDecoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = dict(payload["config"])
    model = LatentPromptTokenDecoder(
        latent_dim=int(config["latent_dim"]),
        vocab_size=int(config["vocab_size"]),
        max_prompt_tokens=int(config["max_prompt_tokens"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device_name))
    model.eval()
    return model, payload


def _decode_token_ids(token_ids: list[int], model_id: str | None) -> str | None:
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as exc:  # pragma: no cover - depends on local tokenizer availability
        return f"<tokenizer decode failed: {exc}>"


def decode_interpolation_prompts(
    prompt_decoder_checkpoint: Path,
    interpolation_dir: Path,
    output_dir: Path | None = None,
    model_id: str | None = None,
    device_name: str = "cpu",
) -> PromptInterpolationDecodeSummary:
    """Decode interpolated latent points into approximate source prompts."""

    model, decoder_payload = load_prompt_token_decoder(prompt_decoder_checkpoint, device_name=device_name)
    config = dict(decoder_payload["config"])
    token_vocab = [int(token_id) for token_id in config["token_id_vocab"]]
    max_latent_chunks = config.get("max_latent_chunks")
    summary = json.loads((interpolation_dir / "interpolation_summary.json").read_text(encoding="utf-8"))
    analysis_dir = Path(summary["analysis_dir"])
    latent_payload = torch.load(analysis_dir / "checkpoint_latents.pt", map_location="cpu")
    latents = latent_payload["latents"].detach().cpu().float()
    pairs = read_jsonl(interpolation_dir / "interpolation_pairs.jsonl")
    alphas = [float(alpha) for alpha in summary.get("alphas", [])]
    output_dir = output_dir or interpolation_dir / "prompt_decoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for pair in pairs:
            a_index = int(pair["a_index"])
            b_index = int(pair["b_index"])
            z_a = latents[a_index]
            z_b = latents[b_index]
            for alpha in alphas:
                z = ((1.0 - alpha) * z_a + alpha * z_b).unsqueeze(0)
                z = _pool_latent_chunks(z, max_latent_chunks=max_latent_chunks).to(model_device)
                logits = model(z, prompt_length=int(config["max_prompt_tokens"]))
                class_ids = torch.argmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
                token_ids = [token_vocab[int(class_id)] for class_id in class_ids]
                rows.append(
                    {
                        "pair_id": pair.get("pair_id"),
                        "pair_type": pair.get("pair_type"),
                        "alpha": alpha,
                        "a_task_id": pair.get("a_task_id"),
                        "b_task_id": pair.get("b_task_id"),
                        "a_prompt": (pair.get("a") or {}).get("prompt"),
                        "b_prompt": (pair.get("b") or {}).get("prompt"),
                        "decoded_prompt_token_ids": token_ids,
                        "decoded_prompt_text": _decode_token_ids(token_ids, model_id),
                        "prompt_decoder_checkpoint": str(prompt_decoder_checkpoint),
                        "prompt_decoder_epoch": decoder_payload.get("prompt_decoder_epoch"),
                        "prompt_decoder_loss": decoder_payload.get("prompt_decoder_loss"),
                    }
                )

    jsonl_path = output_dir / "interpolation_decoded_prompts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    md_path = output_dir / "interpolation_decoded_prompts.md"
    lines = [
        "# Interpolation Prompt Decoder",
        "",
        f"- Prompt decoder checkpoint: `{prompt_decoder_checkpoint}`",
        f"- Prompt decoder epoch: `{decoder_payload.get('prompt_decoder_epoch')}`",
        f"- Prompt decoder loss: `{decoder_payload.get('prompt_decoder_loss')}`",
        f"- Rows: {len(rows)}",
        "",
    ]
    for pair in pairs:
        pair_rows = [row for row in rows if row["pair_id"] == pair.get("pair_id")]
        lines.extend(
            [
                f"## {pair.get('pair_id')} ({pair.get('a_task_id')} -> {pair.get('b_task_id')})",
                "",
                "| alpha | decoded prompt |",
                "| ---: | --- |",
            ]
        )
        for row in pair_rows:
            text = row.get("decoded_prompt_text") or " ".join(str(token_id) for token_id in row["decoded_prompt_token_ids"][:48])
            text = text.replace("\n", "<br>").replace("|", "\\|")
            lines.append(f"| {row['alpha']:.3g} | {text} |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    out_summary = PromptInterpolationDecodeSummary(
        prompt_decoder_checkpoint=str(prompt_decoder_checkpoint),
        interpolation_dir=str(interpolation_dir),
        output_dir=str(output_dir),
        rows=len(rows),
        checkpoint_epoch=summary.get("checkpoint_epoch"),
        artifacts={
            "interpolation_decoded_prompts.jsonl": str(jsonl_path),
            "interpolation_decoded_prompts.md": str(md_path),
        },
    )
    write_json(output_dir / "interpolation_prompt_decode_summary.json", asdict(out_summary))
    return out_summary
