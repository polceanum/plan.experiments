"""Training-curve diagnostics for learned local codecs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .schemas import read_json, read_jsonl, write_json


@dataclass(frozen=True)
class TrainingCurveSummary:
    method: str
    points: int
    first_epoch: int | None
    last_epoch: int | None
    first_loss: float | None
    last_loss: float | None
    min_loss: float | None
    min_loss_epoch: int | None
    improvement_abs: float | None
    improvement_pct: float | None
    monotonic_nonincreasing: bool | None
    increase_steps: int
    max_increase: float | None
    mean_delta: float | None
    delta_std: float | None
    noise_ratio: float | None
    early_mean_delta: float | None
    late_mean_delta: float | None
    shape: str
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingStatusSummary:
    method: str
    rows: int
    current_event: str
    current_epoch: int | None
    total_epochs: int | None
    current_batch: int | None
    total_batches: int | None
    current_loss: float | None
    last_completed_epoch: int | None
    last_completed_loss: float | None
    memory_gb: float | None
    replay_gradients: bool
    training_log_path: str
    status_path: str
    readable_log_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _curve_shape(deltas: list[float], first_loss: float, last_loss: float) -> str:
    if not deltas:
        return "single_point"
    total_change = first_loss - last_loss
    if abs(total_change) < max(abs(first_loss) * 1e-4, 1e-8):
        return "flat"
    increases = [delta for delta in deltas if delta > 0]
    if len(increases) > max(1, len(deltas) // 4):
        return "noisy_decreasing" if total_change > 0 else "noisy_increasing"
    midpoint = max(1, len(deltas) // 2)
    early = [abs(delta) for delta in deltas[:midpoint] if delta < 0]
    late = [abs(delta) for delta in deltas[midpoint:] if delta < 0]
    early_rate = _mean_or_none(early) or 0.0
    late_rate = _mean_or_none(late) or 0.0
    if total_change > 0 and late_rate > early_rate * 1.5 and late_rate > 0:
        return "accelerating_decrease"
    if total_change > 0 and early_rate > late_rate * 1.5 and early_rate > 0:
        return "decelerating_decrease"
    return "steady_decrease" if total_change > 0 else "increasing"


def summarize_training_curve(run_dir: Path, method: str) -> TrainingCurveSummary:
    method = method.lower()
    log_path = run_dir / "compressions" / f"{method}_training.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    rows = [row for row in read_jsonl(log_path) if row.get("loss") is not None]
    if not rows:
        raise ValueError(f"No loss rows found in {log_path}")
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(rows)]
    losses = [float(row["loss"]) for row in rows]
    deltas = [losses[idx] - losses[idx - 1] for idx in range(1, len(losses))]
    increases = [delta for delta in deltas if delta > 0]
    min_idx = min(range(len(losses)), key=losses.__getitem__)
    improvement_abs = losses[0] - losses[-1]
    improvement_pct = (improvement_abs / abs(losses[0])) if losses[0] else None
    delta_std = pstdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None
    mean_delta = _mean_or_none(deltas)
    noise_ratio = None
    if delta_std is not None and mean_delta not in (None, 0.0):
        noise_ratio = delta_std / abs(mean_delta)
    midpoint = max(1, len(deltas) // 2) if deltas else 0
    out_path = run_dir / "compressions" / f"{method}_training_curve.json"
    summary = TrainingCurveSummary(
        method=method,
        points=len(losses),
        first_epoch=epochs[0],
        last_epoch=epochs[-1],
        first_loss=losses[0],
        last_loss=losses[-1],
        min_loss=losses[min_idx],
        min_loss_epoch=epochs[min_idx],
        improvement_abs=improvement_abs,
        improvement_pct=improvement_pct,
        monotonic_nonincreasing=all(delta <= 0 for delta in deltas) if deltas else None,
        increase_steps=len(increases),
        max_increase=max(increases) if increases else 0.0 if deltas else None,
        mean_delta=mean_delta,
        delta_std=delta_std,
        noise_ratio=noise_ratio,
        early_mean_delta=_mean_or_none(deltas[:midpoint]) if deltas else None,
        late_mean_delta=_mean_or_none(deltas[midpoint:]) if deltas[midpoint:] else None,
        shape=_curve_shape(deltas, losses[0], losses[-1]),
        result_path=str(out_path),
    )
    write_json(out_path, {"summary": summary.to_dict(), "points": rows})

    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {"baselines": [], "extra": {}}
    extra = metrics.setdefault("extra", {})
    prefix = f"training_curve_{method}"
    extra[f"{prefix}_path"] = str(out_path)
    extra[f"{prefix}_shape"] = summary.shape
    extra[f"{prefix}_points"] = summary.points
    extra[f"{prefix}_first_loss"] = summary.first_loss
    extra[f"{prefix}_last_loss"] = summary.last_loss
    extra[f"{prefix}_improvement_pct"] = summary.improvement_pct
    extra[f"{prefix}_monotonic_nonincreasing"] = summary.monotonic_nonincreasing
    extra[f"{prefix}_noise_ratio"] = summary.noise_ratio
    write_json(metrics_path, metrics)
    return summary


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _row_loss(row: dict[str, Any]) -> float | None:
    value = row.get("loss", row.get("partial_loss"))
    if value is None:
        return None
    return float(value)


def _human_log_line(row: dict[str, Any]) -> str | None:
    if row.get("event") == "startup":
        shape = row.get("temporal_matrix_shape")
        return (
            f"startup device={row.get('device')} epochs={row.get('epochs')} "
            f"shape={shape} latent_dim={row.get('latent_dim')} hidden_dim={row.get('hidden_dim')} "
            f"replay_weight={row.get('replay_loss_weight')} replay_steps={row.get('replay_loss_steps')}"
        )
    if row.get("event") == "batch_heartbeat":
        return (
            f"epoch {row.get('epoch')}/{row.get('epochs')} "
            f"batch {row.get('batch')}/{row.get('batches')} "
            f"partial_loss={_format_float(row.get('partial_loss'))} "
            f"mse={_format_float((row.get('partial_loss_components') or {}).get('masked_temporal_reconstruction_mse'))} "
            f"replay_kl={_format_float((row.get('partial_loss_components') or {}).get('teacher_forced_generation_replay_kl'))} "
            f"memory_gb={_format_float(row.get('memory_gb'))} "
            f"elapsed_s={_format_float(row.get('elapsed_s'))}"
        )
    if row.get("loss") is not None:
        components = row.get("loss_components") or {}
        return (
            f"epoch {row.get('epoch')}/{row.get('epochs')} done "
            f"loss={_format_float(row.get('loss'))} "
            f"mse={_format_float(components.get('masked_temporal_reconstruction_mse'))} "
            f"replay_kl={_format_float(components.get('teacher_forced_generation_replay_kl'))} "
            f"memory_gb={_format_float(row.get('memory_gb'))} "
            f"elapsed_s={_format_float(row.get('elapsed_s'))}"
        )
    return None


def render_training_status(
    run_dir: Path,
    method: str = "rae_temporal",
    status_path: Path | None = None,
    readable_log_path: Path | None = None,
) -> TrainingStatusSummary:
    method = method.lower()
    log_path = run_dir / "compressions" / f"{method}_training.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    rows = read_jsonl(log_path)
    if not rows:
        raise ValueError(f"No rows found in {log_path}")

    last = rows[-1]
    completed = [row for row in rows if row.get("loss") is not None]
    last_completed = completed[-1] if completed else None
    current_loss = _row_loss(last)
    replay_active = bool(last.get("replay_gradients", False))
    components = last.get("partial_loss_components") or last.get("loss_components") or {}
    if components.get("teacher_forced_generation_replay_kl") not in {None, 0, 0.0}:
        replay_active = True
    if not replay_active and last_completed is not None:
        completed_components = last_completed.get("loss_components") or {}
        if completed_components.get("teacher_forced_generation_replay_kl") not in {None, 0, 0.0}:
            replay_active = True
    status_path = status_path or (run_dir / f"{method}_status.md")

    recent_completed = completed[-8:]
    lines = [
        f"# {method} Training Status",
        "",
        f"Source log: `{log_path}`",
        "",
        f"- Current event: `{last.get('event', 'epoch')}`",
        f"- Epoch: `{last.get('epoch')}/{last.get('epochs')}`",
        f"- Batch: `{last.get('batch', '-')}/{last.get('batches', '-')}`",
        f"- Current loss: `{_format_float(current_loss)}`",
        f"- Memory GB: `{_format_float(last.get('memory_gb'))}`",
        f"- Replay gradients: `{replay_active}`",
        "",
    ]
    if last_completed is not None:
        lines.extend(
            [
                f"- Last completed epoch: `{last_completed.get('epoch')}/{last_completed.get('epochs')}`",
                f"- Last completed loss: `{_format_float(last_completed.get('loss'))}`",
                "",
                "## Recent Completed Epochs",
                "",
            ]
        )
        for row in recent_completed:
            lines.append(
                f"- epoch `{row.get('epoch')}`: loss `{_format_float(row.get('loss'))}`, "
                f"memory `{_format_float(row.get('memory_gb'))}` GB"
            )

    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if readable_log_path is not None:
        readable_log_path.parent.mkdir(parents=True, exist_ok=True)
        readable_lines = [line for row in rows if (line := _human_log_line(row)) is not None]
        readable_log_path.write_text("\n".join(readable_lines) + "\n", encoding="utf-8")

    return TrainingStatusSummary(
        method=method,
        rows=len(rows),
        current_event=str(last.get("event", "epoch")),
        current_epoch=int(last["epoch"]) if last.get("epoch") is not None else None,
        total_epochs=int(last["epochs"]) if last.get("epochs") is not None else None,
        current_batch=int(last["batch"]) if last.get("batch") is not None else None,
        total_batches=int(last["batches"]) if last.get("batches") is not None else None,
        current_loss=current_loss,
        last_completed_epoch=int(last_completed["epoch"]) if last_completed and last_completed.get("epoch") is not None else None,
        last_completed_loss=float(last_completed["loss"]) if last_completed and last_completed.get("loss") is not None else None,
        memory_gb=float(last["memory_gb"]) if last.get("memory_gb") is not None else None,
        replay_gradients=replay_active,
        training_log_path=str(log_path),
        status_path=str(status_path),
        readable_log_path=str(readable_log_path) if readable_log_path is not None else None,
    )
