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