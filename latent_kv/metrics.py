"""Deterministic metric aggregation and report generation."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .schemas import BaselineMetric, read_json, read_jsonl, write_json


def binomial_ci95(accuracy: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n)


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def aggregate_records(records: list[dict[str, Any]], baseline: str = "no_cache") -> list[BaselineMetric]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["benchmark"]].append(record)

    metrics = []
    for benchmark, rows in sorted(groups.items()):
        n = len(rows)
        accuracy = sum(1 for row in rows if row.get("correct")) / n if n else 0.0
        retries = [float(row.get("retry_index", 0)) for row in rows]
        lengths = [float(row.get("generated_tokens") or len(row.get("output_text", "").split())) for row in rows]
        latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
        memories = [float(row["memory_bytes"]) for row in rows if row.get("memory_bytes") is not None]
        failed_then_recovered = [
            row for row in rows if row.get("retry_index", 0) > 0 and row.get("correct")
        ]
        metrics.append(
            BaselineMetric(
                baseline=baseline,
                benchmark=benchmark,
                examples=n,
                accuracy=accuracy,
                ci95=binomial_ci95(accuracy, n),
                retry_efficiency=1.0 / (1.0 + mean(retries)) if retries else 1.0,
                error_recovery_rate=len(failed_then_recovered) / n if n else 0.0,
                mean_reasoning_length=mean(lengths) if lengths else 0.0,
                mean_latency_s=_safe_mean(latencies),
                memory_bytes=_safe_mean(memories),
            )
        )
    return metrics


def metric_to_dict(metric: BaselineMetric) -> dict[str, Any]:
    return metric.__dict__.copy()


def write_metrics(run_dir: Path, metrics: list[BaselineMetric], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "baselines": [metric_to_dict(metric) for metric in metrics],
        "extra": extra or {},
    }
    write_json(run_dir / "metrics.json", payload)


def load_metric_payload(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)
    return {"baselines": [], "extra": {}}


def render_report(run_dir: Path, records: list[dict[str, Any]], metrics_payload: dict[str, Any]) -> str:
    lines = [
        "# Latent KV Run Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Baseline Comparison",
        "",
        "| Baseline | Benchmark | N | Accuracy | CI95 | Retry Efficiency | Recovery | Mean Length | Mean Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metrics_payload.get("baselines", []), key=lambda r: (r["benchmark"], r["baseline"])):
        latency = row["mean_latency_s"]
        latency_text = "" if latency is None else f"{latency:.3f}"
        lines.append(
            "| {baseline} | {benchmark} | {examples} | {accuracy:.3f} | {ci95:.3f} | "
            "{retry_efficiency:.3f} | {error_recovery_rate:.3f} | {mean_reasoning_length:.1f} | "
            "{latency} |".format(**row, latency=latency_text)
        )

    lines.extend(["", "## Qualitative Examples", ""])
    for record in records[:5]:
        status = "correct" if record.get("correct") else "wrong"
        output = str(record.get("output_text", "")).strip().replace("\n", " ")
        if len(output) > 260:
            output = output[:257] + "..."
        lines.extend(
            [
                f"### {record.get('benchmark')} / {record.get('task_id')} ({status})",
                "",
                f"- Target: `{record.get('target')}`",
                f"- Parsed: `{record.get('parsed_answer')}`",
                f"- Output: {output}",
                "",
            ]
        )

    extra = metrics_payload.get("extra", {})
    if extra:
        lines.extend(["## Extra Diagnostics", ""])
        for key, value in sorted(extra.items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    return "\n".join(lines)


def evaluate_run(run_dir: Path, baseline: str = "no_cache") -> dict[str, Any]:
    records = read_jsonl(run_dir / "records.jsonl")
    existing = load_metric_payload(run_dir)
    baseline_metrics = [metric_to_dict(metric) for metric in aggregate_records(records, baseline=baseline)]

    by_key = {
        (row["baseline"], row["benchmark"]): row
        for row in existing.get("baselines", [])
    }
    for row in baseline_metrics:
        by_key[(row["baseline"], row["benchmark"])] = row

    payload = {"baselines": list(by_key.values()), "extra": existing.get("extra", {})}
    write_json(run_dir / "metrics.json", payload)
    report = render_report(run_dir, records, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload

