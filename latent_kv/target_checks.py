"""Compare run metrics against tracked reported targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .baseline_targets import TARGETS
from .schemas import read_json, write_json


BASELINE_ALIASES = {
    "cot": "chain_of_thought",
    "chain_of_thought": "chain_of_thought",
    "self_consistency": "self_consistency",
    "tree_of_thoughts": "tree_of_thoughts",
    "react_hotpotqa": "react_hotpotqa",
}


def _normalize(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _protocol_metadata(extra: dict[str, Any], baseline: str) -> dict[str, Any]:
    payload = extra.get(f"prompt_{baseline}_protocol")
    return payload if isinstance(payload, dict) else {}


def _protocol_audit(target: Any, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dimensions = {
        "model": (target.reported_model, metadata.get("model_id")),
        "prompt": (target.reported_prompt, metadata.get("prompt_family") or metadata.get("protocol_name")),
        "decoding": (target.reported_decoding, metadata.get("decoding_protocol")),
        "split": (target.reported_split, metadata.get("benchmark_split")),
    }
    if target.reported_sample_count is not None:
        dimensions["sample_count"] = (target.reported_sample_count, metadata.get("sample_count"))
    return {
        name: {
            "expected": expected,
            "observed": observed,
            "match": _normalize(expected) == _normalize(observed),
        }
        for name, (expected, observed) in dimensions.items()
    }


def check_targets(run_dir: Path, tolerance: float = 1.0) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    payload = read_json(metrics_path)
    rows = payload.get("baselines", [])
    extra = payload.get("extra", {})
    checks: list[dict[str, Any]] = []
    targets = {(target.name, target.benchmark): target for target in TARGETS}

    for row in rows:
        target_name = BASELINE_ALIASES.get(row["baseline"])
        if not target_name:
            continue
        target = targets.get((target_name, row["benchmark"]))
        if not target:
            continue
        observed = float(row["accuracy"]) * 100.0
        protocol_key = f"prompt_{row['baseline']}_protocol_match"
        protocol_metadata = _protocol_metadata(extra, row["baseline"])
        protocol_dimensions = _protocol_audit(target, protocol_metadata) if protocol_metadata else {}
        if protocol_dimensions:
            protocol_match = all(item["match"] for item in protocol_dimensions.values())
        else:
            protocol_match = bool(extra.get(protocol_key, False))
        target_match = protocol_match and abs(observed - target.reported_value) <= tolerance
        checks.append(
            {
                "baseline": row["baseline"],
                "benchmark": row["benchmark"],
                "observed": observed,
                "target": target.reported_value,
                "metric": target.metric,
                "source": target.source,
                "protocol": target.protocol,
                "protocol_match": protocol_match,
                "protocol_dimensions": protocol_dimensions,
                "target_match": target_match,
                "status": "target_match"
                if target_match
                else ("protocol_mismatch" if not protocol_match else "target_mismatch"),
            }
        )

    payload["target_checks"] = checks
    write_json(metrics_path, payload)
    return checks
