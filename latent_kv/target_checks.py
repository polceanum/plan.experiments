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
                "target_match": target_match,
                "status": "target_match"
                if target_match
                else ("protocol_mismatch" if not protocol_match else "target_mismatch"),
            }
        )

    payload["target_checks"] = checks
    write_json(metrics_path, payload)
    return checks
