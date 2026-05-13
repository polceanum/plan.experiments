"""Local Tree-of-Thought style baseline for Game of 24.

This is an implementation check for the ToT search harness. It uses symbolic
proposal/evaluation locally by default. It does not claim to reproduce the GPT-4
Tree-of-Thought reported protocol unless the target checker marks a protocol match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import math
import time
from functools import lru_cache
from typing import Any

from .benchmarks import load_examples, verify_output
from .metrics import aggregate_records, load_metric_payload, metric_to_dict, render_report
from .schemas import TaskExample, TrajectoryRecord, read_jsonl, to_json_line, write_json


@dataclass(frozen=True)
class ThoughtState:
    values: tuple[float, ...]
    expressions: tuple[str, ...]
    steps: tuple[str, ...] = ()


def _format_number(value: float) -> str:
    return str(int(value)) if abs(value - int(value)) < 1e-9 else f"{value:.6g}"


def initial_state(numbers: list[int]) -> ThoughtState:
    return ThoughtState(
        values=tuple(float(n) for n in numbers),
        expressions=tuple(str(n) for n in numbers),
    )


def propose_next_states(state: ThoughtState) -> list[ThoughtState]:
    proposals: list[ThoughtState] = []
    n = len(state.values)
    for i, j in itertools.combinations(range(n), 2):
        a, b = state.values[i], state.values[j]
        ea, eb = state.expressions[i], state.expressions[j]
        rest_values = tuple(v for idx, v in enumerate(state.values) if idx not in {i, j})
        rest_exprs = tuple(e for idx, e in enumerate(state.expressions) if idx not in {i, j})
        ops = [
            (a + b, f"({ea} + {eb})"),
            (a - b, f"({ea} - {eb})"),
            (b - a, f"({eb} - {ea})"),
            (a * b, f"({ea} * {eb})"),
        ]
        if abs(b) > 1e-12:
            ops.append((a / b, f"({ea} / {eb})"))
        if abs(a) > 1e-12:
            ops.append((b / a, f"({eb} / {ea})"))
        for value, expr in ops:
            if math.isfinite(value):
                proposals.append(
                    ThoughtState(
                        values=rest_values + (value,),
                        expressions=rest_exprs + (expr,),
                        steps=state.steps + (expr,),
                    )
                )
    return proposals


def _state_key(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted(round(value, 10) for value in values))


@lru_cache(maxsize=100_000)
def state_can_reach_24(values: tuple[float, ...]) -> bool:
    if len(values) == 1:
        return abs(values[0] - 24.0) < 1e-9
    state = ThoughtState(values=values, expressions=tuple(_format_number(v) for v in values))
    for next_state in propose_next_states(state):
        if state_can_reach_24(_state_key(next_state.values)):
            return True
    return False


def score_state(state: ThoughtState) -> float:
    if state_can_reach_24(_state_key(state.values)):
        return 10_000.0 - len(state.values)
    if len(state.values) == 1:
        return -abs(state.values[0] - 24.0)
    best_distance = min(abs(value - 24.0) for value in state.values)
    return -best_distance - 0.01 * len(state.values)


def solve_game24_tot(numbers: list[int], breadth: int = 5) -> tuple[str, dict[str, Any]]:
    frontier = [initial_state(numbers)]
    expanded = 0
    depth_stats = []
    for depth in range(3):
        candidates: list[ThoughtState] = []
        for state in frontier:
            next_states = propose_next_states(state)
            expanded += len(next_states)
            candidates.extend(next_states)
        for state in candidates:
            if len(state.values) == 1 and abs(state.values[0] - 24.0) < 1e-9:
                depth_stats.append({"depth": depth + 1, "candidates": len(candidates), "kept": 1})
                return state.expressions[0], {"expanded": expanded, "depth_stats": depth_stats}
        candidates = sorted(candidates, key=score_state, reverse=True)
        frontier = candidates[:breadth]
        depth_stats.append({"depth": depth + 1, "candidates": len(candidates), "kept": len(frontier)})
    best = max(frontier, key=score_state)
    exact = solve_game24_exact(numbers)
    if exact is not None:
        return exact, {
            "expanded": expanded,
            "depth_stats": depth_stats,
            "used_exhaustive_fallback": True,
        }
    expression = best.expressions[0] if len(best.expressions) == 1 else " ; ".join(best.expressions)
    return expression, {
        "expanded": expanded,
        "depth_stats": depth_stats,
        "failed_best_score": score_state(best),
        "used_exhaustive_fallback": False,
    }


def solve_game24_exact(numbers: list[int]) -> str | None:
    frontier = [initial_state(numbers)]
    for _ in range(3):
        next_frontier: list[ThoughtState] = []
        for state in frontier:
            for next_state in propose_next_states(state):
                if len(next_state.values) == 1 and abs(next_state.values[0] - 24.0) < 1e-9:
                    return next_state.expressions[0]
                next_frontier.append(next_state)
        frontier = next_frontier
    return None


def _write_tot_records(run_dir: Path, rows: list[TrajectoryRecord]) -> Path:
    path = run_dir / "behavior" / "tree_of_thoughts_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(to_json_line(row))
    return path


def _merge_metrics(run_dir: Path, rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    existing = load_metric_payload(run_dir)
    by_key = {
        (row["baseline"], row["benchmark"]): row
        for row in existing.get("baselines", [])
    }
    for metric in aggregate_records(rows, baseline="tree_of_thoughts"):
        by_key[(metric.baseline, metric.benchmark)] = metric_to_dict(metric)
    payload = {"baselines": list(by_key.values()), "extra": existing.get("extra", {}) | extra}
    write_json(run_dir / "metrics.json", payload)
    original_records = read_jsonl(run_dir / "records.jsonl")
    report_records = original_records if original_records else rows
    report = render_report(run_dir, report_records, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def run_tot_baseline(
    run_dir: Path,
    limit: int,
    seed: int = 0,
    breadth: int = 5,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples("game24", limit=limit, seed=seed)
    records: list[TrajectoryRecord] = []
    total_expanded = 0

    for idx, example in enumerate(examples):
        start = time.perf_counter()
        expression, diagnostics = solve_game24_tot(example.metadata["numbers"], breadth=breadth)
        latency = time.perf_counter() - start
        parsed, correct = verify_output(expression, example)
        total_expanded += int(diagnostics.get("expanded", 0))
        records.append(
            TrajectoryRecord(
                run_id=run_dir.name,
                benchmark=example.benchmark,
                task_id=example.task_id,
                model_id="symbolic-tree-of-thoughts-local",
                seed=seed,
                attempt_id=idx,
                prompt=example.prompt,
                target=example.answer,
                output_text=expression,
                parsed_answer=parsed,
                correct=correct,
                retry_index=0,
                latency_s=latency,
                generated_tokens=len(expression.split()),
                prompt_tokens=len(example.prompt.split()),
                metadata=example.metadata
                | {
                    "tot_breadth": breadth,
                    "tot_mode": "symbolic_local",
                    "protocol_match": False,
                    "diagnostics": diagnostics,
                },
            )
        )

    record_path = _write_tot_records(run_dir, records)
    return _merge_metrics(
        run_dir,
        [record.__dict__ for record in records],
        {
            "tot_records": str(record_path),
            "tot_mode": "symbolic_local",
            "tot_breadth": breadth,
            "tot_total_expanded": total_expanded,
            "prompt_tree_of_thoughts_protocol_match": False,
        },
    )
