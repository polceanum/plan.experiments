"""Local ReAct-style baseline on a tiny deterministic tool environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .metrics import aggregate_records, load_metric_payload, metric_to_dict, render_report
from .schemas import TrajectoryRecord, read_jsonl, to_json_line, write_json


@dataclass(frozen=True)
class ReactTask:
    task_id: str
    goal: str
    item: str
    target_room: str
    rooms: dict[str, tuple[str, ...]]


REACT_TASKS = [
    ReactTask(
        task_id="react_find_mug",
        goal="Find the mug and finish.",
        item="mug",
        target_room="kitchen",
        rooms={"kitchen": ("mug", "plate"), "office": ("book",), "bedroom": ("lamp",)},
    ),
    ReactTask(
        task_id="react_find_key",
        goal="Find the key and finish.",
        item="key",
        target_room="office",
        rooms={"kitchen": ("spoon",), "office": ("key", "book"), "bedroom": ("blanket",)},
    ),
    ReactTask(
        task_id="react_find_lamp",
        goal="Find the lamp and finish.",
        item="lamp",
        target_room="bedroom",
        rooms={"kitchen": ("cup",), "office": ("paper",), "bedroom": ("lamp", "pillow")},
    ),
]


class TinyToolWorld:
    def __init__(self, task: ReactTask) -> None:
        self.task = task
        self.room: str | None = None
        self.inventory: set[str] = set()
        self.done = False

    def step(self, action: str) -> tuple[str, bool]:
        action = action.strip().lower()
        if action.startswith("go "):
            room = action.removeprefix("go ").strip()
            if room not in self.task.rooms:
                return f"Unknown room: {room}", False
            self.room = room
            items = ", ".join(self.task.rooms[room]) or "nothing"
            return f"You are in the {room}. You see {items}.", False
        if action.startswith("take "):
            item = action.removeprefix("take ").strip()
            if self.room is None:
                return "You are not in a room.", False
            if item not in self.task.rooms[self.room]:
                return f"There is no {item} here.", False
            self.inventory.add(item)
            return f"You took the {item}.", False
        if action == "finish":
            self.done = self.task.item in self.inventory
            return ("Task complete." if self.done else "You do not have the required item."), self.done
        return f"Unknown action: {action}", False


def symbolic_react_policy(task: ReactTask) -> list[tuple[str, str]]:
    trace: list[tuple[str, str]] = []
    for room in task.rooms:
        trace.append((f"I should inspect the {room}.", f"go {room}"))
        if task.item in task.rooms[room]:
            trace.append((f"The {task.item} is here, so I should take it.", f"take {task.item}"))
            trace.append(("I have the required item, so I can finish.", "finish"))
            break
        trace.append((f"The {task.item} is not in the {room}; continue searching.", ""))
    return [(thought, action) for thought, action in trace if action]


def run_react_task(task: ReactTask, max_steps: int = 8) -> tuple[str, bool, list[dict[str, str]]]:
    env = TinyToolWorld(task)
    transcript: list[dict[str, str]] = []
    success = False
    for step_idx, (thought, action) in enumerate(symbolic_react_policy(task)[:max_steps], start=1):
        observation, success = env.step(action)
        transcript.append(
            {
                "step": str(step_idx),
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )
        if success:
            break
    text = "\n".join(
        f"Thought {row['step']}: {row['thought']}\n"
        f"Action {row['step']}: {row['action']}\n"
        f"Observation {row['step']}: {row['observation']}"
        for row in transcript
    )
    return text, success, transcript


def _write_react_records(run_dir: Path, rows: list[TrajectoryRecord]) -> Path:
    path = run_dir / "behavior" / "react_records.jsonl"
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
    for metric in aggregate_records(rows, baseline="react"):
        by_key[(metric.baseline, metric.benchmark)] = metric_to_dict(metric)
    payload = {"baselines": list(by_key.values()), "extra": existing.get("extra", {}) | extra}
    write_json(run_dir / "metrics.json", payload)
    original_records = read_jsonl(run_dir / "records.jsonl")
    report_records = original_records if original_records else rows
    report = render_report(run_dir, report_records, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def run_react_baseline(run_dir: Path, limit: int, max_steps: int = 8) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[TrajectoryRecord] = []
    for idx, task in enumerate(REACT_TASKS[:limit]):
        start = time.perf_counter()
        text, success, transcript = run_react_task(task, max_steps=max_steps)
        latency = time.perf_counter() - start
        records.append(
            TrajectoryRecord(
                run_id=run_dir.name,
                benchmark="react_toolworld",
                task_id=task.task_id,
                model_id="symbolic-react-local",
                seed=0,
                attempt_id=idx,
                prompt=task.goal,
                target=task.item,
                output_text=text,
                parsed_answer=task.item if success else None,
                correct=success,
                retry_index=0,
                latency_s=latency,
                generated_tokens=len(text.split()),
                prompt_tokens=len(task.goal.split()),
                metadata={
                    "react_mode": "symbolic_local",
                    "rooms": {room: list(items) for room, items in task.rooms.items()},
                    "target_room": task.target_room,
                    "transcript": transcript,
                    "protocol_match": False,
                },
            )
        )

    record_path = _write_react_records(run_dir, records)
    return _merge_metrics(
        run_dir,
        [record.__dict__ for record in records],
        {
            "react_records": str(record_path),
            "react_mode": "symbolic_local",
            "react_max_steps": max_steps,
            "prompt_react_protocol_match": False,
        },
    )

