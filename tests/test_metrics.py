from pathlib import Path

from latent_kv.metrics import aggregate_records, evaluate_run
from latent_kv.schemas import TrajectoryRecord, append_jsonl


def test_aggregate_records_by_benchmark():
    records = [
        {"benchmark": "hanoi", "correct": True, "retry_index": 0, "output_text": "a b", "latency_s": 1.0},
        {"benchmark": "hanoi", "correct": False, "retry_index": 1, "output_text": "a", "latency_s": 2.0},
    ]
    metrics = aggregate_records(records, baseline="no_cache")
    assert len(metrics) == 1
    assert metrics[0].accuracy == 0.5
    assert metrics[0].retry_efficiency == 1 / 1.5


def test_evaluate_run_writes_report(tmp_path: Path):
    record = TrajectoryRecord(
        run_id="test",
        benchmark="hanoi",
        task_id="hanoi_0",
        model_id="dry",
        seed=0,
        attempt_id=0,
        prompt="p",
        target="1 -> 3",
        output_text="1 -> 3",
        parsed_answer="[(1, 3)]",
        correct=True,
        retry_index=0,
    )
    append_jsonl(tmp_path / "records.jsonl", record)
    payload = evaluate_run(tmp_path)
    assert payload["baselines"][0]["accuracy"] == 1.0
    assert (tmp_path / "report.md").exists()

