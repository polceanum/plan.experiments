from pathlib import Path

from latent_kv.schemas import write_json
from latent_kv.target_checks import check_targets


def test_target_check_marks_protocol_mismatch(tmp_path: Path):
    write_json(
        tmp_path / "metrics.json",
        {
            "baselines": [
                {
                    "baseline": "cot",
                    "benchmark": "gsm8k",
                    "examples": 1,
                    "accuracy": 0.57,
                    "ci95": 0.0,
                    "retry_efficiency": 1.0,
                    "error_recovery_rate": 0.0,
                    "mean_reasoning_length": 1.0,
                    "mean_latency_s": 0.0,
                    "memory_bytes": None,
                }
            ],
            "extra": {"prompt_cot_protocol_match": False},
        },
    )
    checks = check_targets(tmp_path)
    assert checks[0]["status"] == "protocol_mismatch"


def test_tree_of_thoughts_target_check(tmp_path: Path):
    write_json(
        tmp_path / "metrics.json",
        {
            "baselines": [
                {
                    "baseline": "tree_of_thoughts",
                    "benchmark": "game24",
                    "examples": 1,
                    "accuracy": 0.74,
                    "ci95": 0.0,
                    "retry_efficiency": 1.0,
                    "error_recovery_rate": 0.0,
                    "mean_reasoning_length": 1.0,
                    "mean_latency_s": 0.0,
                    "memory_bytes": None,
                }
            ],
            "extra": {"prompt_tree_of_thoughts_protocol_match": False},
        },
    )
    checks = check_targets(tmp_path)
    assert checks[0]["baseline"] == "tree_of_thoughts"
    assert checks[0]["status"] == "protocol_mismatch"
