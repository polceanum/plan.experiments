import json

from latent_kv.training_diagnostics import summarize_training_curve


def _write_training_log(run_dir, losses):
    path = run_dir / "compressions" / "rae_lstm_training.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, loss in enumerate(losses, start=1):
            handle.write(json.dumps({"epoch": idx, "loss": loss, "method": "rae_lstm"}) + "\n")


def test_training_curve_summary_reports_monotonic_accelerating_drop(tmp_path):
    _write_training_log(tmp_path, [1.0, 0.99, 0.96, 0.90])

    summary = summarize_training_curve(tmp_path, "rae_lstm")

    assert summary.points == 4
    assert summary.monotonic_nonincreasing is True
    assert summary.increase_steps == 0
    assert summary.improvement_abs == 0.09999999999999998
    assert summary.shape == "accelerating_decrease"
    assert (tmp_path / "compressions" / "rae_lstm_training_curve.json").exists()


def test_training_curve_summary_reports_noisy_decrease(tmp_path):
    _write_training_log(tmp_path, [1.0, 0.95, 0.96, 0.90, 0.91, 0.89])

    summary = summarize_training_curve(tmp_path, "rae_lstm")

    assert summary.monotonic_nonincreasing is False
    assert summary.increase_steps == 2
    assert summary.max_increase is not None and summary.max_increase > 0
    assert summary.shape == "noisy_decreasing"