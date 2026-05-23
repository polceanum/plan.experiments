import json

from latent_kv.training_diagnostics import render_training_status, summarize_training_curve


def _write_training_log(run_dir, losses):
    path = run_dir / "compressions" / "rae_temporal_training.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, loss in enumerate(losses, start=1):
            handle.write(json.dumps({"epoch": idx, "loss": loss, "method": "rae_temporal"}) + "\n")


def test_training_curve_summary_reports_monotonic_accelerating_drop(tmp_path):
    _write_training_log(tmp_path, [1.0, 0.99, 0.96, 0.90])

    summary = summarize_training_curve(tmp_path, "rae_temporal")

    assert summary.points == 4
    assert summary.monotonic_nonincreasing is True
    assert summary.increase_steps == 0
    assert summary.improvement_abs == 0.09999999999999998
    assert summary.shape == "accelerating_decrease"
    assert (tmp_path / "compressions" / "rae_temporal_training_curve.json").exists()


def test_training_curve_summary_reports_noisy_decrease(tmp_path):
    _write_training_log(tmp_path, [1.0, 0.95, 0.96, 0.90, 0.91, 0.89])

    summary = summarize_training_curve(tmp_path, "rae_temporal")

    assert summary.monotonic_nonincreasing is False
    assert summary.increase_steps == 2
    assert summary.max_increase is not None and summary.max_increase > 0
    assert summary.shape == "noisy_decreasing"


def test_training_status_renders_human_readable_files(tmp_path):
    path = tmp_path / "compressions" / "rae_temporal_training.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "event": "startup",
            "method": "rae_temporal",
            "epoch": None,
            "epochs": 10,
            "device": "mps",
            "temporal_matrix_shape": [2, 3, 4],
            "latent_dim": 8,
            "hidden_dim": 8,
            "replay_loss_weight": 0.0,
            "replay_loss_steps": 0,
        },
        {
            "event": "batch_heartbeat",
            "epoch": 1,
            "epochs": 10,
            "batch": 2,
            "batches": 4,
            "partial_loss": 0.9,
            "partial_loss_components": {
                "masked_temporal_reconstruction_mse": 0.9,
                "masked_temporal_cosine_distance": 0.1,
                "teacher_forced_generation_replay_kl": 0.0,
            },
            "memory_gb": 1.25,
            "elapsed_s": 12.0,
        },
        {
            "method": "rae_temporal",
            "epoch": 1,
            "epochs": 10,
            "loss": 0.8,
            "loss_components": {
                "masked_temporal_reconstruction_mse": 0.8,
                "masked_temporal_cosine_distance": 0.08,
                "teacher_forced_generation_replay_kl": 0.0,
            },
            "memory_gb": 1.3,
            "replay_gradients": False,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = render_training_status(
        tmp_path,
        "rae_temporal",
        status_path=tmp_path / "status.md",
        readable_log_path=tmp_path / "readable.log",
    )

    assert summary.last_completed_epoch == 1
    assert summary.last_completed_loss == 0.8
    assert "Last completed epoch: `1/10`" in (tmp_path / "status.md").read_text(encoding="utf-8")
    readable = (tmp_path / "readable.log").read_text(encoding="utf-8")
    assert "startup device=mps" in readable
    assert "epoch 1/10 batch 2/4 partial_loss=0.9" in readable
    assert "cosine=0.1" in readable


def test_training_status_marks_replay_active_from_heartbeat_components(tmp_path):
    path = tmp_path / "compressions" / "rae_temporal_training.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "event": "batch_heartbeat",
            "epoch": 2,
            "epochs": 10,
            "batch": 3,
            "batches": 4,
            "partial_loss": 0.5,
            "partial_loss_components": {
                "masked_temporal_reconstruction_mse": 0.49,
                "teacher_forced_generation_replay_kl": 0.2,
            },
            "memory_gb": 1.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = render_training_status(tmp_path, "rae_temporal", status_path=tmp_path / "status.md")

    assert "Replay gradients: `True`" in (tmp_path / "status.md").read_text(encoding="utf-8")
    assert summary.replay_gradients is True
