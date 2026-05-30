import json
from pathlib import Path

from latent_kv.cli import main


def test_brief_cli_outputs_local_brief(capsys):
    assert main(["brief"]) == 0
    out = capsys.readouterr().out
    assert "Generative Latent KV Planning Research Brief" in out
    assert "shared" not in out.lower()


def test_targets_cli_outputs_reported_targets(capsys):
    assert main(["targets"]) == 0
    out = capsys.readouterr().out
    assert "self_consistency" in out
    assert "reported_value" in out


def test_prompt_baseline_help_lists_tiers(capsys):
    try:
        main(["prompt-baseline", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--baseline-tier" in out
    assert "--resume" in out
    assert "--chunk-size" in out
    assert "smoke" in out
    assert "full" in out


def test_collect_prompt_caches_help_lists_config(capsys):
    try:
        main(["collect-prompt-caches", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--cache-mode" in out
    assert "--layer-mode" in out
    assert "--resume" in out


def test_validate_codec_help_lists_method(capsys):
    try:
        main(["validate-codec", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--method" in out
    assert "rae_temporal" in out
    assert "retrieval" in out


def test_replay_fidelity_help_lists_method(capsys):
    try:
        main(["replay-fidelity", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--method" in out
    assert "--limit" in out
    assert "--steps" in out
    assert "rae_temporal" in out


def test_training_curve_help_lists_method(capsys):
    try:
        main(["training-curve", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--method" in out
    assert "rae_temporal" in out
    assert "autoencoder" in out


def test_latent_analysis_help_lists_outputs(capsys):
    try:
        main(["latent-analysis", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--checkpoint" in out
    assert "--output-dir" in out
    assert "--batch-size" in out
    assert "--progress-every-batches" in out
    assert "rae_temporal" in out


def test_latent_interpolate_help_lists_replay_controls(capsys):
    try:
        main(["latent-interpolate", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--analysis-dir" in out
    assert "--alphas" in out
    assert "--pair-mode" in out
    assert "--replay-device" in out
    assert "--max-new-tokens" in out
    assert "--selection" in out
    assert "--min-distance" in out
    assert "--max-distance" in out
    assert "--reconstruction-scan" in out
    assert "--require-convincing-reconstruction" in out


def test_latent_reconstruction_scan_help_lists_controls(capsys):
    try:
        main(["latent-reconstruction-scan", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--analysis-dir" in out
    assert "--replay-device" in out
    assert "--max-new-tokens" in out
    assert "--limit" in out


def test_latent_prompt_decoder_dataset_help_lists_controls(capsys):
    try:
        main(["latent-prompt-decoder-dataset", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--analysis-dir" in out
    assert "--output-dir" in out


def test_latent_prompt_decoder_train_help_lists_controls(capsys):
    try:
        main(["latent-prompt-decoder-train", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--dataset" in out
    assert "--epochs" in out
    assert "--hidden-dim" in out
    assert "--max-prompt-tokens" in out
    assert "--max-latent-chunks" in out
    assert "--num-layers" in out
    assert "--progress-every-batches" in out


def test_corruption_sensitivity_help_lists_alpha(capsys):
    try:
        main(["corruption-sensitivity", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--alpha" in out
    assert "--method" in out
    assert "rae_temporal" in out


def test_compress_help_lists_lstm_hyperparameters(capsys):
    try:
        main(["compress", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--hidden-dim" in out
    assert "--num-layers" in out
    assert "--llm-loss-weight" not in out
    assert "--llm-steps" not in out
    assert "--replay-loss-weight" in out
    assert "--replay-loss-steps" in out
    assert "--replay-loss-every-n-batches" in out
    assert "--cosine-loss-weight" in out
    assert "--prompt-loss-weight" in out
    assert "--prompt-loss-max-tokens" in out
    assert "rae_temporal" in out
    assert "--weight-decay" in out
    assert "--log-every" in out
    assert "--checkpoint-every" in out
    assert "--heartbeat-every-batches" in out
    assert "--train-batch-size" in out
    assert "--resume-checkpoint" in out
    assert "--grad-clip-norm" in out
    assert "--mps-empty-cache-every-batches" in out
    assert "--temporal-num-heads" in out
    assert "--temporal-latent-tokens" in out
    assert "--temporal-decoder-memory-tokens" in out
    assert "rae_temporal_transformer" in out


def test_compress_autoresume_help_lists_restart_controls(capsys):
    try:
        main(["compress-autoresume", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--max-restarts" in out
    assert "--restart-delay-s" in out
    assert "latest numbered checkpoint" in out


def test_compress_autoresume_uses_latest_numbered_checkpoint(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "compressions" / "rae_temporal_transformer_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "rae_temporal_transformer_epoch_000010.pt").write_text("old", encoding="utf-8")
    (checkpoint_dir / "rae_temporal_transformer_epoch_000025.pt").write_text("new", encoding="utf-8")
    (checkpoint_dir / "rae_temporal_transformer_latest.pt").write_text("alias", encoding="utf-8")
    calls = []

    class Completed:
        returncode = 0

    def fake_run(command):
        calls.append(command)
        return Completed()

    monkeypatch.setattr("latent_kv.cli.subprocess.run", fake_run)

    assert main(
        [
            "compress-autoresume",
            "--restart-delay-s",
            "0",
            "--",
            "--run",
            str(tmp_path),
            "--method",
            "rae_temporal_transformer",
            "--epochs",
            "100",
            "--resume-checkpoint",
            "stale.pt",
        ]
    ) == 0

    command = calls[0]
    assert command[:3][-2:] == ["-m", "latent_kv"]
    assert "--resume-checkpoint" in command
    resume_index = command.index("--resume-checkpoint")
    assert command[resume_index + 1].endswith("rae_temporal_transformer_epoch_000025.pt")
    assert "stale.pt" not in command
    events = [
        json.loads(line)
        for line in (tmp_path / "run_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "compress_autoresume_attempt_start" for event in events)


def test_attach_prompt_caches_help_lists_source_records(capsys):
    try:
        main(["attach-prompt-caches", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--source-records" in out
    assert "--cache-mode" in out
    assert "--limit" in out
    assert "--layer-mode" in out
    assert "--resume" in out
