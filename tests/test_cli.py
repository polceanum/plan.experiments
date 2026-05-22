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
