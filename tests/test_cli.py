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
    assert "smoke" in out
    assert "full" in out


def test_collect_prompt_caches_help_lists_config(capsys):
    try:
        main(["collect-prompt-caches", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
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
    assert "--llm-loss-weight" in out
    assert "--llm-steps" in out
    assert "rae_temporal" in out
    assert "--weight-decay" in out
    assert "--log-every" in out
    assert "--checkpoint-every" in out

