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

