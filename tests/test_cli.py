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

