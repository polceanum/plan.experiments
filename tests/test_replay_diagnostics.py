import torch

from latent_kv.replay_diagnostics import _first_replay_token_id, _logit_comparison, _token_rank


class _GenerationConfig:
    repetition_penalty = 1.0


class _Model:
    generation_config = _GenerationConfig()


def test_logit_comparison_reports_top1_and_distribution_drift():
    original = torch.tensor([[0.1, 2.0, -1.0]])
    reconstructed = torch.tensor([[0.2, 1.5, -0.5]])

    comparison = _logit_comparison(original, reconstructed)

    assert comparison["original_top1"] == 1
    assert comparison["reconstructed_top1"] == 1
    assert comparison["top1_match"] is True
    assert comparison["logit_mse"] > 0
    assert comparison["kl_original_to_reconstructed"] >= 0


def test_first_replay_token_prefers_stored_generation_ids():
    bundle = {
        "generation_token_ids": torch.tensor([[7, 8]]),
        "input_ids": torch.tensor([[1, 2]]),
        "last_logits": torch.tensor([[0.0, 10.0, 0.0]]),
    }

    assert _first_replay_token_id(bundle, _Model()) == 7


def test_first_replay_token_falls_back_to_last_logits():
    bundle = {
        "input_ids": torch.tensor([[1, 2]]),
        "last_logits": torch.tensor([[0.0, 10.0, 0.0]]),
    }

    assert _first_replay_token_id(bundle, _Model()) == 1


def test_token_rank_is_one_indexed():
    logits = torch.tensor([[0.2, 4.0, 1.0]])

    assert _token_rank(logits, 1) == 1
    assert _token_rank(logits, 2) == 2