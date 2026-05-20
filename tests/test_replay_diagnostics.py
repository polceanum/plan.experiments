from types import SimpleNamespace

import torch

from latent_kv.replay_diagnostics import _first_replay_token_id, _logit_comparison, _teacher_forced_replay_logits, _token_rank


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


def test_teacher_forced_replay_logits_include_prompt_boundary_distribution():
    class PrefixModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = []

        def forward(
            self,
            input_ids,
            attention_mask,
            past_key_values,
            position_ids=None,
            cache_position=None,
            use_cache=True,
            return_dict=True,
        ):
            del input_ids, use_cache, return_dict
            self.seen.append(
                (
                    int(past_key_values[0][0].shape[-2]),
                    int(attention_mask.shape[-1]),
                    int(position_ids.reshape(-1)[0].item()) if position_ids is not None else int(cache_position.reshape(-1)[0].item()),
                )
            )
            next_past = tuple(
                (
                    torch.cat([key, key[..., -1:, :]], dim=-2),
                    torch.cat([value, value[..., -1:, :]], dim=-2),
                )
                for key, value in past_key_values
            )
            return SimpleNamespace(logits=torch.ones((1, 1, 3)), past_key_values=next_past)

    model = PrefixModel()
    bundle = {
        "input_ids": torch.arange(7, dtype=torch.long).reshape(1, 7),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "generation_config": {"cache_mode": "trajectory", "prompt_tokens": 3},
    }
    cache = ((torch.ones((1, 1, 7, 2)), torch.ones((1, 1, 7, 2))),)

    logits = _teacher_forced_replay_logits(bundle, cache, torch.tensor([10, 11]), model, torch.device("cpu"))

    assert len(logits) == 2
    assert model.seen == [(2, 3, 2), (3, 4, 3)]
