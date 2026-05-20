import torch

from latent_kv.injection import greedy_continue_from_loaded_bundle


class TinyTokenizer:
    eos_token_id = None

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


class PositionalReplayModel:
    def __init__(self):
        self.calls = []
        self.generation_config = type("GenerationConfig", (), {"repetition_penalty": 1.0})()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        vocab = torch.zeros(1, 1, 8)
        vocab[:, :, 3] = 1.0
        return type("Out", (), {"past_key_values": kwargs["past_key_values"], "logits": vocab})()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        position_ids=None,
        cache_position=None,
    ):
        raise AssertionError("Signature only; __call__ is used in this test")


def test_greedy_continue_passes_position_information_when_supported():
    model = PositionalReplayModel()
    cache = ((torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)),)
    bundle = {
        "cache": cache,
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "last_logits": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    }

    output = greedy_continue_from_loaded_bundle(
        bundle=bundle,
        model=model,
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        max_new_tokens=2,
    )

    assert output == "2 3"
    assert model.calls[0]["position_ids"].tolist() == [[2]]
    assert model.calls[0]["cache_position"].tolist() == [2]
    assert model.calls[1]["position_ids"].tolist() == [[3]]
    assert model.calls[1]["cache_position"].tolist() == [3]


def test_greedy_continue_applies_generation_repetition_penalty():
    class RepetitionPenaltyModel(PositionalReplayModel):
        def __init__(self):
            super().__init__()
            self.generation_config = type("GenerationConfig", (), {"repetition_penalty": 2.0})()

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            vocab = torch.zeros(1, 1, 8)
            vocab[:, :, 3] = 1.0
            return type("Out", (), {"past_key_values": kwargs["past_key_values"], "logits": vocab})()

    model = RepetitionPenaltyModel()
    cache = ((torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),)
    bundle = {
        "cache": cache,
        "input_ids": torch.tensor([[2]]),
        "attention_mask": torch.tensor([[1]]),
        "last_logits": torch.tensor([[0.0, 0.0, 10.0, 6.0]]),
    }

    output = greedy_continue_from_loaded_bundle(
        bundle=bundle,
        model=model,
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        max_new_tokens=1,
    )

    assert output == "3"


def test_greedy_continue_replays_trajectory_cache_from_prompt_boundary():
    class TrajectoryReplayModel(PositionalReplayModel):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            vocab = torch.zeros(1, 1, 16)
            vocab[:, :, 6 if len(self.calls) == 1 else 7] = 1.0
            key, value = kwargs["past_key_values"][0]
            next_past = ((torch.cat([key, key[..., -1:, :]], dim=-2), torch.cat([value, value[..., -1:, :]], dim=-2)),)
            return type("Out", (), {"past_key_values": next_past, "logits": vocab})()

    model = TrajectoryReplayModel()
    full_cache = ((torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2)),)
    bundle = {
        "cache": full_cache,
        "input_ids": torch.tensor([[10, 11, 12, 6, 7]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "last_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0]]),
        "generation_config": {
            "cache_mode": "trajectory",
            "prompt_tokens": 3,
            "prompt_input_ids": [[10, 11, 12]],
        },
    }

    output = greedy_continue_from_loaded_bundle(
        bundle=bundle,
        model=model,
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        max_new_tokens=2,
    )

    assert output == "6 7"
    assert model.calls[0]["past_key_values"][0][0].shape[-2] == 2
    assert model.calls[0]["attention_mask"].shape[-1] == 3
    assert model.calls[0]["position_ids"].tolist() == [[2]]
    assert model.calls[1]["past_key_values"][0][0].shape[-2] == 3
    assert model.calls[1]["attention_mask"].shape[-1] == 4
    assert model.calls[1]["position_ids"].tolist() == [[3]]
