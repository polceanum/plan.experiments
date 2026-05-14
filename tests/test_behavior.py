from pathlib import Path

import torch

import latent_kv.behavior as behavior
from latent_kv.cache import CacheTuple, save_cache_bundle
from latent_kv.schemas import CacheMetadata, TrajectoryRecord, append_jsonl, read_jsonl


def _cache() -> CacheTuple:
    return ((torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)


def test_behavior_replay_defaults_to_record_generated_tokens(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "caches" / "cache.pt"
    save_cache_bundle(
        cache_path,
        _cache(),
        CacheMetadata(
            model_id="fake-model",
            tokenizer_id="fake-tokenizer",
            dtype="torch.float32",
            device="cpu",
            layers=1,
            selected_layers=[0],
            selected_heads=None,
            token_count=2,
            cache_path=str(cache_path),
        ),
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
        last_logits=torch.randn(1, 8),
    )
    append_jsonl(
        tmp_path / "records.jsonl",
        TrajectoryRecord(
            run_id=tmp_path.name,
            benchmark="gsm8k",
            task_id="gsm8k_fake",
            model_id="fake-model",
            seed=0,
            attempt_id=0,
            prompt="What is 2+2?",
            target="4",
            output_text="The answer is 4",
            parsed_answer="4",
            correct=True,
            retry_index=0,
            cache_path=str(cache_path),
            generated_tokens=7,
            prompt_tokens=2,
        ),
    )

    budgets: list[int] = []

    def fake_continue_from_bundle(**kwargs):
        budgets.append(kwargs["max_new_tokens"])
        return "The answer is 4" if kwargs["max_new_tokens"] == 7 else "The answer is 0"

    monkeypatch.setattr(behavior, "choose_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(behavior, "load_model_and_tokenizer", lambda model_id, device, local_files_only=True: (object(), object()))
    monkeypatch.setattr(behavior, "greedy_continue_from_loaded_bundle", fake_continue_from_bundle)

    payload = behavior.run_cache_behavioral_baseline(tmp_path, "original_cache")
    rows = read_jsonl(tmp_path / "behavior" / "original_cache_records.jsonl")

    assert budgets == [7]
    assert rows[0]["correct"] is True
    assert rows[0]["metadata"]["replay_max_new_tokens"] == 7
    assert rows[0]["metadata"]["source_generated_tokens"] == 7
    assert payload["extra"]["behavior_original_cache_replay_budget_source"] == "record_generated_tokens"


def test_behavior_replay_cli_budget_overrides_record(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "caches" / "cache.pt"
    save_cache_bundle(
        cache_path,
        _cache(),
        CacheMetadata(
            model_id="fake-model",
            tokenizer_id="fake-tokenizer",
            dtype="torch.float32",
            device="cpu",
            layers=1,
            selected_layers=[0],
            selected_heads=None,
            token_count=2,
            cache_path=str(cache_path),
        ),
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
        last_logits=torch.randn(1, 8),
    )
    append_jsonl(
        tmp_path / "records.jsonl",
        TrajectoryRecord(
            run_id=tmp_path.name,
            benchmark="gsm8k",
            task_id="gsm8k_fake",
            model_id="fake-model",
            seed=0,
            attempt_id=0,
            prompt="What is 2+2?",
            target="4",
            output_text="The answer is 4",
            parsed_answer="4",
            correct=True,
            retry_index=0,
            cache_path=str(cache_path),
            generated_tokens=7,
            prompt_tokens=2,
        ),
    )

    budgets: list[int] = []

    def fake_continue_from_bundle(**kwargs):
        budgets.append(kwargs["max_new_tokens"])
        return "The answer is 4" if kwargs["max_new_tokens"] == 3 else "The answer is 0"

    monkeypatch.setattr(behavior, "choose_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(behavior, "load_model_and_tokenizer", lambda model_id, device, local_files_only=True: (object(), object()))
    monkeypatch.setattr(behavior, "greedy_continue_from_loaded_bundle", fake_continue_from_bundle)

    payload = behavior.run_cache_behavioral_baseline(tmp_path, "original_cache", max_new_tokens=3)
    rows = read_jsonl(tmp_path / "behavior" / "original_cache_records.jsonl")

    assert budgets == [3]
    assert rows[0]["correct"] is True
    assert rows[0]["metadata"]["replay_max_new_tokens"] == 3
    assert payload["extra"]["behavior_original_cache_replay_budget_source"] == "cli"