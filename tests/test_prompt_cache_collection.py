from pathlib import Path

import torch

import latent_kv.prompt_cache_collection as prompt_cache_collection
from latent_kv.benchmarks import load_hanoi
from latent_kv.cache import CacheTuple, load_cache_bundle
from latent_kv.prompt_cache_collection import run_prompt_cache_collection
from latent_kv.schemas import read_json, read_jsonl


def fake_cache() -> CacheTuple:
    return (
        (torch.zeros(1, 2, 3, 2), torch.ones(1, 2, 3, 2)),
        (torch.full((1, 2, 3, 2), 2.0), torch.full((1, 2, 3, 2), 3.0)),
    )


def test_prompt_cache_collection_writes_records_and_bundles(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None
        name_or_path = "fake-tokenizer"

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    def fake_generate_local(model, tokenizer, prompt, device, max_new_tokens, seed, **kwargs):
        return load_hanoi(2)[seed].answer, 0.01, 3

    def fake_capture_prompt_cache(**kwargs):
        return (
            fake_cache(),
            [0, 1],
            3,
            None,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.randn(1, 10),
        )

    monkeypatch.setattr(prompt_cache_collection, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_cache_collection, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_cache_collection, "generate_local", fake_generate_local)
    monkeypatch.setattr(prompt_cache_collection, "capture_prompt_cache", fake_capture_prompt_cache)

    payload = run_prompt_cache_collection(
        run_dir=tmp_path,
        benchmark="hanoi",
        baseline="standard",
        model_id="fake-model",
        limit=2,
        seed=0,
        max_new_tokens=16,
        baseline_tier="custom",
    )

    rows = read_jsonl(tmp_path / "records.jsonl")
    metrics = read_json(tmp_path / "metrics.json")
    assert len(rows) == 2
    assert all(row["cache_path"] for row in rows)
    assert rows[0]["metadata"]["cache_collection"] is True
    assert rows[0]["metadata"]["prompt_protocol"] == "zero_shot_standard"
    bundle = load_cache_bundle(Path(rows[0]["cache_path"]))
    assert bundle["metadata"]["selected_layers"] == [0, 1]
    assert bundle["metadata"]["task_id"] == "hanoi_0000_2d"
    assert bundle["metadata"]["prompt_protocol"] == "zero_shot_standard"
    assert bundle["metadata"]["correct"] is True
    assert bundle["input_ids"].shape == (1, 3)
    assert payload["extra"]["prompt_cache_standard_local_model"] == "fake-model"
    assert metrics["baselines"][0]["baseline"] == "prompt_cache_standard"
