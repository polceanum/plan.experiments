from pathlib import Path

import torch

import latent_kv.prompt_cache_collection as prompt_cache_collection
from latent_kv.benchmarks import load_hanoi
from latent_kv.cache import CacheTuple, load_cache_bundle
from latent_kv.prompt_cache_collection import run_prompt_cache_collection
from latent_kv.prompt_cache_collection import run_existing_prompt_record_cache_collection
from latent_kv.schemas import read_json, read_jsonl
from latent_kv.schemas import TrajectoryRecord, append_jsonl


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

    def fake_generate_local_with_ids(model, tokenizer, prompt, device, max_new_tokens, seed, **kwargs):
        return load_hanoi(2)[seed].answer, 0.01, 3, torch.tensor([[4, 5, 6]])

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
    monkeypatch.setattr(prompt_cache_collection, "generate_local_with_ids", fake_generate_local_with_ids)
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
    assert bundle["generation_token_ids"].tolist() == [[4, 5, 6]]
    assert bundle["generation_config"]["max_new_tokens"] == 16
    assert bundle["generation_config"]["decoding_protocol"] == "greedy"
    assert payload["extra"]["prompt_cache_standard_local_model"] == "fake-model"
    assert metrics["baselines"][0]["baseline"] == "prompt_cache_standard"


def test_prompt_cache_collection_can_capture_full_trajectory(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None
        name_or_path = "fake-tokenizer"

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    def fake_generate_local_with_ids(model, tokenizer, prompt, device, max_new_tokens, seed, **kwargs):
        return load_hanoi(1)[seed].answer, 0.01, 2, torch.tensor([[4, 5]])

    def fake_capture_full_trajectory_cache(**kwargs):
        assert kwargs["generation_token_ids"].tolist() == [[4, 5]]
        return (
            fake_cache(),
            [0, 1],
            3,
            2,
            5,
            None,
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1, 1]]),
            torch.randn(1, 10),
            torch.tensor([[1, 2, 3]]),
        )

    monkeypatch.setattr(prompt_cache_collection, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_cache_collection, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_cache_collection, "generate_local_with_ids", fake_generate_local_with_ids)
    monkeypatch.setattr(prompt_cache_collection, "capture_full_trajectory_cache", fake_capture_full_trajectory_cache)

    run_prompt_cache_collection(
        run_dir=tmp_path,
        benchmark="hanoi",
        baseline="standard",
        model_id="fake-model",
        limit=1,
        seed=0,
        max_new_tokens=16,
        cache_mode="trajectory",
    )

    rows = read_jsonl(tmp_path / "records.jsonl")
    bundle = load_cache_bundle(Path(rows[0]["cache_path"]))
    assert rows[0]["metadata"]["cache_mode"] == "trajectory"
    assert rows[0]["metadata"]["cache_total_tokens"] == 5
    assert rows[0]["metadata"]["cache_generated_tokens"] == 2
    assert bundle["metadata"]["token_count"] == 5
    assert bundle["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert bundle["generation_config"]["cache_mode"] == "trajectory"
    assert bundle["generation_config"]["prompt_tokens"] == 3


def test_existing_prompt_record_cache_collection_preserves_labels(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None
        name_or_path = "fake-tokenizer"

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

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
    monkeypatch.setattr(prompt_cache_collection, "capture_prompt_cache", fake_capture_prompt_cache)

    source = tmp_path / "source.jsonl"
    append_jsonl(
        source,
        TrajectoryRecord(
            run_id="source",
            benchmark="gsm8k",
            task_id="gsm8k_0000",
            model_id="fake-model",
            seed=0,
            attempt_id=0,
            prompt="prompt 0",
            target="18",
            output_text="answer 18",
            parsed_answer="18",
            correct=True,
            retry_index=0,
            generated_tokens=12,
            prompt_tokens=2,
            metadata={"prompt_baseline": "cot", "prompt_protocol": "zero_shot_cot", "decoding_protocol": "greedy"},
        ),
    )
    append_jsonl(
        source,
        TrajectoryRecord(
            run_id="source",
            benchmark="gsm8k",
            task_id="gsm8k_0001",
            model_id="fake-model",
            seed=0,
            attempt_id=1,
            prompt="prompt 1",
            target="20",
            output_text="answer 19",
            parsed_answer="19",
            correct=False,
            retry_index=0,
            generated_tokens=10,
            prompt_tokens=2,
            metadata={"prompt_baseline": "cot", "prompt_protocol": "zero_shot_cot", "decoding_protocol": "greedy"},
        ),
    )

    run_dir = tmp_path / "attached"
    payload = run_existing_prompt_record_cache_collection(
        run_dir=run_dir,
        source_records=source,
        model_id="fake-model",
    )

    rows = read_jsonl(run_dir / "records.jsonl")
    assert [row["correct"] for row in rows] == [True, False]
    assert all(row["cache_path"] for row in rows)
    assert rows[0]["output_text"] == "answer 18"
    assert rows[1]["parsed_answer"] == "19"
    assert rows[0]["metadata"]["cache_attached_from_existing_record"] is True
    bundle = load_cache_bundle(Path(rows[1]["cache_path"]))
    assert bundle["metadata"]["correct"] is False
    assert bundle["metadata"]["prompt_baseline"] == "cot"
    assert bundle["generation_config"]["attached_existing_output"] is True
    assert payload["extra"]["prompt_cache_attached_source_total"] == 2
    assert payload["extra"]["prompt_cache_attached_source_correct"] == 1
    assert payload["extra"]["prompt_cache_attached_source_incorrect"] == 1


def test_existing_prompt_record_cache_collection_can_recapture_trajectory(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None
        name_or_path = "fake-tokenizer"

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    source_cache_path = tmp_path / "source_cache.pt"
    save_cache = fake_cache()
    from latent_kv.cache import CacheMetadata, save_cache_bundle

    save_cache_bundle(
        source_cache_path,
        save_cache,
        CacheMetadata(
            model_id="fake-model",
            tokenizer_id="fake-tokenizer",
            dtype="float32",
            device="cpu",
            layers=2,
            selected_layers=[0, 1],
            selected_heads=None,
            token_count=3,
            cache_path=str(source_cache_path),
        ),
        generation_token_ids=torch.tensor([[8, 9]]),
    )

    def fake_capture_full_trajectory_cache(**kwargs):
        assert kwargs["generation_token_ids"].tolist() == [[8, 9]]
        return (
            fake_cache(),
            [0, 1],
            3,
            2,
            5,
            None,
            torch.tensor([[1, 2, 3, 8, 9]]),
            torch.tensor([[1, 1, 1, 1, 1]]),
            torch.randn(1, 10),
            torch.tensor([[1, 2, 3]]),
        )

    monkeypatch.setattr(prompt_cache_collection, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_cache_collection, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_cache_collection, "capture_full_trajectory_cache", fake_capture_full_trajectory_cache)

    source = tmp_path / "source.jsonl"
    append_jsonl(
        source,
        TrajectoryRecord(
            run_id="source",
            benchmark="gsm8k",
            task_id="gsm8k_0000",
            model_id="fake-model",
            seed=0,
            attempt_id=0,
            prompt="prompt 0",
            target="18",
            output_text="answer 18",
            parsed_answer="18",
            correct=True,
            retry_index=0,
            cache_path=str(source_cache_path),
            generated_tokens=2,
            prompt_tokens=3,
            metadata={"prompt_baseline": "cot", "prompt_protocol": "zero_shot_cot"},
        ),
    )

    run_dir = tmp_path / "attached_trajectory"
    run_existing_prompt_record_cache_collection(
        run_dir=run_dir,
        source_records=source,
        model_id="fake-model",
        cache_mode="trajectory",
    )

    rows = read_jsonl(run_dir / "records.jsonl")
    bundle = load_cache_bundle(Path(rows[0]["cache_path"]))
    assert rows[0]["metadata"]["cache_mode"] == "trajectory"
    assert bundle["input_ids"].tolist() == [[1, 2, 3, 8, 9]]
    assert bundle["generation_token_ids"].tolist() == [[8, 9]]
    assert bundle["generation_config"]["cache_generated_tokens"] == 2


def test_existing_trajectory_recapture_can_tokenize_output_text_when_ids_missing(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None
        name_or_path = "fake-tokenizer"

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            assert text == "answer 18"
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[18, 19]])}

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    source_cache_path = tmp_path / "source_cache_no_ids.pt"
    from latent_kv.cache import CacheMetadata, save_cache_bundle

    save_cache_bundle(
        source_cache_path,
        fake_cache(),
        CacheMetadata(
            model_id="fake-model",
            tokenizer_id="fake-tokenizer",
            dtype="float32",
            device="cpu",
            layers=2,
            selected_layers=[0, 1],
            selected_heads=None,
            token_count=3,
            cache_path=str(source_cache_path),
        ),
    )

    def fake_capture_full_trajectory_cache(**kwargs):
        assert kwargs["generation_token_ids"].tolist() == [[18, 19]]
        return (
            fake_cache(),
            [0, 1],
            3,
            2,
            5,
            None,
            torch.tensor([[1, 2, 3, 18, 19]]),
            torch.tensor([[1, 1, 1, 1, 1]]),
            torch.randn(1, 10),
            torch.tensor([[1, 2, 3]]),
        )

    monkeypatch.setattr(prompt_cache_collection, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_cache_collection, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_cache_collection, "capture_full_trajectory_cache", fake_capture_full_trajectory_cache)

    source = tmp_path / "source.jsonl"
    append_jsonl(
        source,
        TrajectoryRecord(
            run_id="source",
            benchmark="gsm8k",
            task_id="gsm8k_0000",
            model_id="fake-model",
            seed=0,
            attempt_id=0,
            prompt="prompt 0",
            target="18",
            output_text="answer 18",
            parsed_answer="18",
            correct=True,
            retry_index=0,
            cache_path=str(source_cache_path),
            generated_tokens=2,
            prompt_tokens=3,
            metadata={"prompt_baseline": "cot", "prompt_protocol": "zero_shot_cot"},
        ),
    )

    run_dir = tmp_path / "attached_trajectory_fallback"
    run_existing_prompt_record_cache_collection(
        run_dir=run_dir,
        source_records=source,
        model_id="fake-model",
        cache_mode="trajectory",
    )

    rows = read_jsonl(run_dir / "records.jsonl")
    bundle = load_cache_bundle(Path(rows[0]["cache_path"]))
    assert bundle["generation_token_ids"].tolist() == [[18, 19]]
    assert bundle["generation_config"]["generation_token_source"] == "tokenized_output_text"
