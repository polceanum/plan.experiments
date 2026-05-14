from pathlib import Path

import latent_kv.prompt_baselines as prompt_baselines
from latent_kv.benchmarks import load_hanoi
from latent_kv.prompt_baselines import (
    _majority_answer,
    format_prompt,
    prompt_protocol_metadata,
    resolve_baseline_tier,
    run_prompt_baseline,
)
from latent_kv.research_log import append_research_log
from latent_kv.schemas import read_json, read_jsonl


def test_cot_prompt_includes_step_by_step_instruction():
    example = load_hanoi(1)[0]
    prompt = format_prompt(example, "cot")
    assert "Think step by step" in prompt
    assert example.prompt in prompt


def test_self_consistency_majority_answer():
    parsed, text = _majority_answer([("1", "a"), ("2", "b"), ("1", "c")])
    assert parsed == "1"
    assert text == "a"


def test_prompt_protocol_metadata_records_budget_and_split():
    examples = load_hanoi(2)
    metadata = prompt_protocol_metadata(
        baseline="self_consistency",
        model_id="local-model",
        benchmark="hanoi",
        examples=examples,
        limit=2,
        seed=7,
        max_new_tokens=128,
        device_name="mps",
        samples=3,
        temperature=0.8,
        chat_template=True,
        baseline_tier="working",
    )
    assert metadata["protocol_name"] == "zero_shot_cot_self_consistency"
    assert metadata["decoding_protocol"] == "sampled_majority_vote"
    assert metadata["sample_count"] == 3
    assert metadata["benchmark_split"] == "hanoi"
    assert metadata["baseline_tier"] == "working"
    assert metadata["chat_template"] is True
    assert metadata["protocol_match"] is False


def test_resolve_baseline_tier_presets_and_overrides():
    assert resolve_baseline_tier("smoke") == ("smoke", 5, 320)
    assert resolve_baseline_tier("working") == ("working", 20, 320)
    assert resolve_baseline_tier("full", limit=7, max_new_tokens=64) == ("full", 7, 64)


def test_prompt_baseline_streams_records_and_keeps_failures(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    def fake_generate_local(model, tokenizer, prompt, device, max_new_tokens, seed, **kwargs):
        if seed == 0:
            raise RuntimeError("simulated generation failure")
        return load_hanoi(2)[1].answer, 0.01, 3

    monkeypatch.setattr(prompt_baselines, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_baselines, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_baselines, "generate_local", fake_generate_local)

    payload = run_prompt_baseline(
        run_dir=tmp_path,
        benchmark="hanoi",
        baseline="standard",
        model_id="fake-model",
        limit=2,
        seed=0,
        max_new_tokens=16,
        baseline_tier="smoke",
    )

    rows = read_jsonl(tmp_path / "behavior" / "standard_records.jsonl")
    metrics = read_json(tmp_path / "metrics.json")
    assert len(rows) == 2
    assert rows[0]["metadata"]["generation_error"] == "RuntimeError: simulated generation failure"
    assert rows[1]["metadata"]["generation_error"] is None
    assert payload["extra"]["prompt_standard_completed_examples"] == 2
    assert metrics["extra"]["prompt_standard_failed_examples"] == 1


def test_prompt_baseline_resume_skips_existing_records(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        chat_template = None

    calls: list[int] = []

    def fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
        return object(), FakeTokenizer()

    def fake_generate_local(model, tokenizer, prompt, device, max_new_tokens, seed, **kwargs):
        calls.append(seed)
        return load_hanoi(2)[seed].answer, 0.01, 3

    monkeypatch.setattr(prompt_baselines, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(prompt_baselines, "choose_device", lambda device_name: "cpu")
    monkeypatch.setattr(prompt_baselines, "generate_local", fake_generate_local)

    run_prompt_baseline(
        run_dir=tmp_path,
        benchmark="hanoi",
        baseline="standard",
        model_id="fake-model",
        limit=1,
        seed=0,
        max_new_tokens=16,
        baseline_tier="custom",
    )
    run_prompt_baseline(
        run_dir=tmp_path,
        benchmark="hanoi",
        baseline="standard",
        model_id="fake-model",
        limit=2,
        seed=0,
        max_new_tokens=16,
        baseline_tier="custom",
        resume=True,
    )

    rows = read_jsonl(tmp_path / "behavior" / "standard_records.jsonl")
    metrics = read_json(tmp_path / "metrics.json")
    assert calls == [0, 1]
    assert [row["task_id"] for row in rows] == ["hanoi_0000_2d", "hanoi_0001_3d"]
    assert metrics["extra"]["prompt_standard_completed_examples"] == 2
    assert metrics["extra"]["prompt_standard_resume"] is True


def test_append_research_log(tmp_path: Path):
    path = tmp_path / "RESEARCH_LOG.md"
    append_research_log(path, "run", ["worked"], ["failed"], ["todo"])
    text = path.read_text(encoding="utf-8")
    assert "worked" in text
    assert "failed" in text
    assert "todo" in text

