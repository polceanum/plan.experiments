from pathlib import Path

from latent_kv.benchmarks import load_hanoi
from latent_kv.prompt_baselines import _majority_answer, format_prompt, prompt_protocol_metadata, resolve_baseline_tier
from latent_kv.research_log import append_research_log


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


def test_append_research_log(tmp_path: Path):
    path = tmp_path / "RESEARCH_LOG.md"
    append_research_log(path, "run", ["worked"], ["failed"], ["todo"])
    text = path.read_text(encoding="utf-8")
    assert "worked" in text
    assert "failed" in text
    assert "todo" in text

