from pathlib import Path

from latent_kv.benchmarks import load_hanoi
from latent_kv.prompt_baselines import _majority_answer, format_prompt
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


def test_append_research_log(tmp_path: Path):
    path = tmp_path / "RESEARCH_LOG.md"
    append_research_log(path, "run", ["worked"], ["failed"], ["todo"])
    text = path.read_text(encoding="utf-8")
    assert "worked" in text
    assert "failed" in text
    assert "todo" in text

