import pytest

from latent_kv.benchmarks import (
    extract_last_number,
    load_examples,
    load_game24,
    load_humaneval,
    load_hanoi,
    load_sudoku,
    verify_game24,
    verify_hanoi,
    verify_numeric,
    verify_sudoku,
)
from latent_kv.schemas import TaskExample


def test_extract_last_number_normalizes_commas_and_float_suffix():
    assert extract_last_number("The answer is 1,234.0") == "1234"


def test_numeric_verifier():
    example = TaskExample("gsm8k", "x", "prompt", "#### 42")
    parsed, correct = verify_numeric("After working it out, final answer: 42", example)
    assert parsed == "42"
    assert correct


def test_hanoi_solution_verifier_accepts_adapter_answer():
    example = load_hanoi(1)[0]
    parsed, correct = verify_hanoi(example.answer, example)
    assert parsed is not None
    assert correct


def test_sudoku_verifier_accepts_adapter_answer():
    example = load_sudoku(1)[0]
    parsed, correct = verify_sudoku(example.answer, example)
    assert parsed is not None
    assert correct


def test_game24_verifier_accepts_adapter_answer():
    example = load_game24(1)[0]
    parsed, correct = verify_game24(example.answer, example)
    assert parsed is not None
    assert correct


def test_all_loader_uses_fixed_local_tasks_without_humaneval():
    examples = load_examples("all", limit=1, seed=0)
    assert {example.benchmark for example in examples} >= {"hanoi", "sudoku", "game24"}


def test_humaneval_requires_explicit_enable_flag(monkeypatch):
    monkeypatch.delenv("LATENT_KV_ENABLE_HUMANEVAL", raising=False)
    with pytest.raises(RuntimeError, match="HumanEval is disabled"):
        load_humaneval(1)
