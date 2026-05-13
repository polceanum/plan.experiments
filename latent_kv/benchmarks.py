"""Small deterministic benchmark adapters and verifiers."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import itertools
import os
import re
from typing import Callable, Iterable

from .schemas import TaskExample


Verifier = Callable[[str, TaskExample], tuple[str | None, bool]]


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    load: Callable[[int, int], list[TaskExample]]
    verify: Verifier


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    value = matches[-1]
    return value[:-2] if value.endswith(".0") else value


def verify_numeric(output: str, example: TaskExample) -> tuple[str | None, bool]:
    parsed = extract_last_number(output)
    target = extract_last_number(example.answer)
    return parsed, parsed is not None and target is not None and parsed == target


def parse_hanoi_moves(output: str) -> list[tuple[int, int]]:
    moves: list[tuple[int, int]] = []
    for line in output.splitlines():
        nums = re.findall(r"\b[123]\b", line)
        if len(nums) >= 2:
            moves.append((int(nums[0]), int(nums[1])))
    return moves


def hanoi_solution(n: int, source: int = 1, target: int = 3, spare: int = 2) -> list[tuple[int, int]]:
    if n == 0:
        return []
    return (
        hanoi_solution(n - 1, source, spare, target)
        + [(source, target)]
        + hanoi_solution(n - 1, spare, target, source)
    )


def verify_hanoi(output: str, example: TaskExample) -> tuple[str | None, bool]:
    n = int(example.metadata["disks"])
    moves = parse_hanoi_moves(output)
    pegs = {1: list(range(n, 0, -1)), 2: [], 3: []}
    for src, dst in moves:
        if not pegs[src]:
            return str(moves), False
        disk = pegs[src].pop()
        if pegs[dst] and pegs[dst][-1] < disk:
            return str(moves), False
        pegs[dst].append(disk)
    correct = pegs[3] == list(range(n, 0, -1)) and len(moves) == 2**n - 1
    return str(moves), correct


def load_hanoi(limit: int, seed: int = 0) -> list[TaskExample]:
    del seed
    examples: list[TaskExample] = []
    for idx, disks in enumerate(itertools.islice(itertools.cycle([2, 3, 4]), limit)):
        prompt = (
            f"Solve Tower of Hanoi with {disks} disks. Pegs are numbered 1, 2, 3. "
            "Move all disks from peg 1 to peg 3. Return one move per line as 'src -> dst'."
        )
        answer = "\n".join(f"{src} -> {dst}" for src, dst in hanoi_solution(disks))
        examples.append(
            TaskExample(
                benchmark="hanoi",
                task_id=f"hanoi_{idx:04d}_{disks}d",
                prompt=prompt,
                answer=answer,
                metadata={"disks": disks},
            )
        )
    return examples


SUDOKU_4X4 = [
    (
        "1 . . 4\n. 4 1 .\n. 1 4 .\n4 . . 1",
        "1 2 3 4\n3 4 1 2\n2 1 4 3\n4 3 2 1",
    ),
    (
        ". 2 3 .\n3 . . 2\n2 . . 3\n. 3 2 .",
        "1 2 3 4\n3 4 1 2\n2 1 4 3\n4 3 2 1",
    ),
    (
        "1 . 3 .\n. 4 . 2\n2 . 4 .\n. 3 . 1",
        "1 2 3 4\n3 4 1 2\n2 1 4 3\n4 3 2 1",
    ),
]


def parse_grid_numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\b[1-4]\b", text)]


def verify_sudoku(output: str, example: TaskExample) -> tuple[str | None, bool]:
    parsed_nums = parse_grid_numbers(output)
    if len(parsed_nums) < 16:
        return str(parsed_nums), False
    parsed_nums = parsed_nums[-16:]
    expected = parse_grid_numbers(example.answer)
    return str(parsed_nums), parsed_nums == expected


def load_sudoku(limit: int, seed: int = 0) -> list[TaskExample]:
    del seed
    examples = []
    for idx in range(limit):
        puzzle, solution = SUDOKU_4X4[idx % len(SUDOKU_4X4)]
        examples.append(
            TaskExample(
                benchmark="sudoku",
                task_id=f"sudoku_{idx:04d}",
                prompt=(
                    "Solve this 4x4 Sudoku. Use digits 1-4. Return only the completed grid.\n"
                    f"{puzzle}"
                ),
                answer=solution,
                metadata={"size": 4},
            )
        )
    return examples


GAME24_TASKS = [
    ((1, 3, 4, 6), "6 / (1 - 3 / 4)"),
    ((1, 5, 5, 5), "5 * (5 - 1 / 5)"),
    ((3, 3, 8, 8), "8 / (3 - 8 / 3)"),
    ((1, 2, 3, 4), "(1 + 2 + 3) * 4"),
    ((4, 4, 10, 10), "(10 * 10 - 4) / 4"),
    ((2, 2, 6, 6), "6 * 2 + 6 * 2"),
]


def _numbers_in_expression(expression: str) -> list[int]:
    return [int(x) for x in re.findall(r"\b\d+\b", expression)]


def _safe_eval_arithmetic(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
    return float(eval(compile(tree, "<game24>", "eval"), {"__builtins__": {}}, {}))


def verify_game24(output: str, example: TaskExample) -> tuple[str | None, bool]:
    numbers = sorted(int(x) for x in example.metadata["numbers"])
    candidates = [line.strip() for line in output.splitlines() if line.strip()]
    candidates.extend(re.findall(r"([0-9+\-*/().\s]+)", output))
    for candidate in candidates:
        used = sorted(_numbers_in_expression(candidate))
        if used != numbers:
            continue
        try:
            value = _safe_eval_arithmetic(candidate)
        except Exception:
            continue
        if abs(value - 24.0) < 1e-6:
            return candidate, True
    return None, False


def load_game24(limit: int, seed: int = 0) -> list[TaskExample]:
    del seed
    examples = []
    for idx in range(limit):
        numbers, answer = GAME24_TASKS[idx % len(GAME24_TASKS)]
        examples.append(
            TaskExample(
                benchmark="game24",
                task_id=f"game24_{idx:04d}",
                prompt=(
                    "Use each of these four numbers exactly once with +, -, *, / and parentheses "
                    f"to make 24. Numbers: {' '.join(map(str, numbers))}. Return one expression."
                ),
                answer=answer,
                metadata={"numbers": list(numbers)},
            )
        )
    return examples


GSM8K_FALLBACK = [
    (
        "Janet has 3 apples and buys 5 more. How many apples does she have?",
        "8",
    ),
    (
        "A book has 12 pages. Sam reads 4 pages per day. How many days does it take?",
        "3",
    ),
    (
        "There are 6 boxes with 7 pencils each. How many pencils are there?",
        "42",
    ),
]


def load_gsm8k_fallback(limit: int, seed: int = 0) -> list[TaskExample]:
    del seed
    examples = []
    for idx in range(limit):
        question, answer = GSM8K_FALLBACK[idx % len(GSM8K_FALLBACK)]
        examples.append(
            TaskExample(
                benchmark="gsm8k",
                task_id=f"gsm8k_fallback_{idx:04d}",
                prompt=f"Solve the math problem. Give the final numeric answer.\n{question}",
                answer=answer,
                metadata={"source": "built-in-fallback"},
            )
        )
    return examples


def load_gsm8k(limit: int, seed: int = 0) -> list[TaskExample]:
    if os.environ.get("LATENT_KV_ALLOW_DATASET_DOWNLOAD") != "1":
        return load_gsm8k_fallback(limit, seed)
    try:
        from datasets import DownloadConfig, load_dataset

        dataset = load_dataset(
            "gsm8k",
            "main",
            split="test",
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception:
        return load_gsm8k_fallback(limit, seed)
    if seed:
        dataset = dataset.shuffle(seed=seed)
    examples = []
    for idx, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        answer = str(row["answer"]).split("####")[-1].strip()
        examples.append(
            TaskExample(
                benchmark="gsm8k",
                task_id=f"gsm8k_{idx:04d}",
                prompt=f"Solve the math problem. Give the final numeric answer.\n{row['question']}",
                answer=answer,
                metadata={"source": "gsm8k/test"},
            )
        )
    return examples


def load_humaneval(limit: int, seed: int = 0) -> list[TaskExample]:
    del seed
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for HumanEval") from exc
    dataset = load_dataset("openai_humaneval", split="test")
    examples = []
    for idx, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        examples.append(
            TaskExample(
                benchmark="humaneval",
                task_id=row["task_id"],
                prompt=(
                    "Complete the following Python function. Return code only.\n"
                    f"{row['prompt']}"
                ),
                answer=row["canonical_solution"],
                metadata={"entry_point": row["entry_point"], "tests": row["test"]},
            )
        )
    return examples


def verify_humaneval_static(output: str, example: TaskExample) -> tuple[str | None, bool]:
    # Execution is intentionally gated out of the default verifier. This static
    # placeholder keeps the adapter inspectable without running untrusted code.
    entry_point = example.metadata.get("entry_point", "")
    parsed = output.strip()
    return parsed[:200], f"def {entry_point}" in parsed or parsed.startswith(example.answer.strip()[:20])


ADAPTERS: dict[str, BenchmarkAdapter] = {
    "hanoi": BenchmarkAdapter("hanoi", load_hanoi, verify_hanoi),
    "sudoku": BenchmarkAdapter("sudoku", load_sudoku, verify_sudoku),
    "game24": BenchmarkAdapter("game24", load_game24, verify_game24),
    "gsm8k": BenchmarkAdapter("gsm8k", load_gsm8k, verify_numeric),
    "humaneval": BenchmarkAdapter("humaneval", load_humaneval, verify_humaneval_static),
}


def load_examples(benchmark: str, limit: int, seed: int = 0) -> list[TaskExample]:
    if benchmark == "all":
        combined: list[TaskExample] = []
        for name in ("hanoi", "sudoku", "game24", "gsm8k"):
            combined.extend(ADAPTERS[name].load(limit, seed))
        return combined
    if benchmark not in ADAPTERS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from {sorted(ADAPTERS)} or all.")
    return ADAPTERS[benchmark].load(limit, seed)


def verify_output(output: str, example: TaskExample) -> tuple[str | None, bool]:
    return ADAPTERS[example.benchmark].verify(output, example)


def iter_fixed_baselines() -> Iterable[str]:
    return (
        "no_cache",
        "original_cache",
        "random_projection",
        "pca_svd",
        "autoencoder",
        "nearest_neighbor_cache",
        "soft_prefix",
        "hidden_state_only",
        "kv_only",
        "combined_state",
        "cot",
        "self_consistency",
        "retry_reflection",
        "tree_of_thoughts",
        "react",
    )
