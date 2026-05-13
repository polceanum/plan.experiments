"""Reported baseline targets and reproduction policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineTarget:
    name: str
    benchmark: str
    metric: str
    reported_value: float
    source: str
    protocol: str
    reported_model: str
    reported_prompt: str
    reported_decoding: str
    reported_split: str
    reported_sample_count: int | None = None
    local_status: str = "target_only"


TARGETS: tuple[BaselineTarget, ...] = (
    BaselineTarget(
        name="chain_of_thought",
        benchmark="gsm8k",
        metric="accuracy_percent",
        reported_value=57.0,
        source="https://arxiv.org/pdf/2201.11903",
        protocol="PaLM 540B with eight chain-of-thought exemplars on GSM8K.",
        reported_model="PaLM 540B",
        reported_prompt="eight_shot_chain_of_thought",
        reported_decoding="greedy",
        reported_split="gsm8k_test",
    ),
    BaselineTarget(
        name="self_consistency",
        benchmark="gsm8k",
        metric="accuracy_percent",
        reported_value=74.4,
        source="https://arxiv.org/abs/2203.11171",
        protocol="PaLM 540B chain-of-thought with sampled reasoning paths and majority answer.",
        reported_model="PaLM 540B",
        reported_prompt="chain_of_thought",
        reported_decoding="sampled_majority_vote",
        reported_split="gsm8k_test",
    ),
    BaselineTarget(
        name="tree_of_thoughts",
        benchmark="game24",
        metric="success_percent",
        reported_value=74.0,
        source="https://arxiv.org/pdf/2305.10601",
        protocol="GPT-4 Tree of Thoughts on Game of 24 with breadth b=5.",
        reported_model="GPT-4",
        reported_prompt="tree_of_thoughts_breadth_5",
        reported_decoding="search_breadth_5",
        reported_split="game24_reported_set",
        local_status="not_implemented",
    ),
    BaselineTarget(
        name="react_hotpotqa",
        benchmark="hotpotqa",
        metric="exact_match_percent",
        reported_value=27.4,
        source="https://arxiv.org/pdf/2210.03629",
        protocol="PaLM 540B ReAct prompting on HotpotQA.",
        reported_model="PaLM 540B",
        reported_prompt="react_hotpotqa",
        reported_decoding="reported_react_decoding",
        reported_split="hotpotqa_reported_split",
        local_status="not_implemented",
    ),
)


def target_dicts() -> list[dict[str, object]]:
    return [asdict(target) for target in TARGETS]
