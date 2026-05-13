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
    local_status: str = "target_only"


TARGETS: tuple[BaselineTarget, ...] = (
    BaselineTarget(
        name="chain_of_thought",
        benchmark="gsm8k",
        metric="accuracy_percent",
        reported_value=57.0,
        source="https://arxiv.org/pdf/2201.11903",
        protocol="PaLM 540B with eight chain-of-thought exemplars on GSM8K.",
    ),
    BaselineTarget(
        name="self_consistency",
        benchmark="gsm8k",
        metric="accuracy_percent",
        reported_value=74.4,
        source="https://arxiv.org/abs/2203.11171",
        protocol="PaLM 540B chain-of-thought with sampled reasoning paths and majority answer.",
    ),
    BaselineTarget(
        name="tree_of_thoughts",
        benchmark="game24",
        metric="success_percent",
        reported_value=74.0,
        source="https://arxiv.org/pdf/2305.10601",
        protocol="GPT-4 Tree of Thoughts on Game of 24 with breadth b=5.",
        local_status="not_implemented",
    ),
    BaselineTarget(
        name="react_hotpotqa",
        benchmark="hotpotqa",
        metric="exact_match_percent",
        reported_value=27.4,
        source="https://arxiv.org/pdf/2210.03629",
        protocol="PaLM 540B ReAct prompting on HotpotQA.",
        local_status="not_implemented",
    ),
)


def target_dicts() -> list[dict[str, object]]:
    return [asdict(target) for target in TARGETS]
