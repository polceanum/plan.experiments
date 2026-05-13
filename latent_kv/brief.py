"""Local distilled research brief.

This is intentionally a self-contained brief rather than a raw shared-chat URL.
The repository should remain useful without network access or external context.
"""

RESEARCH_BRIEF = """# Generative Latent KV Planning Research Brief

## Core Hypothesis

Transformer KV caches are not only inference artifacts. They are internal
computational states that can encode partial plans, reasoning trajectories,
working memory, verification behaviour, constraint state, and continuation
biases. If those states can be compressed into stable latent representations,
then they can potentially be retrieved, edited, interpolated, generated, and
injected back into an LLM to improve reasoning behaviour.

## Scientific Claim

The primary claim to test is behavioural: latent KV states contain reusable
planning structure. Reconstruction loss, cosine similarity, and logit matching
are useful diagnostics, but they are secondary to task accuracy, retry
efficiency, recovery from failure, and generalization under fixed benchmarks.

## System Goal

Build a deterministic experimental loop:

1. collect reasoning/planning trajectories and KV caches,
2. compress full or partial KV states,
3. compare against fixed baselines,
4. inject reconstructed, retrieved, edited, interpolated, or generated states,
5. report behavioural outcomes in a human-inspectable way after every run.

## Baseline Discipline

The project should be judged against no-cache/default generation, original-cache
replay, random projection, PCA/SVD, learned autoencoders, nearest-neighbour
retrieval, soft-prefix conditioning, hidden-state-only variants, KV-only
variants, and standard prompting baselines such as chain-of-thought,
self-consistency, and retry/reflection where applicable.

## Later Fine-Tuning Direction

The initial prototype can inject generated planning states into a frozen LLM.
The architecture should also leave room to fine-tune the base LLM so it learns
to cooperate with a generative planning module. That later phase compares
frozen-LLM injection against jointly adapted planning-aware decoding.

## Practical Defaults

Local runs should be CPU/MPS-friendly and deterministic. Benchmark realism
should not be sacrificed: start with small fixed slices of GSM8K, Tower of
Hanoi, Sudoku/constraint-grid tasks, and an explicitly gated HumanEval adapter.
Speed is a secondary metric, not the headline claim.
"""


def get_brief() -> str:
    return RESEARCH_BRIEF

