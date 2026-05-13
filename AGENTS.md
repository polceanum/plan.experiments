# Agent Handoff: Generative Latent KV Planning

## Project Purpose

This repository is a research prototype for testing whether transformer KV
caches can be treated as latent planning states. The core hypothesis is that
KV caches encode more than token history: they may contain reusable planning
structure, partial reasoning trajectories, working memory, verification state,
and continuation biases.

The scientific question is behavioural:

> Do compressed, retrieved, edited, interpolated, or generated KV states improve
> reasoning/planning behaviour compared with strong fixed baselines?

Do not frame this as a pure cache-compression or speed project. Reconstruction
loss, logit similarity, and latency are diagnostics. Task accuracy, recovery,
retry efficiency, planning consistency, and generalization are the main signals.

See `docs/LATENT_PLANNING_NOTES.md` for the latent-planning design principles that
future agents should preserve.

## Non-Negotiable Constraints

- Run all Python work through the `orpheus` conda environment.
- LLMs must run locally on this machine. Do not use API keys or hosted model
  services.
- Model loading should remain local-only. The code uses
  `from_pretrained(..., local_files_only=True)` and sets offline HuggingFace
  flags when loading local models.
- GSM8K uses a built-in fallback slice unless
  `LATENT_KV_ALLOW_DATASET_DOWNLOAD=1` is set. When using the cached real GSM8K
  split, keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Distinguish local implementation checks from exact reported-protocol
  reproduction.
  Current local models are much smaller than the PaLM/GPT-4-scale systems used
  by several referenced baselines.

## Repo Map

- `latent_kv/brief.py`: distilled research brief.
- `latent_kv/benchmarks.py`: deterministic benchmark adapters and verifiers.
- `latent_kv/cache.py`: local model loading, KV capture, serialization, replay
  helpers.
- `latent_kv/compressors.py`: random projection, PCA/SVD, autoencoder, and
  retrieval compression baselines.
- `latent_kv/behavior.py`: behavioural replay of original/reconstructed caches.
- `latent_kv/prompt_baselines.py`: standard, CoT, self-consistency, and
  retry/reflection prompt baselines.
- `latent_kv/tot_baseline.py`: local Tree-of-Thought style Game of 24 baseline.
- `latent_kv/react_baseline.py`: local ReAct-style toolworld baseline.
- `latent_kv/target_checks.py`: compare runs against tracked reported targets while
  marking protocol mismatches.
- `docs/RESEARCH_LOG.md`: update this after every meaningful run.
- `docs/BASELINE_TARGETS.md`: reported targets and reproduction status.
- `docs/LATENT_PLANNING_NOTES.md`: latent-planning design notes for validators,
  novelty, diversity, filtering, and protocol discipline.
- `runs/`: ignored experiment artifacts.

## Current Known State

Tests pass:

```bash
conda run -n orpheus pytest
```

Implemented and smoke-tested:

- KV collection and replay on local HuggingFace models.
- Cache compression baselines: random projection, PCA/SVD, autoencoder,
  retrieval.
- Behavioural replay baselines: original cache plus reconstructed cache variants.
- Prompt baselines: standard, CoT, self-consistency, retry/reflection.
- Local ToT implementation check on Game of 24.
- Local ReAct implementation check on tiny toolworld.

Recent real local GSM8K result:

- Run: `runs/gsm8k_qwen_chat5`
- Model: local `Qwen/Qwen2.5-0.5B-Instruct` snapshot with chat template.
- Dataset: cached real GSM8K test split, 5 examples.
- Results:
  - standard: 2/5
  - CoT: 1/5
  - retry_reflection: 2/5
  - self_consistency with 3 samples: 2/5
- Interpretation: local implementation check only, not a PaLM 540B protocol
  reproduction. `check-targets` correctly marks protocol mismatch.

## Useful Commands

Print project brief:

```bash
conda run -n orpheus python -m latent_kv brief
```

Run tests:

```bash
conda run -n orpheus pytest
```

Run local ToT and ReAct checks:

```bash
conda run -n orpheus python -m latent_kv tot-baseline --run runs/tot_smoke --limit 6 --breadth 5
conda run -n orpheus python -m latent_kv react-baseline --run runs/react_smoke --limit 3
```

Run a real local GSM8K prompt sweep using cached data/model files:

```bash
LATENT_KV_ALLOW_DATASET_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n orpheus python -m latent_kv prompt-baseline \
  --run runs/gsm8k_qwen_chat5 \
  --benchmark gsm8k \
  --limit 5 \
  --baseline standard cot retry_reflection \
  --model-id /Users/mike/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --device auto \
  --max-new-tokens 320 \
  --seed 0
```

Check reported target status:

```bash
conda run -n orpheus python -m latent_kv check-targets --run runs/gsm8k_qwen_chat5
```

Append a research log entry:

```bash
conda run -n orpheus python -m latent_kv log \
  --title "short experiment title" \
  --worked "what worked" \
  --did-not-work "what failed or was inconclusive" \
  --todo "next concrete action"
```

## How To Continue

Best next tasks:

1. Add exact few-shot CoT prompt packs from the reported CoT protocol, instead
   of generic zero-shot CoT prompts.
2. Scale the Qwen GSM8K chat-template run from 5 examples to 20 or 50 examples,
   probably without self-consistency unless running overnight.
3. Run the same 5-example GSM8K protocol on local TinyLlama for comparison.
4. Add a local-LLM proposal/evaluator mode for Tree of Thoughts, while keeping
   the symbolic solver as a sanity oracle.
5. Add ReAct adapters for a real local environment, preferably Minigrid/BabyAI
   if available in `orpheus`.
6. Begin connecting cache latents to these benchmark baselines: collect KV
   traces for successful/failed prompt baseline attempts, compress them, and
   evaluate cache replay/editing effects.
7. Add novelty and structural-quality metrics: valid novel successes per cached
   example, nearest-neighbour distance, duplicate rate, edit distance,
   operator/tool-call distribution, and positions of verification/correction
   steps.
8. Add a candidate filtering mode for generated latents: sample several latent
   KV states, replay all locally, filter by verifier/structural metrics, and
   report both raw and filtered success rates.

## Reporting Rules

After every meaningful run:

- Inspect `runs/<run_id>/report.md`.
- Run `check-targets` when the run touches a tracked reported baseline.
- Update `docs/RESEARCH_LOG.md` using the `latent_kv log` command.
- Record protocol caveats clearly. If model/prompt/decoding/split do not match
  the reported protocol, mark it as a local check, not a reproduction.
- For generated latent/cache experiments, always report validity, novelty, and
  structural metrics alongside reconstruction and accuracy.
