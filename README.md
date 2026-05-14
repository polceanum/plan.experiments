# Generative Latent KV Planning

This repository is a research prototype for treating transformer KV caches as
latent planning states. The goal is not just to compress caches, but to test
whether compressed, retrieved, edited, interpolated, or generated internal
states improve downstream behaviour on fixed reasoning and planning tasks.

Fresh agents should start with [AGENTS.md](AGENTS.md). It contains the project
purpose, local-only constraints, current state, and next recommended tasks.
The latent-planning design principles are summarized in
[docs/LATENT_PLANNING_NOTES.md](docs/LATENT_PLANNING_NOTES.md).

All development and experiment commands should run in the `orpheus` conda
environment:

```bash
conda run -n orpheus python -m latent_kv brief
conda run -n orpheus pytest
```

Model execution is local-only by default. The code calls HuggingFace
`from_pretrained(..., local_files_only=True)`, so it will use weights already
present on this computer and will fail clearly instead of contacting a remote
API or service.

Dataset loading is also conservative by default for smoke runs. GSM8K uses a
small built-in fallback slice unless `LATENT_KV_ALLOW_DATASET_DOWNLOAD=1` is set
in the environment.
HumanEval is disabled unless `LATENT_KV_ENABLE_HUMANEVAL=1` is set, and even
then it is loaded from the local dataset cache only.

## First Smoke Run

The model-facing commands are intentionally deterministic and small by default.
They expect HuggingFace model weights to already be available locally.

```bash
conda run -n orpheus python -m latent_kv collect --benchmark hanoi --limit 3 --run-id smoke_hanoi
conda run -n orpheus python -m latent_kv compress --run runs/smoke_hanoi --method random --latent-dim 64
conda run -n orpheus python -m latent_kv behavior --run runs/smoke_hanoi --baseline original_cache random
conda run -n orpheus python -m latent_kv prompt-baseline --run runs/smoke_hanoi --benchmark hanoi --baseline standard cot
conda run -n orpheus python -m latent_kv evaluate --run runs/smoke_hanoi
```

Local checks for the next reported baselines:

```bash
conda run -n orpheus python -m latent_kv tot-baseline --run runs/tot_smoke --limit 6 --breadth 5
conda run -n orpheus python -m latent_kv react-baseline --run runs/react_smoke --limit 3
```

Artifacts are written under `runs/<run_id>/` and ignored by git:

- `records.jsonl`
- `caches/*.pt`
- `metrics.json`
- `report.md`
- `plots/*.png`

## Research Shape

The baseline-first path is deliberate. Every useful latent KV method should be
compared against fixed no-cache, original-cache, random projection, PCA/SVD,
autoencoder, retrieval, soft-prefix, and hidden-state/KV ablations before it is
treated as meaningful progress.

The `compress` command writes reconstruction artifacts. The `behavior` command
is the important scoring step: it replays original or reconstructed caches
through the local model and adds behavioural baseline rows to `metrics.json` and
`report.md`.

## Research Log And Targets

Use the log after each meaningful experiment:

```bash
conda run -n orpheus python -m latent_kv log \
  --title "hanoi_real baseline sweep" \
  --worked "cache replay baselines completed" \
  --did-not-work "small model did not solve Hanoi" \
  --todo "repeat on larger local model"
```

Tracked reported targets can be inspected with:

```bash
conda run -n orpheus python -m latent_kv targets
```

The target registry distinguishes local implementation checks from exact
reported-protocol reproductions. Matching reported numbers requires the source
model scale, prompt, decoding settings, and benchmark protocol.

## Prompt Baseline Tiers

Prompt baselines support named budget tiers so a quick check and a larger run
use the same command path and write comparable protocol metadata:

| Tier | Limit | Max New Tokens | Purpose |
|---|---:|---:|---|
| `smoke` | 5 | 320 | Quick model-level check for plumbing and reports |
| `working` | 20 | 320 | Small local baseline tier before early comparisons |
| `comparison` | 100 | 320 | Stronger local comparison floor for candidate methods |
| `full` | 1319 | 320 | Full GSM8K-test-sized run once protocols are frozen |

Explicit `--limit` and `--max-new-tokens` values override the selected tier.
Prompt-baseline records are written incrementally, and `metrics.json` plus
`report.md` are refreshed after each completed example. If a long run fails or
is interrupted, inspect the partial `behavior/*_records.jsonl`, `metrics.json`,
and shell log for the last completed task and any recorded generation errors.

Example quick check:

```bash
LATENT_KV_ALLOW_DATASET_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n orpheus python -m latent_kv prompt-baseline \
  --run runs/baseline_qwen_smoke \
  --benchmark gsm8k \
  --baseline-tier smoke \
  --baseline standard cot retry_reflection \
  --model-id /path/to/local/model/snapshot
```

Example full-sized run using the same protocol path:

```bash
LATENT_KV_ALLOW_DATASET_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n orpheus python -m latent_kv prompt-baseline \
  --run runs/baseline_qwen_full \
  --benchmark gsm8k \
  --baseline-tier full \
  --baseline standard cot retry_reflection \
  --model-id /path/to/local/model/snapshot
```

## Config-Driven Cache Collection

Latent-manifold experiments should start from cache-backed prompt records: one
row per task plus a replayable KV cache bundle under `caches/`. The
`collect-prompt-caches` command accepts a versioned config, resolves the local
HuggingFace model profile, writes `resolved_config.json`, then captures caches
for the selected prompt protocol.

```bash
LATENT_KV_ALLOW_DATASET_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n orpheus python -m latent_kv collect-prompt-caches \
  --run runs/qwen_cache_smoke \
  --config configs/qwen_gsm8k_cot_cache_smoke.yaml
```

The resolved config records model-derived cache dimensions such as layer count,
KV heads, head dimension, selected layers, bytes per token, and storage estimate.
These values are derived from the local model config rather than hardcoded, so
larger local models can be added through additional config files.

Compression artifacts can be checked against the point-codec contract before
behavioural replay. The validator confirms there is one latent point per cache,
decodes reconstructed cache vectors back to the original KV shapes, checks for
finite tensors, and verifies replay metadata is present.

```bash
conda run -n orpheus python -m latent_kv compress \
  --run runs/qwen_cache_smoke \
  --method retrieval \
  --latent-dim 8

conda run -n orpheus python -m latent_kv validate-codec \
  --run runs/qwen_cache_smoke \
  --method retrieval
```
