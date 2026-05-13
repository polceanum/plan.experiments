# Research Log

This log tracks experiment progress, failures, and next actions. Keep entries
short and concrete so they can be updated after every run.

## 2026-05-13

### Worked

- Created the `latent_kv` research scaffold with local-only model execution.
- Verified `conda run -n orpheus pytest` passes.
- Captured a real KV cache from `EleutherAI/pythia-70m-deduped` using local weights.
- Ran behavioural replay baselines for original, random projection, PCA/SVD, autoencoder, and retrieval caches.
- Confirmed run reports include baseline rows and qualitative examples.

### Did Not Work / Caveats

- `EleutherAI/pythia-70m-deduped` did not solve the Hanoi smoke prompt; this is expected for plumbing, not a scientific result.
- Existing KV replay uses legacy tuple caches; Transformers warns this will be removed after the installed version range.
- Compression baselines run, but learned/generative planning modules are not yet trained.
- Reported prompt baselines still need protocol matching against each exact prompt set, model, decoding, and benchmark split.

### Left To Do

- Implement local prompt baselines: standard prompting, chain-of-thought, self-consistency, and retry/reflection.
- Add reported-target metadata for each baseline and track whether we are running an exact reproduction or a local approximation.
- Add SOTA/protocol checks that compare local results with reported numbers only when the model/protocol match.
- Implement soft-prefix and Tree-of-Thought style search baselines.
- Upgrade cache replay to the modern Transformers `Cache` interface when practical.

## 2026-05-13 - ToT/ReAct Local Baseline Scaffold

### Worked

- Added Game of 24 as a deterministic benchmark with safe arithmetic expression verification.
- Added a local symbolic Tree-of-Thought style search harness for Game of 24.
- Added a tiny deterministic ReAct-style tool environment with thought/action/observation traces.

### Did Not Work / Caveats

- The ToT baseline is a local symbolic harness, not a GPT-4 reproduction of the reported protocol.
- The ReAct baseline is a tiny local environment, not HotpotQA, FEVER, ALFWorld, or WebShop.

### Left To Do

- Add local-LLM proposal/evaluator modes for Tree of Thoughts.
- Add ReAct adapters for real local environments, starting with Minigrid/BabyAI if installed.
- Add exact prompt packs and protocol metadata for any baseline we can reproduce with local weights.


## 2026-05-13 - prompt baseline and target-check scaffold

### Worked

- added structured research log and reported-target registry
- implemented local standard, CoT, self-consistency, and retry/reflection prompt baselines
- added target checks that mark local runs as protocol mismatches unless the reported protocol is matched

### Did Not Work / Caveats

- small Pythia prompt smoke did not solve the tiny Hanoi/GSM8K examples
- GSM8K dataset loading attempted network metadata before fallback; patched fallback to be default unless explicitly enabled

### Left To Do

- implement exact prompt/protocol packs where local model availability makes reproduction possible
- implement Tree-of-Thought and ReAct baselines

## 2026-05-13 - ToT and ReAct local baselines

### Worked

- added Game of 24 benchmark and safe verifier
- added symbolic local Tree-of-Thought baseline with breadth-limited search
- added symbolic local ReAct toolworld baseline with thought/action/observation traces

### Did Not Work / Caveats

- these are local implementation checks, not GPT-4/PaLM protocol reproductions

### Left To Do

- add local-LLM proposal/evaluator mode for ToT
- add ReAct adapters for Minigrid/BabyAI or another installed local environment

## 2026-05-13 - real GSM8K local Qwen sweep

### Worked

- loaded cached GSM8K test split locally with offline flags and escalated cache-lock access
- ran Qwen2.5-0.5B-Instruct local chat-template GSM8K sweep on 5 real test examples
- standard and retry_reflection reached 2/5; CoT reached 1/5; self_consistency with 3 samples reached 2/5

### Did Not Work / Caveats

- first 20-example attempt used fallback data and raw prompts; patched chat-template prompting afterward
- self-consistency is slow locally, about 95 seconds mean latency per example for 3 samples

### Left To Do

- increase GSM8K sample size to 20 or 50 overnight with chat-template prompts
- add exact few-shot CoT exemplars matching the reported CoT protocol instead of generic zero-shot CoT
- run same 5-example sweep on TinyLlama local weights for comparison

## 2026-05-13 - Latent Planning Design Note Update

### Worked

- Consolidated design lessons for the latent-KV work.
- Added `docs/LATENT_PLANNING_NOTES.md` with planning-focused notes on validators,
  novelty, structural quality, candidate filtering, protocol discipline, and
  robust AE baselines.
- Updated `AGENTS.md` so future agents preserve the project framing.

### Did Not Work / Caveats

- These design notes are research assumptions and should be validated
  experimentally, not treated as settled claims.

### Left To Do

- Implement novelty metrics for generated/retrieved cache states.
- Implement structural quality metrics for reasoning traces, tool calls, and
  correction/verification positions.
- Add candidate generation plus automatic filtering for generated latent KV
  states.
