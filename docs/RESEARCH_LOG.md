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

## 2026-05-13 - Qwen baseline protocol audit smoke

### Worked

- Ran local Qwen2.5-0.5B-Instruct GSM8K baseline smoke with standard, CoT, and retry/reflection on 2 cached real test examples.
- Protocol metadata was written for each prompt baseline and check-targets reported dimension-level protocol mismatches.

### Did Not Work / Caveats

- This is a tiny local smoke check, not a reported PaLM 540B CoT reproduction; model and prompt protocol intentionally mismatch the reported target.

### Left To Do

- Scale the protocol-audited Qwen GSM8K run to a fixed 20-example tier before using it as a stronger local baseline.

## 2026-05-13 - Qwen 5-example baseline sanity check

### Worked

- Ran protocol-audited Qwen2.5-0.5B-Instruct GSM8K local baseline check on 5 cached real test examples with 320 max new tokens.
- The earlier 2-example 0.5 tie disappeared: standard reached 2/5, retry_reflection reached 2/5, and CoT reached 1/5 with different task-level success patterns.

### Did Not Work / Caveats

- Still a small local check, not a reported PaLM 540B protocol reproduction; check-targets marks CoT as protocol_mismatch for model and prompt.

### Left To Do

- Use a fixed 20-example GSM8K tier next for a stronger local baseline before comparing latent-KV methods.

## 2026-05-13 - Prompt baseline tier smoke validation

### Worked

- Validated the new prompt-baseline --baseline-tier smoke path on local Qwen2.5-0.5B-Instruct with cached real GSM8K examples.
- The smoke run wrote baseline_tier metadata into metrics/report artifacts and check-targets produced the expected protocol_mismatch audit.

### Did Not Work / Caveats

- This remains a 5-example local smoke tier, not a strong comparison floor or reported-protocol reproduction.

### Left To Do

- Commit the tier/protocol changes, then use --baseline-tier working or comparison for stronger local baseline runs.

## 2026-05-14 - Qwen full-tier baseline interrupted after standard

### Worked

- Completed the full 1319-example cached GSM8K standard prompt baseline for local Qwen2.5-0.5B-Instruct: 465/1319 correct, accuracy 0.353, no generation errors.
- The streamed-artifact safeguard preserved a partial CoT run through 156 examples: 61/156 correct, accuracy 0.391, no generation errors.

### Did Not Work / Caveats

- A reboot interrupted the wrapper during CoT before retry_reflection started; run.log only contains the launch lines because conda/tee output was buffered, but JSONL, metrics, and report artifacts survived.
- This remains a local Qwen zero-shot protocol check, not a PaLM 540B reported-protocol reproduction; check-targets marks CoT as protocol_mismatch for model and prompt.

### Left To Do

- Resume or rerun the full tier for CoT and retry_reflection, preferably with unbuffered Python/stdout handling if live terminal logs are important.

## 2026-05-14 - Qwen full-tier standard and CoT baselines completed

### Worked

- Completed the full 1319-example cached GSM8K standard and CoT baselines for local Qwen2.5-0.5B-Instruct with streamed JSONL artifacts.
- Standard reached 465/1319 correct, accuracy 0.353, with no generation errors.
- Zero-shot CoT reached 498/1319 correct, accuracy 0.378, with no generation errors.
- The resumed CoT-only run stopped after CoT and did not start retry_reflection.
- check-targets recorded the expected protocol_mismatch: decoding and split match the tracked CoT target, but model and prompt differ from PaLM 540B eight-shot CoT.

### Did Not Work / Caveats

- This is a local Qwen zero-shot CoT baseline, not a reported PaLM 540B reproduction.
- retry_reflection remains intentionally deferred for a later overnight run.

### Left To Do

- Start latent-KV method experiments against the completed standard/CoT local baseline floor.
- Run retry_reflection later with --resume when convenient, ideally overnight or after checking laptop load.

## 2026-05-14 - Qwen 10-cache labelled RAE smoke

### Worked

- Saved the KV RAE end-to-end flow diagram; collected 10 local Qwen GSM8K CoT prompt caches with verifier labels; source prompt-cache accuracy was 2/10; retrieval and rae_lstm compression artifacts preserved source_labels; validate-codec passed for 10/10 retrieval and 10/10 rae_lstm caches; rae_lstm 5-epoch training telemetry showed loss decreasing from 1.02272 to 1.02106.

### Did Not Work / Caveats

- Behavioural replay with 64-token continuations reached 0/10 for original_cache, retrieval, and rae_lstm despite no replay errors. Treat this as a replay-protocol smoke, not evidence against the latent method; likely needs better continuation prompt/position handling and longer or protocol-matched decoding.

### Left To Do

- Improve behavioural replay protocol for prompt-cache continuations; add labelled latent diagnostics separating solved vs unsolved source caches; scale cache collection beyond 10 examples after replay checks are better understood.

## 2026-05-14 - Prompt-cache replay protocol fix

### Worked

- Added generation-token metadata to new cache bundles, including generated token IDs and the local generation config used for the source sequence.
- Changed behavioural replay to use each source record's original generated-token count by default, preserving variable plan lengths while still allowing explicit CLI overrides.
- Matched local greedy replay to the model generation stack by passing explicit replay positions where supported and applying the model's repetition penalty before argmax.
- A fresh one-example Qwen cache replay now reproduces the stored correct source answer with the record-derived 295-token budget.
- Re-running the 10-cache smoke with record-derived budgets gave original_cache 2/10, matching the source prompt-cache accuracy; retrieval and rae_lstm remained 0/10 with no replay errors.

### Did Not Work / Caveats

- Decoded-cache behavioural replay still fails on the 10-cache smoke even though structural validation passes, so the current RAE/retrieval cache reconstructions are not behaviourally faithful yet.
- The replay path still uses legacy tuple caches, which Transformers warns will be removed in a future version.

### Left To Do

- Add token-level replay fidelity diagnostics comparing original generated IDs against replay IDs.
- Improve decoded-cache quality before scaling behavioural claims; keep retrieval as the nearest-neighbour sanity baseline.

## 2026-05-14 - chunk-projected RAE reconstruction smoke

### Worked

- Added masked variable-length loss, latent-conditioned positional decoder inputs, final+mean encoder summary, and a chunk projection layer before the LSTM. Full tests passed under conda orpheus. On runs/qwen_cache_smoke_10, structural validation stayed 10/10 and mean validation MSE improved from the prior best 27.23005 to 27.08518.

### Did Not Work / Caveats

- The improvement did not yet translate into GSM8K behavioural replay: rae_lstm remained 0/10 while original_cache and source prompt-cache stay 2/10. This is still a local Qwen 10-example implementation check, not a reported-protocol reproduction.

### Left To Do

- Try more expressive one-point decoders or loss terms aligned with replay-sensitive KV structure, such as layer/head-aware chunking, per-layer normalization, residual reconstruction from retrieval/mean cache, or token-position-aware objectives before rerunning behavioural replay.

## 2026-05-14 - wider projected RAE reconstruction check

### Worked

- Fixed RAE artifact latent encoding so saved point latents use the same masked-normalized variable-length inputs as training; added a variable-length regression test. With chunk projection, a wider hidden_dim=256, latent_dim=256, 80-epoch local run on runs/qwen_cache_smoke_10 reached 10/10 structurally valid reconstructions and mean validation MSE 16.66987, down from 27.08518.

### Did Not Work / Caveats

- Despite the large reconstruction-MSE improvement, GSM8K behavioural replay remained 0/10 for rae_lstm; reconstructed continuations averaged 79.3 tokens versus 261.9 source replay budget, suggesting lower MSE alone is not yet preserving replay dynamics.

### Left To Do

- Add replay-sensitive diagnostics and losses: token-level replay agreement, first-token KL/logit similarity from reconstructed caches, layer/head-wise MSE, and optionally train against a residual or layer-aware representation before another behaviour run.

## 2026-05-14 - replay-fidelity and training-curve diagnostics

### Worked

- Added replay-fidelity diagnostics that compare original vs reconstructed-cache logits after the first replay token, plus training-curve summaries for learned codec logs. Tests passed: 52 total. On runs/qwen_cache_smoke_10, rae_lstm replay-fidelity showed mean logit cosine 0.30649, mean logit MSE 18.61020, KL(original||reconstructed) 15.90055, and top-1 match rate 0/10; retrieval was worse with cosine 0.03710 and MSE 24.07538. The rae_lstm training curve was monotonic nonincreasing and classified as accelerating_decrease: loss 0.99270 -> 0.93307 across sampled epochs 1..80, with late mean delta -0.02690 versus early -0.00292.

### Did Not Work / Caveats

- Lower reconstruction MSE still does not preserve the first cache-dependent next-token distribution; this explains why behavioural replay remains 0/10 despite structurally valid reconstructed caches. The current training log is sparse because log_every was 20, so curve-shape diagnostics are useful but coarse.

### Left To Do

- Run future learned-codec experiments with denser log_every for curve inspection; add logit/KL-sensitive training objectives or per-layer/head diagnostics before relying on MSE as the main reconstruction signal.

## 2026-05-14 - RAE objective and replay failure diagnosis

### Worked

- Double-checked the RAE objective: training uses masked reconstruction MSE on normalized cache vectors with AdamW weight decay only; no KL term is used. Added explicit artifact/telemetry metadata: objective=masked_reconstruction_mse_no_kl, kl_loss_weight=0.0, and loss_components. A 240-epoch weight-decayed RAE comparison improved validation MSE to 5.56810 and produced a monotonic accelerating training curve, but replay-fidelity still had top-1 match 0/10. Record inspection showed no replay errors; reconstructed-cache outputs are degenerate token streams. Token diagnostics show the first replay token remains 'To', but the next cache-dependent top token flips from original ' solve'/' determine'/' calculate' to fragments like 's', 'I', newline, '.', 'ar', or ' PLL'.

### Did Not Work / Caveats

- The problem is not a KL-loss issue and not a verifier/parser artifact. Global cache metrics are misleading: per-record whole-vector cosine is high (about 0.92-0.96), yet behaviour collapses. Layer diagnostics showed poor relative reconstruction for some value-cache slices, especially early-layer values, so flattened MSE/cosine can miss replay-critical KV distortions.

### Left To Do

- Next try layer/head-aware or relative reconstruction losses, with separate key/value diagnostics and possibly logit-matching replay loss. Treat VAE/KL as a separate generative-latent experiment; for faithful RAE reconstruction, adding KL would likely hurt unless carefully beta-scheduled.

## 2026-05-14 - fixed variable-length cache vector alignment

### Worked

- Found the main reconstruction bug: variable-length prompt caches were flattened tightly and padded only at the end, so layer/key/value boundaries shifted across examples and training columns mixed unrelated KV coordinates. Fixed compression to pad each layer/key/value tensor to shared token shapes before flattening, while saving compact original-shape reconstructions for existing validation/replay consumers. Tests passed: 52 total. On runs/qwen_cache_smoke_10, aligned 240-epoch RAE improved mean validation MSE from 5.56810 to 0.36029, KL(original||reconstructed) from 13.58 to 0.875, logit cosine from 0.454 to 0.788, and first cache-dependent top-1 match from 0/10 to 6/10. A 600-epoch run improved validation MSE further to 0.33457 and logit cosine to 0.83974, with top-1 still 6/10.

### Did Not Work / Caveats

- Behavioural GSM8K replay remains 0/10. After the alignment fix the outputs are no longer garbage token streams; they are fluent generic math solutions that often ignore the original problem details. Longer training is now decelerating and only marginally improves replay fidelity, so the remaining issue is rollout-faithful reconstruction, not the earlier flattened-vector alignment bug.

### Left To Do

- Add multi-step replay-fidelity diagnostics and train with replay-sensitive objectives or layer/head/value-aware losses; first-step top-1 is no longer enough because coherent continuations can still drift away from the source problem over several tokens.

## 2026-05-14 - latent-only high-capacity RAE rollout check

### Worked

- Added multi-step teacher-forced replay-fidelity diagnostics and explicit RAE artifact metadata marking decoder_source=latent_only with no retrieval or per-cache residuals. Full tests passed: 52. On runs/qwen_cache_smoke_10, a high-capacity aligned RAE (latent_dim=512, hidden_dim=512, 400 epochs, weight_decay=0.001) reached mean validation MSE 0.21915 and train MSE 0.15832. Multi-step replay fidelity improved strongly: first-step logit cosine 0.92635, KL 0.75303, and 8-step top-1 match rates [0.6, 0.6, 1.0, 0.9, 0.8, 0.8, 0.7, 0.8].

### Did Not Work / Caveats

- Behavioural GSM8K replay remained 0/10. The failure is no longer corrupted token streams; the model now produces fluent but generic math solutions that drift from the original problem. This means no local residual or nearest-neighbour correction should be used to make reconstruction look good, because that would bypass the meaning of the latent point.

### Left To Do

- Keep the one-point latent-only contract. Next improve the learned latent itself with objectives that preserve rollout behaviour, such as multi-step teacher-forced logit matching, layer/head/value-aware weighting, or training on richer cache collections; treat VAE/KL as a separate generative-latent experiment rather than a patch for RAE reconstruction.

## 2026-05-15 - oracle exact-vector replay check

### Worked

- Created an isolated oracle artifact for runs/qwen_cache_smoke_10 with reconstructed vectors set exactly to the original flattened prompt caches.
- Exact-vector oracle under the rae_lstm behavior branch validated at 0.0 MSE, 10/10 valid caches, 8-step top-1 replay-fidelity 1.0, KL 0.0, and byte-for-byte identical full-budget outputs to original_cache.
- Aligned-to-compact conversion was separately checked to round-trip original cache vectors exactly, with max_abs_error 0.0 and max_mse 0.0.

### Did Not Work / Caveats

- The learned high-capacity rae_lstm artifact still has mean reconstruction MSE 0.21915 and semantic free-running drift, so the remaining 0/10 behavior is not caused by reconstruction plumbing.

### Left To Do

- Drive the latent-only decoder toward much lower replay-critical error before testing other techniques; prioritize reconstruction fidelity and cache-sensitive diagnostics over local residual patchups.

## 2026-05-15 - success-only CoT cache replay slice

### Worked

- Built runs/qwen_cache_cot_success20 from the first 20 correct CoT records in runs/baseline_tier_qwen_full/behavior/cot_records.jsonl; the full prompt-only CoT run had 498/1319 correct examples.
- Original-cache replay on the success-only slice preserved 20/20 correct answers, confirming these cached prompts are a clean solved target set.
- Rank-20 PCA/SVD latent-only reconstruction had near-zero reconstruction error and preserved 20/20 behavior, with 8-step replay top-1 match 1.0.

### Did Not Work / Caveats

- The current 512-dim LSTM RAE trained for 500 epochs with weight_decay=0.001 reached train MSE 0.20094 and replay-fidelity top-1 0.65, but free-running behavior was 0/20 on the solved slice.

### Left To Do

- Use this success-only slice as the primary reconstruction target while improving the learned latent-only decoder; more epochs alone are unlikely to close the gap unless architecture/loss fidelity improves.

## 2026-05-15 - RAE corruption sensitivity smoke

### Worked

- Added a corruption-sensitivity diagnostic that replays cache interpolations original + alpha*(reconstructed-original) without changing the latent-only decoder path.
- On runs/qwen_cache_cot_success20 with method rae_lstm, a 3-record smoke curve showed alpha=0.0 preserved 3/3 solved behaviours while alpha=0.1 and alpha=1.0 both fell to 0/3.

### Did Not Work / Caveats

- The long 10-record, 7-alpha free-running sweep was too slow for interactive iteration and was stopped; future sweeps should use fewer records/alphas or a faster teacher-forced variant.

### Left To Do

- Use the sensitivity curve to set reconstruction targets: the current learned RAE error is far outside the behavioural tolerance, so reduce replay-critical error before broadening techniques.
