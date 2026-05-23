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

## 2026-05-15 - Frozen-LLM replay-aware RAE smoke

### Worked

- Added replay-train for latent-only RAE training through a frozen local LLM using teacher-forced replay KL plus masked cache MSE. Two-cache Qwen success-slice smoke ran end-to-end; loss decreased monotonically over 5 epochs and replay KL dropped from 0.0943 to 0.0831.

### Did Not Work / Caveats

- This was only a tiny smoke run with latent_dim=16, hidden_dim=16, limit=2, and one replay token, so it is not evidence that full-scale behavior is restored. Reconstruction MSE remains far above the observed behavioral tolerance.

### Left To Do

- Scale replay-aware training on runs/qwen_cache_cot_success20 with more records, more capacity, and 2-8 teacher-forced replay steps; compare against the 1-layer MSE-only RAE and PCA controls.

## 2026-05-15 - Modest frozen-LLM replay-aware RAE run

### Worked

- Ran rae_lstm_replay_modest5 on runs/qwen_cache_cot_success20 with latent_dim=256, hidden_dim=256, one LSTM layer, limit=5, two teacher-forced replay steps, and 30 epochs. Training loss decreased monotonically from 0.8619 to 0.8247; teacher-forced replay KL dropped from 0.4054 to 0.0412. The resulting artifact validated 5/5 caches and replay-fidelity top-1 matched original logits for both measured steps.

### Did Not Work / Caveats

- Free-running behavior was still 0/5. The model matched the first teacher-forced replay decisions but drifted badly during full generation; reconstruction MSE remained high around 0.35 per-cache validation mean, far above the earlier corruption-tolerance threshold.

### Left To Do

- Next test longer teacher-forced horizons and/or replay loss on more continuation steps before increasing depth. Keep capacity modest, and compare whether multi-step replay KL predicts free-running recovery better than first-step top-1.

## 2026-05-15 - Full-sequence replay-aware RAE scaling

### Worked

- Added replay-train gradient accumulation via --train-batch-size after the all-at-once full20/s4 run hit MPS OOM around epoch 10. Re-ran full20/s4 with latent_dim=512, hidden_dim=512, one LSTM layer, train_batch_size=1, 40 epochs, and replay_weight=0.2. Training replay KL dropped from 1.2667 to 0.0494; four-step replay fidelity reached top-1 rates [1.0, 0.85, 1.0, 1.0]. Also trained full20/s8 for 25 epochs; 8-step replay KL dropped from 1.4062 to 0.2543 and improved late-step fidelity compared with the s4 model.

### Did Not Work / Caveats

- Neither full20/s4 nor full20/s8 restored free-running GSM8K behavior; both scored 0/20. Validation MSE remains around 0.42-0.44 mean compact-cache MSE, still far above the earlier corruption-tolerance threshold. Teacher-forced top-1 agreement over 4-8 steps is not sufficient to keep long greedy generations on task.

### Left To Do

- Next improvements should target reconstruction fidelity and longer rollout stability: consider increasing MSE pressure/normalization quality, curriculum from MSE-only pretraining into replay loss, and measuring more rollout steps before spending more on depth.

## 2026-05-15 - Initialized replay-aware RAE continuation

### Worked

- Added --init-method to replay-train so frozen-LLM replay training can initialize from an existing MSE-trained RAE. The initialized full20/s8 run used latent_dim=512, hidden_dim=512, one LSTM layer, chunk_dim=4096, mse_weight=2.0, replay_weight=0.2, and train_batch_size=1. Training loss decreased monotonically from 1.0443 to 0.8649 over 40 epochs, then a 20-epoch continuation dropped loss from 0.8628 to 0.8401 and replay KL from 0.1655 to 0.0821. Final 8-step replay fidelity improved to mean KL 0.0370 with top-1 rates [1.0, 0.9, 1.0, 1.0, 0.85, 1.0, 0.95, 0.8].

### Did Not Work / Caveats

- Free-running GSM8K behavior remains 0/20. Reconstruction validation mean remains around 0.34 compact-cache MSE, still far above the corruption-tolerance threshold. Longer training improves replay-fidelity curves but does not yet recover long greedy reasoning.

### Left To Do

- Next focus should be better reconstruction capacity/objective before more replay-only training: stronger MSE pretraining, larger but still shallow decoder capacity, or loss weighting by layer/head/token positions that matter for replay.

## 2026-05-15 - Temporal token-state RAE codec

### Worked

- Replaced the artificial chunk sequence assumption with an explicit rae_temporal codec whose sequence axis is real token time. Each temporal step is the full KV state for one token across layers, key/value tensors, heads, and head dimensions. Added temporal codec support to replay-train via --codec-kind temporal and smoke-tested both MSE reconstruction and frozen-LLM replay paths on runs/qwen_cache_cot_success20.

### Did Not Work / Caveats

- The short 20-epoch temporal MSE smoke validated 20/20 caches but reconstruction MSE was still high around 1.94 mean validation MSE. A tiny temporal replay smoke validated caches but started with very high replay KL, as expected from random temporal weights. This fixes the representation, not the fidelity problem yet.

### Left To Do

- Train rae_temporal longer and/or initialize temporal replay from a stronger temporal MSE artifact. Compare temporal curves against the previous chunked RAE without treating chunked results as latent-planning evidence.

## 2026-05-15 - Remove chunk-sequence RAE

### Worked

- Removed the previous flattened-chunk LSTM RAE and the misleading replay-training command surface; temporal token-state seq2seq RAE is now the only learned sequence codec exposed by the CLI.

### Did Not Work / Caveats

- No new training run was performed in this cleanup step; older chunk-based artifacts remain historical only and are no longer advertised as trainable methods.

### Left To Do

- Train the temporal seq2seq codec for longer on success20 now that the model axis matches token time.

## 2026-05-15 - Temporal RAE frozen-LLM gradient signal

### Worked

- Added optional frozen-LLM prompt-state transition KL gradients to rae_temporal training while preserving the temporal seq2seq point-codec contract; updated docs/KV_RAE_FLOW.md to describe token-time states, one latent point, decoded KV sequence, and LLM replay.

### Did Not Work / Caveats

- No full Qwen training run was launched in this edit; the new LLM-gradient path was validated with a fake frozen-LLM unit test rather than local Qwen weights.

### Left To Do

- Run a full success20 rae_temporal experiment with and without --llm-loss-weight, then compare validation MSE, replay fidelity, and behavioural replay.

## 2026-05-15 - Temporal RAE decoded cache solves one Qwen GSM8K task

### Worked

- Ran runs/qwen_cache_cot_success20_temporal_llm_smoke using one solved Qwen GSM8K cache. A tiny 1-epoch temporal+frozen-LLM run validated structurally but decoded behavior failed 0/1. A higher-capacity latent_dim=1024 hidden_dim=1024 run with --llm-loss-weight 0.001 and --llm-steps 1 for 500 epochs reduced compact reconstruction MSE to 9.50e-05 and recovered decoded-cache behavior: original_cache 1/1, rae_temporal 1/1, parsed answer 18.

### Did Not Work / Caveats

- Lower-capacity/shorter runs did not solve after decoding: 1 epoch produced repetitive text, and an 80-epoch latent_dim=256 hidden_dim=256 run parsed 1. A 200-epoch latent_dim=1024 run became coherent but answered 32, so solved behavior appears to require much tighter reconstruction than 0.0287 MSE on this example.

### Left To Do

- Scale carefully from one-cache overfit to a small multi-cache success slice, reporting decoded-cache task accuracy first, then reconstruction MSE and frozen-LLM transition KL as diagnostics.

## 2026-05-15 - Temporal RAE frozen-LLM five-cache decoded behavior

### Worked

- Scaled the temporal point codec from one-cache overfit to a five-cache Qwen GSM8K success slice. The first 500-epoch run with latent_dim=1024 hidden_dim=1024 llm_loss_weight=0.001 validated 5/5 decoded caches and solved 1/5 after decoding. A longer 1500-epoch run with llm_loss_weight=0.0001 reduced mean validation MSE to 2.27e-04 and improved decoded-cache behavior to 3/5 while original_cache stayed 5/5.

### Did Not Work / Caveats

- The 500-epoch five-cache run had mean MSE 2.34e-02 and only 1/5 decoded-cache accuracy. The 1500-epoch run still failed two tasks: gsm8k_0006 parsed 160 instead of 260, and gsm8k_0024 parsed 14.625 instead of 26, so near-exact reconstruction remains necessary and MSE around 2e-4 is not universally sufficient.

### Left To Do

- Next run should either push five-cache reconstruction lower, add more prompt-transition positions for the frozen-LLM loss, or compare against an MSE-only temporal run at the same capacity to isolate the value of frozen-LLM gradients.

## 2026-05-15 - Temporal RAE ten-cache scaling check

### Worked

- Scaled the temporal point codec to 10 solved Qwen GSM8K CoT caches in runs/qwen_cache_cot_success20_temporal_llm_10_big. A larger latent_dim=1536 hidden_dim=1536 attempt hit MPS OOM, so the stable latent_dim=1024 hidden_dim=1024 configuration trained for 2000 epochs with llm_loss_weight=0.0001 and llm_steps=1. Training loss fell from 1.00219 to 0.000104, compact compression MSE was 2.30e-04, validation found 10/10 structurally valid decoded caches with mean reconstruction MSE 2.68e-04, and original_cache control stayed 10/10.

### Did Not Work / Caveats

- Decoded-cache behaviour did not scale beyond the earlier success band: rae_temporal solved 3/10. Correct decoded tasks were gsm8k_0006, gsm8k_0033, and gsm8k_0034; failures were coherent arithmetic drift rather than invalid cache replay. The 1536/1536 capacity increase is not currently feasible on MPS alongside Qwen.

### Left To Do

- Try a more targeted fidelity objective before another long scaling run: more prompt-transition positions, layer/head/value-weighted reconstruction, or a staged MSE-to-LLM-loss schedule; keep behavioural decoded-cache accuracy as the headline metric.

## 2026-05-16 - Interrupted full Qwen CoT temporal RAE feasibility run

### Worked

- Launched temporal RAE training on runs/qwen_cache_cot_full_attached with 1,319 attached CoT cache records, latent_dim=1024, hidden_dim=1024, llm_loss_weight=0.0001, llm_steps=2, train_batch_size=1, log_every=100, and checkpoint_every=1000. The startup event and epoch-1 checkpoint/log were written successfully; frozen LLM gradient loss ran on mps without crashing.

### Did Not Work / Caveats

- The run was stopped after epoch 1. Epoch 1 took about 1,473 seconds, making a 20,000-epoch run impractical at this configuration. The compression CLI writes checkpoints but does not currently support restoring from rae_temporal_latest.pt, and the training JSONL is cleared on startup, so relaunching would start a fresh run unless checkpoint-resume support is added.

### Left To Do

- Add checkpoint restore support for rae_temporal or rerun a shorter feasibility schedule first; consider reducing llm_steps, training on a success/failure slice, or using denser early logging before attempting another long full-cache run.

## 2026-05-17 - Epoch-10 temporal RAE latent category PCA analysis

### Worked

- Added reusable latent-analysis CLI utilities; generated category annotations, checkpoint latents, PCA CSV, and PNG plots for 1,319 Qwen GSM8K CoT cache records from rae_temporal epoch 10.

### Did Not Work / Caveats

- Initial extractor materialized the full cache matrix and used too much CPU RAM; replaced it with batch streaming and added progress logging. Category labels remain heuristic exploratory annotations, not benchmark ground truth.

### Left To Do

- Inspect PCA plots by correctness/category, compare later checkpoints against epoch 10, and consider a stronger manual or local-LLM category audit for ambiguous GSM8K prompts.

## 2026-05-17 - Latent interpolation replay workflow

### Worked

- Added a reusable latent-interpolate CLI that selects correct-correct endpoint pairs, decodes alpha-step temporal RAE latent interpolations, replays each decoded cache under both endpoint prompt contexts, and writes pair/replay/sequence artifacts for inspection.

### Did Not Work / Caveats

- Intermediate points do not have independent GSM8K labels; automated correctness is endpoint-relative only. Full replay sweeps can be expensive because source generations are often hundreds of tokens.

### Left To Do

- Run and inspect the first 50-pair mixed interpolation sweep, then compare decoded transition outputs by same-category versus cross-category pairs.

## 2026-05-17 - Representative latent interpolation inspection sweep

### Worked

- Added spread/min-distance/max-distance pair selection, target/prompt-overlap filtering, and a compact interpolation_inspection.md report; generated a 6-pair representative epoch-10 sweep with 7 alpha points and 64-token CPU continuations.

### Did Not Work / Caveats

- Endpoint-target accuracy remains zero; decoded continuations are readable but often generic or semantically drifted, so this is qualitative evidence that epoch-10 latents are not yet preserving endpoint task plans under interpolation.

### Left To Do

- Rerun the representative sweep on later checkpoints after loss improves, and compare whether endpoint-alpha decodes become more task-specific before scaling to more pairs.

## 2026-05-18 - Epoch 20 reconstructed interpolation probe

### Worked

- Epoch 20 latent analysis artifacts were reused to filter interpolation endpoints through a reconstruction scan; the probe100 scan found 2 decoded-correct endpoints and produced 1 cross-category reconstructed-correct interpolation bridge with 18 replay rows.

### Did Not Work / Caveats

- The decoded reconstructions often produced the correct numeric answer while reasoning about the wrong scenario, so endpoint numeric correctness alone is not a faithful solved-plan filter.

### Left To Do

- Run a broader reconstruction scan on later checkpoints and add semantic/structural faithfulness checks before using reconstructed-correct pairs as interpretive evidence.

## 2026-05-18 - Epoch 30 convincing reconstruction probe

### Worked

- Added a local prompt-faithfulness gate for reconstruction scans and interpolation endpoint filtering; epoch 30 latent artifacts were extracted and a 50-endpoint CPU probe completed with no replay failures.

### Did Not Work / Caveats

- The probe found 1 numeric-correct decoded reconstruction but 0 convincing reconstructions: the numeric-correct row reused almost none of the prompt-specific content and reasoned about a generic purchase instead of the source problem.

### Left To Do

- Rerun convincing reconstruction scans on later checkpoints; consider adding teacher-forced token/logit faithfulness or semantic overlap metrics before scaling interpolation endpoint selection.

## 2026-05-18 - Teacher-forced replay KL training path

### Worked

- Added a separate rae_temporal teacher-forced generated-token replay KL loss, controlled by --replay-loss-weight and --replay-loss-steps, so replay-sensitive gradients can be tested without using final task correctness as supervision. Epoch-10 MSE-only artifacts were extracted and a 25-endpoint reconstruction probe ran with no replay failures.

### Did Not Work / Caveats

- Epoch-10 MSE-only decoding still produced 0 convincing reconstructions in the probe; the one numeric-correct row reasoned about the wrong purchase scenario, so MSE around 0.398 is not yet enough for faithful replay.

### Left To Do

- Launch a separate MPS teacher-forced replay-KL run and compare checkpoint replay/faithfulness against the clean MSE-only epoch-10 baseline.

## 2026-05-18 - Deprecate prompt-prefix KL objective

### Worked

- Removed the active prompt-prefix frozen-LLM KL path from temporal RAE training, kept zero-valued legacy CLI compatibility for old launch scripts, and documented teacher-forced generated-token replay KL as the replay-sensitive auxiliary objective. Cleaned stale derived artifacts from the original full attached run while preserving raw caches and records needed by the active replay run.

### Did Not Work / Caveats

- The active launchd replay process was already started before this cleanup, so it still prints the old frozen-loss startup label until the next run, but it was launched with llm_loss_weight=0 and replay_loss_weight=0.01.

### Left To Do

- Continue monitoring the replay-KL run for its first startup event, heartbeat rows, and epoch-10 checkpoint before rerunning reconstruction scans.

## 2026-05-18 - Clarify replay KL step budget

### Worked

- Documented that replay_loss_steps is the number of teacher-forced generated tokens supervised by replay KL, so the current replay_loss_steps=4 run is a prefix-replay experiment rather than full generated-chain replay.

### Did Not Work / Caveats

- Full reasoning replay supervision would require using most or all original generated tokens, which is likely much more expensive than the current four-step diagnostic.

### Left To Do

- Let the current prefix-replay run reach an early checkpoint, then decide whether to launch a separate full-replay or longer-prefix run after measuring speed and reconstruction faithfulness.

## 2026-05-18 - Add temporal RAE resume and MPS cleanup

### Worked

- Added rae_temporal resume-from-checkpoint support, optional gradient clipping, periodic MPS cache cleanup, and an optional replay-loss batch subsampling knob. The epoch-12 MPS OOM was captured in the structured training log; epoch-10 checkpoint and PCA artifacts remain usable.

### Did Not Work / Caveats

- Batch size 2 still eventually OOMed inside teacher-forced replay KL on MPS, which points to memory accumulation/fragmentation from repeated frozen-LLM replay forwards rather than simple mini-batch size alone.

### Left To Do

- Relaunch from epoch 10 with batch size 1, gradient clipping, and per-batch MPS cache cleanup; use replay-loss subsampling only if OOMs persist.

## 2026-05-19 - Epoch 20 interpolation candidate-plan framing

### Worked

- Generated epoch-20 latent/PCA artifacts and a CPU-safe tiny interpolation rerun with candidate-plan quality reporting. Added per-row quality flags and a human-readable candidate-plan report so interpolation middle points are judged for coherence and completeness rather than endpoint target correctness.

### Did Not Work / Caveats

- A longer CPU replay sample with 128-token continuations was too slow while MPS training remained active and was stopped before producing rows. The tiny epoch-20 sample produced no inspectable candidate plans: all six rows showed placeholder drift and most were truncated.

### Left To Do

- Rerun candidate-plan interpolation on a later checkpoint, or pause the training run briefly to run a faster MPS replay sweep with longer continuations; use candidate quality as the first filter before manual interpretation.

## 2026-05-19 - Pivot to full-trajectory latent planning

### Worked

- Stopped the active prompt-cache replay-KL run after confirming it encoded prompt-prefix caches rather than full prompt+reasoning trajectories. Added trajectory cache collection via --cache-mode trajectory for both fresh prompt-cache collection and attaching caches to existing records with saved generation token IDs. Added latent-prompt-decoder-dataset to export latent/problem-prompt pairs for the prompt-decoder path.

### Did Not Work / Caveats

- The stopped run remains useful as a prompt-state continuation experiment, but it cannot validate full latent-plan interpolation because middle points do not expose their recovered problem prompts and the RAE was not trained on full reasoning trajectories.

### Left To Do

- Collect a new trajectory-cache dataset from the preserved Qwen records/generation token IDs, train a temporal RAE on full prompt+reasoning caches, and add an actual latent-to-prompt decoder head using the exported prompt-decoder dataset.

## 2026-05-19 - Trajectory cache smoke recapture

### Worked

- Ran a one-record Qwen trajectory-cache recapture smoke from the preserved prompt-cache records. The source attached bundles lacked exact generation_token_ids, so trajectory recapture now falls back to tokenizing saved output_text and marks generation_token_source=tokenized_output_text. Smoke artifact has 422 total cache tokens: 128 prompt tokens plus 294 continuation tokens.

### Did Not Work / Caveats

- The fallback is not guaranteed byte-for-byte identical to the original generation token ids when text normalization differs, so future fresh collections should prefer --cache-mode trajectory at generation time to save exact token ids.

### Left To Do

- Use the trajectory smoke to launch a small full-trajectory RAE check before scaling to all 1,319 records.

## 2026-05-19 - Full-trajectory RAE smoke10

### Worked

- Recaptured 10 Qwen GSM8K CoT records as full prompt+generated-reasoning trajectory caches using --cache-mode trajectory. Token lengths ranged from 259 to 487 total tokens, with 89-174 prompt tokens and 161-320 continuation tokens. A 20-epoch MPS temporal RAE smoke with latent_dim=256, hidden_dim=256, batch_size=1 completed without OOM, using tensor shape [10, 487, 6144] and about 0.97GB MPS memory. Training loss decreased from 1.00148 to 0.96076, and epoch-20 latent/PCA plus prompt-decoder dataset artifacts were generated.

### Did Not Work / Caveats

- This was only a plumbing/convergence smoke: reconstruction loss remains high, the dataset has only 10 records, and exact original generation token IDs were unavailable for old attached records, so continuation tokens were reconstructed by tokenizing saved output text.

### Left To Do

- Run a longer small trajectory RAE or scale trajectory recapture to a larger subset once we decide the fallback-tokenized outputs are acceptable; add full-trajectory-specific reconstruction/replay inspection rather than using prompt-continuation interpolation reports.

## 2026-05-19 - All-layer full-trajectory RAE subset smoke

### Worked

- Fixed compact cache serialization so full-trajectory cache files no longer store giant tensor-view backing storage. Captured a 200-record all-layer prompt+reasoning trajectory subset and trained rae_temporal on MPS for 20 epochs without OOM. Training was monotonic: loss 0.96495 to 0.65132, checkpointing and PCA/category artifacts worked, and reconstruction scans replayed without runtime failures.

### Did Not Work / Caveats

- The epoch-20 decoded reconstructions are not yet semantically usable: a 20 solved-endpoint CPU scan found 0 solved and 0 convincing reconstructions, with outputs mostly blank or short generic fragments. This means interpolation artifacts would not be scientifically interpretable yet.

### Left To Do

- Continue all-layer trajectory training much longer or improve capacity/objective before interpolation; compare against upper-layer artifacts only as a feasibility side check, not as a clean global-state interpolation result.

## 2026-05-19 - Solved-only all-layer trajectory RAE launch

### Worked

- Cleaned stale smoke/probe and prompt-prefix artifacts, preserved source records inside the active run, recaptured 498 source-correct Qwen GSM8K CoT plans as all-layer full prompt+reasoning trajectory caches, and launched a persistent MPS rae_temporal MSE-only warmup. Startup succeeded with temporal_matrix_shape [498, 499, 6144]; epoch 1 completed at loss 0.89692 and epoch 2 is decreasing with stable memory.

### Did Not Work / Caveats

- The active warmup has no teacher-forced replay KL gradients yet: replay_loss_weight=0, replay_loss_steps=0, and replay_gradients=false. This is intentional as a clean reconstruction baseline, but it does not yet include the behavior-aligned frozen-LLM token-level signal.

### Left To Do

- Let the MSE-only run reach an early checkpoint such as epoch 10, run reconstruction/PCA diagnostics, then resume from that checkpoint with conservative teacher-forced generated-token replay KL if reconstruction is stable.

## 2026-05-19 - Inspectable temporal RAE training status

### Worked

- Added a training-status CLI that renders structured rae_temporal JSONL telemetry into a markdown status file and readable line-oriented log.
- Started a lightweight launchd status mirror for the active solved498 run, refreshing the readable files every 30 seconds without touching training state.

### Did Not Work / Caveats

- The original launchd stdout/stderr files remain empty for the active training job because it was started through buffered conda run; the JSONL telemetry is the reliable source of truth.

### Left To Do

- Use conda run --no-capture-output and PYTHONUNBUFFERED=1 for future persistent training jobs, and keep the status mirror pattern for long runs.

## 2026-05-19 - Epoch 10 solved-only trajectory RAE analysis

### Worked

- Generated epoch-10 latent/PCA artifacts for runs/qwen_cache_cot_full_trajectory_all_solved498: 498 solved full-trajectory records, checkpoint epoch 10, PCA explained variance ratios PC1=0.6662 and PC2=0.2013.
- Ran a CPU reconstruction probe on 25 decoded endpoints with 192-token budget; cache validation passed with no replay runtime failures.
- Exported the epoch-10 latent-to-prompt-decoder dataset with 498 rows for the future problem-decoder path.

### Did Not Work / Caveats

- Epoch-10 decoded reconstructions are not semantically usable yet: 0/25 solved, 0/25 convincing, 22/25 empty outputs, and the remaining outputs were very short/generic fragments.

### Left To Do

- Do not run interpolation as interpretive evidence from epoch 10; let the MSE-only trajectory RAE train longer and retry reconstruction scans at later checkpoints before adding replay-KL or interpolation sweeps.

## 2026-05-19 - Scratch replay-KL trajectory RAE launch

### Worked

- Stopped the warm-start replay-KL branch, removed its bulky run directory, copied the solved498 full-trajectory caches into a clean scratch replay-KL run, and launched MPS training from random initialization.
- Scratch run uses latent_dim=1024 hidden_dim=1024, lr=1e-4, replay_loss_weight=0.001, replay_loss_steps=2, replay_loss_every_n_batches=2, batch_size=1, grad_clip_norm=0.5, and checkpoints every 5 epochs.

### Did Not Work / Caveats

- The previous warm-start replay-KL branch was stable, but it was not the intended scratch experiment; it was removed to keep the active run unambiguous.

### Left To Do

- Monitor the scratch replay-KL run through its readable status/log files until replay target loading completes and epoch-1 heartbeats appear; compare early replay KL/MSE trends against the removed warm-start notes only as informal context.

## 2026-05-20 - Epoch 20 scratch replay-KL trajectory RAE analysis

### Worked

- Generated epoch-20 latent/PCA artifacts for the scratch replay-KL solved498 full-trajectory run. All 498 solved trajectory records encoded successfully; PCA PC1/PC2 explained variance ratios were 0.6832 and 0.1946.
- Ran a CPU reconstruction probe on 50 decoded endpoints with 256-token budget; cache validation had 0 replay runtime failures.
- Exported the epoch-20 latent-to-prompt-decoder dataset with 498 rows.

### Did Not Work / Caveats

- Epoch-20 decoded reconstructions are still not semantically usable: 0/50 solved, 0/50 convincing, 46/50 empty outputs, and the non-empty outputs were short or task-drifted fragments.

### Left To Do

- Do not generate interpretive interpolation artifacts from epoch 20. Let the scratch replay-KL run continue to later checkpoints; retry reconstruction scans at epoch 50 or consider increasing replay steps/weight only after checking whether MSE continues improving without OOM.

## 2026-05-20 - Fix full-trajectory replay KL prefix slicing

### Worked

- Found that replay KL for full-trajectory caches was teacher-forcing after the completed prompt+solution cache. Updated training and replay diagnostics to slice trajectory caches back to the prompt prefix before replaying generated reasoning tokens.

### Did Not Work / Caveats

- Epoch-20 reconstruction scans that continued after complete trajectories were misleading: original full-trajectory caches often immediately emit EOS as well, so empty continuations were not by themselves evidence of decoder collapse.

### Left To Do

- Restart the replay-KL scratch run from a clean directory with the corrected objective, then rerun epoch-20 latent/reconstruction analysis.

## 2026-05-20 - Fix trajectory reconstruction scan replay boundary

### Worked

- Found that reconstruction scans for full-trajectory caches were replaying from the end of the completed prompt+solution cache. Updated cache injection to replay trajectory bundles from the prompt boundary by slicing the cache prefix and recomputing continuation logits from the decoded prompt prefix. Original trajectory caches now reproduce source reasoning under the fixed path.

### Did Not Work / Caveats

- Epoch-10 decoded reconstructions still do not solve tasks; after the replay fix they emit non-empty generic reasoning templates, which suggests the codec is not yet reconstructing task-specific trajectory state at this loss level.

### Left To Do

- Use prompt-boundary reconstruction scans for future full-trajectory checkpoints. Recheck at epoch 20/30 and consider stronger replay loss or more replay steps only if task-specificity remains absent after MSE improves.

## 2026-05-20 - Fix training replay bundle trajectory metadata

### Worked

- Found a second replay-boundary bug in temporal RAE training: target replay logits used the full trajectory bundle metadata, but predicted replay logits used stripped bundles without generation_config, so reconstructed caches were replayed from the full trajectory length. Preserved generation_config for training replay bundles and added a regression test that all replay-gradient calls use the trajectory prompt boundary.

### Did Not Work / Caveats

- The active corrected run up to epoch 15 was still trained with this target/predicted replay-boundary mismatch and should not be treated as a valid replay-KL experiment.

### Left To Do

- Restart the full-trajectory replay-KL run from scratch using the fixed training metadata path, then rerun prompt-boundary reconstruction analysis at epoch 10/20.

## 2026-05-20 - Include prompt-boundary first-token replay KL

### Worked

- Found that teacher-forced replay KL compared logits only after consuming generated tokens, so it missed the prompt-boundary distribution that selects the first reasoning token. Updated training and replay diagnostics to produce one KL target per generated token starting at the prompt boundary, with regression tests for trajectory prefix positions.

### Did Not Work / Caveats

- The fixedmeta run had correct trajectory metadata but still lacked first-token replay KL, so it should be restarted before treating results as valid.

### Left To Do

- Restart full-trajectory replay-KL training once more with first-token replay included; analyze at epoch 10/20 using prompt-boundary reconstruction scans.

## 2026-05-20 - Audit replay KL numerics and fidelity metrics

### Worked

- Ran another full-trajectory audit while firsttok training continued. Verified a real source-cache oracle ranks the first four stored generated tokens at rank 1 from the prompt boundary. Updated replay KL to compute teacher probabilities in float32 instead of model dtype, and fixed replay-fidelity diagnostics so first-token and second-token source ranks are measured against the correct replay steps after the first-token KL shift.

### Did Not Work / Caveats

- No additional boundary or cache-shape bug was found in training/reconstruction/interpolation paths during this pass. Existing private _next_logits_after_cache_token remains unused.

### Left To Do

- Let the firsttok run continue; analyze the first complete checkpoint with prompt-boundary reconstruction scans.

## 2026-05-20 - Audit checkpoint and replay KL logging reliability

### Worked

- While the firsttok full-trajectory replay-KL run continued, audited checkpoint writing and sparse replay-KL accounting. Added atomic checkpoint saves so analysis cannot catch half-written .pt files, and split replay-KL logging into sampled KL plus effective sparse-objective KL.

### Did Not Work / Caveats

- These reliability fixes do not affect the already-running Python process; they are for future starts/checkpoints after restart. No new cache-boundary or replay-target bug was found.

### Left To Do

- Let the firsttok run reach a complete checkpoint, then analyze with prompt-boundary reconstruction scans; use the sampled/effective KL distinction when interpreting future logs.

## 2026-05-21 - Overfit capacity probes for full-trajectory temporal RAE

### Worked

- Created copied-cache overfit probes for 8 solved trajectories and one solved trajectory while the main firsttok run continued. The 8-record replay-KL probe with latent/hidden 2048, replay_steps=8, replay_weight=0.01 reduced replay KL early but did not reconstruct solved plans after 10 epochs. The one-record MSE-only probe also failed to overfit, ending near normalized MSE 0.98 after 25 epochs.

### Did Not Work / Caveats

- The current latent-only TemporalLSTMAutoEncoder did not memorize even one full trajectory under MSE-only training, so poor reconstruction is likely an architecture/optimization/capacity issue rather than only insufficient dataset epochs or final-answer bias.

### Left To Do

- Try a stronger trajectory codec: chunked/hierarchical latents or a decoder with token-wise cross-attention/MLP conditioning instead of a single latent repeated through an LSTM. Use one-record overfit as the required gate before restarting large-scale training.

## 2026-05-21 - Replace global trajectory latent with structured temporal codec

### Worked

- Found the one-record full-trajectory overfit failure was architectural: the single global TemporalLSTMAutoEncoder stayed near normalized MSE 1.0. Added a structured temporal chunk codec and made rae_temporal use it by default with tokenwise chunk_size=1. The one-record gate now drops immediately, e.g. default rae_temporal latent_dim=512 hidden_dim=1024 reached normalized MSE 0.431 by epoch 40, and a larger tokenwise probe reached 0.290 by epoch 120. Checkpoint, latent-analysis, and interpolation paths now preserve structured latent shapes for decoding while flattening only for PCA/distance.

### Did Not Work / Caveats

- Chunk sizes larger than one and wider hidden dimensions did not solve replay by themselves: chunk_size=16 reached about 0.406 after 60 epochs, and chunk_size=8 with a larger model still ended around 0.400 after 300 epochs. Behavioural replay from these intermediate-MSE checkpoints can still emit blank or generic continuations, so lower reconstruction error is still required before interpreting interpolations.

### Left To Do

- Use the structured rae_temporal codec for the next real run, starting with solved-only full trajectories and conservative replay KL only after the MSE-only gate reaches much lower error. Treat old single-global-latent trajectory runs as deprecated diagnostics, not evidence about the fixed codec.

## 2026-05-21 - Add token-level prompt decoder path

### Worked

- Added a real task/prompt decoder path rather than a character-level toy: `latent-prompt-decoder-dataset` now exports prompt token IDs from trajectory cache bundles, and `latent-prompt-decoder-train` trains a token-level decoder where learned prompt-position queries attend to structured latent memory. The decoder trains over the compact set of prompt token IDs observed in the dataset and stores the mapping back to original tokenizer IDs.
- Smoke-tested the path on epoch-570 solved498 structured latents: 498 rows, latent shape `[498, 499, 512]`, compact prompt vocabulary size 2,837 over original Qwen token IDs, one CPU smoke epoch with hidden_dim 64 / one layer completed and wrote model, decoded-token rows, and training history.
- Updated `docs/KV_RAE_FLOW.md` to distinguish the plan/cache decoder from the task/prompt decoder and to require token-level prompt recovery metrics before using recovered prompts for interpolation interpretation.

### Did Not Work / Caveats

- A first attempt at a character-level prompt decoder was rejected as too toy-like and removed before commit. A direct full-Qwen-vocabulary projection was also too slow for routine CPU checks, so the implemented prompt decoder uses a compact observed-token vocabulary and stores the original-token mapping.

### Left To Do

- Train the prompt decoder for a meaningful number of epochs on a stable RAE checkpoint, then evaluate exact prompt-token recovery and use recovered prompts alongside decoded cache plans for interpolation analysis.

## 2026-05-22 - Decode interpolation prompts from prompt-decoder checkpoints

### Worked

- Added periodic prompt-decoder checkpoints (`prompt_token_decoder_latest.pt` and numbered epoch snapshots) so long CPU decoder runs become inspectable before the final epoch.
- Added `latent-prompt-decode-interpolations`, which rebuilds interpolation alpha latents from `checkpoint_latents.pt` and `interpolation_pairs.jsonl`, then decodes each alpha into approximate prompt tokens with a saved token-level prompt decoder.
- Smoke-ran the command on the epoch-570 interpolation line with the old one-epoch smoke decoder; it wrote 5 decoded-prompt rows and a markdown table, confirming the artifact path works end to end.

### Did Not Work / Caveats

- The one-epoch smoke decoder output is not semantically meaningful. It mostly repeats common prompt tokens, so it should be treated as a plumbing check only. The already-running full prompt decoder was launched before periodic checkpointing existed, so it cannot be used until it finishes or is relaunched with the updated code.

### Left To Do

- Use `prompt_token_decoder_latest.pt` from the next full prompt-decoder run to decode interpolation prompts, then compare endpoint prompt recovery before reading middle-alpha prompts as candidate recovered tasks.

## 2026-05-22 - Add joint prompt-head and replay-KL objective

### Worked

- Stopped the separate MSE-only structured RAE and standalone prompt-decoder jobs to prepare a single joint run.
- Added an auxiliary token-level prompt head directly to `rae_temporal` training, controlled by `--prompt-loss-weight` and `--prompt-loss-*` options. The head uses the same structured latent produced by the RAE encoder and optimizes prompt-token CE/KL against source prompt token IDs from cache bundles.
- The existing frozen local LLM generated-token replay KL remains the plan/cache behavioural objective. When both losses are enabled, the objective is masked temporal KV MSE plus teacher-forced replay KL plus prompt-token CE/KL.
- RAE checkpoints now include the prompt-head state/config, so later interpolation artifacts can decode both recovered prompts and reconstructed plan/cache trajectories from the same latent points. Full tests passed: 106.

### Did Not Work / Caveats

- This is a new joint objective, so the first MPS launch should be watched for memory pressure. Replay KL is still the expensive part because it routes reconstructed caches through the frozen local LLM.

### Left To Do

- Launch the joint full run conservatively on MPS with replay KL sampled every few batches, then inspect epoch-1/epoch-10 logs for prompt CE, replay KL, and MSE balance before increasing replay steps or weights.

## 2026-05-22 - Add transformer temporal point codec

### Worked

- Implemented rae_temporal_transformer as a transformer encoder/decoder trajectory codec that defaults to one latent point per full temporal KV trajectory, preserving PCA/interpolation semantics while providing a stronger sequence model than the old global LSTM bottleneck. Added checkpoint metadata, CLI flags for transformer heads and latent-token capacity, docs, and tests.

### Did Not Work / Caveats

- This is implementation and smoke coverage only; no full MPS training run has been launched yet, and the active joint LSTM/chunk run remains untouched.

### Left To Do

- Run a small transformer point-codec overfit gate, then launch a solved498 full-trajectory run with --method rae_temporal_transformer --temporal-latent-tokens 1 and compare reconstruction, replay KL, prompt decode, and interpolation quality against the structured chunk codec.

## 2026-05-23 - Add hybrid KV cosine reconstruction loss

### Worked

- Added --cosine-loss-weight for temporal RAE training so normalized masked MSE can be complemented by masked temporal KV cosine distance; telemetry, readable status logs, checkpoints, artifacts, CLI help, and tests now report the cosine component separately from replay KL and prompt CE.

### Did Not Work / Caveats

- Cosine is not a replacement for MSE because direction-only matching cannot preserve cache scale; the real run still needs checkpoint analysis before treating interpolation quality as meaningful.

### Left To Do

- Relaunch the transformer point-codec run with MSE plus cosine plus sampled teacher-forced replay KL and prompt CE, then analyze early checkpoints for reconstruction faithfulness and latent interpolation quality.

## 2026-05-23 - Add setup-stage training telemetry

### Worked

- Added setup_start, temporal-matrix, normalization, model-construction, replay-target progress, and prompt-target setup events to temporal RAE JSONL logs so long runs are inspectable before epoch 1 starts. Readable training-status output now renders setup rows too.

### Did Not Work / Caveats

- The first hybrid restart showed that stdout alone is still too sparse during expensive setup; JSONL setup rows are the reliable path.

### Left To Do

- Restart the transformer point-codec run from scratch and confirm setup events plus cosine loss appear in the logs before allowing the run to continue.

## 2026-05-23 - Fix silent reconstruction replay scans

### Worked

- Added pre-row progress messages to reconstruction scans and interpolation replay, including replay model loading and max_new_tokens, so slow CPU replay no longer appears hung before the first completed row. Verified epoch-30 one-row reconstruction scan with max_new_tokens=16 completed and wrote artifacts.

### Did Not Work / Caveats

- The default reconstruction/interpolation replay budget still expands to max(512, source_generated_tokens), which is too expensive to run casually while MPS training is active.

### Left To Do

- Use explicit small --max-new-tokens values for smoke replay scans during active training, and reserve full-budget reconstruction/interpolation sweeps for quieter windows.
