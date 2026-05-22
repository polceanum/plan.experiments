# KV RAE End-to-End Flow

This note describes the `rae_temporal` trajectory-codec paths in the latent-KV
planning prototype. The original prompt-cache path used this contract:

```text
problem prompt KV-cache temporal sequence
  -> one latent problem-state point
  -> decoded KV-cache temporal sequence
  -> local LLM continuation
```

This is useful, but it is not the full latent-plan trajectory experiment. The
full-trajectory path we now want to test uses:

```text
problem prompt + generated reasoning KV-cache temporal sequence
  -> structured latent trajectory state
  -> decoded KV-cache temporal sequence
  -> replay/inspect recovered plan
```

Earlier `rae_temporal` experiments collapsed the whole temporal cache sequence
into a single global latent vector. That shape was too destructive for
full-trajectory caches: a one-record MSE overfit probe stayed near normalized
MSE 1.0. The current default `rae_temporal` codec is therefore structured: it
encodes token-time chunks, with the default `--temporal-chunk-size 1` creating
one latent slot per temporal KV token. PCA and pair selection can flatten that
structured latent for geometry, but decoding preserves the chunk/token latent
shape.

The RAE decoder reconstructs transformer KV state. It does not directly decode
an answer string. The local LLM then performs normal autoregressive decoding
from the reconstructed cache, and the task verifier decides whether that replay
or generated continuation solved the problem.

## Inspectable Training Logs

Long `rae_temporal` runs write authoritative telemetry to
`<run>/compressions/rae_temporal_training.jsonl`. Plain launchd stdout files may
remain empty when the process is started through `conda run`, because stdout can
be captured or buffered outside the training loop. Treat the JSONL file as the
source of truth.

For human inspection, render the JSONL into a compact markdown status file and a
readable line-oriented log:

```bash
conda run -n orpheus python -m latent_kv training-status \
  --run runs/<run_id> \
  --method rae_temporal \
  --output runs/<run_id>/rae_temporal_status.md \
  --readable-log runs/<run_id>/rae_temporal_readable.log
```

Persistent runs should either use `conda run --no-capture-output` and
`PYTHONUNBUFFERED=1` for direct stdout visibility, or run a tiny periodic
`training-status` mirror alongside the training job. The mirror does not touch
model state; it only rewrites the status/readable files from JSONL telemetry.

During training, the local LLM may also be used as a frozen differentiable
critic through teacher-forced generated-token replay KL. Gradients flow through
reconstructed KV tensors into the RAE, but the LLM weights stay frozen. This is
optional and auxiliary; the codec still learns temporal cache reconstruction
from a structured latent trajectory state. It is not a final-answer correctness
loss.

## Full Flow

```text
                         problem instance
                               |
                               v
                    +---------------------+
                    | format prompt       |
                    | standard / CoT / ...|
                    +---------------------+
                               |
                               v
                    prompt token sequence
                    [tok_1, tok_2, ..., tok_N]
                               |
                               v
        +------------------------------------------------+
        | local LLM forward pass over prompt tokens      |
        | transformer layers 1..L                        |
        | attention creates K,V states per layer/token   |
        +------------------------------------------------+
                               |
                               v
        prompt KV-cache temporal sequence / problem state
        [state_1, state_2, ..., state_N]

        where state_t contains every selected layer/head K,V value
        for prompt token tok_t:

        {
          layer_1:  K,V for tok_t
          layer_2:  K,V for tok_t
          ...
          layer_L:  K,V for tok_t
        }
                               |
                               v
                    +---------------------+
                    | save cache bundle   |
                    | + replay metadata   |
                    | + verifier labels   |
                    +---------------------+
                               |
                               v
          source cache datapoint: did this sequence solve?
          correct / incorrect, target, parsed answer, protocol
                               |
                               v
================================================================================
                            RAE POINT CODEC
================================================================================
                               |
                               v
                    +---------------------+
                    | align cache shapes  |
                    | pad token axis only |
                    +---------------------+
                               |
                               v
        temporal cache matrix
        [state_1, state_2, ..., state_N, pad, ...]
        shape: [tokens, token_state_dim]
                               |
                               v
                    +---------------------+
                    | token mask          |
                    | marks real tokens   |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | normalize states    |
                    | using real tokens   |
                    +---------------------+
                               |
                               v
        RAE input sequence
        [state_1_norm, state_2_norm, ..., state_N_norm]
                               |
                               v
                    +---------------------+
                    | structured encoder  |
                    | reads token chunks  |
                    +---------------------+
                               |
                               v
        per-chunk masked encoded state
                               |
                               v
                    +---------------------+
                    | chunk encoder -> z  |
                    +---------------------+
                               |
                               v
        structured latent plan state
        z: [num_chunks, latent_dim]
        carries source_labels:
        task_id, correct, target, parsed_answer, prompt_protocol
                               |
                               v
                    +---------------------+
                    | z -> decoder state  |
                    | + temporal positions|
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | LSTM decoder        |
                    | emits token states  |
                    +---------------------+
                               |
                               v
        reconstructed temporal sequence
        [state'_1, state'_2, ..., state'_N, pad, ...]
                               |
                               v
                    +---------------------+
                    | apply token mask    |
                    | denormalize states  |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | restore layer/K/V   |
                    | tensor layout       |
                    +---------------------+
                               |
                               v
        reconstructed KV-cache temporal sequence
        {
          layer_1:  K',V' over tok_1..tok_N
          layer_2:  K',V' over tok_1..tok_N
          ...
          layer_L:  K',V' over tok_1..tok_N
        }
                               |
                               v
================================================================================
                       OPTIONAL FROZEN-LLM TRAINING SIGNAL
================================================================================
                               |
                               v
                    +---------------------+
                    | freeze local LLM    |
                    | no weight updates   |
                    +---------------------+
                               |
                               v
        compare original and reconstructed generated-token transitions

        for selected generated reasoning positions t:
          original prefix cache state_1..state_t
          reconstructed prefix cache state'_1..state'_t
          same next original generated token tok_{t+1}
                               |
                               v
                    +---------------------+
                    | frozen LLM forward  |
                    | with original cache |
                    +---------------------+
                               |
                               v
                    original transition logits
                               |
                               v
                    +---------------------+
                    | frozen LLM forward  |
                    | with decoded cache  |
                    +---------------------+
                               |
                               v
                    reconstructed transition logits
                               |
                               v
                    +---------------------+
                    | KL(original || rec) |
                    | + reconstruction MSE|
                    +---------------------+
                               |
                               v
        gradients flow through decoded KV cache into the RAE only

        Important: this is not final-answer supervision and not a verifier
        reward. It is a differentiable state-quality loss on the reconstructed
        temporal cache sequence: does the decoded cache induce the same local
        next-token distribution as the original solved trajectory?
                               |
                               v
================================================================================
                              LLM REPLAY
================================================================================
                               |
                               v
                    +---------------------+
                    | validate cache      |
                    | shapes finite meta  |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | resume local LLM    |
                    | from decoded KV     |
                    +---------------------+
                               |
                               v
        generated continuation token sequence
        [out_tok_1, out_tok_2, ..., out_tok_M]
                               |
                               v
                    +---------------------+
                    | decode tokens       |
                    | to output text      |
                    +---------------------+
                               |
                               v
        model answer / reasoning continuation
                               |
                               v
                    +---------------------+
                    | task verifier       |
                    | numeric / solver /  |
                    | unit tests / etc.   |
                    +---------------------+
                               |
                               v
        behavioural outcome of the decoded/generated plan
        correct? valid? novel? structurally plausible?
```

## Short Version

```text
Problem
  -> prompt text
  -> prompt tokens
  -> LLM prompt forward pass
  -> original temporal KV-cache sequence
  -> verifier-labelled cache datapoint
  -> align / temporalize / normalize
  -> structured chunk/token encoder over token time
  -> latent plan state z
  -> chunk/token decoder over token time
  -> reconstructed temporal KV-cache sequence
  -> optional frozen-LLM generated-token replay KL gradients during training
  -> LLM continuation from reconstructed cache
  -> output text
  -> verifier outcome
```

This short version is the prompt-cache path. It compresses the problem state and
then asks the LLM to generate a continuation from that state. Interpolating
these points may produce changed problem states and therefore changed
continuations, but there is no direct decoded problem prompt for the middle
points. That makes it hard to judge whether an interpolated continuation is
correct, because the problem it belongs to is latent.

## Full-Trajectory Path

For full latent-plan experiments, collect cache bundles with
`--cache-mode trajectory`:

```text
Problem
  -> prompt text
  -> local LLM generates reasoning/answer tokens
  -> concatenate prompt tokens + generated tokens
  -> LLM forward pass over the full token sequence
  -> full temporal KV-cache trajectory
  -> RAE encodes/decodes the whole prompt+plan cache sequence
```

The resulting cache sequence includes both the problem prompt prefix and the
generated reasoning continuation. This matches the scientific object:

```text
[problem statement, reasoning steps, answer]
  -> structured latent trajectory state
  -> reconstructed [problem statement, reasoning steps, answer] cache state
```

The decoded cache is still not literally text by itself; KV caches are not
invertible transcripts. But this path trains the bottleneck on the full
trajectory rather than only the prompt prefix, so reconstructions and
interpolations should preserve much more of the reasoning plan structure.

For interpolation experiments, a solved-only training subset is often the
cleaner target. It learns a manifold of successful reasoning trajectories,
reduces compute, and avoids spending capacity on source generations whose
plans already failed. Mixed solved/failed runs can still be useful for
diagnostics, but they are harder to interpret as a latent space of good plans.

## Prompt Decoder Path

Prompt-cache interpolation can still be useful if we also learn to decode the
problem prompt from each latent point:

```text
z -> prompt decoder head -> recovered problem prompt
z -> RAE decoder -> KV problem state / trajectory state
recovered prompt + decoded state -> generated plan
```

This gives interpolated points an inspectable problem statement. Then middle
points can be evaluated as ordinary candidate tasks: does the decoded/recovered
problem make sense, and does the generated plan solve that recovered problem?

The first implementation artifact for this path is
`latent-prompt-decoder-dataset`, which exports latent vectors paired with their
source problem prompts and prompt token IDs recovered from the trajectory cache
bundles. The paired training command is `latent-prompt-decoder-train`, which
trains a token-level latent-to-prompt decoder head. Prompt-decoder training now
writes `prompt_token_decoder_latest.pt` plus numbered checkpoints, so long
decoder runs can be inspected without waiting for the final epoch.

This task decoder is intentionally not a character-level toy. It follows the
same structured-decoder principle as the trajectory RAE: latent chunks are
projected as a memory sequence, learned prompt-position queries attend to that
memory, and a token classifier predicts the original prompt token IDs. Prompt
recovery should be reported with token accuracy and exact prompt-token match
before decoded prompts are used to interpret interpolation points.

Once a prompt-decoder checkpoint exists, `latent-prompt-decode-interpolations`
can project each stored interpolation alpha back into approximate prompt tokens.
Those decoded prompts are interpretive artifacts, not ground truth. A low-loss
prompt decoder should make endpoint prompts recoverable first; only then should
middle-alpha decoded prompts be read as candidate recovered tasks.

## Two Decoders

```text
Plan/cache decoder:
  z -> RAE decoder -> reconstructed full-trajectory KV cache sequence

Task/prompt decoder:
  z -> token-level prompt decoder -> recovered problem prompt tokens

Local LLM replay:
  recovered or endpoint prompt context + reconstructed KV cache -> output text
```

Keeping those separate is important. The RAE learns a bottlenecked cache-state
representation. Behavioural usefulness is measured only after local LLM replay
and task verification.

## Latent Interpolation

Interpolating between two RAE latent points creates an interpolated cache state,
not a new natural-language problem. A point on the line between solved tasks A
and B should therefore be decoded as:

```text
z_alpha = (1 - alpha) * z_A + alpha * z_B
z_alpha -> reconstructed KV cache sequence
```

The decoded cache is then replayed under endpoint prompt contexts. In the
current workflow every interpolation step is replayed once with A's prompt
tokens/logits and once with B's prompt tokens/logits. Automated correctness is
only checked against the endpoint target for that replay context, and this is
not the main success criterion for middle points. Intermediate points have no
dataset labels and may represent different latent tasks or partial plans, so
their outputs should be inspected for coherence, completeness, arithmetic
self-consistency, and structural drift rather than forced agreement with either
endpoint answer.

This endpoint-context replay is an implementation limitation of the prompt-cache
path: an interpolated prompt-state cache needs a prompt/logit context to become
text. Conceptually, the latent line is still A -> alpha points -> B. If the same
alpha renders differently under A and B contexts, treat that as instability or
context dependence in the latent point.

Interpolation artifacts should always preserve the endpoint prompts, original
outputs, decoded intermediate outputs, alpha values, endpoint categories, and
cache-validation status. Candidate-plan reports should also include local
quality flags, such as whether the continuation is inspectable, appears
truncated, contains arithmetic structure, or drifts into a generic placeholder
template. This makes it possible to review whether the path appears to preserve,
blend, or corrupt reasoning structure.

## Training Objective

The base objective is masked temporal reconstruction MSE over real prompt-token
and generated-token KV states:

```text
loss = MSE(original_temporal_cache, reconstructed_temporal_cache)
```

When `--replay-loss-weight` is greater than zero, training adds a frozen-LLM
teacher-forced replay KL term over generated reasoning tokens:

```text
loss = reconstruction_mse
     + replay_loss_weight * KL(
         frozen_llm_logits(original_cache, next_original_generated_token),
         frozen_llm_logits(reconstructed_cache, next_original_generated_token)
       )
```

The LLM receives gradients through `past_key_values`, but all LLM parameters are
frozen. Only the RAE encoder/decoder weights update. The older prompt-prefix
transition KL path is deprecated because it can reward matching prompt-state
continuations rather than reconstructing the generated reasoning trajectory.

`--replay-loss-steps` controls how many generated tokens receive this
teacher-forced KL supervision. A small value, such as the current exploratory
`4`, is a prefix-replay diagnostic: it checks whether the reconstructed cache
induces the same first few generated-token transitions. It does not force a
full reasoning replay. For GSM8K CoT records whose generations are often
hundreds of tokens long, full replay supervision would require a much larger
step budget, such as the original generation length or the collection cap. That
is scientifically closer to "replay the whole reasoning process," but it is
also much more expensive because each supervised token requires an additional
frozen-LLM forward through the reconstructed cache.

The current recommended curriculum is:

```text
1. Train an all-layer, full-trajectory, solved-only RAE with cache MSE only.
2. Inspect reconstruction loss, checkpoint health, and decoded-cache replay
   diagnostics at early checkpoints.
3. Resume from a stable MSE checkpoint with a small teacher-forced replay KL
   weight, e.g. --replay-loss-weight 0.01 and --replay-loss-steps 4.
4. Increase replay_loss_steps only if memory and speed remain stable.
```

This keeps final task correctness out of the loss while adding gradients that
are closer to the behavioural question: whether the reconstructed cache drives
the frozen local LLM through the same reasoning-token transitions.

## Variable Source Lengths

Plan/cache sequences keep their original token lengths. Different prompts and
different reasoning continuations naturally produce different token counts, and
those lengths are part of the behavioural trace. For batching, the codec pads
the token axis to a common maximum length and stores token masks, original
vector lengths, and cache shapes. The scientific object is one structured latent
trajectory per original variable-length temporal sequence, not one global vector.

Replay should use the source record's original generation budget unless an
experiment explicitly overrides it. Cache bundles for new prompt-cache
collections store generated token IDs and generation config so replay fidelity
can be checked against the original local generation path.

## Current Qwen Smoke Dimensions

For the local Qwen2.5-0.5B-Instruct success20 cache run, the temporal codec sees
approximately:

```text
seq_len = 191 prompt tokens after token-axis padding
token_dim = 6144 KV values per token state
```

Larger runs should increase `latent_dim`, `hidden_dim`, and epochs deliberately,
and compare against retrieval/PCA baselines plus original-cache replay.
