# KV RAE End-to-End Flow

This note describes the current `rae_temporal` point-codec path in the
latent-KV planning prototype. The key contract is:

```text
entire prompt KV-cache temporal sequence -> one latent plan point -> decoded KV-cache temporal sequence
```

The RAE decoder reconstructs transformer KV state. It does not directly decode
an answer string. The local LLM then performs normal autoregressive decoding
from the reconstructed cache, and the task verifier decides whether that replay
or generated continuation solved the problem.

During training, the local LLM may also be used as a frozen differentiable
critic. Gradients flow through reconstructed KV tensors into the RAE, but the
LLM weights stay frozen. This LLM loss is optional and auxiliary; the codec
still learns sequence-to-point-to-sequence reconstruction.

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
        prompt KV-cache temporal sequence / planning trace
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
                    | LSTM encoder        |
                    | reads token time    |
                    +---------------------+
                               |
                               v
        encoder summary: last hidden + masked mean encoded state
                               |
                               v
                    +---------------------+
                    | linear summary -> z |
                    +---------------------+
                               |
                               v
        one latent plan point
        z: [latent_dim]
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
        compare original and reconstructed prompt-state transitions

        for selected prompt positions t:
          original prefix cache state_1..state_t
          reconstructed prefix cache state'_1..state'_t
          same next prompt token tok_{t+1}
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

        Important: this is not answer-string decoding and not continuation
        prediction as the main task. It is a differentiable state-quality loss
        on the reconstructed temporal cache sequence.
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
  -> LSTM encoder over token time
  -> latent plan point z
  -> LSTM decoder over token time
  -> reconstructed temporal KV-cache sequence
  -> optional frozen-LLM prompt-transition KL gradients during training
  -> LLM continuation from reconstructed cache
  -> output text
  -> verifier outcome
```

## Two Decoders

```text
RAE decoding:
  z -> reconstructed KV cache sequence

LLM decoding:
  reconstructed KV cache -> next tokens -> output text
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
only checked against the endpoint target for that replay context. Intermediate
points have no dataset labels, so their outputs must be inspected as transition
evidence rather than benchmark examples.

Interpolation artifacts should always preserve the endpoint prompts, original
outputs, decoded intermediate outputs, alpha values, endpoint categories, and
cache-validation status. This makes it possible to review whether the path
appears to preserve, blend, or corrupt reasoning structure.

## Training Objective

The base objective is masked temporal reconstruction MSE over real prompt-token
states:

```text
loss = MSE(original_temporal_cache, reconstructed_temporal_cache)
```

When `--llm-loss-weight` is greater than zero, training adds a frozen-LLM
prompt-transition KL term:

```text
loss = reconstruction_mse
     + llm_loss_weight * KL(
         frozen_llm_logits(original_prefix_cache, next_prompt_token),
         frozen_llm_logits(reconstructed_prefix_cache, next_prompt_token)
       )
```

The LLM receives gradients through `past_key_values`, but all LLM parameters are
frozen. Only the RAE encoder/decoder weights update.

## Variable Source Lengths

Plan/cache sequences keep their original token lengths. Different prompts and
different reasoning continuations naturally produce different token counts, and
those lengths are part of the behavioural trace. For batching, the codec pads
the token axis to a common maximum length and stores token masks, original
vector lengths, and cache shapes. The scientific object remains one latent
point per original variable-length temporal sequence.

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
