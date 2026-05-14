# KV RAE End-to-End Flow

This note describes the current `rae_lstm` point-codec path in the latent-KV
planning prototype. The key contract is:

```text
entire prompt KV-cache sequence -> one latent plan point -> decoded KV-cache sequence
```

The RAE decoder reconstructs transformer KV state. It does not directly decode
an answer string. The local LLM then performs normal autoregressive decoding
from the reconstructed cache, and the task verifier decides whether that replay
or generated continuation solved the problem.

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
        prompt KV-cache sequence / planning-state trace
        {
          layer_1:  K,V over tok_1..tok_N
          layer_2:  K,V over tok_1..tok_N
          ...
          layer_L:  K,V over tok_1..tok_N
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
                    | flatten KV cache    |
                    +---------------------+
                               |
                               v
        flat cache vector x
        [all selected layer/token/head K,V values ...]
                               |
                               v
                    +---------------------+
                    | normalize x         |
                    | (x - mean) / std    |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | split into chunks   |
                    +---------------------+
                               |
                               v
        RAE input sequence
        [chunk_1, chunk_2, ..., chunk_T]
                               |
                               v
                    +---------------------+
                    | LSTM encoder        |
                    | reads chunk sequence|
                    +---------------------+
                               |
                               v
                    final encoder hidden state h_T
                               |
                               v
                    +---------------------+
                    | linear h_T -> z     |
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
                    | linear z -> h_0     |
                    | decoder init state  |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    | LSTM decoder        |
                    | emits chunk sequence|
                    +---------------------+
                               |
                               v
        reconstructed RAE sequence
        [chunk'_1, chunk'_2, ..., chunk'_T]
                               |
                               v
                    +---------------------+
                    | concat chunks       |
                    | crop padding        |
                    | denormalize         |
                    +---------------------+
                               |
                               v
        reconstructed flat cache vector x'
                               |
                               v
                    +---------------------+
                    | unflatten by saved  |
                    | KV shapes           |
                    +---------------------+
                               |
                               v
        reconstructed KV-cache sequence
        {
          layer_1:  K',V' over tok_1..tok_N
          layer_2:  K',V' over tok_1..tok_N
          ...
          layer_L:  K',V' over tok_1..tok_N
        }
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
  -> original KV-cache sequence
  -> verifier-labelled cache datapoint
  -> flatten / normalize / chunk
  -> LSTM encoder
  -> latent plan point z
  -> LSTM decoder
  -> reconstructed KV-cache sequence
  -> LLM continuation from cache
  -> output text
  -> verifier outcome
```

## Two Decoders

```text
RAE decoding:
  z -> reconstructed KV cache

LLM decoding:
  reconstructed KV cache -> next tokens -> output text
```

Keeping those separate is important. The RAE learns a bottlenecked cache-state
representation. Behavioural usefulness is measured only after local LLM replay
and task verification.

## Current Qwen Smoke Dimensions

For the local Qwen2.5-0.5B-Instruct smoke runs, one 128-token prompt cache over
all 24 layers flattens to 786,432 values. With `chunk_dim=8192`, the RAE sees a
96-step chunk sequence. Tiny smoke settings have used `latent_dim=16` and
`hidden_dim=32`; larger runs should increase these deliberately and compare
against retrieval/PCA baselines.