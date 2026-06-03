# Latent Planning Design Notes

## Why It Matters Here

This project treats transformer KV caches as latent planning states. The useful
research pattern is not just "sequence compression"; it is testing whether a
latent bottleneck can preserve high-level plan structure, long-range
dependencies, verification behaviour, and quality-relevant structural
properties while producing novel valid candidates.

## Design Principles To Carry Forward

- Train and evaluate on well-formed structured trajectories first. In this repo,
  that means exact-verifier tasks like Game of 24, Hanoi, Sudoku, code tests, and
  GSM8K answer checks before open-ended tasks.
- Keep a validator/oracle in the loop. This project should use task verifiers,
  execution tests, cache-shape checks, replay checks, and behavioural scoring.
- Novelty matters. A generated or retrieved latent state should be compared
  against training/cache memories so we can distinguish copying from useful
  generalization.
- Structural quality matters. Do not stop at accuracy or reconstruction loss:
  track plan length, reasoning length, action/operator distributions, solution
  trajectories, verification behaviour, and distance/diversity between
  generated outputs.
- Interpolated latent points may define their own valid tasks. For middle
  points between two solved examples, success does not require answering either
  endpoint's dataset label. A strong interpolation can instead decode to a
  coherent invented or mutated problem plus a reasoning trace that correctly
  solves that problem.
- Use automatic filtering. Generated latent states may be noisy; it is acceptable
  to sample many candidates, replay/score them, and keep only candidates that
  satisfy behavioural and structural filters.
- Keep domains/protocols explicit. Every result should be labeled with the
  benchmark, model, prompt protocol, cache slice, replay protocol, and local-vs-
  reproduction status.
- Start with robust autoencoder baselines. AE/PCA/retrieval should remain
  strong first baselines before VAE/diffusion/generative planners.
- Keep fixed hyperparameter and seed discipline when comparing methods. A method
  should not win because it received more tuning than its baselines.
- Future controllability is important. The KV analogue is conditioning latents
  on desired reasoning depth, verification tendency, retry budget, task family,
  or plan length.

## Metrics To Preserve

Add or preserve these metric families where possible:

- **Validity / well-formedness**: task verifier success, executable-code pass
  rate, legal move rate, cache replay compatibility.
- **Generative factor analogue**: number of valid novel behavioural successes
  divided by the number of training/cache examples used.
- **Novelty**: nearest-neighbour distance to cached trajectories or latent
  states; exact duplicate rate; output sequence edit distance.
- **Structural pacing**: positions of verification steps, correction steps,
  subgoals, tool calls, or decisive operations within the generated reasoning.
- **Operator/action distribution**: action frequencies in Hanoi/Game24/ReAct,
  tool-call distributions, verifier/correction phrase distributions.
- **Diversity**: pairwise edit distance or embedding distance across generated
  solutions.
- **Interpolation task/solution validity**: for middle-alpha samples, judge
  whether there is an implied or recovered problem, whether the plan solves that
  problem, and whether the problem/solution pair is novel rather than a direct
  endpoint copy. Endpoint-target correctness is a reconstruction diagnostic, not
  the primary metric for interpolated samples.
- **Protocol robustness**: average and standard deviation across seeds, fixed
  sample budgets, and fixed benchmark slices.

## Practical Implications For This Repo

- Add "candidate generation + filtering" as an explicit mode for latent KV
  experiments: generate multiple cache latents, replay each locally, filter by
  verifier and structural metrics, then report both raw and filtered success.
- When training generative latent modules, report both reconstruction diagnostics
  and behavioural validity/novelty diagnostics.
- For every generated cache result, compare against nearest-neighbour retrieval
  to prove the method is not merely memorizing cached successful states.
- Keep symbolic/local tasks as sanity domains because they give exact
  validators. Use open-ended tasks only after the structured pipeline is stable.
