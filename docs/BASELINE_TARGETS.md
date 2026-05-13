# Baseline Targets

These targets are for protocol validation, not promises that local small models
will reach reported numbers. A local run should only be considered a reproduction
when the model family, prompt, decoding settings, benchmark split, and sample
count match the reported protocol.

| Baseline | Reported Target | Reproduction Status |
|---|---:|---|
| Chain-of-thought prompting on GSM8K with PaLM 540B | 57% solve rate in the reported CoT setup | Target only; local models are approximations |
| Self-consistency on GSM8K with PaLM 540B | 56.5% CoT to 74.4% self-consistency | Target only; needs exact prompt/sample protocol |
| Tree of Thoughts on Game of 24 with GPT-4 | 74% success for ToT with breadth 5 | Local symbolic harness implemented; GPT-4 protocol not reproduced |
| ReAct on HotpotQA/Fever with PaLM 540B | HotpotQA/Fever prompt baselines reported in ReAct Table 1 | Not implemented |
| ReAct on ALFWorld/WebShop | +34% / +10% absolute success-rate gains over prior methods | Not implemented |

## Policy

- Local models must run in `conda run -n orpheus ...`.
- No API keys or hosted model services are allowed.
- If required weights are not present locally, the command should fail rather
  than download silently.
- Reports should distinguish:
  - `local_check`: baseline implementation ran locally.
  - `protocol_match`: model/prompt/decoding/split match the reported protocol.
  - `target_match`: metric is within the accepted tolerance of the reported target.
