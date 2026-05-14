"""Local prompt baselines for reasoning and planning tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch

from .benchmarks import load_examples, verify_output
from .cache import choose_device, load_model_and_tokenizer, set_seed
from .metrics import aggregate_records, load_metric_payload, metric_to_dict, render_report
from .schemas import TaskExample, TrajectoryRecord, read_jsonl, to_json_line, write_json


@dataclass(frozen=True)
class PromptProtocol:
    name: str
    baseline: str
    prompt_family: str
    decoding: str
    reported_protocol: bool = False


@dataclass(frozen=True)
class BaselineTier:
    name: str
    limit: int
    max_new_tokens: int
    description: str


BASELINE_TIERS: dict[str, BaselineTier] = {
    "custom": BaselineTier(
        name="custom",
        limit=3,
        max_new_tokens=96,
        description="Manual/default CLI budget for ad hoc checks.",
    ),
    "smoke": BaselineTier(
        name="smoke",
        limit=5,
        max_new_tokens=320,
        description="Quick model-level baseline check for plumbing and reports.",
    ),
    "working": BaselineTier(
        name="working",
        limit=20,
        max_new_tokens=320,
        description="Small but useful local baseline tier before early comparisons.",
    ),
    "comparison": BaselineTier(
        name="comparison",
        limit=100,
        max_new_tokens=320,
        description="Stronger local comparison floor for candidate methods.",
    ),
    "full": BaselineTier(
        name="full",
        limit=1319,
        max_new_tokens=320,
        description="Full GSM8K test-set sized run; use only when protocols are frozen.",
    ),
}


PROMPT_PROTOCOLS: dict[str, PromptProtocol] = {
    "standard": PromptProtocol(
        name="zero_shot_standard",
        baseline="standard",
        prompt_family="direct_task_prompt",
        decoding="greedy",
    ),
    "cot": PromptProtocol(
        name="zero_shot_chain_of_thought",
        baseline="cot",
        prompt_family="zero_shot_chain_of_thought",
        decoding="greedy",
    ),
    "self_consistency": PromptProtocol(
        name="zero_shot_cot_self_consistency",
        baseline="self_consistency",
        prompt_family="zero_shot_chain_of_thought",
        decoding="sampled_majority_vote",
    ),
    "retry_reflection": PromptProtocol(
        name="zero_shot_retry_reflection",
        baseline="retry_reflection",
        prompt_family="retry_reflection",
        decoding="greedy",
    ),
}


def resolve_baseline_tier(
    tier_name: str,
    limit: int | None = None,
    max_new_tokens: int | None = None,
) -> tuple[str, int, int]:
    if tier_name not in BASELINE_TIERS:
        raise ValueError(f"Unknown baseline tier: {tier_name}")
    tier = BASELINE_TIERS[tier_name]
    return (
        tier.name,
        int(limit if limit is not None else tier.limit),
        int(max_new_tokens if max_new_tokens is not None else tier.max_new_tokens),
    )


def format_prompt(example: TaskExample, baseline: str) -> str:
    baseline = baseline.lower()
    if baseline == "standard":
        return example.prompt
    if baseline == "cot":
        return (
            "Answer the problem. Think step by step, then finish with a line "
            "`The answer is <answer>`.\n\n"
            f"{example.prompt}"
        )
    if baseline == "retry_reflection":
        return (
            "Solve the task. First produce a concise attempt. Then check the attempt for errors. "
            "If needed, correct it. Finish with the final answer only after the check.\n\n"
            f"{example.prompt}"
        )
    raise ValueError(f"Unsupported prompt baseline: {baseline}")


def benchmark_split(examples: list[TaskExample]) -> str:
    sources = sorted({str(example.metadata.get("source", example.benchmark)) for example in examples})
    return "+".join(sources) if sources else "unknown"


def prompt_protocol_metadata(
    *,
    baseline: str,
    model_id: str,
    benchmark: str,
    examples: list[TaskExample],
    limit: int,
    seed: int,
    max_new_tokens: int,
    device_name: str,
    samples: int,
    temperature: float,
    chat_template: bool,
    baseline_tier: str = "custom",
) -> dict[str, Any]:
    protocol = PROMPT_PROTOCOLS[baseline]
    effective_samples = samples if baseline == "self_consistency" else 1
    return {
        "baseline": baseline,
        "protocol_name": protocol.name,
        "prompt_family": protocol.prompt_family,
        "decoding_protocol": protocol.decoding,
        "reported_protocol": protocol.reported_protocol,
        "protocol_match": False,
        "model_id": model_id,
        "benchmark": benchmark,
        "benchmark_split": benchmark_split(examples),
        "baseline_tier": baseline_tier,
        "limit": limit,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "device": device_name,
        "sample_count": effective_samples,
        "temperature": temperature if baseline == "self_consistency" else None,
        "chat_template": chat_template,
        "local_files_only": True,
    }


@torch.no_grad()
def generate_local(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    seed: int,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> tuple[str, float, int]:
    set_seed(seed)
    if getattr(tokenizer, "chat_template", None):
        encoded_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        encoded = {"input_ids": encoded_ids}
        attention_mask = torch.ones_like(encoded_ids)
        encoded["attention_mask"] = attention_mask
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    start = time.perf_counter()
    generate_kwargs = {
        **encoded,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    generated = model.generate(**generate_kwargs)
    latency = time.perf_counter() - start
    new_tokens = int(generated.shape[-1] - encoded["input_ids"].shape[-1])
    text = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text, latency, new_tokens


def _majority_answer(samples: list[tuple[str | None, str]]) -> tuple[str | None, str]:
    parsed_values = [parsed for parsed, _ in samples if parsed is not None]
    if not parsed_values:
        return None, samples[0][1] if samples else ""
    winner = Counter(parsed_values).most_common(1)[0][0]
    for parsed, text in samples:
        if parsed == winner:
            return winner, text
    return winner, samples[0][1]


def _write_prompt_records(run_dir: Path, baseline: str, rows: list[TrajectoryRecord]) -> Path:
    path = run_dir / "behavior" / f"{baseline}_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(to_json_line(row))
    return path


def _prepare_prompt_records(run_dir: Path, baseline: str, resume: bool = False) -> Path:
    path = run_dir / "behavior" / f"{baseline}_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text("", encoding="utf-8")
    elif not path.exists():
        path.write_text("", encoding="utf-8")
    return path


def _append_prompt_record(path: Path, row: TrajectoryRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(to_json_line(row))


def _merge_metrics(run_dir: Path, baseline: str, rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    existing = load_metric_payload(run_dir)
    by_key = {
        (row["baseline"], row["benchmark"]): row
        for row in existing.get("baselines", [])
    }
    for metric in aggregate_records(rows, baseline=baseline):
        by_key[(metric.baseline, metric.benchmark)] = metric_to_dict(metric)
    payload = {"baselines": list(by_key.values()), "extra": existing.get("extra", {}) | extra}
    write_json(run_dir / "metrics.json", payload)
    original_records = read_jsonl(run_dir / "records.jsonl")
    report_records = original_records if original_records else rows
    report = render_report(run_dir, report_records, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def run_prompt_baseline(
    run_dir: Path,
    benchmark: str,
    baseline: str,
    model_id: str,
    limit: int,
    seed: int,
    max_new_tokens: int,
    device_name: str = "auto",
    samples: int = 5,
    temperature: float = 0.7,
    baseline_tier: str = "custom",
    resume: bool = False,
) -> dict[str, Any]:
    baseline = baseline.lower()
    if baseline not in PROMPT_PROTOCOLS:
        raise ValueError("baseline must be standard, cot, self_consistency, or retry_reflection")

    run_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(benchmark, limit=limit, seed=seed)
    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(model_id, device, local_files_only=True)
    uses_chat_template = bool(getattr(tokenizer, "chat_template", None))
    protocol_metadata = prompt_protocol_metadata(
        baseline=baseline,
        model_id=model_id,
        benchmark=benchmark,
        examples=examples,
        limit=limit,
        seed=seed,
        max_new_tokens=max_new_tokens,
        device_name=device_name,
        samples=samples,
        temperature=temperature,
        chat_template=uses_chat_template,
        baseline_tier=baseline_tier,
    )
    record_path = _prepare_prompt_records(run_dir, baseline, resume=resume)
    records: list[dict[str, Any]] = read_jsonl(record_path) if resume else []
    completed_task_ids = {str(row.get("task_id")) for row in records}
    extra = {
        f"prompt_{baseline}_records": str(record_path),
        f"prompt_{baseline}_local_model": model_id,
        f"prompt_{baseline}_benchmark": benchmark,
        f"prompt_{baseline}_baseline_tier": baseline_tier,
        f"prompt_{baseline}_protocol": protocol_metadata,
        f"prompt_{baseline}_protocol_match": False,
        f"prompt_{baseline}_resume": resume,
    }
    payload = _merge_metrics(run_dir, baseline, records, extra)

    for idx, example in enumerate(examples):
        if example.task_id in completed_task_ids:
            print(
                f"[{idx + 1}/{len(examples)}] {baseline} {example.task_id}: skipped existing record",
                flush=True,
            )
            continue
        start = time.perf_counter()
        error = None
        if baseline == "self_consistency":
            prompt_used = format_prompt(example, "cot")
        else:
            prompt_used = format_prompt(example, baseline)
        try:
            if baseline == "self_consistency":
                sample_outputs: list[tuple[str | None, str]] = []
                vote_counts: Counter[str] = Counter()
                total_latency = 0.0
                total_tokens = 0
                for sample_idx in range(samples):
                    text, latency, tokens = generate_local(
                        model,
                        tokenizer,
                        prompt_used,
                        device,
                        max_new_tokens=max_new_tokens,
                        seed=seed + idx * 1000 + sample_idx,
                        do_sample=True,
                        temperature=temperature,
                    )
                    parsed, _ = verify_output(text, example)
                    sample_outputs.append((parsed, text))
                    vote_counts[str(parsed) if parsed is not None else "<unparsed>"] += 1
                    total_latency += latency
                    total_tokens += tokens
                parsed, output_text = _majority_answer(sample_outputs)
                correct = parsed is not None and parsed == verify_output(example.answer, example)[0]
                latency_s = total_latency
                generated_tokens = total_tokens
                sample_metadata = {
                    "self_consistency_votes": dict(sorted(vote_counts.items())),
                    "self_consistency_unparsed": vote_counts.get("<unparsed>", 0),
                }
            else:
                output_text, latency_s, generated_tokens = generate_local(
                    model,
                    tokenizer,
                    prompt_used,
                    device,
                    max_new_tokens=max_new_tokens,
                    seed=seed + idx,
                    do_sample=False,
                )
                parsed, correct = verify_output(output_text, example)
                sample_metadata = {}
        except Exception as exc:
            output_text = ""
            parsed = None
            correct = False
            latency_s = time.perf_counter() - start
            generated_tokens = 0
            sample_metadata = {}
            error = f"{type(exc).__name__}: {exc}"

        record = TrajectoryRecord(
            run_id=run_dir.name,
            benchmark=example.benchmark,
            task_id=example.task_id,
            model_id=model_id,
            seed=seed,
            attempt_id=idx,
            prompt=prompt_used,
            target=example.answer,
            output_text=output_text,
            parsed_answer=parsed,
            correct=bool(correct) and error is None,
            retry_index=0,
            latency_s=latency_s,
            generated_tokens=generated_tokens,
            prompt_tokens=len(prompt_used.split()),
            metadata=example.metadata
            | {
                "prompt_baseline": baseline,
                "prompt_protocol": protocol_metadata["protocol_name"],
                "prompt_family": protocol_metadata["prompt_family"],
                "decoding_protocol": protocol_metadata["decoding_protocol"],
                "local_files_only": True,
                "samples": samples if baseline == "self_consistency" else 1,
                "temperature": temperature if baseline == "self_consistency" else None,
                "chat_template": uses_chat_template,
                "protocol_match": False,
                "generation_error": error,
            }
            | sample_metadata,
        )
        record_dict = record.__dict__
        records.append(record_dict)
        completed_task_ids.add(example.task_id)
        _append_prompt_record(record_path, record)
        payload = _merge_metrics(
            run_dir,
            baseline,
            records,
            extra
            | {
                f"prompt_{baseline}_completed_examples": len(records),
                f"prompt_{baseline}_failed_examples": sum(
                    1 for row in records if (row.get("metadata") or {}).get("generation_error")
                ),
            },
        )
        print(
            f"[{idx + 1}/{len(examples)}] {baseline} {example.task_id}: "
            f"correct={record.correct} error={error is not None}",
            flush=True,
        )

    return _merge_metrics(
        run_dir,
        baseline,
        records,
        extra
        | {
            f"prompt_{baseline}_completed_examples": len(records),
            f"prompt_{baseline}_failed_examples": sum(
                1 for row in records if (row.get("metadata") or {}).get("generation_error")
            )
        },
    )
