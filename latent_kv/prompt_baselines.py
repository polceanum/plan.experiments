"""Local prompt baselines for reasoning and planning tasks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

import torch

from .benchmarks import load_examples, verify_output
from .cache import choose_device, load_model_and_tokenizer, set_seed
from .metrics import aggregate_records, load_metric_payload, metric_to_dict, render_report
from .schemas import TaskExample, TrajectoryRecord, read_jsonl, to_json_line, write_json


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
) -> dict[str, Any]:
    baseline = baseline.lower()
    if baseline not in {"standard", "cot", "self_consistency", "retry_reflection"}:
        raise ValueError("baseline must be standard, cot, self_consistency, or retry_reflection")

    run_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(benchmark, limit=limit, seed=seed)
    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(model_id, device, local_files_only=True)
    records: list[TrajectoryRecord] = []

    for idx, example in enumerate(examples):
        if baseline == "self_consistency":
            prompt = format_prompt(example, "cot")
            sample_outputs: list[tuple[str | None, str]] = []
            total_latency = 0.0
            total_tokens = 0
            for sample_idx in range(samples):
                text, latency, tokens = generate_local(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=max_new_tokens,
                    seed=seed + idx * 1000 + sample_idx,
                    do_sample=True,
                    temperature=temperature,
                )
                parsed, _ = verify_output(text, example)
                sample_outputs.append((parsed, text))
                total_latency += latency
                total_tokens += tokens
            parsed, output_text = _majority_answer(sample_outputs)
            correct = parsed is not None and parsed == verify_output(example.answer, example)[0]
            latency_s = total_latency
            generated_tokens = total_tokens
            prompt_used = prompt
        else:
            prompt_used = format_prompt(example, baseline)
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

        records.append(
            TrajectoryRecord(
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
                correct=bool(correct),
                retry_index=0,
                latency_s=latency_s,
                generated_tokens=generated_tokens,
                prompt_tokens=len(prompt_used.split()),
                metadata=example.metadata
                | {
                    "prompt_baseline": baseline,
                    "local_files_only": True,
                    "samples": samples if baseline == "self_consistency" else 1,
                    "temperature": temperature if baseline == "self_consistency" else None,
                },
            )
        )

    record_path = _write_prompt_records(run_dir, baseline, records)
    return _merge_metrics(
        run_dir,
        baseline,
        [record.__dict__ for record in records],
        {
            f"prompt_{baseline}_records": str(record_path),
            f"prompt_{baseline}_local_model": model_id,
            f"prompt_{baseline}_benchmark": benchmark,
            f"prompt_{baseline}_protocol_match": False,
        },
    )
