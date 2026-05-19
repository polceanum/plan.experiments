"""Cache-backed prompt protocol collection for latent point experiments."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .benchmarks import load_examples, verify_output
from .cache import (
    capture_full_trajectory_cache,
    capture_prompt_cache,
    choose_device,
    load_cache_bundle,
    load_model_and_tokenizer,
    save_cache_bundle,
)
from .experiment_config import ResolvedExperimentConfig, write_resolved_config
from .metrics import evaluate_run, load_metric_payload, render_report
from .prompt_baselines import format_prompt, generate_local_with_ids, prompt_protocol_metadata
from .schemas import CacheMetadata, TrajectoryRecord, append_jsonl, read_jsonl, write_json


def _existing_task_ids(path: Path) -> set[str]:
    return {str(row.get("task_id")) for row in read_jsonl(path)}


def _validate_cache_mode(cache_mode: str) -> str:
    cache_mode = cache_mode.lower()
    if cache_mode not in {"prompt", "trajectory"}:
        raise ValueError(f"Unsupported cache_mode: {cache_mode}")
    return cache_mode


def _tokenize_output_text(tokenizer: Any, output_text: str):
    tokenized = tokenizer(output_text, add_special_tokens=False, return_tensors="pt")
    return tokenized["input_ids"]


def run_prompt_cache_collection(
    run_dir: Path,
    benchmark: str,
    baseline: str,
    model_id: str,
    limit: int,
    seed: int,
    max_new_tokens: int,
    device_name: str = "auto",
    baseline_tier: str = "custom",
    layer_mode: str = "all",
    capture_hidden: bool = False,
    resume: bool = False,
    resolved_config: ResolvedExperimentConfig | None = None,
    cache_mode: str = "prompt",
) -> dict[str, Any]:
    baseline = baseline.lower()
    cache_mode = _validate_cache_mode(cache_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "records.jsonl"
    if not resume:
        record_path.write_text("", encoding="utf-8")
    elif not record_path.exists():
        record_path.write_text("", encoding="utf-8")
    if resolved_config is not None:
        write_resolved_config(run_dir, resolved_config)

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
        samples=1,
        temperature=0.0,
        chat_template=uses_chat_template,
        baseline_tier=baseline_tier,
    )
    completed = _existing_task_ids(record_path) if resume else set()

    for idx, example in enumerate(examples):
        if example.task_id in completed:
            print(f"[{idx + 1}/{len(examples)}] {baseline} {example.task_id}: skipped existing cache", flush=True)
            continue
        prompt_used = format_prompt(example, baseline)
        start = time.perf_counter()
        error = None
        try:
            output_text, latency_s, generated_tokens, generation_token_ids = generate_local_with_ids(
                model,
                tokenizer,
                prompt_used,
                device,
                max_new_tokens=max_new_tokens,
                seed=seed + idx,
                do_sample=False,
            )
            parsed, correct = verify_output(output_text, example)
            if cache_mode == "trajectory":
                (
                    cache,
                    selected_layers,
                    prompt_tokens,
                    cache_generated_tokens,
                    cache_total_tokens,
                    hidden,
                    input_ids,
                    attention_mask,
                    last_logits,
                    prompt_input_ids,
                ) = capture_full_trajectory_cache(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_used,
                    generation_token_ids=generation_token_ids,
                    device=device,
                    layer_mode=layer_mode,
                    capture_hidden=capture_hidden,
                    use_chat_template=True,
                )
            else:
                (
                    cache,
                    selected_layers,
                    prompt_tokens,
                    hidden,
                    input_ids,
                    attention_mask,
                    last_logits,
                ) = capture_prompt_cache(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_used,
                    device=device,
                    layer_mode=layer_mode,
                    capture_hidden=capture_hidden,
                    use_chat_template=True,
                )
                cache_generated_tokens = 0
                cache_total_tokens = prompt_tokens
                prompt_input_ids = input_ids
        except Exception as exc:
            output_text = ""
            parsed = None
            correct = False
            latency_s = time.perf_counter() - start
            generated_tokens = 0
            prompt_tokens = len(prompt_used.split())
            cache = None
            selected_layers = []
            hidden = None
            input_ids = None
            attention_mask = None
            last_logits = None
            generation_token_ids = None
            cache_generated_tokens = 0
            cache_total_tokens = prompt_tokens
            prompt_input_ids = None
            error = f"{type(exc).__name__}: {exc}"

        cache_path = None
        if cache is not None:
            cache_path = cache_dir / f"{example.benchmark}_{baseline}_{idx:04d}.pt"
            metadata = CacheMetadata(
                model_id=model_id,
                tokenizer_id=getattr(tokenizer, "name_or_path", model_id),
                dtype=str(cache[0][0].dtype) if cache else "unknown",
                device=str(device),
                layers=len(cache),
                selected_layers=selected_layers,
                selected_heads=None,
                token_count=cache_total_tokens,
                cache_path=str(cache_path),
                benchmark=example.benchmark,
                task_id=example.task_id,
                prompt_baseline=baseline,
                prompt_protocol=protocol_metadata["protocol_name"],
                target=example.answer,
                parsed_answer=parsed,
                correct=bool(correct) and error is None,
                generation_error=error,
            )
            save_cache_bundle(
                cache_path,
                cache,
                metadata,
                hidden_states=hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                last_logits=last_logits,
                generation_token_ids=generation_token_ids,
                generation_config={
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "decoding_protocol": protocol_metadata["decoding_protocol"],
                    "seed": seed + idx,
                    "cache_mode": cache_mode,
                    "prompt_tokens": prompt_tokens,
                    "cache_generated_tokens": cache_generated_tokens,
                    "cache_total_tokens": cache_total_tokens,
                    "prompt_input_ids": prompt_input_ids.tolist() if prompt_input_ids is not None else None,
                },
            )

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
            cache_path=str(cache_path) if cache_path is not None else None,
            latency_s=latency_s,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            metadata=example.metadata
            | {
                "prompt_baseline": baseline,
                "prompt_protocol": protocol_metadata["protocol_name"],
                "prompt_family": protocol_metadata["prompt_family"],
                "decoding_protocol": protocol_metadata["decoding_protocol"],
                "cache_collection": True,
                "cache_mode": cache_mode,
                "cache_total_tokens": cache_total_tokens,
                "cache_generated_tokens": cache_generated_tokens,
                "cache_layer_mode": layer_mode,
                "chat_template": uses_chat_template,
                "local_files_only": True,
                "generation_error": error,
                "resolved_config": str(run_dir / "resolved_config.json") if resolved_config is not None else None,
            },
        )
        append_jsonl(record_path, record)
        print(
            f"[{idx + 1}/{len(examples)}] {baseline} {example.task_id}: "
            f"correct={record.correct} cache={record.cache_path is not None} error={error is not None}",
            flush=True,
        )

    payload = evaluate_run(run_dir, baseline=f"prompt_cache_{baseline}")
    payload = load_metric_payload(run_dir)
    extra = payload.setdefault("extra", {})
    extra[f"prompt_cache_{baseline}_records"] = str(record_path)
    extra[f"prompt_cache_{baseline}_local_model"] = model_id
    extra[f"prompt_cache_{baseline}_protocol"] = protocol_metadata
    extra[f"prompt_cache_{baseline}_cache_dir"] = str(cache_dir)
    extra[f"prompt_cache_{baseline}_resume"] = resume
    extra[f"prompt_cache_{baseline}_cache_mode"] = cache_mode
    write_json(run_dir / "metrics.json", payload)
    report = render_report(run_dir, read_jsonl(record_path), payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def run_existing_prompt_record_cache_collection(
    run_dir: Path,
    source_records: Path,
    model_id: str,
    device_name: str = "auto",
    layer_mode: str = "all",
    capture_hidden: bool = False,
    resume: bool = False,
    cache_mode: str = "prompt",
    limit: int | None = None,
) -> dict[str, Any]:
    cache_mode = _validate_cache_mode(cache_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "records.jsonl"
    if not resume:
        record_path.write_text("", encoding="utf-8")
    elif not record_path.exists():
        record_path.write_text("", encoding="utf-8")

    source_rows = read_jsonl(source_records)
    if limit is not None:
        source_rows = source_rows[: int(limit)]
    completed = _existing_task_ids(record_path) if resume else set()
    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(model_id, device, local_files_only=True)
    uses_chat_template = bool(getattr(tokenizer, "chat_template", None))

    for idx, source in enumerate(source_rows):
        task_id = str(source.get("task_id") or f"record_{idx:04d}")
        if task_id in completed:
            print(f"[{idx + 1}/{len(source_rows)}] {task_id}: skipped existing cache", flush=True)
            continue
        prompt_used = str(source.get("prompt") or "")
        metadata = dict(source.get("metadata") or {})
        baseline = str(metadata.get("prompt_baseline") or "existing")
        protocol = str(metadata.get("prompt_protocol") or "existing_prompt_record")
        start = time.perf_counter()
        error = None
        try:
            if cache_mode == "trajectory":
                source_cache_path = source.get("cache_path")
                if not source_cache_path:
                    raise ValueError("trajectory cache attachment requires source cache_path with generation_token_ids")
                source_bundle = load_cache_bundle(Path(str(source_cache_path)))
                generation_token_ids = source_bundle.get("generation_token_ids")
                if generation_token_ids is None or int(generation_token_ids.numel()) == 0:
                    generation_token_ids = _tokenize_output_text(tokenizer, str(source.get("output_text") or ""))
                    generation_token_source = "tokenized_output_text"
                else:
                    generation_token_source = "source_generation_token_ids"
                (
                    cache,
                    selected_layers,
                    prompt_tokens,
                    cache_generated_tokens,
                    cache_total_tokens,
                    hidden,
                    input_ids,
                    attention_mask,
                    last_logits,
                    prompt_input_ids,
                ) = capture_full_trajectory_cache(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_used,
                    generation_token_ids=generation_token_ids,
                    device=device,
                    layer_mode=layer_mode,
                    capture_hidden=capture_hidden,
                    use_chat_template=True,
                )
            else:
                (
                    cache,
                    selected_layers,
                    prompt_tokens,
                    hidden,
                    input_ids,
                    attention_mask,
                    last_logits,
                ) = capture_prompt_cache(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_used,
                    device=device,
                    layer_mode=layer_mode,
                    capture_hidden=capture_hidden,
                    use_chat_template=True,
                )
                generation_token_ids = None
                cache_generated_tokens = 0
                cache_total_tokens = prompt_tokens
                prompt_input_ids = input_ids
        except Exception as exc:
            cache = None
            selected_layers = []
            prompt_tokens = int(source.get("prompt_tokens") or len(prompt_used.split()))
            hidden = None
            input_ids = None
            attention_mask = None
            last_logits = None
            generation_token_ids = None
            cache_generated_tokens = 0
            cache_total_tokens = prompt_tokens
            prompt_input_ids = None
            error = f"{type(exc).__name__}: {exc}"

        cache_path = None
        if cache is not None:
            cache_path = cache_dir / f"{source.get('benchmark', 'task')}_{baseline}_{idx:04d}.pt"
            parsed_answer = source.get("parsed_answer")
            metadata_payload = CacheMetadata(
                model_id=model_id,
                tokenizer_id=getattr(tokenizer, "name_or_path", model_id),
                dtype=str(cache[0][0].dtype) if cache else "unknown",
                device=str(device),
                layers=len(cache),
                selected_layers=selected_layers,
                selected_heads=None,
                token_count=cache_total_tokens,
                cache_path=str(cache_path),
                benchmark=source.get("benchmark"),
                task_id=task_id,
                prompt_baseline=baseline,
                prompt_protocol=protocol,
                target=source.get("target"),
                parsed_answer=str(parsed_answer) if parsed_answer is not None else None,
                correct=bool(source.get("correct")) and error is None,
                generation_error=error,
            )
            save_cache_bundle(
                cache_path,
                cache,
                metadata_payload,
                hidden_states=hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                last_logits=last_logits,
                generation_token_ids=generation_token_ids,
                generation_config={
                    "max_new_tokens": source.get("generated_tokens"),
                    "decoding_protocol": metadata.get("decoding_protocol"),
                    "seed": source.get("seed"),
                    "source_records": str(source_records),
                    "attached_existing_output": True,
                    "cache_mode": cache_mode,
                    "generation_token_source": generation_token_source if cache_mode == "trajectory" else None,
                    "prompt_tokens": prompt_tokens,
                    "cache_generated_tokens": cache_generated_tokens,
                    "cache_total_tokens": cache_total_tokens,
                    "prompt_input_ids": prompt_input_ids.tolist() if prompt_input_ids is not None else None,
                },
            )

        record = TrajectoryRecord(
            run_id=run_dir.name,
            benchmark=str(source.get("benchmark") or "unknown"),
            task_id=task_id,
            model_id=model_id,
            seed=int(source.get("seed") or 0),
            attempt_id=int(source.get("attempt_id") if source.get("attempt_id") is not None else idx),
            prompt=prompt_used,
            target=str(source.get("target") or ""),
            output_text=str(source.get("output_text") or ""),
            parsed_answer=source.get("parsed_answer"),
            correct=bool(source.get("correct")) and error is None,
            retry_index=int(source.get("retry_index") or 0),
            cache_path=str(cache_path) if cache_path is not None else None,
            latency_s=time.perf_counter() - start,
            generated_tokens=source.get("generated_tokens"),
            prompt_tokens=prompt_tokens,
            metadata=metadata
            | {
                "cache_collection": True,
                "cache_mode": cache_mode,
                "cache_total_tokens": cache_total_tokens,
                "cache_generated_tokens": cache_generated_tokens,
                "cache_attached_from_existing_record": True,
                "cache_source_records": str(source_records),
                "cache_layer_mode": layer_mode,
                "chat_template": uses_chat_template,
                "local_files_only": True,
                "generation_error": error,
            },
        )
        append_jsonl(record_path, record)
        print(
            f"[{idx + 1}/{len(source_rows)}] {task_id}: "
            f"source_correct={bool(source.get('correct'))} cache={record.cache_path is not None} error={error is not None}",
            flush=True,
        )

    payload = evaluate_run(run_dir, baseline="prompt_cache_attached")
    payload = load_metric_payload(run_dir)
    rows = read_jsonl(record_path)
    extra = payload.setdefault("extra", {})
    extra["prompt_cache_attached_source_records"] = str(source_records)
    extra["prompt_cache_attached_local_model"] = model_id
    extra["prompt_cache_attached_cache_dir"] = str(cache_dir)
    extra["prompt_cache_attached_cache_mode"] = cache_mode
    extra["prompt_cache_attached_source_total"] = len(rows)
    extra["prompt_cache_attached_source_correct"] = sum(1 for row in rows if row.get("correct") is True)
    extra["prompt_cache_attached_source_incorrect"] = sum(1 for row in rows if row.get("correct") is not True)
    extra["prompt_cache_attached_resume"] = resume
    write_json(run_dir / "metrics.json", payload)
    report = render_report(run_dir, rows, payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload
