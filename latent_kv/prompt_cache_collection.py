"""Cache-backed prompt protocol collection for latent point experiments."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .benchmarks import load_examples, verify_output
from .cache import (
    capture_prompt_cache,
    choose_device,
    load_model_and_tokenizer,
    save_cache_bundle,
)
from .experiment_config import ResolvedExperimentConfig, write_resolved_config
from .metrics import evaluate_run, load_metric_payload, render_report
from .prompt_baselines import format_prompt, generate_local, prompt_protocol_metadata
from .schemas import CacheMetadata, TrajectoryRecord, append_jsonl, read_jsonl, write_json


def _existing_task_ids(path: Path) -> set[str]:
    return {str(row.get("task_id")) for row in read_jsonl(path)}


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
) -> dict[str, Any]:
    baseline = baseline.lower()
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
                token_count=prompt_tokens,
                cache_path=str(cache_path),
            )
            save_cache_bundle(
                cache_path,
                cache,
                metadata,
                hidden_states=hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                last_logits=last_logits,
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
    write_json(run_dir / "metrics.json", payload)
    report = render_report(run_dir, read_jsonl(record_path), payload)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return payload
