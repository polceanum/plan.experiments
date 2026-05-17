"""Command line interface for the latent KV research prototype."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Sequence

from .baseline_targets import target_dicts
from .behavior import run_cache_behavioral_baseline
from .benchmarks import load_examples, verify_output
from .brief import get_brief
from .cache import choose_device, collect_one, load_model_and_tokenizer, set_seed
from .codec_validation import validate_reconstructed_artifact
from .compressors import run_compression
from .corruption_sensitivity import score_corruption_sensitivity
from .experiment_config import resolve_experiment_config
from .injection import greedy_continue_from_bundle, validate_bundle_for_injection
from .metrics import evaluate_run
from .prompt_cache_collection import run_existing_prompt_record_cache_collection, run_prompt_cache_collection
from .prompt_baselines import BASELINE_TIERS, resolve_baseline_tier, run_prompt_baseline
from .react_baseline import run_react_baseline
from .replay_diagnostics import score_replay_fidelity
from .research_log import append_research_log
from .schemas import TrajectoryRecord, append_jsonl, read_json, read_jsonl, write_json
from .target_checks import check_targets
from .tot_baseline import run_tot_baseline
from .training_diagnostics import summarize_training_curve


SMOKE_MODEL = "EleutherAI/pythia-70m-deduped"
MAIN_MODEL = "EleutherAI/pythia-410m-deduped"


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _append_run_event(run_dir: Path | None, event: dict[str, object]) -> None:
    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_unix": time.time(),
        **event,
    }
    path = run_dir / "run_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_dir_from_args(args: argparse.Namespace) -> Path | None:
    if hasattr(args, "run") and getattr(args, "run"):
        return Path(str(getattr(args, "run")))
    if hasattr(args, "run_id") and getattr(args, "run_id"):
        return _run_dir(str(getattr(args, "run_id")), str(getattr(args, "runs_root", "runs")))
    return None


def _run_dir(run_id: str, root: str = "runs") -> Path:
    return Path(root) / run_id


def _update_run_index(run_dir: Path, fields: dict[str, object]) -> None:
    index_path = run_dir.parent / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["run_id", "created_at", "benchmark", "model_id", "records", "path"]
    exists = index_path.exists()
    with index_path.open("a", encoding="utf-8") as handle:
        if not exists:
            handle.write(",".join(header) + "\n")
        row = {
            "run_id": run_dir.name,
            "created_at": str(int(time.time())),
            "benchmark": str(fields.get("benchmark", "")),
            "model_id": str(fields.get("model_id", "")),
            "records": str(fields.get("records", "")),
            "path": str(run_dir),
        }
        handle.write(",".join(row[key] for key in header) + "\n")


def _dry_run_output(example) -> str:
    if example.benchmark == "hanoi":
        return example.answer
    if example.benchmark == "sudoku":
        return example.answer
    if example.benchmark == "gsm8k":
        return example.answer
    return example.answer[:400]


def cmd_brief(_: argparse.Namespace) -> int:
    print(get_brief())
    return 0


def cmd_targets(_: argparse.Namespace) -> int:
    import json

    print(json.dumps(target_dicts(), indent=2, sort_keys=True))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args.run_id, args.runs_root)
    cache_dir = run_dir / "caches"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    examples = load_examples(args.benchmark, args.limit, args.seed)

    if args.dry_run:
        for idx, example in enumerate(examples):
            output = _dry_run_output(example)
            parsed, correct = verify_output(output, example)
            record = TrajectoryRecord(
                run_id=args.run_id,
                benchmark=example.benchmark,
                task_id=example.task_id,
                model_id="dry-run-oracle",
                seed=args.seed,
                attempt_id=0,
                prompt=example.prompt,
                target=example.answer,
                output_text=output,
                parsed_answer=parsed,
                correct=correct,
                retry_index=0,
                cache_path=None,
                generated_tokens=len(output.split()),
                prompt_tokens=len(example.prompt.split()),
                metadata=example.metadata | {"dry_run_index": idx},
            )
            append_jsonl(run_dir / "records.jsonl", record)
        _update_run_index(
            run_dir,
            {"benchmark": args.benchmark, "model_id": "dry-run-oracle", "records": len(examples)},
        )
        evaluate_run(run_dir, baseline="no_cache")
        print(f"Wrote dry-run records to {run_dir}")
        return 0

    device = choose_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    for idx, example in enumerate(examples):
        cache_path = cache_dir / f"{example.benchmark}_{example.task_id}_{idx:04d}.pt"
        record = collect_one(
            run_id=args.run_id,
            example=example,
            model=model,
            tokenizer=tokenizer,
            model_id=args.model_id,
            device=device,
            cache_path=cache_path,
            verify_fn=verify_output,
            seed=args.seed + idx,
            max_new_tokens=args.max_new_tokens,
            layer_mode=args.layer_mode,
            capture_hidden=args.capture_hidden,
        )
        append_jsonl(run_dir / "records.jsonl", record)
        print(f"[{idx + 1}/{len(examples)}] {example.task_id}: correct={record.correct}")

    _update_run_index(
        run_dir,
        {"benchmark": args.benchmark, "model_id": args.model_id, "records": len(examples)},
    )
    evaluate_run(run_dir, baseline="no_cache")
    print(f"Wrote records to {run_dir}")
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    result = run_compression(
        run_dir=run_dir,
        method=args.method,
        latent_dim=args.latent_dim,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        llm_model_id=args.model_id,
        llm_device_name=args.device,
        llm_loss_weight=args.llm_loss_weight,
        llm_steps=args.llm_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        heartbeat_every_batches=args.heartbeat_every_batches,
        train_batch_size=args.train_batch_size,
    )
    metrics_path = run_dir / "metrics.json"
    payload = read_json(metrics_path) if metrics_path.exists() else {"baselines": [], "extra": {}}
    extra = payload.setdefault("extra", {})
    extra[f"compression_{result.method}_mse"] = result.reconstruction_mse
    extra[f"compression_{result.method}_latent_dim"] = result.latent_dim
    write_json(metrics_path, payload)
    evaluate_run(run_dir, baseline="no_cache")
    print(f"{result.method}: mse={result.reconstruction_mse:.6g}, latents={result.latent_path}")
    return 0


def _write_basic_plots(run_dir: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/latent_kv_matplotlib")
        import matplotlib.pyplot as plt
    except Exception:
        return
    payload = read_json(run_dir / "metrics.json")
    rows = payload.get("baselines", [])
    if not rows:
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['baseline']}\n{row['benchmark']}" for row in rows]
    values = [row["accuracy"] for row in rows]
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Baseline Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "baseline_accuracy.png")
    plt.close()


def cmd_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if args.behavioral_baseline:
        payload = run_cache_behavioral_baseline(
            run_dir=run_dir,
            baseline=args.behavioral_baseline,
            max_new_tokens=args.max_new_tokens,
            device_name=args.device,
            model_id=args.model_id,
            limit=args.limit,
        )
    else:
        payload = evaluate_run(run_dir, baseline=args.baseline)
    _write_basic_plots(run_dir)
    print(f"Wrote metrics and report to {run_dir}")
    print(f"Baselines: {len(payload.get('baselines', []))}")
    return 0


def cmd_behavior(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    baselines = args.baseline
    if baselines == ["all"]:
        baselines = ["original_cache", "random", "pca_svd", "autoencoder", "rae_temporal", "retrieval"]
    payload = None
    for baseline in baselines:
        payload = run_cache_behavioral_baseline(
            run_dir=run_dir,
            baseline=baseline,
            max_new_tokens=args.max_new_tokens,
            device_name=args.device,
            model_id=args.model_id,
            limit=args.limit,
        )
        print(f"Scored behavioural baseline: {baseline}")
    _write_basic_plots(run_dir)
    print(f"Wrote behavioural metrics and report to {run_dir}")
    print(f"Baselines: {len((payload or {}).get('baselines', []))}")
    return 0


def cmd_prompt_baseline(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    baselines = args.baseline
    if baselines == ["all"]:
        baselines = ["standard", "cot", "self_consistency", "retry_reflection"]
    tier_name, limit, max_new_tokens = resolve_baseline_tier(
        args.baseline_tier,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    if args.chunk_size > 0:
        return _run_prompt_baseline_chunks(args, baselines, tier_name, limit, max_new_tokens)
    payload = None
    for baseline in baselines:
        payload = run_prompt_baseline(
            run_dir=run_dir,
            benchmark=args.benchmark,
            baseline=baseline,
            model_id=args.model_id,
            limit=limit,
            seed=args.seed,
            max_new_tokens=max_new_tokens,
            device_name=args.device,
            samples=args.samples,
            temperature=args.temperature,
            baseline_tier=tier_name,
            resume=args.resume,
        )
        print(f"Scored local prompt baseline: {baseline}")
    _write_basic_plots(run_dir)
    print(f"Wrote prompt-baseline metrics and report to {run_dir}")
    print(f"Baselines: {len((payload or {}).get('baselines', []))}")
    return 0


def _existing_prompt_record_count(run_dir: Path, baseline: str) -> int:
    record_path = run_dir / "behavior" / f"{baseline}_records.jsonl"
    if not record_path.exists():
        return 0
    return len(read_jsonl(record_path))


def _run_prompt_baseline_chunks(
    args: argparse.Namespace,
    baselines: list[str],
    tier_name: str,
    limit: int,
    max_new_tokens: int,
) -> int:
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not args.resume:
        raise ValueError("--chunk-size requires --resume so child chunks append/skip existing records")
    if len(baselines) != 1:
        raise ValueError("--chunk-size currently supports exactly one --baseline at a time")

    baseline = baselines[0]
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    completed = _existing_prompt_record_count(run_dir, baseline)
    while completed < limit:
        next_limit = min(limit, completed + args.chunk_size)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "latent_kv",
            "prompt-baseline",
            "--run",
            str(run_dir),
            "--benchmark",
            args.benchmark,
            "--baseline",
            baseline,
            "--baseline-tier",
            tier_name,
            "--limit",
            str(next_limit),
            "--seed",
            str(args.seed),
            "--max-new-tokens",
            str(max_new_tokens),
            "--samples",
            str(args.samples),
            "--temperature",
            str(args.temperature),
            "--model-id",
            args.model_id,
            "--device",
            args.device,
            "--resume",
        ]
        print(
            f"[chunk] {baseline}: completed={completed}/{limit}; launching up to {next_limit}",
            flush=True,
        )
        subprocess.run(cmd, check=True)
        after = _existing_prompt_record_count(run_dir, baseline)
        if after <= completed:
            raise RuntimeError(
                f"chunk made no progress for {baseline}: still at {after}/{limit}"
            )
        completed = after
    _write_basic_plots(run_dir)
    print(f"Completed chunked prompt baseline: {baseline} {completed}/{limit}")
    return 0


def cmd_collect_prompt_caches(args: argparse.Namespace) -> int:
    resolved = resolve_experiment_config(Path(args.config)) if args.config else None
    config_dataset = resolved.dataset if resolved is not None else {}
    config_prompt = resolved.prompt if resolved is not None else {}
    config_model = resolved.model if resolved is not None else {}
    config_cache = resolved.cache if resolved is not None else {}
    tier_name, limit, max_new_tokens = resolve_baseline_tier(
        args.baseline_tier or str(config_prompt.get("baseline_tier") or "custom"),
        limit=args.limit if args.limit is not None else config_dataset.get("limit"),
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else config_prompt.get("max_new_tokens"),
    )
    payload = run_prompt_cache_collection(
        run_dir=Path(args.run),
        benchmark=args.benchmark or str(config_dataset.get("benchmark") or "hanoi"),
        baseline=args.baseline or str(config_prompt.get("baseline") or "standard"),
        model_id=args.model_id or str(config_model.get("model_id") or SMOKE_MODEL),
        limit=limit,
        seed=args.seed if args.seed is not None else int(config_dataset.get("seed") or 0),
        max_new_tokens=max_new_tokens,
        device_name=args.device or str(config_model.get("device") or "auto"),
        baseline_tier=tier_name,
        layer_mode=args.layer_mode or str(config_cache.get("layer_mode") or "all"),
        capture_hidden=args.capture_hidden or bool(config_cache.get("capture_hidden") or False),
        resume=args.resume,
        resolved_config=resolved,
    )
    _write_basic_plots(Path(args.run))
    print(f"Wrote cache-backed prompt records to {args.run}")
    print(f"Baselines: {len(payload.get('baselines', []))}")
    return 0


def cmd_attach_prompt_caches(args: argparse.Namespace) -> int:
    payload = run_existing_prompt_record_cache_collection(
        run_dir=Path(args.run),
        source_records=Path(args.source_records),
        model_id=args.model_id or SMOKE_MODEL,
        device_name=args.device or "auto",
        layer_mode=args.layer_mode or "all",
        capture_hidden=args.capture_hidden,
        resume=args.resume,
    )
    _write_basic_plots(Path(args.run))
    extra = payload.get("extra", {})
    print(f"Wrote attached prompt-cache records to {args.run}")
    print(f"Records: {extra.get('prompt_cache_attached_source_total')}")
    print(f"Source correct: {extra.get('prompt_cache_attached_source_correct')}")
    print(f"Source incorrect: {extra.get('prompt_cache_attached_source_incorrect')}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    append_research_log(
        log_path=Path(args.path),
        title=args.title,
        worked=args.worked or [],
        did_not_work=args.did_not_work or [],
        todo=args.todo or [],
    )
    print(f"Updated {args.path}")
    return 0


def cmd_tot_baseline(args: argparse.Namespace) -> int:
    payload = run_tot_baseline(
        run_dir=Path(args.run),
        limit=args.limit,
        seed=args.seed,
        breadth=args.breadth,
    )
    _write_basic_plots(Path(args.run))
    print(f"Wrote Tree-of-Thought metrics and report to {args.run}")
    print(f"Baselines: {len(payload.get('baselines', []))}")
    return 0


def cmd_react_baseline(args: argparse.Namespace) -> int:
    payload = run_react_baseline(
        run_dir=Path(args.run),
        limit=args.limit,
        max_steps=args.max_steps,
    )
    _write_basic_plots(Path(args.run))
    print(f"Wrote ReAct metrics and report to {args.run}")
    print(f"Baselines: {len(payload.get('baselines', []))}")
    return 0


def cmd_check_targets(args: argparse.Namespace) -> int:
    import json

    checks = check_targets(Path(args.run), tolerance=args.tolerance)
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0


def cmd_validate_codec(args: argparse.Namespace) -> int:
    import json

    validation = validate_reconstructed_artifact(Path(args.run), args.method)
    print(json.dumps(validation.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_replay_fidelity(args: argparse.Namespace) -> int:
    import json

    summary = score_replay_fidelity(
        run_dir=Path(args.run),
        method=args.method,
        model_id=args.model_id,
        device_name=args.device,
        limit=args.limit,
        steps=args.steps,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_training_curve(args: argparse.Namespace) -> int:
    import json

    summary = summarize_training_curve(Path(args.run), args.method)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_corruption_sensitivity(args: argparse.Namespace) -> int:
    import json

    summary = score_corruption_sensitivity(
        run_dir=Path(args.run),
        method=args.method,
        alphas=args.alpha,
        model_id=args.model_id,
        device_name=args.device,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    bundle_path = Path(args.cache)
    summary = validate_bundle_for_injection(bundle_path)
    if args.validate_only:
        print(summary)
        return 0
    model_id = args.model_id or summary.get("model_id")
    if not model_id:
        raise ValueError("--model-id is required when the bundle metadata has no model_id")
    text = greedy_continue_from_bundle(
        bundle_path=bundle_path,
        model_id=str(model_id),
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latent-kv")
    sub = parser.add_subparsers(dest="command", required=True)

    brief = sub.add_parser("brief", help="Print the distilled local research brief")
    brief.set_defaults(func=cmd_brief)

    targets = sub.add_parser("targets", help="Print reported targets tracked by the project")
    targets.set_defaults(func=cmd_targets)

    collect = sub.add_parser("collect", help="Collect benchmark trajectories and KV caches")
    collect.add_argument("--benchmark", default="hanoi", choices=["hanoi", "sudoku", "game24", "gsm8k", "humaneval", "all"])
    collect.add_argument("--limit", type=int, default=3)
    collect.add_argument("--run-id", default="smoke")
    collect.add_argument("--runs-root", default="runs")
    collect.add_argument("--model-id", default=SMOKE_MODEL)
    collect.add_argument("--device", default="auto")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--max-new-tokens", type=int, default=96)
    collect.add_argument("--layer-mode", default="all", help="all, lower, middle, upper, or comma-separated indices")
    collect.add_argument("--capture-hidden", action="store_true")
    collect.add_argument("--dry-run", action="store_true", help="Create verified oracle records without loading an LLM")
    collect.set_defaults(func=cmd_collect)

    compress = sub.add_parser("compress", help="Run a compression baseline on saved caches")
    compress.add_argument("--run", required=True)
    compress.add_argument("--method", default="random", choices=["random", "pca_svd", "autoencoder", "rae_temporal", "retrieval"])
    compress.add_argument("--latent-dim", type=int, default=64)
    compress.add_argument("--seed", type=int, default=0)
    compress.add_argument("--epochs", type=int, default=1)
    compress.add_argument("--lr", type=float, default=1e-3)
    compress.add_argument("--weight-decay", type=float, default=1e-2)
    compress.add_argument("--hidden-dim", type=int, default=128, help="Hidden size for sequence codecs such as rae_temporal.")
    compress.add_argument("--num-layers", type=int, default=1, help="LSTM depth for sequence codecs such as rae_temporal.")
    compress.add_argument("--model-id", default=None, help="Frozen local LLM for optional rae_temporal prompt-state KL gradients.")
    compress.add_argument("--device", default="auto", help="Device for optional frozen-LLM prompt-state gradients.")
    compress.add_argument("--llm-loss-weight", type=float, default=0.0, help="Weight for frozen-LLM prompt-state transition KL in rae_temporal training.")
    compress.add_argument("--llm-steps", type=int, default=1, help="Prompt-token state transitions per cache for optional frozen-LLM KL.")
    compress.add_argument("--log-every", type=int, default=1, help="Write/print learned-codec training progress every N epochs.")
    compress.add_argument("--checkpoint-every", type=int, default=0, help="Save rae_temporal model checkpoints every N epochs; 0 disables periodic checkpoints.")
    compress.add_argument(
        "--heartbeat-every-batches",
        type=int,
        default=100,
        help="For rae_temporal, write intra-epoch heartbeat events every N mini-batches; 0 disables heartbeats.",
    )
    compress.add_argument("--train-batch-size", type=int, default=0, help="Mini-batch size for rae_temporal training; 0 uses all records at once.")
    compress.set_defaults(func=cmd_compress)

    inject = sub.add_parser("inject", help="Validate or replay a saved cache bundle")
    inject.add_argument("--cache", required=True)
    inject.add_argument("--model-id", default=None)
    inject.add_argument("--device", default="auto")
    inject.add_argument("--max-new-tokens", type=int, default=32)
    inject.add_argument("--validate-only", action="store_true")
    inject.set_defaults(func=cmd_inject)

    evaluate = sub.add_parser("evaluate", help="Aggregate metrics and write report.md")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--baseline", default="no_cache")
    evaluate.add_argument(
        "--behavioral-baseline",
        default=None,
        help="Replay a local cache baseline through the local model and score task behaviour.",
    )
    evaluate.add_argument("--model-id", default=None)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Replay token budget. Defaults to each record's original generated_tokens when omitted.",
    )
    evaluate.add_argument("--limit", type=int, default=None, help="Limit behavioural replay records when --behavioral-baseline is used.")
    evaluate.set_defaults(func=cmd_evaluate)

    behavior = sub.add_parser("behavior", help="Run local behavioural replay baselines from saved caches")
    behavior.add_argument("--run", required=True)
    behavior.add_argument(
        "--baseline",
        nargs="+",
        default=["original_cache"],
    )
    behavior.add_argument("--model-id", default=None)
    behavior.add_argument("--device", default="auto")
    behavior.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Replay token budget. Defaults to each record's original generated_tokens when omitted.",
    )
    behavior.add_argument("--limit", type=int, default=None, help="Limit cache-backed records for partial artifacts or smoke runs.")
    behavior.set_defaults(func=cmd_behavior)

    validate_codec = sub.add_parser(
        "validate-codec",
        help="Validate that compression latents decode to replay-compatible cache shapes",
    )
    validate_codec.add_argument("--run", required=True)
    validate_codec.add_argument("--method", required=True, help="Compression artifact method, e.g. rae_temporal or retrieval.")
    validate_codec.set_defaults(func=cmd_validate_codec)

    replay_fidelity = sub.add_parser(
        "replay-fidelity",
        help="Compare original vs reconstructed-cache logits after the first replay step",
    )
    replay_fidelity.add_argument("--run", required=True)
    replay_fidelity.add_argument("--method", required=True, help="Compression artifact method, e.g. rae_temporal.")
    replay_fidelity.add_argument("--model-id", default=None)
    replay_fidelity.add_argument("--device", default="auto")
    replay_fidelity.add_argument("--limit", type=int, default=None)
    replay_fidelity.add_argument("--steps", type=int, default=1, help="Teacher-forced generated tokens to compare.")
    replay_fidelity.set_defaults(func=cmd_replay_fidelity)

    training_curve = sub.add_parser(
        "training-curve",
        help="Summarize learned-codec training loss dynamics",
    )
    training_curve.add_argument("--run", required=True)
    training_curve.add_argument("--method", required=True, help="Learned codec method, e.g. autoencoder or rae_temporal.")
    training_curve.set_defaults(func=cmd_training_curve)

    corruption = sub.add_parser(
        "corruption-sensitivity",
        help="Replay original-to-reconstructed cache interpolations to measure behavioural robustness",
    )
    corruption.add_argument("--run", required=True)
    corruption.add_argument("--method", required=True, help="Compression artifact method, e.g. rae_temporal.")
    corruption.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    corruption.add_argument("--model-id", default=None)
    corruption.add_argument("--device", default="auto")
    corruption.add_argument("--limit", type=int, default=None)
    corruption.add_argument("--max-new-tokens", type=int, default=None)
    corruption.set_defaults(func=cmd_corruption_sensitivity)

    prompt = sub.add_parser("prompt-baseline", help="Run local prompt baselines without remote APIs")
    prompt.add_argument("--run", required=True)
    prompt.add_argument("--benchmark", default="hanoi", choices=["hanoi", "sudoku", "game24", "gsm8k", "humaneval", "all"])
    prompt.add_argument(
        "--baseline",
        nargs="+",
        default=["standard"],
        choices=["all", "standard", "cot", "self_consistency", "retry_reflection"],
    )
    prompt.add_argument("--model-id", default=SMOKE_MODEL)
    prompt.add_argument("--device", default="auto")
    prompt.add_argument(
        "--baseline-tier",
        default="custom",
        choices=sorted(BASELINE_TIERS),
        help="Named baseline budget preset. Explicit --limit or --max-new-tokens override the preset values.",
    )
    prompt.add_argument("--limit", type=int, default=None)
    prompt.add_argument("--seed", type=int, default=0)
    prompt.add_argument("--max-new-tokens", type=int, default=None)
    prompt.add_argument("--samples", type=int, default=5)
    prompt.add_argument("--temperature", type=float, default=0.7)
    prompt.add_argument("--resume", action="store_true", help="Append missing examples and skip existing task IDs.")
    prompt.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Run prompt-baseline in resumable child-process chunks of N examples; useful for long MPS runs.",
    )
    prompt.set_defaults(func=cmd_prompt_baseline)

    prompt_cache = sub.add_parser(
        "collect-prompt-caches",
        help="Collect prompt-protocol outputs plus replayable KV cache bundles",
    )
    prompt_cache.add_argument("--run", required=True)
    prompt_cache.add_argument("--config", default=None, help="YAML/JSON experiment config to resolve and save with the run.")
    prompt_cache.add_argument("--benchmark", default=None, choices=["hanoi", "sudoku", "game24", "gsm8k", "humaneval", "all"])
    prompt_cache.add_argument("--baseline", default=None, choices=["standard", "cot", "retry_reflection"])
    prompt_cache.add_argument("--model-id", default=None)
    prompt_cache.add_argument("--device", default=None)
    prompt_cache.add_argument(
        "--baseline-tier",
        default=None,
        choices=sorted(BASELINE_TIERS),
        help="Named baseline budget preset. Explicit --limit or --max-new-tokens override the preset values.",
    )
    prompt_cache.add_argument("--limit", type=int, default=None)
    prompt_cache.add_argument("--seed", type=int, default=None)
    prompt_cache.add_argument("--max-new-tokens", type=int, default=None)
    prompt_cache.add_argument("--layer-mode", default=None, help="all, lower, middle, upper, or comma-separated indices")
    prompt_cache.add_argument("--capture-hidden", action="store_true")
    prompt_cache.add_argument("--resume", action="store_true", help="Append missing examples and skip existing task IDs.")
    prompt_cache.set_defaults(func=cmd_collect_prompt_caches)

    attach_cache = sub.add_parser(
        "attach-prompt-caches",
        help="Attach replayable prompt KV caches to an existing prompt-record JSONL without regenerating outputs",
    )
    attach_cache.add_argument("--run", required=True)
    attach_cache.add_argument("--source-records", required=True, help="Existing prompt baseline JSONL to preserve labels and outputs from.")
    attach_cache.add_argument("--model-id", default=None)
    attach_cache.add_argument("--device", default=None)
    attach_cache.add_argument("--layer-mode", default=None, help="all, lower, middle, upper, or comma-separated indices")
    attach_cache.add_argument("--capture-hidden", action="store_true")
    attach_cache.add_argument("--resume", action="store_true", help="Append missing examples and skip existing task IDs.")
    attach_cache.set_defaults(func=cmd_attach_prompt_caches)

    log = sub.add_parser("log", help="Append a structured research-log entry")
    log.add_argument("--path", default="docs/RESEARCH_LOG.md")
    log.add_argument("--title", required=True)
    log.add_argument("--worked", action="append", default=[])
    log.add_argument("--did-not-work", action="append", default=[])
    log.add_argument("--todo", action="append", default=[])
    log.set_defaults(func=cmd_log)

    tot = sub.add_parser("tot-baseline", help="Run a local Tree-of-Thought style Game of 24 baseline")
    tot.add_argument("--run", required=True)
    tot.add_argument("--limit", type=int, default=6)
    tot.add_argument("--seed", type=int, default=0)
    tot.add_argument("--breadth", type=int, default=5)
    tot.set_defaults(func=cmd_tot_baseline)

    react = sub.add_parser("react-baseline", help="Run a local ReAct-style tool-environment baseline")
    react.add_argument("--run", required=True)
    react.add_argument("--limit", type=int, default=3)
    react.add_argument("--max-steps", type=int, default=8)
    react.set_defaults(func=cmd_react_baseline)

    check = sub.add_parser("check-targets", help="Compare run metrics against tracked reported targets")
    check.add_argument("--run", required=True)
    check.add_argument("--tolerance", type=float, default=1.0, help="Allowed absolute percentage-point delta")
    check.set_defaults(func=cmd_check_targets)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = _run_dir_from_args(args)
    start = time.perf_counter()
    _append_run_event(
        run_dir,
        {
            "event": "command_start",
            "command": args.command,
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "args": vars(args),
            "pid": os.getpid(),
        },
    )
    try:
        exit_code = int(args.func(args))
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        _append_run_event(
            run_dir,
            {
                "event": "command_error",
                "command": args.command,
                "elapsed_s": time.perf_counter() - start,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
            },
        )
        raise
    _append_run_event(
        run_dir,
        {
            "event": "command_finish",
            "command": args.command,
            "elapsed_s": time.perf_counter() - start,
            "exit_code": exit_code,
            "pid": os.getpid(),
        },
    )
    return exit_code
