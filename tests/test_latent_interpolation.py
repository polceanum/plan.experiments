from pathlib import Path
from types import SimpleNamespace

import torch

import latent_kv.latent_interpolation as interpolation
from latent_kv.cache import CacheTuple, save_cache_bundle
from latent_kv.latent_interpolation import (
    candidate_plan_quality,
    interpolate_latents,
    parse_alphas,
    reconstruction_faithfulness,
    run_latent_interpolation,
    run_reconstruction_scan,
    select_interpolation_pairs,
)
from latent_kv.schemas import CacheMetadata, TrajectoryRecord, append_jsonl


def _cache(offset: int = 0) -> CacheTuple:
    key = torch.tensor([[[[float(offset)], [float(offset + 1)]]]])
    value = torch.tensor([[[[float(offset + 2)], [float(offset + 3)]]]])
    return ((key, value),)


def test_parse_alphas_and_interpolate_latents():
    assert parse_alphas("0,0.25,1") == [0.0, 0.25, 1.0]

    a = torch.tensor([0.0, 2.0])
    b = torch.tensor([2.0, 4.0])
    z = interpolate_latents(a, b, 0.25)

    assert torch.allclose(z, torch.tensor([0.5, 2.5]))


def test_reconstruction_faithfulness_rejects_right_number_wrong_story():
    prompt = "Solve the math problem. Give the final numeric answer. Harry slept 9 hours. James slept 6 hours. How many more hours did Harry sleep?"
    wrong_story = "There are 3 people in the group and each person gets a toy. The answer is 3."
    faithful_story = "Harry slept 9 hours and James slept 6 hours, so Harry slept 3 more hours. The answer is 3."

    assert not reconstruction_faithfulness(prompt, wrong_story, decoded_correct=True)["convincing"]
    assert reconstruction_faithfulness(prompt, faithful_story, decoded_correct=True)["convincing"]
    assert not reconstruction_faithfulness(prompt, faithful_story, decoded_correct=False)["convincing"]


def test_candidate_plan_quality_does_not_require_endpoint_correctness():
    output = "Let x be the number of books. First compute 21 - 3 = 18. Then 18 / 2 = 9. Therefore, the answer is 9."

    quality = candidate_plan_quality(output, replay_error=None)

    assert quality["inspectable"]
    assert quality["potentially_solved"]


def test_candidate_plan_quality_flags_token_cap_truncation():
    output = "Let x be the number of books. First compute 21 - 3 = 18."

    quality = candidate_plan_quality(output, replay_error=None, hit_max_tokens=True)

    assert quality["appears_truncated"]


def test_select_interpolation_pairs_filters_correct_and_mixed_modes():
    latents = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 1.0],
        ]
    )
    annotations = [
        {"task_id": "a", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "b", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "c", "correct": True, "primary_category": "rate_time_work"},
        {"task_id": "d", "correct": False, "primary_category": "rate_time_work"},
        {"task_id": "e", "correct": True, "primary_category": "fractions_ratios_percents"},
    ]

    pairs = select_interpolation_pairs(latents, annotations, pairs=2, pair_mode="mixed")

    assert {pair.pair_type for pair in pairs} == {"same_category", "cross_category"}
    assert all(annotations[pair.a_index]["correct"] and annotations[pair.b_index]["correct"] for pair in pairs)
    assert len({tuple(sorted((pair.a_index, pair.b_index))) for pair in pairs}) == len(pairs)


def test_select_interpolation_pairs_flattens_structured_latents_for_distance():
    latents = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    annotations = [
        {"task_id": "a", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "b", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "c", "correct": True, "primary_category": "rate_time_work"},
    ]

    pairs = select_interpolation_pairs(latents, annotations, pairs=1, pair_mode="same_category")

    assert len(pairs) == 1
    assert {pairs[0].a_index, pairs[0].b_index} == {0, 1}


def test_select_interpolation_pairs_can_spread_and_filter_near_duplicates():
    latents = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.4, 0.6],
            [0.0, 1.0],
        ]
    )
    annotations = [
        {"task_id": "a", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "b", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "c", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "d", "correct": True, "primary_category": "money_price_profit"},
    ]
    records = [
        {"prompt": "Solve the math problem. Give the final numeric answer. Apples cost 1 dollar.", "target": "1"},
        {"prompt": "Solve the math problem. Give the final numeric answer. Apples cost 1 dollar.", "target": "1"},
        {"prompt": "Solve the math problem. Give the final numeric answer. A train travels for several hours.", "target": "2"},
        {"prompt": "Solve the math problem. Give the final numeric answer. A bakery sells many cakes.", "target": "3"},
    ]

    pairs = select_interpolation_pairs(
        latents,
        annotations,
        records=records,
        pairs=1,
        pair_mode="same_category",
        selection="spread",
        min_distance=0.05,
        max_distance=1.5,
        max_prompt_overlap=0.5,
    )

    assert pairs[0].distance >= 0.05
    assert pairs[0].distance <= 1.5
    assert {pairs[0].a_task_id, pairs[0].b_task_id} != {"a", "b"}


def test_select_interpolation_pairs_can_require_reconstructed_correct_endpoints():
    latents = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    annotations = [
        {"task_id": "a", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "b", "correct": True, "primary_category": "money_price_profit"},
        {"task_id": "c", "correct": True, "primary_category": "rate_time_work"},
        {"task_id": "d", "correct": True, "primary_category": "rate_time_work"},
    ]

    pairs = select_interpolation_pairs(
        latents,
        annotations,
        pairs=1,
        pair_mode="cross_category",
        eligible_indices={1, 3},
    )

    assert len(pairs) == 1
    assert {pairs[0].a_index, pairs[0].b_index} == {1, 3}


class _FakeRAE(torch.nn.Module):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = int(z.shape[0])
        return torch.zeros(batch, 2, 2)


class _ShapeCheckingRAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_shape: tuple[int, ...] | None = None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(z.shape)
        return torch.zeros(int(z.shape[0]), 2, 2)


def test_decode_latent_to_cache_preserves_structured_latent_shape():
    model = _ShapeCheckingRAE()

    cache = interpolation._decode_latent_to_cache(
        z=torch.zeros(2, 3),
        endpoint_shapes=[((1, 1, 2, 1), (1, 1, 2, 1))],
        aligned_shapes=[((1, 1, 2, 1), (1, 1, 2, 1))],
        model=model,
        mean=torch.zeros(1, 1, 2),
        std=torch.ones(1, 1, 2),
    )

    assert model.seen_shape == (1, 2, 3)
    assert cache[0][0].shape == (1, 1, 2, 1)


def _write_interpolation_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    for idx, category in enumerate(["money_price_profit", "rate_time_work"]):
        cache_path = run_dir / "caches" / f"{idx}.pt"
        metadata = CacheMetadata(
            model_id="fake-model",
            tokenizer_id="fake-tokenizer",
            dtype="torch.float32",
            device="cpu",
            layers=1,
            selected_layers=[0],
            selected_heads=None,
            token_count=2,
            cache_path=str(cache_path),
        )
        save_cache_bundle(
            cache_path,
            _cache(idx),
            metadata,
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
            last_logits=torch.zeros(1, 4),
            generation_token_ids=torch.tensor([[3, 4]]),
        )
        append_jsonl(
            run_dir / "records.jsonl",
            TrajectoryRecord(
                run_id="run",
                benchmark="gsm8k",
                task_id=f"task_{idx}",
                model_id="fake-model",
                seed=0,
                attempt_id=0,
                prompt=f"Solve the math problem. Give the final numeric answer. {'Apples cost money' if idx == 0 else 'Trains travel daily'}",
                target=str(idx + 1),
                output_text="The answer is 1",
                parsed_answer="1",
                correct=True,
                retry_index=0,
                cache_path=str(cache_path),
                generated_tokens=4,
            ),
        )
    analysis_dir.mkdir(parents=True)
    rows = [
        {
            "task_id": "task_0",
            "correct": True,
            "primary_category": "money_price_profit",
            "categories": ["money_price_profit"],
            "difficulty_proxy": "single_step",
            "category_notes": "",
        },
        {
            "task_id": "task_1",
            "correct": True,
            "primary_category": "rate_time_work",
            "categories": ["rate_time_work"],
            "difficulty_proxy": "single_step",
            "category_notes": "",
        },
    ]
    with (analysis_dir / "task_categories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(__import__("json").dumps(row) + "\n")
    checkpoint_path = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    torch.save(
        {
            "latents": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "checkpoint_metadata": {"checkpoint_path": str(checkpoint_path), "checkpoint_epoch": 1, "seq_len": 2},
            "annotations": rows,
        },
        analysis_dir / "checkpoint_latents.pt",
    )
    return run_dir


def test_run_latent_interpolation_writes_inspectable_rows(tmp_path: Path, monkeypatch):
    run_dir = _write_interpolation_run(tmp_path)
    checkpoint = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"

    monkeypatch.setattr(
        interpolation,
        "_load_checkpoint_model",
        lambda path: (
            _FakeRAE(),
            {
                "normalization_mean": torch.zeros(1, 1, 2),
                "normalization_std": torch.ones(1, 1, 2),
            },
        ),
    )
    monkeypatch.setattr(interpolation, "load_model_and_tokenizer", lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()))
    replay_budgets: list[int] = []
    monkeypatch.setattr(
        interpolation,
        "greedy_continue_from_loaded_bundle",
        lambda **kwargs: replay_budgets.append(kwargs["max_new_tokens"]) or "The answer is 1",
    )

    summary = run_latent_interpolation(
        run_dir,
        checkpoint_path=checkpoint,
        pairs=1,
        alphas=[0.0, 1.0],
        pair_mode="cross_category",
        replay_device_name="cpu",
        progress_every=0,
    )

    assert summary.pairs == 1
    assert summary.replay_rows == 4
    assert replay_budgets == [512, 512, 512, 512]
    rows = (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_replays.jsonl").read_text(encoding="utf-8")
    assert '"replay_context": "a"' in rows
    assert '"replay_context": "b"' in rows
    assert "endpoint_prompt" in rows
    assert "decoded_output" in rows
    assert "candidate_plan_quality" in rows
    assert "replay_generation" in rows
    assert (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_sequences.md").exists()
    assert (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_candidate_plans.md").exists()
    transition_report = run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_plan_transition_tables.md"
    assert transition_report.exists()
    assert "Endpoint A original -> decoded/interpolated alpha rows -> Endpoint B original" in transition_report.read_text(
        encoding="utf-8"
    )
    latent_line_report = run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_latent_line.md"
    assert latent_line_report.exists()
    latent_line_text = latent_line_report.read_text(encoding="utf-8")
    assert "Endpoint A solved plan -> interpolated latent points -> Endpoint B solved plan" in latent_line_text
    assert "Middle points are judged as self-contained candidate plans" in latent_line_text
    report = (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_inspection.md").read_text(encoding="utf-8")
    assert "Endpoint A solved plan" in report
    assert "Decoded A reconstruction" in report
    assert "Decoded B reconstruction" in report
    assert "Endpoint B solved plan" in report
    solved_report = (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_inspection_solved_reconstructions.md")
    assert solved_report.exists()


def test_run_latent_interpolation_can_filter_from_reconstruction_scan(tmp_path: Path, monkeypatch):
    run_dir = _write_interpolation_run(tmp_path)
    checkpoint = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    scan_path = run_dir / "analysis" / "scan" / "reconstruction_replays.jsonl"
    scan_path.parent.mkdir(parents=True)
    scan_path.write_text(
        "\n".join(
            [
                '{"index": 0, "decoded_correct": true, "replay_error": null}',
                '{"index": 1, "decoded_correct": true, "replay_error": null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        interpolation,
        "_load_checkpoint_model",
        lambda path: (
            _FakeRAE(),
            {
                "normalization_mean": torch.zeros(1, 1, 2),
                "normalization_std": torch.ones(1, 1, 2),
            },
        ),
    )
    monkeypatch.setattr(interpolation, "load_model_and_tokenizer", lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(interpolation, "greedy_continue_from_loaded_bundle", lambda **kwargs: "The answer is 1")

    summary = run_latent_interpolation(
        run_dir,
        checkpoint_path=checkpoint,
        pairs=1,
        alphas=[0.0, 1.0],
        pair_mode="cross_category",
        replay_device_name="cpu",
        progress_every=0,
        reconstruction_scan_path=scan_path,
    )

    assert summary.pairs == 1
    assert summary.reconstruction_scan_path == str(scan_path)
    assert summary.eligible_endpoint_count == 2


def test_run_latent_interpolation_can_require_convincing_scan_rows(tmp_path: Path, monkeypatch):
    run_dir = _write_interpolation_run(tmp_path)
    checkpoint = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    scan_path = run_dir / "analysis" / "scan" / "reconstruction_replays.jsonl"
    scan_path.parent.mkdir(parents=True)
    scan_path.write_text(
        "\n".join(
            [
                '{"index": 0, "decoded_correct": true, "decoded_convincing": true, "replay_error": null}',
                '{"index": 1, "decoded_correct": true, "decoded_convincing": true, "replay_error": null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        interpolation,
        "_load_checkpoint_model",
        lambda path: (
            _FakeRAE(),
            {
                "normalization_mean": torch.zeros(1, 1, 2),
                "normalization_std": torch.ones(1, 1, 2),
            },
        ),
    )
    monkeypatch.setattr(interpolation, "load_model_and_tokenizer", lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(interpolation, "greedy_continue_from_loaded_bundle", lambda **kwargs: "The answer is 1")

    summary = run_latent_interpolation(
        run_dir,
        checkpoint_path=checkpoint,
        pairs=1,
        alphas=[0.0, 1.0],
        pair_mode="cross_category",
        replay_device_name="cpu",
        progress_every=0,
        reconstruction_scan_path=scan_path,
        require_convincing_reconstruction=True,
    )

    assert summary.pairs == 1
    assert summary.require_convincing_reconstruction


def test_run_reconstruction_scan_writes_endpoint_replay_rows(tmp_path: Path, monkeypatch):
    run_dir = _write_interpolation_run(tmp_path)
    checkpoint = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    replay_budgets: list[int] = []

    monkeypatch.setattr(
        interpolation,
        "_load_checkpoint_model",
        lambda path: (
            _FakeRAE(),
            {
                "normalization_mean": torch.zeros(1, 1, 2),
                "normalization_std": torch.ones(1, 1, 2),
            },
        ),
    )
    monkeypatch.setattr(interpolation, "load_model_and_tokenizer", lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(
        interpolation,
        "greedy_continue_from_loaded_bundle",
        lambda **kwargs: replay_budgets.append(kwargs["max_new_tokens"])
        or {
            "text": "The answer is 1",
            "generated_tokens": 3,
            "max_new_tokens": kwargs["max_new_tokens"],
            "hit_max_tokens": False,
            "stop_reason": "eos",
        },
    )

    summary = run_reconstruction_scan(
        run_dir,
        checkpoint_path=checkpoint,
        replay_device_name="cpu",
        max_new_tokens=4,
        progress_every=0,
    )

    assert summary.scanned == 2
    assert summary.solved_reconstructions == 1
    assert summary.convincing_reconstructions == 0
    assert summary.max_new_tokens == 4
    assert summary.max_effective_new_tokens == 4
    assert summary.token_budget_policy == "fixed"
    assert replay_budgets == [4, 4]
    rows = (run_dir / "analysis" / "reconstruction_scan_epoch_1" / "reconstruction_replays.jsonl").read_text(encoding="utf-8")
    assert "decoded_correct" in rows
    assert "decoded_convincing" in rows
    assert "faithfulness" in rows
    assert "original_output" in rows
    assert "replay_generation" in rows


def test_run_reconstruction_scan_defaults_to_generous_source_generation_budget(tmp_path: Path, monkeypatch):
    run_dir = _write_interpolation_run(tmp_path)
    checkpoint = run_dir / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    replay_budgets: list[int] = []

    monkeypatch.setattr(
        interpolation,
        "_load_checkpoint_model",
        lambda path: (
            _FakeRAE(),
            {
                "normalization_mean": torch.zeros(1, 1, 2),
                "normalization_std": torch.ones(1, 1, 2),
            },
        ),
    )
    monkeypatch.setattr(interpolation, "load_model_and_tokenizer", lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(
        interpolation,
        "greedy_continue_from_loaded_bundle",
        lambda **kwargs: replay_budgets.append(kwargs["max_new_tokens"])
        or {
            "text": "The answer is 1",
            "generated_tokens": 3,
            "max_new_tokens": kwargs["max_new_tokens"],
            "hit_max_tokens": False,
            "stop_reason": "eos",
        },
    )

    summary = run_reconstruction_scan(
        run_dir,
        checkpoint_path=checkpoint,
        replay_device_name="cpu",
        progress_every=0,
    )

    assert replay_budgets == [512, 512]
    assert summary.max_new_tokens is None
    assert summary.max_effective_new_tokens == 512
    assert summary.token_budget_policy == "max_512_or_source_generated_tokens"
    assert summary.token_cap_hits == 0
