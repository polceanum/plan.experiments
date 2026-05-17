from pathlib import Path
from types import SimpleNamespace

import torch

import latent_kv.latent_interpolation as interpolation
from latent_kv.cache import CacheTuple, save_cache_bundle
from latent_kv.latent_interpolation import (
    interpolate_latents,
    parse_alphas,
    run_latent_interpolation,
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


class _FakeRAE(torch.nn.Module):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = int(z.shape[0])
        return torch.zeros(batch, 2, 2)


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
                prompt=f"Prompt {idx}",
                target="1",
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
            "checkpoint_metadata": {"checkpoint_path": str(checkpoint_path), "checkpoint_epoch": 1},
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
    monkeypatch.setattr(interpolation, "greedy_continue_from_loaded_bundle", lambda **kwargs: "The answer is 1")

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
    rows = (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_replays.jsonl").read_text(encoding="utf-8")
    assert '"replay_context": "a"' in rows
    assert '"replay_context": "b"' in rows
    assert "endpoint_prompt" in rows
    assert "decoded_output" in rows
    assert (run_dir / "analysis" / "interpolations_epoch_1" / "interpolation_sequences.md").exists()
