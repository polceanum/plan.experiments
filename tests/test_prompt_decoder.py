from pathlib import Path

import torch

from latent_kv.prompt_decoder import export_prompt_decoder_dataset, train_prompt_decoder
from latent_kv.schemas import TrajectoryRecord, append_jsonl, read_json, read_jsonl


def test_export_prompt_decoder_dataset_pairs_latents_with_prompts(tmp_path: Path):
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "records.jsonl",
        TrajectoryRecord(
            run_id="run",
            benchmark="gsm8k",
            task_id="gsm8k_0000",
            model_id="fake",
            seed=0,
            attempt_id=0,
            prompt="Solve 1+1.",
            target="2",
            output_text="The answer is 2",
            parsed_answer="2",
            correct=True,
            retry_index=0,
            prompt_tokens=4,
            generated_tokens=5,
            metadata={"cache_mode": "trajectory"},
        ),
    )
    torch.save(
        {
            "latents": torch.ones(1, 3),
            "annotations": [{"task_id": "gsm8k_0000", "primary_category": "arithmetic_add_subtract", "categories": ["arithmetic_add_subtract"]}],
            "checkpoint_metadata": {"epoch": 1},
        },
        analysis_dir / "checkpoint_latents.pt",
    )

    summary = export_prompt_decoder_dataset(run_dir, analysis_dir)

    rows = read_jsonl(analysis_dir / "prompt_decoder" / "prompt_decoder_rows.jsonl")
    saved_summary = read_json(analysis_dir / "prompt_decoder" / "prompt_decoder_summary.json")
    payload = torch.load(analysis_dir / "prompt_decoder" / "prompt_decoder_dataset.pt", map_location="cpu")
    assert summary.rows == 1
    assert saved_summary["latent_dim"] == 3
    assert saved_summary["latent_shape"] == [1, 3]
    assert rows[0]["prompt"] == "Solve 1+1."
    assert rows[0]["cache_mode"] == "trajectory"
    assert torch.equal(payload["latents"], torch.ones(1, 3))


def test_export_prompt_decoder_dataset_reports_structured_latent_shape(tmp_path: Path):
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "records.jsonl",
        TrajectoryRecord(
            run_id="run",
            benchmark="gsm8k",
            task_id="gsm8k_0000",
            model_id="fake",
            seed=0,
            attempt_id=0,
            prompt="Solve 2+2.",
            target="4",
            output_text="The answer is 4",
            parsed_answer="4",
            correct=True,
            retry_index=0,
            metadata={"cache_mode": "trajectory"},
        ),
    )
    torch.save(
        {
            "latents": torch.ones(1, 2, 5),
            "annotations": [{"task_id": "gsm8k_0000"}],
            "checkpoint_metadata": {"epoch": 1},
        },
        analysis_dir / "checkpoint_latents.pt",
    )

    summary = export_prompt_decoder_dataset(run_dir, analysis_dir)
    saved_summary = read_json(analysis_dir / "prompt_decoder" / "prompt_decoder_summary.json")

    assert summary.latent_dim == 5
    assert summary.latent_shape == [1, 2, 5]
    assert saved_summary["latent_dim"] == 5
    assert saved_summary["latent_shape"] == [1, 2, 5]


def test_train_prompt_decoder_writes_model_and_decodes_rows(tmp_path: Path):
    dataset_dir = tmp_path / "prompt_decoder"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "prompt_decoder_dataset.pt"
    rows = [
        {"index": 0, "task_id": "a", "prompt": "Solve A.", "prompt_token_ids": [3, 4, 5]},
        {"index": 1, "task_id": "b", "prompt": "Solve B.", "prompt_token_ids": [3, 4, 6]},
    ]
    torch.save(
        {
            "latents": torch.tensor([[[1.0, 0.0], [0.5, 0.0]], [[0.0, 1.0], [0.0, 0.5]]]),
            "rows": rows,
            "checkpoint_metadata": {"epoch": 1},
        },
        dataset_path,
    )

    summary = train_prompt_decoder(
        dataset_path,
        epochs=2,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        max_prompt_tokens=4,
        max_latent_chunks=1,
        log_every=0,
        progress_every_batches=0,
    )

    output_dir = dataset_dir / "prompt_decoder_model"
    saved_summary = read_json(output_dir / "prompt_decoder_train_summary.json")
    decoded = read_jsonl(output_dir / "decoded_prompt_tokens.jsonl")
    assert summary.rows == 2
    assert summary.vocab_size == 4
    assert saved_summary["original_token_vocab_size"] == 7
    assert saved_summary["max_prompt_tokens"] == 3
    assert saved_summary["max_latent_chunks"] == 1
    assert saved_summary["progress_every_batches"] == 0
    assert (output_dir / "prompt_token_decoder.pt").exists()
    assert len(decoded) == 2
    assert "decoded_prompt_token_ids" in decoded[0]
