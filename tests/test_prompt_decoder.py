from pathlib import Path

import torch

from latent_kv.prompt_decoder import export_prompt_decoder_dataset
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
