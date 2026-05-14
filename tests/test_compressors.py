from pathlib import Path

import torch

from latent_kv.cache import CacheTuple, save_cache_bundle
from latent_kv.behavior import _load_reconstructed_cache
from latent_kv.codec_validation import validate_cache_against_bundle, validate_reconstructed_artifact
from latent_kv.compressors import run_compression
from latent_kv.schemas import CacheMetadata, TrajectoryRecord, append_jsonl


def _cache(offset: int) -> CacheTuple:
    key = torch.arange(offset, offset + 8, dtype=torch.float32).reshape(1, 1, 4, 2)
    value = torch.arange(offset + 8, offset + 16, dtype=torch.float32).reshape(1, 1, 4, 2)
    return ((key, value),)


def _write_run(tmp_path: Path) -> None:
    for idx in range(2):
        cache_path = tmp_path / "caches" / f"{idx}.pt"
        metadata = CacheMetadata(
            model_id="fake",
            tokenizer_id="fake",
            dtype="torch.float32",
            device="cpu",
            layers=1,
            selected_layers=[0],
            selected_heads=None,
            token_count=4,
            cache_path=str(cache_path),
        )
        save_cache_bundle(
            cache_path,
            _cache(idx),
            metadata,
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            last_logits=torch.randn(1, 10),
        )
        append_jsonl(
            tmp_path / "records.jsonl",
            TrajectoryRecord(
                run_id=tmp_path.name,
                benchmark="hanoi",
                task_id=f"task_{idx}",
                model_id="fake",
                seed=0,
                attempt_id=0,
                prompt="p",
                target="t",
                output_text="o",
                parsed_answer="o",
                correct=bool(idx),
                retry_index=0,
                cache_path=str(cache_path),
            ),
        )


def test_random_projection_compression_writes_artifacts(tmp_path: Path):
    _write_run(tmp_path)
    result = run_compression(tmp_path, method="random", latent_dim=2, seed=0)
    assert result.records == 2
    assert result.reconstruction_mse is not None
    assert Path(result.latent_path).exists()
    assert Path(result.artifact_path).exists()
    payload = torch.load(result.latent_path, map_location="cpu")
    assert payload["lengths"] == [16, 16]
    assert len(payload["cache_paths"]) == 2
    assert payload["codec_contract"]["point_codec"] is True
    assert payload["latents"].shape[0] == 2


def test_reconstructed_cache_can_be_mapped_back_to_original_shapes(tmp_path: Path):
    _write_run(tmp_path)
    run_compression(tmp_path, method="retrieval", latent_dim=2, seed=0)
    cache, mse = _load_reconstructed_cache(tmp_path, "retrieval", str(tmp_path / "caches" / "0.pt"))
    assert len(cache) == 1
    assert cache[0][0].shape == (1, 1, 4, 2)
    assert mse is not None


def test_reconstructed_artifact_validation_reports_decodable_points(tmp_path: Path):
    _write_run(tmp_path)
    run_compression(tmp_path, method="retrieval", latent_dim=2, seed=0)
    validation = validate_reconstructed_artifact(tmp_path, "retrieval")

    assert validation.records == 2
    assert validation.one_point_per_cache is True
    assert validation.valid_caches == 2
    assert validation.invalid_caches == 0
    assert validation.mean_reconstruction_mse is not None
    assert (tmp_path / "compressions" / "retrieval_validation.json").exists()


def test_cache_validation_catches_shape_mismatch(tmp_path: Path):
    _write_run(tmp_path)
    bundle = torch.load(tmp_path / "caches" / "0.pt", map_location="cpu")
    bad_cache = ((torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 4, 2)),)

    result = validate_cache_against_bundle(bad_cache, bundle)

    assert result.valid is False
    assert result.shape_match is False
    assert any("key_shape" in error for error in result.errors)
