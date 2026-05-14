from pathlib import Path

import torch

from latent_kv.cache import CacheTuple, flatten_cache, load_cache_bundle, save_cache_bundle
from latent_kv.behavior import _load_reconstructed_cache
from latent_kv.codec_validation import validate_cache_against_bundle, validate_reconstructed_artifact
from latent_kv.compressors import ChunkedLSTMAutoEncoder, run_compression
from latent_kv.schemas import CacheMetadata, TrajectoryRecord, append_jsonl
from latent_kv.schemas import read_jsonl


def _cache(offset: int) -> CacheTuple:
    key = torch.arange(offset, offset + 8, dtype=torch.float32).reshape(1, 1, 4, 2)
    value = torch.arange(offset + 8, offset + 16, dtype=torch.float32).reshape(1, 1, 4, 2)
    return ((key, value),)


def _cache_with_tokens(offset: int, tokens: int) -> CacheTuple:
    values = torch.arange(offset, offset + tokens * 4, dtype=torch.float32)
    key = values[: tokens * 2].reshape(1, 1, tokens, 2)
    value = values[tokens * 2 :].reshape(1, 1, tokens, 2)
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


def _write_variable_length_run(tmp_path: Path) -> None:
    for idx, tokens in enumerate([4, 3]):
        cache_path = tmp_path / "caches" / f"{idx}.pt"
        metadata = CacheMetadata(
            model_id="fake",
            tokenizer_id="fake",
            dtype="torch.float32",
            device="cpu",
            layers=1,
            selected_layers=[0],
            selected_heads=None,
            token_count=tokens,
            cache_path=str(cache_path),
        )
        save_cache_bundle(
            cache_path,
            _cache_with_tokens(idx * 100, tokens),
            metadata,
            input_ids=torch.ones(1, tokens, dtype=torch.long),
            attention_mask=torch.ones(1, tokens, dtype=torch.long),
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
    assert payload["source_labels"][0]["task_id"] == "task_0"
    assert payload["source_labels"][0]["correct"] is False
    assert payload["source_labels"][1]["correct"] is True
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


def test_chunked_lstm_autoencoder_maps_sequence_to_one_point():
    model = ChunkedLSTMAutoEncoder(input_dim=10, latent_dim=3, chunk_dim=4, hidden_dim=5)
    x = torch.randn(2, 10)

    z = model.encode(x)
    decoded = model.decode(z)

    assert z.shape == (2, 3)
    assert decoded.shape == x.shape


def test_lstm_rae_compression_writes_decodable_point_artifact(tmp_path: Path):
    _write_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_lstm",
        latent_dim=3,
        seed=0,
        epochs=1,
        chunk_dim=4,
        hidden_dim=5,
        weight_decay=0.02,
        log_every=1,
    )
    validation = validate_reconstructed_artifact(tmp_path, "rae_lstm")
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    assert payload["latents"].shape == (2, 3)
    assert payload["codec_contract"]["input_representation"] == "chunked_flattened_full_cache"
    assert artifact["chunk_dim"] == 4
    assert artifact["hidden_dim"] == 5
    assert artifact["weight_decay"] == 0.02
    assert artifact["masked_loss"] is True
    assert artifact["objective"] == "masked_reconstruction_mse_no_kl"
    assert artifact["kl_loss_weight"] == 0.0
    assert artifact["regularization"] == "adamw_weight_decay_only"
    assert artifact["decoder_conditioning"] == "latent_repeated_input_plus_learned_position"
    assert artifact["latent_summary"] == "last_hidden_plus_mean_encoded"
    assert artifact["chunk_projection"] == "linear_layernorm_gelu"
    assert artifact["latent_encoding_input"] == "masked_normalized_cache"
    assert len(artifact["training_history"]) == 1
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_lstm_training.jsonl")
    assert training_rows[0]["method"] == "rae_lstm"
    assert training_rows[0]["epoch"] == 1
    assert training_rows[0]["masked_loss"] is True
    assert training_rows[0]["objective"] == "masked_reconstruction_mse_no_kl"
    assert training_rows[0]["loss_components"]["kl"] == 0.0
    assert training_rows[0]["valid_values"] == 32
    assert payload["training_log_path"].endswith("rae_lstm_training.jsonl")
    assert validation.records == 2
    assert validation.one_point_per_cache is True
    assert validation.valid_caches == 2


def test_lstm_rae_latents_use_masked_normalized_inputs(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_lstm",
        latent_dim=3,
        seed=0,
        epochs=1,
        chunk_dim=4,
        hidden_dim=5,
        log_every=0,
    )
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    vectors = []
    max_length = max(payload["lengths"])
    for cache_path in payload["cache_paths"]:
        vector = flatten_cache(load_cache_bundle(Path(cache_path))["cache"])
        vectors.append(torch.nn.functional.pad(vector, (0, max_length - vector.numel())))
    x = torch.stack(vectors)
    valid_mask = torch.zeros_like(x, dtype=torch.bool)
    for row_idx, length in enumerate(payload["lengths"]):
        valid_mask[row_idx, : int(length)] = True
    normalized = ((x - artifact["normalization_mean"]) / artifact["normalization_std"]).masked_fill(~valid_mask, 0.0)

    model = ChunkedLSTMAutoEncoder(
        input_dim=x.shape[1],
        latent_dim=artifact["latent_dim"],
        chunk_dim=artifact["chunk_dim"],
        hidden_dim=artifact["hidden_dim"],
    )
    model.load_state_dict(artifact["state_dict"])
    with torch.no_grad():
        expected = model.encode(normalized)

    assert payload["lengths"] == [16, 12]
    assert torch.allclose(payload["latents"], expected)
