from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import latent_kv.compressors as compressors
from latent_kv.cache import CacheTuple, flatten_cache, load_cache_bundle, save_cache_bundle
from latent_kv.behavior import _load_reconstructed_cache
from latent_kv.codec_validation import validate_cache_against_bundle, validate_reconstructed_artifact
from latent_kv.compressors import (
    TemporalChunkedAutoEncoder,
    TemporalLSTMAutoEncoder,
    TemporalPositionwiseAutoEncoder,
    TemporalTransformerAutoEncoder,
    _latent_pairwise_separation_loss,
    _temporal_model_class,
    _compact_reconstructions,
    _temporal_matrix,
    _temporal_token_mask,
    load_cache_matrix,
    run_compression,
)
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
            generation_token_ids=torch.tensor([[5, 6]]),
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


def _write_trajectory_run(tmp_path: Path) -> None:
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
            input_ids=torch.tensor([[1, 2, 5, 6]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            last_logits=torch.randn(1, 10),
            generation_token_ids=torch.tensor([[5, 6]]),
            generation_config={"cache_mode": "trajectory", "prompt_tokens": 2},
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
                correct=True,
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


class _FakeFrozenLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)

    def forward(self, input_ids, attention_mask, past_key_values, use_cache=True, return_dict=True, **kwargs):
        del input_ids, attention_mask, use_cache, return_dict, kwargs
        cache_score = sum((key.float().mean() + value.float().mean()) for key, value in past_key_values)
        logits = torch.stack([cache_score, -cache_score, cache_score * 0.5], dim=0).reshape(1, 1, 3)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _fake_load_model_and_tokenizer(model_id, device, local_files_only=True):
    del model_id, local_files_only
    return _FakeFrozenLM().to(device), object()


class _DtypeCheckingLM(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))

    def forward(self, input_ids, attention_mask, past_key_values, use_cache=True, return_dict=True, **kwargs):
        del input_ids, attention_mask, use_cache, return_dict, kwargs
        for key, value in past_key_values:
            assert key.dtype == self.weight.dtype
            assert value.dtype == self.weight.dtype
        logits = torch.ones((1, 1, 3), dtype=self.weight.dtype, device=self.weight.device)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_teacher_forced_replay_casts_cache_to_model_dtype():
    model = _DtypeCheckingLM(torch.float16)
    bundle = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    cache = ((torch.ones((1, 1, 2, 2), dtype=torch.float32), torch.ones((1, 1, 2, 2), dtype=torch.float32)),)

    logits = compressors._teacher_forced_generation_logits(
        bundle,
        cache,
        torch.tensor([3], dtype=torch.long),
        model,
        torch.device("cpu"),
    )

    assert len(logits) == 1
    assert logits[0].dtype == torch.float16


class _PrefixCheckingLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.seen: list[tuple[int, int, int]] = []

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        position_ids=None,
        cache_position=None,
        use_cache=True,
        return_dict=True,
    ):
        del input_ids, use_cache, return_dict
        self.seen.append(
            (
                int(past_key_values[0][0].shape[-2]),
                int(attention_mask.shape[-1]),
                int(position_ids.reshape(-1)[0].item()) if position_ids is not None else int(cache_position.reshape(-1)[0].item()),
            )
        )
        next_past = tuple(
            (
                torch.cat([key, key[..., -1:, :]], dim=-2),
                torch.cat([value, value[..., -1:, :]], dim=-2),
            )
            for key, value in past_key_values
        )
        logits = torch.ones((1, 1, 3), device=self.weight.device)
        return SimpleNamespace(logits=logits, past_key_values=next_past)


def test_teacher_forced_replay_slices_full_trajectory_cache_to_prompt_prefix():
    model = _PrefixCheckingLM()
    bundle = {
        "input_ids": torch.arange(7, dtype=torch.long).reshape(1, 7),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "generation_config": {"cache_mode": "trajectory", "prompt_tokens": 3},
    }
    cache = ((torch.ones((1, 1, 7, 2)), torch.ones((1, 1, 7, 2))),)

    compressors._teacher_forced_generation_logits(
        bundle,
        cache,
        torch.tensor([10, 11], dtype=torch.long),
        model,
        torch.device("cpu"),
    )

    assert model.seen == [(2, 3, 2), (3, 4, 3)]


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


def test_temporal_lstm_autoencoder_maps_token_sequence_to_one_point():
    model = TemporalLSTMAutoEncoder(token_dim=4, max_tokens=5, latent_dim=3, hidden_dim=6)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 3)
    assert decoded.shape == x.shape


def test_temporal_positionwise_autoencoder_maps_token_sequence_to_one_point():
    model = TemporalPositionwiseAutoEncoder(token_dim=4, max_tokens=5, latent_dim=3, hidden_dim=6)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 3)
    assert decoded.shape == x.shape


def test_temporal_chunked_autoencoder_maps_token_sequence_to_chunked_points():
    model = TemporalChunkedAutoEncoder(token_dim=4, max_tokens=5, latent_dim=3, hidden_dim=6, chunk_size=2)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 3, 3)
    assert decoded.shape == x.shape


def test_temporal_transformer_autoencoder_maps_token_sequence_to_one_point():
    model = TemporalTransformerAutoEncoder(
        token_dim=4,
        max_tokens=5,
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        latent_tokens=1,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 3)
    assert decoded.shape == x.shape


def test_temporal_transformer_one_point_can_expand_decoder_memory():
    model = TemporalTransformerAutoEncoder(
        token_dim=4,
        max_tokens=5,
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        latent_tokens=1,
        decoder_memory_tokens=4,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 3)
    assert decoded.shape == x.shape
    assert model.decoder_memory_tokens == 4


def test_latent_pairwise_separation_loss_penalizes_collapsed_codes():
    collapsed = torch.ones(3, 4)
    separated = torch.eye(3, 4)

    assert _latent_pairwise_separation_loss(collapsed, margin=0.25).item() > 0
    assert _latent_pairwise_separation_loss(separated, margin=0.25).item() == pytest.approx(0.0)
    assert _latent_pairwise_separation_loss(collapsed[:1], margin=0.25).item() == pytest.approx(0.0)


def test_temporal_transformer_autoencoder_can_use_multiple_latent_tokens():
    model = TemporalTransformerAutoEncoder(
        token_dim=4,
        max_tokens=5,
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        latent_tokens=2,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.ones((2, 5), dtype=torch.bool)

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 2, 3)
    assert decoded.shape == x.shape


def test_temporal_transformer_can_flatten_multiple_latent_tokens_to_one_point():
    model = TemporalTransformerAutoEncoder(
        token_dim=4,
        max_tokens=5,
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        latent_tokens=2,
        decoder_memory_tokens=3,
        flatten_latent_tokens=True,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.ones((2, 5), dtype=torch.bool)

    z = model.encode(x, token_mask=mask)
    decoded = model.decode(z)

    assert z.shape == (2, 6)
    assert decoded.shape == x.shape


def test_temporal_rae_compression_uses_token_time_axis(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        num_layers=1,
        weight_decay=0.02,
        log_every=1,
    )
    validation = validate_reconstructed_artifact(tmp_path, "rae_temporal")
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    assert payload["latents"].shape == (2, 4, 3)
    assert payload["codec_contract"]["input_representation"] == "temporal_full_cache_token_states"
    assert payload["codec_contract"]["structured_latent_per_cache"] is True
    assert payload["codec_contract"]["one_global_latent_vector_per_cache"] is False
    assert payload["lengths"] == [16, 12]
    assert payload["reconstructed"].shape[1] == 16
    assert artifact["codec_kind"] == "temporal_chunked_rae"
    assert artifact["temporal_codec_kind"] == "chunked"
    assert artifact["chunk_size"] == 1
    assert artifact["num_chunks"] == 4
    assert artifact["seq_len"] == 4
    assert artifact["token_dim"] == 4
    assert artifact["input_representation"] == "temporal_full_cache_token_states"
    assert artifact["latent_encoding_input"] == "masked_normalized_temporal_token_cache"
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")
    # Only check rows that have 'method' (skip startup event)
    epoch_rows = [row for row in training_rows if "method" in row]
    assert epoch_rows[0]["method"] == "rae_temporal"
    assert epoch_rows[0]["valid_tokens"] == 7
    assert epoch_rows[0]["valid_values"] == 28
    assert validation.records == 2
    assert validation.one_point_per_cache is True
    assert validation.valid_caches == 2


def test_temporal_transformer_compression_preserves_point_codec_contract(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        weight_decay=0.02,
        log_every=1,
        temporal_num_heads=2,
        temporal_latent_tokens=1,
        temporal_decoder_memory_tokens=3,
    )
    validation = validate_reconstructed_artifact(tmp_path, "rae_temporal_transformer")
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    assert validation.records == 2
    assert validation.valid_caches == 2
    assert payload["latents"].shape == (2, 3)
    assert payload["codec_contract"]["input_representation"] == "temporal_full_cache_token_states"
    assert payload["codec_contract"]["structured_latent_per_cache"] is False
    assert payload["codec_contract"]["one_global_latent_vector_per_cache"] is True
    assert artifact["codec_kind"] == "temporal_transformer_rae"
    assert artifact["temporal_codec_kind"] == "transformer"
    assert artifact["temporal_num_heads"] == 2
    assert artifact["temporal_latent_tokens"] == 1
    assert artifact["temporal_decoder_memory_tokens"] == 3


def test_temporal_transformer_flattened_latent_tokens_still_save_one_point(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        weight_decay=0.02,
        log_every=1,
        temporal_num_heads=2,
        temporal_latent_tokens=2,
        temporal_decoder_memory_tokens=3,
        temporal_flatten_latent_tokens=True,
    )
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    assert payload["latents"].shape == (2, 6)
    assert payload["codec_contract"]["structured_latent_per_cache"] is False
    assert payload["codec_contract"]["one_global_latent_vector_per_cache"] is True
    assert payload["effective_latent_dim"] == 6
    assert artifact["temporal_latent_tokens"] == 2
    assert artifact["temporal_flatten_latent_tokens"] is True
    assert artifact["effective_latent_dim"] == 6


def test_temporal_mlp_rae_compression_writes_codec_kind(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal_mlp",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        log_every=1,
    )
    artifact = torch.load(result.artifact_path, map_location="cpu")
    checkpoint = torch.load(
        tmp_path / "compressions" / "rae_temporal_mlp_checkpoints" / "rae_temporal_mlp_latest.pt",
        map_location="cpu",
    ) if (tmp_path / "compressions" / "rae_temporal_mlp_checkpoints" / "rae_temporal_mlp_latest.pt").exists() else None

    assert artifact["codec_kind"] == "temporal_positionwise_rae"
    assert artifact["temporal_codec_kind"] == "positionwise"
    if checkpoint is not None:
        assert checkpoint["temporal_codec_kind"] == "positionwise"


def test_temporal_chunked_rae_compression_writes_codec_metadata(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal_chunked",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        log_every=1,
        checkpoint_every=1,
        temporal_chunk_size=2,
    )
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")
    checkpoint = torch.load(
        tmp_path / "compressions" / "rae_temporal_chunked_checkpoints" / "rae_temporal_chunked_latest.pt",
        map_location="cpu",
    )

    assert payload["latents"].shape == (2, 2, 3)
    assert artifact["codec_kind"] == "temporal_chunked_rae"
    assert artifact["temporal_codec_kind"] == "chunked"
    assert artifact["chunk_size"] == 2
    assert artifact["num_chunks"] == 2
    assert checkpoint["temporal_codec_kind"] == "chunked"
    assert checkpoint["chunk_size"] == 2


def test_temporal_rae_rejects_deprecated_prompt_state_gradients(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)
    monkeypatch.setattr(compressors, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)

    with pytest.raises(ValueError, match="llm_loss_weight is deprecated"):
        run_compression(
            tmp_path,
            method="rae_temporal",
            latent_dim=3,
            seed=0,
            epochs=1,
            hidden_dim=5,
            llm_loss_weight=0.01,
            llm_steps=1,
            log_every=1,
        )


def test_temporal_rae_can_use_teacher_forced_generation_replay_gradients(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)
    monkeypatch.setattr(compressors, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)

    result = run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        replay_loss_weight=0.01,
        replay_loss_steps=2,
        log_every=1,
    )
    artifact = torch.load(result.artifact_path, map_location="cpu")
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")

    assert artifact["teacher_forced_generation_replay_gradients"] is True
    assert artifact["teacher_forced_generation_replay_kl_weight"] == 0.01
    assert artifact["teacher_forced_generation_replay_steps"] == 2
    epoch_rows = [row for row in training_rows if "replay_gradients" in row]
    assert epoch_rows[0]["replay_gradients"] is True
    assert "teacher_forced_generation_replay_kl" in epoch_rows[0]["loss_components"]
    assert "frozen_llm_prompt_transition_kl" not in epoch_rows[0]["loss_components"]


def test_temporal_rae_can_use_masked_cosine_reconstruction_gradients(tmp_path: Path):
    _write_run(tmp_path)

    result = run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=4,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        cosine_loss_weight=0.25,
        temporal_num_heads=2,
        temporal_latent_tokens=1,
        log_every=1,
        checkpoint_every=1,
    )
    artifact = torch.load(result.artifact_path, map_location="cpu")
    checkpoint = torch.load(
        tmp_path / "compressions" / "rae_temporal_transformer_checkpoints" / "rae_temporal_transformer_latest.pt",
        map_location="cpu",
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_transformer_training.jsonl")
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert artifact["masked_temporal_cosine_distance_weight"] == 0.25
    assert artifact["masked_temporal_cosine_distance_gradients"] is True
    assert checkpoint["cosine_loss_weight"] == 0.25
    assert epoch_rows[0]["cosine_loss_gradients"] is True
    assert epoch_rows[0]["loss_components"]["masked_temporal_cosine_distance"] >= 0
    assert epoch_rows[0]["loss"] == pytest.approx(
        epoch_rows[0]["loss_components"]["masked_temporal_reconstruction_mse"]
        + 0.25 * epoch_rows[0]["loss_components"]["masked_temporal_cosine_distance"]
    )


def test_temporal_rae_skips_nonfinite_gradient_update(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)

    def fake_clip_grad_norm_(parameters, max_norm, **kwargs):
        del parameters, max_norm, kwargs
        return torch.tensor(float("nan"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=4,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        train_batch_size=1,
        grad_clip_norm=0.25,
        temporal_num_heads=2,
        temporal_latent_tokens=1,
        log_every=1,
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_transformer_training.jsonl")
    skipped_rows = [row for row in training_rows if row.get("event") == "nonfinite_gradient_skipped"]
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert len(skipped_rows) == 2
    assert epoch_rows[0]["skipped_optimizer_steps"] == 2


def test_temporal_rae_skips_finite_weight_nonfinite_forward_batch(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)
    original_decode = TemporalTransformerAutoEncoder.decode
    calls = {"decode": 0}

    def decode_once_with_nan(self, z):
        decoded = original_decode(self, z)
        calls["decode"] += 1
        if calls["decode"] == 1:
            return torch.full_like(decoded, float("nan"))
        return decoded

    monkeypatch.setattr(TemporalTransformerAutoEncoder, "decode", decode_once_with_nan)

    run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=4,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        train_batch_size=1,
        grad_clip_norm=0.25,
        temporal_num_heads=2,
        temporal_latent_tokens=1,
        log_every=1,
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_transformer_training.jsonl")
    skipped_rows = [row for row in training_rows if row.get("event") == "nonfinite_forward_skipped"]
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert len(skipped_rows) == 1
    assert skipped_rows[0]["parameters_finite"] is True
    assert skipped_rows[0]["input_finite"] is True
    assert skipped_rows[0]["reconstruction_finite"] is False
    assert epoch_rows[0]["skipped_forward_batches"] == 1


def test_temporal_rae_reports_optimizer_parameter_corruption(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)
    captured_kwargs = []

    class CorruptingAdamW(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            captured_kwargs.append(kwargs)
            super().__init__(
                params,
                lr=kwargs["lr"],
                weight_decay=kwargs.get("weight_decay", 0.0),
            )

        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
            return result

    monkeypatch.setattr(torch.optim, "AdamW", CorruptingAdamW)

    run_compression(
        tmp_path,
        method="rae_temporal_transformer",
        latent_dim=4,
        seed=0,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        train_batch_size=1,
        grad_clip_norm=0.25,
        temporal_num_heads=2,
        temporal_latent_tokens=1,
        optimizer_eps=1e-5,
        optimizer_foreach=False,
        log_every=1,
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_transformer_training.jsonl")
    corruption_rows = [row for row in training_rows if row.get("event") == "nonfinite_parameters_after_optimizer_step"]
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert captured_kwargs[0]["eps"] == pytest.approx(1e-5)
    assert captured_kwargs[0]["foreach"] is False
    assert len(corruption_rows) == 2
    assert corruption_rows[0]["bad_parameter_count"] >= 1
    assert corruption_rows[0]["action"] == "restored_parameters_reset_optimizer_state_skipped_batch"
    assert corruption_rows[0]["restored_parameters_finite"] is True
    assert corruption_rows[0]["optimizer_state_reset"] == "all"
    assert epoch_rows[0]["skipped_optimizer_steps"] == 2


def test_temporal_rae_can_use_prompt_token_auxiliary_gradients(tmp_path: Path):
    _write_run(tmp_path)

    result = run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=8,
        prompt_loss_weight=0.01,
        prompt_loss_max_tokens=4,
        prompt_loss_hidden_dim=16,
        prompt_loss_num_layers=1,
        prompt_loss_num_heads=4,
        log_every=1,
        checkpoint_every=1,
    )
    artifact = torch.load(result.artifact_path, map_location="cpu")
    checkpoint = torch.load(
        tmp_path / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_latest.pt",
        map_location="cpu",
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert artifact["prompt_decoder_gradients"] is True
    assert artifact["prompt_token_reconstruction_ce_weight"] == 0.01
    assert artifact["prompt_loss_compact_vocab"] == 4
    assert artifact["prompt_decoder_config"]["max_prompt_tokens"] == 4
    assert "prompt_decoder_state_dict" in artifact
    assert checkpoint["prompt_loss_weight"] == 0.01
    assert "prompt_decoder_state_dict" in checkpoint
    assert epoch_rows[0]["prompt_decoder_gradients"] is True
    assert epoch_rows[0]["loss_components"]["prompt_token_reconstruction_ce"] > 0


def test_temporal_rae_resumes_prompt_decoder_and_optimizer_state(tmp_path: Path):
    _write_run(tmp_path)

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=8,
        prompt_loss_weight=0.01,
        prompt_loss_max_tokens=4,
        prompt_loss_hidden_dim=16,
        prompt_loss_num_layers=1,
        prompt_loss_num_heads=4,
        log_every=1,
        checkpoint_every=1,
    )
    checkpoint = tmp_path / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=2,
        hidden_dim=8,
        prompt_loss_weight=0.01,
        prompt_loss_max_tokens=4,
        prompt_loss_hidden_dim=16,
        prompt_loss_num_layers=1,
        prompt_loss_num_heads=4,
        log_every=1,
        checkpoint_every=1,
        resume_checkpoint_path=checkpoint,
    )
    artifact = torch.load(tmp_path / "compressions" / "rae_temporal_artifact.pt", map_location="cpu")
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")

    assert artifact["resume_epoch"] == 1
    assert artifact["resume_optimizer_state"] is True
    assert artifact["resume_prompt_decoder_state"] is True
    assert any(row.get("resume_prompt_decoder_state") is True for row in training_rows)


def test_temporal_rae_reports_sampled_replay_kl_observations(tmp_path: Path, monkeypatch):
    _write_run(tmp_path)
    monkeypatch.setattr(compressors, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        replay_loss_weight=0.01,
        replay_loss_steps=1,
        replay_loss_every_n_batches=2,
        train_batch_size=1,
        log_every=1,
    )
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")
    epoch_rows = [row for row in training_rows if row.get("method") == "rae_temporal"]

    assert epoch_rows[0]["replay_loss_every_n_batches"] == 2
    assert epoch_rows[0]["replay_loss_observations"] == 1
    assert epoch_rows[0]["loss_components"]["teacher_forced_generation_replay_kl"] > 0
    assert epoch_rows[0]["loss_components"]["teacher_forced_generation_replay_kl_effective"] * 2 == pytest.approx(
        epoch_rows[0]["loss_components"]["teacher_forced_generation_replay_kl"]
    )


def test_temporal_rae_replay_gradients_preserve_trajectory_prompt_boundary(tmp_path: Path, monkeypatch):
    _write_trajectory_run(tmp_path)
    monkeypatch.setattr(compressors, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)
    initial_cache_tokens = []

    def fake_teacher_forced_generation_logits(bundle, cache, token_ids, model, device):
        del token_ids, model, device
        initial_cache_tokens.append(compressors._teacher_forced_initial_cache_tokens(bundle, cache))
        cache_score = sum((key.float().mean() + value.float().mean()) for key, value in cache)
        return [torch.stack([cache_score, -cache_score, cache_score * 0.5], dim=0).reshape(1, 1, 3)]

    monkeypatch.setattr(compressors, "_teacher_forced_generation_logits", fake_teacher_forced_generation_logits)

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        replay_loss_weight=0.01,
        replay_loss_steps=1,
        log_every=1,
    )

    assert initial_cache_tokens
    assert set(initial_cache_tokens) == {2}


def test_temporal_rae_writes_periodic_checkpoints(tmp_path: Path):
    _write_variable_length_run(tmp_path)

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=2,
        hidden_dim=5,
        checkpoint_every=1,
        log_every=0,
    )

    checkpoint_dir = tmp_path / "compressions" / "rae_temporal_checkpoints"
    first = torch.load(checkpoint_dir / "rae_temporal_epoch_000001.pt", map_location="cpu")
    latest = torch.load(checkpoint_dir / "rae_temporal_latest.pt", map_location="cpu")

    assert first["epoch"] == 1
    assert latest["epoch"] == 2
    assert latest["latent_dim"] == 3
    assert latest["hidden_dim"] == 5
    assert latest["seq_len"] == 4
    assert latest["token_dim"] == 4
    assert "state_dict" in latest
    assert "normalization_mean" in latest
    assert not list(checkpoint_dir.glob("*.tmp"))


def test_temporal_rae_can_resume_from_checkpoint(tmp_path: Path):
    _write_variable_length_run(tmp_path)

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        checkpoint_every=1,
        log_every=1,
    )
    checkpoint = tmp_path / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_epoch_000001.pt"
    first_checkpoint = torch.load(checkpoint, map_location="cpu")

    assert first_checkpoint["optimizer_state_saved"] is True
    assert "optimizer_state_dict" in first_checkpoint
    assert first_checkpoint["optimizer_state_dict"]["state"]

    run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=2,
        hidden_dim=5,
        checkpoint_every=1,
        log_every=1,
        resume_checkpoint_path=checkpoint,
        grad_clip_norm=1.0,
        mps_empty_cache_every_batches=1,
    )

    latest = torch.load(tmp_path / "compressions" / "rae_temporal_checkpoints" / "rae_temporal_latest.pt", map_location="cpu")
    artifact = torch.load(tmp_path / "compressions" / "rae_temporal_artifact.pt", map_location="cpu")
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")

    assert latest["epoch"] == 2
    assert latest["grad_clip_norm"] == 1.0
    assert artifact["resume_epoch"] == 1
    assert artifact["resume_checkpoint_path"] == str(checkpoint)
    assert artifact["resume_optimizer_state"] is True
    assert any(row.get("resume_epoch") == 1 for row in training_rows)
    assert any(row.get("resume_optimizer_state") is True for row in training_rows)


def test_temporal_rae_can_train_in_mini_batches(tmp_path: Path):
    _write_variable_length_run(tmp_path)

    result = run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=2,
        hidden_dim=5,
        train_batch_size=1,
        log_every=1,
    )

    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")
    training_rows = read_jsonl(tmp_path / "compressions" / "rae_temporal_training.jsonl")

    assert payload["latents"].shape == (2, 4, 3)
    assert artifact["train_batch_size"] == 1
    epoch_rows = [row for row in training_rows if "train_batch_size" in row]
    assert epoch_rows[0]["train_batch_size"] == 1


def test_temporal_rae_latents_use_masked_normalized_token_states(tmp_path: Path):
    _write_variable_length_run(tmp_path)
    result = run_compression(
        tmp_path,
        method="rae_temporal",
        latent_dim=3,
        seed=0,
        epochs=1,
        hidden_dim=5,
        log_every=0,
    )
    payload = torch.load(result.latent_path, map_location="cpu")
    artifact = torch.load(result.artifact_path, map_location="cpu")

    cache_matrix = load_cache_matrix(tmp_path)
    temporal = _temporal_matrix(cache_matrix.matrix, cache_matrix.aligned_shapes)
    token_mask = _temporal_token_mask(cache_matrix.shapes, cache_matrix.aligned_shapes)
    normalized = ((temporal - artifact["normalization_mean"]) / artifact["normalization_std"]) * token_mask.unsqueeze(-1).float()

    model_class = _temporal_model_class(artifact["temporal_codec_kind"])
    model_kwargs = {
        "token_dim": artifact["token_dim"],
        "max_tokens": artifact["seq_len"],
        "latent_dim": artifact["latent_dim"],
        "hidden_dim": artifact["hidden_dim"],
        "num_layers": artifact["num_layers"],
    }
    if artifact["temporal_codec_kind"] == "chunked":
        model_kwargs["chunk_size"] = artifact["chunk_size"]
    model = model_class(
        **model_kwargs,
    )
    model.load_state_dict(artifact["state_dict"])
    with torch.no_grad():
        expected = model.encode(normalized, token_mask=token_mask)

    assert payload["lengths"] == [16, 12]
    assert payload["reconstructed"].shape[1] == 16
    assert torch.allclose(payload["latents"], expected)


def test_aligned_cache_vectors_round_trip_to_original_compact_vectors(tmp_path: Path):
    _write_variable_length_run(tmp_path)

    cache_matrix = load_cache_matrix(tmp_path)
    compact = _compact_reconstructions(cache_matrix.matrix, cache_matrix.shapes, cache_matrix.aligned_shapes)

    for idx, cache_path in enumerate(cache_matrix.paths):
        original = flatten_cache(load_cache_bundle(Path(cache_path))["cache"])
        round_tripped = compact[idx, : cache_matrix.lengths[idx]]
        assert torch.equal(round_tripped, original)
