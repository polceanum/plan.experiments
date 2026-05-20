from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import latent_kv.compressors as compressors
from latent_kv.cache import CacheTuple, flatten_cache, load_cache_bundle, save_cache_bundle
from latent_kv.behavior import _load_reconstructed_cache
from latent_kv.codec_validation import validate_cache_against_bundle, validate_reconstructed_artifact
from latent_kv.compressors import (
    TemporalLSTMAutoEncoder,
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

    assert payload["latents"].shape == (2, 3)
    assert payload["codec_contract"]["input_representation"] == "temporal_full_cache_token_states"
    assert payload["lengths"] == [16, 12]
    assert payload["reconstructed"].shape[1] == 16
    assert artifact["codec_kind"] == "temporal_lstm_rae"
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
    assert any(row.get("resume_epoch") == 1 for row in training_rows)


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

    assert payload["latents"].shape == (2, 3)
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

    model = TemporalLSTMAutoEncoder(
        token_dim=artifact["token_dim"],
        max_tokens=artifact["seq_len"],
        latent_dim=artifact["latent_dim"],
        hidden_dim=artifact["hidden_dim"],
        num_layers=artifact["num_layers"],
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
