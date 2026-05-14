from pathlib import Path

from latent_kv.experiment_config import resolve_experiment_config, resolve_model_profile


def write_fake_model_config(path: Path) -> Path:
    model_dir = path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
{
  "model_type": "fake-qwen",
  "hidden_size": 896,
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "torch_dtype": "bfloat16",
  "max_position_embeddings": 32768
}
""".strip(),
        encoding="utf-8",
    )
    return model_dir


def test_resolve_model_profile_derives_kv_dimensions(tmp_path: Path):
    model_dir = write_fake_model_config(tmp_path)
    profile = resolve_model_profile(str(model_dir), profile_name="fake")

    assert profile.layers == 24
    assert profile.attention_heads == 14
    assert profile.kv_heads == 2
    assert profile.head_dim == 64
    assert profile.grouped_query_attention is True
    assert profile.kv_values_per_token_all_layers == 6144
    assert profile.bytes_per_token_all_layers == 12288


def test_resolve_experiment_config_keeps_model_sizes_config_driven(tmp_path: Path):
    model_dir = write_fake_model_config(tmp_path)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
name: fake_experiment
model:
  profile: fake-profile
  model_id: {model_dir}
dataset:
  benchmark: gsm8k
  limit: 500
  seed: 0
prompt:
  baseline: cot
  max_new_tokens: 320
cache:
  layer_mode: upper
  storage_dtype: bfloat16
codec:
  type: rae_lstm_seq2seq
  latent_dim: 128
training:
  epochs: 2
scale:
  examples: 500
  estimated_tokens_per_cache: 100
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_experiment_config(config_path)

    assert resolved.name == "fake_experiment"
    assert resolved.model_profile.profile_name == "fake-profile"
    assert resolved.selected_layers == list(range(16, 24))
    assert resolved.selected_kv_values_per_token == 2 * 8 * 2 * 64
    assert resolved.selected_bytes_per_token == 2 * 8 * 2 * 64 * 2
    assert resolved.estimated_cache_bytes == 500 * 100 * resolved.selected_bytes_per_token
