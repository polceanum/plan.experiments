from pathlib import Path

import torch

from latent_kv.cache import (
    CacheTuple,
    cache_num_bytes,
    cache_shapes,
    flatten_cache,
    load_cache_bundle,
    save_cache_bundle,
    select_layers,
    unflatten_cache,
)
from latent_kv.schemas import CacheMetadata


def fake_cache() -> CacheTuple:
    return (
        (torch.arange(12).reshape(1, 2, 3, 2), torch.arange(12, 24).reshape(1, 2, 3, 2)),
        (torch.arange(24, 36).reshape(1, 2, 3, 2), torch.arange(36, 48).reshape(1, 2, 3, 2)),
    )


def test_flatten_unflatten_round_trip():
    cache = fake_cache()
    vector = flatten_cache(cache)
    rebuilt = unflatten_cache(vector, cache_shapes(cache))
    assert len(rebuilt) == len(cache)
    for (key_a, value_a), (key_b, value_b) in zip(cache, rebuilt):
        assert torch.equal(key_a.float(), key_b)
        assert torch.equal(value_a.float(), value_b)


def test_select_upper_layer_and_count_bytes():
    cache = fake_cache()
    selected, indices = select_layers(cache, "upper")
    assert indices == [1]
    assert len(selected) == 1
    assert cache_num_bytes(cache) == sum(t.numel() * t.element_size() for layer in cache for t in layer)


def test_cache_bundle_round_trip(tmp_path: Path):
    cache = fake_cache()
    metadata = CacheMetadata(
        model_id="model",
        tokenizer_id="tok",
        dtype="torch.float32",
        device="cpu",
        layers=2,
        selected_layers=[0, 1],
        selected_heads=None,
        token_count=3,
        cache_path=str(tmp_path / "cache.pt"),
    )
    save_cache_bundle(
        tmp_path / "cache.pt",
        cache,
        metadata,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        last_logits=torch.randn(1, 10),
        generation_token_ids=torch.tensor([[4, 5]]),
        generation_config={"max_new_tokens": 2},
    )
    bundle = load_cache_bundle(tmp_path / "cache.pt")
    assert bundle["metadata"]["model_id"] == "model"
    assert bundle["input_ids"].shape == (1, 3)
    assert bundle["generation_token_ids"].tolist() == [[4, 5]]
    assert bundle["generation_config"] == {"max_new_tokens": 2}
    assert len(bundle["cache"]) == 2


def test_cache_bundle_compacts_tensor_views(tmp_path: Path):
    cache = fake_cache()
    metadata = CacheMetadata(
        model_id="model",
        tokenizer_id="tok",
        dtype="torch.float32",
        device="cpu",
        layers=2,
        selected_layers=[0, 1],
        selected_heads=None,
        token_count=3,
        cache_path=str(tmp_path / "cache.pt"),
    )
    logits_view = torch.randn(32, 128)[-1:]
    assert logits_view.untyped_storage().nbytes() > logits_view.numel() * logits_view.element_size()

    save_cache_bundle(
        tmp_path / "cache.pt",
        cache,
        metadata,
        last_logits=logits_view,
    )

    bundle = load_cache_bundle(tmp_path / "cache.pt")
    saved_logits = bundle["last_logits"]
    assert saved_logits.untyped_storage().nbytes() == saved_logits.numel() * saved_logits.element_size()
