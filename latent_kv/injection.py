"""Replay and injection helpers for saved cache bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .cache import CacheTuple, cache_to_device, load_cache_bundle, load_model_and_tokenizer


@torch.no_grad()
def greedy_continue_from_loaded_bundle(
    bundle: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 32,
    cache_override: CacheTuple | None = None,
    last_logits_override: torch.Tensor | None = None,
) -> str:
    cache = cache_override if cache_override is not None else bundle["cache"]
    input_ids = bundle.get("input_ids")
    attention_mask = bundle.get("attention_mask")
    logits = last_logits_override if last_logits_override is not None else bundle.get("last_logits")
    if input_ids is None or logits is None:
        raise ValueError("Bundle must contain input_ids and last_logits for cache replay")

    past = cache_to_device(cache, device)
    generated: list[int] = []
    prompt_len = int(input_ids.shape[-1])
    current_mask = (
        attention_mask.to(device)
        if attention_mask is not None
        else torch.ones((1, prompt_len), dtype=torch.long, device=device)
    )
    next_logits = logits.to(device)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        current_mask = torch.cat(
            [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=device)],
            dim=-1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=current_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def greedy_continue_from_bundle(
    bundle_path: Path,
    model_id: str,
    max_new_tokens: int = 32,
    device_name: str = "auto",
) -> str:
    from .cache import choose_device

    device = choose_device(device_name)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    bundle = load_cache_bundle(bundle_path)
    return greedy_continue_from_loaded_bundle(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
    )


def validate_bundle_for_injection(bundle_path: Path) -> dict[str, Any]:
    bundle = load_cache_bundle(bundle_path)
    cache = bundle["cache"]
    metadata = bundle.get("metadata", {})
    return {
        "path": str(bundle_path),
        "layers": len(cache),
        "has_input_ids": bundle.get("input_ids") is not None,
        "has_last_logits": bundle.get("last_logits") is not None,
        "selected_layers": metadata.get("selected_layers", []),
        "model_id": metadata.get("model_id"),
    }
