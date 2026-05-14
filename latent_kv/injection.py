"""Replay and injection helpers for saved cache bundles."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from .cache import CacheTuple, cache_to_device, load_cache_bundle, load_model_and_tokenizer


def _forward_parameters(model: Any) -> set[str]:
    try:
        return set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return set()


def _eos_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for batch_idx in range(adjusted.shape[0]):
        for token_id in torch.unique(token_ids[batch_idx]).tolist():
            score = adjusted[batch_idx, int(token_id)]
            adjusted[batch_idx, int(token_id)] = score * penalty if score < 0 else score / penalty
    return adjusted


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
    forward_parameters = _forward_parameters(model)
    eos_ids = _eos_ids(tokenizer)
    repetition_penalty = float(getattr(getattr(model, "generation_config", None), "repetition_penalty", 1.0) or 1.0)
    token_history = input_ids.to(device)

    for _ in range(max_new_tokens):
        scored_logits = _apply_repetition_penalty(next_logits, token_history, repetition_penalty)
        next_token = torch.argmax(scored_logits, dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        token_history = torch.cat([token_history, next_token], dim=-1)
        current_mask = torch.cat(
            [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=device)],
            dim=-1,
        )
        model_kwargs = {
            "input_ids": next_token,
            "attention_mask": current_mask,
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
        }
        replay_position = torch.tensor([[prompt_len + len(generated) - 1]], dtype=torch.long, device=device)
        if "position_ids" in forward_parameters:
            model_kwargs["position_ids"] = replay_position
        if "cache_position" in forward_parameters:
            model_kwargs["cache_position"] = replay_position.reshape(-1)
        outputs = model(**model_kwargs)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        if generated[-1] in eos_ids:
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
