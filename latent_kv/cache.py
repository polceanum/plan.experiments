"""KV cache capture, serialization, flattening, and replay utilities."""

from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Any

import torch

from .schemas import CacheMetadata, TaskExample, TrajectoryRecord


CacheTuple = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_past_key_values(past: Any) -> CacheTuple:
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if not isinstance(past, (tuple, list)):
        raise TypeError(f"Unsupported past_key_values type: {type(past)!r}")
    normalized = []
    for layer in past:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Each cache layer must contain key and value tensors")
        normalized.append((layer[0].detach().cpu(), layer[1].detach().cpu()))
    return tuple(normalized)


def select_layers(cache: CacheTuple, mode: str = "all") -> tuple[CacheTuple, list[int]]:
    n = len(cache)
    if mode == "all":
        indices = list(range(n))
    elif mode == "lower":
        indices = list(range(0, max(1, n // 3)))
    elif mode == "middle":
        start = n // 3
        stop = max(start + 1, (2 * n) // 3)
        indices = list(range(start, stop))
    elif mode == "upper":
        indices = list(range((2 * n) // 3, n))
    else:
        indices = [int(part) for part in mode.split(",") if part.strip()]
    return tuple(cache[idx] for idx in indices), indices


def cache_num_bytes(cache: CacheTuple) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in cache for tensor in layer)


def flatten_cache(cache: CacheTuple) -> torch.Tensor:
    parts = [tensor.reshape(-1).float().cpu() for layer in cache for tensor in layer]
    return torch.cat(parts) if parts else torch.empty(0)


def cache_shapes(cache: CacheTuple) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(tuple(k.shape), tuple(v.shape)) for k, v in cache]


def unflatten_cache(vector: torch.Tensor, shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> CacheTuple:
    vector = vector.detach().cpu().float().reshape(-1)
    offset = 0
    layers = []
    for key_shape, value_shape in shapes:
        key_size = int(torch.tensor(key_shape).prod().item())
        value_size = int(torch.tensor(value_shape).prod().item())
        key = vector[offset : offset + key_size].reshape(key_shape)
        offset += key_size
        value = vector[offset : offset + value_size].reshape(value_shape)
        offset += value_size
        layers.append((key, value))
    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} values but shapes consumed {offset}")
    return tuple(layers)


def save_cache_bundle(
    path: Path,
    cache: CacheTuple,
    metadata: CacheMetadata,
    hidden_states: Any = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    last_logits: torch.Tensor | None = None,
    generation_token_ids: torch.Tensor | None = None,
    generation_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache": cache,
            "metadata": metadata.__dict__,
            "shapes": cache_shapes(cache),
            "hidden_states": hidden_states,
            "input_ids": input_ids.detach().cpu() if input_ids is not None else None,
            "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
            "last_logits": last_logits.detach().cpu() if last_logits is not None else None,
            "generation_token_ids": generation_token_ids.detach().cpu() if generation_token_ids is not None else None,
            "generation_config": generation_config or {},
        },
        path,
    )


def load_cache_bundle(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def cache_to_device(cache: CacheTuple, device: torch.device, dtype: torch.dtype | None = None) -> CacheTuple:
    moved = []
    for key, value in cache:
        moved.append(
            (
                key.to(device=device, dtype=dtype or key.dtype),
                value.to(device=device, dtype=dtype or value.dtype),
            )
        )
    return tuple(moved)


def load_model_and_tokenizer(model_id: str, device: torch.device, local_files_only: bool = True):
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_prompt_for_model(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    use_chat_template: bool = True,
) -> dict[str, torch.Tensor]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        encoded = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def capture_prompt_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    layer_mode: str = "all",
    capture_hidden: bool = False,
    use_chat_template: bool = True,
) -> tuple[CacheTuple, list[int], int, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = encode_prompt_for_model(tokenizer, prompt, device, use_chat_template=use_chat_template)
    outputs = model(
        **encoded,
        use_cache=True,
        output_hidden_states=capture_hidden,
        return_dict=True,
    )
    cache = normalize_past_key_values(outputs.past_key_values)
    selected, selected_layers = select_layers(cache, layer_mode)
    hidden = None
    if capture_hidden and outputs.hidden_states is not None:
        hidden = tuple(t.detach().cpu() for t in outputs.hidden_states)
    return (
        selected,
        selected_layers,
        int(encoded["input_ids"].shape[-1]),
        hidden,
        encoded["input_ids"].detach().cpu(),
        encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).detach().cpu(),
        outputs.logits[:, -1, :].detach().cpu(),
    )


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, int, float]:
    set_seed(seed)
    encoded = encode_prompt_for_model(tokenizer, prompt, device, use_chat_template=True)
    start = time.perf_counter()
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = time.perf_counter() - start
    new_tokens = int(generated.shape[-1] - encoded["input_ids"].shape[-1])
    text = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text, new_tokens, latency


def collect_one(
    run_id: str,
    example: TaskExample,
    model: Any,
    tokenizer: Any,
    model_id: str,
    device: torch.device,
    cache_path: Path,
    verify_fn: Any,
    seed: int,
    max_new_tokens: int,
    layer_mode: str,
    capture_hidden: bool,
) -> TrajectoryRecord:
    output_text, generated_tokens, latency = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=example.prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    parsed, correct = verify_fn(output_text, example)
    cache, selected_layers, prompt_tokens, hidden, input_ids, attention_mask, last_logits = capture_prompt_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=example.prompt,
        device=device,
        layer_mode=layer_mode,
        capture_hidden=capture_hidden,
    )
    metadata = CacheMetadata(
        model_id=model_id,
        tokenizer_id=getattr(tokenizer, "name_or_path", model_id),
        dtype=str(cache[0][0].dtype) if cache else "unknown",
        device=str(device),
        layers=len(cache),
        selected_layers=selected_layers,
        selected_heads=None,
        token_count=prompt_tokens,
        cache_path=str(cache_path),
        benchmark=example.benchmark,
        task_id=example.task_id,
        target=example.answer,
        parsed_answer=parsed,
        correct=correct,
    )
    save_cache_bundle(
        cache_path,
        cache,
        metadata,
        hidden_states=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        last_logits=last_logits,
    )
    return TrajectoryRecord(
        run_id=run_id,
        benchmark=example.benchmark,
        task_id=example.task_id,
        model_id=model_id,
        seed=seed,
        attempt_id=0,
        prompt=example.prompt,
        target=example.answer,
        output_text=output_text,
        parsed_answer=parsed,
        correct=correct,
        retry_index=0,
        cache_path=str(cache_path),
        hidden_path=None,
        latency_s=latency,
        generated_tokens=generated_tokens,
        prompt_tokens=prompt_tokens,
        memory_bytes=cache_num_bytes(cache),
        metadata=example.metadata | {"layer_mode": layer_mode},
    )
