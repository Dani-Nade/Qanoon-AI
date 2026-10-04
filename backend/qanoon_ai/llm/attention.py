"""Memory-safe attention and CUDA failure handling for local generation."""

from __future__ import annotations

ATTENTION_IMPLEMENTATION = "sdpa_repeat_kv"


def register_attention() -> str:
    """Register SDPA that repeats KV heads instead of passing ``enable_gqa``.

    Transformers passes ``enable_gqa=True`` to SDPA for grouped-query models such
    as Qwen3. The memory-efficient kernel does not support GQA and flash attention
    is not built for Windows, so PyTorch silently falls back to the math kernel,
    which materialises the full attention matrix: ~4.6 GB per layer for a 4k-token
    prompt. Repeating the KV heads keeps the memory-efficient kernel (~30 MB).
    """
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import repeat_kv
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        groups = getattr(module, "num_key_value_groups", 1)
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if is_causal is None:
            is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal,
        )
        return output.transpose(1, 2).contiguous(), None

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, attention)
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, sdpa_mask)
    return ATTENTION_IMPLEMENTATION


def is_recoverable_oom(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.OutOfMemoryError)


def is_fatal_cuda_error(error: BaseException) -> bool:
    """A CUDA runtime error leaves the process's CUDA context unusable until restart."""
    import torch

    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(error, accelerator_error):
        return True
    return isinstance(error, RuntimeError) and "CUDA error" in str(error)
