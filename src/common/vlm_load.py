"""Shared vLLM load/destroy for every generation script in this repo, plus
per-model-family vision-token-budget alignment.

Gemma 4 and Qwen3.5/3.6 tokenize images completely differently: Gemma has a
small, discrete, bounded "visual token budget" (70/140/280/560/1120 tokens
per image — defaulting to 280 if nothing overrides it); Qwen (a
Qwen2-VL-style dynamic-resolution processor) scales token count directly
with an image's native resolution, uncapped up to ~4096x4096px unless
bounded via max_pixels. Left at their defaults, the two model families see
meaningfully different amounts of visual detail per image — a confound for
any accuracy comparison between them, not just the max_model_len overflow
it can also cause on a large/oddly-shaped image. DEFAULT_VISION_TOKEN_BUDGET
aligns both families to the same effective per-image token count (Gemma's
own maximum tier) so a lower/upper-bound or cross-model comparison isn't
secretly also a "how much of the image did each model get to see"
comparison.

NOT VERIFIED AGAINST A LIVE MODEL — the exact runtime kwarg names
(image_seq_length for Gemma4Processor, max_pixels for
Qwen2VLImageProcessorFast) come from each model's published processor
config on HuggingFace, not from an actual vLLM run in this environment (no
GPU available while writing this). On first real use, confirm both models
actually pick this up — e.g. compare a rendered prompt's token count for
the same image before/after, or check vLLM's startup log for an
"unrecognized mm_processor_kwargs" warning — before trusting a full run's
results.
"""
from __future__ import annotations

import gc

DEFAULT_VISION_TOKEN_BUDGET = 1120  # Gemma's own maximum tier

# Qwen2VL-style processors count tokens in (merge_size * patch_size)^2-pixel
# units (patch_size=16, merge_size=2 for Qwen3.5/Qwen3.6's
# Qwen2VLImageProcessorFast) -> 32x32 px per token.
_QWEN_PIXELS_PER_TOKEN = (16 * 2) ** 2


def mm_processor_kwargs_for(model_id: str, token_budget: int) -> dict:
    """{} for any model family not specifically handled here — vLLM then
    just uses that model's own default image tokenization, same as every
    load_llm() call in this repo before this module existed."""
    name = model_id.lower()
    if "gemma" in name:
        return {"image_seq_length": token_budget}
    if "qwen" in name:
        return {"max_pixels": token_budget * _QWEN_PIXELS_PER_TOKEN}
    return {}


def _best_supported_dtype() -> str:
    """bfloat16 needs compute capability >= 8.0 (Ampere+ — A100, RTX 30xx+,
    H100...); older cards (TITAN RTX, RTX 20xx, V100 — capability 7.x)
    raise vLLM's own ValueError at engine startup ("Bfloat16 is only
    supported on GPUs with compute capability of at least 8.0") if handed
    dtype="bfloat16" unconditionally, as this used to do. Auto-detecting
    means a job scheduled onto whatever GPU the cluster happens to hand
    out never hard-fails on this — it just quietly uses float16 there
    instead of needing a per-script --dtype flag threaded through every
    caller of load_llm()."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return "bfloat16"
    except Exception:
        pass
    return "float16"


def normalize_quantization(quantization: str | None) -> tuple[str | None, str | None]:
    """(quantization, load_format) kwargs actually valid for vLLM's LLM(...).

    Every --quantize/--judge_quantize flag in this repo has offered "4bit"/
    "8bit" as choices since early on, passed straight through to
    LLM(quantization=...) with no translation — vLLM has never accepted
    those literal strings (confirmed on a real run: "Unknown quantization
    method: 4bit. Must be one of ['awq', 'fp8', ..., 'bitsandbytes', ...]").
    On-the-fly bnb quantization in vLLM needs BOTH quantization=
    "bitsandbytes" and load_format="bitsandbytes" together; vLLM's bnb
    integration doesn't expose 4-bit vs. 8-bit as an independent choice the
    way the CLI flag implies, so both "4bit" and "8bit" map to the same
    "bitsandbytes" mechanism here — kept as two CLI choices only so
    existing --quantize 4bit/8bit invocations across every script (self_cot,
    vision_baseline, desc_augmented_vision_v2, cot_yaml, verifier_cot, ...)
    keep working unchanged rather than needing every call site updated.
    "awq"/"gptq" are real vLLM quantization method names already and pass
    through as-is, load_format untouched (None = vLLM's own default)."""
    if quantization in ("4bit", "8bit"):
        return "bitsandbytes", "bitsandbytes"
    return quantization, None


def load_llm(
    model_id: str,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
    vision_token_budget: int | None = DEFAULT_VISION_TOKEN_BUDGET,
    caller_tag: str = "",
):
    """vision_token_budget=None or 0 skips mm_processor_kwargs entirely
    (falls back to each model's own default tokenization) — pass None for a
    text-only script, or 0 (via a --vision_token_budget CLI flag) to opt out
    of alignment for a deliberate cheaper/faster exploratory run."""
    from vllm import LLM

    kwargs = mm_processor_kwargs_for(model_id, vision_token_budget) if vision_token_budget else {}
    tag = f"[{caller_tag}] " if caller_tag else ""
    dtype = _best_supported_dtype()
    quantization, load_format = normalize_quantization(quantization)
    print(f"\n{tag}Loading {model_id} via vLLM (dtype={dtype})"
          + (f" (mm_processor_kwargs={kwargs})" if kwargs else ""))
    llm = LLM(
        model=model_id,
        dtype=dtype,
        quantization=quantization,
        load_format=load_format or "auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=enforce_eager,
        mm_processor_kwargs=kwargs or None,
    )
    return llm, llm.get_tokenizer()


def destroy_llm(llm) -> None:
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
