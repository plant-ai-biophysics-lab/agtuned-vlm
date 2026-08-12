"""Shared chat-template rendering for every vLLM generation script in this
repo (vision_baseline.py, desc_augmented_vision_v2.py, self_cot.py,
cot_yaml.py, vision_assessment/assess.py) — was duplicated identically
across all of them before this module existed.

Disabling "extended thinking" (Qwen3's <think>...</think> reasoning traces,
on by default) matters here specifically because every one of these scripts
does MULTIPLE turns per image (cot_yaml.py alone does up to 5 steps + a
classification turn, more with --per_step) — a hidden thinking block on
every turn compounds into both much slower generation and a much larger
accumulated conversation history than a model that answers directly, which
is the most likely explanation for Qwen models needing a larger
--max_model_len and running noticeably slower than Gemma on the same data.

apply_chat_template(..., enable_thinking=False) alone isn't reliable: it
raises TypeError for tokenizers whose chat template doesn't accept
enable_thinking as a kwarg, and the naive fallback (used everywhere in this
repo until now) was to just render the template plainly on that TypeError —
which leaves thinking mode ON for exactly those tokenizers, Qwen's included.
vision_assessment/assess.py had already discovered and fixed this once, for
its own text-only prompt builder only: on that TypeError, merge a literal
"/no_think" into the system turn instead (Qwen's own documented convention
for disabling thinking via plain text rather than a template kwarg). That
fix never made it into any of the multimodal (image) prompt builders in
this repo, including assess.py's own — this module centralizes it so a
future fix like this only ever needs to happen in one place.
"""
from __future__ import annotations

from PIL import Image as PILImage


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with "/no_think" merged into the first
    system turn's text (or a new leading system turn if there isn't one) —
    never mutates the caller's own messages/dicts, since callers reuse the
    same history object across turns."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = " ".join(filter(None, [_text_of(m["content"]), "/no_think"]))
            return out
    return [{"role": "system", "content": "/no_think"}] + out


def render_chat_template(tokenizer, messages: list[dict]) -> str:
    """Render a full chat `messages` list (system/user/assistant turns,
    already built by the caller — a user turn may carry an image content
    block, e.g. [{"type": "image"}, {"type": "text", "text": ...}]) via the
    tokenizer's chat template, attempting to disable extended thinking —
    see module docstring for why this matters and the three-tier fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            _inject_no_think(messages), tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_multimodal_prompt(
    tokenizer, image: PILImage.Image, question: str, system_prompt: str = ""
) -> dict:
    """Convenience wrapper for the common single-turn case (one image +
    one question, no prior history) — vision_baseline.py /
    desc_augmented_vision_v2.py / vision_assessment/assess.py's shape.
    Multi-turn callers (self_cot.py, cot_yaml.py) build their own `messages`
    list across turns and call render_chat_template directly."""
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [user_msg]
    return {"prompt": render_chat_template(tokenizer, messages), "multi_modal_data": {"image": image}}
