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


def render_chat_template(
    tokenizer, messages: list[dict], enable_thinking: bool = False, chat_template: str | None = None,
    tools: list[dict] | None = None,
) -> str:
    """Render a full chat `messages` list (system/user/assistant turns,
    already built by the caller — a user turn may carry an image content
    block, e.g. [{"type": "image"}, {"type": "text", "text": ...}]) via the
    tokenizer's chat template, attempting to disable extended thinking by
    default — see module docstring for why this matters and the
    three-tier fallback. enable_thinking=True skips both the
    enable_thinking=False kwarg and the "/no_think" injection, letting
    whatever the tokenizer's chat template does by default (thinking ON
    for Qwen3) through unchanged — for callers deliberately A/B testing
    thinking vs. no-thinking (e.g. verifier_cot.py's --enable_thinking).

    chat_template (default None — use the tokenizer's own bundled
    template): raw Jinja text passed straight through as
    apply_chat_template's own `chat_template` override kwarg (standard HF
    tokenizer feature). Exists because some models' bundled templates have
    NO enable_thinking support at all — confirmed for Gemma-4, whose stock
    template silently ignores enable_thinking (TypeError caught by the
    fallback below, same as any tokenizer without the kwarg) and needs a
    separate template file (e.g. vLLM's tool_chat_template_gemma4.jinja,
    vendored at src/config/tool_chat_template_gemma4.jinja) that defines
    its own enable_thinking-gated <|channel>thought...<channel|> emission.
    When given, skips straight to the override render (the enable_thinking
    kwarg is still forwarded and respected by that custom template) rather
    than going through the Qwen-oriented TypeError/no_think fallback chain,
    which doesn't apply once a custom template is already in play.

    tools (default None): forwarded straight through as apply_chat_template's
    own `tools` kwarg (standard HF function-calling feature — a list of
    OpenAI-style {"type": "function", "function": {name, description,
    parameters}} schemas). Renders each tool's definition into the prompt so
    a tool-calling-trained model (e.g. Qwen3/Qwen2.5's Hermes-style
    <tool_call>{"name":...,"arguments":...}</tool_call> convention) knows
    what it can call. A model/template with no tools support raises
    TypeError same as an unsupported enable_thinking kwarg would — not
    caught here, since silently dropping `tools` would leave the ReAct loop
    (src/baselines/react_agent.py) generating against a model that was
    never told its own tool list exists. Confirm the target model's chat
    template actually supports `tools` before relying on this (Qwen3/
    Qwen2.5 do; Gemma-4's bundled template does not — same gap as its
    missing enable_thinking support, see module docstring above)."""
    if chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template=chat_template, enable_thinking=enable_thinking, tools=tools,
        )
    if enable_thinking:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=tools,
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            _inject_no_think(messages), tokenize=False, add_generation_prompt=True, tools=tools,
        )
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)


def build_multimodal_prompt(
    tokenizer, image: PILImage.Image, question: str, system_prompt: str = "",
    enable_thinking: bool = False, chat_template: str | None = None,
) -> dict:
    """Convenience wrapper for the common single-turn case (one image +
    one question, no prior history) — vision_baseline.py /
    desc_augmented_vision_v2.py / vision_assessment/assess.py's shape.
    Multi-turn callers (self_cot.py, cot_yaml.py) build their own `messages`
    list across turns and call render_chat_template directly.

    enable_thinking (default False, matches render_chat_template's own
    default — thinking suppressed): forwarded straight through. Every
    existing caller leaves this False and gets the exact original
    behavior.

    chat_template (default None): forwarded straight through to
    render_chat_template — see its docstring."""
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [user_msg]
    return {
        "prompt": render_chat_template(
            tokenizer, messages, enable_thinking=enable_thinking, chat_template=chat_template,
        ),
        "multi_modal_data": {"image": image},
    }
