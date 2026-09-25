"""Core ReAct loop over a locally-served vLLM model's own tool-calling
support — Thought/Action/Observation via whatever tag/syntax the model's
own chat template trains it to emit for a tool call, not a hosted API. Kept
consistent with every other generation script in this repo: same load_llm/
vLLM machinery (src/common/vlm_load.py), same render_chat_template
(src/common/chat_prompt.py, now tools-aware) — this loop is just another
multi-turn caller of it, in the same vein as self_cot.py/cot_yaml.py's own
growing-message-history pattern.

Tool-call syntax is NOT standardized across model families the way JSON
itself is — each model reproduces whatever its own chat template rendered
for past tool_calls turns, and different families disagree (Qwen/Hermes:
a real JSON object inside <tool_call> tags; Gemma-4's vendored template:
a custom non-JSON <|tool_call>call:NAME{...}<tool_call|> syntax with its
own quoting token). src/baselines/tool_call_formats.py holds one parser per
family and tool_call_format_for(model_id) picks the right one by model
name — see that module's docstring for the full detail and provenance of
each grammar, and its "NOT verified against a live model" caveat.

This module is model/task-agnostic — run_episode() takes a task string
(optionally an image) and a tool list, nothing here knows about agml
datasets or classification. The eval-harness-specific piece (heldout AgML
images, per-episode checkpointing, scoring against ground truth) is
react_baseline.py, which calls this — including where episodes are
actually persisted: this module does NOT write a checkpoint row or own a
per-episode directory. It writes at most one shared JSONL log (`log_path`,
one line per model turn, tagged by `episode_key`) — react_baseline.py opens
ONE such log per model, not one per episode, specifically to avoid a
directory-per-image explosion at scale (thousands of images -> thousands
of folders was the wrong shape, fixed here).

Image support: an optional `image` (PIL) attaches to the FIRST user turn
only (`[{"type": "image"}, {"type": "text", "text": task}]`, same content
shape build_multimodal_prompt/chat_prompt.py already uses elsewhere) and is
re-attached via `multi_modal_data` on EVERY generate() call — vLLM has no
persistent multimodal state across separate .generate() calls the way a
real multi-turn chat session would, so each turn's full re-rendered prompt
needs the same image handed back alongside it. This is the exact same
pattern self_cot.py/cot_yaml.py already use for their own multi-turn
image + growing-history loops (`{"prompt": ..., "multi_modal_data":
{"image": image}}`), not a new one invented for this loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image as PILImage

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

from src.baselines.tool_call_formats import TOOL_CALL_EXTRACTORS
from src.baselines.tools import TOOL_SCHEMAS, ToolError, dispatch_tool
from src.common.chat_prompt import render_chat_template
from src.common.io import append_jsonl, load_prompts

_REACT_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "react_prompts.yaml"
DEFAULT_SYSTEM_PROMPT = load_prompts(_REACT_PROMPTS_PATH)["default_system_prompt"]

# Tool results are truncated before going back into the conversation — a
# large retrieve()/grep() payload would otherwise eat the context budget
# for the rest of the episode (each turn re-renders the FULL message
# history, same growing-prompt pattern as self_cot.py/cot_yaml.py). The
# model can always narrow its query and re-call.
_MAX_TOOL_RESULT_CHARS = 20_000


@dataclass
class EpisodeResult:
    result: str | None
    stop_reason: str          # "finished" | "no_tool_call" | "max_iterations"
    n_steps: int
    tools_used: list[str] = field(default_factory=list)  # tool names called, in order, "finish" excluded
    trajectory: list[dict] = field(default_factory=list)  # raw message history, for logging/debugging


def run_episode(
    llm,
    tokenizer,
    task: str,
    *,
    image: PILImage.Image | None = None,
    episode_key: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: list[dict] = TOOL_SCHEMAS,
    tool_call_format: str = "hermes",
    sandbox_root: Path | None = None,
    max_iterations: int = 20,
    gen_max_tokens: int = 1024,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    chat_template: str | None = None,
    log_path: Path | None = None,
    verbose: bool = True,
) -> EpisodeResult:
    """Runs one ReAct episode to completion (or exhaustion) against an
    already-loaded vLLM `llm`/`tokenizer` (src/common/vlm_load.py's
    load_llm) — loading/destroying the model is the caller's job (same
    convention as every other run_model() in this repo), so a harness can
    load once and call run_episode() per task/image without reloading.

    `verbose` (default True): print one short line per turn (step index +
    what happened — tool call(s), no tool call, or finish) as the episode
    runs, not just a summary after it's done. A slow episode (generation
    can genuinely take minutes per turn) would otherwise leave the
    terminal silent with no way to tell it's still working vs. hung — this
    is cheap stdout noise, not a real cost, so it defaults on. Set False
    for a quieter batch run once you trust the loop and just want
    react_baseline.py's own per-episode summary lines.

    `image`, if given, attaches to the FIRST user turn only (the task
    itself) and is re-sent via multi_modal_data on every generate() call
    thereafter — see module docstring. None (default) keeps this a
    plain-text episode, unchanged from before image support existed.

    `episode_key`: an opaque identifier for this episode (e.g. a
    dataset/class/image-name triple, or an ad-hoc key for a manual test) —
    stamped onto every log line this call writes so a SHARED `log_path`
    across many episodes can still be split back apart per episode
    afterward. Purely a label; nothing here parses or validates it.

    `tool_call_format`: "hermes" (Qwen3/Qwen2.5) or "gemma4"
    (src/config/tool_chat_template_gemma4.jinja) — picks which of
    tool_call_formats.py's parsers reads the model's raw generation. Get
    this wrong for the served model and every episode silently ends
    stop_reason="no_tool_call" on turn 1 (no exception — there's no way to
    distinguish "wrong parser" from "model chose not to call a tool" from
    parsed output alone). react_baseline.py picks this automatically via
    tool_call_format_for(model_id); pass it explicitly here only if calling
    run_episode() directly.

    `sandbox_root`, if given, is passed through to the `write` tool as its
    confinement root — every write call is refused if it'd resolve outside
    it (see tools.py's module docstring). NOT created here — `write`'s own
    implementation creates it (and any subdirectory) lazily, on first
    actual write, via target.parent.mkdir(...). An episode that never calls
    `write` (the common case) creates no directory at all; the caller
    decides what `sandbox_root` even points at (react_baseline.py scopes it
    per-episode under one shared workspace/ directory, not a directory
    named after the episode itself, so only episodes that actually write
    something leave a trace on disk).

    `log_path`, if given, gets one JSON line per model turn
    ({"episode_key": episode_key, "step": i, "raw_text": ...}) via
    append_jsonl — pass the SAME path across many run_episode() calls (one
    shared file per model, same convention as judge_log.jsonl/
    verifier_log.jsonl elsewhere in this repo) rather than one log file per
    episode; filter by episode_key afterward if you need one episode's
    turns in isolation.

    Each turn re-renders the FULL message history through the chat
    template (system + user task + every past assistant/tool turn) — the
    same multi-turn pattern self_cot.py/cot_yaml.py already use, not a new
    one. Multiple tool calls in a single turn are all executed and their
    results folded into ONE "tool" role message (rather than one message
    per call correlated by an id) — neither the Hermes nor the gemma4
    convention carries a per-call id to correlate against, only call order,
    so a single combined observation avoids relying on an id that doesn't
    exist."""
    if SamplingParams is None:
        raise RuntimeError("vllm not installed — this only runs on the sys_platform == 'linux' GPU environment.")
    extract_tool_calls = TOOL_CALL_EXTRACTORS.get(tool_call_format)
    if extract_tool_calls is None:
        raise ValueError(f"Unknown tool_call_format {tool_call_format!r} — must be one of {sorted(TOOL_CALL_EXTRACTORS)}.")

    first_user_content = [{"type": "image"}, {"type": "text", "text": task}] if image is not None else task
    messages: list[dict] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + [{"role": "user", "content": first_user_content}]
    tools_used: list[str] = []

    # skip_special_tokens=False, UNCONDITIONALLY — not just when
    # enable_thinking is True (an earlier version of this line was
    # `skip_special_tokens=not enable_thinking`, matching vision_baseline.py/
    # verifier_cot.py's own SamplingParams, which only ever need this for
    # Gemma-4's thinking channel). This loop ALSO needs it for
    # tool_call_formats.py's "gemma4" parser: <|tool_call>/<tool_call|> and
    # the <|"|> quote token are REAL special tokens in the vendored
    # template's vocabulary too, same bracket family as <|channel>/
    # <channel|>, not literal text — confirmed from a real cluster run with
    # enable_thinking OFF (the default): raw_text came back as
    # `call:search_agml{query:coffee leaf rust}`, missing every one of those
    # markers, so extract_tool_calls_gemma4 matched nothing even though the
    # model clearly intended to call a tool. skip_special_tokens=True was
    # stripping them regardless of enable_thinking. Harmless for "hermes"
    # (Qwen's <tool_call> tags are plain text, not special tokens — already
    # established elsewhere in this repo for its <think> tags too), so
    # there's no reason to gate this on tool_call_format either — always off.
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=gen_max_tokens, skip_special_tokens=False,
    )

    for step in range(max_iterations):
        if verbose:
            # vLLM's own generate() shows a tqdm progress bar per call too,
            # but that renders as carriage-return updates — useless in a
            # redirected/SLURM log file. This is a plain line specifically
            # so a log file (not just an interactive terminal) still shows
            # forward progress during the slow part (generation itself).
            print(f"    [react] step {step}: generating...", flush=True)
        prompt = render_chat_template(
            tokenizer, messages, enable_thinking=enable_thinking, chat_template=chat_template, tools=tools,
        )
        # Same shape self_cot.py/cot_yaml.py already pass for their own
        # multi-turn image + growing-history loops — vLLM's .generate() has
        # no persistent multimodal state across separate calls, so `image`
        # is re-attached every turn, not just the first.
        gen_input = {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
        outputs = llm.generate([gen_input], sampling_params)
        raw_text = outputs[0].outputs[0].text
        messages.append({"role": "assistant", "content": raw_text})
        if log_path is not None:
            append_jsonl(log_path, {"episode_key": episode_key, "step": step, "raw_text": raw_text})

        calls = extract_tool_calls(raw_text)
        if not calls:
            if verbose:
                print(f"    [react] step {step}: no tool call found (stop_reason=no_tool_call)")
            # No tool call found — the loop has nothing to observe next.
            # Treat as an incomplete episode rather than silently returning
            # whatever text the model produced (finish is the only
            # sanctioned way to end with a real result).
            return EpisodeResult(result=None, stop_reason="no_tool_call", n_steps=step + 1, tools_used=tools_used, trajectory=messages)

        finish_call = next((c for c in calls if c["name"] == "finish"), None)
        if finish_call is not None:
            if verbose:
                print(f"    [react] step {step}: finish")
            return EpisodeResult(
                result=finish_call["arguments"].get("result"), stop_reason="finished",
                n_steps=step + 1, tools_used=tools_used, trajectory=messages,
            )

        if verbose:
            print(f"    [react] step {step}: {', '.join(c['name'] for c in calls)}")
        observations = []
        for call in calls:
            tools_used.append(call["name"])
            try:
                output = dispatch_tool(call["name"], call["arguments"], sandbox_root=sandbox_root)
                content = output if isinstance(output, str) else json.dumps(output)
                content = content[:_MAX_TOOL_RESULT_CHARS]
            except ToolError as e:
                content = f"ERROR: {e}"
            observations.append({"name": call["name"], "content": content})

        if tool_call_format == "gemma4":
            # gemma4's OWN template renders a plain role="tool" message as
            # NOTHING — its per-message loop is wrapped in `{%- if
            # message['role'] != 'tool' -%}`, so that role is skipped
            # entirely; tool responses only render via a "Legacy:
            # tool_responses embedded on the assistant message (Google/
            # Gemma native)" path, i.e. set directly on the SAME assistant
            # message dict just appended above — confirmed against a real
            # run: without this, the model never saw its own tool results
            # and kept re-issuing the identical call turn after turn
            # (verified: 20 consecutive identical `search_agml{query:
            # "coffee leaf rust"}` calls in one real episode) until
            # max_iterations. format_tool_response_block (the template's
            # own renderer for this field) accepts a plain string per
            # response, wrapped in <|"|> quotes — no need to preserve dict
            # structure here.
            messages[-1]["tool_responses"] = [
                {"name": o["name"], "response": o["content"]} for o in observations
            ]
        else:
            # hermes (Qwen/Hermes convention): a plain role="tool" message
            # IS rendered by that family's own chat template, no special-
            # cased attachment needed.
            combined = "\n\n".join(f"[{o['name']}]\n{o['content']}" for o in observations)
            messages.append({"role": "tool", "content": combined})

    return EpisodeResult(result=None, stop_reason="max_iterations", n_steps=max_iterations, tools_used=tools_used, trajectory=messages)
