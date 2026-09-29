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
one or two lines per model turn — "generation" always, "tool_response"
only on a turn with tool calls — tagged by `episode_key`, see
run_episode's own docstring) — react_baseline.py opens ONE such log per
model, not one per episode, specifically to avoid a directory-per-image
explosion at scale (thousands of images -> thousands
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
from src.common.progress import ProgressBar

_REACT_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "react_prompts.yaml"
_REACT_PROMPTS = load_prompts(_REACT_PROMPTS_PATH)
DEFAULT_SYSTEM_PROMPT = _REACT_PROMPTS["default_system_prompt"]
NUDGE_PROMPT = _REACT_PROMPTS["nudge_prompt"]
WRAP_UP_PROMPT = _REACT_PROMPTS["wrap_up_prompt"]
AUTO_VERIFY_PROMPT = _REACT_PROMPTS["auto_verify_prompt"]

# Tool results are truncated before going back into the conversation — a
# large retrieve()/grep() payload would otherwise eat the context budget
# for the rest of the episode (each turn re-renders the FULL message
# history, same growing-prompt pattern as self_cot.py/cot_yaml.py). The
# model can always narrow its query and re-call.
_MAX_TOOL_RESULT_CHARS = 20_000

# Fraction of max_model_len at which the wrap-up nudge fires (once per
# episode) — before the hard context_overflow cutoff at 100%, not instead
# of it: a nudge can itself fail (model ignores it, or there just isn't
# room left even for a short reply), so the proactive overflow check still
# runs every turn regardless of whether this already fired.
_WRAP_UP_THRESHOLD = 0.8


@dataclass
class EpisodeResult:
    result: str | None
    stop_reason: str          # "finished" | "no_tool_call" | "max_iterations" | "context_overflow" | "error"
    n_steps: int
    summary: str | None = None        # finish()'s own "what I investigated" — see tools.py's finish schema
    justification: str | None = None  # finish()'s own "why this answer" — separate from `result` itself
    tools_used: list[str] = field(default_factory=list)  # tool names called, in order, "finish" excluded
    nudges_used: int = 0      # see run_episode's `max_nudges` — how many of them this episode needed
    wrap_up_nudged: bool = False  # did the 80%-of-max_model_len wrap-up nudge fire? see _WRAP_UP_THRESHOLD
    auto_verified: bool = False  # did the automatic finish-time verification gate fire? False if the
                                 # model called `verify` itself before finishing (no auto-gate needed),
                                 # or if the episode never reached a finish attempt at all.
    error: str | None = None  # set only for stop_reason in {"context_overflow", "error"} — see run_episode
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
    max_nudges: int = 1,
    gen_max_tokens: int = 1024,
    max_model_len: int | None = None,
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

    `max_model_len` (default None = no check): the served model's own
    context window (same value passed to load_llm()). Each turn re-renders
    the FULL growing message history, and _MAX_TOOL_RESULT_CHARS only caps
    a SINGLE tool result's size — nothing caps the CUMULATIVE growth across
    many turns, so a real episode making several retrieve()/search_agml()
    calls can genuinely exceed the context window. Confirmed on a real
    cluster run: vLLM raised VLLMValidationError (a hard crash, not a
    graceful truncation) when a prompt hit 24577 tokens against a 24576
    max_model_len — with no try/except anywhere, that took the ENTIRE
    batch process down, not just this one episode, and would keep crashing
    identically on every retry (already-checkpointed episodes before it
    are safe, but nothing after it in the batch ever runs). When given,
    this is checked BEFORE every generate() call (prompt token count +
    gen_max_tokens + a 1200-token margin for vision-token expansion an
    `image`'s plain-text tokenization can't account for, against
    max_model_len) and ends the episode with stop_reason="context_overflow"
    instead of ever sending the doomed request — `error` on the returned
    EpisodeResult has the specific numbers. A real (not just anticipated)
    exception from generate() itself is also now caught and returned as
    stop_reason="error" rather than propagating, as a second line of
    defense for whatever this estimate doesn't anticipate. Also gates a
    separate, earlier warning: at _WRAP_UP_THRESHOLD (80%) of max_model_len,
    WRAP_UP_PROMPT fires once as a "user" turn telling the model to call
    `finish` now rather than keep gathering — recovers an answer that would
    otherwise be lost entirely once the hard 100% cutoff above hits (that
    ends the episode with NO result at all). `wrap_up_nudged` on the
    returned EpisodeResult records whether this fired.

    `max_nudges` (default 1): a turn with no tool call at all doesn't
    immediately end the episode — NUDGE_PROMPT (react_prompts.yaml) is
    injected as one "user" turn and the model gets another try, up to this
    many times, before actually returning stop_reason="no_tool_call". Real,
    observed failure mode this recovers from: even with an explicit "call
    finish, don't just answer in prose" system-prompt instruction, the
    model sometimes writes a complete, well-cited answer as plain text and
    never calls `finish` — discarding that answer outright (the original,
    nudge-free behavior) throws away a result that was right there. Costs
    one extra turn (counts against max_iterations) only on a genuine miss;
    0 disables nudging entirely, back to the original all-or-nothing check.

    Verification: `verify` is a real tool (tools.py) the model can call at
    ANY point, not just before finishing — see its own docstring for why
    it makes no network/LLM call of its own (a deliberate "reflection
    point", not a second verifier model). If `finish` is called WITHOUT
    the model ever having called `verify` first, this loop auto-injects
    AUTO_VERIFY_PROMPT as one "user" turn (same shape as the nudges above)
    instead of accepting that finish immediately — a MANDATORY gate, not
    left to the model's own discretion, since this repo has repeatedly
    found the model isn't a reliable judge of when to double-check itself.
    Fires at most once per episode: the second `finish` call (same or
    revised answer) is always accepted, regardless of whether the model
    called `verify` again in between. `auto_verified` on the returned
    EpisodeResult records whether this gate fired.

    `finish` itself now takes 3 arguments (tools.py's own schema):
    `result` (the answer, unchanged), `summary` (what was investigated —
    which sources/tools, what was found), `justification` (why this
    specific answer — how the gathered evidence supports it). Populated on
    the returned EpisodeResult as `summary`/`justification`, separate from
    `result` itself — a structured audit trail per episode, not just
    whatever prose the model happened to put in `result`.

    `verbose` (default True): show a live ProgressBar (src/common/
    progress.py — same one vision_baseline.py etc. use) tracking steps used
    against max_iterations, its description updated each turn with what
    just happened (generating / tool name(s) / finish / nudge). A slow
    episode (generation can genuinely take minutes per turn) would
    otherwise leave the terminal silent with no way to tell it's still
    working vs. hung — this is cheap stdout noise, not a real cost, so it
    defaults on. Set False for a quieter batch run once you trust the loop
    and just want react_baseline.py's own per-episode summary lines.

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

    `log_path`, if given, gets JSON lines via append_jsonl — pass the SAME
    path across many run_episode() calls (one shared file per model, same
    convention as judge_log.jsonl/verifier_log.jsonl elsewhere in this
    repo) rather than one log file per episode; filter by episode_key
    afterward if you need one episode's turns in isolation. Two line
    "type"s, both tagged with the same episode_key/step so they can be
    paired back up: {"type": "generation", "raw_text": ...} — the model's
    own output for that turn (one per turn, always) — and, only on a turn
    that made tool call(s), {"type": "tool_response", "observations":
    [{"name", "content"}, ...]} — what each tool actually returned. The
    second type exists because a tool's response is computed by OUR code,
    not generated by the model, so it's never part of raw_text and would
    otherwise never be recorded anywhere (real gap, found by actually
    reading a trajectory closely — a tool call with no way to see what it
    returned isn't very useful for debugging).

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
    nudges_used = 0
    wrap_up_nudged = False
    has_verified = False    # True once the model has called `verify` at least once
    auto_verified = False   # True once the automatic finish-time verification gate has fired

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

    # ProgressBar (not vLLM's own per-call tqdm bar — that's carriage-
    # return-driven and gets corrupted by any other print interleaved with
    # it, and produces useless output in a redirected/SLURM log file
    # anyway; this one throttles to whole-percent-point LINES there
    # instead). total=max_iterations since that's this episode's real step
    # budget; desc updated each turn with what just happened. Wrapped in
    # try/finally purely to commit the bar's trailing newline on a real
    # terminal (finish()'s own completed=True would force n up to total,
    # misrepresenting an episode that ended early — not what we want here).
    pbar = ProgressBar(total=max_iterations, desc=episode_key or "episode") if verbose else None
    try:
        for step in range(max_iterations):
            prompt = render_chat_template(
                tokenizer, messages, enable_thinking=enable_thinking, chat_template=chat_template, tools=tools,
            )
            if max_model_len is not None:
                prompt_tokens = len(tokenizer(prompt)["input_ids"])
                # +1200: a fixed margin for vision-token expansion a plain
                # text tokenization of `prompt` can't account for (the
                # rendered string carries one small image placeholder;
                # vLLM's own processor expands that to vision_token_budget
                # tokens, typically ~1120 — see vlm_load.py's
                # DEFAULT_VISION_TOKEN_BUDGET — only once generation
                # actually runs). Approximate, not exact, deliberately on
                # the generous side.
                image_margin = 1200 if image is not None else 0
                if prompt_tokens + gen_max_tokens + image_margin > max_model_len:
                    msg = (f"prompt has ~{prompt_tokens} tokens (+{gen_max_tokens} requested output"
                           f"{f', +{image_margin} image margin' if image_margin else ''}) > "
                           f"max_model_len={max_model_len}")
                    if pbar is not None:
                        pbar.desc = f"{episode_key + ' ' if episode_key else ''}context overflow"
                        pbar.update(1)
                    return EpisodeResult(
                        result=None, stop_reason="context_overflow", n_steps=step, error=msg,
                        tools_used=tools_used, nudges_used=nudges_used, wrap_up_nudged=wrap_up_nudged,
                        auto_verified=auto_verified, trajectory=messages,
                    )
                if not wrap_up_nudged and prompt_tokens >= _WRAP_UP_THRESHOLD * max_model_len:
                    # One-time warning BEFORE the hard cutoff above — losing
                    # the episode entirely to context_overflow (no result,
                    # even though real progress was made) is worse than
                    # asking the model to wrap up a turn early. Consumes one
                    # `step` (a real turn, counted against max_iterations)
                    # rather than a free pass, same convention as the
                    # no-tool-call nudge above.
                    wrap_up_nudged = True
                    if pbar is not None:
                        pbar.desc = f"{episode_key + ' ' if episode_key else ''}wrap-up nudge"
                        pbar.update(1)
                    messages.append({"role": "user", "content": WRAP_UP_PROMPT})
                    continue

            # Same shape self_cot.py/cot_yaml.py already pass for their own
            # multi-turn image + growing-history loops — vLLM's .generate()
            # has no persistent multimodal state across separate calls, so
            # `image` is re-attached every turn, not just the first.
            gen_input = {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
            try:
                outputs = llm.generate([gen_input], sampling_params)
            except Exception as e:
                # Second line of defense — whatever the token-count estimate
                # above didn't anticipate. One bad episode must never take
                # the whole batch down with it (confirmed as a real,
                # reproducible risk, not hypothetical — see max_model_len's
                # own docstring).
                if pbar is not None:
                    pbar.desc = f"{episode_key + ' ' if episode_key else ''}error"
                    pbar.update(1)
                return EpisodeResult(
                    result=None, stop_reason="error", n_steps=step, error=str(e),
                    tools_used=tools_used, nudges_used=nudges_used, wrap_up_nudged=wrap_up_nudged,
                    auto_verified=auto_verified, trajectory=messages,
                )
            raw_text = outputs[0].outputs[0].text
            messages.append({"role": "assistant", "content": raw_text})
            if log_path is not None:
                append_jsonl(log_path, {
                    "episode_key": episode_key, "step": step, "type": "generation", "raw_text": raw_text,
                })

            calls = extract_tool_calls(raw_text)
            if not calls:
                if nudges_used < max_nudges:
                    # Real, observed failure mode (not hypothetical): even
                    # with an explicit "call finish, don't just answer in
                    # prose" system-prompt instruction, the model sometimes
                    # writes a complete, well-cited answer as plain text and
                    # never calls `finish`. Give it one more try with a
                    # reminder rather than discarding an answer that was
                    # right there — see NUDGE_PROMPT / max_nudges docstring.
                    nudges_used += 1
                    if pbar is not None:
                        pbar.desc = f"{episode_key + ' ' if episode_key else ''}nudge ({nudges_used}/{max_nudges})"
                        pbar.update(1)
                    messages.append({"role": "user", "content": NUDGE_PROMPT})
                    continue
                if pbar is not None:
                    pbar.desc = f"{episode_key + ' ' if episode_key else ''}no tool call"
                    pbar.update(1)
                # Still nothing after every nudge — the loop has nothing to
                # observe next. Treat as an incomplete episode rather than
                # silently returning whatever text the model produced
                # (finish is the only sanctioned way to end with a real
                # result).
                return EpisodeResult(result=None, stop_reason="no_tool_call", n_steps=step + 1, tools_used=tools_used, nudges_used=nudges_used, wrap_up_nudged=wrap_up_nudged, auto_verified=auto_verified, trajectory=messages)

            finish_call = next((c for c in calls if c["name"] == "finish"), None)
            if finish_call is not None:
                if not has_verified and not auto_verified:
                    # Mandatory gate: `finish` without the model ever having
                    # called `verify` itself doesn't end the episode — one
                    # auto-injected reflection first, same "inject a user
                    # turn and continue" shape as the nudges above. Fires at
                    # most once (auto_verified latches True), so a second
                    # finish call is always accepted regardless of what the
                    # model does with the reflection.
                    auto_verified = True
                    if pbar is not None:
                        pbar.desc = f"{episode_key + ' ' if episode_key else ''}auto-verify"
                        pbar.update(1)
                    messages.append({"role": "user", "content": AUTO_VERIFY_PROMPT})
                    continue
                if pbar is not None:
                    pbar.desc = f"{episode_key + ' ' if episode_key else ''}finish"
                    pbar.update(1)
                return EpisodeResult(
                    result=finish_call["arguments"].get("result"), stop_reason="finished",
                    n_steps=step + 1, tools_used=tools_used, nudges_used=nudges_used,
                    summary=finish_call["arguments"].get("summary"),
                    justification=finish_call["arguments"].get("justification"),
                    wrap_up_nudged=wrap_up_nudged, auto_verified=auto_verified, trajectory=messages,
                )

            if pbar is not None:
                pbar.desc = f"{episode_key + ' ' if episode_key else ''}{', '.join(c['name'] for c in calls)}"
                pbar.update(1)
            observations = []
            for call in calls:
                tools_used.append(call["name"])
                if call["name"] == "verify":
                    has_verified = True
                try:
                    output = dispatch_tool(call["name"], call["arguments"], sandbox_root=sandbox_root)
                    content = output if isinstance(output, str) else json.dumps(output)
                    content = content[:_MAX_TOOL_RESULT_CHARS]
                except ToolError as e:
                    content = f"ERROR: {e}"
                observations.append({"name": call["name"], "content": content})

            if log_path is not None:
                # Separate log line, same episode_key/step as the generation
                # line above (distinguished by "type") — the tool's own
                # response is computed by OUR code, not generated by the
                # model, so it was never part of raw_text and never got
                # logged anywhere before this. Without it, the viewer (or
                # anyone reading trajectory_log.jsonl by hand) could see
                # every call the model made but never what any of them
                # actually returned — a real gap, not a hypothetical one.
                append_jsonl(log_path, {
                    "episode_key": episode_key, "step": step, "type": "tool_response",
                    "observations": observations,
                })

            if tool_call_format == "gemma4":
                # gemma4's OWN template renders a plain role="tool" message
                # as NOTHING — its per-message loop is wrapped in `{%- if
                # message['role'] != 'tool' -%}`, so that role is skipped
                # entirely; tool responses only render via a "Legacy:
                # tool_responses embedded on the assistant message (Google/
                # Gemma native)" path, i.e. set directly on the SAME
                # assistant message dict just appended above — confirmed
                # against a real run: without this, the model never saw its
                # own tool results and kept re-issuing the identical call
                # turn after turn (verified: 20 consecutive identical
                # `search_agml{query: "coffee leaf rust"}` calls in one real
                # episode) until max_iterations. format_tool_response_block
                # (the template's own renderer for this field) accepts a
                # plain string per response, wrapped in <|"|> quotes — no
                # need to preserve dict structure here.
                messages[-1]["tool_responses"] = [
                    {"name": o["name"], "response": o["content"]} for o in observations
                ]
            else:
                # hermes (Qwen/Hermes convention): a plain role="tool"
                # message IS rendered by that family's own chat template,
                # no special-cased attachment needed.
                combined = "\n\n".join(f"[{o['name']}]\n{o['content']}" for o in observations)
                messages.append({"role": "tool", "content": combined})

        return EpisodeResult(result=None, stop_reason="max_iterations", n_steps=max_iterations, tools_used=tools_used, nudges_used=nudges_used, wrap_up_nudged=wrap_up_nudged, auto_verified=auto_verified, trajectory=messages)
    finally:
        if pbar is not None and pbar.is_tty:
            pbar.stream.write("\n")
            pbar.stream.flush()
