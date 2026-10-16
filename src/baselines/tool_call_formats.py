"""Model-family-specific tool-call parsers for the ReAct loop
(react_agent.py). extract_tool_calls in earlier versions of this module was
a single Hermes/Qwen-only regex — that silently breaks on any other model
family's own tool-calling syntax, which is NOT a standardized wire format
the way e.g. JSON itself is; each model is trained against whatever its own
chat template renders for tool_calls in PAST turns, and is expected to
reproduce that same syntax when emitting a NEW one. Swapping the served
model means swapping the parser too, not just the model id.

Two formats implemented here:

- **hermes** — Qwen3/Qwen2.5's convention (also NousResearch Hermes models,
  vLLM's own "hermes" tool parser): a real JSON object inside
  <tool_call>...</tool_call> tags — {"name": ..., "arguments": {...}}.

- **gemma4** — reverse-engineered from src/config/tool_chat_template_gemma4.jinja
  (vendored from vllm-project/vllm) rather than guessed: that template's own
  history-rendering logic for an assistant `tool_calls` turn (lines ~253-274)
  emits exactly <|tool_call>call:NAME{key:value,...}<tool_call|>, where
  string values are wrapped in a custom quote token (<|"|>...<|"|>, NOT a
  plain double quote) and keys are unquoted bare text — NOT valid JSON, so
  json.loads can't be reused here.

  The GRAMMAR is confirmed against a real cluster run — a raw generation
  with enable_thinking off came back as `call:search_agml{query:coffee leaf
  rust}`, keys/structure exactly matching this format, just missing every
  <|tool_call>/<tool_call|>/<|"|> marker. Root cause (fixed in
  react_agent.py's run_episode): those markers are REAL special tokens in
  this template's vocabulary too (same bracket family as
  <|channel>/<channel|>), and vLLM's SamplingParams had
  skip_special_tokens=True whenever enable_thinking was off, stripping them
  before this parser ever saw them — extract_tool_calls_gemma4 correctly
  matched zero calls on that raw text, because the markers it depends on
  genuinely weren't there. skip_special_tokens is now unconditionally False
  for this loop, not just when enable_thinking is on. If a rerun after that
  fix still doesn't show <|tool_call>call:... in trajectory_log.jsonl,
  something deeper is wrong (this parser's grammar, not just the special-
  token stripping) — check whether vLLM ships its own registered tool-call
  parser for this exact template (`python -c "from
  vllm.entrypoints.openai.tool_parsers import ToolParserManager as T;
  print(list(T.tool_parsers))"` on the remote machine) and prefer that over
  this hand-written one if so.

tool_call_format_for(model_id) picks between them by model name, same
"gemma"/"qwen" substring convention src/common/vlm_load.py's own
mm_processor_kwargs_for already uses.
"""
from __future__ import annotations

import json
import re

# ── hermes (Qwen3/Qwen2.5) ──────────────────────────────────────────────────

_HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls_hermes(text: str) -> list[dict]:
    """Parses Hermes/Qwen-style <tool_call>{"name": ..., "arguments": {...}}
    </tool_call> blocks out of a raw generation. Malformed JSON inside a tag
    is silently skipped (treated as if the model hadn't called anything) —
    a real generation could still contain useful reasoning prose around a
    malformed block, so this doesn't raise."""
    calls = []
    for m in _HERMES_TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if name:
            calls.append({"name": name, "arguments": obj.get("arguments") or {}})
    return calls


# ── gemma4 (tool_chat_template_gemma4.jinja) ────────────────────────────────

_GEMMA4_QUOTE = '<|"|>'
_GEMMA4_CALL_START_RE = re.compile(r"<\|tool_call>call:([^{<]+)\{")
_GEMMA4_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class _Gemma4ParseError(ValueError):
    pass


def _gemma4_find_matching_brace(s: str, open_idx: int) -> int:
    """Index of the `}` matching the `{` at s[open_idx], skipping over any
    quoted-string spans (via the custom <|"|> token) so a literal `{`/`}`
    inside a string value can't desync the depth count."""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        if s.startswith(_GEMMA4_QUOTE, i):
            end = s.find(_GEMMA4_QUOTE, i + len(_GEMMA4_QUOTE))
            if end == -1:
                raise _Gemma4ParseError("unterminated quoted string")
            i = end + len(_GEMMA4_QUOTE)
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise _Gemma4ParseError("unterminated object")


def _gemma4_parse_value(s: str, i: int) -> tuple[object, int]:
    """(value, next_index) parsing one value starting at s[i]."""
    if s.startswith(_GEMMA4_QUOTE, i):
        end = s.find(_GEMMA4_QUOTE, i + len(_GEMMA4_QUOTE))
        if end == -1:
            raise _Gemma4ParseError("unterminated quoted string")
        return s[i + len(_GEMMA4_QUOTE): end], end + len(_GEMMA4_QUOTE)
    if s.startswith("true", i):
        return True, i + 4
    if s.startswith("false", i):
        return False, i + 5
    if s.startswith("null", i):
        return None, i + 4
    if i < len(s) and s[i] == "{":
        end = _gemma4_find_matching_brace(s, i)
        return _gemma4_parse_object(s, i + 1, end), end + 1
    if i < len(s) and s[i] == "[":
        return _gemma4_parse_array(s, i)
    m = _GEMMA4_NUMBER_RE.match(s, i)
    if m:
        text = m.group(0)
        return (float(text) if "." in text else int(text)), m.end()
    raise _Gemma4ParseError(f"cannot parse value at {s[i:i + 30]!r}")


def _gemma4_parse_array(s: str, open_idx: int) -> tuple[list, int]:
    assert s[open_idx] == "["
    items: list = []
    i = open_idx + 1
    if i < len(s) and s[i] == "]":
        return items, i + 1
    while True:
        value, i = _gemma4_parse_value(s, i)
        items.append(value)
        if i < len(s) and s[i] == ",":
            i += 1
            continue
        if i < len(s) and s[i] == "]":
            return items, i + 1
        raise _Gemma4ParseError(f"expected ',' or ']' at {s[i:i + 30]!r}")


def _gemma4_parse_object(s: str, start: int, end: int) -> dict:
    """Parses the flat key:value,key:value body between (exclusive) start
    and end — keys are bare unquoted text up to the next `:`."""
    obj: dict = {}
    i = start
    while i < end:
        colon = s.find(":", i)
        if colon == -1 or colon >= end:
            raise _Gemma4ParseError(f"expected ':' after key at {s[i:i + 30]!r}")
        key = s[i:colon].strip()
        value, i = _gemma4_parse_value(s, colon + 1)
        obj[key] = value
        if i < end and s[i] == ",":
            i += 1
    return obj


def extract_tool_calls_gemma4(text: str) -> list[dict]:
    """Parses <|tool_call>call:NAME{key:value,...}<tool_call|> blocks — see
    module docstring for the grammar and its provenance. A block that fails
    to parse (grammar mismatch — e.g. the live model didn't actually follow
    the template's own convention) is skipped rather than raised, same
    silently-skip-malformed behavior as extract_tool_calls_hermes, logged
    to stderr since a persistent mismatch here means every episode will
    silently end stop_reason="no_tool_call"."""
    calls = []
    for m in _GEMMA4_CALL_START_RE.finditer(text):
        name = m.group(1).strip()
        open_brace = m.end() - 1  # the '{' the regex matched up to
        try:
            close_brace = _gemma4_find_matching_brace(text, open_brace)
            arguments = _gemma4_parse_object(text, open_brace + 1, close_brace)
        except _Gemma4ParseError as e:
            import sys
            print(f"[tool_call_formats] gemma4: failed to parse call to {name!r}: {e}", file=sys.stderr)
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls


TOOL_CALL_EXTRACTORS = {
    "hermes": extract_tool_calls_hermes,
    "gemma4": extract_tool_calls_gemma4,
}


def tool_call_format_for(model_id: str) -> str:
    """"gemma4" or "hermes" by model name — same substring convention
    src/common/vlm_load.py's mm_processor_kwargs_for already uses for
    picking per-family vision-token kwargs. Defaults to "hermes" for any
    other family (Qwen and most other tool-calling-trained open models use
    this convention)."""
    return "gemma4" if "gemma" in model_id.lower() else "hermes"
