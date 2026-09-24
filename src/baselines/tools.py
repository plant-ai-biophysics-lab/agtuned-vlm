"""Tool implementations for the ReAct baseline (react_agent.py) — read/glob/
grep/write over the local filesystem, finish as the loop's own termination
signal, and retrieve/search_agml proxied straight to agml-agent's Python API
(not its MCP server). A batch eval harness has no need for MCP's real
value-add (several concurrent clients sharing one live server process); a
plain import call is fewer moving parts and has no connection-timeout/cold-
cache failure mode to work around (see agml-agent's own README on that —
uvx's cold `agml` install alone can exceed a typical 30s MCP connect
timeout).

Every tool function here takes the parsed `tool_use` block's `input: dict`
and returns either a JSON-serializable value (stringified before going back
to the model) or raises ToolError with a message the model can react to —
never lets a real exception (a missing file, a bad regex, a network error)
propagate up into the agent loop and kill the whole episode.
"""
from __future__ import annotations

import re
from pathlib import Path


class ToolError(Exception):
    """Raised by a tool implementation — the message goes back to the model
    as an `is_error` tool_result, not a crash."""


# ── read ─────────────────────────────────────────────────────────────────────

def _read(input: dict, **_) -> str:
    path = Path(input["path"])
    if not path.exists():
        raise ToolError(f"No such file: {path}")
    if path.is_dir():
        raise ToolError(f"{path} is a directory, not a file — use glob to list its contents.")
    offset = int(input.get("offset") or 0)
    limit = input.get("limit")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = offset + int(limit) if limit else None
    lines = lines[offset:end]
    return "\n".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(lines))


# ── glob ─────────────────────────────────────────────────────────────────────

_MAX_GLOB_MATCHES = 500

def _glob(input: dict, **_) -> str:
    root = Path(input.get("root") or ".")
    if not root.exists():
        raise ToolError(f"No such directory: {root}")
    matches = sorted(str(p) for p in root.glob(input["pattern"]))
    if not matches:
        return "(no matches)"
    if len(matches) > _MAX_GLOB_MATCHES:
        matches = matches[:_MAX_GLOB_MATCHES] + [f"... truncated at {_MAX_GLOB_MATCHES} matches, narrow the pattern"]
    return "\n".join(matches)


# ── grep ─────────────────────────────────────────────────────────────────────

_MAX_GREP_MATCHES = 200

def _grep(input: dict, **_) -> str:
    try:
        regex = re.compile(input["pattern"])
    except re.error as e:
        raise ToolError(f"Invalid regex {input['pattern']!r}: {e}")

    root = Path(input.get("path") or ".")
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = [p for p in root.rglob(input.get("glob") or "*") if p.is_file()]
    else:
        raise ToolError(f"No such file or directory: {root}")

    matches = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{path}:{i}: {line.strip()}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    matches.append(f"... truncated at {_MAX_GREP_MATCHES} matches, narrow --pattern/--glob")
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"


# ── write (sandboxed) ────────────────────────────────────────────────────────
# No human confirmation step — this runs unattended (batch/cluster), unlike
# Claude Code's own permission prompts. The sandbox is the only safety net:
# any path that resolves outside sandbox_root is refused rather than
# executed, so a hallucinating episode can't touch anything beyond its own
# workspace directory.

def _write(input: dict, *, sandbox_root: Path | None = None, **_) -> str:
    if sandbox_root is None:
        raise ToolError("write is disabled — no sandbox_root configured for this episode.")
    sandbox_resolved = sandbox_root.resolve()
    target = (sandbox_resolved / input["path"]).resolve()
    if target != sandbox_resolved and sandbox_resolved not in target.parents:
        raise ToolError(
            f"Refused: {input['path']!r} resolves outside this episode's sandbox "
            f"({sandbox_resolved}). Only paths under the sandbox may be written."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(input["content"], encoding="utf-8")
    return f"Wrote {len(input['content'])} bytes to {target.relative_to(sandbox_resolved)}"


# ── agml-agent: retrieve / search_agml ──────────────────────────────────────
# Imported lazily (inside the function, not at module load) so this module
# stays importable even before `uv sync` has pulled agml-agent in — the
# import error surfaces as a normal ToolError the model sees, not a crash at
# process startup.

def _retrieve(input: dict, **_) -> dict:
    try:
        from agml_agent.retrieve import retrieve
    except ImportError as e:
        raise ToolError(f"agml-agent not installed ({e}) — `uv sync` should pull it in.")
    return retrieve(
        input["class_name"],
        crop=input.get("crop", ""),
        task=input.get("task", ""),
        scientific_name=input.get("scientific_name"),
        aliases=input.get("aliases"),
        sources=input.get("sources"),
    )


def _search_agml(input: dict, **_) -> dict:
    try:
        from agml_agent.search_agml import search_agml
    except ImportError as e:
        raise ToolError(f"agml-agent not installed ({e}) — `uv sync` should pull it in.")
    return search_agml(
        query=input.get("query", ""),
        exact_name=input.get("exact_name"),
        ml_task=input.get("ml_task"),
        ag_task=input.get("ag_task"),
        limit=int(input.get("limit") or 10),
    )


# ── finish ───────────────────────────────────────────────────────────────────
# Never dispatched through TOOL_DISPATCH — react_agent.run_episode() checks
# for a `finish` tool_use block before reaching the dispatch step at all,
# and ends the episode there. Its schema still needs a real entry in
# TOOL_SCHEMAS (below) so the model can actually call it.


# OpenAI/HF function-calling schema shape ({"type": "function", "function":
# {name, description, parameters}}) — the format `apply_chat_template`'s own
# `tools=` kwarg expects (see render_chat_template's docstring,
# src/common/chat_prompt.py), and what Qwen3/Qwen2.5's chat template renders
# into a Hermes-style <tool_call>{"name":...,"arguments":...}</tool_call>
# convention the model is trained to emit. NOT the same shape Anthropic's
# Messages API uses (name/description/input_schema, flat) — this repo talks
# to the model-under-test over vLLM, not the Anthropic API, so this is the
# only schema format that matters here.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file, optionally a line range. Returns "
                            "1-indexed, tab-prefixed lines (like `cat -n`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "0-indexed line to start from."},
                    "limit": {"type": "integer", "description": "Max lines to return."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a glob pattern (e.g. '**/*.py') "
                            "under a root directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "root": {"type": "string", "description": "Defaults to '.'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex pattern, under a "
                            "file or directory. Returns matching lines as "
                            "'path:line_no: text'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex."},
                    "path": {"type": "string", "description": "File or directory to search. Defaults to '.'."},
                    "glob": {"type": "string", "description": "Restrict to files matching this glob when path is a directory, e.g. '*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file. Confined to this episode's own "
                            "sandbox directory — a path outside it is refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the sandbox root."},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Live grounding text/images for a disease, pest, "
                            "species, or quality class, from EPPO, GBIF, UC "
                            "IPM/APS, Bugwood, USDA. Returns {source_name: "
                            "[chunk, ...]}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "crop": {"type": "string"},
                    "task": {"type": "string", "enum": ["disease", "pest", "species", "quality"]},
                    "scientific_name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these source names (default: all active)."},
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agml",
            "description": "Search or look up datasets in the AgML catalog "
                            "(class labels, licenses, benchmark results). "
                            "Returns {results: [...], total_matched: N}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "exact_name": {"type": "string", "description": "Returns full metadata + benchmark results for one dataset."},
                    "ml_task": {"type": "string", "enum": ["image_classification", "object_detection", "semantic_segmentation"]},
                    "ag_task": {"type": "string", "description": "e.g. disease_classification, weed_detection, quality_classification."},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the episode and report the final answer. Call "
                            "this exactly once, when the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "The complete final answer."},
                },
                "required": ["result"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "read": _read,
    "glob": _glob,
    "grep": _grep,
    "write": _write,
    "retrieve": _retrieve,
    "search_agml": _search_agml,
}


def dispatch_tool(name: str, tool_input: dict, *, sandbox_root: Path | None = None):
    """Runs one tool call, returning a JSON-serializable result. Raises
    ToolError (never a bare exception) for any failure the model should be
    allowed to see and react to."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        raise ToolError(f"Unknown tool: {name!r}")
    return fn(tool_input, sandbox_root=sandbox_root)
