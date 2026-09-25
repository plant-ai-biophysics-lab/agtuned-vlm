"""ReAct baseline (Paper 2/3 candidate) — a tool-using agent
(src/baselines/react_agent.py) evaluated as an alternative to the
single-shot vision_baseline.py / verifier_cot.py pipelines. Same vLLM
model-serving path as every other script in this repo (src/common/
vlm_load.py) — the model being tested does its own tool calling via its
chat template, no external API. read/glob/grep/write are local (write
sandboxed to a per-episode subfolder under one shared workspace/, created
lazily only if actually used — see below); retrieve/search_agml are
agml-agent's Python functions, called directly (no MCP connection) — see
src/baselines/tools.py.

Default model is RedHatAI's quantized Gemma-4 (RedHatAI/gemma-4-31B-it-
FP8-dynamic, same checkpoint used elsewhere in this repo), served with
--chat_template_path pointing at the vendored
src/config/tool_chat_template_gemma4.jinja — Gemma-4's own bundled template
has no `tools` support at all (same gap as its missing enable_thinking
support, see src/common/chat_prompt.py's module docstring), but this
vendored template does declare and parse tool calls, in its own
non-Hermes syntax (src/baselines/tool_call_formats.py's "gemma4" parser).
tool_call_format_for(args.model) picks the right parser automatically;
override with --tool_call_format if using some other tool-calling model
(e.g. a Qwen3/Qwen2.5 checkpoint, "hermes").

Output layout — ONE checkpoint row per episode, NOT one folder per episode
(an earlier version of this script did the latter; at even a few thousand
images that's a few thousand tiny folders with no easy resume-by-scanning,
the wrong shape given every other stage in this repo already solves this
with a flat per-model checkpoint CSV — see src/common/vision_eval.py's
append_ckpt_row/load_ckpt/archive_ckpt_if_rerun, reused here unchanged):

    <output_dir>/<safe_model_name>/
        ckpt_<safe_model_name>.csv   # one row per episode (REACT_CKPT_FIELDS below);
                                      # resumable — load_ckpt() skips (agml_dataset,
                                      # class_label, image_name) triples already present
        trajectory_log.jsonl         # SHARED across every episode, one line per model
                                      # turn, each tagged with its own episode_key —
                                      # same append-only convention as judge_log.jsonl/
                                      # verifier_log.jsonl elsewhere in this repo
        workspace/<safe_episode_key>/  # created lazily, ONLY for an episode that
                                        # actually calls `write` at least once — most
                                        # episodes (plain classification, no `write`
                                        # use) leave no directory here at all

Each checkpoint row also carries `elapsed_seconds` — that episode's own
run_episode() wall-clock time (excludes the one-time model load/destroy
around the whole batch) — so a slow outlier is visible straight from the
checkpoint. See docs/stage3_react_baseline.md and app/react_baseline_viewer.py
(a Streamlit viewer over this exact checkpoint + trajectory log shape).

Three ways to run (checked in this order — --sample_images wins over
--tasks_jsonl wins over --task):

- `--task "..."` — a single ad-hoc, text-only episode (quick manual test),
  no agml_dataset/class_label/image — image_name defaults to an
  auto-generated "adhoc_<timestamp>_<hash>" key unless --episode_key is
  given.
- `--tasks_jsonl <path>` — the batch/harness mode: one JSON object per
  line, each with at minimum a "task" key and, optionally,
  "agml_dataset"/"class_label"/"image_name"/"image_path" keys. A row WITH
  "image_path" gets that image actually loaded (PIL, RGB) and attached to
  the episode (react_agent.py's `image` param) — a REAL classification
  episode, comparable to vision-baseline/verifier-cot, not a different
  task. A row without it is text-only, same as --task mode. Every row's
  model/tokenizer load happens ONCE, shared across every episode in the
  file — episodes are NOT destroy/reload cycles.
- `--sample_images` — the same thing as --tasks_jsonl over
  build_react_tasks.py's output, but IN-MEMORY, no intermediate file:
  same active_datasets/load_hf_samples/sample_class_images sampling AND the
  same cls_question() prompt text vision_baseline.py itself uses (pass
  --rubric to instead use the SAME fill-in-the-blank criteria chain
  vision-baseline's own --rubric flag AND verifier_cot.py's own
  k-candidate generation step always use — makes this comparable to
  "Baseline with rubric"/verifier-cot instead of plain vision-baseline;
  --config/--prompts/--manifest/--images_root/--n_samples/--subsample/
  --match_images_from mirror build_react_tasks.py's own flags exactly, see
  its docstring). Use react-build-tasks instead (a separate command, same
  underlying code) when you want the task list persisted/reviewable/
  reusable across multiple react-baseline runs rather than resampled every
  invocation.

Only tool-calling-capable models are valid here — confirmed for the
default Gemma-4 + vendored template combination, and for Qwen3/Qwen2.5
(their own bundled templates already support tools).

Usage:
    uv run python main.py react-baseline \\
        --task "What crops is early blight commonly found on?"

    # 3-task text-only smoke test (infra check, no images) — see docs/stage3_react_baseline.md
    uv run python main.py react-baseline --tasks_jsonl dataset/react_smoke_test_tasks.jsonl

    # real classification episodes, images included, sampled on the fly:
    uv run python main.py react-baseline --sample_images --n_samples 3

    # same, but rubric-wrapped (comparable to Stage 2 / verifier-cot instead of plain vision-baseline):
    uv run python main.py react-baseline --sample_images --n_samples 3 --rubric

    # persisted/reusable task file instead of --sample_images:
    uv run python main.py react-build-tasks --n_samples 3 --output outputs_final/react_baseline/tasks/react_tasks_smoke3.jsonl
    uv run python main.py react-baseline --tasks_jsonl outputs_final/react_baseline/tasks/react_tasks_smoke3.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import hashlib
import re
import time
from datetime import datetime

import yaml
from PIL import Image as PILImage

from src.baselines.build_react_tasks import build_tasks
from src.baselines.react_agent import run_episode
from src.baselines.tool_call_formats import TOOL_CALL_EXTRACTORS, tool_call_format_for
from src.common.io import load_jsonl
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name

REACT_CKPT_FIELDS = [
    # (agml_dataset, class_label, image_name) is the exact triple
    # load_ckpt()/append_ckpt_row() (src/common/vision_eval.py) key resume
    # and dedup on — reused unchanged rather than reinventing it. For a
    # non-image ad-hoc --task run, agml_dataset/class_label are left blank
    # and image_name alone carries the episode's key (either
    # --episode_key, or an auto-generated one) — still gives correct
    # resume/dedup behavior against a repeated invocation.
    "agml_dataset", "class_label", "image_name",
    "task",               # the exact prompt text sent as the first user turn
    "result",             # finish()'s reported answer — blank if the episode didn't finish
    "stop_reason",        # "finished" | "no_tool_call" | "max_iterations"
    "n_steps",
    "tools_used",         # tool names called, in order, ";"-joined — "finish" excluded (see stop_reason)
    "tool_call_format",
    "elapsed_seconds",    # wall-clock time for this episode's run_episode() call alone (excludes
                          # model load/destroy, which happen once outside the per-episode loop) —
                          # rounded to 0.1s. Lets you spot a slow outlier (stuck near
                          # --max_iterations, or a slow retrieve()/search_agml() call) straight from
                          # the checkpoint without re-reading trajectory_log.jsonl timestamps.
]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(s: str) -> str:
    """Filesystem-safe folder-name component for a workspace/ subfolder —
    collapses any run of non-alnum/underscore/dot/dash characters (path
    separators, spaces, etc.) into a single underscore."""
    return _UNSAFE_CHARS_RE.sub("_", s).strip("_") or "episode"


def _auto_episode_key(task: str) -> str:
    digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:8]
    return f"adhoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}"


def _load_episodes(args: argparse.Namespace) -> list[dict]:
    """Each episode dict has agml_dataset/class_label/image_name/task/
    image_path — from --sample_images (real AgML images, sampled in-memory,
    no intermediate file), --tasks_jsonl (batch mode, reading a file build_
    react_tasks.py or --sample_images itself already wrote), or a single
    row built from --task (ad-hoc mode, always image_path=None — text-only).
    Missing optional keys in a --tasks_jsonl line default to "" (agml_dataset/
    class_label) same as REACT_CKPT_FIELDS's own blank convention for a
    non-image task; image_path defaults to None (text-only row) if absent."""
    if args.sample_images:
        # build_tasks() (build_react_tasks.py) takes an argparse.Namespace
        # and only reads .config/.prompts/.manifest/.images_root/.n_samples/
        # .subsample/.match_images_from off it — this parser defines those
        # same dests, so `args` is passed straight through unchanged rather
        # than repackaging it into a second Namespace.
        return build_tasks(args)
    if args.tasks_jsonl:
        rows = load_jsonl(Path(args.tasks_jsonl))
        episodes = []
        for r in rows:
            if "task" not in r:
                raise ValueError(f"{args.tasks_jsonl}: row missing required \"task\" key: {r}")
            episodes.append({
                "agml_dataset": r.get("agml_dataset", ""),
                "class_label": r.get("class_label", ""),
                "image_name": r.get("image_name") or _auto_episode_key(r["task"]),
                "task": r["task"],
                "image_path": r.get("image_path"),
            })
        return episodes
    return [{
        "agml_dataset": "",
        "class_label": "",
        "image_name": args.episode_key or _auto_episode_key(args.task),
        "task": args.task,
        "image_path": None,
    }]


def main(args: argparse.Namespace) -> None:
    model_dir = Path(args.output_dir) / safe_model_name(args.model)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(args.model)}.csv"
    trajectory_log_path = model_dir / "trajectory_log.jsonl"
    workspace_root = model_dir / "workspace"

    archive_ckpt_if_rerun(ckpt_path, args.rerun, extra_logs=("trajectory_log.jsonl",))
    _, seen = load_ckpt(ckpt_path, fields=REACT_CKPT_FIELDS)

    episodes = _load_episodes(args)
    pending = [e for e in episodes if (e["agml_dataset"], e["class_label"], e["image_name"]) not in seen]
    n_skipped = len(episodes) - len(pending)
    if n_skipped:
        print(f"[react-baseline] skipping {n_skipped} episode(s) already in {ckpt_path.name} (resume).")
    if not pending:
        print("[react-baseline] nothing to run — every episode already checkpointed.")
        return

    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None
    tool_call_format = args.tool_call_format or tool_call_format_for(args.model)
    print(f"[react-baseline] {len(pending)} episode(s) pending, tool_call_format={tool_call_format!r} "
          + ("(--tool_call_format)" if args.tool_call_format else f"(auto-detected from --model {args.model!r})"))

    llm, tokenizer = load_llm(
        args.model,
        quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        # Set unconditionally, not just when a --tasks_jsonl row actually
        # carries image_path — this is a per-model-LOAD kwarg (vLLM's
        # mm_processor_kwargs), not per-request, so it has to be right
        # before we know which specific rows in this batch have images.
        # Harmless for a text-only episode (never touched unless
        # multi_modal_data is actually passed on that generate() call).
        vision_token_budget=args.vision_token_budget,
        caller_tag="react-baseline",
    )
    try:
        for i, ep in enumerate(pending, start=1):
            key = "/".join(filter(None, [ep["agml_dataset"], ep["class_label"], ep["image_name"]]))
            image = PILImage.open(ep["image_path"]).convert("RGB") if ep.get("image_path") else None
            t0 = time.time()
            result = run_episode(
                llm, tokenizer, ep["task"],
                image=image,
                episode_key=key,
                tool_call_format=tool_call_format,
                sandbox_root=workspace_root / _safe_component(key),
                max_iterations=args.max_iterations,
                gen_max_tokens=args.gen_max_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                chat_template=chat_template,
                log_path=trajectory_log_path,
                verbose=not args.quiet,
            )
            elapsed = time.time() - t0
            append_ckpt_row(ckpt_path, {
                "agml_dataset": ep["agml_dataset"],
                "class_label": ep["class_label"],
                "image_name": ep["image_name"],
                "task": ep["task"],
                "result": result.result or "",
                "stop_reason": result.stop_reason,
                "n_steps": result.n_steps,
                "tools_used": ";".join(result.tools_used),
                "tool_call_format": tool_call_format,
                "elapsed_seconds": round(elapsed, 1),
            }, fields=REACT_CKPT_FIELDS)
            print(f"[react-baseline] ({i}/{len(pending)}) {key!r} -> {result.stop_reason} "
                  f"in {result.n_steps} step(s), {elapsed:.1f}s")
    finally:
        destroy_llm(llm)

    print(f"\nDone. Checkpoint: {ckpt_path}")
    print(f"Trajectory log: {trajectory_log_path}")


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--react_config", default="src/config/react_baseline_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--react_config", default=_pre_args.react_config,
        help="YAML file of defaults (model, max_iterations, etc.) — edit it directly for "
             "routine changes instead of passing flags every time. Any flag also passed on the "
             "command line overrides this file.",
    )
    parser.add_argument("--task", default=None, help="A single freeform ad-hoc task. Ignored if --tasks_jsonl is given.")
    parser.add_argument("--episode_key", default=None,
                         help="Checkpoint key for --task mode (goes in the image_name column). "
                              "Default: auto-generated from a timestamp + hash of the task text.")
    parser.add_argument("--tasks_jsonl", default=None,
                         help="Batch mode: path to a JSONL file, one episode per line, each a JSON "
                              "object with a required \"task\" key and optional \"agml_dataset\"/"
                              "\"class_label\"/\"image_name\" keys. Takes precedence over --task, "
                              "ignored if --sample_images is given.")
    parser.add_argument("--sample_images", action="store_true",
                         help="Sample real AgML images in-memory (same code build_react_tasks.py's "
                              "own CLI uses — active_datasets/load_hf_samples/sample_class_images/"
                              "cls_question/apply_rubric) instead of reading --tasks_jsonl from disk. "
                              "Takes precedence over both --tasks_jsonl and --task. Use "
                              "react-build-tasks instead when you want the task list persisted/"
                              "reviewable/reusable across multiple react-baseline runs rather than "
                              "resampled every invocation.")
    parser.add_argument("--config", default="src/config/datasets.yaml",
                         help="Only used with --sample_images.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml",
                         help="Only used with --sample_images.")
    parser.add_argument("--manifest", default="dataset/selected_images.json",
                         help="Only used with --sample_images.")
    parser.add_argument("--images_root", default="dataset/images",
                         help="Only used with --sample_images.")
    parser.add_argument("--n_samples", type=int, default=3,
                         help="Images/class. Only used with --sample_images. Kept small (3) by "
                              "default — raise it once react-baseline itself is trusted.")
    parser.add_argument("--subsample", nargs="+", default=None,
                         help="Restrict --sample_images to these dataset(s) only (agml_name, "
                              "space-separated) — same convention as verifier_cot.py's own "
                              "--subsample.")
    parser.add_argument("--match_images_from", default=None,
                         help="Only used with --sample_images. Path to an existing "
                              "single-prediction-per-row checkpoint (e.g. outputs_final/"
                              "vision_baseline_thinking/<model>/ckpt_<model>.csv) — replays that "
                              "run's own sampled image names for a true apples-to-apples image set, "
                              "same convention as verifier_cot.py's own flag of this name.")
    parser.add_argument("--rubric", action="store_true",
                         help="Only used with --sample_images. Wrap the classification question "
                              "through the same fill-in-the-blank criteria chain vision_baseline.py's "
                              "own --rubric flag AND verifier_cot.py's own k-candidate generation "
                              "step always use — pass this to compare react-baseline against "
                              "'Baseline with rubric' (Stage 2) or verifier-cot rather than plain "
                              "vision-baseline. Off by default (plain question).")
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                         help="Must be a tool-calling-capable model. Gemma-4 needs "
                              "--chat_template_path pointing at the vendored gemma4 template "
                              "(its own bundled template has no tools support) — set as this "
                              "flag's own default below. Qwen3/Qwen2.5 work with no override.")
    parser.add_argument("--tool_call_format", choices=sorted(TOOL_CALL_EXTRACTORS), default=None,
                         help="Which of tool_call_formats.py's parsers reads the model's raw "
                              "generation. Default: auto-detected from --model (tool_call_format_for) "
                              "— override only if the auto-detection would pick wrong.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint (+ trajectory_log.jsonl) aside, "
                              "timestamped, and regenerate every episode from scratch. Old files "
                              "are archived, never deleted — same convention as every other stage.")
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--gen_max_tokens", type=int, default=1024,
                         help="max_tokens per turn. Raise well above this with --enable_thinking, "
                              "same reasoning as every other generation script in this repo.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress run_episode()'s per-turn progress lines (step index + "
                              "generating/tool-call/finish) — keeps just react-baseline's own "
                              "per-episode summary lines. On by default (verbose) since a single "
                              "turn's generation can take real minutes with otherwise zero terminal "
                              "output to tell it apart from a hang.")
    parser.add_argument("--enable_thinking", action="store_true",
                         help="Model's own extended thinking, off by default — same convention as "
                              "chat_prompt.py's render_chat_template (see its module docstring).")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja",
                         help="Override the tokenizer's own bundled chat template with a Jinja file "
                              "at this path — REQUIRED for the default Gemma-4 model (its own bundled "
                              "template has no tools/enable_thinking support at all). Pass '' (empty "
                              "string) to use the tokenizer's own bundled template instead — needed "
                              "for a Qwen3/Qwen2.5 model, whose own template already supports both.")
    parser.add_argument("--output_dir", default="outputs_final/react_baseline/")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET,
                         help="Only matters for --tasks_jsonl rows carrying image_path — aligns "
                              "per-image token count across model families, same as vision_baseline.py's "
                              "own flag of this name (src/common/vlm_load.py).")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")

    if _pre_args.react_config and Path(_pre_args.react_config).exists():
        _cfg = _load_yaml(Path(_pre_args.react_config)) or {}
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests
        if _unknown:
            print(f"[warn] {_pre_args.react_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    args = parser.parse_args()
    if not args.chat_template_path:
        args.chat_template_path = None
    if not args.task and not args.tasks_jsonl and not args.sample_images:
        parser.error("one of --task, --tasks_jsonl, or --sample_images is required")
    main(args)
