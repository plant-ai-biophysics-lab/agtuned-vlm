"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 3 —
summarize Step 2's box-level explanation traces into ONE
skill_memory_detection.md file. Text-only map-reduce, same mechanic
heldout_summarize.py (Stage 3's own Step 3) uses:

  MAP: group all explanations by `error_type` (false_positive/missed/
  loose_match/worked — Step 2's own failure/success taxonomy; no
  separate trace_type dimension the way classification's bucket_category
  x trace_type is, since error_type already encodes worked-vs-failure
  directly). A bucket whose raw explanation text would exceed
  --max_words_per_chunk is split further into word-bounded chunks. ONE
  summarization call per chunk, producing a short bullet list of
  recurring, GENERALIZABLE VISUAL/SPATIAL patterns (not coordinate
  numbers — detection_skill_map_summarize_prompt's own rule).

  REDUCE: ONE final call merges every chunk-level bullet summary into a
  single Markdown reference document, organized into "What Worked" /
  "What Didn't Work" (split into False Positives / Missed Instances /
  Loose or Imprecise Boxes) sections.

Map-step chunk summaries are cached to <output_dir>/chunk_summaries.json
after the map phase — same resumability heldout_summarize.py's own
cache gives: a crash/interrupt during the (cheap, ~1 call) reduce phase
doesn't lose the (more expensive, N-chunk) map phase's work.

Usage:
    uv run python main.py heldout-detection-summarize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — pred_bbox_json/gt_bbox_json/explanation can exceed the 131072-byte default.
import json
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_channel_thought
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.vision_eval import safe_model_name
from src.common.progress import ProgressBar

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_ERROR_TYPE_LABEL = {
    "false_positive": "cases where it predicted a box but there was no real instance there (false positives)",
    "missed": "cases where it missed a real instance entirely (no box predicted at all)",
    "loose_match": "cases where its box was a real match but not tight (too loose/large, or should have been split)",
    "worked": "cases where it correctly identified and tightly matched a real instance",
}
_ERROR_TYPES = ("false_positive", "missed", "loose_match", "worked")


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_chunks(rows: list[dict], max_words_per_chunk: int) -> dict[str, list[list[dict]]]:
    """{error_type: [chunk_rows, ...]} — a bucket's rows are kept in one
    chunk unless their total explanation word count exceeds
    max_words_per_chunk, in which case they're split sequentially into
    multiple word-bounded chunks — same logic heldout_summarize.py's own
    build_chunks uses, just keyed by error_type alone instead of a
    (category, trace_type) tuple."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["error_type"]].append(r)

    chunked: dict[str, list[list[dict]]] = {}
    for error_type, bucket_rows in buckets.items():
        chunks: list[list[dict]] = []
        current: list[dict] = []
        current_words = 0
        for r in bucket_rows:
            w = len(r["explanation"].split())
            if current and current_words + w > max_words_per_chunk:
                chunks.append(current)
                current, current_words = [], 0
            current.append(r)
            current_words += w
        if current:
            chunks.append(current)
        chunked[error_type] = chunks
    return chunked


def compact_error_type_blocks(
    error_type: str, summaries: list[dict], max_words_per_group: int,
    llm, tokenizer, map_template: str, chat_template: str | None, args: argparse.Namespace,
) -> list[dict]:
    """Recursively re-condense one error_type's own chunk summaries (via
    the SAME map_template — structurally identical task whether the N
    items being condensed are raw explanations (map phase) or already-
    summarized chunks (here): condense N bullet-texts into 4-8 bullets)
    until their total word count fits comfortably inside a single reduce
    call's context budget.

    Needed because build_chunks' own word-bounded chunking only bounds
    EACH chunk's own size, not how many chunks a bucket produces in
    total — classification's real run produced ~11 map chunks total
    (heldout_summarize.py never needed this), but a real detection run
    produced 110 (box-level granularity, not row-level), and simply
    concatenating all 110 summaries into one reduce prompt blew
    --max_model_len entirely (24577 input tokens requested against a
    24576-token window, leaving ZERO room for --reduce_max_tokens of
    actual output — a real failure, not a close call)."""
    summaries = sorted(summaries, key=lambda x: x["chunk_index"])
    level = 0
    while True:
        total_words = sum(len(s["summary"].split()) for s in summaries)
        if len(summaries) <= 1 or total_words <= max_words_per_group:
            return summaries
        level += 1
        groups: list[list[dict]] = []
        current: list[dict] = []
        current_words = 0
        for s in summaries:
            w = len(s["summary"].split())
            if current and current_words + w > max_words_per_group:
                groups.append(current)
                current, current_words = [], 0
            current.append(s)
            current_words += w
        if current:
            groups.append(current)
        if len(groups) >= len(summaries):
            # Grouping didn't reduce the count (every item alone already
            # exceeds the budget) — stop here rather than looping forever.
            return summaries
        print(f"    [reduce-compact] {error_type} level {level}: {len(summaries)} summaries "
              f"({total_words} words) -> {len(groups)} group(s)")
        new_summaries = []
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=args.map_max_tokens, seed=args.seed,
            skip_special_tokens=not args.enable_thinking,
        )
        for gi, group in enumerate(groups):
            explanations_block = "\n".join(f"{j+1}. {s['summary']}" for j, s in enumerate(group))
            text = map_template.format(
                error_type_label=_ERROR_TYPE_LABEL[error_type], n=len(group),
                explanations_block=explanations_block)
            hist = [{"role": "user", "content": text}]
            prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking,
                                           chat_template=chat_template)
            out = llm.generate([prompt], sampling_params=gen_params)[0]
            raw = out.outputs[0].text
            _, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
            new_summaries.append({
                "error_type": error_type, "chunk_index": gi,
                "n_rows": sum(s["n_rows"] for s in group), "summary": cleaned.strip(),
            })
        summaries = new_summaries


def run_model(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.detection_skill_prompts))
    map_template = hprompts["detection_skill_map_summarize_prompt"]
    reduce_template = hprompts["detection_skill_reduce_summarize_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    with open(args.ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("explanation", "").strip()]
    if not rows:
        raise SystemExit(f"ERROR: {args.ckpt} has no non-blank explanation rows.")

    chunked = build_chunks(rows, args.max_words_per_chunk)
    flat_chunks = [(et, i, chunk) for et, chunks in chunked.items() for i, chunk in enumerate(chunks)]
    n_total_calls = len(flat_chunks)
    print(f"  {len(rows)} explanation row(s), {len(chunked)} error_type bucket(s), "
          f"{n_total_calls} map chunk(s):")
    for et in _ERROR_TYPES:
        chunks = chunked.get(et, [])
        if not chunks:
            continue
        sizes = [len(c) for c in chunks]
        print(f"    {et}: {sum(sizes)} row(s) -> {len(chunks)} chunk(s) {sizes}")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "chunk_summaries.json"
    if args.rerun and cache_path.exists():
        cache_path.unlink()
    chunk_summaries: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    pending = [(et, i, chunk) for et, i, chunk in flat_chunks if f"{et}|{i}" not in chunk_summaries]

    llm = tokenizer = None
    if pending:
        print(f"  [map] {len(pending)}/{n_total_calls} chunk(s) pending")
        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_detection_summarize_map",
        )
        pbar = ProgressBar(total=len(pending), desc="Map")
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=args.map_max_tokens, seed=args.seed,
            skip_special_tokens=not args.enable_thinking,
        )
        for et, i, chunk in pending:
            explanations_block = "\n".join(
                f"{j+1}. [{r.get('category', '?')}] {r['explanation']}" for j, r in enumerate(chunk))
            text = map_template.format(
                error_type_label=_ERROR_TYPE_LABEL[et], n=len(chunk), explanations_block=explanations_block)
            hist = [{"role": "user", "content": text}]
            prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking,
                                           chat_template=chat_template)
            out = llm.generate([prompt], sampling_params=gen_params)[0]
            raw = out.outputs[0].text
            _, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
            chunk_summaries[f"{et}|{i}"] = {
                "error_type": et, "chunk_index": i, "n_rows": len(chunk), "summary": cleaned.strip(),
            }
            cache_path.write_text(json.dumps(chunk_summaries, indent=2))
            pbar.update()
        pbar.finish()
    else:
        print("  [map] all chunks already cached")

    by_error_type: dict[str, list[dict]] = defaultdict(list)
    for v in chunk_summaries.values():
        by_error_type[v["error_type"]].append(v)

    if llm is None:
        # Needed even if the map phase itself had nothing pending — the
        # compaction pass below may still require real generation calls.
        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_detection_summarize_reduce",
        )

    print("  [reduce] compacting each error_type bucket's own chunk summaries to fit the "
          f"reduce call's context budget (--max_words_per_reduce_group {args.max_words_per_reduce_group}) ...")
    blocks = []
    for et in _ERROR_TYPES:
        summaries = by_error_type.get(et, [])
        if not summaries:
            continue
        compacted = compact_error_type_blocks(
            et, summaries, args.max_words_per_reduce_group, llm, tokenizer, map_template,
            chat_template, args)
        for v in compacted:
            blocks.append(f"### {et} (n={v['n_rows']})\n{v['summary']}")

    print("  [reduce] merging all (compacted) chunk summaries into the final .md ...")
    chunk_summaries_block = "\n\n".join(blocks)
    reduce_text = reduce_template.format(chunk_summaries_block=chunk_summaries_block)

    hist = [{"role": "user", "content": reduce_text}]
    prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking, chat_template=chat_template)
    reduce_params = SamplingParams(
        temperature=0.0, max_tokens=args.reduce_max_tokens, seed=args.seed,
        skip_special_tokens=not args.enable_thinking,
    )
    out = llm.generate([prompt], sampling_params=reduce_params)[0]
    raw = out.outputs[0].text
    _, md_text = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
    md_text = md_text.strip()

    destroy_llm(llm)
    del tokenizer

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text)
    print(f"\n  Skill memory file: {out_path} ({len(md_text.split())} words)")
    print(f"  Chunk summaries cache: {cache_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--ckpt", required=True,
                         help="heldout-detection-explain-traces' own output checkpoint CSV.")
    parser.add_argument("--detection_skill_prompts", default="src/config/detection_skill_prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_detection_summarize/",
                         help="Where the chunk_summaries.json cache is written.")
    parser.add_argument("--output_md", default="outputs_final/heldout_detection_summarize/skill_memory_detection.md",
                         help="Final skill-memory file — this is what Step 4 ablates and, if it "
                              "helps, gets injected into detection-multiclass.")
    parser.add_argument("--max_words_per_chunk", type=int, default=8000,
                         help="An error_type bucket whose raw explanation text exceeds this is split "
                              "into multiple map chunks instead of one call.")
    parser.add_argument("--max_words_per_reduce_group", type=int, default=1500,
                         help="Per error_type, chunk summaries are recursively re-condensed "
                              "(compact_error_type_blocks) until their own total word count is under "
                              "this, BEFORE the single final reduce call — otherwise a bucket that "
                              "produced many map chunks (box-level granularity can easily exceed "
                              "classification's own ~11-chunk real run) can blow the reduce call's "
                              "entire --max_model_len with zero room left for output (a real failure "
                              "on a real run: 110 chunks concatenated requested 24577 input tokens "
                              "against a 24576-token window). 1500 words/error_type x 4 error_types "
                              "= 6000 words (~8000 tokens) max going into the final reduce call, "
                              "comfortably under --max_model_len with room for --reduce_max_tokens.")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--map_max_tokens", type=int, default=2048,
                         help="Short bullet-list output per chunk, but real headroom for a thinking "
                              "trace first.")
    parser.add_argument("--reduce_max_tokens", type=int, default=4096,
                         help="The final .md is longer than any one map summary.")
    parser.add_argument("--rerun", action="store_true",
                         help="Discard the chunk_summaries.json cache and redo the map phase from "
                              "scratch. Without this, an existing cache is reused and only missing "
                              "chunks are (re)computed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
