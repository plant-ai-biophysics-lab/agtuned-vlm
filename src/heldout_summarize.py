"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 3 — summarize
Step 2's explanation traces into ONE skill-memory .md file. Text-only
map-reduce (no images — these are already-written explanations about
images, not the images themselves):

  MAP: group all explanations by (heuristic category, trace_type) —
  disease/pest/deficiency/healthy x worked/didnt_work, 8 buckets — via
  bucket_category() (heldout_zeroshot.py, same heuristic the Step 1
  trend analysis used). A bucket whose raw explanation text would
  exceed --max_words_per_chunk is split further into word-bounded
  chunks. ONE summarization call per chunk, producing a short bullet
  list of recurring, GENERALIZABLE patterns — never restating individual
  explanations, never facts about a single image. (A real run: 827
  explanations / ~101K words split into 8 buckets, 2 of which needed
  splitting into 2 chunks each — ~11 map calls total, not 827.)

  REDUCE: ONE final call merges every chunk-level bullet summary into a
  single Markdown reference document, organized into "What Worked" /
  "What Didn't Work" / "Common Diseases, Pests, and Deficiencies to
  Know" sections.

Map-step chunk summaries are cached to <output_dir>/chunk_summaries.json
after the map phase — a crash/interrupt during the (cheap, ~1 call)
reduce phase doesn't lose the (more expensive, ~11 calls) map phase's
work; re-running with the cache already present skips straight to
reduce unless --rerun.

Usage:
    uv run python main.py heldout-summarize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --ckpt outputs_final/heldout_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_thinking
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.vision_eval import safe_model_name
from src.common.progress import ProgressBar
from src.heldout_zeroshot import bucket_category_for_domain

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_TRACE_TYPE_LABEL = {"worked": "cases it got RIGHT", "didnt_work": "cases it got WRONG"}


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_chunks(rows: list[dict], max_words_per_chunk: int,
                  bucket_fn=bucket_category_for_domain("disease_pest")) -> dict[tuple[str, str], list[list[dict]]]:
    """{(category, trace_type): [chunk_rows, ...]} — a bucket's rows are
    kept in one chunk unless their total explanation word count exceeds
    max_words_per_chunk, in which case they're split sequentially into
    multiple word-bounded chunks (not by row count — explanation length
    varies a lot, so a fixed row count per chunk would under/overshoot
    the context budget depending on which rows landed in it). `bucket_fn`
    defaults to disease/pest's bucket_category; run_model passes
    bucket_category_quality instead for --domain quality (a single flat
    "quality" bucket — see that function's own docstring for why)."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(bucket_fn(r), r["trace_type"])].append(r)

    chunked: dict[tuple[str, str], list[list[dict]]] = {}
    for key, bucket_rows in buckets.items():
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
        chunked[key] = chunks
    return chunked


def compact_bucket_blocks(
    cat: str, tt: str, summaries: list[dict], max_words_per_group: int,
    llm, tokenizer, map_template: str, chat_template: str | None, args: argparse.Namespace,
) -> list[dict]:
    """Recursively re-condense one (category, trace_type) bucket's own
    chunk summaries (via the SAME map_template — structurally identical
    task whether the N items being condensed are raw explanations [map
    phase] or already-summarized chunks [here]) until their total word
    count fits comfortably inside a single reduce call's context budget.

    Detection analog (heldout_detection_summarize.py's own
    compact_error_type_blocks) was built after a real run produced 110
    map chunks (box-level granularity) that blew --max_model_len when
    concatenated into one reduce prompt. Classification's own real runs
    hadn't needed this (disease/pest: ~11 chunks total) — but a real
    --domain quality run produced 81 chunks for its single flat "quality"
    bucket (no sub-category split to spread chunks across), totaling
    ~18.7K tokens of chunk-summary text alone against a 24576-token
    window — leaving so little room that --reduce_max_tokens couldn't
    help no matter how high it was set; the model's output was being cut
    off by the context window itself, not a max_tokens cap."""
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
        print(f"    [reduce-compact] {cat}/{tt} level {level}: {len(summaries)} summaries "
              f"({total_words} words) -> {len(groups)} group(s)")
        new_summaries = []
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=args.map_max_tokens, seed=args.seed,
            skip_special_tokens=not args.enable_thinking,
        )
        for gi, group in enumerate(groups):
            explanations_block = "\n".join(f"{j+1}. {s['summary']}" for j, s in enumerate(group))
            text = map_template.format(
                category=cat, trace_type_label=_TRACE_TYPE_LABEL[tt], n=len(group),
                explanations_block=explanations_block,
            )
            hist = [{"role": "user", "content": text}]
            prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking,
                                           chat_template=chat_template)
            out = llm.generate([prompt], sampling_params=gen_params)[0]
            raw = out.outputs[0].text
            _, cleaned = extract_thinking(raw)
            new_summaries.append({
                "category": cat, "trace_type": tt, "chunk_index": gi,
                "n_rows": sum(s["n_rows"] for s in group), "summary": cleaned.strip(),
            })
        summaries = new_summaries


def run_model(model_id: str, args: argparse.Namespace) -> None:
    domain = getattr(args, "domain", "disease_pest")
    prefix = f"heldout_{domain}_" if domain != "disease_pest" else "heldout_"
    hprompts = load_yaml(Path(args.heldout_prompts))
    map_template = hprompts[f"{prefix}map_summarize_prompt"]
    reduce_template = hprompts[f"{prefix}reduce_summarize_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None
    bucket_fn = bucket_category_for_domain(domain)

    with open(args.ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("explanation", "").strip()]
    if not rows:
        raise SystemExit(f"ERROR: {args.ckpt} has no non-blank explanation rows.")

    chunked = build_chunks(rows, args.max_words_per_chunk, bucket_fn=bucket_fn)
    flat_chunks = [(cat, tt, i, chunk) for (cat, tt), chunks in chunked.items() for i, chunk in enumerate(chunks)]
    n_total_calls = len(flat_chunks)
    print(f"  {len(rows)} explanation row(s), {len(chunked)} (category, trace_type) bucket(s), "
          f"{n_total_calls} map chunk(s):")
    for (cat, tt), chunks in sorted(chunked.items()):
        sizes = [len(c) for c in chunks]
        print(f"    {cat}/{tt}: {sum(sizes)} row(s) -> {len(chunks)} chunk(s) {sizes}")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "chunk_summaries.json"
    if args.rerun and cache_path.exists():
        cache_path.unlink()
    chunk_summaries: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    pending = [(cat, tt, i, chunk) for cat, tt, i, chunk in flat_chunks if f"{cat}|{tt}|{i}" not in chunk_summaries]

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
            caller_tag="heldout_summarize_map",
        )
        pbar = ProgressBar(total=len(pending), desc="Map")
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=args.map_max_tokens, seed=args.seed,
            skip_special_tokens=not args.enable_thinking,
        )
        for cat, tt, i, chunk in pending:
            explanations_block = "\n".join(f"{j+1}. {r['explanation']}" for j, r in enumerate(chunk))
            text = map_template.format(
                category=cat, trace_type_label=_TRACE_TYPE_LABEL[tt], n=len(chunk),
                explanations_block=explanations_block,
            )
            hist = [{"role": "user", "content": text}]
            prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking,
                                           chat_template=chat_template)
            out = llm.generate([prompt], sampling_params=gen_params)[0]
            raw = out.outputs[0].text
            # Unconditional — see extract_thinking's own docstring; strips
            # either Gemma-4's channel format or Qwen's <think> tags.
            _, cleaned = extract_thinking(raw)
            chunk_summaries[f"{cat}|{tt}|{i}"] = {
                "category": cat, "trace_type": tt, "chunk_index": i, "n_rows": len(chunk),
                "summary": cleaned.strip(),
            }
            cache_path.write_text(json.dumps(chunk_summaries, indent=2))
            pbar.update()
        pbar.finish()
    else:
        print("  [map] all chunks already cached")

    # Reduce — one final call, separate (small, cheap) LLM load if the
    # map phase already destroyed its own instance, so a map-only rerun
    # (map cache already complete) doesn't need to reload the model
    # unnecessarily before realizing reduce is all that's left — but
    # here they share one load since both run in the same invocation.
    print("  [reduce] merging all chunk summaries into the final .md ...")
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in chunk_summaries.values():
        by_bucket[(v["category"], v["trace_type"])].append(v)

    if llm is None:
        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_summarize_reduce",
        )

    compacted: dict[tuple[str, str], list[dict]] = {}
    for (cat, tt), summaries in by_bucket.items():
        compacted[(cat, tt)] = compact_bucket_blocks(
            cat, tt, summaries, args.max_words_per_reduce_group, llm, tokenizer, map_template,
            chat_template, args,
        )

    by_section: dict[str, list[dict]] = defaultdict(list)
    for (cat, tt), summaries in compacted.items():
        for v in summaries:
            by_section[tt].append(v)
    blocks = []
    for tt in ("worked", "didnt_work"):
        for v in sorted(by_section.get(tt, []), key=lambda x: x["category"]):
            blocks.append(f"### {v['category']} — {_TRACE_TYPE_LABEL[tt]} (n={v['n_rows']})\n{v['summary']}")
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
    _, md_text = extract_thinking(raw)
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
    parser.add_argument("--domain", choices=["disease_pest", "quality", "species"], default="disease_pest",
                         help="Selects both the bucketing heuristic (disease/pest/deficiency/healthy "
                              "vs. a single flat 'quality' bucket) and which heldout_prompts.yaml "
                              "template pair to use (heldout_* vs heldout_quality_*).")
    parser.add_argument("--ckpt", required=True,
                         help="heldout-explain-traces's own output checkpoint CSV.")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_summarize/",
                         help="Where the chunk_summaries.json cache is written.")
    parser.add_argument("--output_md", default="outputs_final/heldout_summarize/skill_memory.md",
                         help="Final skill-memory file — this is what Step 4 ablates and, if it "
                              "helps, gets injected into vision-baseline/verifier-cot.")
    parser.add_argument("--max_words_per_chunk", type=int, default=8000,
                         help="A (category, trace_type) bucket whose raw explanation text exceeds "
                              "this is split into multiple map chunks instead of one call.")
    parser.add_argument("--max_words_per_reduce_group", type=int, default=1500,
                         help="A (category, trace_type) bucket whose MAP-phase chunk summaries "
                              "together exceed this many words are recursively re-condensed "
                              "(compact_bucket_blocks) before the final reduce call — prevents a "
                              "bucket with many map chunks (e.g. --domain quality's single flat "
                              "bucket) from blowing --max_model_len when concatenated into one "
                              "reduce prompt, a real failure found on a 81-chunk quality run.")
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
    parser.add_argument("--max_model_len", type=int, default=24576,
                         help="Same default every other stage in this pipeline uses. The largest real "
                              "bucket (pest/didnt_work) needs ~19-20K input tokens alone, plus up to "
                              "--map_max_tokens (2048) generated on top — still comfortably under "
                              "this, so raise it only if your own run's bucket sizes differ a lot or "
                              "KV-cache memory errors regardless (lower --gpu_memory_utilization or "
                              "raise this together, per vLLM's own suggestion in that error).")
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
