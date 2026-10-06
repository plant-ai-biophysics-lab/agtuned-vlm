"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 3b — self-
optimization loop: EDIT an existing skill-memory.md directly against
per-task eval-split scores, instead of re-summarizing from Step 2's
traces (heldout-summarize). One round = one call of this script:

  1. Read an already-judged `heldout-zeroshot --split eval
     --skill_memory_path <current skill>` checkpoint (the SAME eval
     split every round — see the caveat below).
  2. Compute judge_accuracy per bucket_category (disease/pest/
     deficiency/healthy), same heuristic Step 1/3 already use.
  3. Sample up to --n_examples_per_bucket wrong rows per bucket (the
     model's own wrong prediction, the real ground truth, its own
     stated reasoning, the expert's observations — NOT images; this is
     a text-only edit call, same convention as heldout-summarize's
     reduce phase).
  4. ONE LLM call: current skill.md text + the score breakdown + the
     miss examples -> a FULL REVISED skill.md (heldout_skill_edit_prompt
     in heldout_prompts.yaml). The call never sees Step 2's original
     explanation traces — only the skill file's own current wording and
     this round's own failures, so each round edits what's actually
     there rather than re-deriving from scratch.

Run this once, inject the output into a fresh `heldout-zeroshot --split
eval --skill_memory_path <this round's output>` run, then feed THAT
checkpoint into the next round. Each round's output REPLACES (not
merges onto) the input skill file, so the document stays a single,
bounded-size, directly-editable artifact across rounds rather than an
ever-growing accretion.

**Explicit, stated caveat (do not lose track of this)**: every round's
edit is steered DIRECTLY by this fixed eval split's own scores — this is
intentionally optimizing against the same n=199 split repeatedly, not
the lessons split. That is a real overfitting risk (acknowledged,
chosen anyway per docs/stage3_context_memory.md's Step 3b section) —
track every round's per-task scores (--round_log) so a non-monotonic or
noisy trajectory is visible, and treat any single round's number as
provisional, not a result, until confirmed on a split this loop never
touched.

Usage (round N -> round N+1):
    uv run python main.py heldout-skill-optimize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --eval_ckpt outputs_final/heldout_zeroshot_eval_optimize_round0/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --skill_memory_path src/config/skill_memory_disease-pest.md \\
        --output_md outputs_final/heldout_skill_optimize/skill_memory_round1.md \\
        --round_log outputs_final/heldout_skill_optimize/round_log.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — explanations/reasoning fields can exceed the 131072-byte default on a full run.
import json
import random
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_channel_thought
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.heldout_zeroshot import bucket_category

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_BUCKETS = ["disease", "pest", "deficiency", "healthy"]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def compute_task_scores(rows: list[dict]) -> dict[str, dict]:
    """{bucket: {"accuracy", "n"}} — same bucket_category heuristic Step
    1/3 use, applied to each row's own class_label (ground truth)."""
    per_bucket: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        jc = r.get("judged_correct")
        if jc in (None, ""):
            continue
        per_bucket[bucket_category(r["class_label"])].append(int(jc))
    out = {}
    for b in _BUCKETS:
        vals = per_bucket.get(b, [])
        out[b] = {"accuracy": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}
    return out


def build_task_scores_block(scores: dict[str, dict]) -> str:
    lines = []
    for b in _BUCKETS:
        s = scores[b]
        if s["n"] == 0:
            continue
        lines.append(f"- {b}: {s['accuracy']:.1%} accuracy (n={s['n']})")
    return "\n".join(lines)


def sample_miss_examples(rows: list[dict], n_per_bucket: int, seed: int) -> dict[str, list[dict]]:
    """{bucket: [row, ...]} — up to n_per_bucket WRONG rows per bucket,
    sampled (not just the first N) so repeated rounds against the same
    eval split don't always see the identical miss subset."""
    rng = random.Random(seed)
    misses: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct") in (None, ""):
            continue
        if int(r["judged_correct"]) == 1:
            continue
        misses[bucket_category(r["class_label"])].append(r)
    out = {}
    for b in _BUCKETS:
        pool = misses.get(b, [])
        out[b] = rng.sample(pool, min(n_per_bucket, len(pool)))
    return out


def build_miss_examples_block(examples_by_bucket: dict[str, list[dict]]) -> str:
    blocks = []
    for b in _BUCKETS:
        for r in examples_by_bucket.get(b, []):
            blocks.append(
                f"[{b}] {r.get('crop', '?')} plant — model answered "
                f"\"{(r.get('extracted_answer') or r.get('prediction') or '').strip()}\", actual answer: "
                f"\"{r['class_label']}\". Model's own reasoning: \"{(r.get('reasoning') or '').strip()[:500]}\". "
                f"Expert's observations: \"{(r.get('details') or '').strip()[:500]}\"."
            )
    return "\n\n".join(blocks)


def run_model(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.heldout_prompts))
    edit_template = hprompts["heldout_skill_edit_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    with open(args.eval_ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: {args.eval_ckpt} has no rows.")
    n_judged = sum(1 for r in rows if r.get("judged_correct") not in (None, ""))
    if n_judged == 0:
        raise SystemExit(f"ERROR: {args.eval_ckpt} has no judged rows — was this run without --skip_judge?")

    scores = compute_task_scores(rows)
    task_scores_block = build_task_scores_block(scores)
    print("  Per-task scores this round:")
    for line in task_scores_block.splitlines():
        print(f"   {line}")

    examples_by_bucket = sample_miss_examples(rows, args.n_examples_per_bucket, args.seed)
    miss_examples_block = build_miss_examples_block(examples_by_bucket)
    n_examples = sum(len(v) for v in examples_by_bucket.values())
    print(f"  {n_examples} miss example(s) sampled across {len(_BUCKETS)} bucket(s)")

    current_skill_md = Path(args.skill_memory_path).read_text(encoding="utf-8").strip()
    edit_text = edit_template.format(
        current_skill_md=current_skill_md, task_scores_block=task_scores_block,
        miss_examples_block=miss_examples_block,
    )

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="heldout_skill_optimize_edit",
    )
    hist = [{"role": "user", "content": edit_text}]
    prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking, chat_template=chat_template)
    gen_params = SamplingParams(
        temperature=0.0, max_tokens=args.edit_max_tokens, seed=args.seed,
        skip_special_tokens=not args.enable_thinking,
    )
    out = llm.generate([prompt], sampling_params=gen_params)[0]
    raw = out.outputs[0].text
    _, md_text = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
    md_text = md_text.strip()

    destroy_llm(llm)
    del tokenizer

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text)
    print(f"\n  Revised skill memory: {out_path} ({len(md_text.split())} words)")

    if args.round_log:
        log_path = Path(args.round_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(log_path.read_text()) if log_path.exists() else []
        overall_vals = [int(r["judged_correct"]) for r in rows if r.get("judged_correct") not in (None, "")]
        history.append({
            "round_input_skill": str(args.skill_memory_path),
            "round_output_skill": str(out_path),
            "eval_ckpt": str(args.eval_ckpt),
            "overall_judge_accuracy": round(sum(overall_vals) / len(overall_vals), 4),
            "task_scores": scores,
            "n_examples_shown": n_examples,
        })
        log_path.write_text(json.dumps(history, indent=2))
        print(f"  Round log ({len(history)} round(s)): {log_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--eval_ckpt", required=True,
                         help="An already-judged heldout-zeroshot --split eval checkpoint, run WITH "
                              "--skill_memory_path pointed at the CURRENT skill file (this round's "
                              "input) — the scores driving this round's edit must reflect that exact "
                              "document, not a different one.")
    parser.add_argument("--skill_memory_path", required=True,
                         help="The CURRENT skill file text to revise — same file --eval_ckpt was "
                              "generated against.")
    parser.add_argument("--output_md", required=True,
                         help="Where this round's revised skill file is written — a NEW path (e.g. "
                              "skill_memory_round<N+1>.md), never overwriting --skill_memory_path, so "
                              "every round's version stays on disk for inspection/rollback.")
    parser.add_argument("--n_examples_per_bucket", type=int, default=5,
                         help="Wrong-row examples sampled per task bucket and shown to the edit call.")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--round_log", default=None,
                         help="Append this round's scores/paths to this JSON file (a list, one entry "
                              "per round) — track the trajectory across rounds rather than only ever "
                              "looking at the latest number.")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--edit_max_tokens", type=int, default=4096,
                         help="The revised document is at least as long as the input skill file.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Also controls which miss examples are sampled — vary this across rounds "
                              "if you want each round to see a different miss subset.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
