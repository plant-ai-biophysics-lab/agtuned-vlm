"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 3b — automated
self-optimization loop: starting from one skill-memory.md, repeatedly
(1) score it on the held-out eval split, (2) EDIT it directly against
that round's own per-task scores + a sample of its own mistakes
(NOT a re-summarize from Step 2's original traces), (3) repeat for
--max_rounds — then report the single best-scoring round's file.

One invocation runs the WHOLE loop (no more manually alternating
`heldout-zeroshot`/`heldout-skill-optimize` calls round by round):

  round 0: score --init_skill_path as-is (no edit yet) — the starting
    point every later round is judged against.
  round i (1..max_rounds): edit round (i-1)'s skill file against round
    (i-1)'s own scores/mistakes -> round i's skill file; score it.

Each round writes its own checkpoint + skill.md under
--output_dir/round<i>/, and appends ONE entry to --round_log (a single
JSON file, list of round records) with:
  - overall + per-task-bucket judge_accuracy
  - a MECHANICAL diff (added/removed lines, via difflib — not trusted
    to the editing model's own self-report) between round (i-1)'s and
    round i's skill text
  - the editing model's own stated CHANGELOG (what it changed and why,
    parsed from the ---CHANGELOG--- section heldout_skill_edit_prompt
    asks for) — the diff is the objective "what", the changelog is the
    model's own "why"
Resumable: a round already present in --round_log (and whose output
files still exist) is skipped on a re-run, so a --max_rounds run split
across multiple SLURM jobs just continues from the last completed
round.

After the last round, the round with the HIGHEST overall judge_accuracy
across ALL rounds (round 0 included — a loop that only makes things
worse should report round 0, not a forced "optimized" file) is copied
to --output_dir/skill_memory_best.md.

**Explicit, stated caveat (do not lose track of this)**: every round's
edit is steered DIRECTLY by this fixed eval split's own scores — this
is intentionally optimizing against the same n=199 split repeatedly,
not the lessons split. That is a real overfitting risk (acknowledged,
chosen anyway per docs/stage3_context_memory.md's Step 3b section) —
read the full --round_log trajectory, not just which round "won", and
treat the selected file as provisional until confirmed on a split this
loop never touched.

Usage:
    uv run python main.py heldout-skill-optimize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --init_skill_path src/config/skill_memory_disease-pest.md \\
        --max_rounds 10 \\
        --output_dir outputs_final/heldout_skill_optimize/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — explanations/reasoning fields can exceed the 131072-byte default on a full run.
import difflib
import json
import random
import re
import shutil
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_channel_thought
from src.common.vision_eval import safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.heldout_zeroshot import bucket_category, build_arg_parser as build_zeroshot_arg_parser, run_model as run_zeroshot

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_BUCKETS = ["disease", "pest", "deficiency", "healthy"]
_CHANGELOG_DELIM = "---CHANGELOG---"


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


def compute_overall_accuracy(rows: list[dict]) -> float | None:
    vals = [int(r["judged_correct"]) for r in rows if r.get("judged_correct") not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else None


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


def diff_skill_text(old_text: str, new_text: str) -> dict:
    """MECHANICAL diff (difflib, not the editing model's own self-report)
    between two skill.md versions — the objective "what changed" that
    sits alongside the model's own stated "why" (the parsed changelog).
    Line-based: this is a short bulleted Markdown doc, not prose, so
    line-level add/remove is already a meaningful unit of change."""
    old_lines = [l for l in old_text.splitlines() if l.strip()]
    new_lines = [l for l in new_text.splitlines() if l.strip()]
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(old_lines[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(new_lines[j1:j2])
    return {"removed_lines": removed, "added_lines": added, "n_removed": len(removed), "n_added": len(added)}


def parse_edit_output(raw: str) -> tuple[str, str, bool]:
    """(revised_md, changelog_text, changelog_parsed) — splits on
    _CHANGELOG_DELIM. changelog_parsed=False means the model didn't
    follow the format and the WHOLE output was treated as the document
    (not a silent failure — logged explicitly so a round with no real
    changelog is distinguishable from a round where parsing broke)."""
    if _CHANGELOG_DELIM in raw:
        doc, _, changelog = raw.partition(_CHANGELOG_DELIM)
        return doc.strip(), changelog.strip(), True
    return raw.strip(), "", False


# The exact bare-category-word failure mode v1/v2 (src/common/heldout_skill.py's
# own docstring history) found — a bullet whose ENTIRE content is just a bare
# category label, nothing else. Checked automatically every round instead of
# requiring a manual read of the full skill.md to catch a regression.
_BARE_CATEGORY_RE = re.compile(
    r"^[-*]\s*\*{0,2}(chewing pests?|sucking pests?|pest|disease|deficiency|viral infection)\*{0,2}:?\s*$",
    re.IGNORECASE,
)
_COMMON_SECTION_RE = re.compile(r"common (diseases|pests|conditions).{0,40}(to know|reference)", re.IGNORECASE)


def check_rule_violations(skill_md: str) -> list[str]:
    """Automated QC against the exact two documented historical failure
    modes (see heldout_prompts.yaml's heldout_skill_edit_prompt rules 1-2)
    — returns a list of violation strings, empty if clean. Not a full
    content-quality check, just a cheap regex tripwire for the specific,
    already-known-bad patterns, run every round so a regression shows up
    in --round_log immediately rather than requiring a manual re-read."""
    violations = []
    for i, line in enumerate(skill_md.splitlines(), start=1):
        if _BARE_CATEGORY_RE.match(line.strip()):
            violations.append(f"line {i}: bare-category-word bullet: {line.strip()!r}")
    if _COMMON_SECTION_RE.search(skill_md):
        violations.append("a 'Common Diseases/Pests/Conditions...' style reference section reappeared")
    return violations


def score_round(model_id: str, skill_memory_path: Path, round_dir: Path, base_args: argparse.Namespace) -> tuple[list[dict], float | None, dict]:
    """Runs heldout-zeroshot's own --split eval generation+judge logic
    (via heldout_zeroshot.run_model, reused directly — not
    reimplemented) against skill_memory_path, writing its checkpoint
    under round_dir. Returns (rows, overall_accuracy, task_scores)."""
    round_args = argparse.Namespace(**vars(base_args))
    round_args.skill_memory_path = str(skill_memory_path)
    round_args.wrapper_only = False
    round_args.split = "eval"
    round_args.output_dir = str(round_dir)
    round_args.rerun = False
    rows, _ckpt_path = run_zeroshot(model_id, round_args)
    return rows, compute_overall_accuracy(rows), compute_task_scores(rows)


def edit_round(model_id: str, llm, tokenizer, current_skill_md: str, task_scores_block: str,
                miss_examples_block: str, edit_template: str, chat_template: str | None,
                args: argparse.Namespace) -> tuple[str, str, bool, str, str]:
    """ONE edit call. Returns (revised_md, changelog_text, changelog_parsed,
    thinking_trace, raw_cleaned_output) — the last two are for debugging a
    round that looks wrong (truncation, a channel-thought-leak, a model
    that ignored the format) without needing to re-run anything; callers
    should persist raw_cleaned_output to disk per round, not just the
    parsed halves, since parse_edit_output's own correctness is exactly
    the kind of thing you'd want to audit from the untouched source."""
    edit_text = edit_template.format(
        current_skill_md=current_skill_md, task_scores_block=task_scores_block,
        miss_examples_block=miss_examples_block,
    )
    hist = [{"role": "user", "content": edit_text}]
    prompt = render_chat_template(tokenizer, hist, enable_thinking=args.enable_thinking, chat_template=chat_template)
    gen_params = SamplingParams(
        temperature=0.0, max_tokens=args.edit_max_tokens, seed=args.seed,
        skip_special_tokens=not args.enable_thinking,
    )
    out = llm.generate([prompt], sampling_params=gen_params)[0]
    raw = out.outputs[0].text
    thinking, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
    revised_md, changelog, changelog_parsed = parse_edit_output(cleaned)
    return revised_md, changelog, changelog_parsed, thinking, cleaned


def run_loop(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.heldout_prompts))
    edit_template = hprompts["heldout_skill_edit_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    round_log_path = Path(args.round_log) if args.round_log else out_dir / "round_log.json"
    history: list[dict] = json.loads(round_log_path.read_text()) if round_log_path.exists() else []
    completed_rounds = {h["round"] for h in history}

    # Base args for score_round — every heldout-zeroshot default
    # (gen_max_tokens, judge settings, chat template, etc.) comes from
    # its OWN build_arg_parser(), not hand-duplicated here, so the two
    # scripts can never silently drift apart on defaults.
    base_args = build_zeroshot_arg_parser().parse_args([])
    base_args.model = model_id
    for flag in ("manifest", "images_root", "prompts", "chat_template_path", "gen_max_tokens",
                 "chunk_size", "quantize", "gpu_memory_utilization", "max_model_len",
                 "vision_token_budget", "max_num_seqs", "enforce_eager", "kv_cache_dtype",
                 "enable_thinking", "skip_judge", "judge_model", "judge_quantize",
                 "judge_confidence_threshold", "judge_chunk_size", "judge_enable_thinking",
                 "judge_max_tokens", "judge_chat_template_path"):
        if hasattr(args, flag):
            setattr(base_args, flag, getattr(args, flag))

    current_skill_path = Path(args.init_skill_path)
    current_skill_text = current_skill_path.read_text(encoding="utf-8").strip()

    for round_i in range(args.max_rounds + 1):  # round 0 = score the initial file as-is, no edit
        round_dir = out_dir / f"round{round_i}"
        round_skill_path = round_dir / "skill_memory.md"

        if round_i in completed_rounds and round_skill_path.exists():
            print(f"  [round {round_i}] already completed — skipping scoring/edit (resume)")
            current_skill_text = round_skill_path.read_text(encoding="utf-8").strip()
            current_skill_path = round_skill_path
            prior = next(h for h in history if h["round"] == round_i)
            # Reload this round's checkpoint rows so the NEXT round's edit
            # step still has what it needs (miss examples, task scores) —
            # without this, resuming mid-loop silently edits with an empty
            # miss-examples block on the first post-resume round.
            ckpt_path = round_dir / safe_model_name(model_id) / f"ckpt_{safe_model_name(model_id)}.csv"
            if ckpt_path.exists():
                with open(ckpt_path, newline="", encoding="utf-8") as f:
                    prev_rows = list(csv.DictReader(f))
            else:
                prev_rows = []
            prev_task_scores = prior["task_scores"]
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === round {round_i}/{args.max_rounds} ===")
        changelog_parsed = True
        thinking_trace = ""
        if round_i == 0:
            # Round 0's "skill file" IS the init file, copied in place so
            # every round (0 included) has a uniform round<i>/skill_memory.md
            # location to select the winner from later.
            shutil.copy(current_skill_path, round_skill_path)
            changelog = "(round 0 — initial file, not yet edited)"
            diff_summary = {"removed_lines": [], "added_lines": [], "n_removed": 0, "n_added": 0}
            print(f"  Scoring initial file as-is: {round_skill_path}")

        if round_i > 0:
            prev_scores_block = build_task_scores_block(prev_task_scores)
            prev_misses = sample_miss_examples(prev_rows, args.n_examples_per_bucket, args.seed + round_i)
            miss_examples_block = build_miss_examples_block(prev_misses)
            n_examples = sum(len(v) for v in prev_misses.values())
            print(f"  [edit] {n_examples} miss example(s) from round {round_i - 1} shown to the editor")

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
            revised_md, changelog, changelog_parsed, thinking_trace, raw_cleaned = edit_round(
                model_id, llm, tokenizer, current_skill_text, prev_scores_block, miss_examples_block,
                edit_template, chat_template, args,
            )
            destroy_llm(llm)
            del tokenizer

            # Raw (pre-parse) output saved to disk EVERY round, not just the
            # parsed halves — if a round looks wrong (truncated, the
            # changelog delimiter missing, a bare-category regression),
            # this is the untouched source to check parse_edit_output's own
            # correctness against, without re-running anything.
            (round_dir / "edit_raw_output.txt").write_text(raw_cleaned)
            if thinking_trace:
                (round_dir / "edit_thinking.txt").write_text(thinking_trace)
            if not changelog_parsed:
                print(f"  [edit] WARNING: '{_CHANGELOG_DELIM}' not found in the model's output — "
                      f"whole response treated as the document, no changelog captured this round.")

            diff_summary = diff_skill_text(current_skill_text, revised_md)
            round_skill_path.write_text(revised_md)
            current_skill_text = revised_md
            print(f"  [edit] +{diff_summary['n_added']} / -{diff_summary['n_removed']} line(s) vs. round {round_i - 1}")

        rule_violations = check_rule_violations(current_skill_text)
        if rule_violations:
            print(f"  [QC] {len(rule_violations)} rule violation(s) in this round's skill.md:")
            for v in rule_violations:
                print(f"    - {v}")

        rows, overall_acc, task_scores = score_round(model_id, round_skill_path, round_dir, base_args)
        n_judged_total = sum(1 for r in rows if r.get("judged_correct") not in (None, ""))
        acc_str = f"{overall_acc:.4f}" if overall_acc is not None else "n/a"
        print(f"  [round {round_i}] overall judge_accuracy={acc_str} (n_judged={n_judged_total}/{len(rows)})")
        if n_judged_total < len(rows):
            print(f"  [QC] WARNING: only {n_judged_total}/{len(rows)} rows judged this round — "
                  f"this round's accuracy is NOT computed over the same row count as other rounds, "
                  f"comparisons against it may be misleading.")
        for b in _BUCKETS:
            s = task_scores[b]
            if s["n"]:
                print(f"    {b}: {s['accuracy']:.4f} (n={s['n']})")

        history = [h for h in history if h["round"] != round_i]
        history.append({
            "round": round_i,
            "skill_path": str(round_skill_path),
            "overall_judge_accuracy": overall_acc,
            "n_rows_total": len(rows),
            "n_judged_total": n_judged_total,
            "task_scores": task_scores,
            "diff_vs_previous_round": diff_summary,
            "changelog": changelog,
            "changelog_parsed": changelog_parsed,
            "rule_violations": rule_violations,
        })
        history.sort(key=lambda h: h["round"])
        round_log_path.write_text(json.dumps(history, indent=2))
        completed_rounds.add(round_i)

        prev_rows, prev_task_scores = rows, task_scores
        current_skill_path = round_skill_path

    scored = [h for h in history if h["overall_judge_accuracy"] is not None]
    if not scored:
        print("\n  No round produced a judged score — nothing to select.")
        return
    best = max(scored, key=lambda h: h["overall_judge_accuracy"])
    best_out_path = out_dir / "skill_memory_best.md"
    shutil.copy(best["skill_path"], best_out_path)
    print(f"\n  Best round: {best['round']} (overall judge_accuracy={best['overall_judge_accuracy']:.4f})")
    print(f"  Copied to: {best_out_path}")
    print(f"  Full trajectory: {round_log_path}")
    if best["round"] == 0:
        print("  [finding] round 0 (the UNEDITED starting file) won — no later round improved on it. "
              "The loop made things worse or flat every time it tried to edit.")
    if best.get("rule_violations"):
        print(f"  [QC] WARNING: the SELECTED best round has {len(best['rule_violations'])} rule "
              f"violation(s) — check them before trusting this file:")
        for v in best["rule_violations"]:
            print(f"    - {v}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    if not Path(args.init_skill_path).exists():
        print(f"ERROR: --init_skill_path {args.init_skill_path!r} does not exist.")
        sys.exit(1)
    run_loop(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--init_skill_path", required=True,
                         help="Starting skill.md — e.g. src/config/skill_memory_disease-pest.md. "
                              "Round 0 scores this file AS-IS before any editing.")
    parser.add_argument("--max_rounds", type=int, required=True,
                         help="Number of EDIT rounds after round 0 (so max_rounds=10 means 11 total "
                              "scoring passes: round 0 plus rounds 1-10).")
    parser.add_argument("--output_dir", default="outputs_final/heldout_skill_optimize/",
                         help="Each round writes to <output_dir>/round<i>/ (its own skill_memory.md + "
                              "heldout-zeroshot checkpoint). The best round's file is copied to "
                              "<output_dir>/skill_memory_best.md at the end.")
    parser.add_argument("--round_log", default=None,
                         help="Default: <output_dir>/round_log.json. A list of per-round records "
                              "(scores, mechanical diff, editor's own changelog) — read this to see "
                              "the FULL trajectory, not just which round won.")
    parser.add_argument("--n_examples_per_bucket", type=int, default=5,
                         help="Wrong-row examples sampled per task bucket and shown to each round's "
                              "edit call.")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--manifest", default="dataset/heldout_selected_images.json")
    parser.add_argument("--images_root", default="dataset/images_heldout")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096,
                         help="Forwarded to each round's heldout-zeroshot scoring pass.")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--edit_max_tokens", type=int, default=4096,
                         help="The revised document is at least as long as the input skill file.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base seed for miss-example sampling — offset by round number so "
                              "different rounds see different miss subsets even against the same "
                              "eval split.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--judge_enable_thinking", action="store_true")
    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--judge_chat_template_path", default=None)

    main(parser.parse_args())
