"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 3b —
automated self-optimization loop for the DETECTION skill-memory file.
Detection analog of heldout_skill_optimize.py (Stage 3's own Step 3b) —
same TRAIN/VAL loop structure, same round-record/resume/selection
machinery, same ---CHANGELOG--- contract (parse_edit_output/
diff_skill_text imported directly, not reimplemented), swapped for
detection's own scoring/bucketing:

  - score_round calls heldout_detection_zeroshot.run_model (not
    heldout_zeroshot.run_model) — box-level IoU-matched precision/
    recall/F1 (deterministic, no judge) instead of judge_accuracy.
  - "task buckets" are Step 2's own error_type taxonomy (false_positive/
    missed/loose_match/worked — bbox_metrics.match_boxes applied fresh
    each round via enumerate_items, imported from
    heldout_detection_explain_traces.py, not reimplemented) instead of
    disease/pest/deficiency/healthy.
  - miss/worked examples shown to the edit call are box-level structural
    facts (predicted/ground-truth coordinates, category, IoU) — NOT a
    written explanation the way classification's examples carry one,
    since there's no per-round "explain traces" pass here (that's Step
    2's own separate, one-time, GPU-heavy script — re-running it every
    optimization round would defeat the point of a cheap scoring loop).
  - the edit prompt (detection_skill_edit_prompt,
    src/config/detection_skill_prompts.yaml) carries detection's own
    VISUAL/SPATIAL-pattern-not-coordinate-number rule and "no bare
    Common Objects list" rule instead of classification's
    specific-diagnosis-name rule.
  - check_rule_violations is DETECTION-specific (channel-leak detection
    reused as-is; "Common Objects/Classes to Know" bare-list detection;
    a NEW heuristic for a likely-literal-coordinate bullet — detection's
    own per-image-memorization failure mode, not classification's
    bare-category-word one).

TRAIN/VAL separation, resumability, and VAL-only round selection are
identical in spirit to Stage 3's own loop — see that script's own
docstring for the full "why" (not re-litigated here).

Usage:
    uv run python main.py heldout-detection-skill-optimize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --init_skill_path outputs_final/heldout_detection_summarize/skill_memory_detection.md \\
        --max_rounds 10 \\
        --output_dir outputs_final/heldout_detection_skill_optimize/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — pred_boxes_json/gt_boxes_json can exceed the 131072-byte default on a full run.
import json
import random
import re
import shutil
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_channel_thought
from src.common.vision_eval import safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.heldout_detection_explain_traces import enumerate_items
from src.heldout_detection_zeroshot import build_arg_parser as build_det_zeroshot_arg_parser, \
    run_model as run_det_zeroshot
from src.heldout_skill_optimize import diff_skill_text, parse_edit_output

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_ERROR_TYPES = ("false_positive", "missed", "loose_match", "worked")
_CHANGELOG_DELIM = "---CHANGELOG---"


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def compute_overall_f1(rows: list[dict]) -> float | None:
    """Macro-average of each row's own f1_at_iou50 — same convention
    heldout_detection_ablation_summary.py's own compute_metrics uses."""
    vals = [float(r["f1_at_iou50"]) for r in rows if r.get("f1_at_iou50") not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_error_type_breakdown(rows: list[dict], iou_threshold: float, loose_iou_threshold: float) -> dict:
    """{error_type: {"count": n, "frac": n/total}} — fresh box-level
    classification of THIS round's own predictions via enumerate_items
    (imported, not reimplemented), the detection analog of
    compute_task_scores' per-bucket accuracy."""
    items = enumerate_items(rows, iou_threshold, loose_iou_threshold)
    total = len(items)
    counts = {et: sum(1 for it in items if it["error_type"] == et) for et in _ERROR_TYPES}
    return {et: {"count": counts[et], "frac": round(counts[et] / total, 4) if total else 0.0}
            for et in _ERROR_TYPES}


def build_error_type_scores_block(breakdown: dict, overall_f1: float | None) -> str:
    lines = [f"- overall F1 (macro-averaged): {overall_f1:.4f}" if overall_f1 is not None else "- overall F1: n/a"]
    for et in _ERROR_TYPES:
        b = breakdown.get(et, {"count": 0, "frac": 0.0})
        if b["count"]:
            lines.append(f"- {et}: {b['count']} box(es) ({b['frac']:.1%} of all box-level outcomes)")
    return "\n".join(lines)


def sample_error_type_examples(
    rows: list[dict], error_types: tuple[str, ...], n_per_bucket: int, seed: int, iou_threshold: float,
    loose_iou_threshold: float,
) -> dict[str, list[dict]]:
    """{error_type: [item, ...]} — up to n_per_bucket box-level items per
    error_type, sampled (not just the first N) from a FRESH enumerate_items
    pass over this round's own rows."""
    items = enumerate_items(rows, iou_threshold, loose_iou_threshold)
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["error_type"] in error_types:
            by_type[it["error_type"]].append(it)
    return {et: rng.sample(by_type.get(et, []), min(n_per_bucket, len(by_type.get(et, []))))
            for et in error_types}


def build_examples_block(examples_by_type: dict[str, list[dict]]) -> str:
    blocks = []
    for et in _ERROR_TYPES:
        for it in examples_by_type.get(et, []):
            category = (it["gt_box"] or it["pred_box"]).get("category_name", "?")
            pred_text = f"{[round(v, 1) for v in it['pred_box']['bbox']]}" if it["pred_box"] else "(none predicted)"
            gt_text = f"{[round(v, 1) for v in it['gt_box']['bbox']]}" if it["gt_box"] else "(no real instance here)"
            blocks.append(
                f"[{et}] {it['row'].get('agml_dataset', '?')} — category '{category}'. "
                f"Predicted box: {pred_text}. Ground truth box: {gt_text}. IoU={it['iou']:.2f}."
            )
    return "\n\n".join(blocks)


def build_prior_round_feedback_block(changelog: str | None, prior_breakdown: dict | None,
                                      prior_f1: float | None, enabled: bool) -> str:
    """Detection analog of heldout_skill_optimize.py's own
    build_prior_round_feedback_block — see that function's docstring for
    the full "why" (not re-litigated here). Empty unless
    --feedback_prior_round."""
    if not enabled or changelog is None or prior_breakdown is None:
        return ""
    scores_block = build_error_type_scores_block(prior_breakdown, prior_f1)
    return (
        "\nFor reference, here is what you changed in the PREVIOUS round of "
        "editing this document, and the box-level outcomes that edit actually "
        "produced (use this to judge whether your last change helped, hurt, "
        "or had no real effect before making another one):\n\n"
        f"Previous round's changelog:\n{changelog}\n\n"
        f"Resulting box-level outcomes:\n{scores_block}\n"
    )


# Reused as-is — a channel-thought leak is a model/template-level failure,
# not domain-specific (same bug class heldout_skill_optimize.py's own
# --edit_enable_thinking default-off was added to prevent).
_CHANNEL_MARKER_RE = re.compile(r"<\|channel>|<channel\|>")
# Detection's own "Common Diseases..." equivalent.
_COMMON_OBJECTS_SECTION_RE = re.compile(r"common (objects|classes|categories).{0,40}(to know|reference)", re.IGNORECASE)
# A bullet that looks like it's citing a literal pixel coordinate/offset
# instead of a visual/spatial pattern — detection's own per-image-
# memorization failure mode (Step 3's map-reduce rule, carried over here).
_COORDINATE_BULLET_RE = re.compile(
    r"^[-*]\s.*(\bmove\b.*\b(left|right|up|down|px|pixels?)\b|\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])",
    re.IGNORECASE,
)


def check_detection_rule_violations(skill_md: str) -> list[str]:
    """Detection-specific QC — see this module's own docstring for why
    classification's check_rule_violations (bare-category-word regex) is
    not reused as-is: detection's real failure modes are different
    (coordinate-citing bullets, a 'Common Objects' list), not bare
    diagnosis-category words that will never appear in this content."""
    violations: list[str] = []
    if _CHANNEL_MARKER_RE.search(skill_md):
        violations.append("channel-thought leak: literal <|channel> marker found in the document")
    for i, line in enumerate(skill_md.splitlines(), 1):
        if _COORDINATE_BULLET_RE.match(line.strip()):
            violations.append(f"line {i}: bullet looks like it cites a literal coordinate/offset, "
                               f"not a visual/spatial pattern: {line.strip()!r}")
    if _COMMON_OBJECTS_SECTION_RE.search(skill_md):
        violations.append("a 'Common Objects/Classes to Know' style reference section reappeared")
    return violations


def score_round(model_id: str, skill_memory_path: Path, split_dir: Path, split: str,
                 base_args: argparse.Namespace) -> tuple[list[dict], float | None, dict]:
    """Runs heldout-detection-zeroshot's own --split {lessons,eval}
    generation+scoring logic (via heldout_detection_zeroshot.run_model,
    reused directly — not reimplemented) against skill_memory_path.
    Returns (rows, overall_f1, error_type_breakdown)."""
    round_args = argparse.Namespace(**vars(base_args))
    round_args.skill_memory_path = str(skill_memory_path)
    round_args.split = split
    round_args.output_dir = str(split_dir)
    round_args.rerun = False
    rows, _ckpt_path = run_det_zeroshot(model_id, round_args)
    breakdown = compute_error_type_breakdown(rows, base_args.iou_threshold, base_args.loose_iou_threshold)
    return rows, compute_overall_f1(rows), breakdown


def edit_round(model_id: str, llm, tokenizer, current_skill_md: str, task_scores_block: str,
                miss_examples_block: str, worked_examples_block: str, prior_round_feedback_block: str,
                edit_template: str, chat_template: str | None, args: argparse.Namespace) -> tuple[str, str, bool, str, str]:
    """ONE edit call — same shape as heldout_skill_optimize.py's own
    edit_round, parse_edit_output reused directly for the ---CHANGELOG---
    split."""
    edit_text = edit_template.format(
        current_skill_md=current_skill_md, task_scores_block=task_scores_block,
        miss_examples_block=miss_examples_block, worked_examples_block=worked_examples_block,
        prior_round_feedback_block=prior_round_feedback_block,
    )
    hist = [{"role": "user", "content": edit_text}]
    prompt = render_chat_template(tokenizer, hist, enable_thinking=args.edit_enable_thinking, chat_template=chat_template)
    gen_params = SamplingParams(
        temperature=0.0, max_tokens=args.edit_max_tokens, seed=args.seed,
        skip_special_tokens=not args.edit_enable_thinking,
    )
    out = llm.generate([prompt], sampling_params=gen_params)[0]
    raw = out.outputs[0].text
    thinking, cleaned = extract_channel_thought(raw) if args.edit_enable_thinking else ("", raw)
    revised_md, changelog, changelog_parsed = parse_edit_output(cleaned)
    return revised_md, changelog, changelog_parsed, thinking, cleaned


def run_loop(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.detection_skill_prompts))
    edit_template = hprompts["detection_skill_edit_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    round_log_path = Path(args.round_log) if args.round_log else out_dir / "round_log.json"
    history: list[dict] = json.loads(round_log_path.read_text()) if round_log_path.exists() else []
    completed_rounds = {h["round"] for h in history}

    # Base args for score_round — every heldout-detection-zeroshot
    # default (gen_max_tokens, chat template, etc.) comes from its OWN
    # build_arg_parser(), not hand-duplicated here.
    base_args = build_det_zeroshot_arg_parser().parse_args([])
    base_args.model = model_id
    for flag in ("manifest", "config", "images_root", "prompts", "chat_template_path", "gen_max_tokens",
                 "quantize", "gpu_memory_utilization", "max_model_len", "vision_token_budget",
                 "max_num_seqs", "enforce_eager", "kv_cache_dtype", "enable_thinking", "iou_threshold"):
        if hasattr(args, flag):
            setattr(base_args, flag, getattr(args, flag))
    base_args.loose_iou_threshold = args.loose_iou_threshold  # not a heldout-detection-zeroshot flag — used
    # directly by score_round/compute_error_type_breakdown, not forwarded to run_det_zeroshot.

    current_skill_path = Path(args.init_skill_path)
    current_skill_text = current_skill_path.read_text(encoding="utf-8").strip()
    prev_changelog: str | None = None

    for round_i in range(args.max_rounds + 1):  # round 0 = score the initial file as-is, no edit
        round_dir = out_dir / f"round{round_i}"
        round_skill_path = round_dir / "skill_memory.md"
        train_dir = round_dir / "train"
        val_dir = round_dir / "val"

        if round_i in completed_rounds and round_skill_path.exists():
            print(f"  [round {round_i}] already completed — skipping scoring/edit (resume)")
            current_skill_text = round_skill_path.read_text(encoding="utf-8").strip()
            current_skill_path = round_skill_path
            prior = next(h for h in history if h["round"] == round_i)
            ckpt_path = train_dir / safe_model_name(model_id) / f"ckpt_{safe_model_name(model_id)}.csv"
            if ckpt_path.exists():
                with open(ckpt_path, newline="", encoding="utf-8") as f:
                    prev_train_rows = list(csv.DictReader(f))
            else:
                prev_train_rows = []
            prev_train_breakdown = prior["train_error_type_breakdown"]
            prev_train_f1 = prior["train_overall_f1"]
            prev_changelog = prior["changelog"]
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === round {round_i}/{args.max_rounds} ===")
        changelog_parsed = True
        thinking_trace = ""
        if round_i == 0:
            shutil.copy(current_skill_path, round_skill_path)
            changelog = "(round 0 — initial file, not yet edited)"
            diff_summary = {"removed_lines": [], "added_lines": [], "n_removed": 0, "n_added": 0}
            print(f"  Scoring initial file as-is: {round_skill_path}")

        if round_i > 0:
            prev_scores_block = build_error_type_scores_block(prev_train_breakdown, prev_train_f1)
            prev_misses = sample_error_type_examples(
                prev_train_rows, ("false_positive", "missed", "loose_match"), args.n_examples_per_bucket,
                args.seed + round_i, args.iou_threshold, args.loose_iou_threshold)
            miss_examples_block = build_examples_block(prev_misses)
            n_examples = sum(len(v) for v in prev_misses.values())
            prev_worked = sample_error_type_examples(
                prev_train_rows, ("worked",), args.n_examples_per_bucket, args.seed + round_i,
                args.iou_threshold, args.loose_iou_threshold)
            worked_examples_block = build_examples_block(prev_worked)
            n_worked = sum(len(v) for v in prev_worked.values())
            print(f"  [edit] {n_examples} TRAIN miss example(s) + {n_worked} TRAIN worked example(s) "
                  f"from round {round_i - 1} shown to the editor")
            prior_round_feedback_block = build_prior_round_feedback_block(
                prev_changelog, prev_train_breakdown, prev_train_f1, args.feedback_prior_round)
            if args.feedback_prior_round:
                print(f"  [edit] prior round's changelog + resulting box-level outcomes also shown to the editor")

            llm, tokenizer = load_llm(
                model_id, quantization=args.quantize,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager,
                vision_token_budget=args.vision_token_budget,
                kv_cache_dtype=args.kv_cache_dtype,
                caller_tag="heldout_detection_skill_optimize_edit",
            )
            revised_md, changelog, changelog_parsed, thinking_trace, raw_cleaned = edit_round(
                model_id, llm, tokenizer, current_skill_text, prev_scores_block, miss_examples_block,
                worked_examples_block, prior_round_feedback_block, edit_template, chat_template, args,
            )
            destroy_llm(llm)
            del tokenizer

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

        rule_violations = check_detection_rule_violations(current_skill_text)
        if rule_violations:
            print(f"  [QC] {len(rule_violations)} rule violation(s) in this round's skill.md:")
            for v in rule_violations:
                print(f"    - {v}")

        train_rows, train_f1, train_breakdown = score_round(model_id, round_skill_path, train_dir, "lessons", base_args)
        print(f"  [round {round_i}] TRAIN F1={train_f1 if train_f1 is not None else 'n/a'} (n={len(train_rows)})")

        val_rows, val_f1, val_breakdown = score_round(model_id, round_skill_path, val_dir, "eval", base_args)
        print(f"  [round {round_i}] VAL   F1={val_f1 if val_f1 is not None else 'n/a'} (n={len(val_rows)}) "
              f"<- used for round selection")

        for label, breakdown in (("TRAIN", train_breakdown), ("VAL", val_breakdown)):
            for et in _ERROR_TYPES:
                b = breakdown.get(et, {"count": 0, "frac": 0.0})
                if b["count"]:
                    print(f"    {label} {et}: {b['count']} ({b['frac']:.1%})")

        history = [h for h in history if h["round"] != round_i]
        history.append({
            "round": round_i,
            "skill_path": str(round_skill_path),
            "train_overall_f1": train_f1,
            "train_n_rows_total": len(train_rows),
            "train_error_type_breakdown": train_breakdown,
            "val_overall_f1": val_f1,
            "val_n_rows_total": len(val_rows),
            "val_error_type_breakdown": val_breakdown,
            "diff_vs_previous_round": diff_summary,
            "changelog": changelog,
            "changelog_parsed": changelog_parsed,
            "rule_violations": rule_violations,
        })
        history.sort(key=lambda h: h["round"])
        round_log_path.write_text(json.dumps(history, indent=2))
        completed_rounds.add(round_i)

        prev_train_rows, prev_train_breakdown, prev_train_f1 = train_rows, train_breakdown, train_f1
        prev_changelog = changelog
        current_skill_path = round_skill_path

    # Selection is ALWAYS by VAL F1 — TRAIN never decides the winner.
    scored = [h for h in history if h["val_overall_f1"] is not None]
    if not scored:
        print("\n  No round produced a scored VAL F1 — nothing to select.")
        return
    best = max(scored, key=lambda h: h["val_overall_f1"])
    best_out_path = out_dir / "skill_memory_best.md"
    shutil.copy(best["skill_path"], best_out_path)
    print(f"\n  Best round: {best['round']} (VAL F1={best['val_overall_f1']:.4f}, "
          f"TRAIN F1={best['train_overall_f1']})")
    print(f"  Copied to: {best_out_path}")
    print(f"  Full trajectory: {round_log_path}")
    if best["round"] == 0:
        print("  [finding] round 0 (the UNEDITED starting file) won — no later round improved on it.")
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
                         help="Starting skill.md — e.g. heldout-detection-summarize's own output "
                              "(outputs_final/heldout_detection_summarize/skill_memory_detection.md). "
                              "Round 0 scores this file AS-IS before any editing.")
    parser.add_argument("--max_rounds", type=int, required=True)
    parser.add_argument("--output_dir", default="outputs_final/heldout_detection_skill_optimize/")
    parser.add_argument("--round_log", default=None)
    parser.add_argument("--n_examples_per_bucket", type=int, default=10,
                         help="Box-level examples sampled per error_type bucket (10 = 40 total across "
                              "4 buckets) and shown to each round's edit call — same default "
                              "classification's own loop uses, for consistency.")
    parser.add_argument("--feedback_prior_round", action=argparse.BooleanOptionalAction, default=False,
                         help="Same flag/semantics as heldout-skill-optimize's own — see that "
                              "script's --help for the full explanation. Default off.")
    parser.add_argument("--detection_skill_prompts", default="src/config/detection_skill_prompts.yaml")
    parser.add_argument("--manifest", default="dataset/detection_heldout_selected_images.json")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--images_root", default="dataset/images_detection_heldout")
    parser.add_argument("--prompts", default="src/config/prompts_detection.yaml")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Controls ONLY the scoring pass — NOT the edit call "
                              "(--edit_enable_thinking, default OFF).")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096)
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="Matching threshold — same value Step 1/2's own scoring used.")
    parser.add_argument("--loose_iou_threshold", type=float, default=0.7,
                         help="A real match below this (but above --iou_threshold) is loose_match; "
                              "at or above it, worked. Same value Step 2's own default.")
    parser.add_argument("--edit_enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                         help="OFF by default — same reasoning heldout-skill-optimize's own flag "
                              "documents (document-rewrite task, not the visual task thinking was "
                              "validated for; a real bug was found from defaulting this on for an "
                              "analogous task).")
    parser.add_argument("--edit_max_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
