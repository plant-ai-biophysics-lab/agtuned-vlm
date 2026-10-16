"""
Paper 2, Stage 0 — Judge QA review export. See docs/stage0_judge_improvements.md's
"Judge QA review export" section.

Builds ONE review-ready CSV of DISAGREEMENTS between two judge-checkpoint runs
over the SAME classification checkpoint — e.g. the original inline-judged
`vision_baseline_thinking` checkpoint (judged under the hybrid template as it
was worded at generation time) vs. a `judge-checkpoint --rejudge` variant run
after hand-editing `judge_classification_hybrid_template`'s wording. No GPU,
no model calls — pure post-hoc sampling from two already-judged
VISION_BASELINE_CKPT_FIELDS-shaped CSVs (same schema on both sides: both are
either a vision-baseline checkpoint or a judge-checkpoint output copied from
one, so `judged_correct`/`judge_reasoning`/`raw_response`/etc. all line up by
column name).

A row qualifies if --baseline_ckpt's `judged_correct` differs from
--variant_ckpt's `judged_correct` on the same (agml_dataset, class_label,
image_name) key — i.e. the template wording change actually flipped a
verdict. Stratified by task_type (--n_per_task rows per task_type, capped by
however many disagreements actually exist).

Usage:
    uv run python main.py judge-qa-export \\
        --baseline_ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --variant_ckpt outputs_final/judge_variants/hybrid_v2/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --n_per_task 50 --seed 42 \\
        --output_csv dataset/judge_qa_review/judge_qa_review_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import random
import re
from collections import defaultdict

_PRED_TRUNCATE = 300
_REASON_TRUNCATE = 300
_RAW_RESPONSE_TRUNCATE = 2000  # longer — a thinking trace is verbose and truncates well before
                                # judge_reasoning would, so this needs real headroom to keep the
                                # actual <|channel>thought...<channel|> reasoning, not just the tail


def _judged_bool(v: str) -> bool:
    return str(v).strip() in ("1", "True", "true")


_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*(\d+)')


def _extract_confidence(judge_reasoning: str) -> str:
    """The judge's own stated confidence (1 or 2 in this repo's JSON verdict
    schema — see src/common/metrics.py's _parse_judge_response) — already
    present in judge_reasoning (the cleaned, truncated judge response text),
    just never pulled out as its own column before. Regex rather than
    json.loads since the judge's response is often markdown-fenced and/or
    not strictly valid JSON — this only needs the one field."""
    m = _CONFIDENCE_RE.search(judge_reasoning or "")
    return m.group(1) if m else ""


def _load_dedup(path: Path) -> dict[tuple, dict]:
    """Load a VISION_BASELINE_CKPT_FIELDS-shaped checkpoint CSV, keeping the
    LAST occurrence of each (agml_dataset, class_label, image_name) key — a
    checkpoint directory containing both a merged file and its own shard
    files would otherwise silently double-count a row."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("agml_dataset", ""), r.get("class_label", ""), r.get("image_name", ""))
        by_key[key] = r
    return by_key


_FIELDNAMES = [
    "agml_dataset", "class_label", "image_name", "task_type",
    "ground_truth", "extracted_answer", "prediction", "raw_response",
    "baseline_judged_correct", "baseline_judge_confidence", "baseline_judge_reasoning",
    "variant_judged_correct", "variant_judge_confidence", "variant_judge_reasoning",
    "reviewer_verdict", "visual_detail_score", "categorical_score", "failure_category",
    "which_judge_wrong", "notes",
]

# Reviewer-scoring columns — always blank in the export; documented here so
# the meaning ships with the file (a reviewer opening this CSV cold has no
# other place to read what a "3" means):
#   visual_detail_score  (1-5) — how much concrete, checkable visual evidence
#     the model's OWN reasoning actually cites (specific colors/shapes/
#     textures/measurements) vs. vague assertion. 1 = no real description
#     ("it looks diseased"), 5 = specific, diagnostic-grade detail.
#   categorical_score  (0/1/2) — how far the prediction is from the ground
#     truth, mirroring Paper 1 Appendix C.1's distinction: 0 = correct,
#     1 = specificity gap (right general category, wrong specific member —
#     e.g. "leaf spot disease" for "Cercospora"), 2 = causal-category
#     mismatch (different kingdom/cause entirely — e.g. a fungal name for
#     an insect-damage ground truth).
#   which_judge_wrong  ("baseline" / "variant" / "both" / "neither") — the
#     reviewer's own call on which of the two verdicts is actually correct.


def _build_row(key: tuple, baseline_row: dict, variant_row: dict) -> dict:
    ds, cls, img = key
    baseline_reasoning = baseline_row.get("judge_reasoning", "") or ""
    variant_reasoning = variant_row.get("judge_reasoning", "") or ""
    return {
        "agml_dataset": ds, "class_label": cls, "image_name": img,
        "task_type": variant_row.get("task_type", ""),
        "ground_truth": variant_row.get("ground_truth", ""),
        "extracted_answer": variant_row.get("resolved_class", ""),
        "prediction": (variant_row.get("prediction", "") or "")[:_PRED_TRUNCATE],
        "raw_response": (variant_row.get("raw_response", "") or "")[:_RAW_RESPONSE_TRUNCATE],
        "baseline_judged_correct": baseline_row.get("judged_correct", ""),
        "baseline_judge_confidence": _extract_confidence(baseline_reasoning),
        "baseline_judge_reasoning": baseline_reasoning[:_REASON_TRUNCATE],
        "variant_judged_correct": variant_row.get("judged_correct", ""),
        "variant_judge_confidence": _extract_confidence(variant_reasoning),
        "variant_judge_reasoning": variant_reasoning[:_REASON_TRUNCATE],
        "reviewer_verdict": "", "visual_detail_score": "", "categorical_score": "",
        "failure_category": "", "which_judge_wrong": "", "notes": "",
    }


def build_export(baseline_csv: Path, variant_csv: Path, n_per_task: int, seed: int) -> tuple[list[dict], dict]:
    """Returns (output_rows, stats) — stats records how many disagreements
    existed per task_type vs. how many were actually sampled, so a task_type
    with fewer real disagreements than n_per_task is visible in the printed
    summary rather than silently under-filled."""
    baseline = _load_dedup(baseline_csv)
    variant = _load_dedup(variant_csv)
    common_keys = sorted(set(baseline) & set(variant))

    candidates: dict[str, list[tuple]] = defaultdict(list)
    for key in common_keys:
        b, v = baseline[key], variant[key]
        if _judged_bool(b.get("judged_correct", "")) != _judged_bool(v.get("judged_correct", "")):
            candidates[v.get("task_type", "")].append(key)

    output_rows: list[dict] = []
    stats: dict = {}
    for task_type, cand_list in sorted(candidates.items()):
        task_rng = random.Random(f"{seed}|{task_type}")  # deterministic per task_type
        shuffled = cand_list[:]
        task_rng.shuffle(shuffled)
        sampled = shuffled[:n_per_task]
        stats[task_type] = {"n_candidates": len(cand_list), "n_sampled": len(sampled)}
        for key in sampled:
            output_rows.append(_build_row(key, baseline[key], variant[key]))

    return output_rows, stats


def main(args: argparse.Namespace) -> None:
    output_rows, stats = build_export(
        Path(args.baseline_ckpt), Path(args.variant_ckpt), args.n_per_task, args.seed,
    )
    if not output_rows:
        print("No disagreements found — check --baseline_ckpt/--variant_ckpt paths, or the two "
              "verdicts simply agree on every row.")
        return

    out_path = Path(args.output_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for r in output_rows:
            w.writerow(r)

    print(f"Wrote {len(output_rows)} disagreement row(s) to {out_path}\n")
    for task_type, s in stats.items():
        flag = "" if s["n_sampled"] == args.n_per_task else f"  [only {s['n_candidates']} candidate(s) available]"
        print(f"  {task_type}: {s['n_sampled']}/{args.n_per_task}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseline_ckpt", required=True,
        help="A VISION_BASELINE_CKPT_FIELDS-shaped checkpoint CSV already judged once — e.g. "
             "vision-baseline's own inline-judged checkpoint, or an earlier judge-checkpoint variant.",
    )
    parser.add_argument(
        "--variant_ckpt", required=True,
        help="A judge-checkpoint output (same underlying rows as --baseline_ckpt, re-judged under a "
             "different hybrid-template wording) — compared against --baseline_ckpt for verdict "
             "disagreements.",
    )
    parser.add_argument("--n_per_task", type=int, default=50,
                         help="Disagreement rows sampled per task_type — capped by however many real "
                              "candidates exist for that task_type.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="dataset/judge_qa_review/judge_qa_review.csv",
                         help="dataset/ (not outputs_final/) so this is git-tracked for reproducibility "
                              "— outputs_final/ is entirely gitignored (see .gitignore's \"outputs_*\"). "
                              "Name it with the variant's own version, e.g. "
                              "judge_qa_review_v2.csv for a hybrid_v2 comparison, so re-running against "
                              "a later variant never overwrites an earlier review.")
    main(parser.parse_args())
