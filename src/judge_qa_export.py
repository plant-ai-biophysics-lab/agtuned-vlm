"""
Paper 2, Stage 2 — Judge QA review export (docs/stage2_verifier_corrections.md
is the verifier's own diagnostic doc; this is the judge-template-comparison
counterpart, no doc file yet — see the "LLM as a judge improvements" note).

Builds ONE review-ready CSV for a human reviewer to hand-check the judge's
verdicts, comparing the `current` and `hybrid` judge templates on the same
checkpoint (e.g. vision_baseline_thinking's rechecked_T0.7.csv from each
template's judge-template-sensitivity leniency run). No GPU, no model calls
— pure post-hoc sampling from two already-judged CSVs.

Three sampling pools, each stratified by task_type (--n_per_task rows per
task_type per pool, capped by however many candidates actually exist):

  targeted_hybrid  — hybrid's judged_correct_full_trajectory disagrees with
                      a cheap, deterministic text-overlap check between
                      extracted_answer and ground_truth (exact match,
                      substring, or neither). Calibrates hybrid's real
                      failure rate against a human oracle — targeted_current
                      is deliberately NOT built, since Paper 1 (Appendix
                      C.1) already established current's leniency; review
                      budget goes to the new, uncertain quantity (hybrid)
                      instead.
  disagreement     — current and hybrid give different verdicts on the same
                      row. Tells you which template is actually right more
                      often where template choice matters.
  random_control    — unconditioned sample (judged by hybrid). Baseline for
                      "is judging reliable in general," since the other two
                      pools are deliberately biased toward suspicious cases.

A row CAN appear in more than one pool (e.g. a targeted_hybrid hit that's
also a disagreement hit) — kept, not deduped away, since each pool answers
a different question and a reviewer should see the row under each lens
that flagged it. Each occurrence is its own output row, tagged by `pool`.

Both source CSVs are deduped on (agml_dataset, class_label, image_name)
before sampling (keeps the LAST-seen row per key) — judge-template-
sensitivity runs on a directory containing both a merged checkpoint and its
own already-merged shard files silently double-judge every row (see
judge_leniency.py's load_ckpt_rows / _is_excluded_ckpt, fixed this session);
deduping here means a stale duplicate can't skew which rows get sampled.

Usage:
    uv run python main.py judge-qa-export \\
        --current_csv outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8_thinking/current/leniency/rechecked_T0.7.csv \\
        --hybrid_csv outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8_thinking/hybrid/leniency/rechecked_T0.7.csv \\
        --n_per_task 50 --seed 42 \\
        --output_csv ~/Downloads/judge_qa_review.csv
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


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower()).strip()


def _overlap(extracted: str, ground_truth: str) -> str:
    """"exact_match" / "substring_match" / "none" — a cheap, deterministic
    stand-in for "does this look right," never a claim about actual
    correctness (that's what the human reviewer's own verdict is for)."""
    e, g = _normalize(extracted), _normalize(ground_truth)
    if not e or not g:
        return "none"
    if e == g:
        return "exact_match"
    if e in g or g in e:
        return "substring_match"
    return "none"


def _judged_bool(v: str) -> bool:
    return str(v).strip() in ("1", "True", "true")


def _load_dedup(path: Path) -> dict[tuple, dict]:
    """Load a rechecked_T<temp>.csv, keeping the LAST occurrence of each
    (agml_dataset, class_label, image_name) key — see module docstring for
    why duplicates can exist."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("agml_dataset", ""), r.get("class_label", ""), r.get("image_name", ""))
        by_key[key] = r
    return by_key


_FIELDNAMES = [
    "pool", "agml_dataset", "class_label", "image_name", "task_type",
    "ground_truth", "extracted_answer", "prediction", "overlap_heuristic",
    "current_judged_correct", "current_judge_reasoning",
    "hybrid_judged_correct", "hybrid_judge_reasoning",
    "reviewer_verdict", "failure_category", "which_template_wrong", "notes",
]


def _build_row(pool: str, key: tuple, current_row: dict, hybrid_row: dict, overlap_value: str = "") -> dict:
    ds, cls, img = key
    return {
        "pool": pool,
        "agml_dataset": ds, "class_label": cls, "image_name": img,
        "task_type": hybrid_row.get("task_type", ""),
        "ground_truth": hybrid_row.get("ground_truth", ""),
        "extracted_answer": hybrid_row.get("resolved_class", ""),
        "prediction": (hybrid_row.get("prediction", "") or "")[:_PRED_TRUNCATE],
        "overlap_heuristic": overlap_value,
        "current_judged_correct": current_row.get("judged_correct_full_trajectory", ""),
        "current_judge_reasoning": (current_row.get("judge_reasoning_full_trajectory", "") or "")[:_REASON_TRUNCATE],
        "hybrid_judged_correct": hybrid_row.get("judged_correct_full_trajectory", ""),
        "hybrid_judge_reasoning": (hybrid_row.get("judge_reasoning_full_trajectory", "") or "")[:_REASON_TRUNCATE],
        "reviewer_verdict": "", "failure_category": "", "which_template_wrong": "", "notes": "",
    }


def build_export(
    current_csv: Path, hybrid_csv: Path, n_per_task: int, seed: int,
) -> tuple[list[dict], dict]:
    """Returns (output_rows, stats) — stats records how many candidates
    existed per (pool, task_type) vs. how many were actually sampled, so a
    task_type/pool with fewer real candidates than n_per_task is visible
    in the printed summary rather than silently under-filled."""
    current = _load_dedup(current_csv)
    hybrid = _load_dedup(hybrid_csv)
    common_keys = sorted(set(current) & set(hybrid))

    # candidates[pool][task_type] -> list of (key, overlap_value)
    candidates: dict[str, dict[str, list[tuple[tuple, str]]]] = {
        "targeted_hybrid": defaultdict(list),
        "disagreement": defaultdict(list),
        "random_control": defaultdict(list),
    }

    for key in common_keys:
        h, c = hybrid[key], current[key]
        task_type = h.get("task_type", "")
        h_judged = _judged_bool(h.get("judged_correct_full_trajectory", ""))
        c_judged = _judged_bool(c.get("judged_correct_full_trajectory", ""))
        ov = _overlap(h.get("resolved_class", ""), h.get("ground_truth", ""))
        overlap_says_ok = ov in ("exact_match", "substring_match")

        if h_judged != overlap_says_ok:
            candidates["targeted_hybrid"][task_type].append((key, ov))
        if c_judged != h_judged:
            candidates["disagreement"][task_type].append((key, ""))
        candidates["random_control"][task_type].append((key, ""))

    output_rows: list[dict] = []
    stats: dict = {}
    for pool, by_task in candidates.items():
        stats[pool] = {}
        for task_type, cand_list in sorted(by_task.items()):
            pool_rng = random.Random(f"{seed}|{pool}|{task_type}")  # deterministic per (pool, task_type)
            shuffled = cand_list[:]
            pool_rng.shuffle(shuffled)
            sampled = shuffled[:n_per_task]
            stats[pool][task_type] = {"n_candidates": len(cand_list), "n_sampled": len(sampled)}
            for key, ov in sampled:
                output_rows.append(_build_row(pool, key, current[key], hybrid[key], overlap_value=ov))

    return output_rows, stats


def main(args: argparse.Namespace) -> None:
    output_rows, stats = build_export(
        Path(args.current_csv), Path(args.hybrid_csv), args.n_per_task, args.seed,
    )
    if not output_rows:
        print("No candidate rows found — check --current_csv/--hybrid_csv paths.")
        return

    out_path = Path(args.output_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for r in output_rows:
            w.writerow(r)

    print(f"Wrote {len(output_rows)} row(s) to {out_path}\n")
    for pool, by_task in stats.items():
        print(f"  {pool}:")
        for task_type, s in by_task.items():
            flag = "" if s["n_sampled"] == args.n_per_task else f"  [only {s['n_candidates']} candidate(s) available]"
            print(f"    {task_type}: {s['n_sampled']}/{args.n_per_task}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--current_csv",
        default="outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8_thinking/current/leniency/rechecked_T0.7.csv",
    )
    parser.add_argument(
        "--hybrid_csv",
        default="outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8_thinking/hybrid/leniency/rechecked_T0.7.csv",
    )
    parser.add_argument("--n_per_task", type=int, default=50,
                         help="Rows sampled per (pool, task_type) — capped by however many real "
                              "candidates exist for that combination.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="dataset/judge_qa_review/judge_qa_review.csv",
                         help="dataset/ (not outputs_final/) so this is git-tracked for reproducibility — "
                              "outputs_final/ is entirely gitignored (see .gitignore's \"outputs_*\").")
    main(parser.parse_args())
