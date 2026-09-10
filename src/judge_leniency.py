"""Judge leniency baseline — Phase 1 of the judge-improvements evaluation plan
("LLM as a judge improvements" note). Loads the judge model and, at EACH of
a swept set of temperatures, freshly re-judges every resolvable row TWICE —
once on the full model response (pred_key="prediction"), once on just the
already-extracted resolved_class string — then compares the two verdicts.
Same judge, same template, same temperature on both sides at a given sweep
point; only the input text differs. See src/common/leniency_analysis.py's
module docstring for why this is the correct leniency comparison (not a
judge-vs-resolve_class comparison, which conflates judge leniency with
resolve_class's own string-matching limits).

Both sides are re-judged fresh at every temperature (no reuse of the
checkpoint's already-stored judged_correct, even for temperatures close to
the pipeline's normal greedy default) so leniency_gap is computed under the
exact same sampling conditions on both sides at every sweep point — this
also makes each temperature's leniency_gap directly comparable to Phase 2's
(src/judge_consistency.py) flip_rate at that same temperature, for the
leniency-vs-instability correlation plot.

Needs a GPU/judge model — NOT a pure-CPU step. Input is an already-produced
classification checkpoint CSV (vision_baseline.py / self_cot.py /
cot_yaml.py / desc_augmented_vision_v2.py's ckpt_*.csv, or any CSV sharing
CKPT_FIELDS' judged_correct/resolved_class/resolvable/task_type columns)
that has already been through the judge phase at least once (judged_correct
populated — used only to know which rows are worth including, not reused
as a value).

Writes, per checkpoint:
  leniency_gap_T<temp>.png        — one per temperature: full-response vs.
                                     resolved-class-only judge accuracy, per
                                     task_type (resolvable rows only)
  leniency_gap_by_temperature.png — leniency_gap per task_type, grouped by
                                     temperature (signed y-axis)
  resolve_failures.txt            — how many/which fraction of rows had no
                                     resolved_class at all (excluded from
                                     every temperature's comparison)
  rechecked_T<temp>.csv           — per temperature, per-row: prediction,
                                     resolved_class, judged_correct_full_trajectory
                                     (full-response verdict at this T),
                                     judged_correct (resolved-class verdict
                                     at this T)
  leniency_gap.json               — leniency_gap by_temperature +
                                     resolve_failures, machine-readable —
                                     this is what --leniency_gap_json in
                                     src/judge_consistency.py reads

Usage:
    uv run python main.py judge-leniency \\
        --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \\
        --output_dir ablation/judge_leniency/vision_baseline_e4b/ \\
        --label "Vision Baseline — Gemma-4-E4B-it" \\
        --temperatures 0.1 0.5 1.0 \\
        --judge_model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from src.common.ablation_judge_recheck import rejudge_condition
from src.common.io import load_prompts
from src.common.leniency_analysis import (
    compute_leniency_gap, plot_leniency_gap, plot_leniency_gap_by_temperature,
    resolve_failure_report, write_resolve_failure_report,
)
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm


def load_ckpt_rows(ckpt_arg: str) -> list[dict]:
    """--ckpt accepts a single CSV path, a glob pattern, or a directory
    (globs ckpt_*.csv inside it) — concatenates every match, since a
    resumed/collated run can be split across multiple ckpt_*.csv files.

    Excludes "*.bak.csv" — vision_eval.py's --rerun archives the prior
    checkpoint as ckpt_<model>.<timestamp>.bak.csv before starting fresh, so
    a plain ckpt_*.csv glob over a model's output directory would otherwise
    pick up both the live checkpoint and its own archived predecessor and
    double-count every already-resumed row."""
    p = Path(ckpt_arg)
    if p.is_dir():
        paths = sorted(f for f in p.glob("ckpt_*.csv") if ".bak." not in f.name)
    else:
        matches = sorted(m for m in glob.glob(ckpt_arg) if ".bak." not in Path(m).name)
        paths = [Path(m) for m in matches] if matches else [p]
    if not paths:
        raise FileNotFoundError(f"No checkpoint CSV found for --ckpt {ckpt_arg!r}")

    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main(args: argparse.Namespace) -> None:
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("judged_correct", "resolved_class", "resolvable", "task_type") if c not in rows[0]]
    if missing:
        raise ValueError(
            f"--ckpt is missing expected column(s) {missing} — leniency_analysis needs "
            "CKPT_FIELDS-shaped rows (vision_baseline.py / self_cot.py / cot_yaml.py / "
            "desc_augmented_vision_v2.py's checkpoint CSV shape)."
        )
    rows = [r for r in rows if str(r.get("judged_correct", "")).strip() != ""]
    if not rows:
        raise ValueError("--ckpt has no judged rows (judged_correct is blank everywhere) — run the judge phase first.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.ckpt).stem

    # resolve_failures is computed over EVERY row (before filtering) so it
    # reports what fraction of the whole dataset gets excluded from the
    # leniency comparison below, not just a count within an already-filtered set.
    failures = resolve_failure_report(rows)
    write_resolve_failure_report(failures, output_dir / "resolve_failures.txt", model_label=label)

    resolvable_rows = [r for r in rows if str(r.get("resolvable", "0")) == "1" and r.get("resolved_class")]
    if not resolvable_rows:
        raise ValueError(
            "No rows have a non-empty resolved_class — nothing to re-judge. "
            f"See {output_dir / 'resolve_failures.txt'} for the unresolved breakdown."
        )
    print(f"  {len(resolvable_rows)}/{len(rows)} row(s) have a resolved_class "
          f"({failures['overall']['pct_unresolved']:.2f}% excluded as unresolved) — re-judging those.")

    prompts = load_prompts(Path(args.prompts))
    judge_llm, judge_tok = _load_judge_llm(
        args.judge_model, quantization=args.judge_quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
    )
    fields = ["agml_dataset", "class_label", "image_name", "task_type", "ground_truth",
              "resolved_class", "correct", "prediction",
              "judged_correct_full_trajectory", "judge_reasoning_full_trajectory",
              "judged_correct", "judge_reasoning"]

    results_by_temp: dict[float, dict] = {}
    try:
        for temperature in args.temperatures:
            full_rechecked = rejudge_condition(
                resolvable_rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                log_path=output_dir / f"full_response_T{temperature}_judge_log.jsonl",
                pred_key="prediction", template_key="judge_classification_template",
                progress_desc=f"Re-judging {label} (full response, T={temperature})",
                temperature=temperature, seed=0,
            )
            resolved_rechecked = rejudge_condition(
                resolvable_rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                log_path=output_dir / f"resolved_class_T{temperature}_judge_log.jsonl",
                pred_key="resolved_class", template_key="judge_classification_template",
                progress_desc=f"Re-judging {label} (resolved class, T={temperature})",
                temperature=temperature, seed=0,
            )

            # Same row order/length (both are resolvable_rows re-judged
            # independently) — zip by index to pair each row's two fresh
            # verdicts into the shape compute_leniency_gap expects.
            combined = []
            for base, full_r, resolved_r in zip(resolvable_rows, full_rechecked, resolved_rechecked):
                combined.append({
                    **base,
                    "judged_correct_full_trajectory": full_r["judged_correct"],
                    "judge_reasoning_full_trajectory": full_r["judge_reasoning"],
                    "judged_correct": resolved_r["judged_correct"],
                    "judge_reasoning": resolved_r["judge_reasoning"],
                })

            results = compute_leniency_gap(combined)
            results_by_temp[temperature] = results
            plot_leniency_gap(results, output_dir / f"leniency_gap_T{temperature}.png")

            with open(output_dir / f"rechecked_T{temperature}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in combined:
                    w.writerow(r)

            d = results["overall"]
            print(f"  T={temperature}: leniency_gap={d['leniency_gap']:+.4f} "
                  f"(full_response={d['full_response_rate']:.4f}, resolved_class={d['resolved_class_rate']:.4f})")
    finally:
        _destroy_judge_llm(judge_llm)
        del judge_tok

    plot_leniency_gap_by_temperature(results_by_temp, output_dir / "leniency_gap_by_temperature.png")

    (output_dir / "leniency_gap.json").write_text(json.dumps(
        {"label": label, "n_rows_total": len(rows), "n_rows_rejudged": len(resolvable_rows),
         "temperatures": args.temperatures,
         "leniency_gap_by_temperature": {str(t): d for t, d in results_by_temp.items()},
         "resolve_failures": failures},
        indent=2,
    ))

    print(f"Wrote leniency_gap_T<temp>.png, leniency_gap_by_temperature.png, resolve_failures.txt, "
          f"rechecked_T<temp>.csv, leniency_gap.json to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_leniency/")
    parser.add_argument("--label", default=None, help="Chart title / report header label (defaults to the ckpt filename).")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                         help="Swept judge temperatures — matches src/judge_consistency.py's default so "
                              "leniency_gap(T) and flip_rate(T) line up for the correlation plot. Both the "
                              "full-response and resolved-class sides are freshly re-judged at each value "
                              "(no reuse of the checkpoint's stored judged_correct).")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()
    main(args)
