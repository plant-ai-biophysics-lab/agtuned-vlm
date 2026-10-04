"""
Paper 2, Stage 4 — comparison bar chart: vision-baseline vs.
--verifier_type top3 --top3_mode verifier_rubric (round 1 only) vs. that
same checkpoint after verifier-cot-top3-refine. No GPU, no model calls —
pure post-hoc comparison of three already-judged checkpoint CSVs.

--baseline_ckpt is FILTERED to just the agml_dataset(s) actually present
in --verifier_rubric_ckpt (the smoketest's own dataset scope) — comparing
against the baseline's full dataset coverage would conflate "the
smoketest only touched 3 datasets" with a real performance gap.

Metric: judge_accuracy (fraction of judged_correct == 1 among judged
rows) — the same metric this stage has been reading as the trustworthy
one throughout (not the strict resolve_class-based `accuracy`, which
undercounts on free-text-vs-class-list naming mismatches). `accuracy` is
still reported in the JSON summary for reference.

Usage:
    uv run python main.py verifier-cot-comparison-summary \\
        --baseline_ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --verifier_rubric_ckpt outputs_final/verifier_cot_top3_verifier_rubric_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv \\
        --refined_ckpt outputs_final/verifier_cot_top3_verifier_rubric_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json

# Red reserved entirely for the baseline reference line — same
# categorical-color convention detection_comparison_summary.py uses.
_CONDITION_COLORS = {
    "Top3 verifier_rubric (round 1)": "dodgerblue",
    "Top3 verifier_rubric + refine": "mediumorchid",
}


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_accuracy_metrics(rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("judged_correct") not in (None, "")]
    resolvable = [r for r in rows if r.get("correct") not in (None, "")]
    judge_accuracy = sum(int(r["judged_correct"]) for r in judged) / len(judged) if judged else 0.0
    accuracy = sum(int(r["correct"]) for r in resolvable) / len(resolvable) if resolvable else 0.0
    return {
        "n": len(rows), "n_judged": len(judged),
        "judge_accuracy": round(judge_accuracy, 4), "accuracy": round(accuracy, 4),
    }


def plot_comparison(baseline: dict, conditions: dict[str, dict], out_path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_names = list(conditions.keys())
    vals = [conditions[c]["judge_accuracy"] for c in condition_names]
    colors = [_CONDITION_COLORS.get(c, "gray") for c in condition_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(condition_names, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=10)

    ax.axhline(baseline["judge_accuracy"], color="crimson", linestyle="--", linewidth=2, zorder=5,
               label=f"Baseline (reference) = {baseline['judge_accuracy']:.3f}")

    ax.set_ylabel("judge_accuracy")
    ax.set_ylim(0, 1.05)
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Top3 verifier_rubric vs. refinement vs. baseline{subtitle}\n"
                 f"(n={conditions[condition_names[0]]['n']}, baseline n={baseline['n']})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    baseline_rows = load_rows(Path(args.baseline_ckpt))
    verifier_rubric_rows = load_rows(Path(args.verifier_rubric_ckpt))
    refined_rows = load_rows(Path(args.refined_ckpt))
    if not baseline_rows:
        raise SystemExit(f"ERROR: {args.baseline_ckpt} has no rows.")
    if not verifier_rubric_rows:
        raise SystemExit(f"ERROR: {args.verifier_rubric_ckpt} has no rows.")
    if not refined_rows:
        raise SystemExit(f"ERROR: {args.refined_ckpt} has no rows.")

    smoketest_datasets = {r["agml_dataset"] for r in verifier_rubric_rows}
    baseline_filtered = [r for r in baseline_rows if r["agml_dataset"] in smoketest_datasets]
    missing = smoketest_datasets - {r["agml_dataset"] for r in baseline_filtered}
    if missing:
        print(f"  [note] {sorted(missing)} present in --verifier_rubric_ckpt but not found in "
              f"--baseline_ckpt — baseline comparison excludes them.")
    if not baseline_filtered:
        raise SystemExit(f"ERROR: none of {sorted(smoketest_datasets)} found in --baseline_ckpt "
                          f"{args.baseline_ckpt} — is this the right baseline checkpoint?")

    baseline = compute_accuracy_metrics(baseline_filtered)
    conditions = {
        "Top3 verifier_rubric (round 1)": compute_accuracy_metrics(verifier_rubric_rows),
        "Top3 verifier_rubric + refine": compute_accuracy_metrics(refined_rows),
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.refined_ckpt).parent / "comparison"
    out_path = out_dir / "comparison.png"
    plot_comparison(baseline, conditions, out_path, title=args.title)

    result = {"baseline_datasets": sorted(smoketest_datasets), "baseline": baseline, **conditions}
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"  Baseline (filtered to {sorted(smoketest_datasets)}): "
          f"judge_accuracy={baseline['judge_accuracy']:.3f} accuracy={baseline['accuracy']:.3f} (n={baseline['n']})")
    for name, m in conditions.items():
        print(f"  {name:<32s} judge_accuracy={m['judge_accuracy']:.3f} accuracy={m['accuracy']:.3f} (n={m['n']})")
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_ckpt", required=True,
                         help="vision-baseline checkpoint — filtered down to just the datasets "
                              "present in --verifier_rubric_ckpt before computing its metrics.")
    parser.add_argument("--verifier_rubric_ckpt", required=True,
                         help="A graded --verifier_type top3 --top3_mode verifier_rubric checkpoint "
                              "(round 1, before refinement) — also defines which datasets the "
                              "baseline comparison is filtered to.")
    parser.add_argument("--refined_ckpt", required=True,
                         help="That same checkpoint after verifier-cot-top3-refine.")
    parser.add_argument("--out_dir", default=None, help="Default: <refined_ckpt's own dir>/comparison/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
