"""
Paper 2, Stage 4 — Verifier score correlation plot. See
docs/stage4_classification_verifier.md's "Read the result" sections —
until now, `verifier_score_correlation`/`top3_pool_analysis` were only a
handful of numbers in metrics.json, read by eye, no chart. No GPU, no
model calls — reads an already-judged `verifier-cot` checkpoint CSV
directly (`verifier_score`/`judged_correct` columns), recomputes
point-biserial r itself (not trusted from a possibly-stale metrics.json),
and renders:

  1. verifier_score distribution split by judged correctness (box + jittered
     strip, one point per row) — does the score actually separate correct
     from incorrect trajectories, or is it all overlapping noise?
  2. The same split as an overlaid histogram — a different view of the
     same question, useful when n is large enough that individual strip
     points stop being legible.

Both panels in ONE figure (one PNG), annotated with point-biserial r and
both group means, so a single image is the answer to "does this signal
work."

Usage:
    uv run python main.py verifier-score-correlation-plot \\
        --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import random

from src.common.bootstrap import point_biserial

# Semantic good/bad pair — reserved for exactly this correct/incorrect
# split, not a categorical hue cycled across unrelated series.
_COLOR_CORRECT = "seagreen"
_COLOR_INCORRECT = "firebrick"


def load_scored_rows(ckpt_path: Path) -> tuple[list[float], list[int], str]:
    """Returns (verifier_scores, judged_binary, judge_key) — judge_key is
    "judged_correct" if any row has one (the normal case), else falls back
    to the deterministic "correct" column (e.g. a --skip_judge run), same
    fallback verifier_cot.py's own metrics computation uses. Only rows
    with BOTH a verifier_score and a value for judge_key are kept."""
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vs_rows = [r for r in rows if r.get("verifier_score") not in (None, "")]
    judge_key = "judged_correct" if any(r.get("judged_correct") not in (None, "") for r in vs_rows) else "correct"
    scored = [r for r in vs_rows if r.get(judge_key) not in (None, "")]
    scores = [float(r["verifier_score"]) for r in scored]
    binary = [int(r[judge_key]) for r in scored]
    return scores, binary, judge_key


def plot_correlation(scores: list[float], binary: list[int], judge_key: str, out_path: Path, title: str = "",
                      seed: int = 42) -> dict:
    """Renders the two-panel figure, returns the summary dict also written
    alongside it as JSON (same fields verifier_cot.py's own
    verifier_score_correlation block carries, so the two are directly
    comparable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct_scores = [s for s, c in zip(scores, binary) if c]
    incorrect_scores = [s for s, c in zip(scores, binary) if not c]
    r_value = point_biserial(scores, binary)
    mean_correct = sum(correct_scores) / len(correct_scores) if correct_scores else None
    mean_incorrect = sum(incorrect_scores) / len(incorrect_scores) if incorrect_scores else None

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1: box + jittered strip, one point per row.
    groups = [incorrect_scores, correct_scores]
    labels = [f"Incorrect (n={len(incorrect_scores)})", f"Correct (n={len(correct_scores)})"]
    colors = [_COLOR_INCORRECT, _COLOR_CORRECT]
    bp = ax_box.boxplot(groups, tick_labels=labels, widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("black")
    rng = random.Random(seed)
    for i, (group, color) in enumerate(zip(groups, colors), start=1):
        jitter = [i + rng.uniform(-0.15, 0.15) for _ in group]
        ax_box.scatter(jitter, group, color=color, alpha=0.5, s=14, zorder=3, edgecolors="none")
    ax_box.set_ylabel("verifier_score")
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.set_title("Distribution by outcome")
    ax_box.grid(alpha=0.3, axis="y")

    # Panel 2: overlaid histogram, same two groups.
    bins = [i / 20 for i in range(21)]
    ax_hist.hist(incorrect_scores, bins=bins, color=_COLOR_INCORRECT, alpha=0.5, label=labels[0], density=True)
    ax_hist.hist(correct_scores, bins=bins, color=_COLOR_CORRECT, alpha=0.5, label=labels[1], density=True)
    if mean_incorrect is not None:
        ax_hist.axvline(mean_incorrect, color=_COLOR_INCORRECT, linestyle="--", linewidth=2)
    if mean_correct is not None:
        ax_hist.axvline(mean_correct, color=_COLOR_CORRECT, linestyle="--", linewidth=2)
    ax_hist.set_xlabel("verifier_score")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Overlaid distribution")
    ax_hist.legend(fontsize=8, loc="upper center")
    ax_hist.grid(alpha=0.3, axis="y")

    r_str = f"{r_value:.3f}" if r_value is not None else "n/a"
    mc_str = f"{mean_correct:.3f}" if mean_correct is not None else "n/a"
    mi_str = f"{mean_incorrect:.3f}" if mean_incorrect is not None else "n/a"
    subtitle = f": {title}" if title else ""
    fig.suptitle(
        f"verifier_score vs. {judge_key}{subtitle}\n"
        f"point-biserial r={r_str}   mean correct={mc_str}   mean incorrect={mi_str}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n": len(scores), "against": judge_key, "point_biserial_r": r_value,
        "mean_verifier_score_correct": mean_correct, "mean_verifier_score_incorrect": mean_incorrect,
    }


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    scores, binary, judge_key = load_scored_rows(ckpt_path)
    if not scores:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows with both verifier_score and a judged/correct "
                          f"value — was this checkpoint run with --verifier_type simple or top3, and not "
                          f"--skip_judge?")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "correlation"
    out_path = out_dir / "verifier_score_correlation.png"
    summary = plot_correlation(scores, binary, judge_key, out_path, title=args.title, seed=args.seed)

    json_path = out_dir / "verifier_score_correlation.json"
    json_path.write_text(json.dumps(summary, indent=2))

    r_str = f"{summary['point_biserial_r']:.3f}" if summary["point_biserial_r"] is not None else "n/a"
    print(f"  point-biserial r={r_str} (n={summary['n']}, against={judge_key})")
    print(f"  mean verifier_score: correct={summary['mean_verifier_score_correct']:.3f} "
          f"incorrect={summary['mean_verifier_score_incorrect']:.3f}")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A verifier-cot checkpoint CSV (VERIFIER_CKPT_FIELDS-shaped) — "
                              "ckpt_<model>.csv for --verifier_type simple, ckpt_<model>_top3.csv for top3.")
    parser.add_argument("--out_dir", default=None,
                         help="Default: <ckpt's own directory>/correlation/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    parser.add_argument("--seed", type=int, default=42, help="Jitter seed for the strip plot (cosmetic only).")
    main(parser.parse_args())
