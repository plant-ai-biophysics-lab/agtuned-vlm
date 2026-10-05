"""
Paper 2 Stage 5 — Does the box verifier's per-box confidence score
actually track real correctness? See docs/stage5_detection_verifier.md.

detection-box-verifier-summary's threshold sweep answers "what single
global confidence cutoff maximizes F1" — at confidence_threshold=0.0,
the sweep's own answer is "none, keep everything." That is a statement
about THIS dataset's score distribution relative to a single global
cutoff; it says nothing about whether the confidence score itself is a
real signal. A verifier whose scores correlate strongly with per-box
correctness but whose "good" and "bad" distributions overlap too much
for ANY single cutoff to cleanly separate them can legitimately sweep
to threshold=0.0 while still being a usable signal (e.g. for ranking,
for --top_k selection, or for flagging low-confidence boxes for human
review) rather than a useless one.

This script answers that directly, no GPU, no model: for every row,
match ALL of its predicted boxes against ground truth (full set,
confidence-agnostic — the same bbox_metrics.match_boxes IoU matching
every other stage uses) to label each individual predicted box
correct (matched to some GT box) or incorrect (unmatched), then
computes the point-biserial correlation between that box's own stored
"confidence" (from detection-box-verifier/detection-box-refine's
per-box grading) and its correct/incorrect label, pooled across every
box in the checkpoint. Same point_biserial helper + two-panel
distribution-split plot verifier-score-correlation-plot uses for the
classification verifier's own score, just applied at the BOX level
instead of the row level.

Usage:
    uv run python main.py detection-confidence-correlation \\
        --ckpt outputs_final/detection_box_refine/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --iou_threshold 0.3
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # Python's 131072-byte default is too small for a full
# (non-smoketest) run's pred_boxes_json/refinement_stats fields — many boxes x full
# refinement history per row routinely exceeds it, raising _csv.Error: field larger
# than field limit. 2**31-1 (not sys.maxsize) because csv's C extension passes this
# straight to a C long, which overflows on some platforms at sys.maxsize.
import json
import random

from src.common.bbox_metrics import match_boxes
from src.common.bootstrap import point_biserial

_COLOR_CORRECT = "seagreen"
_COLOR_INCORRECT = "firebrick"


def load_box_confidence_labels(ckpt_path: Path, iou_threshold: float) -> tuple[list[float], list[int]]:
    """Returns (confidences, matched_binary) — one entry per predicted box
    that carries a "confidence" value (boxes with none, e.g. a plain
    detection-multiclass checkpoint never graded, are skipped). Matching
    is confidence-AGNOSTIC (every predicted box in the row competes for a
    GT match, same as Variant A's own scoring) — the point is to label
    each box's REAL correctness independent of its own confidence score,
    then ask whether that score tracks the label."""
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    confidences: list[float] = []
    matched_binary: list[int] = []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        matches, _unmatched_pred, _unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        matched_pred_idx = {pi for pi, _gi, _iou in matches}
        for pi, box in enumerate(pred_boxes):
            conf = box.get("confidence")
            if conf is None:
                continue
            confidences.append(float(conf))
            matched_binary.append(1 if pi in matched_pred_idx else 0)
    return confidences, matched_binary


def plot_correlation(confidences: list[float], binary: list[int], iou_threshold: float,
                      out_path: Path, title: str = "", seed: int = 42) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct = [c for c, b in zip(confidences, binary) if b]
    incorrect = [c for c, b in zip(confidences, binary) if not b]
    r_value = point_biserial(confidences, binary)
    mean_correct = sum(correct) / len(correct) if correct else None
    mean_incorrect = sum(incorrect) / len(incorrect) if incorrect else None

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.5))

    groups = [incorrect, correct]
    labels = [f"Unmatched (n={len(incorrect)})", f"Matched to GT (n={len(correct)})"]
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
    ax_box.set_ylabel("per-box confidence")
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.set_title("Distribution by outcome")
    ax_box.grid(alpha=0.3, axis="y")

    bins = [i / 20 for i in range(21)]
    ax_hist.hist(incorrect, bins=bins, color=_COLOR_INCORRECT, alpha=0.5, label=labels[0], density=True)
    ax_hist.hist(correct, bins=bins, color=_COLOR_CORRECT, alpha=0.5, label=labels[1], density=True)
    if mean_incorrect is not None:
        ax_hist.axvline(mean_incorrect, color=_COLOR_INCORRECT, linestyle="--", linewidth=2)
    if mean_correct is not None:
        ax_hist.axvline(mean_correct, color=_COLOR_CORRECT, linestyle="--", linewidth=2)
    ax_hist.set_xlabel("per-box confidence")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Overlaid distribution")
    ax_hist.legend(fontsize=8, loc="upper center")
    ax_hist.grid(alpha=0.3, axis="y")

    r_str = f"{r_value:.3f}" if r_value is not None else "n/a"
    mc_str = f"{mean_correct:.3f}" if mean_correct is not None else "n/a"
    mi_str = f"{mean_incorrect:.3f}" if mean_incorrect is not None else "n/a"
    subtitle = f": {title}" if title else ""
    fig.suptitle(
        f"per-box confidence vs. GT-matched (IoU>={iou_threshold}){subtitle}\n"
        f"point-biserial r={r_str}   mean matched={mc_str}   mean unmatched={mi_str}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_boxes": len(confidences), "iou_threshold": iou_threshold, "point_biserial_r": r_value,
        "mean_confidence_matched": mean_correct, "mean_confidence_unmatched": mean_incorrect,
    }


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    confidences, binary = load_box_confidence_labels(ckpt_path, args.iou_threshold)
    if not confidences:
        raise SystemExit(f"ERROR: {ckpt_path} has no predicted boxes with a \"confidence\" field — was this "
                          f"checkpoint produced by detection-box-verifier or detection-box-refine, not a plain "
                          f"detection-baseline/detection-multiclass checkpoint?")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "confidence_correlation"
    out_path = out_dir / "box_confidence_correlation.png"
    summary = plot_correlation(confidences, binary, args.iou_threshold, out_path, title=args.title, seed=args.seed)

    json_path = out_dir / "box_confidence_correlation.json"
    json_path.write_text(json.dumps(summary, indent=2))

    r_str = f"{summary['point_biserial_r']:.3f}" if summary["point_biserial_r"] is not None else "n/a"
    print(f"  point-biserial r={r_str} (n_boxes={summary['n_boxes']}, iou_threshold={args.iou_threshold})")
    print(f"  mean confidence: matched={summary['mean_confidence_matched']:.3f} "
          f"unmatched={summary['mean_confidence_unmatched']:.3f}")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {json_path}")
    print(
        "\n  A positive r here (even a modest one) means the verifier's confidence score DOES track real "
        "correctness, independent of whether detection-box-verifier-summary's own sweep found a single global "
        "--confidence_threshold worth using — the score can still be useful for ranking/--top_k/human-review "
        "flagging even when it isn't useful as a single hard cutoff."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A detection-box-verifier or detection-box-refine checkpoint CSV — needs "
                              "per-box \"confidence\" in pred_boxes_json.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="IoU threshold used to label each box matched/unmatched against ground truth "
                              "(confidence-agnostic — every box in the row competes for a match regardless "
                              "of its own score). Match this to whatever --iou_threshold you settled on via "
                              "detection-box-verifier-summary/detection-final-metrics.")
    parser.add_argument("--out_dir", default=None, help="Default: <ckpt's own directory>/confidence_correlation/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    parser.add_argument("--seed", type=int, default=42, help="Jitter seed for the strip plot (cosmetic only).")
    main(parser.parse_args())
