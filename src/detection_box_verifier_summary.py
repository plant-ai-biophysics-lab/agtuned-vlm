"""
Paper 2 Stage 4, Step 1-report — confidence/IoU threshold sweep over an
existing detection-box-verifier checkpoint (no GPU, no model).

detection-box-verifier already grades every predicted box's confidence
ONCE and stores it on the box itself (pred_boxes_json's "confidence"
field) — --confidence_threshold at run time only decides where that
checkpoint's OWN pred_boxes_filtered_json/metrics.json drew the line.
This script re-derives precision/recall/F1 at MANY different
(confidence_threshold, iou_threshold) combinations directly from the
checkpoint's already-stored per-box confidence scores — pure CPU
re-filtering + re-matching, no new grading calls needed — to answer
"what threshold combination actually maximizes F1," instead of trusting
whatever --confidence_threshold the run happened to use.

Same macro-averaging convention as src/common/bbox_summary.py's own
compute_at_thresholds (per-image precision/recall/F1 first, then
averaged across images) — not pooled/micro-averaged — so numbers at the
checkpoint's own --confidence_threshold/--iou_threshold here match its
own metrics.json exactly, letting you trust the rest of the sweep.

Usage:
    uv run python main.py detection-box-verifier-summary \\
        --ckpt outputs_final/detection_box_verifier_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_metrics import match_boxes, precision_recall_f1  # noqa: E402

# Same 3-point IoU grid src/common/bbox_summary.py's own threshold sweep
# uses (0.3/0.5/0.75) — kept consistent rather than inventing a different
# IoU grid just for this script. Confidence gets a finer grid (0.0-0.9 in
# steps of 0.1) since that's the genuinely new dimension this checkpoint
# can actually sweep (bbox_summary's own sweep has no confidence score to
# sweep at all — detection-baseline/--rubric checkpoints never had one).
IOU_THRESHOLDS = [0.3, 0.5, 0.75]
CONFIDENCE_THRESHOLDS = [round(0.1 * i, 1) for i in range(10)]


def load_rows(ckpt_path: Path) -> list[dict]:
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_grid(
    rows: list[dict], conf_thresholds: list[float] = CONFIDENCE_THRESHOLDS,
    iou_thresholds: list[float] = IOU_THRESHOLDS,
) -> dict[tuple[float, float], dict[str, float]]:
    """{(confidence_threshold, iou_threshold): {"precision","recall","f1"}},
    macro-averaged per-row, recomputed directly from each row's stored
    pred_boxes_json (filtered by confidence) and gt_boxes_json."""
    parsed = []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        parsed.append((pred_boxes, gt_boxes))

    out: dict[tuple[float, float], dict[str, float]] = {}
    for conf_thr in conf_thresholds:
        for iou_thr in iou_thresholds:
            precisions, recalls, f1s = [], [], []
            for pred_boxes, gt_boxes in parsed:
                filtered = [b for b in pred_boxes if b.get("confidence", 0.0) >= conf_thr]
                matches, unmatched_pred, unmatched_gt = match_boxes(filtered, gt_boxes, iou_thr)
                m = precision_recall_f1(matches, unmatched_pred, unmatched_gt)
                precisions.append(m["precision"])
                recalls.append(m["recall"])
                f1s.append(m["f1"])
            n = len(precisions)
            out[(conf_thr, iou_thr)] = {
                "precision": round(sum(precisions) / n, 4) if n else 0.0,
                "recall": round(sum(recalls) / n, 4) if n else 0.0,
                "f1": round(sum(f1s) / n, 4) if n else 0.0,
            }
    return out


def plot_sweep(grid: dict[tuple[float, float], dict[str, float]], out_path: Path, title: str = "") -> tuple[float, float, dict]:
    """F1 vs. confidence_threshold, one line per IoU threshold — matches
    bbox_summary.plot_iou_thresholds's own "one line per condition" style
    rather than introducing a new heatmap convention. Each line's own
    peak is marked; the single best (confidence_threshold, iou_threshold)
    across the whole grid gets a star + printed summary. Returns
    (best_conf_thr, best_iou_thr, best_metrics) for the caller to print/
    write out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iou_thrs = sorted({k[1] for k in grid})
    conf_thrs = sorted({k[0] for k in grid})
    colors = plt.cm.viridis([i / max(len(iou_thrs) - 1, 1) for i in range(len(iou_thrs))])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    best_key = max(grid, key=lambda k: grid[k]["f1"])
    best_conf_thr, best_iou_thr = best_key
    best_metrics = grid[best_key]

    for iou_thr, color in zip(iou_thrs, colors):
        f1s = [grid[(c, iou_thr)]["f1"] for c in conf_thrs]
        ax.plot(conf_thrs, f1s, marker="o", markersize=5, linewidth=1.8, color=color,
                 label=f"IoU={iou_thr}")
        peak_idx = max(range(len(f1s)), key=lambda i: f1s[i])
        ax.annotate(f"{f1s[peak_idx]:.3f}", (conf_thrs[peak_idx], f1s[peak_idx]),
                     ha="center", va="bottom", fontsize=7, xytext=(0, 5), textcoords="offset points")

    ax.scatter([best_conf_thr], [best_metrics["f1"]], marker="*", s=300, color="crimson",
                zorder=5, label=f"best: conf≥{best_conf_thr}, IoU={best_iou_thr} (F1={best_metrics['f1']:.3f})")

    ax.set_xlabel("confidence_threshold")
    ax.set_ylabel("F1")
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Confidence/IoU threshold sweep{subtitle}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best_conf_thr, best_iou_thr, best_metrics


def write_grid_csv(grid: dict[tuple[float, float], dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence_threshold", "iou_threshold", "precision", "recall", "f1"])
        for (conf_thr, iou_thr), m in sorted(grid.items()):
            w.writerow([conf_thr, iou_thr, m["precision"], m["recall"], m["f1"]])
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    if not any(r.get("pred_boxes_json") for r in rows):
        raise SystemExit(f"ERROR: {ckpt_path} has no pred_boxes_json — is this a "
                          f"detection-box-verifier checkpoint, not a plain detection-baseline one?")

    grid = compute_grid(rows)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "threshold_sweep"
    plot_path = out_dir / "f1_vs_confidence_threshold.png"
    csv_path = out_dir / "threshold_sweep.csv"

    best_conf_thr, best_iou_thr, best_metrics = plot_sweep(grid, plot_path, title=args.title)
    write_grid_csv(grid, csv_path)

    print(f"  Saved: {plot_path}")
    print(f"\n  Best: confidence_threshold={best_conf_thr}, iou_threshold={best_iou_thr} -> "
          f"precision={best_metrics['precision']:.3f} recall={best_metrics['recall']:.3f} "
          f"f1={best_metrics['f1']:.3f} (n={len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                     help="A detection-box-verifier (BOX_VERIFIER_CKPT_FIELDS-shaped) checkpoint "
                          "CSV — needs per-box \"confidence\" in pred_boxes_json.")
    ap.add_argument("--out_dir", default=None, help="Default: <ckpt's own directory>/threshold_sweep/")
    ap.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(ap.parse_args())
