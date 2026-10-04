"""
Paper 2 Stage 5-report — comparison bar chart: baseline detection vs.
verifier (no refine) vs. verifier+refine (unfiltered) vs.
verifier+refine+filter, precision/recall/F1 side by side (no GPU, no
model). Four conditions, computed at the SAME --iou_threshold so the
comparison is apples-to-apples regardless of what IoU threshold each
checkpoint's own run originally used:

- **Baseline** (--baseline_ckpt, detection-baseline/detection-multiclass)
  — every box, no confidence filtering (plain checkpoints have no
  per-box confidence at all). Shown as a DASHED RED REFERENCE LINE per
  metric, not a bar — the question this whole stage exists to answer is
  "does verification+refinement actually clear this," which a line you
  compare bars against shows more directly than a 4th bar would.
- **Verifier, no refine** (--verifier_ckpt, detection-box-verifier) —
  grades but never corrects anything, filtered at --confidence_threshold.
- **Verifier + refine** (--refine_ckpt, detection-box-refine) —
  crop/grade/correct-and-reverify, UNFILTERED (every surviving box kept
  regardless of confidence — confidence_threshold=0.0).
- **Verifier + refine + filter** (--refine_ckpt again) — same checkpoint,
  filtered at --confidence_threshold this time.

Usage:
    uv run python main.py detection-comparison-summary \\
        --baseline_ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --verifier_ckpt outputs_final/detection_box_verifier/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --refine_ckpt outputs_final/detection_box_refine/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --confidence_threshold 0.6 --iou_threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_metrics import match_boxes, precision_recall_f1  # noqa: E402
from src.detection_box_verifier_summary import load_rows  # noqa: E402

# Fixed categorical order for the 3 bar conditions — red is deliberately
# excluded from this set, reserved entirely for the baseline reference
# line, so a bar is never confused with the line it's being compared
# against.
_CONDITION_COLORS = {
    "Verifier (no refine)": "dodgerblue",
    "Verifier + refine": "mediumorchid",
    "Verifier + refine + filter": "darkorange",
}
_METRICS = ["precision", "recall", "f1"]
_METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}


def compute_overall_metrics(rows: list[dict], confidence_threshold: float | None, iou_threshold: float) -> dict:
    """Macro-averaged (per-row) precision/recall/F1, pooled across every
    row given — same convention every other script in this stage uses.
    confidence_threshold=None is a true no-op (baseline checkpoints have
    no "confidence" key at all); a box missing "confidence" is always
    kept regardless of the threshold value (never dropped as if scored
    0) — same semantics as detection_qc_plot.py's own _filter_by_confidence."""
    precisions, recalls, f1s = [], [], []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        if confidence_threshold is not None:
            pred_boxes = [b for b in pred_boxes
                          if b.get("confidence") is None or b["confidence"] >= confidence_threshold]
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        m = precision_recall_f1(*match_boxes(pred_boxes, gt_boxes, iou_threshold))
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
    n = len(rows)
    return {
        "precision": round(sum(precisions) / n, 4) if n else 0.0,
        "recall": round(sum(recalls) / n, 4) if n else 0.0,
        "f1": round(sum(f1s) / n, 4) if n else 0.0,
        "n": n,
    }


def plot_comparison(
    baseline: dict, conditions: dict[str, dict], out_path: Path, title: str = "",
) -> None:
    """Grouped bar chart — one group per metric (Precision/Recall/F1),
    one bar per condition within each group, fixed categorical color
    order (_CONDITION_COLORS). Baseline drawn as a short dashed red line
    segment confined to each metric's own group (not one flat line across
    the whole chart — baseline's precision/recall/F1 are 3 different
    values, so a single flat line would be wrong for at least 2 of the 3
    groups)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_names = list(conditions.keys())
    n_cond = len(condition_names)
    bar_width = 0.8 / n_cond
    x = list(range(len(_METRICS)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cond in enumerate(condition_names):
        vals = [conditions[cond][m] for m in _METRICS]
        offsets = [xi + (i - (n_cond - 1) / 2) * bar_width for xi in x]
        color = _CONDITION_COLORS.get(cond, "gray")
        bars = ax.bar(offsets, vals, width=bar_width, color=color, label=cond)
        for xo, v in zip(offsets, vals):
            ax.annotate(f"{v:.3f}", (xo, v), ha="center", va="bottom", fontsize=8, xytext=(0, 3),
                        textcoords="offset points")

    # Baseline reference line — one short dashed-red segment per metric
    # group, at that metric's own baseline value, spanning just that
    # group's own bar cluster width.
    half_span = 0.4
    for xi, m in zip(x, _METRICS):
        v = baseline[m]
        ax.plot([xi - half_span, xi + half_span], [v, v], color="crimson", linestyle="--", linewidth=2,
                zorder=5)
        ax.annotate(f"baseline={v:.3f}", (xi + half_span, v), ha="right", va="bottom", fontsize=7.5,
                    color="crimson", xytext=(0, 3), textcoords="offset points")

    ax.plot([], [], color="crimson", linestyle="--", linewidth=2, label="Baseline (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in _METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Detection pipeline comparison{subtitle}", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    baseline_rows = load_rows(Path(args.baseline_ckpt))
    verifier_rows = load_rows(Path(args.verifier_ckpt))
    refine_rows = load_rows(Path(args.refine_ckpt))
    if not baseline_rows:
        raise SystemExit(f"ERROR: {args.baseline_ckpt} has no rows.")
    if not verifier_rows:
        raise SystemExit(f"ERROR: {args.verifier_ckpt} has no rows.")
    if not refine_rows:
        raise SystemExit(f"ERROR: {args.refine_ckpt} has no rows.")

    baseline = compute_overall_metrics(baseline_rows, None, args.iou_threshold)
    conditions = {
        "Verifier (no refine)": compute_overall_metrics(verifier_rows, args.confidence_threshold, args.iou_threshold),
        "Verifier + refine": compute_overall_metrics(refine_rows, 0.0, args.iou_threshold),
        "Verifier + refine + filter": compute_overall_metrics(refine_rows, args.confidence_threshold, args.iou_threshold),
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.refine_ckpt).parent / "comparison"
    out_path = out_dir / f"comparison_conf{args.confidence_threshold}_iou{args.iou_threshold}.png"
    plot_comparison(baseline, conditions, out_path, title=args.title)

    result = {
        "confidence_threshold": args.confidence_threshold, "iou_threshold": args.iou_threshold,
        "baseline": baseline, **conditions,
    }
    json_path = out_dir / f"comparison_conf{args.confidence_threshold}_iou{args.iou_threshold}.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"  Baseline:                     P={baseline['precision']:.3f} R={baseline['recall']:.3f} "
          f"F1={baseline['f1']:.3f} (n={baseline['n']})")
    for name, m in conditions.items():
        print(f"  {name:<28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['n']})")
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_ckpt", required=True,
                         help="detection-baseline or detection-multiclass checkpoint (no confidence "
                              "field — scored unfiltered, class-agnostic).")
    parser.add_argument("--verifier_ckpt", required=True, help="detection-box-verifier checkpoint.")
    parser.add_argument("--refine_ckpt", required=True,
                         help="detection-box-refine checkpoint — used twice, once unfiltered and "
                              "once filtered.")
    parser.add_argument("--confidence_threshold", type=float, required=True,
                         help="E.g. the best value detection-box-verifier-summary's sweep reported.")
    parser.add_argument("--iou_threshold", type=float, required=True,
                         help="Applied to ALL FOUR conditions for a fair comparison, independent of "
                              "whatever IoU threshold each checkpoint's own original run used.")
    parser.add_argument("--out_dir", default=None,
                         help="Default: <refine_ckpt's own directory>/comparison/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
