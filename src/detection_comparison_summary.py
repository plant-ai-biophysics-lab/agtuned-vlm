"""
Paper 2 Stage 6-report — comparison bar chart: baseline detection vs.
verifier (no refine) vs. verifier+refine (unfiltered) vs.
verifier+refine+filter, precision/recall/F1 side by side (no GPU, no
model). Up to four conditions (--verifier_ckpt is optional — omit it to
compare just Baseline/Verifier+refine/Verifier+refine+filter), computed
at the SAME --iou_threshold so the comparison is apples-to-apples
regardless of what IoU threshold each checkpoint's own run originally
used:

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
from src.detection_final_metrics import compute_variant_b  # noqa: E402

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
    baseline_b: dict | None = None, conditions_b: dict[str, dict] | None = None,
) -> None:
    """Grouped bar chart — one group per metric (Precision/Recall/F1),
    one bar per condition within each group, fixed categorical color
    order (_CONDITION_COLORS). Baseline drawn as a short dashed red line
    segment confined to each metric's own group (not one flat line across
    the whole chart — baseline's precision/recall/F1 are 3 different
    values, so a single flat line would be wrong for at least 2 of the 3
    groups).

    baseline_b/conditions_b (optional) — the SAME shape as baseline/
    conditions but carrying Variant B (per-image ORACLE sub-class
    selection, see detection_final_metrics.py) instead of Variant A
    (deployable). When given, each condition's bar is paired with a
    second, HATCHED bar in the same hue immediately beside it (texture
    as the secondary encoding for "oracle," color stays the condition's
    own identity — never a new hue for "oracle," which would make it
    look like a 4th condition rather than a ceiling on the other three),
    and baseline gets a second dotted goldenrod reference line alongside
    its existing dashed crimson one. This answers "how far is each
    condition from its own upper bound," not just "how do the
    conditions compare to each other.\""""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_names = list(conditions.keys())
    n_cond = len(condition_names)
    show_b = conditions_b is not None
    # Each condition occupies one slot; within a slot, 1 bar (Variant A
    # only) or 2 bars side by side (A solid, B hatched) when show_b.
    slot_width = 0.8 / n_cond
    bar_width = slot_width / 2 if show_b else slot_width
    x = list(range(len(_METRICS)))

    fig, ax = plt.subplots(figsize=(11, 6.5) if show_b else (9, 6))
    for i, cond in enumerate(condition_names):
        color = _CONDITION_COLORS.get(cond, "gray")
        slot_center = [xi + (i - (n_cond - 1) / 2) * slot_width for xi in x]

        vals_a = [conditions[cond][m] for m in _METRICS]
        offsets_a = [c - bar_width / 2 for c in slot_center] if show_b else slot_center
        ax.bar(offsets_a, vals_a, width=bar_width, color=color, label=cond, zorder=3)
        for xo, v in zip(offsets_a, vals_a):
            ax.annotate(f"{v:.3f}", (xo, v), ha="center", va="bottom", fontsize=7.5, xytext=(0, 3),
                        textcoords="offset points")

        if show_b:
            vals_b = [conditions_b[cond][m] for m in _METRICS]
            offsets_b = [c + bar_width / 2 for c in slot_center]
            ax.bar(offsets_b, vals_b, width=bar_width, color=color, alpha=0.45, hatch="//",
                   edgecolor=color, linewidth=0.5, zorder=3)
            for xo, v in zip(offsets_b, vals_b):
                ax.annotate(f"{v:.3f}", (xo, v), ha="center", va="bottom", fontsize=7.5, xytext=(0, 3),
                            textcoords="offset points", color="dimgray")

    # Baseline reference line(s) — one short dashed-red segment per metric
    # group at Variant A's own baseline value, spanning just that group's
    # own bar cluster width; a second dotted-goldenrod segment for
    # Variant B's oracle baseline when show_b.
    half_span = 0.4
    for xi, m in zip(x, _METRICS):
        v = baseline[m]
        ax.plot([xi - half_span, xi + half_span], [v, v], color="crimson", linestyle="--", linewidth=2,
                zorder=5)
        ax.annotate(f"baseline={v:.3f}", (xi + half_span, v), ha="right", va="bottom", fontsize=7.5,
                    color="crimson", xytext=(0, 3), textcoords="offset points")
        if baseline_b is not None:
            vb = baseline_b[m]
            ax.plot([xi - half_span, xi + half_span], [vb, vb], color="goldenrod", linestyle=":",
                    linewidth=2, zorder=5)
            ax.annotate(f"baseline oracle={vb:.3f}", (xi - half_span, vb), ha="left", va="bottom",
                        fontsize=7.5, color="goldenrod", xytext=(0, 3), textcoords="offset points")

    ax.plot([], [], color="crimson", linestyle="--", linewidth=2, label="Baseline (reference)")
    if baseline_b is not None:
        ax.plot([], [], color="goldenrod", linestyle=":", linewidth=2, label="Baseline oracle (Variant B)")
    if show_b:
        ax.bar([], [], color="white", edgecolor="black", hatch="//", label="Oracle upper bound (Variant B)")
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
    refine_rows = load_rows(Path(args.refine_ckpt))
    if not baseline_rows:
        raise SystemExit(f"ERROR: {args.baseline_ckpt} has no rows.")
    if not refine_rows:
        raise SystemExit(f"ERROR: {args.refine_ckpt} has no rows.")

    baseline = compute_overall_metrics(baseline_rows, None, args.iou_threshold)
    conditions = {}
    if args.verifier_ckpt:
        verifier_rows = load_rows(Path(args.verifier_ckpt))
        if not verifier_rows:
            raise SystemExit(f"ERROR: {args.verifier_ckpt} has no rows.")
        conditions["Verifier (no refine)"] = compute_overall_metrics(
            verifier_rows, args.confidence_threshold, args.iou_threshold)
    conditions["Verifier + refine"] = compute_overall_metrics(refine_rows, 0.0, args.iou_threshold)
    conditions["Verifier + refine + filter"] = compute_overall_metrics(
        refine_rows, args.confidence_threshold, args.iou_threshold)

    baseline_b = conditions_b = None
    if args.show_variant_b:
        baseline_b = compute_variant_b(baseline_rows, 0.0, args.iou_threshold)["overall"]
        conditions_b = {}
        if args.verifier_ckpt:
            conditions_b["Verifier (no refine)"] = compute_variant_b(
                verifier_rows, args.confidence_threshold, args.iou_threshold)["overall"]
        conditions_b["Verifier + refine"] = compute_variant_b(refine_rows, 0.0, args.iou_threshold)["overall"]
        conditions_b["Verifier + refine + filter"] = compute_variant_b(
            refine_rows, args.confidence_threshold, args.iou_threshold)["overall"]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.refine_ckpt).parent / "comparison"
    suffix = "_variantB" if args.show_variant_b else ""
    out_path = out_dir / f"comparison_conf{args.confidence_threshold}_iou{args.iou_threshold}{suffix}.png"
    plot_comparison(baseline, conditions, out_path, title=args.title,
                     baseline_b=baseline_b, conditions_b=conditions_b)

    result = {
        "confidence_threshold": args.confidence_threshold, "iou_threshold": args.iou_threshold,
        "baseline": baseline, **conditions,
    }
    if args.show_variant_b:
        result["baseline_variant_b_oracle"] = baseline_b
        result["variant_b_oracle"] = conditions_b
    json_path = out_dir / f"comparison_conf{args.confidence_threshold}_iou{args.iou_threshold}{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"  Baseline:                     P={baseline['precision']:.3f} R={baseline['recall']:.3f} "
          f"F1={baseline['f1']:.3f} (n={baseline['n']})")
    for name, m in conditions.items():
        print(f"  {name:<28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['n']})")
    if args.show_variant_b:
        print(f"\n  Baseline oracle (Variant B):  P={baseline_b['precision']:.3f} R={baseline_b['recall']:.3f} "
              f"F1={baseline_b['f1']:.3f}")
        for name, m in conditions_b.items():
            print(f"  {name:<28s} oracle P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_ckpt", required=True,
                         help="detection-baseline or detection-multiclass checkpoint (no confidence "
                              "field — scored unfiltered, class-agnostic).")
    parser.add_argument("--verifier_ckpt", default=None,
                         help="detection-box-verifier checkpoint (optional — omit to compare just "
                              "Baseline / Verifier+refine / Verifier+refine+filter, e.g. when no "
                              "no-refine checkpoint is available).")
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
    parser.add_argument("--show_variant_b", action="store_true",
                         help="Also compute and overlay Variant B (per-image ORACLE sub-class "
                              "selection, see detection_final_metrics.py) for every condition — a "
                              "second hatched bar beside each condition's own Variant A bar, plus a "
                              "second dotted-goldenrod baseline reference line. Shows each condition's "
                              "own upper bound, not just how conditions compare to each other.")
    main(parser.parse_args())
