"""
Paper 2 Stage 5 — Final metrics at a CHOSEN (confidence_threshold,
iou_threshold) pair, over a detection-box-refine (or
detection-box-verifier) checkpoint (no GPU, no model).

detection-box-verifier-summary sweeps a whole grid and tells you the best
threshold combination; this script computes the actual deliverable
metrics at whichever combination you've settled on, in two variants:

- **Variant A — class-agnostic (all boxes, regardless of sub-class)**:
  the same matching this whole pipeline has used throughout
  (bbox_metrics.match_boxes never reads category_name) — every box that
  passes --confidence_threshold is pooled together and matched against
  ground truth, no sub-class distinction at all. This is literally one
  cell of detection-box-verifier-summary's own sweep grid, reused
  directly (compute_grid), not re-derived.
- **Variant B — per-image oracle sub-class selection (ORACLE/upper
  bound, not deployable)**: for EACH image independently, among the
  distinct category_name values present in its own filtered predictions
  (detection-multiclass's own free-text sub-class vocabulary, e.g.
  "flower: open yellow" vs "flower: bud"), try each ALONE against that
  image's own ground truth and keep whichever gives the highest F1 —
  then score the image on only that sub-class's boxes, macro-averaged
  across the dataset. Deliberately PER-IMAGE, not a per-dataset argmax
  (an earlier version was — see git history/docs/stage5_detection_verifier.md):
  free-text labels vary in wording across images for the same real
  stage ("flower: open" vs "flower: open yellow"), which fragmented a
  per-dataset grouping into many small, individually weak buckets.
  Per-image selection needs no cross-image text matching at all, since
  each image only ever compares among its own locally-present labels.
  The selection itself looks at ground truth, which nothing deployable
  could do — this is an upper bound ("what's the best possible result if
  sub-class selection were perfect"), not a real filter's achievable
  number. Compare against Variant A to see the gap between what's
  achievable today and what perfect selection could theoretically buy.

Usage:
    uv run python main.py detection-final-metrics \\
        --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --confidence_threshold 0.6 --iou_threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_metrics import match_boxes, precision_recall_f1  # noqa: E402
from src.detection_box_verifier_summary import compute_grid, load_rows  # noqa: E402


def compute_variant_b(rows: list[dict], confidence_threshold: float, iou_threshold: float) -> dict:
    """{dataset: {"precision", "recall", "f1", "n_pred", "n_gt",
    "n_matched", "chosen_subclass_counts": {subclass: n_images}}}

    PER-IMAGE oracle sub-class selection — not a per-dataset argmax (see
    module docstring for why that was replaced, 2026-10-02: free-text
    sub-class labels vary in wording ACROSS images — "flower: open" vs
    "flower: open yellow" for the same real stage — which fragmented a
    per-dataset grouping into many small, individually weak buckets).
    For EACH image independently: among the distinct category_name
    values present in its own filtered predictions, try each ALONE
    against that image's own ground truth, keep whichever gives the
    highest F1 (ties broken by higher precision), and score the image on
    ONLY that sub-class's boxes. No cross-image text comparison is ever
    needed — each image only ever compares among its own locally-present
    labels. Macro-averaged across every image in the dataset (an image
    with zero predictions still contributes, scored as an empty
    prediction set).

    ORACLE METRIC: the selection itself looks at ground truth to decide
    which boxes to keep, which nothing deployable could do — this is an
    upper bound ("what's the best possible result if sub-class selection
    were perfect"), not a number a real filter could achieve on its own.
    Compare against Variant A (ground-truth-agnostic, actually
    deployable) to see the gap between what's achievable today and what
    perfect sub-class selection could theoretically buy."""
    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["agml_dataset"], []).append(r)

    out: dict[str, dict] = {}
    for dataset, ds_rows in by_dataset.items():
        precisions, recalls, f1s = [], [], []
        n_pred_total = n_gt_total = n_matched_total = 0
        chosen_counts: dict[str, int] = {}

        for r in ds_rows:
            pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
            pred_boxes = [b for b in pred_boxes if b.get("confidence") is None or b["confidence"] >= confidence_threshold]
            gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
            subclasses = sorted({str(b.get("category_name")) for b in pred_boxes if b.get("category_name")})

            best_sub, best_m = None, None
            for sub in subclasses:
                sub_boxes = [b for b in pred_boxes if str(b.get("category_name")) == sub]
                cand_m = precision_recall_f1(*match_boxes(sub_boxes, gt_boxes, iou_threshold))
                if best_m is None or (cand_m["f1"], cand_m["precision"]) > (best_m["f1"], best_m["precision"]):
                    best_sub, best_m = sub, cand_m
            if best_m is None:
                best_m = precision_recall_f1(*match_boxes([], gt_boxes, iou_threshold))

            if best_sub is not None:
                chosen_counts[best_sub] = chosen_counts.get(best_sub, 0) + 1
            precisions.append(best_m["precision"])
            recalls.append(best_m["recall"])
            f1s.append(best_m["f1"])
            n_pred_total += best_m["n_pred"]
            n_gt_total += best_m["n_gt"]
            n_matched_total += best_m["n_matched"]

        n = len(ds_rows)
        out[dataset] = {
            "precision": round(sum(precisions) / n, 4) if n else 0.0,
            "recall": round(sum(recalls) / n, 4) if n else 0.0,
            "f1": round(sum(f1s) / n, 4) if n else 0.0,
            "n_pred": n_pred_total, "n_gt": n_gt_total, "n_matched": n_matched_total,
            "chosen_subclass_counts": dict(sorted(chosen_counts.items(), key=lambda kv: -kv[1])),
        }
    return out


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")

    # Variant A: one cell of the same sweep grid detection-box-verifier-
    # summary computes — reused directly, not re-derived, so this number
    # always matches what the sweep itself would report at this (conf,
    # iou) combination.
    grid = compute_grid(rows, conf_thresholds=[args.confidence_threshold], iou_thresholds=[args.iou_threshold])
    variant_a_overall = grid[(args.confidence_threshold, args.iou_threshold)]

    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["agml_dataset"], []).append(r)
    variant_a_per_dataset = {}
    for dataset, ds_rows in sorted(by_dataset.items()):
        ds_grid = compute_grid(ds_rows, conf_thresholds=[args.confidence_threshold], iou_thresholds=[args.iou_threshold])
        variant_a_per_dataset[dataset] = ds_grid[(args.confidence_threshold, args.iou_threshold)]

    variant_b = compute_variant_b(rows, args.confidence_threshold, args.iou_threshold)

    result = {
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "n_rows": len(rows),
        "variant_a_all_subclasses": {
            "overall": variant_a_overall,
            "per_dataset": variant_a_per_dataset,
        },
        "variant_b_oracle_subclass_per_image": variant_b,
    }

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "final_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"final_metrics_conf{args.confidence_threshold}_iou{args.iou_threshold}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  confidence_threshold={args.confidence_threshold}  iou_threshold={args.iou_threshold}")
    print()
    print(f"  Variant A (all sub-classes pooled) — overall: "
          f"P={variant_a_overall['precision']:.3f} R={variant_a_overall['recall']:.3f} "
          f"F1={variant_a_overall['f1']:.3f}")
    for dataset, m in sorted(variant_a_per_dataset.items()):
        print(f"    {dataset}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print()
    print("  Variant B (per-image oracle sub-class selection — ORACLE/upper bound, not deployable):")
    for dataset, m in sorted(variant_b.items()):
        top = next(iter(m["chosen_subclass_counts"]), None)
        top_str = f"{top!r} ({m['chosen_subclass_counts'][top]} image(s))" if top else "n/a"
        print(f"    {dataset}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
              f"most-chosen={top_str}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A detection-box-refine or detection-box-verifier checkpoint CSV — "
                              "needs per-box \"confidence\" in pred_boxes_json.")
    parser.add_argument("--confidence_threshold", type=float, required=True,
                         help="E.g. the best value detection-box-verifier-summary's sweep reported.")
    parser.add_argument("--iou_threshold", type=float, required=True,
                         help="E.g. the best value detection-box-verifier-summary's sweep reported.")
    parser.add_argument("--out_dir", default=None, help="Default: <ckpt's own directory>/final_metrics/")
    main(parser.parse_args())
