"""
Stage 5c — Detection QC plotting: read a detection_baseline.py (or
detection_verifier.py) checkpoint CSV and draw predicted vs. ground-truth
boxes on a sample of images per dataset, for visual sanity-checking.

This is the ONLY reliable way to catch a bbox-coordinate-alignment bug —
predicted boxes plotted against the correct-resolution original image; if
they're systematically offset/scaled, the model's coordinates are relative
to an internally-rescaled view (see src/common/vlm_load.py's
mm_processor_kwargs_for docstring) and need a rescale-back fix, not
something worth guessing about from numbers alone. Run this immediately
after a baseline checkpoint's first rows exist, on a handful of images,
BEFORE trusting any run at scale — see docs/stage5_object_detection.md.

Usage:
    uv run python main.py detection-qc-plot \\
      --ckpt outputs_final/detection_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
      --n_per_dataset 3

Stage 2.5 side-by-side mode — pass --compare_ckpt (a second
DET_CKPT_FIELDS-shaped checkpoint, e.g. the --rubric run) to plot both
conditions' predictions against the same ground truth, side by side, for
the same sampled images (only images scored by BOTH checkpoints are
eligible):
    uv run python main.py detection-qc-plot \\
      --ckpt outputs_final/detection_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
      --compare_ckpt outputs_final/detection_baseline_rubric/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
      --compare_labels Baseline,Rubric --n_per_dataset 3
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
from collections import defaultdict

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes, precision_recall_f1
from src.common.bbox_viz import plot_predictions, plot_predictions_comparison


def _filter_by_confidence(boxes: list[dict], threshold: float | None) -> list[dict]:
    """Boxes with no "confidence" key at all (a plain detection-baseline/
    detection-multiclass checkpoint, never graded) are always KEPT, not
    dropped — missing confidence means "never filtered," not "score 0."
    threshold=None (default) is a no-op, same list back unchanged."""
    if threshold is None:
        return boxes
    return [b for b in boxes if b.get("confidence") is None or b["confidence"] >= threshold]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to a DET_CKPT_FIELDS-shaped checkpoint CSV.")
    ap.add_argument("--compare_ckpt", default=None,
                     help="Stage 2.5 — a second checkpoint (e.g. the --rubric run) to plot "
                          "side by side with --ckpt against the same ground truth.")
    ap.add_argument("--compare_labels", default="Condition A,Condition B",
                     help="Comma-separated panel titles for --ckpt,--compare_ckpt.")
    ap.add_argument("--images_root", default="dataset/images_detection")
    ap.add_argument("--out_dir", default=None,
                     help="Default: <ckpt's own directory>/qc_plots/")
    ap.add_argument("--n_per_dataset", type=int, default=3, help="Sample images per dataset.")
    ap.add_argument("--confidence_threshold", type=float, default=None,
                     help="Drop predicted boxes with confidence below this before plotting — for a "
                          "detection-box-refine/detection-box-verifier checkpoint's own per-box "
                          "confidence. Boxes with no confidence at all (a plain detection-baseline/"
                          "detection-multiclass checkpoint, never graded) are always kept, not "
                          "dropped. Off by default (plots every box, same as before this flag "
                          "existed). When set, the title's P/R/matched numbers are RECOMPUTED on "
                          "the filtered set (at --iou_threshold), not the checkpoint's own stored "
                          "(unfiltered) columns — so what's printed always matches what's drawn.")
    ap.add_argument("--iou_threshold", type=float, default=0.5,
                     help="Only used to recompute P/R/matched for the title when "
                          "--confidence_threshold filters the box set — the drawn boxes themselves "
                          "don't depend on this.")
    ap.add_argument("--color_by_subclass", action="store_true",
                     help="Color each predicted box by its own category_name value instead of a "
                          "flat lime — for detection-multiclass checkpoints, where category_name "
                          "already carries a sub-class/condition (e.g. \"flower: bud\" vs "
                          "\"flower: open flower\"), not just the dataset's one coarse class. "
                          "Ground truth stays dashed red either way. Off by default — a plain "
                          "detection-baseline checkpoint's category_name is just the dataset's one "
                          "class repeated on every box, so this would add a legend with no real "
                          "information.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: {ckpt_path} has no rows.")
        sys.exit(1)

    compare_rows_by_key: dict[tuple[str, str], dict] | None = None
    if args.compare_ckpt:
        compare_path = Path(args.compare_ckpt)
        with open(compare_path, newline="", encoding="utf-8") as f:
            compare_rows = list(csv.DictReader(f))
        if not compare_rows:
            print(f"ERROR: {compare_path} has no rows.")
            sys.exit(1)
        compare_rows_by_key = {(r["agml_dataset"], r["image_name"]): r for r in compare_rows}
        rows = [r for r in rows if (r["agml_dataset"], r["image_name"]) in compare_rows_by_key]
        if not rows:
            print("ERROR: --ckpt and --compare_ckpt share no overlapping (dataset, image) pairs.")
            sys.exit(1)
    label_a, label_b = (args.compare_labels.split(",", 1) + ["Condition B"])[:2]

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "qc_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dataset[r["agml_dataset"]].append(r)

    rng = random.Random(args.seed)
    n_plotted, n_missing_image = 0, 0
    for dataset, ds_rows in sorted(by_dataset.items()):
        sample = rng.sample(ds_rows, min(args.n_per_dataset, len(ds_rows)))
        ds_out = out_dir / dataset
        for r in sample:
            img_path = Path(args.images_root) / dataset / r["image_name"]
            if not img_path.exists():
                print(f"  [skip] {dataset}/{r['image_name']} — image not found at {img_path}")
                n_missing_image += 1
                continue
            image = PILImage.open(img_path).convert("RGB")
            pred_boxes = _filter_by_confidence(json.loads(r.get("pred_boxes_json") or "[]"),
                                                args.confidence_threshold)
            gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
            out_path = ds_out / f"{Path(r['image_name']).stem}.png"

            if compare_rows_by_key is not None:
                r_b = compare_rows_by_key[(dataset, r["image_name"])]
                pred_boxes_b = _filter_by_confidence(json.loads(r_b.get("pred_boxes_json") or "[]"),
                                                      args.confidence_threshold)
                if args.confidence_threshold is not None:
                    m_a = precision_recall_f1(*match_boxes(pred_boxes, gt_boxes, args.iou_threshold))
                    m_b = precision_recall_f1(*match_boxes(pred_boxes_b, gt_boxes, args.iou_threshold))
                    title = (f"{dataset} / {r['image_name']} (conf>={args.confidence_threshold})\n"
                             f"{label_a}: P={m_a['precision']} R={m_a['recall']} "
                             f"| {label_b}: P={m_b['precision']} R={m_b['recall']}")
                else:
                    title = (f"{dataset} / {r['image_name']}\n"
                             f"{label_a}: P={r.get('precision_at_iou50', '?')} R={r.get('recall_at_iou50', '?')} "
                             f"| {label_b}: P={r_b.get('precision_at_iou50', '?')} R={r_b.get('recall_at_iou50', '?')}")
                plot_predictions_comparison(image, pred_boxes, pred_boxes_b, gt_boxes,
                                             label_a, label_b, out_path=out_path, title=title,
                                             color_by_subclass=args.color_by_subclass)
            else:
                if args.confidence_threshold is not None:
                    m = precision_recall_f1(*match_boxes(pred_boxes, gt_boxes, args.iou_threshold))
                    title = (f"{dataset} / {r['image_name']} (conf>={args.confidence_threshold}, "
                             f"IoU>={args.iou_threshold})\n"
                             f"pred={m['n_pred']} gt={m['n_gt']} matched={m['n_matched']} "
                             f"P={m['precision']} R={m['recall']} IoU={m['mean_iou_matched']}")
                else:
                    title = (f"{dataset} / {r['image_name']}\n"
                             f"pred={r.get('n_pred', len(pred_boxes))} gt={r.get('n_gt', len(gt_boxes))} "
                             f"matched={r.get('n_matched', '?')} "
                             f"P={r.get('precision_at_iou50', '?')} R={r.get('recall_at_iou50', '?')} "
                             f"IoU={r.get('mean_iou_matched', '?')}")
                plot_predictions(image, pred_boxes, gt_boxes, out_path=out_path, title=title,
                                  color_by_subclass=args.color_by_subclass)
            n_plotted += 1

    print(f"\nPlotted {n_plotted} image(s) across {len(by_dataset)} dataset(s) -> {out_dir}")
    if n_missing_image:
        print(f"  {n_missing_image} row(s) skipped — cached image not found (moved/deleted "
              f"since the baseline run, or --images_root doesn't match what generated this checkpoint).")
    print("\nVisually confirm: do predicted (solid lime) boxes land on the actual objects, "
          "roughly matching ground-truth (dashed red) boxes where present? A systematic "
          "offset/scale mismatch across most images (not just occasional misses) means the "
          "model's coordinates are relative to an internally-rescaled image view, not this "
          "original-resolution one — see this script's module docstring.")


if __name__ == "__main__":
    main()
