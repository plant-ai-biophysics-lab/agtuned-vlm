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
import json
import random
from collections import defaultdict

from PIL import Image as PILImage

from src.common.bbox_viz import plot_predictions, plot_predictions_comparison


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
            pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
            gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
            out_path = ds_out / f"{Path(r['image_name']).stem}.png"

            if compare_rows_by_key is not None:
                r_b = compare_rows_by_key[(dataset, r["image_name"])]
                pred_boxes_b = json.loads(r_b.get("pred_boxes_json") or "[]")
                title = (f"{dataset} / {r['image_name']}\n"
                         f"{label_a}: P={r.get('precision_at_iou50', '?')} R={r.get('recall_at_iou50', '?')} "
                         f"| {label_b}: P={r_b.get('precision_at_iou50', '?')} R={r_b.get('recall_at_iou50', '?')}")
                plot_predictions_comparison(image, pred_boxes, pred_boxes_b, gt_boxes,
                                             label_a, label_b, out_path=out_path, title=title)
            else:
                title = (f"{dataset} / {r['image_name']}\n"
                         f"pred={r.get('n_pred', len(pred_boxes))} gt={r.get('n_gt', len(gt_boxes))} "
                         f"matched={r.get('n_matched', '?')} "
                         f"P={r.get('precision_at_iou50', '?')} R={r.get('recall_at_iou50', '?')} "
                         f"IoU={r.get('mean_iou_matched', '?')}")
                plot_predictions(image, pred_boxes, gt_boxes, out_path=out_path, title=title)
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
