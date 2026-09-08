"""One-off retroactive correction for detection_baseline.py / detection_verifier.py
checkpoints affected by the Gemma 0-1000-normalized-box bug (see
src/common/bbox_parse.py's denormalize_box_1000/needs_1000_scale_boxes and
docs/stage5_object_detection.md).

Every checkpoint row's `raw_prediction` (or, for detection_verifier.py,
`verifier_feedback` does NOT carry the winning candidate's raw text, so this
only applies to detection_baseline.py's checkpoint, which does) is preserved
verbatim, so this reprocesses predicted boxes from that raw text using the
ACTUAL image on disk (for its real width/height) instead of re-running any
model — no GPU/inference needed, but it DOES need the real image files, so
run this on whichever machine already has dataset/images_detection/
populated (NOT the code-editing-only local dev environment this fix was
written in).

Usage:
    uv run python scripts/fix_detection_bbox_scale.py \\
        --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \\
        --images_root dataset/images_detection \\
        --model google/gemma-4-E4B-it

Writes a corrected copy alongside the original (suffixed .rescaled.csv) —
the original is never overwritten, so you can diff/compare before deciding
to replace it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes, precision_recall_f1
from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes, needs_1000_scale_boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="detection_baseline.py checkpoint to correct.")
    ap.add_argument("--images_root", default="dataset/images_detection")
    ap.add_argument("--model", required=True, help="Model id this checkpoint was generated with "
                                                     "(determines whether the 0-1000 fix applies at all).")
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not needs_1000_scale_boxes(args.model):
        print(f"  {args.model} is not a model family confirmed to need the 0-1000 rescale "
              f"(see bbox_parse.needs_1000_scale_boxes) — nothing to do.")
        return

    ckpt_path = Path(args.ckpt)
    rows = list(csv.DictReader(open(ckpt_path, encoding="utf-8")))
    print(f"  Reprocessing {len(rows)} row(s) from {ckpt_path}")

    n_fixed, n_missing_image, n_changed_metrics = 0, 0, 0
    for r in rows:
        img_path = Path(args.images_root) / r["agml_dataset"] / r["image_name"]
        if not img_path.exists():
            n_missing_image += 1
            continue
        img = PILImage.open(img_path)
        raw_boxes = extract_boxes(r.get("raw_prediction", ""))
        pred_boxes = [
            {**b, "bbox": clamp_box(denormalize_box_1000(b["bbox"], img.width, img.height), img.width, img.height)}
            for b in raw_boxes
        ]
        gt_boxes = json.loads(r["gt_boxes_json"])
        matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, args.iou_threshold)
        metrics = precision_recall_f1(matches, unmatched_pred, unmatched_gt)

        old_f1 = r.get("f1_at_iou50", "")
        r["pred_boxes_json"] = json.dumps(pred_boxes)
        r["n_pred"] = metrics["n_pred"]
        r["n_gt"] = metrics["n_gt"]
        r["n_matched"] = metrics["n_matched"]
        r["precision_at_iou50"] = metrics["precision"]
        r["recall_at_iou50"] = metrics["recall"]
        r["f1_at_iou50"] = metrics["f1"]
        r["mean_iou_matched"] = metrics["mean_iou_matched"]
        n_fixed += 1
        if old_f1 and str(round(metrics["f1"], 4)) != old_f1:
            n_changed_metrics += 1

    if n_missing_image:
        print(f"  [warn] {n_missing_image} row(s) skipped — image not found under {args.images_root} "
              f"(left unchanged in the output file)")
    print(f"  Rescaled {n_fixed} row(s), {n_changed_metrics} had a different f1_at_iou50 after correction")

    out_path = ckpt_path.with_suffix(".rescaled.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")

    judged_rows = [r for r in rows if r.get("f1_at_iou50")]
    if judged_rows:
        mean_p = sum(float(r["precision_at_iou50"]) for r in judged_rows) / len(judged_rows)
        mean_r = sum(float(r["recall_at_iou50"]) for r in judged_rows) / len(judged_rows)
        mean_f1 = sum(float(r["f1_at_iou50"]) for r in judged_rows) / len(judged_rows)
        mean_iou = sum(float(r["mean_iou_matched"]) for r in judged_rows) / len(judged_rows)
        print(f"  Corrected overall: precision={mean_p:.4f} recall={mean_r:.4f} "
              f"f1={mean_f1:.4f} mean_iou={mean_iou:.4f} (n={len(judged_rows)})")


if __name__ == "__main__":
    main()
