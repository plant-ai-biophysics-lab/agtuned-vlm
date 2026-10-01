"""
Stage 4 — Detection Box Verifier: per-box confidence grading over an
existing detection-multiclass checkpoint. See
docs/stage4_detection_verifier.md's "Planned: per-box verifier" section
for the full design rationale — summarized here:

detection-multiclass (this stage's own generation step) deliberately
OVER-generates — every growth stage/condition it can see, not just
ground truth's one labeled convention. This script is what prunes that
pool back down: for each predicted box, crop it out of the source image
(padded, see crop_box) and show ONLY that crop to a verifier model —
independently of the other boxes and the full-image context, a genuine
second opinion, not the same generation call grading its own output.
Each crop is graded against src/config/verifier_prompts_detection.yaml's
box_rubric (4 criteria) independently on a 1-3 scale, summed/normalized
to a confidence in [0, 1] — same summed-rubric mechanic
src/verifier_cot.py's --verifier_type simple uses for a whole
classification trajectory (its own _parse_criterion_scores/
_normalize_criterion_scores are reused directly, unchanged), just
applied per-box instead of per-trajectory. No ground truth is shown to
the grader — purely a visual plausibility check against the box's OWN
stated category/sub-type label.

Boxes scoring below --confidence_threshold are dropped into a SEPARATE
filtered set (pred_boxes_filtered_json) before re-scoring precision/
recall/F1 against ground truth — the original unfiltered
pred_boxes_json/metrics are preserved unchanged alongside the filtered
ones, so a run's own checkpoint directly shows the before/after effect
of filtering rather than only the filtered number. Scoring itself stays
class-agnostic either way (src/common/bbox_metrics.match_boxes never
reads category_name) — this script only changes WHICH boxes make it
into that matching step, never how matching itself works.

Usage:
    uv run python main.py detection-box-verifier \\
        --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --output_dir outputs_final/detection_box_verifier_smoketest/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import time
from collections import defaultdict

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes, precision_recall_f1
from src.common.chat_prompt import build_multimodal_prompt
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.progress import ProgressBar
from src.verifier_cot import _count_rubric_criteria, _normalize_criterion_scores, _parse_criterion_scores

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_VERIFIER_PROMPTS_DETECTION_PATH = Path(__file__).parent / "config" / "verifier_prompts_detection.yaml"

# One row per (dataset, class, image) — same granularity as
# detection_baseline.py's own DET_CKPT_FIELDS, plus the filtered-set
# columns this stage adds. pred_boxes_json carries the FULL pool (every
# box the input checkpoint had), each box dict now also holding
# "confidence"/"criterion_scores"/"verifier_feedback" — gt_boxes_json and
# the plain n_pred/n_matched/precision/recall/f1/mean_iou_matched columns
# are an UNCHANGED pass-through of the input checkpoint's own numbers
# (never recomputed here), so a reader can compare them directly against
# this row's own *_filtered columns without re-deriving the "before"
# picture from a different file.
BOX_VERIFIER_CKPT_FIELDS = [
    "agml_dataset", "class_label", "task_type", "image_name",
    "pred_boxes_json", "gt_boxes_json",
    "n_pred", "n_gt", "n_matched", "precision_at_iou50", "recall_at_iou50",
    "f1_at_iou50", "mean_iou_matched",
    "pred_boxes_filtered_json",
    "n_pred_filtered", "n_matched_filtered", "precision_at_iou50_filtered",
    "recall_at_iou50_filtered", "f1_at_iou50_filtered", "mean_iou_matched_filtered",
]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def crop_box(image: PILImage.Image, bbox: list[float], pad_frac: float = 0.15) -> PILImage.Image:
    """Crop `bbox` ([x1,y1,x2,y2] pixel) out of `image`, padded by
    pad_frac * the box's own width/height on each side (NOT a fixed pixel
    count — detection images vary wildly in resolution, so a fixed pad
    would be way too generous on a small crop and negligible on a large
    one), clamped to the image's own bounds. Padding gives the verifier
    model context it needs to judge plausibility (criterion 1/4) without
    being so loose it re-includes neighboring instances (criterion 3).
    Degenerate (zero-area) input boxes are floored to a minimum 1px crop
    rather than raising, so a single malformed box can't abort a whole
    batch."""
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_x, pad_y = w * pad_frac, h * pad_frac
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(image.width, max(cx1 + 1, int(x2 + pad_x)))
    cy2 = min(image.height, max(cy1 + 1, int(y2 + pad_y)))
    return image.crop((cx1, cy1, cx2, cy2))


def grade_boxes(
    llm, tokenizer, crops: list[PILImage.Image], labels: list[str], rubric: str, grading_template: str,
    system_prompt: str, grading_max_tokens: int = 300, seed: int | None = None,
) -> list[dict]:
    """Grade every (crop, label) pair independently, batched into ONE
    llm.generate() call regardless of how many images/boxes they came
    from — same batching convention verifier_cot.py's run_group_top3 uses
    for its own per-candidate grading. Returns one {"confidence",
    "criterion_scores", "feedback"} dict per input pair, same order."""
    n_criteria = _count_rubric_criteria(rubric)
    prompts = [
        build_multimodal_prompt(
            tokenizer, crop, grading_template.format(rubric=rubric, label=label),
            system_prompt=system_prompt,
        )
        for crop, label in zip(crops, labels)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=grading_max_tokens, seed=seed)
    outputs = llm.generate(prompts, sampling_params=params)
    results = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        criterion_scores = _parse_criterion_scores(text, n_criteria)
        confidence = _normalize_criterion_scores(criterion_scores, n_criteria)
        results.append({
            "confidence": round(confidence, 4),
            "criterion_scores": criterion_scores,
            "feedback": text,
        })
    return results


def run_model(model_id: str, args: argparse.Namespace) -> None:
    vprompts = load_yaml(_VERIFIER_PROMPTS_DETECTION_PATH)
    rubric = vprompts["box_rubric"]
    grading_template = vprompts["box_grading_prompt"]
    system_prompt = vprompts.get("assessment_system", "")

    ckpt_path = Path(args.ckpt)
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    if not input_rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    out_ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(out_ckpt_path, args.rerun)
    rows, seen = load_ckpt(out_ckpt_path, fields=BOX_VERIFIER_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} rows already scored in {out_ckpt_path.name}")

    pending = [
        r for r in input_rows
        if (r["agml_dataset"], r["class_label"], r["image_name"]) not in seen
    ]
    if not pending:
        print("  Nothing to do — all rows already checkpointed.")
        return

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in pending:
        by_dataset[r["agml_dataset"]].append(r)
    print(f"  {len(by_dataset)} dataset(s), {len(pending)} pending row(s)")

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="detection_box_verifier",
    )

    t_start = time.perf_counter()
    pbar = ProgressBar(total=len(by_dataset), desc="Datasets")
    n_missing_image = 0
    for dataset, ds_rows in by_dataset.items():
        pbar.desc = f"Datasets ({dataset})"
        # Flatten every box across every row in this dataset group into
        # one (row_idx, box_idx, crop, label) list, so ALL of this
        # dataset's crops go through ONE batched llm.generate() call
        # regardless of how many boxes any individual image has.
        flat: list[tuple[int, int]] = []
        crops: list[PILImage.Image] = []
        labels: list[str] = []
        pred_boxes_per_row: list[list[dict]] = []
        for ri, r in enumerate(ds_rows):
            img_path = Path(args.images_root) / dataset / r["image_name"]
            pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
            pred_boxes_per_row.append(pred_boxes)
            if not pred_boxes:
                continue
            if not img_path.exists():
                print(f"  [skip] {dataset}/{r['image_name']} — image not found at {img_path}")
                n_missing_image += 1
                continue
            image = PILImage.open(img_path).convert("RGB")
            for bi, box in enumerate(pred_boxes):
                flat.append((ri, bi))
                crops.append(crop_box(image, box["bbox"], pad_frac=args.crop_pad_frac))
                labels.append(str(box.get("category_name") or "object"))

        if flat:
            grades = grade_boxes(
                llm, tokenizer, crops, labels, rubric, grading_template, system_prompt,
                grading_max_tokens=args.grading_max_tokens, seed=args.seed,
            )
            for (ri, bi), grade in zip(flat, grades):
                pred_boxes_per_row[ri][bi] = {
                    **pred_boxes_per_row[ri][bi],
                    "confidence": grade["confidence"],
                    "criterion_scores": grade["criterion_scores"],
                    "verifier_feedback": grade["feedback"],
                }

        for ri, r in enumerate(ds_rows):
            pred_boxes = pred_boxes_per_row[ri]
            gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
            filtered = [b for b in pred_boxes if b.get("confidence", 0.0) >= args.confidence_threshold]
            matches, unmatched_pred, unmatched_gt = match_boxes(filtered, gt_boxes, args.iou_threshold)
            filtered_metrics = precision_recall_f1(matches, unmatched_pred, unmatched_gt)

            row = {
                "agml_dataset": r["agml_dataset"], "class_label": r["class_label"],
                "task_type": r.get("task_type", ""), "image_name": r["image_name"],
                "pred_boxes_json": json.dumps(pred_boxes), "gt_boxes_json": json.dumps(gt_boxes),
                "n_pred": r.get("n_pred", len(pred_boxes)), "n_gt": r.get("n_gt", len(gt_boxes)),
                "n_matched": r.get("n_matched", ""),
                "precision_at_iou50": r.get("precision_at_iou50", ""),
                "recall_at_iou50": r.get("recall_at_iou50", ""),
                "f1_at_iou50": r.get("f1_at_iou50", ""),
                "mean_iou_matched": r.get("mean_iou_matched", ""),
                "pred_boxes_filtered_json": json.dumps(filtered),
                "n_pred_filtered": filtered_metrics["n_pred"],
                "n_matched_filtered": filtered_metrics["n_matched"],
                "precision_at_iou50_filtered": filtered_metrics["precision"],
                "recall_at_iou50_filtered": filtered_metrics["recall"],
                "f1_at_iou50_filtered": filtered_metrics["f1"],
                "mean_iou_matched_filtered": filtered_metrics["mean_iou_matched"],
            }
            append_ckpt_row(out_ckpt_path, row, fields=BOX_VERIFIER_CKPT_FIELDS)
            rows.append(row)
        pbar.update()
    pbar.finish()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Inference time: {elapsed:.1f}s")
    if n_missing_image:
        print(f"  {n_missing_image} row(s) skipped scoring (image not found) — their original "
              f"pred_boxes_json is preserved WITHOUT confidence/criterion_scores added.")
    destroy_llm(llm)
    del tokenizer

    if rows:
        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            return sum(vals) / len(vals) if vals else 0.0

        metrics = {
            "n_total": len(rows),
            "unfiltered": {
                "mean_precision_at_iou50": round(_mean("precision_at_iou50"), 4),
                "mean_recall_at_iou50": round(_mean("recall_at_iou50"), 4),
                "mean_f1_at_iou50": round(_mean("f1_at_iou50"), 4),
            },
            "filtered": {
                "confidence_threshold": args.confidence_threshold,
                "mean_precision_at_iou50": round(_mean("precision_at_iou50_filtered"), 4),
                "mean_recall_at_iou50": round(_mean("recall_at_iou50_filtered"), 4),
                "mean_f1_at_iou50": round(_mean("f1_at_iou50_filtered"), 4),
            },
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Unfiltered: precision={metrics['unfiltered']['mean_precision_at_iou50']:.3f} "
              f"recall={metrics['unfiltered']['mean_recall_at_iou50']:.3f} "
              f"f1={metrics['unfiltered']['mean_f1_at_iou50']:.3f}")
        print(f"  Filtered (>={args.confidence_threshold}): "
              f"precision={metrics['filtered']['mean_precision_at_iou50']:.3f} "
              f"recall={metrics['filtered']['mean_recall_at_iou50']:.3f} "
              f"f1={metrics['filtered']['mean_f1_at_iou50']:.3f} (n={len(rows)})")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                         help="Verifier model — defaults to the same quantized checkpoint every "
                              "other judge/grading call in this repo uses.")
    parser.add_argument("--ckpt", required=True,
                         help="A detection-multiclass (DET_CKPT_FIELDS-shaped) checkpoint CSV to "
                              "grade boxes from.")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--output_dir", default="outputs_final/detection_box_verifier/")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="IoU threshold for counting a FILTERED prediction as a true-positive "
                              "match (re-scored against this; the unfiltered precision/recall/f1 "
                              "columns are passed through from --ckpt as-is, at whatever threshold "
                              "that run used).")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                         help="Boxes with confidence below this are dropped from "
                              "pred_boxes_filtered_json before re-scoring. No principled default yet "
                              "— tune this once you've seen a real confidence-score distribution "
                              "(metrics.json's filtered vs. unfiltered precision/recall/f1).")
    parser.add_argument("--crop_pad_frac", type=float, default=0.15,
                         help="Pad each box crop by this fraction of its own width/height per side "
                              "before showing it to the verifier — too tight loses context the model "
                              "needs to judge plausibility, too loose re-includes neighboring "
                              "instances (box_rubric criterion 3).")
    parser.add_argument("--grading_max_tokens", type=int, default=300,
                         help="max_tokens for the per-box grading call (4 short criterion lines — "
                              "this is deliberately NOT --gen_max_tokens-sized, there's no thinking "
                              "step here).")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192,
                         help="A box crop + short grading prompt + 4-line response needs nowhere "
                              "near detection-multiclass's own 24576 — see that script's own "
                              "--max_model_len help text for why that one's sized the way it is.")
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
