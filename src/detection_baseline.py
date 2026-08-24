"""
Stage 5b — Detection Baseline: image + direct box-localization question only,
no intermediate reasoning. Spatial-task analog of vision_baseline.py (Stage
2a) — the lower bound Detection-Verifier (src/detection_verifier.py, Stage
5d) is compared against at an equal image budget. See
docs/stage5_object_detection.md.

Classification-only fields (resolve_class, rouge, judge phase) don't apply
here — scoring is IoU-based (src/common/bbox_metrics.py) against the
ground-truth boxes load_hf_detection_samples already cached alongside each
image, computed and checkpointed per-row directly (no separate judge phase).

Per-model output subfolder + a per-sample checkpoint CSV (flushed to disk
immediately after each sample is scored) so a run on a time-limited remote
job can resume exactly where it left off:
  <output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics.json

Usage:
    uv run python main.py detection-baseline --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
import time

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes, precision_recall_f1
from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes, needs_1000_scale_boxes
from src.common.chat_prompt import build_multimodal_prompt
from src.common.dataset_config import active_datasets, load_hf_detection_samples, load_manifest
from src.common.io import load_prompts
from src.common.progress import ProgressBar
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
    _DET_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    _DET_PARAMS = None

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)

# One row per (dataset, class, image) — same granularity as vision_baseline's
# CKPT_FIELDS, but "class_label" here is the PRIMARY category the manifest
# balanced on (see download_and_sample_detection.py), not the only label —
# gt_boxes_json/pred_boxes_json carry the FULL per-image box list.
DET_CKPT_FIELDS = [
    "agml_dataset", "class_label", "task_type", "image_name",
    "raw_prediction", "pred_boxes_json", "gt_boxes_json",
    "n_pred", "n_gt", "n_matched", "precision_at_iou50", "recall_at_iou50",
    "f1_at_iou50", "mean_iou_matched",
]


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def run_batch(llm, tokenizer, images: list, question: str, system_prompt: str = "") -> list[str]:
    prompts = [build_multimodal_prompt(tokenizer, img, question, system_prompt=system_prompt) for img in images]
    outputs = llm.generate(prompts, sampling_params=_DET_PARAMS)
    return [_clean(o.outputs[0].text) for o in outputs]


def det_question(class_names: list[str], prompts: dict) -> str:
    names = ", ".join(class_names) if class_names else "object"
    return prompts["detection_baseline"].format(class_names=names)


def score_row(pred_boxes: list[dict], gt_boxes: list[dict], iou_threshold: float = 0.5) -> dict:
    matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    return precision_recall_f1(matches, unmatched_pred, unmatched_gt)


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    manifest = load_manifest(Path(args.manifest))
    rescale_1000 = needs_1000_scale_boxes(model_id)
    if rescale_1000:
        print(f"  [bbox] {model_id}: rescaling predicted boxes from Gemma's native 0-1000 "
              f"normalized scale to real pixel coordinates (see bbox_parse.needs_1000_scale_boxes)")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=DET_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} samples already scored in {ckpt_path.name}")

    # Collect (dataset, class_names, [(image_name, path, boxes)]) groups —
    # one call to run_batch per dataset (all images share the same
    # class_names, hence the same question), unlike classification's
    # per-(dataset,class) grouping (class_names here means EVERY class in
    # the dataset, not one — a detection prompt asks about the whole
    # vocabulary at once, not one class at a time).
    groups = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip:
            continue
        try:
            samples = load_hf_detection_samples(ds.agml_name, manifest, Path(args.images_root))
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest")
            continue
        # "seen" keys are (agml_dataset, class_label, image_name) — but
        # class_label here would require re-deriving the primary category
        # before even checking resume state, so instead just check
        # image_name alone against every seen triple for this dataset
        # (image_name is already unique per (dataset, row_idx), so this is
        # an equally safe resume check without the extra work).
        seen_names = {img for d, _, img in seen if d == ds.agml_name}
        pending = [s for s in samples if Path(s["path"]).name not in seen_names]
        if not pending:
            continue
        groups.append((ds, pending))

    if not groups:
        print("  Nothing to do — all samples already checkpointed.")
    else:
        n_pending = sum(len(g[1]) for g in groups)
        print(f"  {len(groups)} dataset group(s), {n_pending} pending sample(s)")

        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            caller_tag="detection_baseline",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Datasets")
        for ds, samples in groups:
            pbar.desc = f"Datasets ({ds.agml_name})"
            images = [PILImage.open(s["path"]).convert("RGB") for s in samples]
            question = det_question(ds.clean_classes, prompts)

            preds = run_batch(llm, tokenizer, images, question, system_prompt=system_prompt)

            for s, img, pred in zip(samples, images, preds):
                gt_boxes = s["boxes"]
                pred_boxes_raw = extract_boxes(pred)
                pred_boxes = []
                for b in pred_boxes_raw:
                    bbox = denormalize_box_1000(b["bbox"], img.width, img.height) if rescale_1000 else b["bbox"]
                    pred_boxes.append({**b, "bbox": clamp_box(bbox, img.width, img.height)})
                metrics = score_row(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
                primary_label = gt_boxes[0]["category_name"] if gt_boxes else ""
                row = {
                    "agml_dataset": ds.agml_name, "class_label": primary_label,
                    "task_type": ds.task_type, "image_name": Path(s["path"]).name,
                    "raw_prediction": pred,
                    "pred_boxes_json": json.dumps(pred_boxes),
                    "gt_boxes_json": json.dumps(gt_boxes),
                    "n_pred": metrics["n_pred"], "n_gt": metrics["n_gt"], "n_matched": metrics["n_matched"],
                    "precision_at_iou50": metrics["precision"], "recall_at_iou50": metrics["recall"],
                    "f1_at_iou50": metrics["f1"], "mean_iou_matched": metrics["mean_iou_matched"],
                }
                append_ckpt_row(ckpt_path, row, fields=DET_CKPT_FIELDS)
                rows.append(row)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer

    if rows:
        mean_p = sum(float(r["precision_at_iou50"]) for r in rows) / len(rows)
        mean_r = sum(float(r["recall_at_iou50"]) for r in rows) / len(rows)
        mean_f1 = sum(float(r["f1_at_iou50"]) for r in rows) / len(rows)
        metrics = {"n_total": len(rows), "mean_precision_at_iou50": round(mean_p, 4),
                   "mean_recall_at_iou50": round(mean_r, 4), "mean_f1_at_iou50": round(mean_f1, 4)}
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Overall: precision={mean_p:.3f} recall={mean_r:.3f} f1={mean_f1:.3f} (n={len(rows)})")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Single model to run.")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts_detection.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images_detection.json")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--output_dir", default="outputs_final/detection_baseline/")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="IoU threshold for counting a prediction as a true-positive match.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")

    main(parser.parse_args())
