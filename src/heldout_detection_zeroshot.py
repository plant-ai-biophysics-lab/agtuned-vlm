"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 1 —
zero-shot detection prediction on the held-out TRAIN (lessons) / VAL
(eval) pool Step 0 built. Detection analog of heldout_zeroshot.py
(Stage 3's own Step 1).

Reuses `detection-multiclass`'s own generation+scoring mechanism
directly — `detection_baseline.py`'s `run_batch`/`score_row`/
`DET_CKPT_FIELDS` (imported, not duplicated) and `detection_multiclass.py`'s
own prompt key (`detection_multiclass`/`detection_multiclass_json`) — NOT
plain `detection-baseline`, consistent with Step −1's own choice (real
local metrics showed multiclass beats baseline on F1/recall).

Reads `dataset/detection_heldout_selected_images.json` directly (Step
0's own output) — NOT `active_datasets()`/`DatasetConfig`, since these
15 TRAIN+VAL datasets are deliberately marked `skip: true` to stay OUT
of that normal AgML-manifest pipeline entirely. Per-dataset class
list/crop for the prompt's `{class_names}`/`{crop}` slots come straight
from `src/config/detection_datasets.yaml`'s own `classes`/`crop`
fields for that `agml_dataset` — not a `DatasetConfig` object (no
aliases to fold in for these datasets).

`--split {lessons,eval}`: `lessons` = TRAIN (this step's own primary
use — feeds Step 2's explain-traces), `eval` = VAL (Step 3b's
self-optimization loop scoring pass later, and Step 4's ablation) —
same script handles both, just filtered to that split's own rows.
`DET_CKPT_FIELDS` already carries `agml_dataset`/`class_label`/
`image_name` natively (unlike classification's `HELDOUT_CKPT_FIELDS`,
which had to repurpose those 3 names — detection's own checkpoint
schema already matches `load_ckpt`'s hardcoded resume-key contract with
no repurposing needed).

`class_label` is the FIRST ground-truth box's own `category_name` (same
convention `detection_multiclass.py`'s own `run_model` uses) — a
summary label for display/resume-keying only; scoring itself is always
class-agnostic (`bbox_metrics.match_boxes` never reads `category_name`).

Usage:
    uv run python main.py heldout-detection-zeroshot \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --split lessons \\
        --output_dir outputs_final/heldout_detection_zeroshot_lessons/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from collections import defaultdict

from PIL import Image as PILImage

from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes, extract_boxes_json, \
    needs_1000_scale_boxes, uses_json_box_format
from src.common.dataset_config import crop_display
from src.common.io import load_prompts
from src.common.progress import ProgressBar
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

from src.detection_baseline import DET_CKPT_FIELDS, run_batch, score_row


def load_heldout_detection_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_dataset_meta(path: Path) -> dict:
    """{agml_name: {"classes": [...], "crop": ...}} from
    src/config/detection_datasets.yaml — only the 2 fields this script's
    own prompt-building needs, not a full DatasetConfig object."""
    import yaml
    data = yaml.safe_load(path.read_text())
    return {e["agml_name"]: {"classes": e.get("classes") or [], "crop": e.get("crop")} for e in data["datasets"]}


def det_question_heldout(classes: list[str], crop, prompts: dict, use_json: bool) -> str:
    """Same shape as detection_multiclass.py's own det_question_multiclass,
    built from a plain classes list + crop value instead of a
    DatasetConfig object (no configured aliases to fold in for these
    datasets)."""
    names = ", ".join(classes) if classes else "object"
    key = "detection_multiclass_json" if use_json else "detection_multiclass"
    return prompts[key].format(class_names=names, crop=crop_display(crop))


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None
    manifest = load_heldout_detection_manifest(Path(args.manifest))
    dataset_meta = load_dataset_meta(Path(args.config))
    rescale_1000 = needs_1000_scale_boxes(model_id)
    use_json = uses_json_box_format(model_id)
    if rescale_1000:
        print(f"  [bbox] {model_id}: rescaling predicted boxes from Gemma's native 0-1000 "
              f"normalized scale to real pixel coordinates (see bbox_parse.needs_1000_scale_boxes)")
    if use_json:
        print(f"  [bbox] {model_id}: requesting Gemma's native box_2d JSON output format "
              f"instead of <box> tags (see bbox_parse.uses_json_box_format)")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=DET_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} samples already scored in {ckpt_path.name}")

    selected = {k: r for k, r in manifest["rows"].items() if r["split"] == args.split}
    if not selected:
        raise SystemExit(f"ERROR: no rows with split={args.split!r} in {args.manifest}.")
    print(f"  {len(selected)} row(s) in split={args.split!r} ({len(manifest['rows'])} total in manifest)")

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for key, r in selected.items():
        by_dataset[r["agml_dataset"]].append(r)

    groups = []
    for agml_name, ds_rows in by_dataset.items():
        seen_names = {img for d, _, img in seen if d == agml_name}
        pending = [r for r in ds_rows if Path(r["image_path"]).name not in seen_names]
        if not pending:
            continue
        groups.append((agml_name, pending))

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
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_detection_zeroshot",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Datasets")
        for agml_name, ds_rows in groups:
            pbar.desc = f"Datasets ({agml_name})"
            meta = dataset_meta.get(agml_name, {"classes": [], "crop": None})
            images = [PILImage.open(r["image_path"]).convert("RGB") for r in ds_rows]
            question = det_question_heldout(meta["classes"], meta["crop"], prompts, use_json=use_json)

            results = run_batch(
                llm, tokenizer, images, question, system_prompt=system_prompt,
                enable_thinking=args.enable_thinking, chat_template=chat_template,
                gen_max_tokens=args.gen_max_tokens,
            )

            for r, img, result in zip(ds_rows, images, results):
                pred = result["prediction"]
                gt_boxes = r["boxes"]
                pred_boxes_raw = extract_boxes_json(pred) if use_json else extract_boxes(pred)
                pred_boxes = []
                for b in pred_boxes_raw:
                    bbox = denormalize_box_1000(b["bbox"], img.width, img.height) if rescale_1000 else b["bbox"]
                    pred_boxes.append({**b, "bbox": clamp_box(bbox, img.width, img.height)})
                metrics = score_row(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
                primary_label = gt_boxes[0]["category_name"] if gt_boxes else ""
                row = {
                    "agml_dataset": agml_name, "class_label": primary_label,
                    "task_type": "object_detection", "image_name": Path(r["image_path"]).name,
                    "raw_prediction": pred,
                    "reasoning": result["reasoning"],
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
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--manifest", default="dataset/detection_heldout_selected_images.json")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--split", choices=["lessons", "eval"], default="lessons",
                         help="lessons=TRAIN (this step's own primary use), eval=VAL (Step 3b/4 "
                              "later) — never use eval to write the skill-memory file.")
    parser.add_argument("--prompts", default="src/config/prompts_detection.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_detection_zeroshot/")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
