"""
Stage 6 — Detection Multi-Class Generation: same single-shot,
no-intermediate-reasoning box prediction as detection_baseline.py (Stage
2), except the prompt asks the model to report EVERY visible instance of
the target object at ANY growth stage/condition, labeling each box with
its own specific stage (e.g. "bud", "open flower", "dead flower") instead
of just the dataset's coarse class name. See
docs/stage6_detection_verifier.md and
src/config/prompts_detection.yaml's detection_multiclass header comment
for the full motivation (a real QC finding: ground truth on
GEMINI_cowpea_flower_detection only labels open/senesced blooms as
"flower", but the model correctly detects buds and dead flowers too —
deflating precision under single-class scoring, not a real localization
bug).

This is the FIRST piece of Stage 6 (generation only) — this script does
NOT do the planned per-box confidence-grading verifier pass yet, see
docs/stage6_detection_verifier.md's "Planned: per-box verifier" section
for that design. Scoring here is already class-agnostic
(bbox_metrics.match_boxes never reads category_name), so precision/
recall/F1 against ground truth work exactly like detection_baseline.py's
own — this script only changes the PROMPT and the stage/condition label
each predicted box carries in pred_boxes_json's "category_name" field.

Deliberately reuses detection_baseline.py's run_batch/score_row/
DET_CKPT_FIELDS/box-parsing pipeline as-is (imported, not duplicated) —
Stage 2's own baseline script is mid-run right now and is NOT touched by
this file. Same CLI surface/defaults (thinking-enabled by default,
--gen_max_tokens 4096, vendored Gemma-4 chat template) as
detection_baseline.py, just a different prompt key and a 2-dataset
smoketest instead of 1.

Usage:
    uv run python main.py detection-multiclass \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --output_dir outputs_final/detection_multiclass_smoketest/ \\
        --smoketest
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

from PIL import Image as PILImage

from src.common.dataset_config import active_datasets, crop_display, load_hf_detection_samples, load_manifest
from src.common.io import load_prompts
from src.common.progress import ProgressBar
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes, extract_boxes_json, \
    needs_1000_scale_boxes, uses_json_box_format

from src.detection_baseline import DET_CKPT_FIELDS, run_batch, score_row

# GEMINI_cowpea_flower_detection: the same 990-image dataset
# detection_baseline.py's own --smoketest uses (see that module's
# SMOKETEST_DATASET), where the stage-mismatch finding motivating this
# script was first observed. GEMINI_cowpea_pod_detection: the second
# growth-stage-varying dataset from this conversation (young/mature/dried
# pods) — both need to be present in --config/--manifest (see
# docs/stage2_object_detection.md's smoke test section for the
# download-detection-datasets --only pattern; run that for
# GEMINI_cowpea_pod_detection too if it isn't already downloaded).
SMOKETEST_DATASETS = ["GEMINI_cowpea_flower_detection", "GEMINI_cowpea_pod_detection"]


def det_question_multiclass(ds, prompts: dict, use_json: bool = False) -> str:
    """Same shape as detection_baseline.det_question, but selects the
    detection_multiclass(/_json) prompt key instead of detection_baseline
    — see that key's header comment in prompts_detection.yaml."""
    names_list = ds.classes_with_aliases_display()
    names = ", ".join(names_list) if names_list else "object"
    key = "detection_multiclass_json" if use_json else "detection_multiclass"
    return prompts[key].format(class_names=names, crop=crop_display(ds.crop))


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None
    manifest = load_manifest(Path(args.manifest))
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

    all_active = list(active_datasets(Path(args.config)))
    if args.smoketest:
        wanted = set(SMOKETEST_DATASETS)
        all_active = [ds for ds in all_active if ds.agml_name in wanted]
        found = {ds.agml_name for ds in all_active}
        missing = wanted - found
        if missing:
            raise SystemExit(
                f"--smoketest: {sorted(missing)} not found (or not active) in --config "
                f"{args.config} — catalog/migrate them first (see "
                f"docs/stage2_object_detection.md's smoke test section)."
            )
        print(f"  [smoketest] restricting to {sorted(found)}")

    groups = []
    for ds in all_active:
        if ds.skip:
            continue
        try:
            samples = load_hf_detection_samples(ds.agml_name, manifest, Path(args.images_root))
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest"
                  + (f" — run download-detection-datasets --only '^{ds.agml_name}$' "
                     f"--skip_existing first" if args.smoketest else ""))
            continue
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
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="detection_multiclass",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Datasets")
        for ds, samples in groups:
            pbar.desc = f"Datasets ({ds.agml_name})"
            images = [PILImage.open(s["path"]).convert("RGB") for s in samples]
            question = det_question_multiclass(ds, prompts, use_json=use_json)

            results = run_batch(
                llm, tokenizer, images, question, system_prompt=system_prompt,
                enable_thinking=args.enable_thinking, chat_template=chat_template,
                gen_max_tokens=args.gen_max_tokens,
            )

            for s, img, result in zip(samples, images, results):
                pred = result["prediction"]
                gt_boxes = s["boxes"]
                pred_boxes_raw = extract_boxes_json(pred) if use_json else extract_boxes(pred)
                pred_boxes = []
                for b in pred_boxes_raw:
                    bbox = denormalize_box_1000(b["bbox"], img.width, img.height) if rescale_1000 else b["bbox"]
                    pred_boxes.append({**b, "bbox": clamp_box(bbox, img.width, img.height)})
                # Class-agnostic matching (bbox_metrics.match_boxes never
                # reads category_name) — precision/recall/F1 here count a
                # predicted box as a true positive purely on IoU overlap
                # with a ground-truth box, regardless of what stage label
                # the model gave it or what single class name ground
                # truth uses. The per-box "category_name" field (the
                # model's own stage/condition label) is preserved in
                # pred_boxes_json for inspection/QC even though scoring
                # ignores it.
                metrics = score_row(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
                primary_label = gt_boxes[0]["category_name"] if gt_boxes else ""
                row = {
                    "agml_dataset": ds.agml_name, "class_label": primary_label,
                    "task_type": ds.task_type, "image_name": Path(s["path"]).name,
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
    parser.add_argument("--model", default=None, help="Single model to run.")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts_detection.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images_detection_10.json")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--output_dir", default="outputs_final/detection_multiclass/")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="IoU threshold for counting a prediction as a true-positive match.")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Same mechanism/default as detection_baseline.py's own "
                              "--enable_thinking — on by default. Pass --no-enable_thinking to "
                              "opt out.")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja",
                         help="Same as detection_baseline.py's own flag — pass \"\" to unset for a "
                              "non-Gemma model.")
    parser.add_argument("--gen_max_tokens", type=int, default=4096,
                         help="Same default as detection_baseline.py — real headroom for a "
                              "thinking trace before box output.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--smoketest", action="store_true",
                         help=f"Restrict the run to just {SMOKETEST_DATASETS!r}, for validating "
                              "the multi-class prompt end to end before committing to a full "
                              "multi-dataset run. Needs both datasets present in --config and "
                              "--manifest.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None,
                         help="None keeps vLLM's default (same dtype as the model). \"fp8\" halves "
                              "KV-cache memory -- useful for a long --max_model_len on a "
                              "memory-constrained GPU, independent of --quantize (see vlm_load.load_llm).")

    main(parser.parse_args())
