"""
Stage 5, Step 2 — Detection Box Refine: iterative correct-and-reverify
loop on top of detection-box-verifier's own crop/grade mechanism. See
docs/stage5_detection_verifier.md for the full design rationale —
summarized here.

detection-box-verifier only GRADES (confidence in, nothing out but a
score). This script takes action on a failing grade instead of just
filtering it out: for each predicted box, grade its crop against
box_rubric's 3 criteria (same mechanism, same grading call — see
src/detection_box_verifier.py's grade_boxes, reused directly), and for
ANY criterion scoring <= 2, apply that criterion's own corrective
generation call, then re-grade. Priority order when more than one
criterion fails at once — confirmed design choice, not arbitrary:

  criterion 3 (multiple instances merged) -> SPLIT first
  criterion 2 (box not tight)             -> TIGHTEN second
  criterion 1 (not a real instance)       -> RECLASSIFY/DROP last

Split-then-tighten-then-reclassify, not the rubric's own 1/2/3 order:
splitting a box changes how many boxes exist, so tightening or
reclassifying BEFORE a needed split would just waste a correction on a
box about to be replaced anyway. Each correction fully replaces the
box's prior state (new bbox from split/tighten, new category_name from
reclassify) and the box is re-graded fresh next round — a box's scores
from a PRIOR round are never trusted to still apply after any
correction.

Corrections, each a real generation call (unlike detection-box-verifier,
which never generates anything):
  - split: shown the SAME padded crop box_grading_prompt was shown (not
    the full image — the crop is exactly what the grader found lacking),
    asked to localize EACH separate instance within it. Each returned
    box_2d is reported relative to the CROP (0-1000 scale) and remapped
    back into the source image's own pixel coordinates (see
    remap_crop_box_to_image) before being treated as a new candidate box.
  - tighten: same crop, asked for a single tighter box for the same
    instance, remapped the same way.
  - reclassify: same crop, shown the OTHER category_name labels already
    present elsewhere in this same image's own pool (detection-multiclass's
    own free-text sub-class vocabulary for this image — not a fixed
    taxonomy, not the dataset's ground truth), asked which (if any) this
    crop actually matches. No match -> the box is dropped, not kept with
    its original (apparently wrong) label.

A box batches through AT MOST ONE correction per round (whichever of
split/tighten/reclassify its own current grade triggers, in that
priority order) — not all three per round, since each correction
invalidates the box's prior grade anyway, making a second correction in
the same round against a stale grade meaningless. --max_rounds (default
2) caps how many grade-correct-reverify rounds a box gets before it's
kept as-is and flagged "exhausted" rather than looped forever.

Usage:
    uv run python main.py detection-box-refine \\
        --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --output_dir outputs_final/detection_box_refine_smoketest/ \\
        --smoketest
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
from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes_json
from src.common.chat_prompt import build_multimodal_prompt
from src.common.metrics import extract_channel_thought
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.progress import ProgressBar
from src.detection_box_verifier import (
    _VERIFIER_PROMPTS_DETECTION_PATH, crop_box, crop_box_region, grade_boxes, load_yaml,
)
from src.detection_multiclass import SMOKETEST_DATASETS

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

# Same row shape as detection_box_verifier.py's own BOX_VERIFIER_CKPT_FIELDS
# minus the unfiltered/filtered split (there's only ONE pool here — the
# refined one — not an unfiltered-vs-filtered pair) plus refinement_stats,
# a per-row JSON summary of how many split/tighten/reclassify corrections
# and drops this row's boxes went through.
BOX_REFINE_CKPT_FIELDS = [
    "agml_dataset", "class_label", "task_type", "image_name",
    "pred_boxes_json", "gt_boxes_json",
    "n_pred", "n_gt", "n_matched", "precision_at_iou50", "recall_at_iou50",
    "f1_at_iou50", "mean_iou_matched",
    "refinement_stats",
]


def remap_crop_box_to_image(
    crop_bbox_0_1000: list[float], crop_region: tuple[int, int, int, int], image_w: int, image_h: int,
) -> list[float]:
    """A box_2d reported relative to a CROP (0-1000 normalized scale,
    y-first per Gemma's own native convention — same as
    detection_multiclass/detection_baseline's own box parsing) ->
    [x1,y1,x2,y2] pixel coordinates in the SOURCE image. crop_region must
    be the EXACT (cx1,cy1,cx2,cy2) crop_box_region() returned for the
    crop the model was actually shown — using a re-derived approximation
    here would silently misalign every corrected box."""
    cx1, cy1, cx2, cy2 = crop_region
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    px1, py1, px2, py2 = denormalize_box_1000(crop_bbox_0_1000, crop_w, crop_h)
    image_bbox = [cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2]
    return clamp_box(image_bbox, image_w, image_h)


def do_split(llm, tokenizer, items, system_prompt, vprompts, gen_max_tokens, seed,
             enable_thinking: bool = True, chat_template: str | None = None):
    """items: [(image, box, crop_region), ...]. Returns one list of new
    box dicts per item, same order — [] means the model found nothing
    real here (drop), a single-element list means it disagreed that a
    split was needed (treated as a corrected replacement, not a true
    split), 2+ means a real split. enable_thinking/chat_template: same
    mechanism as detection_box_verifier.grade_boxes — a correction call
    is a real generation decision (how many instances, where), so it gets
    the same reasoning step detection-multiclass's own box generation
    does."""
    template = vprompts["box_split_prompt"]
    prompts = []
    for image, box, crop_region in items:
        crop = image.crop(crop_region)
        text = template.format(category=box["category_name"])
        prompts.append(build_multimodal_prompt(tokenizer, crop, text, system_prompt=system_prompt,
                                                 enable_thinking=enable_thinking, chat_template=chat_template))
    params = SamplingParams(temperature=0.0, max_tokens=gen_max_tokens, seed=seed,
                             skip_special_tokens=not enable_thinking)
    outputs = llm.generate(prompts, sampling_params=params)
    results = []
    for (image, box, crop_region), out in zip(items, outputs):
        raw = out.outputs[0].text
        _, cleaned = extract_channel_thought(raw) if enable_thinking else ("", raw)
        parsed = extract_boxes_json(cleaned)
        new_boxes = []
        for p in parsed:
            img_bbox = remap_crop_box_to_image(p["bbox"], crop_region, image.width, image.height)
            new_boxes.append({"bbox": img_bbox, "category_name": p.get("category_name") or box["category_name"]})
        results.append(new_boxes)
    return results


def do_tighten(llm, tokenizer, items, system_prompt, vprompts, gen_max_tokens, seed,
               enable_thinking: bool = True, chat_template: str | None = None):
    """items: [(image, box, crop_region), ...]. Returns one tightened
    [x1,y1,x2,y2] (or None, meaning drop) per item, same order."""
    template = vprompts["box_tighten_prompt"]
    prompts = []
    for image, box, crop_region in items:
        crop = image.crop(crop_region)
        text = template.format(category=box["category_name"])
        prompts.append(build_multimodal_prompt(tokenizer, crop, text, system_prompt=system_prompt,
                                                 enable_thinking=enable_thinking, chat_template=chat_template))
    params = SamplingParams(temperature=0.0, max_tokens=gen_max_tokens, seed=seed,
                             skip_special_tokens=not enable_thinking)
    outputs = llm.generate(prompts, sampling_params=params)
    results = []
    for (image, box, crop_region), out in zip(items, outputs):
        raw = out.outputs[0].text
        _, cleaned = extract_channel_thought(raw) if enable_thinking else ("", raw)
        parsed = extract_boxes_json(cleaned)
        if not parsed:
            results.append(None)
            continue
        results.append(remap_crop_box_to_image(parsed[0]["bbox"], crop_region, image.width, image.height))
    return results


def do_reclassify(llm, tokenizer, items, system_prompt, vprompts, grading_max_tokens, seed,
                   enable_thinking: bool = True, chat_template: str | None = None):
    """items: [(image, box, crop_region, candidate_labels), ...]. Returns
    the matched candidate label (or None, meaning drop — no match/"none")
    per item, same order. Only ever returns a label from that item's own
    candidate list — never accepts free text back, so a malformed/
    off-list response is treated the same as "none"."""
    template = vprompts["box_reclassify_prompt"]
    prompts = []
    for image, box, crop_region, candidates in items:
        crop = image.crop(crop_region)
        candidate_text = "\n".join(f"- {c}" for c in candidates) if candidates else "(no other labels found elsewhere in this image)"
        text = template.format(candidate_labels=candidate_text)
        prompts.append(build_multimodal_prompt(tokenizer, crop, text, system_prompt=system_prompt,
                                                 enable_thinking=enable_thinking, chat_template=chat_template))
    params = SamplingParams(temperature=0.0, max_tokens=grading_max_tokens, seed=seed,
                             skip_special_tokens=not enable_thinking)
    outputs = llm.generate(prompts, sampling_params=params)
    results = []
    for (image, box, crop_region, candidates), out in zip(items, outputs):
        raw = out.outputs[0].text
        _, cleaned = extract_channel_thought(raw) if enable_thinking else ("", raw)
        text = cleaned.strip()
        match = next((c for c in candidates if c.strip().lower() == text.lower()), None)
        results.append(match)
    return results


def refine_group(
    llm, tokenizer, images_by_name: dict, boxes_by_image: dict[str, list[dict]],
    rubric: str, grading_template: str, vprompts: dict, system_prompt: str,
    other_labels_by_image: dict[str, list[str]],
    crop_pad_frac: float, grading_max_tokens: int, gen_max_tokens: int, max_rounds: int, seed: int,
    enable_thinking: bool = True, chat_template: str | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """One dataset group's worth of images, run through the full
    round-based grade -> correct -> reverify loop (see module docstring).
    Returns (finalized_boxes_by_image, stats) — stats is a flat dict of
    call/drop counts for this group, folded into each row's own
    refinement_stats at the caller."""
    active: dict[str, list[dict]] = {img: list(boxes) for img, boxes in boxes_by_image.items()}
    finalized: dict[str, list[dict]] = {img: [] for img in boxes_by_image}
    stats = {"rounds_run": 0, "split_calls": 0, "tighten_calls": 0, "reclassify_calls": 0, "dropped": 0}

    for round_i in range(max_rounds):
        flat = [(img, b) for img, boxes in active.items() for b in boxes]
        if not flat:
            break
        stats["rounds_run"] = round_i + 1

        crops = [crop_box(images_by_name[img], b["bbox"], crop_pad_frac) for img, b in flat]
        labels = [b["category_name"] for img, b in flat]
        grades = grade_boxes(llm, tokenizer, crops, labels, rubric, grading_template, system_prompt,
                              grading_max_tokens=grading_max_tokens, seed=seed,
                              enable_thinking=enable_thinking, chat_template=chat_template)
        for (img, b), g in zip(flat, grades):
            b["criterion_scores"] = g["criterion_scores"]
            b["confidence"] = g["confidence"]
            # One snapshot per round this box (or its lineage, across a
            # split) was graded — bbox/label/scores AT that round, plus
            # which correction (if any) was decided from them, filled in
            # just below. detection_box_verifier_summary.py's
            # plot_refinement_timeline reads this to show how a box
            # actually changed step by step, not just its final state.
            b.setdefault("history", []).append({
                "round": round_i + 1, "bbox": list(b["bbox"]), "category_name": b["category_name"],
                "criterion_scores": dict(b["criterion_scores"]), "confidence": b["confidence"], "action": None,
            })

        needs_split, needs_tighten, needs_reclassify, passed = [], [], [], []
        for img, b in flat:
            c = {int(k): int(v) for k, v in b["criterion_scores"].items()}
            c1, c2, c3 = c.get(1, 3), c.get(2, 3), c.get(3, 3)
            if c3 <= 2:
                b["history"][-1]["action"] = "split"
                needs_split.append((img, b))
            elif c2 <= 2:
                b["history"][-1]["action"] = "tighten"
                needs_tighten.append((img, b))
            elif c1 <= 2:
                b["history"][-1]["action"] = "reclassify"
                needs_reclassify.append((img, b))
            else:
                b["history"][-1]["action"] = "passed"
                passed.append((img, b))

        for img, b in passed:
            b["refinement_status"] = "passed"
            b["refinement_rounds"] = round_i + 1
            finalized[img].append(b)

        new_active: dict[str, list[dict]] = {img: [] for img in active}

        if needs_split:
            items = [(images_by_name[img], b, crop_box_region(images_by_name[img], b["bbox"], crop_pad_frac))
                      for img, b in needs_split]
            results = do_split(llm, tokenizer, items, system_prompt, vprompts, gen_max_tokens, seed,
                               enable_thinking=enable_thinking, chat_template=chat_template)
            stats["split_calls"] += len(items)
            for (img, b), new_boxes in zip(needs_split, results):
                if not new_boxes:
                    stats["dropped"] += 1
                    continue
                # split_group_size: how many boxes THIS specific split
                # call produced (1 = model disagreed a split was needed,
                # 2+ = a real split) — recorded explicitly at the moment
                # of splitting, not inferred later from history, so
                # "did splitting actually grow the box count" can be
                # checked directly per split event, not just assumed.
                for nb in new_boxes:
                    # Each new box starts from a COPY of the parent's
                    # history up through the split decision — so a
                    # genuine split (2+ results) shows the same shared
                    # lineage up to that point, then diverges from there.
                    nb["history"] = [dict(h) for h in b["history"]]
                    nb["history"][-1]["split_group_size"] = len(new_boxes)
                    new_active[img].append(nb)

        if needs_tighten:
            items = [(images_by_name[img], b, crop_box_region(images_by_name[img], b["bbox"], crop_pad_frac))
                      for img, b in needs_tighten]
            results = do_tighten(llm, tokenizer, items, system_prompt, vprompts, gen_max_tokens, seed,
                                 enable_thinking=enable_thinking, chat_template=chat_template)
            stats["tighten_calls"] += len(items)
            for (img, b), new_bbox in zip(needs_tighten, results):
                if new_bbox is None:
                    stats["dropped"] += 1
                    continue
                b["bbox"] = new_bbox
                new_active[img].append(b)

        if needs_reclassify:
            items = [
                (images_by_name[img], b, crop_box_region(images_by_name[img], b["bbox"], crop_pad_frac),
                 [l for l in other_labels_by_image.get(img, []) if l != b["category_name"]])
                for img, b in needs_reclassify
            ]
            results = do_reclassify(llm, tokenizer, items, system_prompt, vprompts, grading_max_tokens, seed,
                                     enable_thinking=enable_thinking, chat_template=chat_template)
            stats["reclassify_calls"] += len(items)
            for (img, b), new_label in zip(needs_reclassify, results):
                if new_label is None:
                    stats["dropped"] += 1
                    continue
                b["category_name"] = new_label
                new_active[img].append(b)

        active = new_active

    # Final grading pass for anything still active after max_rounds — a
    # box corrected during the LAST round's loop body above was never
    # re-graded before this point, so without this its stored
    # confidence/criterion_scores would reflect the PRE-correction box,
    # not what it actually ended up as. Needed so
    # detection-box-verifier-summary's threshold sweep (the only place
    # filtering happens — this script applies none itself) has an
    # accurate confidence to threshold on for every box in the output,
    # not just ones that happened to pass outright before the round cap.
    flat = [(img, b) for img, boxes in active.items() for b in boxes]
    if flat:
        crops = [crop_box(images_by_name[img], b["bbox"], crop_pad_frac) for img, b in flat]
        labels = [b["category_name"] for img, b in flat]
        grades = grade_boxes(llm, tokenizer, crops, labels, rubric, grading_template, system_prompt,
                              grading_max_tokens=grading_max_tokens, seed=seed,
                              enable_thinking=enable_thinking, chat_template=chat_template)
        for (img, b), g in zip(flat, grades):
            b["criterion_scores"] = g["criterion_scores"]
            b["confidence"] = g["confidence"]
            b.setdefault("history", []).append({
                "round": max_rounds, "bbox": list(b["bbox"]), "category_name": b["category_name"],
                "criterion_scores": dict(b["criterion_scores"]), "confidence": b["confidence"],
                "action": "exhausted",
            })

    for img, boxes in active.items():
        for b in boxes:
            b["refinement_status"] = "exhausted"
            b["refinement_rounds"] = max_rounds
            finalized[img].append(b)

    return finalized, stats


def run_model(model_id: str, args: argparse.Namespace) -> None:
    vprompts = load_yaml(_VERIFIER_PROMPTS_DETECTION_PATH)
    rubric = vprompts["box_rubric"]
    grading_template = vprompts["box_grading_prompt"]
    system_prompt = vprompts.get("assessment_system", "")
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    ckpt_path = Path(args.ckpt)
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    if not input_rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")

    if args.smoketest:
        wanted = set(SMOKETEST_DATASETS)
        input_rows = [r for r in input_rows if r["agml_dataset"] in wanted]
        found = {r["agml_dataset"] for r in input_rows}
        missing = wanted - found
        if missing:
            print(f"  [smoketest] WARNING: {sorted(missing)} not found in --ckpt")
        print(f"  [smoketest] restricting to {sorted(found)} ({len(input_rows)} row(s))")
        if not input_rows:
            raise SystemExit(f"--smoketest: none of {sorted(wanted)} found in {ckpt_path}.")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    out_ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(out_ckpt_path, args.rerun)
    rows, seen = load_ckpt(out_ckpt_path, fields=BOX_REFINE_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} rows already refined in {out_ckpt_path.name}")

    pending = [r for r in input_rows if (r["agml_dataset"], r["class_label"], r["image_name"]) not in seen]
    if not pending:
        print("  Nothing to do — all rows already checkpointed.")
        return

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in pending:
        by_dataset[r["agml_dataset"]].append(r)
    print(f"  {len(by_dataset)} dataset(s), {len(pending)} pending row(s), max_rounds={args.max_rounds}")

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="detection_box_refine",
    )

    t_start = time.perf_counter()
    pbar = ProgressBar(total=len(by_dataset), desc="Datasets")
    n_missing_image = 0
    for dataset, ds_rows in by_dataset.items():
        pbar.desc = f"Datasets ({dataset})"

        images_by_name: dict[str, PILImage.Image] = {}
        boxes_by_image: dict[str, list[dict]] = {}
        gt_by_image: dict[str, list[dict]] = {}
        other_labels_by_image: dict[str, list[str]] = {}
        for r in ds_rows:
            img_path = Path(args.images_root) / dataset / r["image_name"]
            pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
            gt_by_image[r["image_name"]] = json.loads(r.get("gt_boxes_json") or "[]")
            other_labels_by_image[r["image_name"]] = sorted({
                str(b.get("category_name")) for b in pred_boxes if b.get("category_name")
            })
            if not pred_boxes:
                boxes_by_image[r["image_name"]] = []
                continue
            if not img_path.exists():
                print(f"  [skip] {dataset}/{r['image_name']} — image not found at {img_path}")
                n_missing_image += 1
                boxes_by_image[r["image_name"]] = []
                continue
            images_by_name[r["image_name"]] = PILImage.open(img_path).convert("RGB")
            # Strip any stale grading from the input checkpoint — every
            # box is re-graded fresh in round 1 against the CURRENT
            # (possibly just-changed) 3-criterion rubric.
            boxes_by_image[r["image_name"]] = [
                {"bbox": b["bbox"], "category_name": b.get("category_name") or "object"} for b in pred_boxes
            ]

        has_images = {img: boxes for img, boxes in boxes_by_image.items() if img in images_by_name and boxes}
        finalized, stats = refine_group(
            llm, tokenizer, images_by_name, has_images, rubric, grading_template, vprompts, system_prompt,
            other_labels_by_image, crop_pad_frac=args.crop_pad_frac, grading_max_tokens=args.grading_max_tokens,
            gen_max_tokens=args.gen_max_tokens, max_rounds=args.max_rounds, seed=args.seed,
            enable_thinking=args.enable_thinking, chat_template=chat_template,
        )

        for r in ds_rows:
            image_name = r["image_name"]
            refined_boxes = finalized.get(image_name, boxes_by_image.get(image_name, []))
            gt_boxes = gt_by_image[image_name]
            matches, unmatched_pred, unmatched_gt = match_boxes(refined_boxes, gt_boxes, args.iou_threshold)
            metrics = precision_recall_f1(matches, unmatched_pred, unmatched_gt)
            row = {
                "agml_dataset": r["agml_dataset"], "class_label": r["class_label"],
                "task_type": r.get("task_type", ""), "image_name": image_name,
                "pred_boxes_json": json.dumps(refined_boxes), "gt_boxes_json": json.dumps(gt_boxes),
                "n_pred": metrics["n_pred"], "n_gt": metrics["n_gt"], "n_matched": metrics["n_matched"],
                "precision_at_iou50": metrics["precision"], "recall_at_iou50": metrics["recall"],
                "f1_at_iou50": metrics["f1"], "mean_iou_matched": metrics["mean_iou_matched"],
                "refinement_stats": json.dumps(stats),
            }
            append_ckpt_row(out_ckpt_path, row, fields=BOX_REFINE_CKPT_FIELDS)
            rows.append(row)
        pbar.update()
    pbar.finish()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Inference time: {elapsed:.1f}s")
    if n_missing_image:
        print(f"  {n_missing_image} row(s) had no source image found — their boxes pass through unrefined.")
    destroy_llm(llm)
    del tokenizer

    if rows:
        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            return sum(vals) / len(vals) if vals else 0.0

        metrics = {
            "n_total": len(rows),
            "mean_precision_at_iou50": round(_mean("precision_at_iou50"), 4),
            "mean_recall_at_iou50": round(_mean("recall_at_iou50"), 4),
            "mean_f1_at_iou50": round(_mean("f1_at_iou50"), 4),
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Refined: precision={metrics['mean_precision_at_iou50']:.3f} "
              f"recall={metrics['mean_recall_at_iou50']:.3f} f1={metrics['mean_f1_at_iou50']:.3f} "
              f"(n={len(rows)})")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                         help="Verifier/refiner model — defaults to the same quantized checkpoint "
                              "every other judge/grading call in this repo uses.")
    parser.add_argument("--ckpt", required=True,
                         help="A detection-multiclass (or detection-box-verifier) checkpoint CSV to "
                              "refine boxes from. Any pre-existing confidence/criterion_scores on its "
                              "boxes are ignored — every box is re-graded fresh in round 1.")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--output_dir", default="outputs_final/detection_box_refine/")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="IoU threshold for the FINAL refined box set's own precision/recall/F1.")
    parser.add_argument("--crop_pad_frac", type=float, default=0.15,
                         help="Same meaning as detection-box-verifier's own flag — padding per side "
                              "when cropping a box for grading/correction.")
    parser.add_argument("--max_rounds", type=int, default=2,
                         help="Grade -> correct -> reverify rounds per box before it's kept as-is "
                              "and flagged \"exhausted\" in refinement_stats rather than looped "
                              "forever.")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Same mechanism/default as detection_multiclass.py's own "
                              "--enable_thinking — on by default, for both the grading call and all "
                              "three correction calls (split/tighten/reclassify). "
                              "detection-multiclass's own boxes are generated WITH thinking, so "
                              "grading/correcting them with no reasoning step was a weaker, "
                              "not-apples-to-apples pass. Pass --no-enable_thinking to opt out.")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja",
                         help="Same as detection_multiclass.py's own flag — pass \"\" to unset for a "
                              "non-Gemma model.")
    parser.add_argument("--grading_max_tokens", type=int, default=1024,
                         help="max_tokens for the grading call and the reclassify correction — "
                              "bumped from the original 300 to leave headroom for a thinking trace "
                              "before the short criterion/label output, now that --enable_thinking "
                              "defaults to on.")
    parser.add_argument("--gen_max_tokens", type=int, default=1024,
                         help="max_tokens for the split/tighten correction calls — bumped from the "
                              "original 300 for the same reason as --grading_max_tokens.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--smoketest", action="store_true",
                         help=f"Restrict --ckpt's rows to just {SMOKETEST_DATASETS!r} — same "
                              f"convention as detection-multiclass/detection-box-verifier.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096,
                         help="A crop + a short correction/grading prompt needs nowhere near "
                              "detection-multiclass's own 24576 — see detection-box-verifier's own "
                              "--max_model_len help text for the same reasoning.")
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
