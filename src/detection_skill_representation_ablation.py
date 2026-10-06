"""
Paper 2, Stage 4 DRAFT (docs/stage4_detection_context_memory_DRAFT.md),
Step -1 — box-representation ablation, THREE conditions. Answers two
questions BEFORE the detection skill-memory pipeline's own Step 2
(explain traces) gets built for real:

  1. Raw coordinates vs. rendered overlay: when asking a model to
     explain what's wrong with a box prediction, should it be shown
     (a) the plain full image + the raw [x1,y1,x2,y2] coordinates as
     text, or (b) the full image with both boxes already drawn on it
     (predicted solid lime, ground truth dashed red)? VLMs are not
     generally reliable at spatial reasoning from raw numbers alone —
     (b) turns it into a vision task instead of a numeric-reasoning
     task — but this should be CHECKED, not assumed.
  2. Full image vs. cropped: given an overlay IS used, does a tight,
     padded crop around just the box(es) in question (c) help (less
     distraction, more resolution on small/occluded objects) or hurt
     (loses surrounding context a judgment like "should this be split
     into two instances" needs)?

Source checkpoint is a `detection-multiclass` run, NOT plain
detection-baseline — real local metrics
(outputs_final/detection_multiclass vs. outputs_final/detection_baseline,
RedHatAI/gemma-4-31B-it-FP8-dynamic) showed multiclass winning on F1
(0.3773 vs. 0.3531) and recall (0.4283 vs. 0.3501), trading away some
precision (0.3997 vs. 0.4392) to get there — the real pipeline builds
on multiclass, so this ablation does too.

Samples up to --n_samples box-level DISCREPANCIES (not every box — only
the interesting cases: false positives, missed instances, loose
matches), roughly round-robin across those 3 error types, THREE calls
per sample (one per condition — 3 x --n_samples total generation calls,
ONE shared vLLM load). For a sample with two boxes (worked/loose_match
— see find_discrepancies), condition (c)'s crop is the PADDED UNION of
both boxes, not just one, so neither box gets cut off. For EACH sample,
renders ONE comparison image: question at top, the full annotated image
and the cropped annotated image side by side, and all three conditions'
model responses below, labeled (a)/(b)/(c).

No scoring, no metric — this is a qualitative, read-it-yourself gate
(per explicit design choice: do not commit to any one representation
from intuition alone, but also don't force a premature automated metric
on 30 qualitative samples either).

Usage:
    uv run python main.py detection-skill-representation-ablation \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --n_samples 30
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — pred_boxes_json/gt_boxes_json can exceed the 131072-byte default on a full run.
import json
import random
import textwrap

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes
from src.common.bbox_viz import plot_predictions
from src.common.chat_prompt import build_multimodal_prompt
from src.common.metrics import extract_channel_thought
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_LOOSE_IOU_THRESHOLD = 0.7  # a matched pair below this (but above the matching threshold itself,
# so it's still a "real" match, just not a tight one) counts as a "loose_match" sample — distinct
# from an unmatched pred (false_positive) or unmatched gt (missed).


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def find_discrepancies(rows: list[dict], iou_threshold: float) -> list[dict]:
    """One entry per box-level DISCREPANCY (not every box) across the
    whole checkpoint: {"row", "pred_box", "gt_box", "error_type", "iou"}
    — pred_box/gt_box is None for a missed/false_positive sample (only
    one side exists), both present for a loose_match."""
    out: list[dict] = []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        for pi in unmatched_pred:
            out.append({"row": r, "pred_box": pred_boxes[pi], "gt_box": None,
                        "error_type": "false_positive", "iou": 0.0})
        for gi in unmatched_gt:
            out.append({"row": r, "pred_box": None, "gt_box": gt_boxes[gi],
                        "error_type": "missed", "iou": 0.0})
        for pi, gi, iou in matches:
            if iou < _LOOSE_IOU_THRESHOLD:
                out.append({"row": r, "pred_box": pred_boxes[pi], "gt_box": gt_boxes[gi],
                            "error_type": "loose_match", "iou": iou})
    return out


def union_crop_region(image: PILImage.Image, boxes: list[list[float]], pad_frac: float = 0.25) -> tuple[int, int, int, int]:
    """The clamped, padded crop region (cx1,cy1,cx2,cy2) covering the
    UNION of every box in `boxes` (1 or 2, here) — NOT
    detection_box_verifier.py's own crop_box_region, which crops around
    a single box only; a loose_match/worked sample's predicted and
    ground-truth boxes don't fully overlap by definition, so cropping
    around just one of them risks cutting the other one off entirely.
    pad_frac is relative to the UNION box's own width/height (same
    not-a-fixed-pixel-count reasoning as crop_box's own docstring —
    detection images vary wildly in resolution)."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_x, pad_y = w * pad_frac, h * pad_frac
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(image.width, max(cx1 + 1, int(x2 + pad_x)))
    cy2 = min(image.height, max(cy1 + 1, int(y2 + pad_y)))
    return cx1, cy1, cx2, cy2


def shift_box(box: dict, dx: float, dy: float) -> dict:
    """A copy of `box` with its bbox shifted by (-dx,-dy) — remaps a box
    reported in the SOURCE image's coordinate frame into a crop's own
    frame (crop origin becomes (0,0)), same remapping
    detection_box_refine.py's split/tighten corrections need and get
    from crop_box_region's own exposed offset; here the offset is the
    crop region's own (cx1,cy1), not a per-correction one."""
    x1, y1, x2, y2 = box["bbox"]
    return {**box, "bbox": [x1 - dx, y1 - dy, x2 - dx, y2 - dy]}


def sample_round_robin(discrepancies: list[dict], n_samples: int, seed: int) -> list[dict]:
    """Up to n_samples, round-robin across the 3 error types so no single
    type (false positives are typically the most numerous, from
    detection-multiclass's own deliberate over-generation — every
    plausible box, not pruned) dominates the sample purely by base rate."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {"false_positive": [], "missed": [], "loose_match": []}
    for d in discrepancies:
        by_type[d["error_type"]].append(d)
    for v in by_type.values():
        rng.shuffle(v)

    sampled: list[dict] = []
    types = ["false_positive", "missed", "loose_match"]
    idx = {t: 0 for t in types}
    while len(sampled) < n_samples and any(idx[t] < len(by_type[t]) for t in types):
        for t in types:
            if idx[t] < len(by_type[t]):
                sampled.append(by_type[t][idx[t]])
                idx[t] += 1
                if len(sampled) >= n_samples:
                    break
    return sampled


def build_question(hprompts: dict, d: dict) -> str:
    category = (d["pred_box"] or d["gt_box"]).get("category_name") or d["row"].get("class_label") or "object"
    if d["error_type"] == "false_positive":
        return hprompts["detection_skill_question_false_positive"].format(category=category)
    if d["error_type"] == "missed":
        return hprompts["detection_skill_question_missed"].format(category=category)
    return hprompts["detection_skill_question_loose_match"].format(category=category, iou=d["iou"])


def run_model(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.detection_skill_prompts))
    system_prompt = hprompts["detection_skill_system"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    with open(args.ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: {args.ckpt} has no rows.")

    discrepancies = find_discrepancies(rows, args.iou_threshold)
    if not discrepancies:
        raise SystemExit(f"ERROR: no box-level discrepancies found in {args.ckpt} at "
                          f"--iou_threshold {args.iou_threshold} — nothing to sample.")
    sampled = sample_round_robin(discrepancies, args.n_samples, args.seed)
    counts = {t: sum(1 for s in sampled if s["error_type"] == t) for t in
              ("false_positive", "missed", "loose_match")}
    print(f"  {len(sampled)} sample(s): {counts}")

    out_dir = Path(args.out_dir)
    annotated_dir = out_dir / "annotated"
    cropped_dir = out_dir / "cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render each sample's full + cropped annotated images + build all
    # three conditions' questions BEFORE loading the model, so the
    # (slow, GPU-resident) load window is spent only on generation.
    prepared = []
    for i, d in enumerate(sampled):
        r = d["row"]
        img_path = Path(args.images_root) / r["agml_dataset"] / r["image_name"]
        if not img_path.exists():
            print(f"  [skip] {r['agml_dataset']}/{r['image_name']} — image not found at {img_path}")
            continue
        image = PILImage.open(img_path).convert("RGB")

        pred_list = [d["pred_box"]] if d["pred_box"] else []
        gt_list = [d["gt_box"]] if d["gt_box"] else None
        annotated_path = annotated_dir / f"sample{i:03d}.png"
        plot_predictions(image, pred_list, gt_list, out_path=annotated_path,
                          title=f"{d['error_type']} (IoU={d['iou']:.2f})")

        # Condition (c)'s crop: padded UNION of whichever box(es) this
        # sample has, boxes re-expressed in the crop's own coordinate
        # frame (shift_box) before being drawn, so plot_predictions
        # draws them in the right place on the CROPPED image, not where
        # they'd be on the full one.
        present_boxes = [b["bbox"] for b in (d["pred_box"], d["gt_box"]) if b]
        cx1, cy1, cx2, cy2 = union_crop_region(image, present_boxes, args.crop_pad_frac)
        cropped_image = image.crop((cx1, cy1, cx2, cy2))
        pred_cropped = [shift_box(d["pred_box"], cx1, cy1)] if d["pred_box"] else []
        gt_cropped = [shift_box(d["gt_box"], cx1, cy1)] if d["gt_box"] else None
        cropped_annotated_path = cropped_dir / f"sample{i:03d}.png"
        plot_predictions(cropped_image, pred_cropped, gt_cropped, out_path=cropped_annotated_path,
                          title=f"{d['error_type']} (cropped, IoU={d['iou']:.2f})")

        question = build_question(hprompts, d)
        pred_bbox_text = f"{[round(v, 1) for v in d['pred_box']['bbox']]}" if d["pred_box"] else "(none predicted)"
        gt_bbox_text = f"{[round(v, 1) for v in d['gt_box']['bbox']]}" if d["gt_box"] else "(no real instance here)"
        question_a = question + hprompts["detection_skill_raw_coords_suffix"].format(
            width=image.width, height=image.height, pred_bbox_text=pred_bbox_text, gt_bbox_text=gt_bbox_text,
        )
        question_b = question + hprompts["detection_skill_overlay_suffix"]
        question_c = question + hprompts["detection_skill_overlay_suffix"]

        # Only the image loading/plotting (no model needed) happens here —
        # the actual chat-template-rendered prompts are built AFTER the
        # model loads below, in one pass, not duplicated before and after.
        prepared.append({
            "index": i, "discrepancy": d, "question": question,
            "image_path": img_path, "annotated_path": annotated_path, "cropped_path": cropped_annotated_path,
            "question_a": question_a, "question_b": question_b, "question_c": question_c,
        })

    if not prepared:
        raise SystemExit("ERROR: no sampled images were found on disk — nothing to generate.")

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="detection_skill_representation_ablation",
    )
    for p in prepared:
        image = PILImage.open(p["image_path"]).convert("RGB")
        annotated_image = PILImage.open(p["annotated_path"]).convert("RGB")
        cropped_image = PILImage.open(p["cropped_path"]).convert("RGB")
        p["prompt_a"] = build_multimodal_prompt(tokenizer, image, p["question_a"], system_prompt,
                                                 enable_thinking=args.enable_thinking, chat_template=chat_template)
        p["prompt_b"] = build_multimodal_prompt(tokenizer, annotated_image, p["question_b"], system_prompt,
                                                 enable_thinking=args.enable_thinking, chat_template=chat_template)
        p["prompt_c"] = build_multimodal_prompt(tokenizer, cropped_image, p["question_c"], system_prompt,
                                                 enable_thinking=args.enable_thinking, chat_template=chat_template)

    gen_params = SamplingParams(
        temperature=0.0, max_tokens=args.gen_max_tokens, repetition_penalty=1.1,
        skip_special_tokens=not args.enable_thinking,
    )
    outputs_a = llm.generate([p["prompt_a"] for p in prepared], sampling_params=gen_params)
    outputs_b = llm.generate([p["prompt_b"] for p in prepared], sampling_params=gen_params)
    outputs_c = llm.generate([p["prompt_c"] for p in prepared], sampling_params=gen_params)
    destroy_llm(llm)
    del tokenizer

    for p, out_a, out_b, out_c in zip(prepared, outputs_a, outputs_b, outputs_c):
        raw_a, raw_b, raw_c = out_a.outputs[0].text, out_b.outputs[0].text, out_c.outputs[0].text
        _, text_a = extract_channel_thought(raw_a) if args.enable_thinking else ("", raw_a)
        _, text_b = extract_channel_thought(raw_b) if args.enable_thinking else ("", raw_b)
        _, text_c = extract_channel_thought(raw_c) if args.enable_thinking else ("", raw_c)
        p["response_a"] = text_a.strip()
        p["response_b"] = text_b.strip()
        p["response_c"] = text_c.strip()

    render_comparison_report(prepared, out_dir)
    print(f"\n  {len(prepared)} comparison image(s) written to {out_dir}")


def render_comparison_report(prepared: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for p in prepared:
        d = p["discrepancy"]
        annotated_image = PILImage.open(p["annotated_path"]).convert("RGB")
        cropped_image = PILImage.open(p["cropped_path"]).convert("RGB")

        fig = plt.figure(figsize=(11, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 4.5, 4.5], hspace=0.2, wspace=0.05)

        ax_q = fig.add_subplot(gs[0, :])
        ax_q.axis("off")
        ax_q.text(0.5, 0.5, "\n".join(textwrap.wrap(p["question"], width=140)),
                   ha="center", va="center", fontsize=10, wrap=True)

        ax_full = fig.add_subplot(gs[1, 0])
        ax_full.imshow(annotated_image)
        ax_full.axis("off")
        ax_full.set_title(f"FULL — {d['error_type']} (IoU={d['iou']:.2f})\n"
                           f"{d['row']['agml_dataset']}/{d['row']['image_name']}", fontsize=8.5)

        ax_crop = fig.add_subplot(gs[1, 1])
        ax_crop.imshow(cropped_image)
        ax_crop.axis("off")
        ax_crop.set_title("CROPPED (padded union of box(es))", fontsize=8.5)

        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")
        a_wrapped = "\n".join(textwrap.wrap(p["response_a"], width=160))
        b_wrapped = "\n".join(textwrap.wrap(p["response_b"], width=160))
        c_wrapped = "\n".join(textwrap.wrap(p["response_c"], width=160))
        ax_text.text(0.0, 1.00, f"(a) RAW COORDINATES (full image) condition:\n{a_wrapped}",
                      ha="left", va="top", fontsize=8, family="monospace", wrap=True)
        ax_text.text(0.0, 0.64, f"(b) VISUAL OVERLAY (full image) condition:\n{b_wrapped}",
                      ha="left", va="top", fontsize=8, family="monospace", wrap=True)
        ax_text.text(0.0, 0.28, f"(c) VISUAL OVERLAY (cropped) condition:\n{c_wrapped}",
                      ha="left", va="top", fontsize=8, family="monospace", wrap=True)

        out_path = out_dir / f"comparison_sample{p['index']:03d}_{d['error_type']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--ckpt", required=True,
                         help="A detection-multiclass checkpoint CSV (DET_CKPT_FIELDS-shaped — same "
                              "schema detection-baseline uses, reused directly by detection-multiclass, "
                              "not a different one).")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="Used both for matching (false_positive/missed classification) and as "
                             "the FLOOR for loose_match (a matched pair below --iou_threshold is "
                             "already a non-match handled elsewhere; loose_match is specifically a "
                             "REAL match whose IoU is still under _LOOSE_IOU_THRESHOLD=0.7).")
    parser.add_argument("--crop_pad_frac", type=float, default=0.25,
                         help="Condition (c)'s crop padding, relative to the padded UNION box's own "
                              "width/height (see union_crop_region) — wider than "
                              "detection_box_verifier.py's own single-box crop_box default (0.15) "
                              "since a loose_match/worked sample's two boxes are themselves already "
                              "spread apart, before any padding is added.")
    parser.add_argument("--detection_skill_prompts", default="src/config/detection_skill_prompts.yaml")
    parser.add_argument("--out_dir", default="outputs_final/detection_skill_representation_ablation/")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                         help="OFF by default — this is a short, text-only-output explanation task "
                              "with no established need for extended reasoning, same reasoning "
                              "heldout_skill_optimize.py's --edit_enable_thinking now defaults off "
                              "for too (and defaulting thinking on for an analogous task caused a "
                              "real bug there — see that script's own flag help for the full story).")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=512,
                         help="All three conditions ask for a short 2-4 sentence answer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
