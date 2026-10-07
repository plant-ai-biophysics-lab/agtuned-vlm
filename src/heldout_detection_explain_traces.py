"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 2 —
explain traces on Step 1's LESSONS-split checkpoint, box-level, ALL
discrepancies AND successes (not just misses) — detection analog of
heldout_explain_traces.py (Stage 3's own Step 2).

Ground truth is already in Step 1's checkpoint (gt_boxes_json) — no
reveal step needed, unlike classification's design (which hides the
real label during Step 1's own prediction and only reveals it here);
detection's Step 1 scoring already requires the ground-truth boxes, so
there's nothing new to "reveal", just explain.

Per-box, not per-row, same convention detection_skill_representation_
ablation.py's own Step -1 established and detection_box_verifier.py's
own grading convention before that: ONE box-level item per explanation
call, never a whole image's full box set at once (ambiguous about which
sentence refers to which error otherwise).

Failure/success taxonomy (4 categories, `error_type`) — matches
detection_box_refine.py's own already-validated 3-criterion rubric by
name, plus one more (`worked`) for Stage 3's own "explain what worked,
not just misses" design choice, mirrored here:
  - false_positive     — criterion 1, "not a real instance" (no matching gt)
  - missed              — a real gt instance with no matching prediction
                           at all (detection's analog of classification's
                           "false Healthy" — the model didn't even try)
  - loose_match         — criterion 2/3, "not tight" (too loose/large/cut
                           off, OR covers >1 real instance and needs
                           splitting) — a REAL match (passes --iou_threshold)
                           but below --loose_iou_threshold; the specific
                           "too loose" vs. "needs splitting" distinction is
                           left to the model's own free-text explanation
                           (Step 3's map-reduce summary is what's expected
                           to surface "clustered instances need splitting"
                           as a pattern, not a separately-coded detector).
  - worked              — a tight match (>= --loose_iou_threshold).

NO separate "wrong_class_label" category, despite predicted boxes
carrying a `category_name` — deliberately, matching a precedent already
established everywhere else in this pipeline: `detection_baseline.py`'s
own `score_row` (`match_boxes` + `precision_recall_f1`) NEVER reads
`category_name`, because `detection-multiclass` predicts an
open-vocabulary free-text SUB-class (e.g. "grape: ripe cluster") while
ground truth only ever carries the dataset's one coarse class label
(e.g. "grape") — the two were never meant to be string-compared. A
first version of this module DID compare them directly and produced
real false positives: a correct "grape: ripe cluster" prediction
flagged as `wrong_class_label` against ground truth whose own label was
literally `"1"` (`Project-AgML/grape_detection_syntheticday`'s own
upstream `ClassLabel(names=['grape', '1'])` — a genuinely uninformative
class name, not something this pipeline can fix), and more generally
ANY correct prediction whose free-text sub-class wasn't a byte-for-byte
match to the GT's single coarse label (e.g. "grape" vs. "grape: ripe
cluster" — also not a string match, also a correct prediction). Tight
IoU is all that's checked; a loose-string-matching compromise was
considered and rejected as still unable to resolve genuinely
uninformative GT labels like `"1"`, so category agreement is dropped
entirely rather than partially fixed.

ONE pass over match_boxes per row produces all 4 categories directly
(not Step -1's own find_discrepancies, which only covers false_positive/
missed/loose_match — calling it here AND separately checking `matches`
for `worked` would call match_boxes on the same row twice for no
reason).

box_idx (0, 1, 2, ... per image, in the FIXED deterministic order this
module enumerates items) plus error_type stand in for class_label in
load_ckpt's hardcoded 3-field resume key (agml_dataset/class_label/
image_name) — same repurposing trick heldout_zeroshot.py's own
HELDOUT_CKPT_FIELDS used for a different reason, needed here because
several box-level items share one (agml_dataset, image_name) pair,
which alone isn't a unique resume key. Deterministic given the same
checkpoint CSV: pred_boxes_json/gt_boxes_json are fixed JSON text, and
match_boxes is a deterministic greedy algorithm, so re-running produces
the identical box_idx assignment every time.

Only condition (a) — raw coordinates + plain image — Step -1's own
settled winner (docs/stage4_detection_context_memory.md): no overlay
rendering needed here at all.

Usage:
    uv run python main.py heldout-detection-explain-traces \\
        --ckpt outputs_final/heldout_detection_zeroshot_lessons/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
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

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes
from src.common.chat_prompt import build_multimodal_prompt
from src.common.metrics import extract_channel_thought
from src.common.progress import ProgressBar
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

DET_EXPLAIN_CKPT_FIELDS = [
    "agml_dataset", "class_label", "image_name", "error_type", "category", "box_idx",
    "pred_bbox_json", "gt_bbox_json", "iou", "question", "explanation",
]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def enumerate_items(rows: list[dict], iou_threshold: float, loose_iou_threshold: float) -> list[dict]:
    """One entry per box-level item (discrepancy OR successful match) per
    row: {"row", "pred_box", "gt_box", "error_type", "iou", "box_idx"} —
    see this module's docstring for the 4 error_type values and the
    deterministic box_idx ordering (false_positive, then missed, then
    every match in match_boxes' own return order). category_name is
    deliberately NOT compared here — see this module's own docstring for
    why (tight IoU is "worked", loose IoU is "loose_match", regardless of
    what either box's category_name says)."""
    items: list[dict] = []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        box_idx = 0
        for pi in unmatched_pred:
            items.append({"row": r, "pred_box": pred_boxes[pi], "gt_box": None,
                          "error_type": "false_positive", "iou": 0.0, "box_idx": box_idx})
            box_idx += 1
        for gi in unmatched_gt:
            items.append({"row": r, "pred_box": None, "gt_box": gt_boxes[gi],
                          "error_type": "missed", "iou": 0.0, "box_idx": box_idx})
            box_idx += 1
        for pi, gi, iou in matches:
            error_type = "worked" if iou >= loose_iou_threshold else "loose_match"
            items.append({"row": r, "pred_box": pred_boxes[pi], "gt_box": gt_boxes[gi],
                          "error_type": error_type, "iou": iou, "box_idx": box_idx})
            box_idx += 1
    return items


def build_question(hprompts: dict, item: dict) -> str:
    error_type = item["error_type"]
    if error_type == "false_positive":
        return hprompts["detection_skill_question_false_positive"].format(
            category=item["pred_box"]["category_name"])
    if error_type == "missed":
        return hprompts["detection_skill_question_missed"].format(
            category=item["gt_box"]["category_name"])
    if error_type == "loose_match":
        return hprompts["detection_skill_question_loose_match"].format(
            category=item["gt_box"]["category_name"], iou=item["iou"])
    return hprompts["detection_skill_question_worked"].format(
        category=item["gt_box"]["category_name"], iou=item["iou"])


def build_prompt_text(hprompts: dict, item: dict, width: int, height: int) -> str:
    question = build_question(hprompts, item)
    pred_bbox_text = f"{[round(v, 1) for v in item['pred_box']['bbox']]}" if item["pred_box"] else "(none predicted)"
    gt_bbox_text = f"{[round(v, 1) for v in item['gt_box']['bbox']]}" if item["gt_box"] else "(no real instance here)"
    return question + hprompts["detection_skill_raw_coords_suffix"].format(
        width=width, height=height, pred_bbox_text=pred_bbox_text, gt_bbox_text=gt_bbox_text,
    )


def run_model(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.detection_skill_prompts))
    system_prompt = hprompts["detection_skill_system"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    with open(args.ckpt, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    if not input_rows:
        raise SystemExit(f"ERROR: {args.ckpt} has no rows.")

    items = enumerate_items(input_rows, args.iou_threshold, args.loose_iou_threshold)
    if not items:
        raise SystemExit(f"ERROR: no box-level items found in {args.ckpt} — nothing to explain.")
    counts = {t: sum(1 for it in items if it["error_type"] == t) for t in
              ("false_positive", "missed", "loose_match", "worked")}
    print(f"  {len(items)} box-level item(s) from {len(input_rows)} row(s): {counts}")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=DET_EXPLAIN_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} item(s) already explained in {ckpt_path.name}")

    pending = [it for it in items
               if (it["row"]["agml_dataset"], f"{it['error_type']}_{it['box_idx']}", it["row"]["image_name"])
               not in seen]
    if not pending:
        print("  Nothing to do — all items already checkpointed.")
        return
    print(f"  {len(pending)} pending item(s)")

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="heldout_detection_explain_traces",
    )

    chunk_size = max(args.chunk_size, 1)
    pbar = ProgressBar(total=len(pending), desc="Explaining")
    for i in range(0, len(pending), chunk_size):
        chunk = pending[i : i + chunk_size]
        images, prompts_batch, valid = [], [], []
        for it in chunk:
            img_path = Path(args.images_root) / it["row"]["agml_dataset"] / it["row"]["image_name"]
            if not img_path.exists():
                print(f"  [skip] {it['row']['agml_dataset']}/{it['row']['image_name']} — image not found at {img_path}")
                continue
            image = PILImage.open(img_path).convert("RGB")
            text = build_prompt_text(hprompts, it, image.width, image.height)
            prompts_batch.append(build_multimodal_prompt(
                tokenizer, image, text, system_prompt,
                enable_thinking=args.enable_thinking, chat_template=chat_template))
            valid.append(it)

        if not prompts_batch:
            pbar.update(len(chunk))
            continue

        params = SamplingParams(
            temperature=0.0, max_tokens=args.gen_max_tokens, repetition_penalty=1.1,
            skip_special_tokens=not args.enable_thinking,
        )
        outputs = llm.generate(prompts_batch, sampling_params=params)

        for it, out in zip(valid, outputs):
            raw = out.outputs[0].text
            _, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
            category = (it["gt_box"] or it["pred_box"])["category_name"]
            row = {
                "agml_dataset": it["row"]["agml_dataset"],
                "class_label": f"{it['error_type']}_{it['box_idx']}",
                "image_name": it["row"]["image_name"],
                "error_type": it["error_type"],
                "category": category,
                "box_idx": it["box_idx"],
                "pred_bbox_json": json.dumps(it["pred_box"]) if it["pred_box"] else "",
                "gt_bbox_json": json.dumps(it["gt_box"]) if it["gt_box"] else "",
                "iou": round(it["iou"], 4),
                "question": build_question(hprompts, it),
                "explanation": cleaned.strip(),
            }
            append_ckpt_row(ckpt_path, row, fields=DET_EXPLAIN_CKPT_FIELDS)
            rows.append(row)
        pbar.update(len(chunk))
    pbar.finish()
    destroy_llm(llm)
    del tokenizer

    out_counts = {t: sum(1 for r in rows if r["error_type"] == t) for t in
                  ("false_positive", "missed", "loose_match", "worked")}
    print(f"\n  {len(rows)} total item(s): {out_counts}")
    print(f"  Checkpoint: {ckpt_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--ckpt", required=True,
                         help="heldout-detection-zeroshot's own --split lessons checkpoint CSV — "
                              "ALL box-level items are processed, not just discrepancies.")
    parser.add_argument("--images_root", default="dataset/images_detection_heldout",
                         help="Same root Step 0's download_heldout_detection.py cached into "
                              "(<images_root>/<agml_dataset>/<image_name>).")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                         help="Matching threshold (false_positive/missed vs. a real match) — same "
                              "value Step 1's own scoring used, for consistency.")
    parser.add_argument("--loose_iou_threshold", type=float, default=0.7,
                         help="A real match below this (but above --iou_threshold) is loose_match; "
                              "at or above it, worked — category_name is never compared (see this "
                              "module's own docstring for why). Same threshold value Step -1's own "
                              "ablation used (_LOOSE_IOU_THRESHOLD in "
                              "detection_skill_representation_ablation.py).")
    parser.add_argument("--detection_skill_prompts", default="src/config/detection_skill_prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_detection_explain_traces/")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                         help="OFF by default — same reasoning detection_skill_representation_"
                              "ablation.py's own --enable_thinking default documents (short, "
                              "text-only explanation task, no established need for extended "
                              "reasoning; defaulting thinking on for an analogous document/short-"
                              "answer task caused a real bug in heldout_skill_optimize.py).")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=512,
                         help="Short 2-4 sentence answer, same budget Step -1's ablation used.")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
