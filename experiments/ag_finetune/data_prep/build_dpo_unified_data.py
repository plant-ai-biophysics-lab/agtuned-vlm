"""
Build the combined DPO training pool: (image, question, chosen, rejected)
rows pooled across all 4 tasks — disease_pest, quality, species
(classification) and detection — for the preference-learning alternative to
GRPO (see the conversation that led to this script; GRPO's reward collapses
to zero-signal whenever all K rollouts for a prompt land on the same reward,
which is common for open-set free-text classification against hundreds of
possible labels, and detection's reward never needed a judge call at all so
a GRPO reward function already existed for it — this script is the DPO data
side of the comparison, letting classification move to DPO while detection
can go either way).

Unlike build_grpo_unified_data.py, this does NOT sample rollouts at train
time — every (chosen, rejected) pair is precomputed here, offline, from
checkpoints Stage 3/4's explain-traces pipeline already produced. No live
judge server needed at DPO train time at all (a real simplification vs. the
GRPO GPU layout).

Classification pairs (disease_pest/quality/species), source: each domain's
`heldout_explain_traces` checkpoint (one row PER IMAGE — a single zero-shot
pass only ever produces one judged completion per image, never both a
correct and incorrect one for the same image, so BOTH sides of every pair
are built from that one row, not pulled from two different images):
  - rejected = the row's own `prediction` field verbatim — the model's real
    completion for that image, whether or not it was correct.
  - chosen   = the row's own `explanation` field (an explainer model's
    analysis, grounded in the dataset's expert `details` text, generated
    WITH knowledge of the ground truth — confirmed via
    src/config/heldout_prompts.yaml's `heldout_explain_correct_prompt`/
    `heldout_explain_miss_prompt`), with the final `<answer>` tag appended.
    Included for BOTH trace_type values, not just `didnt_work`:
      - didnt_work: explanation describes what evidence SHOULD have been
        used — directly corrects a wrong answer.
      - worked: explanation CONFIRMS the correct evidence (and explicitly
        flags a "lucky guess" if the model's own stated reasoning was
        actually unsound despite landing on the right answer) — this is a
        MORE reliable articulation of the same correct conclusion than the
        model's own raw first-pass guess, so including it teaches the
        policy to prefer well-grounded reasoning even when it already gets
        the final answer right, not just to fix outright errors.
    Known, accepted limitation (not fixed here): `explanation` is written in
    third-person evaluator voice ("The diagnosis is correct because...",
    "...supports the expert's observation") rather than the policy's own
    first-person diagnostic register the real `prediction` field uses — a
    real style difference between chosen/rejected. Left as-is per the same
    reasoning accepted for detection's reasoning-preamble below: a nudge
    toward more grounded, explicit reasoning is a desirable direction, not
    a shortcut to suppress.

Detection pairs, source: TWO checkpoints together (the per-box
heldout_detection_explain_traces checkpoint alone is NOT enough — it holds
no per-image `raw_prediction` field, only individual box-level explanations
of errors already matched/unmatched by IoU):
  - rejected = the IMAGE-level `heldout_detection_zeroshot` checkpoint's own
    `raw_prediction` field verbatim (real, bare box_2d JSON — the detection
    prompt explicitly asks for ONLY a JSON array, no reasoning).
  - chosen   = every per-box explanation for that image (from the per-box
    explain-traces checkpoint: false_positive/missed/wrong_class_label/
    loose_match rows, grouped by (agml_dataset, image_name)) collated into
    one paragraph, followed by the ground-truth boxes re-rendered into the
    exact box_2d JSON schema the prompt expects (bbox_parse.py's
    normalize_box_1000_y_first — the inverse of denormalize_box_1000).
    Accepted structural asymmetry: chosen has a reasoning preamble, rejected
    doesn't (since the original generation was never asked for one) — see
    this module's own conversation history; a preamble is treated as
    strictly desirable here, not a shortcut to guard against.
  - Only images below --f1_threshold are included — a perfect detection has
    no error to collate an explanation from and nothing to prefer over.

Usage:
    uv run python experiments/ag_finetune/data_prep/build_dpo_unified_data.py \
        --output outputs/dpo_unified_pairs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.io import load_prompts  # noqa: E402
from src.heldout_zeroshot import _cls_question, _quality_question, _species_question  # noqa: E402
from src.heldout_detection_zeroshot import det_question_heldout, load_dataset_meta  # noqa: E402

# (domain, explain_traces_ckpt_dir) — each domain's own heldout_explain_traces
# run; quality/species default to a non-colliding dir name since
# heldout_explain_traces.py's own --output_dir default
# ("outputs_final/heldout_explain_traces/") is domain-agnostic — a real run
# for a 2nd/3rd domain MUST have been pointed at a different --output_dir,
# same convention every other domain-specific stage in this pipeline already
# uses (e.g. "_with_quality_skill"); override via --{domain}_explain_dir if
# a different path was actually used.
# (domain, default_explain_dir, images_root) — images_root matches each
# domain's own download_heldout_*.py output convention (confirmed earlier
# this session: flat images_root/<key>.jpg, no per-dataset subdirectory,
# for all 3 classification sources alike).
_CLASSIFICATION_DOMAINS = [
    ("disease_pest", "outputs_final/heldout_explain_traces", "dataset/images_heldout"),
    ("quality", "outputs_final/heldout_explain_traces_quality", "dataset/images_heldout_quality"),
    ("species", "outputs_final/heldout_explain_traces_species", "dataset/images_heldout_species"),
]
_DETECTION_DATASETS_YAML = "src/config/detection_datasets.yaml"


def safe_model_name(model_id: str) -> str:
    # Mirrors src/common/vision_eval.py's own safe_model_name exactly
    # (duplicated here rather than imported to avoid pulling vision_eval's
    # heavier deps into a pure data-prep script).
    return model_id.replace("/", "_")


def build_classification_pairs(domain: str, explain_dir: Path, images_root: str,
                                model_id: str, prompts: dict) -> list[dict]:
    ckpt = explain_dir / safe_model_name(model_id) / f"ckpt_{safe_model_name(model_id)}.csv"
    if not ckpt.exists():
        print(f"  [skip] {domain}: no explain-traces checkpoint at {ckpt}")
        return []
    with open(ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if domain == "quality":
        question_fn = lambda crop: _quality_question(crop, prompts)
    elif domain == "species":
        question_fn = lambda crop: _species_question(prompts)
    else:
        question_fn = _cls_question

    out = []
    n_worked = n_missed = 0
    for r in rows:
        explanation = (r.get("explanation") or "").strip()
        prediction = (r.get("prediction") or "").strip()
        ground_truth = (r.get("judge_gt_hint") or r.get("class_label") or "").strip()
        if not explanation or not prediction or not ground_truth:
            continue
        crop = r.get("crop", "")
        out.append({
            "task": domain,
            "image_path": f"{images_root}/{r['image_name']}",
            "question": question_fn(crop),
            "chosen": f"{explanation}\n\n<answer>{ground_truth}</answer>",
            "rejected": prediction,
            "crop": crop,
            "agml_dataset": r.get("agml_dataset", ""),
        })
        if r.get("trace_type") == "worked":
            n_worked += 1
        else:
            n_missed += 1
    print(f"  {domain}: {len(out)} pair(s) ({n_worked} worked, {n_missed} didn't)")
    return out


def _collate_detection_explanations(explain_rows: list[dict]) -> dict[tuple[str, str], str]:
    """{(agml_dataset, image_name): collated explanation text} — one entry
    per image, joining every per-box error explanation for that image into
    one paragraph (see this module's own docstring for why)."""
    by_image: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in explain_rows:
        key = (r["agml_dataset"], r["image_name"])
        exp = (r.get("explanation") or "").strip()
        if not exp:
            continue
        by_image[key].append(f"- ({r.get('error_type', '')}, {r.get('category', '')}) {exp}")
    return {k: "\n".join(v) for k, v in by_image.items()}


def build_detection_pairs(zeroshot_ckpt: Path, explain_ckpt: Path, images_root: Path,
                           dataset_meta: dict, detection_prompts: dict, use_json: bool,
                           f1_threshold: float) -> list[dict]:
    from PIL import Image as PILImage
    from src.common.bbox_parse import normalize_box_1000_y_first

    with open(zeroshot_ckpt, newline="", encoding="utf-8") as f:
        zeroshot_rows = list(csv.DictReader(f))
    with open(explain_ckpt, newline="", encoding="utf-8") as f:
        explain_rows = list(csv.DictReader(f))
    collated = _collate_detection_explanations(explain_rows)

    out = []
    n_no_explanation = n_no_meta = n_no_image = n_skipped_perfect = 0
    for r in zeroshot_rows:
        try:
            f1 = float(r.get("f1_at_iou50", "1.0") or "1.0")
        except ValueError:
            f1 = 1.0
        if f1 >= f1_threshold:
            n_skipped_perfect += 1
            continue

        key = (r["agml_dataset"], r["image_name"])
        explanation = collated.get(key)
        if not explanation:
            n_no_explanation += 1
            continue

        meta = dataset_meta.get(r["agml_dataset"])
        if meta is None:
            n_no_meta += 1
            continue

        img_path = images_root / r["agml_dataset"] / r["image_name"]
        if not img_path.exists():
            n_no_image += 1
            continue
        with PILImage.open(img_path) as img:
            img_w, img_h = img.width, img.height

        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        box_2d_list = [
            {"box_2d": normalize_box_1000_y_first(b["bbox"], img_w, img_h), "label": b.get("category_name", "")}
            for b in gt_boxes
        ]
        chosen = f"{explanation}\n\n{json.dumps(box_2d_list)}"
        rejected = (r.get("raw_prediction") or "").strip()
        if not rejected:
            continue

        question = det_question_heldout(meta["classes"], meta["crop"], detection_prompts, use_json=use_json)
        out.append({
            "task": "detection",
            "image_path": str(img_path),
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "crop": meta["crop"],
            "agml_dataset": r["agml_dataset"],
        })

    print(f"  detection: {len(out)} pair(s) "
          f"(skipped: {n_skipped_perfect} perfect/above-threshold, {n_no_explanation} no explanation, "
          f"{n_no_meta} no dataset meta, {n_no_image} image not found)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="src/config/prompts.yaml")
    ap.add_argument("--detection_prompts", default="src/config/prompts_detection.yaml")
    ap.add_argument("--explainer_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                     help="Model used to GENERATE the explain-traces checkpoints (decides the "
                          "checkpoint subdirectory name via safe_model_name) — not necessarily the "
                          "same as --policy_model below, which decides the POLICY's own question "
                          "format/box convention.")
    for domain, default_dir, default_images_root in _CLASSIFICATION_DOMAINS:
        ap.add_argument(f"--{domain}_explain_dir", default=default_dir)
        ap.add_argument(f"--{domain}_images_root", default=default_images_root)
    ap.add_argument("--detection_zeroshot_ckpt", default="outputs_final/heldout_detection_zeroshot/"
                     "RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv",
                     help="IMAGE-level checkpoint (one row per image, has raw_prediction/gt_boxes_json/"
                          "f1_at_iou50) — NOT the per-box explain-traces checkpoint below.")
    ap.add_argument("--detection_explain_ckpt", default="outputs_final/heldout_detection_explain_traces/"
                     "RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv",
                     help="Per-BOX checkpoint (one row per error instance) — collated per-image here.")
    ap.add_argument("--detection_images_root", default="dataset/images_detection_heldout")
    ap.add_argument("--f1_threshold", type=float, default=0.95,
                     help="Detection images with f1_at_iou50 >= this are already-near-perfect — "
                          "excluded, nothing meaningful to prefer over.")
    ap.add_argument("--policy_model", default="unsloth/gemma-4-31B-it",
                     help="Decides detection's box_2d vs <box>-tag question format via the SAME "
                          "bbox_parse.uses_json_box_format(model_id) check the GRPO data-prep script "
                          "and the training scripts both use — must match whatever --model the DPO "
                          "training script trains.")
    ap.add_argument("--output", default="outputs/dpo_unified_pairs.jsonl")
    args = ap.parse_args()

    from src.common.bbox_parse import uses_json_box_format
    use_json = uses_json_box_format(args.policy_model)
    print(f"  Detection question format: use_json={use_json} (decided from --policy_model={args.policy_model!r})")

    prompts = load_prompts(Path(args.prompts))
    detection_prompts = load_prompts(Path(args.detection_prompts))

    all_rows: list[dict] = []
    for domain, _, _ in _CLASSIFICATION_DOMAINS:
        explain_dir = Path(getattr(args, f"{domain}_explain_dir"))
        images_root = getattr(args, f"{domain}_images_root")
        all_rows.extend(build_classification_pairs(domain, explain_dir, images_root, args.explainer_model, prompts))

    zeroshot_ckpt = Path(args.detection_zeroshot_ckpt)
    explain_ckpt = Path(args.detection_explain_ckpt)
    if zeroshot_ckpt.exists() and explain_ckpt.exists():
        dataset_meta = load_dataset_meta(Path(_DETECTION_DATASETS_YAML))
        all_rows.extend(build_detection_pairs(
            zeroshot_ckpt, explain_ckpt, Path(args.detection_images_root),
            dataset_meta, detection_prompts, use_json, args.f1_threshold,
        ))
    else:
        missing = [str(p) for p in (zeroshot_ckpt, explain_ckpt) if not p.exists()]
        print(f"  [skip] detection: missing checkpoint(s): {missing}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n  Total: {len(all_rows)} pair(s)")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
