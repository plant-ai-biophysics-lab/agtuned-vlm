"""
Build the combined GRPO training pool: (image, question, ground_truth, task)
rows pooled across all 4 tasks' held-out `lessons` splits — disease_pest,
quality, species (classification) and detection (object detection).

Deliberately reuses each domain's REAL question-building function from
src/heldout_zeroshot.py / src/heldout_detection_zeroshot.py rather than
reimplementing question text here, so the GRPO policy is trained on the
EXACT same question framing its skill-memory counterpart is evaluated
against (see the approved plan's Data section for why this matters for the
comparison's validity).

Does NOT need the `explanation` text heldout-explain-traces generates —
GRPO samples its own rollouts fresh from the policy at train time and only
needs (image, question, ground_truth) to score them, not a pre-written
explanation. See the approved plan (~/.claude/plans/linear-giggling-honey.md)
for the full reasoning.

Writes one combined JSONL, each row tagged with "task" ("disease_pest" /
"quality" / "species" / "detection") so the training script can apply the
right reward function and can equal-probability-sample across tasks via
datasets.interleave_datasets (not baked into row order here — that's a
training-time decision, not a data-prep one).

Row shape:
  classification: {"task", "image_path", "question", "ground_truth",
                    "crop", "agml_dataset"}
  detection:       {"task": "detection", "image_path", "question",
                     "ground_truth" (JSON-encoded GT boxes list),
                     "crop", "agml_dataset"}

Usage:
    uv run python experiments/ag_finetune/data_prep/build_grpo_unified_data.py \
        --output outputs/grpo_unified_lessons.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.common.io import load_prompts  # noqa: E402
from src.heldout_zeroshot import (  # noqa: E402
    _cls_question, _quality_question, _species_question, build_class_list_text,
)
from src.heldout_detection_zeroshot import det_question_heldout, load_dataset_meta  # noqa: E402

_CLASSIFICATION_DOMAINS = [
    ("disease_pest", "dataset/heldout_selected_images.json"),
    ("quality", "dataset/heldout_quality_selected_images.json"),
    ("species", "dataset/heldout_species_selected_images.json"),
]
_DETECTION_MANIFEST = "dataset/detection_heldout_selected_images.json"
_DETECTION_DATASETS_YAML = "src/config/detection_datasets.yaml"


def build_classification_rows(domain: str, manifest_path: Path, prompts: dict) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"]
    # Same convention heldout_zeroshot.py's own run_judge_phase call uses
    # for held-out domains (extra_keys={"class_list": "class_list_text"}) —
    # ALL rows regardless of split, so the class list itself never differs
    # between what built the skill doc and what trains GRPO.
    class_list_text = build_class_list_text(rows)

    if domain == "quality":
        question_fn = lambda crop: _quality_question(crop, prompts)
    elif domain == "species":
        question_fn = lambda crop: _species_question(prompts)
    else:
        question_fn = _cls_question

    out = []
    for r in rows.values():
        if r.get("split") != "lessons":
            continue
        crop = r.get("crop", "")
        out.append({
            "task": domain,
            "image_path": r["image_path"],
            "question": question_fn(crop),
            "ground_truth": "; ".join(r.get("diagnosis") or []),
            "class_list_text": class_list_text,
            "crop": crop,
            "agml_dataset": r.get("agml_dataset", ""),
        })
    return out


def build_detection_rows(manifest_path: Path, dataset_meta: dict, detection_prompts: dict,
                          use_json: bool) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"]

    out = []
    skipped_no_meta = 0
    for r in rows.values():
        if r.get("split") != "lessons":
            continue
        ds_name = r.get("agml_dataset", "")
        meta = dataset_meta.get(ds_name)
        if meta is None:
            skipped_no_meta += 1
            continue
        question = det_question_heldout(meta["classes"], meta["crop"], detection_prompts, use_json=use_json)
        out.append({
            "task": "detection",
            "image_path": r["image_path"],
            "question": question,
            "ground_truth": json.dumps(r.get("boxes") or []),
            "class_list_text": "",  # unused for detection rows — kept for uniform schema
            "crop": meta["crop"],
            "agml_dataset": ds_name,
        })
    if skipped_no_meta:
        print(f"  [warn] {skipped_no_meta} detection row(s) skipped — agml_dataset not found in "
              f"{_DETECTION_DATASETS_YAML} (shouldn't happen for the 15 Stage-4 held-out datasets).")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="src/config/prompts.yaml")
    ap.add_argument("--detection_prompts", default="src/config/prompts_detection.yaml",
                     help="Separate prompts file detection uses for its own question templates "
                          "(detection_multiclass/detection_multiclass_json keys) — confirmed via "
                          "heldout_detection_zeroshot.py/detection_multiclass.py's own --prompts "
                          "default; these keys do NOT exist in the classification prompts.yaml.")
    ap.add_argument("--output", default="outputs/grpo_unified_lessons.jsonl")
    ap.add_argument("--policy_model", default="unsloth/gemma-4-31B-it",
                     help="Decides whether the detection question asks for box_2d JSON or <box> tags "
                          "— must match whatever --model stage4_grpo_unified.py trains, via the SAME "
                          "bbox_parse.uses_json_box_format(model_id) check that script's own reward "
                          "function uses, so the prompt and the parser never disagree about format.")
    args = ap.parse_args()

    from src.common.bbox_parse import uses_json_box_format
    use_json = uses_json_box_format(args.policy_model)
    print(f"  Detection question format: use_json={use_json} (decided from --policy_model={args.policy_model!r})")

    prompts = load_prompts(Path(args.prompts))
    detection_prompts = load_prompts(Path(args.detection_prompts))

    all_rows: list[dict] = []
    for domain, manifest_path in _CLASSIFICATION_DOMAINS:
        rows = build_classification_rows(domain, Path(manifest_path), prompts)
        print(f"  {domain}: {len(rows)} lessons-split row(s)")
        all_rows.extend(rows)

    dataset_meta = load_dataset_meta(Path(_DETECTION_DATASETS_YAML))
    det_rows = build_detection_rows(Path(_DETECTION_MANIFEST), dataset_meta, detection_prompts, use_json)
    print(f"  detection: {len(det_rows)} lessons-split row(s)")
    all_rows.extend(det_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n  Total: {len(all_rows)} row(s) across {len(_CLASSIFICATION_DOMAINS) + 1} tasks")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
