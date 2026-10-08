"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 0b — download +
cache a held-out slice of AgML's own quality_identification datasets as
a lesson-building source for the quality skill memory, the same role
Digital Green plays for disease/pest (Step 0). Unlike disease/pest,
there's no external quality-labeled dataset to borrow, so 5 of AgML's 24
quality_identification datasets are reserved for this instead (`skip:
true` in src/config/datasets.yaml, each tagged "HELD OUT for Stage 3
quality skill-memory lesson-building" — never sampled into the main AgML
quality_identification eval):
  - VegNet_quality_classification (vegetable)
  - banana_bunch_maturity_classification (banana)
  - coffee_bean_quality_classification (coffee)
  - pomegranate_quality_classification (pomegranate)
  - riseholme_strawberry_classification_2021 (strawberry)
Chosen for crop diversity (5 distinct crops) and task-flavor diversity
(grading/maturity/ripeness/condition), all plain RGB photos (no thermal/
sensor modality, consistent with the main eval's image format) — the
remaining 19 quality_identification datasets are untouched by this
script and stay in the main AgML eval pool exactly as before.

Writes dataset/heldout_quality_selected_images.json in the EXACT same
row shape download_heldout_digitalgreen.py uses (image_path, crop,
diagnosis as a 1-element list, details, country, state, split) plus one
extra field (agml_dataset, the real originating sub-dataset name) — so
every downstream Stage 3 script (heldout_zeroshot.py,
heldout_explain_traces.py, heldout_summarize.py,
heldout_skill_optimize.py) works completely UNCHANGED via
--manifest/--images_root pointed here, with --domain quality selecting
the quality-appropriate question/prompts. "diagnosis" holds the
dataset's own cleaned class label (e.g. "grade 1", "ripe") as a
1-element list purely for schema compatibility with Digital Green's real
multi-label rows — never multi-valued here, since every AgML dataset is
one-label-per-image. "details"/"country"/"state" are always empty — none
of these 5 datasets carries expert free-text notes or location metadata
the way Digital Green does.

Row keys and cached image paths follow Digital Green's own flat
convention (images_root/<key>.jpg, no per-dataset subdirectory) — keys
are "<agml_name>__<row_idx>" (globally unique across all 5
sub-datasets) so heldout_zeroshot.py's `images_root / f"{k}.jpg"` image
lookup needs no changes at all.

--eval_frac (default 0.2) splits each sub-dataset's own rows into a
lesson-building slice and a held-out eval slice independently
(stratified by sub-dataset, which is already effectively stratified by
crop — each of the 5 covers exactly one crop), seeded by --seed.

Usage:
    uv run python scripts/download_heldout_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
from collections import defaultdict

from src.common.dataset_config import _load_dataset_safe, clean_class_list, find_image_column
from scripts.agml_hf.download_and_sample import find_label_column, resolve_class_names

# The 5 datasets.yaml entries marked skip:true for this purpose — kept as
# an explicit list here (not re-derived from datasets.yaml's skip flag,
# which could pick up an unrelated future skip:true addition) so this
# script's scope is self-contained and auditable at a glance.
HELD_OUT_DATASETS = [
    ("VegNet_quality_classification", "vegetable"),
    ("banana_bunch_maturity_classification", "banana"),
    ("coffee_bean_quality_classification", "coffee"),
    ("pomegranate_quality_classification", "pomegranate"),
    ("riseholme_strawberry_classification_2021", "strawberry"),
]


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"hf_repo_id": "Project-AgML/<per-row agml_dataset>", "rows": {}}


def assign_splits(row_keys_by_dataset: dict[str, list[str]], eval_frac: float, seed: int) -> dict[str, str]:
    """Same stratification strategy as download_heldout_digitalgreen.py's
    assign_splits — here stratified by sub-dataset (each already one
    crop) instead of by crop field, since the 5 sub-datasets are 1:1 with
    crop already."""
    rng = random.Random(seed)
    split_by_key: dict[str, str] = {}
    for name, keys in row_keys_by_dataset.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_eval = round(len(keys) * eval_frac)
        for i, key in enumerate(keys):
            split_by_key[key] = "eval" if i < n_eval else "lessons"
    return split_by_key


def main(args: argparse.Namespace) -> None:
    images_root = Path(args.images_root)
    images_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out_path)
    rows = manifest["rows"]
    n_before = len(rows)

    row_keys_by_dataset: dict[str, list[str]] = defaultdict(list)
    for agml_name, crop in HELD_OUT_DATASETS:
        repo_id = f"Project-AgML/{agml_name}"
        print(f"  Loading {repo_id}...")
        ds = _load_dataset_safe(repo_id, None, split="train")
        print(f"  {len(ds)} row(s) in {agml_name}.")
        label_col = find_label_column(ds.features)
        image_col = find_image_column(ds.features)
        class_names = resolve_class_names(ds, label_col)

        for idx in range(len(ds)):
            key = f"{agml_name}__{idx}"
            row_keys_by_dataset[agml_name].append(key)
            img_path = images_root / f"{key}.jpg"
            if key in rows and img_path.exists():
                continue
            raw_label = ds[idx][label_col]
            label = class_names[raw_label] if class_names is not None and isinstance(raw_label, int) else raw_label
            clean_label = clean_class_list([label])[0]
            if not img_path.exists():
                ds[idx][image_col].convert("RGB").save(img_path, format="JPEG")
            rows[key] = {
                "image_path": str(img_path), "crop": crop, "diagnosis": [clean_label],
                "details": "", "country": "", "state": "", "agml_dataset": agml_name, "split": None,
            }

    split_by_key = assign_splits(row_keys_by_dataset, args.eval_frac, args.seed)
    for key, rec in rows.items():
        rec["split"] = split_by_key[key]

    manifest["rows"] = rows
    out_path.write_text(json.dumps(manifest, indent=2))

    n_lessons = sum(1 for r in rows.values() if r["split"] == "lessons")
    n_eval = sum(1 for r in rows.values() if r["split"] == "eval")
    print(f"\n  {len(rows)} row(s) total ({len(rows) - n_before} new this run) across "
          f"{len(HELD_OUT_DATASETS)} held-out quality dataset(s).")
    print(f"  lessons={n_lessons} eval={n_eval} (--eval_frac={args.eval_frac}, --seed={args.seed})")
    print(f"  Manifest: {out_path}")
    print(f"  Images:   {images_root}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_root", default="dataset/images_heldout_quality")
    parser.add_argument("--output", default="dataset/heldout_quality_selected_images.json")
    parser.add_argument("--eval_frac", type=float, default=0.2,
                         help="Fraction of each held-out sub-dataset's own rows reserved for eval, "
                              "never used to write the skill-memory file.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
