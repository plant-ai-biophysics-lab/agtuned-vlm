"""
Paper 2, Stage 3 (docs/stage3_context_memory.md) — download + cache Digital
Green's `Crop_Disease_Images` dataset (989 real images, expert agronomist
labels, CC-BY-4.0) as a HELD-OUT lesson-building source. This dataset must
NEVER be sampled into the normal eval pipeline — kept in its own files for
exactly that reason:

  dataset/images_heldout/<row_idx>.jpg       (not dataset/images/)
  dataset/heldout_selected_images.json       (not dataset/selected_images.json)

Deliberately NOT wired through src/common/dataset_config.py's
DatasetConfig/load_hf_samples/datasets.yaml machinery, for two real
structural reasons, not just caution:
  1. Every one of those assumes `hf_repo_id = f"Project-AgML/{agml_name}"`
     (download_and_sample.py builds it that way) — this dataset isn't
     under that HF org at all.
  2. Its `diagnosis` column can hold MULTIPLE semicolon-separated labels
     in a single row (e.g. "Pod Borer; Boron Deficiency", confirmed via
     the HF datasets-server — rare, ~1/20 rows in a spot check, but
     real) — AgML's one-label-per-image convention (a label IS the
     per-image cache directory) doesn't represent that at all.

--eval_frac (default 0.2) splits rows into a lesson-building slice and a
held-out eval slice, seeded by --seed, stratified by `crop` so both
splits cover the same crops rather than an eval slice that happens to
miss whole crops. The eval slice must never be used to write the
skill-memory .md file this stage builds toward — only to measure whether
it actually generalizes, once built.

Resumable: an image already cached on disk is never re-fetched; a row
already in --output is skipped.

Usage:
    uv run python scripts/download_heldout_digitalgreen.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
from collections import defaultdict

_REPO_ID = "DigiGreen/Crop_Disease_Images"


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"hf_repo_id": _REPO_ID, "rows": {}}


def assign_splits(row_ids_by_crop: dict[str, list[int]], eval_frac: float, seed: int) -> dict[int, str]:
    """Stratified by crop — each crop's own rows are shuffled and split
    independently, so a crop with few images still contributes to both
    splits (rounding down means a crop with 1 row always lands in
    "lessons", never silently dropped from either split's class
    coverage in a way that would be hard to notice)."""
    rng = random.Random(seed)
    split_by_id: dict[int, str] = {}
    for crop, ids in row_ids_by_crop.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_eval = round(len(ids) * eval_frac)
        for i, row_id in enumerate(ids):
            split_by_id[row_id] = "eval" if i < n_eval else "lessons"
    return split_by_id


def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    images_root = Path(args.images_root)
    images_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out_path)
    rows = manifest["rows"]
    n_before = len(rows)

    print(f"  Loading {_REPO_ID} (split=train)...")
    ds = load_dataset(_REPO_ID, split="train")
    print(f"  {len(ds)} row(s) in the dataset.")

    row_ids_by_crop: dict[str, list[int]] = defaultdict(list)
    pending_images: dict[int, object] = {}
    for idx, row in enumerate(ds):
        key = str(idx)
        crop = row.get("crop") or "unknown"
        row_ids_by_crop[crop].append(idx)
        img_path = images_root / f"{idx}.jpg"
        if key in rows and img_path.exists():
            continue
        if not img_path.exists():
            pending_images[idx] = row["image_file"]
        diagnosis_raw = row.get("diagnosis") or ""
        diagnosis = [d.strip() for d in diagnosis_raw.split(";") if d.strip()]
        rows[key] = {
            "image_path": str(img_path), "crop": crop, "diagnosis": diagnosis,
            "details": row.get("details") or "", "country": row.get("country") or "",
            "state": row.get("state") or "", "split": None,  # filled in below, every run
        }

    if pending_images:
        print(f"  Caching {len(pending_images)} new image(s) to {images_root}/...")
        for idx, img in pending_images.items():
            img.convert("RGB").save(images_root / f"{idx}.jpg", format="JPEG")

    # Re-derive splits every run (cheap, deterministic given --seed) so a
    # resumed/extended run's split assignment stays consistent rather than
    # only ever assigning "lessons"/"eval" to rows seen in THIS run.
    split_by_id = assign_splits(row_ids_by_crop, args.eval_frac, args.seed)
    for key, rec in rows.items():
        rec["split"] = split_by_id[int(key)]

    manifest["rows"] = rows
    out_path.write_text(json.dumps(manifest, indent=2))

    n_lessons = sum(1 for r in rows.values() if r["split"] == "lessons")
    n_eval = sum(1 for r in rows.values() if r["split"] == "eval")
    print(f"\n  {len(rows)} row(s) total ({len(rows) - n_before} new this run).")
    print(f"  lessons={n_lessons} eval={n_eval} (--eval_frac={args.eval_frac}, --seed={args.seed})")
    print(f"  Manifest: {out_path}")
    print(f"  Images:   {images_root}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_root", default="dataset/images_heldout")
    parser.add_argument("--output", default="dataset/heldout_selected_images.json")
    parser.add_argument("--eval_frac", type=float, default=0.2,
                         help="Fraction of each crop's rows held out for eval, never used to write "
                              "the skill-memory file.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
