"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 0 —
download + cache a HELD-OUT pool of Project-AgML object-detection
datasets for the detection skill-memory pipeline's lesson-building and
held-out eval. Detection analog of
scripts/download_heldout_digitalgreen.py.

**Decided (not re-derived at run time)**: 5 ENTIRE datasets, chosen for
domain diversity, never sampled into Stage 2/Stage 6's own eval pool at
all (see HELD_OUT_DATASETS below + the corresponding `skip: true` +
notes marker on each one's entry in src/config/detection_datasets.yaml):

  - mango_growth_detection       — growth-stage distinctions (4 classes:
    Early-Fruit/Mature/Premature/Ripe), single crop, moderate size
  - MH_Weed16_weed_detection     — extreme multiclass (15 weed/crop
    species), dense/crowded images, a stress test for the
    "wrong class label on an otherwise-correct box" failure mode
  - ghai_strawberry_fruit_detection — organ-level multiclass (10 classes
    spanning bud/calyx/flower/leaf/stem/fruit at multiple ripeness
    stages), fine part-level distinctions rather than whole-plant stage
  - cashew_detection             — multi-scale (whole tree + individual
    flowers/nuts, 6 classes), a stress test for "multiple instances
    merged" vs. "missed entirely"
  - weed_crop_detection          — crop-vs-weed across MANY species (13
    classes), closest in spirit to classification's own
    weed_identification task_type

Together: growth-stage, extreme-class-count weed-ID, organ/part-level
distinctions, multi-scale/dense clustering, and crop-vs-weed — five
different failure-mode flavors, not five similar datasets.

Unlike download_and_sample_detection.py (Stage 2/6's own sampler, which
only ever selects from `active_datasets()` — i.e. `skip: false`
entries), this script targets these 5 names DIRECTLY regardless of
their `skip` flag in detection_datasets.yaml, reusing
download_and_sample_detection.py's own process_dataset() (imported, not
duplicated) for the class-balanced selection and
src/common/dataset_config.py's load_hf_detection_samples() (imported,
not duplicated) for the actual HF download + local image/annotation
caching — same resumability guarantee as every other Stage 2/4/6
downloader (an already-cached image+sidecar pair is a pure local-disk
read, no HF Hub/network involvement on a second run).

Caches to dataset/images_detection_heldout/<agml_name>/<idx>.jpg (+
.json sidecar) — NOT dataset/images_detection/, so a held-out image can
never collide with or get silently reused by Stage 2/6's own normal
eval image cache. Writes dataset/detection_heldout_selected_images.json
— one record per row (agml_dataset, image_path, boxes, split), split
assigned "lessons"/"eval" (--eval_frac, default 0.2, stratified by
DATASET — not crop, since each of the 5 held-out datasets already
represents its own distinct domain — seeded by --seed).

**Known caveat, inherited not introduced here**: migrate-detection-
datasets regenerates detection_datasets.yaml from scratch on every run
(see that script's own module docstring) and does NOT currently merge
forward prior manual edits — this was already true of the `crop` field
(hand-filled after generation, would be wiped by a naive re-run) before
this script's own `skip: true` + notes markers added a second thing a
re-run would silently undo. Re-apply both by hand if
migrate-detection-datasets is ever re-run.

Usage:
    uv run python main.py download-heldout-detection
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import time
import traceback
from datetime import datetime, timezone

from scripts.agml_hf.download_and_sample import with_retry
from scripts.agml_hf.download_and_sample_detection import process_dataset
from src.common.dataset_config import load_hf_detection_samples

HELD_OUT_DATASETS = [
    "mango_growth_detection",
    "MH_Weed16_weed_detection",
    "ghai_strawberry_fruit_detection",
    "cashew_detection",
    "weed_crop_detection",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/config/hf_detection_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=10,
                     help="Images/class to sample per held-out dataset — same default "
                          "download_and_sample_detection.py uses for Stage 2/6's own pool.")
    ap.add_argument("--eval_frac", type=float, default=0.2,
                     help="Fraction of each held-out dataset's own selected rows assigned to "
                          "\"eval\" (never used for lesson-writing, only for Step 4's ablation) — "
                          "stratified by DATASET, same mechanism download_heldout_digitalgreen.py "
                          "uses stratified by crop.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images_root", default="dataset/images_detection_heldout")
    ap.add_argument("--output", default="dataset/detection_heldout_selected_images.json")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}
    images_root = Path(args.images_root)

    out_path = Path(args.output)
    manifest = json.loads(out_path.read_text()) if out_path.exists() else {
        "generated_at": None, "seed": args.seed, "eval_frac": args.eval_frac,
        "n_per_class": args.n_per_class, "held_out_datasets": HELD_OUT_DATASETS, "rows": {},
    }

    t0 = time.time()
    for i, agml_name in enumerate(HELD_OUT_DATASETS, 1):
        meta = catalog.get(agml_name, {})
        hf_config = meta.get("preferred_config", "default")
        class_names = meta.get("class_names")
        print(f"[{i}/{len(HELD_OUT_DATASETS)}] {agml_name}", end=" ", flush=True)
        try:
            repo_id, config, selected = with_retry(
                process_dataset, agml_name, hf_config, catalog, args.n_per_class, args.seed, None,
            )
            selection_manifest = {
                "datasets": {
                    agml_name: {
                        "hf_repo_id": repo_id, "hf_config": config, "classes": selected,
                        "class_names": class_names, "bbox_convention": meta.get("bbox_convention"),
                    }
                }
            }
            samples = load_hf_detection_samples(agml_name, selection_manifest, images_root, class_names)
            n_total = len(samples)
            print(f"ok ({len(selected)} classes, {n_total} images)")

            # Stratified-by-DATASET lessons/eval split — each of the 5 held-out
            # datasets gets its own 80/20 split rather than pooling all 5
            # together first, so a small dataset's rows can't get entirely
            # absorbed into the other split by chance.
            rng = random.Random(args.seed)
            indices = list(range(n_total))
            rng.shuffle(indices)
            n_eval = max(1, round(n_total * args.eval_frac)) if n_total else 0
            eval_idx = set(indices[:n_eval])

            for idx, s in enumerate(samples):
                row_key = f"{agml_name}/{Path(s['path']).stem}"
                manifest["rows"][row_key] = {
                    "agml_dataset": agml_name,
                    "image_path": s["path"],
                    "boxes": s["boxes"],
                    "split": "eval" if idx in eval_idx else "lessons",
                }
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)

        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    n_lessons = sum(1 for r in manifest["rows"].values() if r["split"] == "lessons")
    n_eval = sum(1 for r in manifest["rows"].values() if r["split"] == "eval")
    print(f"\nDone in {elapsed:.0f}s. {len(manifest['rows'])} total rows "
          f"({n_lessons} lessons, {n_eval} eval) -> {out_path}")


if __name__ == "__main__":
    main()
