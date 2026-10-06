"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 0 —
download + cache the detection skill-memory pipeline's TRAIN/VAL pool.
Detection analog of scripts/download_heldout_digitalgreen.py.

**Final design**: of the ~54-dataset Project-AgML detection catalog, 10
go to TRAIN, 5 to VAL — both marked `skip: true` in
src/config/detection_datasets.yaml (never sampled into Stage 2/Stage 6's
own active eval pool at all). The remaining ~39 are TEST — read for a
final confirmatory score only, NEVER used to generate lessons, so they
stay in Stage 2/6's own active pool untouched (no marking needed; this
script never touches them).

Unlike the lessons/eval split classification's own
download_heldout_digitalgreen.py uses (one pool of IMAGES, stratified
80/20 within itself), this is a split by DATASET MEMBERSHIP — a dataset
is entirely TRAIN or entirely VAL, decided here directly, not derived
from a random per-row split. "split" in the output manifest is
"lessons" for every row from a TRAIN_DATASETS member, "eval" for every
row from a VAL_DATASETS member — same terminology
(lessons/eval) the rest of this pipeline already uses.

TRAIN (10) — generates the lessons/explanations the skill file is built
from:
  - synthetic_cowpea_flower_detection, synthetic_cowpea_pod_detection,
    grape_detection_syntheticday — synthetic data, cheap/scalable
  - mango_growth_detection, MH_Weed16_weed_detection,
    ghai_strawberry_fruit_detection, cashew_detection,
    weed_crop_detection, tomato_ripeness_detection,
    onionfoliageset_detection — 7 more, domain-diverse (growth-stage,
    extreme multiclass weed-ID, organ-level distinctions, multi-scale
    clustering, crop-vs-weed, ripeness, crop/no-crop/foliage)

VAL (5) — scores every round of Step 3b's self-optimization loop;
NEVER shown to the edit call, selects which round's skill file to
keep only:
  - embrapa_wgisd_grape_detection, grape_detection_californianight —
    real grape (grape_detection_californiaday and the GEMINI cowpea
    real counterparts were an earlier pick, moved to TEST instead —
    VAL's job here is a representative general validation set, not a
    strict synthetic/real domain-paired sim-to-real test)
  - wheat_head_counting, CocoaMFDB_detection, coffee_detection — 3
    more, chosen for diversity against TRAIN's own domains (grain/
    counting, a different crop family, multi-scale tree+fruit)

Reuses download_and_sample_detection.py's own process_dataset()
(class-balanced selection) and dataset_config.py's own
load_hf_detection_samples() (HF download + local image/annotation
caching) directly, not reimplemented — same as before.

Caches to dataset/images_detection_heldout/<agml_name>/<idx>.jpg (+
.json sidecar) — NOT dataset/images_detection/, so a held-out image can
never collide with Stage 2/6's own eval cache. Writes
dataset/detection_heldout_selected_images.json — one record per row
(agml_dataset, image_path, boxes, split). Resumable (an already-cached
image/sidecar pair is a pure local-disk read).

**Known caveat, inherited not introduced here**: migrate-detection-
datasets regenerates detection_datasets.yaml from scratch on every run
but IS merge-safe as of 2026-10 (see that script's own module
docstring) — a deliberate `skip: true` holdout (notes starting with
"HELD OUT for Paper 2 Stage 4") survives a re-run automatically. The
`crop` field for an EXISTING dataset also survives; only a brand-new
dataset name gets fresh auto-derivation.

Usage:
    uv run python main.py download-heldout-detection
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import traceback
from datetime import datetime, timezone

from scripts.agml_hf.download_and_sample import with_retry
from scripts.agml_hf.download_and_sample_detection import process_dataset
from src.common.dataset_config import load_hf_detection_samples

TRAIN_DATASETS = [
    "synthetic_cowpea_flower_detection",
    "synthetic_cowpea_pod_detection",
    "grape_detection_syntheticday",
    "mango_growth_detection",
    "MH_Weed16_weed_detection",
    "ghai_strawberry_fruit_detection",
    "cashew_detection",
    "weed_crop_detection",
    "tomato_ripeness_detection",
    "onionfoliageset_detection",
]

VAL_DATASETS = [
    "embrapa_wgisd_grape_detection",
    "grape_detection_californianight",
    "wheat_head_counting",
    "CocoaMFDB_detection",
    "coffee_detection",
]


def download_group(agml_names: list[str], split_label: str, catalog: dict, images_root: Path,
                    n_per_class: int, seed: int, manifest: dict) -> None:
    for i, agml_name in enumerate(agml_names, 1):
        meta = catalog.get(agml_name, {})
        hf_config = meta.get("preferred_config", "default")
        class_names = meta.get("class_names")
        print(f"  [{split_label} {i}/{len(agml_names)}] {agml_name}", end=" ", flush=True)
        try:
            repo_id, config, selected = with_retry(
                process_dataset, agml_name, hf_config, catalog, n_per_class, seed, None,
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
            print(f"ok ({len(selected)} classes, {len(samples)} images)")

            for s in samples:
                row_key = f"{agml_name}/{Path(s['path']).stem}"
                manifest["rows"][row_key] = {
                    "agml_dataset": agml_name,
                    "image_path": s["path"],
                    "boxes": s["boxes"],
                    "split": split_label,
                }
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)

        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/config/hf_detection_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=10,
                     help="Images/class to sample per dataset (both TRAIN and VAL) — same default "
                          "download_and_sample_detection.py uses for Stage 2/6's own pool.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images_root", default="dataset/images_detection_heldout")
    ap.add_argument("--output", default="dataset/detection_heldout_selected_images.json")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}
    images_root = Path(args.images_root)

    out_path = Path(args.output)
    manifest = json.loads(out_path.read_text()) if out_path.exists() else {
        "generated_at": None, "seed": args.seed, "n_per_class": args.n_per_class,
        "train_datasets": TRAIN_DATASETS, "val_datasets": VAL_DATASETS, "rows": {},
    }

    t0 = time.time()
    print(f"TRAIN ({len(TRAIN_DATASETS)} datasets):")
    download_group(TRAIN_DATASETS, "lessons", catalog, images_root, args.n_per_class, args.seed, manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nVAL ({len(VAL_DATASETS)} datasets):")
    download_group(VAL_DATASETS, "eval", catalog, images_root, args.n_per_class, args.seed, manifest)
    out_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    n_lessons = sum(1 for r in manifest["rows"].values() if r["split"] == "lessons")
    n_eval = sum(1 for r in manifest["rows"].values() if r["split"] == "eval")
    print(f"\nDone in {elapsed:.0f}s. {len(manifest['rows'])} total rows "
          f"({n_lessons} TRAIN/lessons from {len(TRAIN_DATASETS)} datasets, "
          f"{n_eval} VAL/eval from {len(VAL_DATASETS)} datasets) -> {out_path}")


if __name__ == "__main__":
    main()
