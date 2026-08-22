"""
Stage 5 — Download detection-type datasets from Project-AgML (HuggingFace),
take a class-balanced subsample of each, and record exactly which images
were chosen to a JSON manifest — detection analog of download_and_sample.py.

Differs from the classification version in one structural way: a detection
image can carry MULTIPLE annotated instances/categories at once, so there's
no single per-row label to balance on the way classification has. This
balances on each row's PRIMARY category (its first annotated instance's
category) as a documented, pragmatic V1 stand-in — see
src/common/dataset_config.py::load_hf_detection_samples, which materializes
the FULL per-row annotation list (every box/category, not just the primary
one) regardless of which category a row was balanced under. Rows with zero
annotated instances are excluded entirely (not useful for detection).

Reuses download_and_sample.py's HF-rate-limit retry helpers directly
(with_retry/is_retryable/RETRYABLE_STATUS) rather than reimplementing them.

Usage (--n_per_class defaults to 10, matching the classification pipeline):
  uv run python main.py download-detection-datasets \\
    --config src/config/detection_datasets.yaml \\
    --catalog src/config/hf_detection_catalog.json \\
    --output dataset/selected_images_detection.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.agml_hf.download_and_sample import with_retry  # noqa: E402
from src.common.dataset_config import active_datasets, DatasetConfig, subsample_class_balanced  # noqa: E402


def process_dataset(agml_name: str, hf_config: str | None, catalog: dict,
                     n_per_class: int, seed: int,
                     fixed_indices: dict[str, list[int]] | None):
    from datasets import load_dataset

    meta = catalog.get(agml_name, {})
    config = hf_config or meta.get("preferred_config", "default")
    repo_id = f"Project-AgML/{agml_name}"
    ds = load_dataset(repo_id, config, split="train")
    class_names = meta.get("class_names")

    def primary_category_name(i: int) -> str | None:
        objs = ds[i].get("objects") or {}
        # Schema key inconsistency confirmed across real Project-AgML
        # detection repos (e.g. CocoaMFDB_detection uses "categories",
        # CBDA_Cotton_Ball_Detection_Augmented uses "category") — check
        # both rather than assuming either.
        cats = objs.get("category") if objs.get("category") is not None else objs.get("categories", [])
        if not cats:
            return None
        cat_id = int(cats[0])
        if class_names and 0 <= cat_id < len(class_names):
            return class_names[cat_id]
        return f"category_{cat_id}"

    if fixed_indices is not None:
        selected = fixed_indices
    else:
        annotated_indices = [i for i in range(len(ds)) if primary_category_name(i) is not None]
        selected = subsample_class_balanced(annotated_indices, primary_category_name, n_per_class, seed)

    return repo_id, config, selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="src/config/detection_datasets.yaml")
    ap.add_argument("--catalog", default="src/config/hf_detection_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=10,
                     help="Images/class to sample — matches the classification pipeline's "
                          "n=10/class default (see docs/stage0a_datasets.md's note on n=10 vs. "
                          "n=25 giving nearly identical numbers at lower cost).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="dataset/selected_images_detection.json")
    ap.add_argument("--only", default=None, help="Optional regex filter on agml_name.")
    ap.add_argument("--from_manifest", default=None,
                     help="Reproduce indices from an existing manifest instead of re-sampling.")
    ap.add_argument("--skip_existing", action="store_true",
                     help="Resume: reuse datasets already recorded in --output.")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}
    datasets = active_datasets(Path(args.config))

    if args.only:
        pattern = re.compile(args.only)
        datasets = [d for d in datasets if pattern.search(d.agml_name)]
        if not datasets:
            print(f"ERROR: --only {args.only!r} matched no active (skip: false) detection datasets.")
            sys.exit(1)

    prior_manifest = json.loads(Path(args.from_manifest).read_text()) if args.from_manifest else None

    out_path = Path(args.output)
    existing_manifest = None
    if args.skip_existing and out_path.exists():
        existing_manifest = json.loads(out_path.read_text())
        if existing_manifest.get("n_per_class") != args.n_per_class:
            print(f"WARNING: --skip_existing requested but existing manifest was built with "
                  f"n_per_class={existing_manifest.get('n_per_class')} != requested "
                  f"{args.n_per_class} — ignoring --skip_existing, reprocessing everything.")
            existing_manifest = None

    manifest = {
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduced_from": args.from_manifest,
        "datasets": {},
        "failed": {},
    }
    if existing_manifest:
        manifest["datasets"] = dict(existing_manifest.get("datasets", {}))

    t0 = time.time()
    n_ok, n_err, n_skipped = 0, 0, 0
    for i, d in enumerate(datasets, 1):
        if existing_manifest and d.agml_name in manifest["datasets"]:
            print(f"[{i}/{len(datasets)}] {d.agml_name} — already done, skipping")
            n_skipped += 1
            continue

        print(f"[{i}/{len(datasets)}] {d.agml_name}", end=" ", flush=True)
        fixed = None
        if prior_manifest and d.agml_name in prior_manifest["datasets"]:
            fixed = prior_manifest["datasets"][d.agml_name]["classes"]
        try:
            repo_id, config, selected = with_retry(
                process_dataset, d.agml_name, d.hf_config, catalog, args.n_per_class, args.seed, fixed,
            )
            manifest["datasets"][d.agml_name] = {
                "hf_repo_id": repo_id,
                "hf_config": config,
                "classes": selected,
                "class_names": catalog.get(d.agml_name, {}).get("class_names"),
                "bbox_convention": catalog.get(d.agml_name, {}).get("bbox_convention"),
            }
            manifest["failed"].pop(d.agml_name, None)
            n_total = sum(len(v) for v in selected.values())
            print(f"ok ({len(selected)} classes, {n_total} images)")
            n_ok += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)
            manifest["failed"][d.agml_name] = {"error": str(e), "traceback": tb}
            n_err += 1

        if i % 5 == 0 or i == len(datasets):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    elapsed = time.time() - t0
    total_images = sum(len(idxs) for ds in manifest["datasets"].values() for idxs in ds["classes"].values())
    print(f"\nDone in {elapsed:.0f}s. {n_ok} datasets ok, {n_skipped} skipped, {n_err} errors -> {args.output}")
    if manifest["failed"]:
        print(f"Failed datasets (re-run with --skip_existing to retry): {sorted(manifest['failed'])}")
    print(f"Total images selected: {total_images:,}")


if __name__ == "__main__":
    main()
