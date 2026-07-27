"""
Stage 0 (prep) — Catalog every dataset in the Project-AgML HuggingFace org.

AgML's full corpus was republished to https://huggingface.co/Project-AgML (259
datasets as of writing, far more than the 25 hand-curated in
src/config/datasets.yaml). There is no bulk API for per-dataset splits/features/
class-label lists, so each repo is introspected individually via
datasets.load_dataset_builder (metadata only — this does NOT download images).

Only classification datasets are relevant to this repo's pipeline (linear probe,
vision/language assessment, CoT are all classification tasks) — detection/
segmentation datasets (bbox/mask features) are tagged and excluded downstream by
scripts/agml_hf/migrate_datasets_yaml.py, not filtered out of the catalog itself,
so the catalog stays a complete, reusable record of the org.

Output: src/config/hf_catalog.json — commit this. It's the reproducible input to
migrate_datasets_yaml.py and download_and_sample.py; re-run only when the
Project-AgML org's dataset list changes.

Usage:
  uv run python scripts/agml_hf/catalog_datasets.py \
    --org Project-AgML \
    --output src/config/hf_catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

DETECTION_HINTS = ("_detection", "_counting")
SEGMENTATION_HINTS = ("_segmentation",)
CLASSIFICATION_HINTS = ("_classification",)


def infer_dataset_type_from_name(repo_name: str) -> str | None:
    name = repo_name.lower()
    if any(h in name for h in SEGMENTATION_HINTS):
        return "segmentation"
    if any(h in name for h in DETECTION_HINTS):
        return "detection"
    if any(h in name for h in CLASSIFICATION_HINTS):
        return "classification"
    return None


def infer_dataset_type_from_features(features) -> str | None:
    keys = {k.lower() for k in features}
    type_names = {str(type(v)).lower() for v in features.values()}
    if any("classlabel" in t for t in type_names) or "label" in keys:
        return "classification"
    if "bbox" in keys or "objects" in keys or "boxes" in keys:
        return "detection"
    if "mask" in keys or "segmentation" in keys or "annotation" in keys:
        return "segmentation"
    return None


def pick_preferred_config(config_entries: dict) -> str:
    if "raw" in config_entries:
        return "raw"
    if "default" in config_entries:
        return "default"
    non_augmented = [c for c in config_entries if not re.search(r"aug|synthetic", c, re.I)]
    if non_augmented:
        return sorted(non_augmented)[0]
    return sorted(config_entries)[0]


def find_label_feature(features):
    for key in ("label", "labels", "class", "class_label"):
        if key in features:
            return features[key]
    for v in features.values():
        if hasattr(v, "names"):
            return v
    return None


def introspect_repo(repo_id: str) -> dict:
    from datasets import load_dataset_builder

    name = repo_id.split("/", 1)[1]
    builder = load_dataset_builder(repo_id, trust_remote_code=False)
    config_names = list(builder.builder_configs.keys()) if builder.builder_configs else [None]

    config_entries = {}
    for cfg_name in config_names:
        b = load_dataset_builder(repo_id, cfg_name, trust_remote_code=False) if cfg_name else builder
        splits = {s: b.info.splits[s].num_examples for s in (b.info.splits or {})}
        features = b.info.features or {}
        label_feature = find_label_feature(features)
        class_names = list(label_feature.names) if label_feature is not None and hasattr(label_feature, "names") else None
        key = cfg_name or "default"
        config_entries[key] = {
            "splits": splits,
            "class_names": class_names,
            "features": {k: str(v) for k, v in features.items()},
        }

    any_cfg = next(iter(config_entries.values()))
    dtype = (
        infer_dataset_type_from_features(any_cfg["features"])
        or infer_dataset_type_from_name(name)
        or "other"
    )

    return {
        "repo_id": repo_id,
        "dataset_type": dtype,
        "preferred_config": pick_preferred_config(config_entries),
        "configs": config_entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="Project-AgML")
    ap.add_argument("--output", default="src/config/hf_catalog.json")
    ap.add_argument("--only", default=None, help="optional regex filter on repo name, for debug/incremental runs")
    ap.add_argument("--skip_existing", action="store_true", help="skip repo ids already present in --output")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    out_path = Path(args.output)
    catalog: dict = {}
    if args.skip_existing and out_path.exists():
        catalog = json.loads(out_path.read_text())

    print(f"Listing datasets for org={args.org}...")
    api = HfApi()
    repo_ids = sorted(d.id for d in api.list_datasets(author=args.org))
    print(f"Found {len(repo_ids)} datasets.")

    if args.only:
        pattern = re.compile(args.only, re.I)
        repo_ids = [r for r in repo_ids if pattern.search(r)]
        print(f"Filtered to {len(repo_ids)} by --only={args.only!r}")

    t0 = time.time()
    n_done, n_err = 0, 0
    for i, repo_id in enumerate(repo_ids, 1):
        name = repo_id.split("/", 1)[1]
        if args.skip_existing and name in catalog and "error" not in catalog[name]:
            continue
        print(f"  [{i}/{len(repo_ids)}] {repo_id}", end=" ", flush=True)
        try:
            catalog[name] = introspect_repo(repo_id)
            print(f"ok ({catalog[name]['dataset_type']}, "
                  f"config={catalog[name]['preferred_config']})")
            n_done += 1
        except Exception as e:
            catalog[name] = {"repo_id": repo_id, "error": str(e)}
            print(f"ERROR: {e}")
            n_err += 1

        if i % 10 == 0 or i == len(repo_ids):
            out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))

    out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {n_done} ok, {n_err} errors. Wrote {out_path}")


if __name__ == "__main__":
    main()
