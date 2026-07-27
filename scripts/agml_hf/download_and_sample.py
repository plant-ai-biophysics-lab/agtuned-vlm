"""
Stage 0 — Download active datasets from Project-AgML (HuggingFace), take a
class-balanced subsample of each, and record exactly which images were chosen
to a JSON manifest for reproducibility.

Replaces the old `agml` pip package's implicit auto-download-on-first-use with
an explicit prefetch against https://huggingface.co/Project-AgML (259 datasets,
vs. the 25 previously hand-curated). Only classification datasets are relevant
to this pipeline — see scripts/agml_hf/migrate_datasets_yaml.py, which already
filtered datasets.yaml down to classification-only.

Sampling is a fixed cap per class (--n_per_class, adjustable), seeded for
reproducibility. The manifest records the exact HF row index selected per
(dataset, class) — re-running with --from_manifest reproduces the identical
image set without re-rolling randomness, and lets later consumers materialize
images by index without depending on random state.

Usage:
  uv run python scripts/agml_hf/download_and_sample.py \
    --config src/config/datasets.yaml \
    --catalog src/config/hf_catalog.json \
    --n_per_class 50 \
    --output dataset/selected_images.json

  # Reproduce a previous selection exactly (e.g. after re-cataloging), no re-sampling:
  uv run python scripts/agml_hf/download_and_sample.py \
    --config src/config/datasets.yaml \
    --from_manifest dataset/selected_images.json \
    --output dataset/selected_images.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import active_datasets, subsample_class_balanced  # noqa: E402


def find_label_column(features) -> str:
    for key in ("label", "labels", "class", "class_label"):
        if key in features:
            return key
    raise KeyError(f"No recognizable label column in features: {list(features)}")


def resolve_class_names(ds, label_col: str) -> list[str] | None:
    feat = ds.features[label_col]
    return list(feat.names) if hasattr(feat, "names") else None


def process_dataset(agml_name: str, hf_config: str | None, catalog: dict,
                     n_per_class: int, seed: int,
                     fixed_indices: dict[str, list[int]] | None):
    from datasets import load_dataset

    config = hf_config or catalog.get(agml_name, {}).get("preferred_config")
    repo_id = f"Project-AgML/{agml_name}"
    ds = load_dataset(repo_id, config, split="train")
    label_col = find_label_column(ds.features)
    class_names = resolve_class_names(ds, label_col)

    def label_fn(i):
        raw = ds[i][label_col]
        return class_names[raw] if class_names is not None and isinstance(raw, int) else raw

    if fixed_indices is not None:
        selected = fixed_indices
    else:
        selected = subsample_class_balanced(list(range(len(ds))), label_fn, n_per_class, seed)

    return repo_id, config, selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/datasets.yaml")
    ap.add_argument("--catalog", default="src/config/hf_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="dataset/selected_images.json")
    ap.add_argument("--include_skipped", action="store_true")
    ap.add_argument("--from_manifest", default=None,
                     help="reproduce indices from an existing manifest instead of re-sampling")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}

    if args.include_skipped:
        from src.common.dataset_config import load_datasets
        datasets = load_datasets(Path(args.config))
    else:
        datasets = active_datasets(Path(args.config))

    prior_manifest = json.loads(Path(args.from_manifest).read_text()) if args.from_manifest else None

    manifest = {
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduced_from": args.from_manifest,
        "datasets": {},
    }

    t0 = time.time()
    n_ok, n_err = 0, 0
    for i, d in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {d.agml_name}", end=" ", flush=True)
        fixed = None
        if prior_manifest and d.agml_name in prior_manifest["datasets"]:
            fixed = prior_manifest["datasets"][d.agml_name]["classes"]
        try:
            repo_id, config, selected = process_dataset(
                d.agml_name, d.hf_config, catalog, args.n_per_class, args.seed, fixed
            )
            manifest["datasets"][d.agml_name] = {
                "hf_repo_id": repo_id,
                "hf_config": config,
                "classes": selected,
            }
            n_total = sum(len(v) for v in selected.values())
            print(f"ok ({len(selected)} classes, {n_total} images)")
            n_ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            n_err += 1

        if i % 5 == 0 or i == len(datasets):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    elapsed = time.time() - t0
    total_images = sum(
        len(idxs) for ds in manifest["datasets"].values() for idxs in ds["classes"].values()
    )
    print(f"\nDone in {elapsed:.0f}s. {n_ok} datasets ok, {n_err} errors, "
          f"{total_images:,} images selected -> {args.output}")


if __name__ == "__main__":
    main()
