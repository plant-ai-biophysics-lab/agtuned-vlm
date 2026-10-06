"""
Stage 0 (prep) — Catalog every dataset in the Project-AgML HuggingFace org.

AgML's full corpus was republished to https://huggingface.co/Project-AgML (259
datasets as of writing, far more than the 25 hand-curated in
src/config/datasets.yaml). There is no bulk API for per-dataset splits/features/
class-label lists, so each repo is introspected individually via
datasets.load_dataset_builder (metadata only — this does NOT download images).

Only classification datasets are relevant to this repo's pipeline right now
(linear probe, vision/language assessment, CoT are all classification tasks).
Every dataset still gets introspected (there's no way to know its type without
loading its builder metadata first), but the two output files split on the
result: classification-type entries (full detail) go to --output;
detection/segmentation/multimodal/other entries go to --excluded_output
(also full detail — the introspection work is already done, no reason to
throw it away) — this repo just isn't using them yet. Revisit
--excluded_output later if/when detection, segmentation, or multimodal/VQA
data becomes relevant to this pipeline; nothing needs re-scanning to do so.

The multimodal check runs BEFORE the classification check in
infer_dataset_type_from_features — a VQA/captioning dataset can carry an
incidental "label" key or ClassLabel feature alongside its text fields (e.g.
{"image", "question", "answer", "label"}), which would otherwise be silently
misclassified as pure classification and auto-included.

Output: src/config/hf_catalog.json (classification only) and
src/config/hf_catalog_excluded.json (everything else) — commit both. The
former is the reproducible input to migrate_datasets_yaml.py and
download_and_sample.py; re-run only when the Project-AgML org's dataset list
changes.

Usage:
  uv run python scripts/agml_hf/catalog_datasets.py \
    --org Project-AgML \
    --output src/config/hf_catalog.json \
    --excluded_output src/config/hf_catalog_excluded.json
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
MULTIMODAL_HINTS = ("_vqa", "_qa", "_caption", "_captioning", "_instruct")

# Feature keys that indicate a VQA/captioning/instruction-style dataset —
# checked BEFORE the classification check below, not after, because a
# multimodal dataset can carry an incidental "label" key or ClassLabel
# feature alongside its text fields (e.g. {"image", "question", "answer",
# "label"}) and would otherwise be silently misclassified as pure
# classification and auto-included in datasets.yaml by migrate_datasets_yaml.py,
# when it actually needs manual review, not auto-inclusion.
MULTIMODAL_FEATURE_KEYS = {
    "question", "answer", "answers", "caption", "captions", "conversations",
    "messages", "instruction", "response", "text", "dialogue", "prompt",
}


def infer_dataset_type_from_name(repo_name: str) -> str | None:
    name = repo_name.lower()
    if any(h in name for h in MULTIMODAL_HINTS):
        return "multimodal"
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
    if keys & MULTIMODAL_FEATURE_KEYS:
        return "multimodal"
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
    from datasets import get_dataset_config_names, load_dataset_builder
    from datasets.data_files import EmptyDatasetError

    name = repo_id.split("/", 1)[1]
    # trust_remote_code intentionally omitted: datasets>=4.0 no longer accepts
    # it for these repos (auto-converted Parquet, no custom loading script to
    # trust in the first place) — passing it raises "BuilderConfig ... doesn't
    # have a 'trust_remote_code' key" / "unexpected keyword argument" on every
    # single repo.
    #
    # get_dataset_config_names(), not load_dataset_builder(repo_id) with no
    # config, to enumerate configs: a repo with multiple named configs and no
    # default (e.g. EfficientMaize_classification -> ['augmented', 'raw'])
    # raises "Config name is missing" from the no-arg builder call before we
    # ever get to list its configs. This works uniformly for single-config
    # repos too — they report back ['default'], which load_dataset_builder
    # accepts explicitly just like any named config.
    #
    # KNOWN REPO-SIDE BUG, confirmed on Project-AgML/synthetic_cowpea_
    # {flower,pod}_detection: the repo's own README YAML declares a config
    # (e.g. "synthetic") whose data_files glob pattern matches ZERO real
    # files — get_dataset_config_names() raises EmptyDatasetError before we
    # ever see a config name, even though the repo's actual data
    # (confirmed via huggingface_hub.list_repo_files: augmented/train-*.parquet
    # for these two) is perfectly loadable. Not something to fix upstream from
    # here — fall back to bypassing config-name resolution ENTIRELY and
    # loading directly from whatever real *.parquet files the repo actually
    # has, via an explicit data_files= glob. General (not hardcoded to these
    # two repo names), in case another AgML repo has the same README bug.
    try:
        config_names = get_dataset_config_names(repo_id)
        fallback_data_files = None
    except EmptyDatasetError:
        from huggingface_hub import list_repo_files
        parquet_files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")]
        if not parquet_files:
            raise
        print(f"[config-name resolution failed, falling back to {len(parquet_files)} real "
              f"*.parquet file(s)] ", end="")
        config_names = [None]
        fallback_data_files = parquet_files

    config_entries = {}
    for cfg_name in config_names:
        if fallback_data_files is not None:
            b = load_dataset_builder(repo_id, data_files=fallback_data_files)
        else:
            b = load_dataset_builder(repo_id, cfg_name)
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
    ap.add_argument("--output", default="src/config/hf_catalog.json",
                    help="Classification-only entries (full detail).")
    ap.add_argument("--excluded_output", default="src/config/hf_catalog_excluded.json",
                    help="Detection/segmentation/multimodal/other entries (also full detail — "
                         "not thrown away, just not used by this pipeline yet).")
    ap.add_argument("--only", default=None, help="optional regex filter on repo name, for debug/incremental runs")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip repo ids already present in --output or --excluded_output")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    out_path = Path(args.output)
    excluded_path = Path(args.excluded_output)
    catalog: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    excluded: dict = json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
    if not args.skip_existing:
        catalog, excluded = {}, {}

    print(f"Listing datasets for org={args.org}...")
    api = HfApi()
    repo_ids = sorted(d.id for d in api.list_datasets(author=args.org))
    print(f"Found {len(repo_ids)} datasets.")

    if args.only:
        pattern = re.compile(args.only, re.I)
        repo_ids = [r for r in repo_ids if pattern.search(r)]
        print(f"Filtered to {len(repo_ids)} by --only={args.only!r}")

    t0 = time.time()
    n_done, n_err, n_excluded = 0, 0, 0
    for i, repo_id in enumerate(repo_ids, 1):
        name = repo_id.split("/", 1)[1]
        already_done = (name in catalog and "error" not in catalog[name]) or \
                       (name in excluded and "error" not in excluded[name])
        if args.skip_existing and already_done:
            continue
        print(f"  [{i}/{len(repo_ids)}] {repo_id}", end=" ", flush=True)
        try:
            entry = introspect_repo(repo_id)
            if entry["dataset_type"] == "classification":
                catalog[name] = entry
                excluded.pop(name, None)
            else:
                excluded[name] = entry
                catalog.pop(name, None)
                n_excluded += 1
            print(f"ok ({entry['dataset_type']}, config={entry['preferred_config']})")
            n_done += 1
        except Exception as e:
            catalog.pop(name, None)
            excluded[name] = {"repo_id": repo_id, "error": str(e)}
            print(f"ERROR: {e}")
            n_err += 1

        if i % 10 == 0 or i == len(repo_ids):
            out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))
            excluded_path.write_text(json.dumps(excluded, indent=2, sort_keys=True))

    out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    excluded_path.write_text(json.dumps(excluded, indent=2, sort_keys=True))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {n_done} ok ({n_excluded} non-classification), {n_err} errors.")
    print(f"  Classification -> {out_path} ({len(catalog)} entries)")
    print(f"  Excluded (revisit later) -> {excluded_path} ({len(excluded)} entries)")


if __name__ == "__main__":
    main()
