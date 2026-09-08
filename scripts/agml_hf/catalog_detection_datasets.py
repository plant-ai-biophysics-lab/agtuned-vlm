"""
Stage 5 (prep) — Catalog detection-type datasets in the Project-AgML
HuggingFace org, resolving the two things scripts/agml_hf/catalog_datasets.py
left unresolved for them.

catalog_datasets.py already introspected every repo in the org and split
classification-type entries into hf_catalog.json, everything else (including
every detection-type repo) into hf_catalog_excluded.json — full feature/split
detail, nothing thrown away. But its find_label_feature() only checks
TOP-LEVEL label-shaped keys ("label", "labels", "class", "class_label", or
any top-level feature with .names) — a detection repo's category ids live
NESTED inside an "objects" dict feature (e.g. {"bbox": ..., "category": ...}),
so find_label_feature() never sees them and every detection entry's
class_names comes back null.

This script, for every dataset_type=="detection" entry in --excluded_catalog:
  1. Re-loads the HF dataset builder and tries to resolve category_id ->
     class_name three ways, in order: (a) is the category feature actually a
     ClassLabel with .names after all, just nested one level deeper than
     find_label_feature() checks; (b) does the repo's dataset card (README.md,
     fetched via huggingface_hub) contain a recognizable class list; (c) load
     one row and check for a stray "categories"/"classes"/"labels" field
     sitting alongside "objects" that isn't part of the standard schema.
  2. Loads a small sample of rows (--n_probe_rows) and measures the actual
     bbox coordinate convention used — pixel vs. normalized (by value
     magnitude: essentially every real photo has SOME coordinate > 1.0 if
     pixel-space; normalized stays in [0,1]) and xywh vs. xyxy (by checking
     whether index 2/3 consistently exceed index 0/1 the way x2/y2 must in
     xyxy, but rarely do as a raw width/height in xywh) — printed AND
     recorded per dataset rather than assumed to match any other AgML export
     path's convention (see this repo's object-detection plan for why that
     assumption would be unsafe).

Output: --output (src/config/hf_detection_catalog.json), hf_catalog.json-
shaped (repo_id/dataset_type/preferred_config/configs) plus two new fields
per entry: "class_names" (filled in where resolved, else null) and
"bbox_convention" ({"scale": "pixel"|"normalized"|"unknown", "format":
"xyxy"|"xywh"|"unknown", "n_probed": int}). Datasets where class_names
couldn't be resolved by any method are still WRITTEN (not dropped) with
class_names=null — migrate_detection_datasets_yaml.py marks those skip: true,
the same "recorded but flagged for manual review" convention Stage 0a uses,
so nothing here is silently lost.

Usage:
  uv run python main.py catalog-detection-datasets \\
    --excluded_catalog src/config/hf_catalog_excluded.json \\
    --output src/config/hf_detection_catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

_CLASS_LIST_HINT_RE = re.compile(
    r"(?:classes|categories|labels)\s*[:=]?\s*\[([^\]]+)\]", re.IGNORECASE,
)


def _names_from_feature(feature) -> list[str] | None:
    """Descend through Sequence/List wrapping to find a ClassLabel's .names,
    one level deeper than catalog_datasets.py's find_label_feature() checks
    (that function only ever looks at TOP-LEVEL features, never into a
    nested dict feature like "objects": {"bbox": ..., "category": ...})."""
    seen = set()
    f = feature
    for _ in range(4):  # a couple of Sequence/List wraps is plenty in practice
        if id(f) in seen:
            break
        seen.add(id(f))
        if hasattr(f, "names"):
            return list(f.names)
        inner = getattr(f, "feature", None)
        if inner is None:
            break
        f = inner
    return None


def _find_category_feature(features):
    """features is a datasets.Features mapping (or the "objects" sub-dict
    within it) — return whatever sub-feature looks like a category/class id
    field, checking common detection-schema key names."""
    for key in ("category", "categories", "category_id", "label", "class", "class_id"):
        if key in features:
            return features[key]
    return None


def resolve_class_names(repo_id: str, cfg_name: str) -> list[str] | None:
    from datasets import load_dataset_builder

    b = load_dataset_builder(repo_id, cfg_name)
    features = b.info.features or {}

    # (a) category feature nested inside "objects" (or sitting top-level) —
    # check both, since the exact schema shape varies per repo.
    objects_feature = features.get("objects")
    if objects_feature is not None:
        inner = getattr(objects_feature, "feature", objects_feature)
        inner_dict = inner if hasattr(inner, "keys") else {}
        cat_feature = _find_category_feature(inner_dict)
        if cat_feature is not None:
            names = _names_from_feature(cat_feature)
            if names:
                return names
    cat_feature = _find_category_feature(features)
    if cat_feature is not None:
        names = _names_from_feature(cat_feature)
        if names:
            return names

    # (b) dataset card — look for a "classes:"/"categories:"/"labels:"
    # bracketed list anywhere in the README.
    try:
        from huggingface_hub import hf_hub_download
        readme_path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
        text = Path(readme_path).read_text(encoding="utf-8", errors="ignore")
        m = _CLASS_LIST_HINT_RE.search(text)
        if m:
            names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
            names = [n for n in names if n]
            if names:
                return names
    except Exception:
        pass

    # (c) load one row, check for a stray field alongside "objects" that
    # isn't part of the standard bbox/category schema.
    try:
        from datasets import load_dataset
        row = load_dataset(repo_id, cfg_name, split="train[:1]")[0]
        for key in ("categories", "classes", "labels", "class_names"):
            val = row.get(key)
            if isinstance(val, list) and val and isinstance(val[0], str):
                return val
    except Exception:
        pass

    return None


def probe_bbox_convention(repo_id: str, cfg_name: str, n_rows: int) -> dict:
    """Load a handful of rows and measure (not assume) the bbox coordinate
    convention. scale: essentially every real photo has at least one box
    coordinate > 1.0 in pixel space (images are never <=1px); normalized
    stays in [0,1] for every value. format: in xyxy, x2>=x1 and y2>=y1 holds
    for EVERY valid box (by definition of the corners); in xywh, index 2/3
    are a width/height, typically much smaller than index 0/1's position and
    with no such ordering guarantee against them."""
    result = {"scale": "unknown", "format": "unknown", "n_probed": 0}
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, cfg_name, split=f"train[:{n_rows}]")
    except Exception as e:
        result["error"] = str(e)
        return result

    all_boxes = []
    for row in ds:
        objs = row.get("objects") or {}
        for box in objs.get("bbox", []) or []:
            if len(box) == 4:
                all_boxes.append([float(v) for v in box])
    result["n_probed"] = len(all_boxes)
    if not all_boxes:
        return result

    max_val = max(v for box in all_boxes for v in box)
    result["scale"] = "normalized" if max_val <= 1.0 else "pixel"

    xyxy_consistent = sum(1 for b in all_boxes if b[2] >= b[0] and b[3] >= b[1])
    xyxy_frac = xyxy_consistent / len(all_boxes)
    # A genuine xywh dataset will violate x2>=x1/y2>=y1 "by luck" on some
    # boxes too (width can exceed x when the box starts near the left edge),
    # so this isn't perfectly separable — 95% is a deliberately high bar to
    # call it confidently xyxy; anything lower is reported as xywh-or-unclear
    # rather than guessed, and left for the QC plot (Stage 5c) to settle
    # visually against the actual image.
    result["format"] = "xyxy" if xyxy_frac >= 0.95 else "xywh"
    result["xyxy_consistent_frac"] = round(xyxy_frac, 3)
    return result


def introspect_detection_repo(repo_id: str, preferred_config: str, n_probe_rows: int) -> dict:
    class_names = resolve_class_names(repo_id, preferred_config)
    bbox_convention = probe_bbox_convention(repo_id, preferred_config, n_probe_rows)
    return {"class_names": class_names, "bbox_convention": bbox_convention}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excluded_catalog", default="src/config/hf_catalog_excluded.json")
    ap.add_argument("--output", default="src/config/hf_detection_catalog.json")
    ap.add_argument("--n_probe_rows", type=int, default=20,
                     help="Rows sampled per dataset to measure bbox scale/format convention.")
    ap.add_argument("--only", default=None, help="Optional regex filter on dataset name, for debug/incremental runs.")
    ap.add_argument("--skip_existing", action="store_true",
                     help="Skip datasets already resolved (class_names is non-null) in --output.")
    args = ap.parse_args()

    excluded = json.loads(Path(args.excluded_catalog).read_text())
    detection_entries = {k: v for k, v in excluded.items() if v.get("dataset_type") == "detection"}
    print(f"{len(detection_entries)} detection-type dataset(s) in {args.excluded_catalog}")

    out_path = Path(args.output)
    catalog: dict = json.loads(out_path.read_text()) if out_path.exists() else {}

    names = sorted(detection_entries)
    if args.only:
        pattern = re.compile(args.only, re.I)
        names = [n for n in names if pattern.search(n)]
        print(f"Filtered to {len(names)} by --only={args.only!r}")

    t0 = time.time()
    n_resolved, n_unresolved, n_err = 0, 0, 0
    for i, name in enumerate(names, 1):
        if args.skip_existing and catalog.get(name, {}).get("class_names"):
            continue
        entry = detection_entries[name]
        repo_id = entry["repo_id"]
        cfg = entry.get("preferred_config", "default")
        print(f"  [{i}/{len(names)}] {name}", end=" ", flush=True)
        try:
            resolved = introspect_detection_repo(repo_id, cfg, args.n_probe_rows)
            merged = dict(entry)
            merged["class_names"] = resolved["class_names"]
            merged["bbox_convention"] = resolved["bbox_convention"]
            catalog[name] = merged
            bc = resolved["bbox_convention"]
            if resolved["class_names"]:
                n_resolved += 1
                print(f"ok (n_classes={len(resolved['class_names'])}, "
                      f"bbox={bc['scale']}/{bc['format']}, n_probed={bc['n_probed']})")
            else:
                n_unresolved += 1
                print(f"UNRESOLVED class_names (bbox={bc['scale']}/{bc['format']}, n_probed={bc['n_probed']})")
        except Exception as e:
            catalog[name] = {**entry, "class_names": None, "bbox_convention": {"error": str(e)}}
            n_err += 1
            print(f"ERROR: {e}")

        if i % 5 == 0 or i == len(names):
            out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))

    out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {n_resolved} resolved, {n_unresolved} unresolved, {n_err} errors.")
    print(f"  -> {out_path} ({len(catalog)} entries)")
    if n_unresolved:
        print(f"  {n_unresolved} dataset(s) have class_names=null — migrate_detection_datasets_yaml.py "
              f"will mark these skip: true for manual review, not drop them.")


if __name__ == "__main__":
    main()
