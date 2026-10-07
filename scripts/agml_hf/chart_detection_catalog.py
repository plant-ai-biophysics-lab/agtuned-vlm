"""
Chart generation for the object-detection catalog — detection analog of
download_and_sample.py's classes_by_task_type_crop_nested_pie.png, kept as
its own standalone script rather than folded into
download_and_sample_detection.py (which has no --plot_only/--chart_dir of
its own yet).

Every detection dataset's task_type is a constant "object_detection" (see
detection_datasets.yaml's own header comment) — there is no disease/pest/
species/quality split to reuse as this chart's inner ring the way
classification's own chart has. Two alternative inner-ring groupings are
built here, as two separate PNGs, so you can pick whichever reads better:

  A) detection_classes_by_cropcategory_nested_pie.png — inner ring = crop
     category (reuses download_and_sample.py's own CROP_CATEGORY/
     _crop_category taxonomy directly), outer ring = individual crop name
     within that category.
  B) detection_classes_by_target_type_nested_pie.png — inner ring = what
     the box is actually drawn around (produce/fruit, plant part, disease
     symptom, pest/damage, weed/non-crop, other) — a NEW heuristic
     (_detection_target_type, below), built by inspecting the real 125
     distinct bbox class names across the 49 currently-active detection
     datasets (disease-symptom classes like "Tomato leaf mosaic virus" turned
     out to be the single largest group — not something the classification
     chart's taxonomy has a slot for at all). Outer ring = crop category.

Both reuse _render_task_nested_donut/_task_outer_entries/_crop_category/
CROP_CATEGORY/_add_leader_labels from download_and_sample.py rather than
duplicating that rendering code — only the TASK_COLOR/_display_task
defaults needed a color_map/display_fn override hook added there (see that
function's docstring) since this chart's inner-ring keys (crop categories /
target types) aren't classification task types.

Usage:
    uv run python scripts/agml_hf/chart_detection_catalog.py \\
        --config src/config/detection_datasets.yaml \\
        --manifest dataset/selected_images_detection_10.json \\
        --chart_dir outputs/agml_hf_detection/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.agml_hf.download_and_sample import (  # noqa: E402
    CROP_CATEGORY, _crop_category, _render_task_nested_donut,
)
from src.common.dataset_config import active_datasets  # noqa: E402

# Crop-category inner ring (chart A) — same categories CROP_CATEGORY already
# defines for the classification chart's outer ring, just promoted to the
# inner ring here. Colors chosen distinct from TASK_COLOR (classification's
# own palette) so a reader never confuses the two charts' color coding.
CROP_CATEGORY_COLOR = {
    "fruit": "#55A868", "vegetable": "#4C72B0", "grain": "#C44E52",
    "legume": "#8172B2", "cash crop": "#CCB974", "herb/spice": "#64B5CD",
    "nut/oilseed": "#937860", "ornamental": "#DA8BC3",
    "non-crop species": "#8C8C8C", "multi-crop / unspecified": "#B0673F",
    "other": "#888888",
}

# Detection target-type inner ring (chart B) — grounded in the real 125
# distinct class names across the 49 active detection datasets (see this
# module's docstring and the dev session that built this: every name was
# printed and manually reviewed, not guessed). Checked in priority order —
# e.g. "Tomato two spotted spider mites leaf" must hit PEST (mite) before
# ORGAN (leaf) would otherwise claim it; "leaf spot" is a disease phrase,
# not "spot" alone (which would false-positive on "spotted").
_PEST_WORDS = ["mite", "aphid", "borer", "weevil", "thrip", "hopper"]
_DISEASE_WORDS = ["scab", "rust", "blight", "mildew", "mold", "mosaic", "bacterial",
                   "septoria", "black rot", "leaf spot", "virus"]
_WEED_WORDS = ["weed", "horseweed", "kochia", "ragweed", "redrootpigweed", "waterhemp", "no_crop"]
_ORGAN_WORDS = ["leaf", "stem", "shoot", "bud", "foliage", "canopy", "crown", "growing", "turning"]
_PRODUCE_WORDS = ["fruit", "berry", "grape", "apple", "avocado", "mango", "orange", "strawberry",
                   "quince", "date", "pod", "bean", "capsicum", "corn", "maize", "onion",
                   "rockmelon", "cacao", "wheat head", "panicle", "sunflower", "ripe", "unripe",
                   "ripening", "mature", "immature", "spoiled", "cavity", "almond", "coffee", "leek"]

TARGET_TYPE_COLOR = {
    "produce_fruit": "#55A868", "plant_part": "#4C72B0", "disease_symptom": "#C44E52",
    "pest_damage": "#DD8452", "weed_noncrop": "#8172B2", "other": "#888888",
}
_TARGET_TYPE_DISPLAY = {
    "produce_fruit": "produce / fruit", "plant_part": "plant part",
    "disease_symptom": "disease symptom", "pest_damage": "pest / damage",
    "weed_noncrop": "weed / non-crop", "other": "other / ambiguous",
}


def _detection_target_type(class_name: str) -> str:
    """What a detection box is actually drawn around, not what crop it's
    on — see this module's docstring for how these buckets were derived
    from the real class-name vocabulary. "other" for bare/ambiguous labels
    (color names, ripeness-stage codes like "T1", ground-truth placeholders
    like "0"/"object") — degrades gracefully, same convention as
    _crop_category's own "other" fallback."""
    n = class_name.lower()
    if any(w in n for w in _PEST_WORDS):
        return "pest_damage"
    if any(w in n for w in _DISEASE_WORDS):
        return "disease_symptom"
    if any(w in n for w in _WEED_WORDS):
        return "weed_noncrop"
    if any(w in n for w in _ORGAN_WORDS):
        return "plant_part"
    if "fruit" in n or any(w in n for w in _PRODUCE_WORDS):
        return "produce_fruit"
    if "flower" in n or "plant" in n:
        return "plant_part"
    return "other"


def _display_crop_category(cat: str) -> str:
    return cat


def _display_target_type(t: str) -> str:
    return _TARGET_TYPE_DISPLAY.get(t, t)


def _detection_crop_label(crop_val) -> str:
    """detection_datasets.yaml's own crop field -> a display label, same
    "no single crop applies" convention download_and_sample.py's classes_by_
    task_type_crop_nested_pie.png already uses for null/"mixed" crop (weed
    detection datasets, multi-crop scenes) — confirmed real value via
    detection_datasets.yaml ('null' and the literal string "mixed" both
    appear)."""
    if isinstance(crop_val, list):
        return " / ".join(crop_val)
    if not crop_val or crop_val == "mixed":
        return "multi-crop / unspecified"
    return crop_val


def make_detection_charts(config_path: Path, manifest_path: Path, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    datasets = active_datasets(config_path)
    ds_by_name = {d.agml_name: d for d in datasets}
    manifest = json.loads(manifest_path.read_text())
    ds_entries = {name: entry for name, entry in manifest["datasets"].items() if name in ds_by_name}

    missing = sorted(set(ds_by_name) - set(ds_entries))
    if missing:
        print(f"  [warn] {len(missing)} active dataset(s) not in manifest, skipped: {missing}")
    if not ds_entries:
        print("  No datasets in manifest overlap the active config — skipping detection charts.")
        return

    # ── Chart A: crop category (inner) -> individual crop (outer) ──────────
    cropcat_counts: Counter = Counter()
    cropcat_by_cat: dict[str, Counter] = defaultdict(Counter)
    for name, entry in ds_entries.items():
        cfg = ds_by_name[name]
        crop_val = cfg.crop
        crop_label = _detection_crop_label(crop_val)
        cat = _crop_category(crop_label)
        n_classes = len(entry["classes"])
        cropcat_counts[cat] += n_classes
        cropcat_by_cat[cat][crop_label] += n_classes

    cropcat_types = sorted(cropcat_counts, key=lambda c: -cropcat_counts[c])
    _render_task_nested_donut(
        chart_dir, cropcat_types, cropcat_counts, cropcat_by_cat,
        top_n=8, unit_name="crops", breakdown_key="crop",
        breakdown_filename="detection_classes_by_cropcategory_breakdown.csv",
        chart_filename="detection_classes_by_cropcategory_nested_pie.png",
        color_map=CROP_CATEGORY_COLOR, display_fn=_display_crop_category,
    )

    # ── Chart B: detection target type (inner) -> crop category (outer) ────
    target_counts: Counter = Counter()
    target_by_type: dict[str, Counter] = defaultdict(Counter)
    for name, entry in ds_entries.items():
        cfg = ds_by_name[name]
        crop_val = cfg.crop
        crop_label = _detection_crop_label(crop_val)
        cat = _crop_category(crop_label)
        for cls in entry["classes"]:
            target = _detection_target_type(cls)
            target_counts[target] += 1
            target_by_type[target][cat] += 1

    target_types = sorted(target_counts, key=lambda t: -target_counts[t])
    _render_task_nested_donut(
        chart_dir, target_types, target_counts, target_by_type,
        top_n=8, unit_name="crop categories", breakdown_key="crop_category",
        breakdown_filename="detection_classes_by_target_type_breakdown.csv",
        chart_filename="detection_classes_by_target_type_nested_pie.png",
        color_map=TARGET_TYPE_COLOR, display_fn=_display_target_type,
    )

    print(f"\n  {len(ds_entries)} active detection dataset(s) charted "
          f"({sum(len(e['classes']) for e in ds_entries.values())} classes total).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="src/config/detection_datasets.yaml")
    ap.add_argument("--manifest", default="dataset/selected_images_detection_10.json")
    ap.add_argument("--chart_dir", default="outputs/agml_hf_detection/")
    args = ap.parse_args()
    make_detection_charts(Path(args.config), Path(args.manifest), Path(args.chart_dir))


if __name__ == "__main__":
    main()
