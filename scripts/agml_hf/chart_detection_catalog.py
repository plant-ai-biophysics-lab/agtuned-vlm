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
import re
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
    "non-crop species": "#8C8C8C", "multi-crop": "#B0673F",
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
# Split out of the old catch-all "other" bucket (2026-10, real review of
# every one of its 62 class names across the full active+held-out pool):
# bare color/stage codes from fruit ripeness-grading datasets (strawberry/
# tomato color classes, date maturity stages 0-3, tea age stages T1-T4,
# pomegranate "mid-growth") — genuinely about WHEN in development the box
# was drawn, not just that it's produce, so kept distinct from
# produce_fruit rather than folded into it. Checked AFTER produce_fruit/
# organ/pest/disease/weed (all above) so a specific match there always
# wins — e.g. "red_spider_mite" hits PEST first, never lands here.
_RIPENESS_WORDS = ["green", "red", "pink", "white", "b_green", "l_green", "lw", "sw",
                     "large_white", "small_white", "mid-growth", "t1", "t2", "t3", "t4"]
# The generic "this is the crop (not the weed)" label in several crop-vs-
# weed binary detection datasets, plus the specific crop-species names used
# the same way in weed_crop_detection (Canola/Field Pea/Flax/Lentil/Sugar
# beet — that dataset's "classes" ARE crop species, not organs/produce).
# Checked LAST (right before the final "other" fallback) since bare "crop"
# or a bare species name could otherwise shadow a more specific match.
_CROP_PLANT_WORDS = ["crop", "canola", "field pea", "flax", "lentil", "sugar beet", "tree"]

TARGET_TYPE_COLOR = {
    "produce_fruit": "#55A868", "plant_part": "#4C72B0", "disease_symptom": "#C44E52",
    "pest_damage": "#DD8452", "weed_noncrop": "#8172B2", "ripeness_stage": "#CCB974",
    "crop_plant": "#64B5CD", "other": "#888888",
}
_TARGET_TYPE_DISPLAY = {
    "produce_fruit": "produce / fruit", "plant_part": "plant part",
    "disease_symptom": "disease symptom", "pest_damage": "pest / damage",
    "weed_noncrop": "weed / non-crop", "ripeness_stage": "ripeness / growth stage",
    "crop_plant": "crop plant, non-weed", "other": "other / ambiguous",
}


def _detection_target_type(class_name: str) -> str:
    """What a detection box is actually drawn around, not what crop it's
    on — see this module's docstring for how these buckets were derived
    from the real class-name vocabulary. "other" for bare/ambiguous labels
    genuinely unresolvable from the class name alone (ground-truth
    placeholders like "0"/"object"/"Maybe") — degrades gracefully, same
    convention as _crop_category's own "other" fallback."""
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
    if any(w in n for w in _RIPENESS_WORDS):
        return "ripeness_stage"
    if any(w in n for w in _CROP_PLANT_WORDS):
        return "crop_plant"
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
        return "multi-crop"
    return crop_val


def _display_dataset_name(agml_name: str) -> str:
    """Turn an agml_name into a short, readable outer-ring sub-label for
    the multi-crop bucket (e.g. "plant_doc_detection" -> "plant doc") —
    used ONLY for datasets whose crop genuinely spans multiple species
    (confirmed real case, 2026-10: with every single-crop dataset's "crop"
    field correctly filled in, "multi-crop" is dominated by one 28-class
    dataset (plant_doc_detection) plus a handful of smaller ones
    (weed_crop_detection, vegetable_crop_early_detection, ...) — a single
    opaque "multi-crop (N)" outer wedge hid which dataset that N actually
    came from. Strips the generic "_detection"/"_detection_augmented"
    suffix (uninformative here — every entry in this file is a detection
    dataset) and converts underscores to spaces; deliberately NOT
    title-cased — matches the lowercase convention every other outer-ring
    crop label in this chart already uses."""
    name = re.sub(r"_detection(_augmented)?$", "", agml_name)
    return name.replace("_", " ")


def _load_detection_heldout_classes(heldout_manifest_path: Path) -> dict[str, list[str]]:
    """The 15 detection datasets held out for Paper 2 Stage 4's
    skill-memory pipeline (`skip: true` in detection_datasets.yaml, notes
    starting "HELD OUT for Paper 2 Stage 4" — see that file's own header
    comment) never appear in the active 10/class sampling manifest
    (dataset/selected_images_detection_10.json), so they're invisible to
    this chart's main loop unless folded in separately here — same role
    _digital_green_class_task_crop_counts/_species_class_task_crop_counts
    play for the classification chart. Returns {agml_dataset: [distinct
    class names]}, same shape as an active manifest entry's "classes" list
    (just a list, not the sampled-index dict — image indices aren't needed
    for a class-count chart), built from
    dataset/detection_heldout_selected_images.json's own row-level
    `boxes[].category_name` (download_heldout_detection.py's output; no
    `--classes` key to read at the dataset level the way the active
    manifest has one). Returns {} if the manifest hasn't been generated yet
    (chart generation shouldn't hard-fail just because this optional extra
    source is missing)."""
    if not heldout_manifest_path.exists():
        return {}
    rows = json.loads(heldout_manifest_path.read_text())["rows"]
    classes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for r in rows.values():
        ds_name = r.get("agml_dataset")
        if not ds_name:
            continue
        for b in r.get("boxes") or []:
            name = (b.get("category_name") or "").strip()
            if name:
                classes_by_dataset[ds_name].add(name)
    return {name: sorted(classes) for name, classes in classes_by_dataset.items()}


def make_detection_charts(
    config_path: Path, manifest_path: Path, chart_dir: Path, outer_label_fontsize: int = 9,
    small_inner_pct_threshold: float = 8,
    detection_heldout_manifest: Path | None = None,
) -> None:
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

    # Fold in the 15 Stage-4-held-out detection datasets alongside the 39
    # active ones, so this chart's totals match the paper's real combined
    # dataset count (54 datasets / 200 classes) rather than only the
    # active-sampling subset. Their crop/category metadata still lives in
    # detection_datasets.yaml (skip: true doesn't remove the entry, just
    # excludes it from active_datasets()) — load_datasets() (unfiltered)
    # resolves it the same way --include_skipped does for the
    # classification chart.
    heldout_classes = (
        _load_detection_heldout_classes(detection_heldout_manifest) if detection_heldout_manifest else {}
    )
    if heldout_classes:
        from src.common.dataset_config import load_datasets
        full_ds_by_name = {d.agml_name: d for d in load_datasets(config_path)}
        n_heldout_classes = 0
        for name, class_names in heldout_classes.items():
            cfg = full_ds_by_name.get(name)
            if cfg is None:
                print(f"  [warn] held-out detection dataset {name!r} not found in {config_path} — skipped")
                continue
            ds_by_name[name] = cfg
            ds_entries[name] = {"classes": class_names}
            n_heldout_classes += len(class_names)
        print(f"  Stage-4 held-out detection: +{len(heldout_classes)} datasets, "
              f"+{n_heldout_classes} classes folded in")

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
        # A genuinely multi-crop dataset (plant_doc_detection spans apple/
        # bell pepper/.../tomato, weed_crop_detection spans several row
        # crops, ...) still gets one shared crop_label ("multi-crop") from
        # _detection_crop_label — fine for the INNER ring (that's correctly
        # one category), but it collapsed the OUTER ring into a single
        # opaque "multi-crop (N)" wedge that hid which dataset(s) N came
        # from. Use the dataset's own name there instead, same role an
        # individual crop name plays for every other category.
        outer_label = crop_label if cat != "multi-crop" else _display_dataset_name(name)
        cropcat_by_cat[cat][outer_label] += n_classes

    cropcat_types = sorted(cropcat_counts, key=lambda c: -cropcat_counts[c])
    _render_task_nested_donut(
        chart_dir, cropcat_types, cropcat_counts, cropcat_by_cat,
        top_n=8, unit_name="crops", breakdown_key="crop",
        breakdown_filename="detection_classes_by_cropcategory_breakdown.csv",
        chart_filename="detection_classes_by_cropcategory_nested_pie.png",
        color_map=CROP_CATEGORY_COLOR, display_fn=_display_crop_category,
        outer_label_fontsize=outer_label_fontsize,
        small_inner_pct_threshold=small_inner_pct_threshold,
        # Several SMALL categories (grain, vegetable) were each naming
        # 3-5 singleton/doubleton crops individually since top_n=8 never
        # truncated a category with fewer than 8 members to begin with —
        # that's what made this chart's top-left corner crowded, not the
        # big fruit category (which top_n=8 already summarized fine).
        # min_outer_count=2 folds any crop appearing only once into its
        # category's "N remaining crops" bucket instead, leaving every
        # other category (including fruit) untouched.
        min_outer_count=2,
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
        outer_label_fontsize=outer_label_fontsize,
        small_inner_pct_threshold=small_inner_pct_threshold,
    )

    print(f"\n  {len(ds_entries)} detection dataset(s) charted "
          f"({sum(len(e['classes']) for e in ds_entries.values())} classes total).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="src/config/detection_datasets.yaml")
    ap.add_argument("--manifest", default="dataset/selected_images_detection_10.json")
    ap.add_argument("--chart_dir", default="outputs/agml_hf_detection/")
    ap.add_argument("--outer_label_fontsize", type=int, default=9,
                     help="Font size for the outer leader-line labels on both nested donuts — "
                          "default 9 matches every other chart in this pipeline.")
    ap.add_argument("--small_inner_pct_threshold", type=float, default=8,
                     help="Inner-ring wedges below this percent get a leader-labeled callout "
                          "outside the ring instead of inline text — default 8 (higher than "
                          "classification's 3) since these two charts have more, smaller inner "
                          "wedges (crop category / target type) that would otherwise overlap.")
    ap.add_argument("--detection_heldout_manifest", default="dataset/detection_heldout_selected_images.json",
                     help="The 15 detection datasets held out for Paper 2 Stage 4's skill-memory "
                          "pipeline (download_heldout_detection.py's output) — their classes are "
                          "folded into both charts alongside the 39 active datasets. Pass '' to "
                          "skip (missing file is also a no-op).")
    args = ap.parse_args()
    make_detection_charts(Path(args.config), Path(args.manifest), Path(args.chart_dir),
                           outer_label_fontsize=args.outer_label_fontsize,
                           small_inner_pct_threshold=args.small_inner_pct_threshold,
                           detection_heldout_manifest=(
                               Path(args.detection_heldout_manifest) if args.detection_heldout_manifest else None
                           ))


if __name__ == "__main__":
    main()
