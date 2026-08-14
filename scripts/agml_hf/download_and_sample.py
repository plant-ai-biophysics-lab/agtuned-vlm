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

Resuming after a partial/rate-limited run: pass --skip_existing to re-use any
dataset already recorded in --output (from a prior run) instead of
re-downloading/re-sampling it, and only process datasets missing or previously
failed. This only applies when --n_per_class matches the existing manifest's
recorded value — changing --n_per_class invalidates every already-capped
dataset (see caveat below), so --skip_existing is ignored (with a warning) if
the cap changed. HuggingFace Hub 429s (rate limiting) are retried per-dataset
with exponential backoff before being recorded as a failure.

Caveat when changing --n_per_class: only classes that exceed the cap are
subsampled at all (smaller classes are unaffected either way), but for classes
that DO get capped, a different --n_per_class is not guaranteed to select a
superset/subset of a previous run's choice — same seed, different sample size,
generally a different draw. Treat each --n_per_class value as producing its own
independent (but internally reproducible) selection, not an incremental one.

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
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import active_datasets, DatasetConfig, subsample_class_balanced  # noqa: E402

TASK_COLOR = {
    "disease": "#4C72B0",
    "pest_damage": "#DD8452",
    "species_identification": "#55A868",
    "quality_identification": "#C44E52",
    # Combined pseudo-task used by the crop nested donut only (see
    # make_charts) — quality_identification and pest_damage are both small
    # enough that their outer-ring labels crowd/overlap next to each other,
    # so that one chart merges them into a single inner-ring section.
    "quality_pest_damage": "#B0673F",
}

RETRYABLE_STATUS = {429, 503}
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 10


def is_retryable(exc: Exception) -> bool:
    from huggingface_hub.utils import HfHubHTTPError
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None)
        if status in RETRYABLE_STATUS:
            return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES or not is_retryable(e):
                raise
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"\n    rate-limited ({e}), retry {attempt}/{MAX_RETRIES} in {wait}s...",
                  end=" ", flush=True)
            time.sleep(wait)


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


def _shade_gradient_by_value(base_hex: str, values: list) -> list:
    """Colors aligned 1:1 with `values`' own order, but lightness is
    assigned by each value's RANK (largest -> darkest shade), not by its
    position in the list. Using position instead of rank would make a
    "N remaining ..." bucket (always appended last) get an arbitrary shade
    unrelated to its actual class count — it usually IS the largest single
    value (so position happened to match rank and looked fine), but not
    always: species' "3 remaining crops (11)" is smaller than
    medicinal_plant (174), so position-based shading made the small bucket
    darkest and the large one lightest — backwards.
    """
    import colorsys
    import matplotlib.colors as mcolors
    import numpy as np

    n = len(values)
    if n <= 1:
        return [base_hex]
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
    lightnesses = np.linspace(0.32, 0.78, n)  # rank 0 (largest value) gets index 0 = darkest
    order = sorted(range(n), key=lambda i: -values[i])  # index of largest value first
    result = [None] * n
    for rank, idx in enumerate(order):
        result[idx] = colorsys.hls_to_rgb(h, lightnesses[rank], s)
    return result


def _display_task(t: str) -> str:
    if t == "quality_pest_damage":
        return "quality/pest damage"
    t = t.replace("_identification", "")
    return "pest/damage" if t == "pest_damage" else t


def _spread_label_positions(natural_ys: list[float], min_gap: float) -> list[float]:
    """Adjust `natural_ys` (one wedge's ideal, unconstrained label y position
    each) to be at least `min_gap` apart while preserving order — standard
    two-pass stacking: a forward pass pushes any position that's too close
    to its (already-adjusted) predecessor up to exactly min_gap away, then a
    backward pass pulls each position back down to exactly min_gap below its
    (already-finalized) successor wherever the forward pass left slack this
    can safely close up. After both passes every adjacent pair is >=
    min_gap apart, guaranteed — each index is only ever touched once per
    pass, and each pass only tightens relative to an already-settled
    neighbor, so neither pass can undo the other's guarantee."""
    order = sorted(range(len(natural_ys)), key=lambda i: natural_ys[i])
    n = len(order)
    pos = [natural_ys[i] for i in order]

    for k in range(1, n):
        if pos[k] - pos[k - 1] < min_gap:
            pos[k] = pos[k - 1] + min_gap
    for k in range(n - 2, -1, -1):
        if pos[k + 1] - pos[k] < min_gap:
            pos[k] = pos[k + 1] - min_gap

    result = [0.0] * n
    for k, idx in enumerate(order):
        result[idx] = pos[k]
    return result


def _add_leader_labels(ax, wedges, labels, radius, xoff, yoff, fontsize, min_gap: float = 0.075) -> None:
    """Leader-line labels around a ring, split into a left column (x<0) and
    a right column (x>=0) — each column's label y-positions are indepedently
    spread apart (see _spread_label_positions) so tightly clustered wedges
    (several thin slices within a narrow angular range) never produce
    overlapping text, regardless of how close their wedges actually are."""
    import numpy as np

    sides: dict[bool, list[int]] = {True: [], False: []}  # True = right (x>=0)
    positions = []
    for wedge, label in zip(wedges, labels):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x, y = np.cos(np.deg2rad(ang)) * radius, np.sin(np.deg2rad(ang)) * radius
        positions.append((ang, x, y))
        sides[x >= 0].append(len(positions) - 1)

    adjusted_y = [0.0] * len(labels)
    for _, idxs in sides.items():
        if not idxs:
            continue
        natural = [yoff * positions[i][2] for i in idxs]
        spread = _spread_label_positions(natural, min_gap)
        for i, y in zip(idxs, spread):
            adjusted_y[i] = y

    for i, label in enumerate(labels):
        ang, x, _ = positions[i]
        ha = "left" if x >= 0 else "right"
        # "angle" (the elbow-shaped connector) computes its bend via a
        # line-intersection that's undefined right at exactly 0/180 deg (a
        # wedge pointing due left/right) and looks stretched/bad well before
        # that too — a wedge within ~20 deg of the horizontal gets a plain
        # straight connector instead; everything else keeps the elbow look.
        nearest_pole = round(ang / 180) * 180
        near_horizontal = abs(ang - nearest_pole) < 20
        connectionstyle = "arc3,rad=0.0" if near_horizontal else f"angle,angleA=0,angleB={ang}"
        ax.annotate(
            label, xy=(x, positions[i][2]), xytext=(xoff * (1 if x >= 0 else -1), adjusted_y[i]),
            horizontalalignment=ha, verticalalignment="center", fontsize=fontsize,
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6, connectionstyle=connectionstyle),
        )


# Broad crop-type taxonomy for the "remaining crops" bucket on the crop
# nested donut — splits what would otherwise be one opaque "N remaining
# crops" wedge into a few named sub-wedges ("N remaining fruit crops",
# "N remaining vegetable crops", ...) so there's still visible structure in
# whatever falls outside the top-N individually-shown crops. Deliberately
# coarse (standard agricultural categories, not a botanical taxonomy) —
# compound "a / b" multi-crop labels are categorized by their first
# component; anything not listed here falls into "other" rather than
# raising, since new crop names get added to datasets.yaml over time.
CROP_CATEGORY = {
    # Fruit
    "apple": "fruit", "banana": "fruit", "citrus": "fruit", "coconut": "fruit",
    "custard_apple": "fruit", "date": "fruit", "date_palm": "fruit", "dragonfruit": "fruit",
    "durian": "fruit", "fig": "fruit", "grape": "fruit", "guava": "fruit", "java_plum": "fruit",
    "jujube": "fruit", "mango": "fruit", "orange": "fruit", "plum": "fruit", "pomegranate": "fruit",
    "sapota": "fruit", "strawberry": "fruit", "fruit": "fruit", "aegle_marmelos": "fruit",
    "hog_plum": "fruit", "lemon": "fruit", "mulberry": "fruit",
    # Vegetable
    "ash_gourd": "vegetable", "cauliflower": "vegetable", "chilli": "vegetable",
    "cucumber": "vegetable", "eggplant": "vegetable", "okra": "vegetable", "onion": "vegetable",
    "spinach": "vegetable", "tomato": "vegetable", "taro": "vegetable", "vegetable": "vegetable",
    "mint": "vegetable",  # botanically an herb, but its only appearance (quality task) was a
                           # lone 4-class wedge landing right at the ring's 0deg/360deg seam —
                           # folded in with vegetable crops to avoid that connector artifact
                           # rather than leaving a single-crop herb/spice sliver there
    # Grain / cereal
    "corn": "grain", "maize": "grain", "paddy": "grain", "rice": "grain", "sorghum": "grain",
    # Legume
    "bean": "legume", "blackgram": "legume", "lentil": "legume", "soybean": "legume",
    # Cash / industrial crop
    "cotton": "cash crop", "jute": "cash crop", "tea": "cash crop", "coffee": "cash crop",
    "arabica_coffee": "cash crop", "sugarcane": "cash crop",
    # Herb / spice / medicinal
    "betel_leaf": "herb/spice", "chitrak": "herb/spice", "lemongrass": "herb/spice",
    "medicinal_plant": "herb/spice", "moringa": "herb/spice",
    # Nut / oilseed
    "pistachio": "nut/oilseed", "sunflower": "nut/oilseed",
    # Ornamental
    "rose": "ornamental", "money_plant": "ornamental",
    # Non-crop
    "weed / non-crop species": "non-crop species",
    "multi-crop / unspecified": "multi-crop / unspecified",
}


def _crop_category(crop_label: str) -> str:
    """Exact match first — "weed / non-crop species" and "multi-crop /
    unspecified" are pipeline-generated fallback labels (see
    _AMBIGUOUS_CROP_LABEL/_AMBIGUOUS_CROP_DEFAULT), single semantic units
    despite containing " / ", not real multi-crop combinations. Only falls
    back to the first "/"-joined component (e.g. "eggplant / tomato" ->
    "eggplant") for genuine multi-crop labels not in CROP_CATEGORY verbatim.
    "other" for anything not found either way, so a new crop name in
    datasets.yaml degrades gracefully instead of crashing chart generation."""
    if crop_label in CROP_CATEGORY:
        return CROP_CATEGORY[crop_label]
    first = crop_label.split(" / ")[0].strip()
    return CROP_CATEGORY.get(first, "other")


def _write_breakdown_csv(
    chart_dir: Path, filename: str, breakdown_key: str,
    class_task_types: list[str], groups_by_task: dict[str, Counter], top_n: int,
    category_fn=None,
) -> Path:
    """Full breakdown (every value, not just what's shown on-chart) —
    shared by the nested donut and the stacked bar, which both truncate
    on-chart and point here (by filename) for the rest. category_fn, when
    given, adds a "category" column (see _crop_category) so the chart's
    "remaining X crops" grouping is traceable back to individual crop names."""
    import csv as _csv

    path = chart_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        header = ["task_type", breakdown_key, "n_classes", "shown_in_chart"]
        if category_fn is not None:
            header.append("category")
        writer.writerow(header)
        for t in class_task_types:
            group = groups_by_task[t]
            top_names = {name for name, _ in group.most_common(top_n)}
            for key, n in group.most_common():
                row = [t, key, n, "top" if key in top_names else "other"]
                if category_fn is not None:
                    row.append(category_fn(key))
                writer.writerow(row)
    print(f"  Saved: {path}")
    return path


def _task_outer_entries(
    class_task_types: list[str], groups_by_task: dict[str, Counter], top_n: int, unit_name: str,
    category_fn=None,
) -> dict[str, list[tuple[str, int, tuple]]]:
    """{task: [(label, value, color), ...]}.

    category_fn=None (the location chart): original behavior — top_n
    individual entries by rank, plus one "N remaining {unit_name}" catch-all.

    category_fn given (the crop chart): every crop is binned into its
    CROP_CATEGORY first — no rank-based top_n, no hand-maintained exception
    list. Within each category, one crop "speaks for" the whole category
    (keeps its own name as the wedge label, absorbing the rest of the
    category's count silently) only if it clearly DOMINATES that category
    (>=_DOMINANCE_FRAC of the category's total, e.g. medicinal_plant among
    herb/spice crops, rice among grain crops) — otherwise the category has
    no single obvious "headline" crop, so it's shown as one combined
    "{category} crops" wedge (e.g. mango + banana + citrus + generic "fruit"
    + everything else fruit-shaped becomes one "fruit crops" wedge, since no
    single one of them represents "fruit" on its own). This is what makes
    "group citrus with fruit" or "group tomato/eggplant with vegetable" a
    property of the taxonomy (CROP_CATEGORY) rather than a per-chart
    exception list that has to be hand-extended every time a new crowded
    crop shows up.

    Result sorted by value, largest first. Colored as a light->dark gradient
    of that task's own color ranked by actual value (see
    _shade_gradient_by_value). Shared by the nested donut's outer ring and
    the stacked bar's per-task segments — same cutoff, same shading, so the
    two chart types agree with each other."""
    _DOMINANCE_FRAC = 0.7
    _MIN_CATEGORY_TOTAL = 5  # categories smaller than this pool into "other"

    result: dict[str, list[tuple[str, int, tuple]]] = {}
    for t in class_task_types:
        group = groups_by_task[t]

        if category_fn is None:
            top = group.most_common(top_n)
            top_names = {name for name, _ in top}
            remainder_items = [(name, n) for name, n in group.items() if name not in top_names]
            entries = list(top)
            if remainder_items:
                entries.append((f"{len(remainder_items)} remaining {unit_name}",
                                 sum(n for _, n in remainder_items)))
        else:
            by_cat: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for name, n in group.items():
                by_cat[category_fn(name)].append((name, n))

            other_items: list[tuple[str, int]] = []
            for cat in [c for c, items in by_cat.items() if sum(n for _, n in items) < _MIN_CATEGORY_TOTAL]:
                other_items.extend(by_cat.pop(cat))
            if other_items:
                by_cat["other"] = other_items

            entries = []
            for cat, items in by_cat.items():
                total = sum(n for _, n in items)
                leader_name, leader_n = max(items, key=lambda kv: kv[1])
                if len(items) == 1 or leader_n / total >= _DOMINANCE_FRAC:
                    entries.append((leader_name, total))
                else:
                    noun = cat if "crop" in cat else f"{cat} crops"
                    entries.append((noun, total))

        entries.sort(key=lambda e: -e[1])
        shades = _shade_gradient_by_value(TASK_COLOR.get(t, "#888888"), [v for _, v in entries])
        result[t] = [(label, val, color) for (label, val), color in zip(entries, shades)]
    return result


def _render_task_nested_donut(
    chart_dir: Path,
    class_task_types: list[str],
    class_task_counts: Counter,
    groups_by_task: dict[str, Counter],
    *,
    top_n: int,
    unit_name: str,
    breakdown_key: str,
    breakdown_filename: str,
    chart_filename: str,
    category_fn=None,
) -> None:
    """Task type (inner ring, by class count) -> some second attribute
    (outer ring) — generic over whatever per-task Counter it's given
    (crop_by_task, location_by_task, ...), so every nested-donut chart in
    this pipeline shares one implementation instead of each being its own
    copy. `unit_name` names what's being bucketed for the "N remaining
    {unit_name}" catch-all label (e.g. "crops", "countries"). `category_fn`
    splits that catch-all into per-category sub-wedges instead — see
    _task_outer_entries / _crop_category."""
    import matplotlib.pyplot as plt
    import numpy as np

    inner_labels = class_task_types
    inner_vals = [class_task_counts[t] for t in inner_labels]
    inner_colors = [TASK_COLOR.get(t, "#888888") for t in inner_labels]
    inner_callouts = [f"{_display_task(t)} ({class_task_counts[t]} classes)" for t in inner_labels]

    _write_breakdown_csv(chart_dir, breakdown_filename, breakdown_key,
                          class_task_types, groups_by_task, top_n, category_fn=category_fn)
    outer_entries = _task_outer_entries(class_task_types, groups_by_task, top_n, unit_name,
                                         category_fn=category_fn)

    outer_vals, outer_colors, outer_labels = [], [], []
    for t in inner_labels:
        for label, val, color in outer_entries[t]:
            outer_vals.append(val)
            outer_colors.append(color)
            outer_labels.append(f"{label} ({val})")

    fig, ax = plt.subplots(figsize=(13, 10))
    ring_w = 0.46  # thick enough that even the smallest task's two-line label fits inside it
    inner_wedges, _ = ax.pie(
        inner_vals, radius=1 - ring_w, colors=inner_colors,
        wedgeprops=dict(width=ring_w, edgecolor="white", linewidth=1.5))
    outer_wedges, _, _ = ax.pie(
        outer_vals, radius=1, colors=outer_colors,
        wedgeprops=dict(width=ring_w, edgecolor="white", linewidth=1),
        autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", pctdistance=1 - ring_w / 2 + 0.03)
    ax.set(aspect="equal")

    # Task name, class count, and percentage as ONE compact text block (3
    # short lines) rather than two separately-positioned elements — keeps
    # them visually grouped together and lets the whole block sit closer to
    # the ring's inner edge (nearer the hole/center), where matplotlib
    # doesn't need to fan two elements across the wedge's full radial span
    # to keep them apart. Contained within even the thinnest wedge.
    ring_inner_r = 1 - 2 * ring_w
    ring_outer_r = 1 - ring_w
    total_inner = sum(inner_vals)
    for wedge, val, label in zip(inner_wedges, inner_vals, inner_callouts):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        rad = np.deg2rad(ang)
        pct = 100 * val / total_inner
        # Small wedges sit further out toward the ring's outer edge, where
        # the arc is wider — near the hole all wedges converge tightly, so a
        # smaller wedge benefits more from the extra room a larger radius
        # gives than a big wedge does.
        frac = 0.58 if pct < 15 else 0.38
        label_r = ring_inner_r + (ring_outer_r - ring_inner_r) * frac
        name, count = label.split(" (")
        ax.text(np.cos(rad) * label_r, np.sin(rad) * label_r,
                 f"{name}\n({count}\n{pct:.0f}%",
                 ha="center", va="center", fontsize=9, linespacing=1.3)

    _add_leader_labels(ax, outer_wedges, outer_labels, radius=1,
                        xoff=1.25, yoff=1.15, fontsize=9)

    plt.tight_layout()
    path = chart_dir / chart_filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Fixed, distinct-from-TASK_COLOR palette so continent color never gets
# confused with the task-type color already used elsewhere in these charts.
CONTINENT_COLOR = {
    "asia": "#8172B2",
    "africa": "#CCB974",
    "europe": "#64B5CD",
    "north_america": "#937860",
    "south_america": "#DA8BC3",
    "oceania": "#8C8C8C",
    "worldwide": "#4D4D4D",
}


def _render_task_continent_grouped_bar(
    chart_dir: Path,
    class_task_types: list[str],
    continent_by_task: dict[str, Counter],
    *,
    breakdown_filename: str,
    chart_filename: str,
    ylabel: str,
) -> None:
    """One bar per (task, continent) pair, grouped side-by-side within each
    task — not stacked, and not broken down to country level on-chart (that
    level of granularity overlapped/crowded once there were ~9 segments in
    small task bars like pest/damage). Continent is naturally few enough
    (~7) to read directly off a shared legend instead."""
    import csv as _csv
    import matplotlib.pyplot as plt
    import numpy as np

    # Full country-level detail (which countries make up each continent, per
    # task) stays available in the CSV even though this chart only shows
    # continent totals.
    with open(chart_dir / breakdown_filename, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["task_type", "continent", "n_classes"])
        for t in class_task_types:
            for continent, n in continent_by_task[t].most_common():
                writer.writerow([t, continent, n])
    print(f"  Saved: {chart_dir / breakdown_filename}")

    # Consistent continent order across every task cluster — by total count
    # across all tasks, descending, so the dominant continent (Asia) always
    # lands in the same relative position.
    totals = Counter()
    for t in class_task_types:
        totals.update(continent_by_task[t])
    continents = [c for c, _ in totals.most_common()]

    display_names = [_display_task(t) for t in class_task_types]
    n_bars = len(continents)
    width = 0.8 / n_bars
    x = np.arange(len(class_task_types))

    fig, ax = plt.subplots(figsize=(max(8, len(class_task_types) * 2.4), 6.5))
    for i, continent in enumerate(continents):
        vals = [continent_by_task[t].get(continent, 0) for t in class_task_types]
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=continent.replace("_", " "),
               color=CONTINENT_COLOR.get(continent, "#888888"),
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_bars, fontsize=9, frameon=False)
    plt.tight_layout()

    path = chart_dir / chart_filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def make_charts(datasets: list[DatasetConfig], manifest: dict, chart_dir: Path) -> None:
    """Summary figures for a download_and_sample run: how many datasets/classes/images
    ended up in the selection, and how it breaks down by task type."""
    import matplotlib.pyplot as plt

    chart_dir.mkdir(parents=True, exist_ok=True)
    ds_by_name = {d.agml_name: d for d in datasets}
    task_type_by_name = {d.agml_name: d.task_type for d in datasets}
    ds_entries = manifest["datasets"]

    # NOT manifest.get("failed", {}) alone — that dict can only ever contain
    # datasets that raised a catchable Python exception *inside* this
    # process. A dataset killed by an external `timeout` wrapper (or any
    # other SIGTERM/SIGKILL) never gets a chance to run its own except
    # block, so it's invisible to manifest["failed"] even though it very
    # much did not succeed. Comparing the full active dataset list (passed
    # in as `datasets`) against what's actually in ds_entries catches every
    # missing dataset regardless of why it's missing — timeout, exception,
    # or simply not attempted yet in a partial run.
    explicit_errors = manifest.get("failed", {})
    missing_names = sorted(d.agml_name for d in datasets if d.agml_name not in ds_entries)
    failed = {
        name: explicit_errors.get(name, "no error captured — likely killed by an external "
                                        "timeout/signal before it could raise a Python exception")
        for name in missing_names
    }

    if not ds_entries:
        print("  No datasets in manifest — skipping charts.")
        return

    # ── 1. Datasets per task type ───────────────────────────────────────────
    task_counts = Counter(task_type_by_name.get(name, "unknown") for name in ds_entries)
    task_types = sorted(task_counts, key=lambda t: -task_counts[t])
    colors = [TASK_COLOR.get(t, "#888888") for t in task_types]

    fig, ax = plt.subplots(figsize=(max(6, len(task_types) * 1.5), 5))
    bars = ax.bar(task_types, [task_counts[t] for t in task_types],
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, task_types):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(task_counts[t]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of datasets")
    title = f"Datasets by task type (n={len(ds_entries)} datasets)"
    if failed:
        title += f"  —  {len(failed)} datasets FAILED, not shown (see summary.txt)"
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "datasets_per_task_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 1b. Classes per EFFECTIVE task type (class_overrides applied) ───────
    # Same idea as chart 1, but counting individual classes rather than whole
    # datasets — a pest/damage class nested inside an otherwise-"disease"
    # dataset counts under pest_damage here, via
    # DatasetConfig.effective_task_type() (src/common/dataset_config.py).
    # Only successfully-downloaded classes counted, same as every other chart.
    class_task_counts = Counter()
    for name, entry in ds_entries.items():
        cfg = ds_by_name.get(name)
        for cls in entry["classes"]:
            eff = cfg.effective_task_type(cls) if cfg else task_type_by_name.get(name, "unknown")
            class_task_counts[eff] += 1

    class_task_types = sorted(class_task_counts, key=lambda t: -class_task_counts[t])
    class_colors = [TASK_COLOR.get(t, "#888888") for t in class_task_types]

    fig, ax = plt.subplots(figsize=(max(6, len(class_task_types) * 1.5), 5))
    bars = ax.bar(class_task_types, [class_task_counts[t] for t in class_task_types],
                   color=class_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, class_task_types):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(class_task_counts[t]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of classes")
    title = f"Classes by effective task type, overrides applied (n={sum(class_task_counts.values())} classes)"
    if failed:
        title += f"  —  {len(failed)} datasets FAILED, not shown"
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "classes_per_task_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Classes per dataset (sorted, long-tail style) ────────────────────
    n_classes = {name: len(entry["classes"]) for name, entry in ds_entries.items()}
    names = sorted(n_classes, key=lambda n: -n_classes[n])
    counts = [n_classes[n] for n in names]
    colors2 = [TASK_COLOR.get(task_type_by_name.get(n, "unknown"), "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 5))
    ax.bar(range(len(names)), counts, color=colors2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(names)))
    fontsize = max(4, 8 - len(names) // 20)
    ax.set_xticklabels(names, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylabel("Number of classes")
    ax.set_title("Classes per dataset")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TASK_COLOR.values()]
    ax.legend(handles, TASK_COLOR.keys(), fontsize=8, loc="upper right")
    plt.tight_layout()
    path = chart_dir / "classes_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. Images selected per dataset (sorted, long-tail style) ────────────
    n_images = {name: sum(len(v) for v in entry["classes"].values()) for name, entry in ds_entries.items()}
    names_by_img = sorted(n_images, key=lambda n: -n_images[n])
    img_counts = [n_images[n] for n in names_by_img]
    colors3 = [TASK_COLOR.get(task_type_by_name.get(n, "unknown"), "#888888") for n in names_by_img]

    fig, ax = plt.subplots(figsize=(max(14, len(names_by_img) * 0.35), 5))
    ax.bar(range(len(names_by_img)), img_counts, color=colors3, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(names_by_img)))
    fontsize = max(4, 8 - len(names_by_img) // 20)
    ax.set_xticklabels(names_by_img, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylabel("Images selected")
    ax.set_title(f"Images selected per dataset (n_per_class cap = {manifest.get('n_per_class')})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TASK_COLOR.values()]
    ax.legend(handles, TASK_COLOR.keys(), fontsize=8, loc="upper right")
    plt.tight_layout()
    path = chart_dir / "images_selected_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 4. Distribution of images-per-class (shows how many classes hit the cap) ──
    per_class_counts = [len(idxs) for entry in ds_entries.values() for idxs in entry["classes"].values()]
    cap = manifest.get("n_per_class")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(per_class_counts, bins=min(30, max(cap, 1) if cap else 30),
            color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    if cap:
        ax.axvline(cap, color="#C44E52", linestyle="--", linewidth=1.5, label=f"cap = {cap}")
        ax.legend()
    ax.set_xlabel("Images selected per class")
    ax.set_ylabel("Number of classes")
    n_at_cap = sum(1 for c in per_class_counts if cap and c >= cap)
    ax.set_title(f"Images-per-class distribution ({n_at_cap}/{len(per_class_counts)} classes at cap)")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "images_per_class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 5. Task-type split: datasets vs classes (after class_overrides) ─────
    # Two separate pies (no titles, no bottom caption) rather than one
    # side-by-side figure — easier to drop individually into a paper/slide
    # without cropping out the other one's chart. class_task_counts computed
    # once, up in chart 1b, and reused here.
    for counts, fname in (
        (task_counts, "datasets_by_task_type_pie.png"),
        (class_task_counts, "classes_by_task_type_pie.png"),
    ):
        types = sorted(counts, key=lambda t: -counts[t])
        vals = [counts[t] for t in types]
        cols = [TASK_COLOR.get(t, "#888888") for t in types]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.pie(vals, labels=types, colors=cols, autopct=lambda p: f"{p:.0f}%\n({round(p * sum(vals) / 100)})",
               startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.5})
        plt.tight_layout()
        path = chart_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 6. Nested donut(s): task type (inner ring) -> crop / location (outer
    # ring) ──────────────────────────────────────────────────────────────
    # More granular version of chart 5's classes_by_task_type_pie.png — same
    # inner ring (task type, by class count), plus an outer ring breaking
    # each task down by some second attribute (crop, or separately location),
    # shaded as a light->dark gradient of that task's own color so the outer
    # ring visually "belongs" to its inner wedge. Both rings are labeled with
    # leader-line callouts (no legend, no in-wedge text) — every wedge gets
    # its label placed outside the ring regardless of how thin that wedge is,
    # so nothing overflows or collides with its own percentage. Only the
    # percentage number stays inside. The rendering itself
    # (_render_task_nested_donut, module-level below) is generic over
    # whatever Counter-per-task grouping it's given — crop and location are
    # two separate calls into the same function, not two copies of this code.

    # "mixed" (explicit in datasets.yaml) and unset/None crop both mean "no
    # single crop applies" — but WHY differs by task, confirmed by inspecting
    # every dataset behind each bucket: every species_identification dataset
    # with an ambiguous crop turned out to be a weed/invasive-species
    # identification set (ACHENY, MH_Weed16, WaterHyacinth,
    # maize_tomato_weed, crop_weeds_greece, plant_seedlings_aarhus,
    # rangeland_weeds_australia) where "crop" doesn't apply at all — not
    # missing metadata. For every other task type, the ambiguous ones
    # (AgriVision4, three_plant_leaf_disease, BPLD, MedLeafX) are genuinely
    # multi-crop-spanning or crop-unidentifiable datasets. Datasets where the
    # specific crops ARE known (e.g. banana_guava_quality_classification)
    # have their crop written as an explicit list in datasets.yaml instead of
    # "mixed" — see DatasetConfig.crop / crop_display() — so they're
    # unaffected by this bucket and appear under their own real crop name(s).
    _AMBIGUOUS_CROP_LABEL = {"species_identification": "weed / non-crop species"}
    _AMBIGUOUS_CROP_DEFAULT = "multi-crop / unspecified"

    TOP_N_CROPS = 8
    crop_by_task: dict[str, Counter] = defaultdict(Counter)
    for name, entry in ds_entries.items():
        cfg = ds_by_name.get(name)
        if cfg is None:
            continue
        crop_val = cfg.crop
        for cls in entry["classes"]:
            eff = cfg.effective_task_type(cls) if cfg else task_type_by_name.get(name, "unknown")
            if isinstance(crop_val, list):
                crop_label = " / ".join(crop_val)
            elif not crop_val or crop_val == "mixed":
                crop_label = _AMBIGUOUS_CROP_LABEL.get(eff, _AMBIGUOUS_CROP_DEFAULT)
            else:
                crop_label = crop_val
            crop_by_task[eff][crop_label] += 1

    _render_task_nested_donut(
        chart_dir, class_task_types, class_task_counts, crop_by_task,
        top_n=TOP_N_CROPS, unit_name="crops", breakdown_key="crop",
        breakdown_filename="classes_by_task_type_crop_breakdown.csv",
        chart_filename="classes_by_task_type_crop_nested_pie.png",
        category_fn=_crop_category,  # bin every crop into CROP_CATEGORY — a
                                      # crop keeps its own name only if it
                                      # dominates its category (e.g.
                                      # medicinal_plant, rice); otherwise the
                                      # whole category is one combined wedge
                                      # (e.g. "fruit crops"). See
                                      # _task_outer_entries.
    )

    # ── 6b. Same nested donut, outer ring by LOCATION (country) instead of
    # crop — a dataset-level attribute (every class in a dataset shares that
    # dataset's location), unlike crop which can vary within a multi-crop
    # dataset. location: [continent, country] per DatasetConfig — see
    # datasets.yaml; country is location[1] (a " / "-joined string for the
    # one dataset that genuinely spans multiple countries, same convention
    # as multi-crop labels). Datasets missing location entirely (shouldn't
    # happen for the 121 currently populated, but the pipeline shouldn't
    # crash if a future dataset is added without one) fall into "unspecified".
    location_by_task: dict[str, Counter] = defaultdict(Counter)
    continent_by_task: dict[str, Counter] = defaultdict(Counter)
    for name, entry in ds_entries.items():
        cfg = ds_by_name.get(name)
        if cfg is None:
            continue
        if cfg.location and len(cfg.location) > 1:
            continent, country = cfg.location[0], cfg.location[1]
        else:
            continent, country = "unspecified", "unspecified"
        for cls in entry["classes"]:
            eff = cfg.effective_task_type(cls) if cfg else task_type_by_name.get(name, "unknown")
            location_by_task[eff][country] += 1
            continent_by_task[eff][continent] += 1

    _render_task_nested_donut(
        chart_dir, class_task_types, class_task_counts, location_by_task,
        top_n=8, unit_name="countries", breakdown_key="country",
        breakdown_filename="classes_by_task_type_location_breakdown.csv",
        chart_filename="classes_by_task_type_location_nested_pie.png",
    )

    # ── 6c. Bar-chart companion to 6b — grouped (not stacked) by continent
    # rather than country, so it pairs with the pie chart in the paper
    # without repeating the pie's country-level crowding in bar form.
    _render_task_continent_grouped_bar(
        chart_dir, class_task_types, continent_by_task,
        breakdown_filename="classes_by_task_type_continent_breakdown.csv",
        chart_filename="classes_by_task_type_continent_grouped_bar.png",
        ylabel="Number of classes",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    total_classes = sum(n_classes.values())
    total_images = sum(n_images.values())
    summary = (
        f"Datasets: {len(ds_entries)} succeeded, {len(failed)} failed\n"
        f"Classes:  {total_classes}\n"
        f"Images:   {total_images:,}\n"
        f"n_per_class cap: {cap}\n"
        f"By task type (datasets): {dict(task_counts)}\n"
        f"By task type (classes, overrides applied): {dict(class_task_counts)}\n"
    )
    if failed:
        summary += f"\nFailed datasets ({len(failed)}) — re-run with --skip_existing to retry:\n"
        for name in sorted(failed):
            err = failed[name]
            msg = err.get("error", err) if isinstance(err, dict) else err  # dict since the
                                                                            # traceback-capture
                                                                            # change; str for
                                                                            # any older manifest
            summary += f"  - {name}: {msg}\n"
    (chart_dir / "summary.txt").write_text(summary)
    print(f"  Saved: {chart_dir / 'summary.txt'}")
    print(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/datasets.yaml")
    ap.add_argument("--catalog", default="src/config/hf_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="dataset/selected_images.json")
    ap.add_argument("--include_skipped", action="store_true")
    ap.add_argument("--only", default=None,
                     help="optional regex filter on agml_name — run against a single dataset "
                          "(or a subset) instead of the full active list. Same convention as "
                          "catalog_datasets.py's --only. Combine with --skip_existing so a "
                          "one-dataset-at-a-time wrapper script (e.g. under an OS-level `timeout`, "
                          "for hangs that never raise a catchable exception) still accumulates "
                          "into the same manifest across separate invocations.")
    ap.add_argument("--from_manifest", default=None,
                     help="reproduce indices from an existing manifest instead of re-sampling")
    ap.add_argument("--skip_existing", action="store_true",
                     help="resume: reuse datasets already recorded (successfully) in --output "
                          "instead of re-downloading/re-sampling them; only process what's "
                          "missing or previously failed. Ignored (with a warning) if --n_per_class "
                          "doesn't match the existing manifest's recorded value.")
    ap.add_argument("--chart_dir", default="outputs/agml_hf/",
                     help="where to write summary figures (datasets/classes/images counts)")
    ap.add_argument("--no_charts", action="store_true", help="skip chart generation")
    ap.add_argument("--plot_only", action="store_true",
                     help="skip download/sampling entirely; just regenerate charts from the "
                          "existing manifest at --output")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}

    if args.include_skipped:
        from src.common.dataset_config import load_datasets
        datasets = load_datasets(Path(args.config))
    else:
        datasets = active_datasets(Path(args.config))

    if args.only:
        pattern = re.compile(args.only)
        datasets = [d for d in datasets if pattern.search(d.agml_name)]
        if not datasets:
            print(f"ERROR: --only {args.only!r} matched no active datasets.")
            sys.exit(1)

    if args.plot_only:
        out_path = Path(args.output)
        if not out_path.exists():
            print(f"ERROR: --plot_only given but {out_path} does not exist.")
            sys.exit(1)
        manifest = json.loads(out_path.read_text())
        make_charts(datasets, manifest, Path(args.chart_dir))
        return

    prior_manifest = json.loads(Path(args.from_manifest).read_text()) if args.from_manifest else None

    existing_manifest = None
    out_path = Path(args.output)
    if args.skip_existing and out_path.exists():
        existing_manifest = json.loads(out_path.read_text())
        if existing_manifest.get("n_per_class") != args.n_per_class:
            print(f"WARNING: --skip_existing requested but existing manifest was built with "
                  f"n_per_class={existing_manifest.get('n_per_class')} != requested "
                  f"{args.n_per_class} — every dataset's sample would be stale under the new "
                  f"cap, so ignoring --skip_existing and reprocessing everything.")
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
                process_dataset, d.agml_name, d.hf_config, catalog, args.n_per_class, args.seed, fixed
            )
            manifest["datasets"][d.agml_name] = {
                "hf_repo_id": repo_id,
                "hf_config": config,
                "classes": selected,
            }
            manifest["failed"].pop(d.agml_name, None)
            n_total = sum(len(v) for v in selected.values())
            print(f"ok ({len(selected)} classes, {n_total} images)")
            n_ok += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)  # full traceback in the job log, not just str(e) — a bug like
                       # "bad operand type for abs(): 'str'" is unlocatable from the
                       # message alone; str/dict-shape kept backward compatible below,
                       # existing readers only ever do sorted(manifest["failed"]) (keys)
            manifest["failed"][d.agml_name] = {"error": str(e), "traceback": tb}
            n_err += 1

        if i % 5 == 0 or i == len(datasets):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    elapsed = time.time() - t0
    total_images = sum(
        len(idxs) for ds in manifest["datasets"].values() for idxs in ds["classes"].values()
    )
    print(f"\nDone in {elapsed:.0f}s. {n_ok} datasets ok, {n_skipped} skipped (already done), "
          f"{n_err} errors -> {args.output}")
    if manifest["failed"]:
        print(f"Failed datasets (re-run with --skip_existing to retry just these): "
              f"{sorted(manifest['failed'])}")
    print(f"Total images selected: {total_images:,}")

    if not args.no_charts:
        print(f"\n=== Generating summary charts -> {args.chart_dir} ===")
        make_charts(datasets, manifest, Path(args.chart_dir))


if __name__ == "__main__":
    main()
