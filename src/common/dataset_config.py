"""Dataset registry — shared by all pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import yaml

DISEASE_TASK_TYPES = {"disease", "pest_damage"}
IDENTIFICATION_TASK_TYPES = {"species_identification", "quality_identification"}


@dataclass
class DatasetConfig:
    agml_name: str
    task_type: str
    classes: list[str]
    crop: Optional[str | list[str]] = None  # list when the dataset spans a small,
                                             # explicitly known set of crops (see
                                             # crop_display()); str otherwise
    n_images: int = 0
    location: list = None
    skip: bool = False
    skip_description: bool = False
    notes: str = ""
    class_overrides: dict = None  # {clean_class_label: task_type}
    class_aliases: dict = None  # {clean_class_label: [alt_name, ...]} — alternate
                                 # names (common/scientific/standard spelling) for
                                 # classes whose raw label is a poor search term
                                 # (regional/vernacular name, dataset-internal code,
                                 # etc.). Opt-in, empty by default; see
                                 # effective_search_terms().
    hf_config: Optional[str] = None  # Project-AgML HF dataset config/split-group, e.g. "raw" vs "augmented"
    excluded_classes: list = None  # [raw_class_label, ...] — classes to skip
                                    # entirely (never sampled by load_hf_samples,
                                    # never counted in downstream charts/metrics via
                                    # excluded_class_pairs()). Opt-in, empty by
                                    # default. Raw manifest-key form (matches this
                                    # dataset's own `classes:` list spelling exactly,
                                    # e.g. "BAS_370" not "BAS 370") — for classes
                                    # that are fundamentally too difficult for the
                                    # task at hand (e.g. bare cultivar-registry codes
                                    # like "BD70" with no visually groundable signal
                                    # to distinguish them from "BD72"), not a
                                    # generic quality filter.
    bbox_convention: Optional[dict] = None  # detection datasets only (Stage 5) —
                                             # {"scale": "pixel"|"normalized",
                                             # "format": "xyxy"|"xywh"}, measured by
                                             # catalog_detection_datasets.py. None
                                             # for classification datasets.

    @property
    def clean_classes(self) -> list[str]:
        return clean_class_list(self.classes)

    @property
    def is_disease_task(self) -> bool:
        return self.task_type in DISEASE_TASK_TYPES

    def effective_task_type(self, class_label: str) -> str:
        """Return the task_type for a specific class, respecting class_overrides."""
        if self.class_overrides:
            if class_label in self.class_overrides:
                return self.class_overrides[class_label]
        return self.task_type

    def classes_with_aliases_display(self) -> list[str]:
        """clean_classes, each optionally annotated with its class_aliases
        (if configured) — the detection analog of effective_search_terms,
        which resolves aliases for ONE class_label at a time (classification
        asks about a single class per image); detection's det_question
        lists every class the image might contain in a single combined
        question instead, so aliases need to be folded inline per class
        rather than returned as a separate priority-ordered list. A class
        with no configured aliases renders unchanged."""
        out = []
        for c in self.clean_classes:
            aliases = self.class_aliases.get(c) if self.class_aliases else None
            if isinstance(aliases, str):
                aliases = [aliases]
            if aliases:
                out.append(f"{c} (also known as: {', '.join(aliases)})")
            else:
                out.append(c)
        return out

    def effective_search_terms(self, class_label: str) -> list[str]:
        """class_label plus any curated alternate names for it, in priority
        order (class_label first, then aliases as configured). Aliases are
        opt-in per class via class_aliases in datasets.yaml — most classes
        have none and this just returns [class_label]."""
        terms = [class_label]
        if self.class_aliases and class_label in self.class_aliases:
            aliases = self.class_aliases[class_label]
            if isinstance(aliases, str):
                aliases = [aliases]
            terms.extend(a for a in aliases if a and a not in terms)
        return terms


def load_datasets(config_path: Path) -> list[DatasetConfig]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    result = []
    for d in raw["datasets"]:
        result.append(DatasetConfig(
            agml_name=d["agml_name"],
            crop=d.get("crop"),
            task_type=d["task_type"],
            n_images=d.get("n_images", 0),
            classes=d.get("classes", []),
            location=d.get("location", []),
            skip=d.get("skip", False),
            skip_description=d.get("skip_description", False),
            notes=d.get("notes", ""),
            class_overrides=d.get("class_overrides", {}),
            class_aliases=d.get("class_aliases", {}),
            hf_config=d.get("hf_config"),
            excluded_classes=d.get("excluded_classes", []),
            bbox_convention=d.get("bbox_convention"),
        ))
    return result


def active_datasets(config_path: Path) -> list[DatasetConfig]:
    return [d for d in load_datasets(config_path) if not d.skip]


def excluded_class_pairs(config_path: Path) -> set[tuple[str, str]]:
    """{(agml_name, cleaned_class_label), ...} across every dataset's
    excluded_classes — the CSV-row-filtering counterpart to
    load_hf_samples's excluded_classes param: checkpoint rows store the
    CLEANED class_label (clean_class_list — underscores -> spaces), not the
    raw manifest key excluded_classes is spelled in, so this applies that
    same cleaning before returning the lookup set. Used by
    src/common/vision_eval.py's load_ckpt to keep every existing
    chart/metric consumer (which reads through load_ckpt, directly or via
    load_all_rows/collate_models) automatically excluding these classes
    without each script needing its own filter."""
    pairs = set()
    for d in load_datasets(config_path):
        if not d.excluded_classes:
            continue
        for cleaned in clean_class_list(d.excluded_classes):
            pairs.add((d.agml_name, cleaned))
    return pairs


def datasets_by_name(config_path: Path) -> dict[str, DatasetConfig]:
    return {d.agml_name: d for d in load_datasets(config_path)}


def prefetch_agml_datasets(datasets: list[DatasetConfig]) -> None:
    """Trigger AgML downloads for all datasets upfront before any model loading."""
    import agml
    names = [d.agml_name for d in datasets]
    print(f"\n=== Prefetching {len(names)} AgML datasets ===")
    for i, name in enumerate(names, 1):
        print(f"  [{i}/{len(names)}] {name}", end=" ", flush=True)
        try:
            agml.data.AgMLDataLoader(name)
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")
    print()


def clean_class_list(raw_classes: list[str]) -> list[str]:
    cleaned = []
    for c in raw_classes:
        if "___" in c:
            cleaned.append(c.split("___", 1)[1].replace("_", " ").strip())
        else:
            cleaned.append(c.replace("_", " ").strip())
    return cleaned


_AMBIGUOUS_CROPS = {"multiple", "mixed"}  # not a real crop name — falls back to
                                           # the generic "plant" instead of leaking
                                           # the literal word into prompts like
                                           # "...in this description of a {crop}?"


def crop_display(crop: Optional[str | list[str]]) -> str:
    """Render a DatasetConfig.crop value for use in a prompt's {crop} slot.
    A list (dataset spans a small, explicitly known set of crops — see
    datasets.yaml) becomes "eggplant or tomato"; None/"multiple"/"mixed"
    (crop unknown or too broad to enumerate) falls back to generic "plant"."""
    if isinstance(crop, list):
        return " or ".join(c.replace("_", " ").strip() for c in crop)
    if crop and crop not in _AMBIGUOUS_CROPS:
        return crop.replace("_", " ").strip()
    return "plant"


def parse_class_label(raw_label: str, dataset_crop: Optional[str | list[str]]) -> tuple[str, str]:
    """Returns (crop, clean_class_label). Splits on ___ for multi-crop datasets."""
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        return parts[0].replace("_", " ").strip(), parts[1].replace("_", " ").strip()
    return crop_display(dataset_crop), raw_label.replace("_", " ").strip()


def subsample_class_balanced(
    items: list[Any],
    label_fn: Callable[[Any], str],
    n_per_class: int,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Group items by label_fn(item), cap each group at n_per_class via seeded,
    without-replacement sampling. Returns {label: [selected indices into items]}.

    Shared sampling primitive for "representative, class-balanced subset" use
    cases (e.g. scripts/agml_hf/download_and_sample.py, linear_probe/probe.py's
    per-class train cap) — a fixed cap per class rather than proportional/scaled,
    since source-dataset class imbalance reflects collection effort, not
    real-world prevalence.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        by_class.setdefault(label_fn(item), []).append(i)
    selected: dict[str, list[int]] = {}
    for label, indices in by_class.items():
        if len(indices) > n_per_class:
            chosen = rng.choice(len(indices), size=n_per_class, replace=False)
            indices = sorted(indices[i] for i in chosen)
        selected[label] = indices
    return selected


def prefetch_hf_datasets(datasets: list[DatasetConfig], hf_catalog_path: Path) -> None:
    """Trigger Project-AgML HuggingFace dataset downloads for all datasets upfront
    (HF-sourced analog of prefetch_agml_datasets, additive — does not replace it)."""
    import json
    from datasets import load_dataset_builder

    catalog = json.loads(Path(hf_catalog_path).read_text())
    print(f"\n=== Prefetching {len(datasets)} HuggingFace (Project-AgML) datasets ===")
    for i, d in enumerate(datasets, 1):
        repo_id = f"Project-AgML/{d.agml_name}"
        config = d.hf_config or catalog.get(d.agml_name, {}).get("preferred_config")
        print(f"  [{i}/{len(datasets)}] {repo_id} (config={config})", end=" ", flush=True)
        try:
            builder = load_dataset_builder(repo_id, config)
            builder.download_and_prepare()
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")
    print()


# ── Shared HF-manifest-based sample loading ─────────────────────────────────
#
# The single loader every downstream stage (linear_probe, cot_vision,
# vision_assessment, ...) should use to actually materialize images once
# scripts/agml_hf/download_and_sample.py has produced a manifest — one
# implementation instead of each stage separately reimplementing "look up
# this dataset's HF repo/config, open it, index into it by row, cache to
# disk." Returns the same [{"path": str, "label": str}, ...] shape the old
# agml-pip-package-based loaders used, so it's a drop-in replacement at
# every call site that previously called an agml-based sample loader.

def load_manifest(manifest_path: Path) -> dict:
    """Load a scripts/agml_hf/download_and_sample.py manifest
    (dataset/selected_images.json) once, to be passed into load_hf_samples()
    per-dataset rather than re-read from disk every call."""
    import json
    return json.loads(Path(manifest_path).read_text())


def missing_from_manifest(datasets: list[DatasetConfig], manifest: dict) -> list[str]:
    """Active dataset names with no successful entry in the manifest — same
    diff-based accounting scripts/agml_hf/download_and_sample.py's
    make_charts() uses (catches anything not downloaded yet, regardless of
    whether it failed with a caught exception or was silently killed by an
    external timeout — see that function's comment for why the diff, not
    manifest["failed"] alone, is the reliable source of truth)."""
    done = set(manifest.get("datasets", {}).keys())
    return [d.agml_name for d in datasets if d.agml_name not in done]


def find_image_column(features) -> str:
    """Image-feature analog of scripts/agml_hf/download_and_sample.py's
    find_label_column — needed here because load_hf_samples() opens the raw
    HF dataset to materialize pixels for a manifest-selected row index."""
    from datasets import Image as HFImage
    for key, feat in features.items():
        if isinstance(feat, HFImage):
            return key
    for key in ("image", "img", "picture", "photo"):
        if key in features:
            return key
    raise KeyError(f"No recognizable image column in features: {list(features)}")


_UNSAFE_LABEL_CHARS = None  # lazily-compiled regex, see _safe_label_dir


def _safe_label_dir(label: str) -> str:
    """Sanitize a class label for use as a filesystem directory name — labels
    can contain spaces, slashes, or other characters that are valid class
    names but not valid path components."""
    global _UNSAFE_LABEL_CHARS
    if _UNSAFE_LABEL_CHARS is None:
        import re
        _UNSAFE_LABEL_CHARS = re.compile(r"[^a-zA-Z0-9_.-]+")
    return _UNSAFE_LABEL_CHARS.sub("_", label).strip("_") or "unlabeled"


def load_hf_samples(
    agml_name: str, manifest: dict, images_root: Path, excluded_classes: list[str] | None = None,
) -> list[dict]:
    """Materialize the images a download_and_sample.py manifest selected for
    one dataset into local JPEG files, returning [{"path": str, "label": str}, ...]
    — the exact shape every embedding/inference consumer already expects.

    Caches to images_root/<agml_name>/<label>/<row_idx>.jpg: a file already
    on disk is reused, never re-fetched. The underlying HF dataset (a full
    `datasets.load_dataset` call) is only opened if at least one image for
    this dataset is actually missing — repeat runs after the first are pure
    local-disk reads, no network/HF Hub involvement at all.

    `excluded_classes` (raw manifest-key form, e.g. DatasetConfig.excluded_classes)
    skips those labels entirely — never materialized, never sampled, never
    evaluated. This is the future-run-facing half of class exclusion; the
    already-checkpointed-row-facing half is excluded_class_pairs() +
    load_ckpt() in src/common/vision_eval.py.

    Raises KeyError if `agml_name` has no entry in manifest["datasets"] —
    i.e. it was never downloaded, or download failed/was skipped. Callers
    iterating multiple datasets should catch this per-dataset (same pattern
    the old agml-based loaders' call sites already used for a missing/broken
    dataset) rather than let one bad dataset abort an entire run.
    """
    entry = manifest.get("datasets", {}).get(agml_name)
    if entry is None:
        raise KeyError(
            f"{agml_name!r} not found in manifest's \"datasets\" — it was never "
            f"successfully downloaded (see summary.txt, or the manifest's "
            f"\"failed\" key, for why)."
        )

    repo_id = entry["hf_repo_id"]
    config = entry["hf_config"]
    classes: dict[str, list[int]] = entry["classes"]
    excluded = set(excluded_classes or [])

    out_dir = Path(images_root) / agml_name
    samples: list[dict] = []
    ds = None
    image_col = None

    for label, indices in classes.items():
        if label in excluded:
            continue
        label_dir = out_dir / _safe_label_dir(label)
        for idx in indices:
            path = label_dir / f"{idx}.jpg"
            if not path.exists():
                if ds is None:
                    from datasets import load_dataset
                    ds = load_dataset(repo_id, config, split="train")
                    image_col = find_image_column(ds.features)
                label_dir.mkdir(parents=True, exist_ok=True)
                ds[idx][image_col].convert("RGB").save(path, format="JPEG")
            samples.append({"path": str(path), "label": label})

    return samples


def load_hf_detection_samples(
    agml_name: str, manifest: dict, images_root: Path, class_names: list[str] | None = None,
) -> list[dict]:
    """Detection analog of load_hf_samples() — NOT a drop-in replacement,
    since a detection row can carry multiple boxes/categories and the
    classification return shape ({"path", "label"}) has no room for that.
    Returns [{"path": str, "boxes": [{"bbox": [...], "category_id": int,
    "category_name": str}, ...]}, ...].

    Caches to images_root/<agml_name>/<row_idx>.jpg (no per-label
    subdirectory, unlike load_hf_samples — a detection image can belong to
    several categories at once, so there's no single label to shard by)
    plus a <row_idx>.json sidecar holding that row's full annotation list,
    so a second run is a pure local-disk read (image + sidecar both
    present) with no HF Hub/network involvement, same resumability
    guarantee as load_hf_samples.

    manifest["datasets"][agml_name]["classes"] maps PRIMARY category name
    (the balancing key download_and_sample_detection.py sampled on) -> row
    indices — used only to enumerate which rows to materialize; the actual
    per-row annotation written to the sidecar includes EVERY box/category
    in that row via the "objects" field, not just the primary one.

    class_names: category_id -> name lookup (from detection_datasets.yaml's
    `classes` list, i.e. catalog_detection_datasets.py's resolved names).
    Falls back to the bare integer id (as a string) if not given or if an
    id falls outside the list — better a visible "category_7" in a QC plot
    than a KeyError aborting the whole run.

    Raises KeyError if `agml_name` has no manifest entry — same contract
    and same per-dataset-catch-and-continue expectation as load_hf_samples.
    """
    import json as _json

    entry = manifest.get("datasets", {}).get(agml_name)
    if entry is None:
        raise KeyError(
            f"{agml_name!r} not found in manifest's \"datasets\" — it was never "
            f"successfully downloaded (see summary.txt, or the manifest's "
            f"\"failed\" key, for why)."
        )

    repo_id = entry["hf_repo_id"]
    config = entry["hf_config"]
    classes: dict[str, list[int]] = entry["classes"]
    names = class_names if class_names is not None else entry.get("class_names")
    bbox_convention = entry.get("bbox_convention") or {}
    box_scale = bbox_convention.get("scale", "pixel")
    box_format = bbox_convention.get("format", "xywh")

    def _category_name(cat_id: int) -> str:
        if names and 0 <= cat_id < len(names):
            return names[cat_id]
        return f"category_{cat_id}"

    def _to_pixel_xyxy(bbox: list[float], img_w: int, img_h: int) -> list[float]:
        """Normalize this dataset's raw box into the pixel-space [x1,y1,x2,y2]
        convention every other Stage 5 module (bbox_parse.py, bbox_metrics.py,
        the detection prompts) assumes — catalog_detection_datasets.py MEASURED
        (not assumed) each dataset's actual scale/format, confirmed via a real
        64-dataset run to vary per-dataset (57 pixel / 7 normalized; 63 xywh /
        1 xyxy). Without this conversion, a normalized [0,1]-range box would
        get clamped down to a ~1x1 pixel region by clamp_box() downstream —
        silently destroying it, not just scoring it wrong."""
        x, y, a, b = bbox
        if box_scale == "normalized":
            x, y, a, b = x * img_w, y * img_h, a * img_w, b * img_h
        if box_format == "xywh":
            return [x, y, x + a, y + b]
        return [x, y, a, b]  # already xyxy

    out_dir = Path(images_root) / agml_name
    samples: list[dict] = []
    ds = None
    image_col = None
    seen_idx: set[int] = set()

    for indices in classes.values():
        for idx in indices:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            img_path = out_dir / f"{idx}.jpg"
            ann_path = out_dir / f"{idx}.json"
            if img_path.exists() and ann_path.exists():
                boxes = _json.loads(ann_path.read_text())
            else:
                if ds is None:
                    from datasets import load_dataset
                    ds = load_dataset(repo_id, config, split="train")
                    image_col = find_image_column(ds.features)
                out_dir.mkdir(parents=True, exist_ok=True)
                row = ds[idx]
                image = row[image_col].convert("RGB")
                image.save(img_path, format="JPEG")
                objs = row.get("objects") or {}
                # Schema key inconsistency confirmed across real Project-AgML
                # detection repos (e.g. CocoaMFDB_detection uses "categories",
                # CBDA_Cotton_Ball_Detection_Augmented uses "category") — check
                # both rather than assuming either.
                cat_list = objs.get("category") if objs.get("category") is not None else objs.get("categories", [])
                boxes = [
                    {"bbox": _to_pixel_xyxy(list(bbox), image.width, image.height),
                     "category_id": int(cat_id), "category_name": _category_name(int(cat_id))}
                    for bbox, cat_id in zip(objs.get("bbox", []), cat_list)
                ]
                ann_path.write_text(_json.dumps(boxes))
            samples.append({"path": str(img_path), "boxes": boxes})

    return samples
