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
    crop: Optional[str] = None
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
        ))
    return result


def active_datasets(config_path: Path) -> list[DatasetConfig]:
    return [d for d in load_datasets(config_path) if not d.skip]


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


def parse_class_label(raw_label: str, dataset_crop: Optional[str]) -> tuple[str, str]:
    """Returns (crop, clean_class_label). Splits on ___ for multi-crop datasets."""
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        return parts[0].replace("_", " ").strip(), parts[1].replace("_", " ").strip()
    crop = dataset_crop if dataset_crop and dataset_crop not in _AMBIGUOUS_CROPS else "plant"
    return crop, raw_label.replace("_", " ").strip()


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


def load_hf_samples(agml_name: str, manifest: dict, images_root: Path) -> list[dict]:
    """Materialize the images a download_and_sample.py manifest selected for
    one dataset into local JPEG files, returning [{"path": str, "label": str}, ...]
    — the exact shape every embedding/inference consumer already expects.

    Caches to images_root/<agml_name>/<label>/<row_idx>.jpg: a file already
    on disk is reused, never re-fetched. The underlying HF dataset (a full
    `datasets.load_dataset` call) is only opened if at least one image for
    this dataset is actually missing — repeat runs after the first are pure
    local-disk reads, no network/HF Hub involvement at all.

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

    out_dir = Path(images_root) / agml_name
    samples: list[dict] = []
    ds = None
    image_col = None

    for label, indices in classes.items():
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
