"""Dataset registry — shared by all pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import yaml

DISEASE_TASK_TYPES = {"disease", "pest_damage"}
IDENTIFICATION_TASK_TYPES = {"weed_identification", "crop_identification"}


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


def parse_class_label(raw_label: str, dataset_crop: Optional[str]) -> tuple[str, str]:
    """Returns (crop, clean_class_label). Splits on ___ for multi-crop datasets."""
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        return parts[0].replace("_", " ").strip(), parts[1].replace("_", " ").strip()
    crop = dataset_crop if dataset_crop and dataset_crop != "multiple" else "plant"
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
