"""Dataset registry — shared by all pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
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
