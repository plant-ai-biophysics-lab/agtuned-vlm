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
    notes: str = ""

    @property
    def clean_classes(self) -> list[str]:
        return clean_class_list(self.classes)

    @property
    def is_disease_task(self) -> bool:
        return self.task_type in DISEASE_TASK_TYPES


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
            notes=d.get("notes", ""),
        ))
    return result


def active_datasets(config_path: Path) -> list[DatasetConfig]:
    return [d for d in load_datasets(config_path) if not d.skip]


def datasets_by_name(config_path: Path) -> dict[str, DatasetConfig]:
    return {d.agml_name: d for d in load_datasets(config_path)}


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
