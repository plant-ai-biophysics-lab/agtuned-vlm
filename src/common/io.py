"""I/O helpers — shared by all pipeline stages."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path


class Tee:
    """Mirror print() to both stdout and a file."""

    def __init__(self, path: Path):
        self._file = open(path, "w", encoding="utf-8")

    def print(self, *args, **kwargs):
        import builtins
        builtins.print(*args, **kwargs)
        builtins.print(*args, **{**kwargs, "file": self._file, "flush": True})

    def close(self):
        self._file.close()


def load_jsonl(path: Path, n_samples: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[:n_samples] if n_samples else records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_prompts(prompts_path: Path) -> dict[str, str]:
    import yaml
    with open(prompts_path) as f:
        return yaml.safe_load(f)


def update_pipeline_results(
    csv_path: Path,
    model: str,
    model_type: str,
    stage: str,
    metrics: dict,
) -> None:
    """
    Upsert a model row in the pipeline results CSV.

    stage: one of "linear_probe", "language_assessment", "vision_assessment"
    metrics: dict with keys matching the stage's columns

    Column mapping:
      linear_probe        → linear_probe_f1
      language_assessment → lang_description_rouge_ss, lang_classification_f1
      vision_assessment   → vision_description_rouge_ss, vision_classification_f1
    """
    COLUMNS = [
        "model_types", "model",
        "linear_probe_f1",
        "lang_description_rouge_ss", "lang_classification_f1",
        "vision_description_rouge_ss", "vision_classification_f1",
    ]

    rows: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["model"]] = row

    if model not in rows:
        rows[model] = {c: "" for c in COLUMNS}
        rows[model]["model"] = model
        rows[model]["model_types"] = model_type

    stage_map = {
        "linear_probe": {"linear_probe_f1": metrics.get("f1_macro", "")},
        "language_assessment": {
            "lang_description_rouge_ss": metrics.get("description_rouge_ss", ""),
            "lang_classification_f1": metrics.get("classification_f1_macro", ""),
        },
        "vision_assessment": {
            "vision_description_rouge_ss": metrics.get("description_rouge_ss", ""),
            "vision_classification_f1": metrics.get("classification_f1_macro", ""),
        },
    }
    rows[model].update(stage_map.get(stage, {}))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
