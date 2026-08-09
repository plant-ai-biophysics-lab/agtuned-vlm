"""Shared per-sample checkpoint + collation helpers for vision_baseline.py /
desc_augmented_vision_v2.py — two-condition (lower/upper bound) vision
classification eval with a per-model output subfolder and cross-model
collation, so a limited-time remote job can resume exactly where it left off.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.common.metrics import aggregate_rouge, compute_classification_metrics

CKPT_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "prediction", "resolved_class", "ground_truth",
    "correct", "contains_match", "resolvable",
    "rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim",
]

_SUMMARY_FIELDS = ["f1_macro", "accuracy", "contains_accuracy", "rouge_ss", "sem_sim"]


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def load_ckpt(path: Path) -> tuple[list[dict], set[tuple[str, str, str]]]:
    """Load an existing checkpoint CSV. Returns (rows, seen-keys) so a resumed
    run can skip (dataset, class_label, image_name) triples already scored."""
    if not path.exists():
        return [], set()
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {(r["agml_dataset"], r["class_label"], r["image_name"]) for r in rows}
    return rows, seen


def append_ckpt_row(path: Path, row: dict) -> None:
    """Flush one scored sample to the checkpoint CSV immediately — a kill
    mid-run loses at most the in-flight batch, not the whole model's progress."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CKPT_FIELDS})


def compute_model_metrics(rows: list[dict]) -> dict:
    """Per-dataset, per-task-type, and overall F1 + ROUGE for one model's rows."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["agml_dataset"]].append(r)
        by_task[r["task_type"]].append(r)

    def _summarize(group_rows: list[dict]) -> dict:
        refs = [r["ground_truth"] for r in group_rows]
        preds = [r.get("resolved_class") or "" for r in group_rows]
        m = compute_classification_metrics(refs, preds)
        m.update(aggregate_rouge(group_rows))
        return m

    return {
        "overall":    _summarize(rows),
        "by_dataset": {ds: _summarize(rs) for ds, rs in by_ds.items()},
        "by_task":    {tt: _summarize(rs) for tt, rs in by_task.items()},
    }


def write_model_metrics(model_dir: Path, metrics: dict) -> None:
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def collate_models(output_dir: Path) -> dict[str, dict]:
    """Scan output_dir/<model>/ckpt_*.csv (one subfolder per model already
    run), recompute metrics per model, return {model_dir_name: metrics}."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_files = list(model_dir.glob("ckpt_*.csv"))
        if not ckpt_files:
            continue
        rows = list(csv.DictReader(open(ckpt_files[0], encoding="utf-8")))
        if not rows:
            continue
        result[model_dir.name] = compute_model_metrics(rows)
    return result


def write_collated(output_dir: Path, collated: dict[str, dict]) -> None:
    (output_dir / "collated_metrics.json").write_text(json.dumps(collated, indent=2))

    fieldnames = ["model", "scope", "key", "n_total", *_SUMMARY_FIELDS]
    with open(output_dir / "collated_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, metrics in collated.items():
            def _row(scope: str, key: str, m: dict) -> dict:
                row = {"model": model, "scope": scope, "key": key, "n_total": m.get("n_total")}
                row.update({k: m.get(k) for k in _SUMMARY_FIELDS})
                return row
            writer.writerow(_row("overall", "", metrics["overall"]))
            for tt, tm in metrics["by_task"].items():
                writer.writerow(_row("task", tt, tm))
            for ds, dm in metrics["by_dataset"].items():
                writer.writerow(_row("dataset", ds, dm))
    print(f"  Saved: {output_dir / 'collated_summary.csv'}")
    print(f"  Saved: {output_dir / 'collated_metrics.json'}")
