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

import numpy as np
from sklearn.metrics import f1_score

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


# ── Cross-script (baseline + desc-aug together) collation ──────────────────
#
# vision_baseline.py's own --collate and desc_augmented_vision_v2.py's own
# --collate (above) each summarize one script's output_dir in isolation —
# useful for a quick per-script progress check. Comparing the two conditions
# against each other (the actual lower-bound/upper-bound question) needs both
# directories at once; that combined view + the CI'd bar chart live in
# vision_baseline_collate.py, built on the two helpers below.

def load_all_rows(output_dir: Path) -> dict[str, list[dict]]:
    """{model_dir_name: [row, ...]} — every ckpt_*.csv under output_dir/<model>/."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_files = list(model_dir.glob("ckpt_*.csv"))
        if not ckpt_files:
            continue
        rows = list(csv.DictReader(open(ckpt_files[0], encoding="utf-8")))
        if rows:
            result[model_dir.name] = rows
    return result


def bootstrap_f1_ci(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """Macro-F1 point estimate + 95% CI, bootstrap-resampled over CLASSES (not
    samples) — each class already has its own well-defined F1 (from its own
    TP/FP/FN), and macro-F1 is just their mean, so resampling which classes
    are in the average is the natural unit of uncertainty here, matching "CI
    based on the classes" rather than per-image noise."""
    refs = [r["ground_truth"] for r in rows]
    preds = [r.get("resolved_class") or "" for r in rows]
    classes = sorted(set(refs))
    if not classes:
        return {"f1_macro": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_classes": 0, "n": len(rows)}

    per_class = f1_score(refs, preds, average=None, labels=classes, zero_division=0)
    point = float(np.mean(per_class))
    if len(classes) < 2:
        return {"f1_macro": round(point, 4), "ci_low": round(point, 4),
                "ci_high": round(point, 4), "n_classes": len(classes), "n": len(rows)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(classes))
    boot_means = np.array([
        per_class[rng.choice(idx, size=len(classes), replace=True)].mean()
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "f1_macro": round(point, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "n_classes": len(classes),
        "n": len(rows),
    }


def compute_condition_stats(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict[str, dict]:
    """{"overall": {...}, "<task_type>": {...}, ...} — bootstrap_f1_ci scoped
    to all rows and to each task_type subset."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_type"]].append(r)
    stats = {"overall": bootstrap_f1_ci(rows, n_boot, seed)}
    for tt, trows in by_task.items():
        stats[tt] = bootstrap_f1_ci(trows, n_boot, seed)
    return stats


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
