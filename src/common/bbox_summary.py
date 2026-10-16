"""Shared 2-(or-more)-condition detection comparison core — IoU-based
precision/recall/F1 bar charts, per-dataset breakdown, and IoU-threshold
sweep, all computed from DET_CKPT_FIELDS-shaped checkpoint rows
(detection_baseline.py's schema, reused unchanged by every detection
condition: plain baseline, --rubric, verifier, ...).

Factored out of detection_summary.py (Stage 5-report, Baseline vs Verifier)
so a second stage-report script (detection_rubric_summary.py, Stage
2.5-report, Baseline vs Rubric) can reuse the identical plotting/stats
logic instead of duplicating it — see method_comparison.py for the same
"shared computation, thin per-stage CLI" pattern used on the classification
side.

Usage
-----
conditions = [("Baseline", "#1f77b4", baseline_rows), ("Rubric", "#2ca02c", rubric_rows)]
keys = keys_intersection([rows for _, _, rows in conditions])
results = collect(conditions, keys, n_boot=2000, seed=42)
plot_bar(results, conditions, len(keys), model_safe, out_path, title="Baseline vs. Rubric")
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

METRICS = ["precision_at_iou50", "recall_at_iou50", "f1_at_iou50", "mean_iou_matched"]
IOU_THRESHOLDS = [0.3, 0.5, 0.75]


def load_rows(ckpt_path: Path) -> list[dict]:
    if not ckpt_path.exists():
        return []
    with open(ckpt_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["agml_dataset"], r["image_name"]) for r in rows}


def keys_intersection(rows_list: list[list[dict]]) -> set[tuple[str, str]]:
    """(agml_dataset, image_name) pairs present in EVERY condition's rows —
    every condition may have been run with a different --n_samples or
    against a still-growing manifest, so comparisons are only fair on the
    exact overlap, never just "the same datasets"."""
    if not rows_list:
        return set()
    keys = _row_keys(rows_list[0])
    for rows in rows_list[1:]:
        keys &= _row_keys(rows)
    return keys


def boot_ci(vals: list[float], rng, n_boot: int = 2000) -> tuple[float, float, float]:
    import numpy as np
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) < 2:
        v = float(arr[0])
        return v, v, v
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def collect(
    conditions: list[tuple[str, str, list[dict]]], keys: set[tuple[str, str]],
    n_boot: int, seed: int,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """{label: {metric: (mean, ci_low, ci_high), "_n": (n, 0, 0)}}, every
    condition filtered to the exact same overlapping keys."""
    import numpy as np
    rng = np.random.default_rng(seed)

    out = {}
    for label, _color, rows in conditions:
        frows = [r for r in rows if (r["agml_dataset"], r["image_name"]) in keys]
        if not frows:
            continue
        out[label] = {}
        for metric in METRICS:
            vals = [float(r[metric]) for r in frows if r.get(metric)]
            out[label][metric] = boot_ci(vals, rng, n_boot)
        out[label]["_n"] = (len(frows), 0, 0)
    return out


def compute_at_thresholds(rows: list[dict], keys: set[tuple[str, str]],
                           thresholds: list[float] = IOU_THRESHOLDS) -> dict[float, dict[str, float]]:
    """Re-derive precision/recall/F1 at every threshold in `thresholds`
    directly from each row's stored pred_boxes_json/gt_boxes_json — the
    checkpoint only ever persists metrics AT THE THRESHOLD THE RUN USED
    (--iou_threshold, default 0.5), so anything else needs recomputing
    from the raw box lists. Per-image F1 first, then averaged (macro),
    matching precision_recall_f1's own per-row convention used everywhere
    else in this pipeline."""
    from src.common.bbox_metrics import match_boxes, precision_recall_f1

    frows = [r for r in rows if (r["agml_dataset"], r["image_name"]) in keys]
    out: dict[float, dict[str, float]] = {}
    for thr in thresholds:
        precisions, recalls, f1s, ious = [], [], [], []
        for r in frows:
            pred = json.loads(r["pred_boxes_json"]) if r.get("pred_boxes_json") else []
            gt = json.loads(r["gt_boxes_json"]) if r.get("gt_boxes_json") else []
            matches, unmatched_pred, unmatched_gt = match_boxes(pred, gt, thr)
            m = precision_recall_f1(matches, unmatched_pred, unmatched_gt)
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            if m["n_matched"]:
                ious.append(m["mean_iou_matched"])
        n = len(frows)
        out[thr] = {
            "precision": round(sum(precisions) / n, 4) if n else 0.0,
            "recall": round(sum(recalls) / n, 4) if n else 0.0,
            "f1": round(sum(f1s) / n, 4) if n else 0.0,
            "mean_iou_matched": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "n_matched": sum(1 for f in f1s if f > 0),
        }
    return out


def plot_bar(results: dict, conditions: list[tuple[str, str, list[dict]]],
             n_images: int, model_safe: str, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    present = [(label, color) for label, color, _rows in conditions if label in results]
    if not present:
        print("  [skip] plot_bar: nothing to plot")
        return

    x = np.arange(len(METRICS))
    n_bars = len(present)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, color) in enumerate(present):
        means, lo_err, hi_err = [], [], []
        for metric in METRICS:
            m, lo, hi = results[label][metric]
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=f"{label} (n={results[label]['_n'][0]})",
                       color=color, edgecolor="black", linewidth=0.6,
                       yerr=[lo_err, hi_err], capsize=4, error_kw={"elinewidth": 1.0, "alpha": 0.75})
        for b, v in zip(bars, means):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", xytext=(0, 3), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision@IoU50", "Recall@IoU50", "F1@IoU50", "Mean IoU\n(matched)"])
    ax.set_ylabel("Score")
    ax.set_title(f"Detection: {title} — {model_safe}\nsame {n_images} images, 95% CI")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_dataset(conditions: list[tuple[str, str, list[dict]]], keys: set[tuple[str, str]],
                     model_safe: str, out_path: Path, title: str, metric: str = "f1_at_iou50") -> None:
    """Per-dataset F1@IoU50 breakdown — detection's analogue of
    classification's by-task chart, since every detection dataset shares
    one task_type (object_detection), dataset is the natural axis of
    variation here instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def _by_dataset(rows):
        out: dict[str, list[float]] = {}
        for r in rows:
            key = (r["agml_dataset"], r["image_name"])
            if key in keys and r.get(metric):
                out.setdefault(r["agml_dataset"], []).append(float(r[metric]))
        return out

    by_ds_by_cond = {label: _by_dataset(rows) for label, _color, rows in conditions}
    datasets = sorted(set().union(*(d.keys() for d in by_ds_by_cond.values())))
    if not datasets:
        print("  [skip] plot_by_dataset: no overlapping rows")
        return

    present = [(label, color) for label, color, _rows in conditions if by_ds_by_cond.get(label)]
    x = np.arange(len(datasets))
    width = 0.8 / max(len(present), 1)
    fig, ax = plt.subplots(figsize=(max(9, len(datasets) * 0.5), 5.5))
    for i, (label, color) in enumerate(present):
        by_ds = by_ds_by_cond[label]
        means = [np.mean(by_ds[d]) if d in by_ds else 0.0 for d in datasets]
        offset = (i - (len(present) - 1) / 2) * width
        ax.bar(x + offset, means, width, label=label, color=color, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Detection: {title} by dataset — {model_safe}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_iou_thresholds(conditions: list[tuple[str, str, list[dict]]], keys: set[tuple[str, str]],
                         model_safe: str, out_path: Path, title: str = "") -> None:
    """F1 at multiple IoU thresholds (0.3/0.5/0.75 by default), one line
    per condition — a single fixed threshold (0.5) can zero out real,
    "close but not tight enough" localizations, so this shows the fuller
    picture without needing per-box confidence scores for a real AP
    curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, rows in conditions:
        if not rows:
            continue
        by_thr = compute_at_thresholds(rows, keys)
        thrs = sorted(by_thr)
        f1s = [by_thr[t]["f1"] for t in thrs]
        ax.plot(thrs, f1s, marker="o", markersize=7, linewidth=1.8, color=color, label=label)
        for t, f1 in zip(thrs, f1s):
            ax.annotate(f"{f1:.3f}", (t, f1), ha="center", va="bottom", fontsize=8, fontweight="bold",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("F1")
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Detection F1 vs. IoU threshold{subtitle} — {model_safe}")
    ax.set_xticks(IOU_THRESHOLDS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_summary_csv(results: dict, conditions: list[tuple[str, str, list[dict]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "mean", "ci_low", "ci_high", "n"])
        for label, _color, _rows in conditions:
            if label not in results:
                continue
            n = results[label]["_n"][0]
            for metric in METRICS:
                m, lo, hi = results[label][metric]
                w.writerow([label, metric, m, lo, hi, n])
    print(f"  Saved: {out_path}")


def write_iou_thresholds_csv(conditions: list[tuple[str, str, list[dict]]], keys: set[tuple[str, str]],
                              out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "iou_threshold", "precision", "recall", "f1", "mean_iou_matched", "n_matched"])
        for label, _color, rows in conditions:
            if not rows:
                continue
            for thr, m in compute_at_thresholds(rows, keys).items():
                w.writerow([label, thr, m["precision"], m["recall"], m["f1"], m["mean_iou_matched"], m["n_matched"]])
    print(f"  Saved: {out_path}")
