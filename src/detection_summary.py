"""
Stage 5-report — Detection-Baseline vs. Detection-Verifier comparison
(no GPU, no model).

Mirrors verifier_summary.py's role for classification: a quick,
apples-to-apples check of how the single-round K-candidate + PPT-tournament
verifier (detection_verifier.py) compares against the no-reasoning
detection_baseline.py, using IoU-based precision/recall/F1 (mean_iou_matched
too) since there is no per-box confidence score to build a real mAP curve
from (see detection_baseline.py's module docstring / bbox_metrics.py).

Every condition is filtered to the EXACT (agml_dataset, image_name) pairs
BOTH runs scored, not merely the same datasets, since the two scripts may
have been run with different --n_samples or against a still-growing
manifest.

Usage:
    uv run python main.py detection-summary \\
        --baseline_dir outputs_final/detection_baseline/ \\
        --verifier_dir outputs_final/detection_verifier/ \\
        --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_metrics import match_boxes, precision_recall_f1  # noqa: E402
from src.common.vision_eval import safe_model_name  # noqa: E402

_METRICS = ["precision_at_iou50", "recall_at_iou50", "f1_at_iou50", "mean_iou_matched"]
_CONDITIONS = [("Baseline", "#1f77b4"), ("Verifier", "#e06b00")]
_IOU_THRESHOLDS = [0.3, 0.5, 0.75]


def _load_rows(ckpt_path: Path) -> list[dict]:
    if not ckpt_path.exists():
        return []
    with open(ckpt_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["agml_dataset"], r["image_name"]) for r in rows}


def _boot_ci(vals: list[float], rng, n_boot: int = 2000) -> tuple[float, float, float]:
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


def collect(baseline_rows: list[dict], verifier_rows: list[dict], keys: set[tuple[str, str]],
            n_boot: int, seed: int) -> dict[str, dict[str, tuple[float, float, float]]]:
    """{condition: {metric: (mean, ci_low, ci_high)}}, both conditions
    filtered to the exact same overlapping (dataset, image) keys."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def _filt(rows):
        return [r for r in rows if (r["agml_dataset"], r["image_name"]) in keys]

    out = {}
    for label, rows in (("Baseline", baseline_rows), ("Verifier", verifier_rows)):
        frows = _filt(rows)
        if not frows:
            continue
        out[label] = {}
        for metric in _METRICS:
            vals = [float(r[metric]) for r in frows if r.get(metric)]
            out[label][metric] = _boot_ci(vals, rng, n_boot)
        out[label]["_n"] = (len(frows), 0, 0)
    return out


def plot(results: dict, n_images: int, model_safe: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    present = [(label, color) for label, color in _CONDITIONS if label in results]
    if not present:
        print("  [skip] nothing to plot")
        return

    x = np.arange(len(_METRICS))
    n_bars = len(present)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, color) in enumerate(present):
        means, lo_err, hi_err = [], [], []
        for metric in _METRICS:
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
    ax.set_title(f"Detection: Baseline vs. Verifier — {model_safe}\nsame {n_images} images, 95% CI")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_dataset(baseline_rows: list[dict], verifier_rows: list[dict], keys: set[tuple[str, str]],
                     model_safe: str, out_path: Path, metric: str = "f1_at_iou50") -> None:
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

    base_by_ds = _by_dataset(baseline_rows)
    ver_by_ds = _by_dataset(verifier_rows)
    datasets = sorted(set(base_by_ds) | set(ver_by_ds))
    if not datasets:
        print("  [skip] plot_by_dataset: no overlapping rows")
        return

    x = np.arange(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(datasets) * 0.5), 5.5))
    for offset, (label, by_ds, color) in zip(
        (-width / 2, width / 2),
        (("Baseline", base_by_ds, "#1f77b4"), ("Verifier", ver_by_ds, "#e06b00")),
    ):
        means = [np.mean(by_ds[d]) if d in by_ds else 0.0 for d in datasets]
        ax.bar(x + offset, means, width, label=label, color=color, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Detection: Baseline vs. Verifier by dataset — {model_safe}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def compute_at_thresholds(rows: list[dict], keys: set[tuple[str, str]],
                           thresholds: list[float] = _IOU_THRESHOLDS) -> dict[float, dict[str, float]]:
    """Re-derive precision/recall/F1 at every threshold in `thresholds`
    directly from each row's stored pred_boxes_json/gt_boxes_json — the
    checkpoint only ever persists metrics AT THE THRESHOLD THE RUN USED
    (--iou_threshold, default 0.5), so anything else needs recomputing
    from the raw box lists, not read off an existing column. No rerun
    needed: the box lists themselves are already exactly what the
    generation run produced, only the matching/scoring threshold changes.

    Per-image F1 is computed first, then averaged across images (macro
    average) — matching precision_recall_f1's own per-row convention
    used everywhere else in this pipeline (score_row in
    detection_baseline.py/detection_verifier.py, mean_f1_at_iou50 in
    metrics.json), NOT a pooled/micro-averaged TP+FP+FN-across-all-images
    score. Mixing the two conventions in the same report is a correctness
    bug, not just a style choice — they can disagree substantially
    whenever per-image box counts vary."""
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


def plot_iou_thresholds(baseline_rows: list[dict], verifier_rows: list[dict], keys: set[tuple[str, str]],
                         model_safe: str, out_path: Path) -> None:
    """F1 at multiple IoU thresholds (0.3/0.5/0.75 by default), one line
    per condition — a single fixed threshold (0.5) can zero out real,
    "close but not tight enough" localizations (verified concretely on
    this project's own data: several predicted boxes landing at IoU
    0.28-0.40 against their correct ground-truth match, all reported as
    non-matches at the standard 0.5 cutoff), so this shows the fuller
    picture without needing per-box confidence scores for a real AP
    curve (see this module's own docstring for why mAP itself isn't
    computable here)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, rows, color in (("Baseline", baseline_rows, "#1f77b4"), ("Verifier", verifier_rows, "#e06b00")):
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
    ax.set_title(f"Detection F1 vs. IoU threshold — {model_safe}")
    ax.set_xticks(_IOU_THRESHOLDS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    model_safe = safe_model_name(args.model)
    baseline_ckpt = Path(args.baseline_dir) / model_safe / f"ckpt_{model_safe}.csv"
    verifier_ckpt = Path(args.verifier_dir) / model_safe / f"ckpt_{model_safe}.csv"

    baseline_rows = _load_rows(baseline_ckpt)
    verifier_rows = _load_rows(verifier_ckpt)
    if not baseline_rows:
        raise SystemExit(f"No baseline checkpoint at {baseline_ckpt}")
    if not verifier_rows:
        print(f"  [skip] no verifier checkpoint yet at {verifier_ckpt} — "
              f"run `detection-verifier --model {args.model}` first")
        return

    keys = _keys(baseline_rows) & _keys(verifier_rows)
    if not keys:
        print("  [skip] baseline and verifier share no overlapping (dataset, image) pairs")
        return
    print(f"  Filtering both conditions to exactly {len(keys)} overlapping image(s)")

    results = collect(baseline_rows, verifier_rows, keys, args.n_boot, args.seed)

    out_dir = Path(args.output_dir) / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(results, len(keys), model_safe, out_dir / "detection_summary.png")
    plot_by_dataset(baseline_rows, verifier_rows, keys, model_safe, out_dir / "detection_summary_by_dataset.png")
    plot_iou_thresholds(baseline_rows, verifier_rows, keys, model_safe, out_dir / "detection_summary_iou_thresholds.png")

    csv_path = out_dir / "detection_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "mean", "ci_low", "ci_high", "n"])
        for label in results:
            n = results[label]["_n"][0]
            for metric in _METRICS:
                m, lo, hi = results[label][metric]
                w.writerow([label, metric, m, lo, hi, n])

    thresholds_csv = out_dir / "detection_summary_iou_thresholds.csv"
    with open(thresholds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "iou_threshold", "precision", "recall", "f1", "mean_iou_matched", "n_matched"])
        for label, rows in (("Baseline", baseline_rows), ("Verifier", verifier_rows)):
            for thr, m in compute_at_thresholds(rows, keys).items():
                w.writerow([label, thr, m["precision"], m["recall"], m["f1"], m["mean_iou_matched"], m["n_matched"]])
    print(f"  Saved: {thresholds_csv}")
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline_dir", default="outputs_final/detection_baseline/")
    p.add_argument("--verifier_dir", default="outputs_final/detection_verifier/")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--output_dir", default="outputs_final/detection_summary/")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
