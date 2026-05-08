"""
Linear Probe — Batch Runner across all AgML datasets
=====================================================
Reads datasets.csv, runs linear_probe.main() for each dataset, then
aggregates results by task (3rd column) and plots a summary bar chart
where each x-axis group is a task and bars are per-backbone metrics.

Usage:
    python run_all_datasets.py --model /path/to/finetuned \\
                               --baseline_model google/gemma-4-E2B-it \\
                               --yolo_model yolov8s-cls.pt \\
                               --output_dir run_all_results

    # Skip datasets already processed (safe to re-run after interruption):
    python run_all_datasets.py --model /path/to/finetuned --output_dir run_all_results

    # Discover-only dry run (no model loading):
    python run_all_datasets.py --output_dir run_all_results --dry_run
"""

import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


DATASETS_CSV = Path(__file__).parent / "datasets.csv"


def load_datasets(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append({"dataset": parts[0], "crop": parts[1], "task": parts[2]})
    return rows


def result_path(output_dir: str, dataset: str) -> Path:
    return Path(output_dir) / dataset / "linear_probe_results.json"


def run_single(dataset_name: str, args: argparse.Namespace, dataset_output_dir: str):
    from linear_probe import main as lp_main

    lp_args = SimpleNamespace(
        model=args.model,
        baseline_model=args.baseline_model,
        dino_model=args.dino_model,
        yolo_model=args.yolo_model,
        agml_dataset=dataset_name,
        output_dir=dataset_output_dir,
        n_samples=args.n_samples,
    )
    lp_main(lp_args)


def load_all_results(output_dir: str, rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        rp = result_path(output_dir, row["dataset"])
        if not rp.exists():
            continue
        with open(rp) as f:
            data = json.load(f)
        for backbone, metrics in data.get("backbones", {}).items():
            records.append({
                "dataset": row["dataset"],
                "task":    row["task"],
                "backbone": backbone,
                "accuracy":    metrics.get("accuracy", 0.0),
                "f1_macro":    metrics.get("f1_macro", 0.0),
                "f1_weighted": metrics.get("f1_weighted", 0.0),
            })
    return records


def aggregate_by_task(records: list[dict]) -> dict:
    """
    Returns {task: {backbone: {metric: mean_value}}}.
    Averages over all datasets within each task.
    """
    from collections import defaultdict

    # task -> backbone -> metric -> [values]
    accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        for metric in ("accuracy", "f1_macro", "f1_weighted"):
            accum[r["task"]][r["backbone"]][metric].append(r[metric])

    aggregated = {}
    for task, backbones in accum.items():
        aggregated[task] = {}
        for backbone, metrics in backbones.items():
            aggregated[task][backbone] = {
                m: round(float(np.mean(vals)), 4)
                for m, vals in metrics.items()
            }
    return aggregated


def plot_summary(aggregated: dict, output_dir: str, metric: str = "accuracy"):
    tasks     = sorted(aggregated.keys())
    backbones = sorted({b for task in aggregated.values() for b in task})

    x      = np.arange(len(tasks))
    n      = len(backbones)
    width  = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2.5), 6))

    for i, backbone in enumerate(backbones):
        vals   = [aggregated[task].get(backbone, {}).get(metric, 0.0) for task in tasks]
        offset = (i - (n - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=backbone)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Linear Probe — {metric.replace('_', ' ').title()} by Task (averaged across datasets)")
    ax.legend(title="Backbone", fontsize=8)
    plt.tight_layout()

    path = os.path.join(output_dir, f"summary_by_task_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_dataset(records: list[dict], output_dir: str, metric: str = "accuracy"):
    """Secondary chart: per-dataset scores grouped by task (small multiples)."""
    from collections import defaultdict

    task_datasets: dict = defaultdict(list)
    for r in records:
        task_datasets[r["task"]].append(r["dataset"])
    # Deduplicate while preserving order
    task_datasets = {t: list(dict.fromkeys(ds)) for t, ds in task_datasets.items()}

    backbones = sorted({r["backbone"] for r in records})

    n_tasks = len(task_datasets)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, (task, datasets) in zip(axes, sorted(task_datasets.items())):
        n  = len(backbones)
        x  = np.arange(len(datasets))
        w  = 0.8 / n

        for i, backbone in enumerate(backbones):
            lookup = {
                r["dataset"]: r[metric]
                for r in records
                if r["backbone"] == backbone and r["task"] == task
            }
            vals   = [lookup.get(ds, 0.0) for ds in datasets]
            offset = (i - (n - 1) / 2) * w
            ax.bar(x + offset, vals, w, label=backbone)

        short_names = [ds.split("_")[0] for ds in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_title(task, fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title() if ax == axes[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Backbone", loc="upper right", fontsize=8)
    fig.suptitle(f"Linear Probe — {metric.replace('_', ' ').title()} per Dataset", fontsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, f"per_dataset_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probe across all AgML datasets and plot task-level summary"
    )
    parser.add_argument("--model", type=str, default=None,
        help="Fine-tuned Gemma4 path or HF ID")
    parser.add_argument("--baseline_model", type=str, default=None,
        help="Base Gemma4 before fine-tuning")
    parser.add_argument("--dino_model", type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 model ID. Pass 'none' to skip.")
    parser.add_argument("--yolo_model", type=str, default=None,
        help="YOLO classification model (e.g. yolo11s-cls.pt)")
    parser.add_argument("--output_dir", type=str, default="run_all_results",
        help="Root output directory (one subdir per dataset)")
    parser.add_argument("--n_samples", type=int, default=None,
        help="Max samples per dataset (stratified subsample). Default: use all.")
    parser.add_argument("--dry_run", action="store_true",
        help="Skip model runs — only aggregate and plot already-saved results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_datasets(DATASETS_CSV)
    print(f"Loaded {len(rows)} datasets from {DATASETS_CSV}")

    # ── Per-dataset runs ───────────────────────────────────────────────────────
    if not args.dry_run:
        failed = []
        for i, row in enumerate(rows):
            dataset = row["dataset"]
            rp      = result_path(args.output_dir, dataset)

            if rp.exists():
                print(f"[{i+1}/{len(rows)}] Skipping {dataset} (already done)")
                continue

            print(f"\n[{i+1}/{len(rows)}] Running: {dataset} (task={row['task']}) ...")
            dataset_out = str(Path(args.output_dir) / dataset)
            os.makedirs(dataset_out, exist_ok=True)

            try:
                run_single(dataset, args, dataset_out)
            except Exception as e:
                print(f"  ERROR on {dataset}: {e}")
                traceback.print_exc()
                failed.append(dataset)

        if failed:
            print(f"\nFailed datasets ({len(failed)}): {failed}")

    # ── Aggregate and plot ─────────────────────────────────────────────────────
    records = load_all_results(args.output_dir, rows)
    if not records:
        print("No results found — nothing to plot.")
        return

    n_completed = len({r["dataset"] for r in records})
    print(f"\nAggregating {len(records)} backbone-dataset records across {n_completed} datasets ...")

    aggregated = aggregate_by_task(records)

    agg_path = os.path.join(args.output_dir, "aggregated_by_task.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved: {agg_path}")

    for metric in ("accuracy", "f1_macro"):
        plot_summary(aggregated, args.output_dir, metric=metric)
        plot_per_dataset(records, args.output_dir, metric=metric)

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
