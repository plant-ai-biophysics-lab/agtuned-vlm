"""
Regenerate all desc_augmented_vision plots without re-running inference.

Usage:
    uv run python src/desc_augmented_vision_plot.py \\
      --models     google/gemma-4-26B-A4B-it \\
      --vis_dir    outputs/vision_assessment/ \\
      --output_dir outputs/desc_augmented_vision/
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.common.dataset_config import datasets_by_name
from src.desc_augmented_vision import (
    _read_csv,
    _safe_name,
    compute_cls_metrics_per_dataset,
    plot_comparison,
    plot_task_comparison,
    plot_overall_comparison,
    plot_transitions,
)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    vis_dir    = Path(args.vis_dir)

    metrics_path = output_dir / "desc_augmented_metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        print("Run desc_augmented_vision.py first.")
        raise SystemExit(1)

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task   = {k: v.task_type for k, v in datasets_cfg.items()}

    baseline_per_model:  dict[str, dict[str, dict]] = {}
    augmented_per_model: dict[str, dict[str, dict]] = {}

    for model_id in args.models:
        safe         = _safe_name(model_id)
        aug_csv      = output_dir / f"aug_cls_{safe}.csv"
        baseline_csv = output_dir / f"baseline_cls_{safe}.csv"

        if not aug_csv.exists():
            print(f"  [skip] {aug_csv.name} not found")
            continue

        aug_rows      = _read_csv(aug_csv)
        baseline_rows = _read_csv(baseline_csv) if baseline_csv.exists() else []

        augmented_per_model[model_id] = compute_cls_metrics_per_dataset(aug_rows)
        baseline_per_model[model_id]  = compute_cls_metrics_per_dataset(baseline_rows)

    if not augmented_per_model:
        print("No augmented CSVs found. Nothing to plot.")
        raise SystemExit(1)

    print("\n=== Generating plots ===")
    plot_comparison(baseline_per_model, augmented_per_model, output_dir)
    plot_task_comparison(baseline_per_model, augmented_per_model, ds_to_task, output_dir)
    plot_overall_comparison(baseline_per_model, augmented_per_model, output_dir)
    plot_transitions(vis_dir, output_dir, ds_to_task, output_dir, args.models)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate desc_augmented_vision plots")
    parser.add_argument("--models",      nargs="+", required=True)
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/")
    parser.add_argument("--output_dir",  default="outputs/desc_augmented_vision/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    args = parser.parse_args()
    main(args)
