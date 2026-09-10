"""
Regenerate all desc_augmented_vision plots without re-running inference.

Usage:
    python main.py aug-vision-plot \\
      --models     google/gemma-4-26B-A4B-it \\
      --vis_dir    outputs/vision_assessment/ \\
      --output_dir outputs/desc_augmented_vision/

    # For the --all_classes run:
    python main.py aug-vision-plot \\
      --models     google/gemma-4-26B-A4B-it \\
      --vis_dir    outputs/vision_assessment/ \\
      --output_dir outputs/desc_augmented_vision/ \\
      --all_classes
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.common.dataset_config import datasets_by_name
from src.common.method_comparison import (
    plot_difficulty_scatter_comparison,
    plot_overall_judge_comparison,
    plot_per_task_full_comparison,
    plot_per_task_judge_comparison,
)
from src.legacy.desc_augmented_vision import (
    _read_csv,
    _safe_name,
    compute_cls_metrics_per_dataset,
    load_clustering_per_dataset,
    plot_aug_distribution,
    plot_comparison,
    plot_delta_per_task,
    plot_difficulty_bins,
    plot_difficulty_bins_clustering,
    plot_difficulty_correlation,
    plot_overall_comparison,
    plot_task_comparison,
    plot_transitions,
)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    vis_dir    = Path(args.vis_dir)
    filter_label = "all" if args.all_classes else "filtered"

    metrics_path = output_dir / f"desc_augmented_metrics_{filter_label}.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        print("Run aug-vision first.")
        raise SystemExit(1)

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task   = {k: v.task_type for k, v in datasets_cfg.items()}

    baseline_per_model:  dict[str, dict[str, dict]] = {}
    augmented_per_model: dict[str, dict[str, dict]] = {}
    baseline_rows_by_model:  dict[str, list[dict]] = {}
    augmented_rows_by_model: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}

    for model_id in args.models:
        safe         = _safe_name(model_id)
        aug_csv      = output_dir / f"aug_cls_{filter_label}_{safe}.csv"
        baseline_csv = output_dir / f"baseline_cls_{filter_label}_{safe}.csv"

        if not aug_csv.exists():
            print(f"  [skip] {aug_csv.name} not found")
            continue

        aug_rows      = _read_csv(aug_csv)
        baseline_rows = _read_csv(baseline_csv) if baseline_csv.exists() else []

        # Merge judged_correct from vision assessment CSV into baseline rows
        vis_cls_csv = vis_dir / f"cls_{safe}.csv"
        if vis_cls_csv.exists() and baseline_rows:
            vis_rows = _read_csv(vis_cls_csv)
            vis_judged = {(r["agml_dataset"], r["image_name"], r["class_label"]): r.get("judged_correct", "")
                          for r in vis_rows}
            for r in baseline_rows:
                key = (r.get("agml_dataset", ""), r.get("image_name", ""), r.get("class_label", ""))
                if key in vis_judged:
                    r["judged_correct"] = vis_judged[key]
            print(f"  Merged judged_correct from {vis_cls_csv.name} into baseline rows")

        augmented_per_model[model_id] = compute_cls_metrics_per_dataset(aug_rows)
        baseline_per_model[model_id]  = compute_cls_metrics_per_dataset(baseline_rows)
        augmented_rows_by_model[model_id] = aug_rows
        baseline_rows_by_model[model_id]  = baseline_rows
        summary[model_id] = {}  # presence flag for plot_aug_distribution
        print(f"  Loaded {aug_csv.name} ({len(aug_rows)} rows)")

    if not augmented_per_model:
        print("No augmented CSVs found. Nothing to plot.")
        raise SystemExit(1)

    print("\n=== Generating plots ===")

    # Shared method comparison charts (same pipeline as cls_per_task_all_errorbar.png)
    for model_id in args.models:
        if model_id not in baseline_rows_by_model or model_id not in augmented_rows_by_model:
            continue
        rows_by_cond = {
            "Vision Baseline": baseline_rows_by_model[model_id],
            "Desc-Aug":        augmented_rows_by_model[model_id],
        }
        plot_per_task_judge_comparison(
            rows_by_cond,
            output_path=output_dir / f"method_cmp_per_task_{filter_label}.png",
            title=f"Desc-Aug — Mean Judged F1 by Task ({filter_label})",
        )
        plot_per_task_full_comparison(
            rows_by_cond,
            output_path=output_dir / f"method_cmp_per_task_full_{filter_label}.png",
            title=f"Desc-Aug — Direct vs Judged F1 by Task ({filter_label})",
        )
        plot_overall_judge_comparison(
            rows_by_cond,
            output_path=output_dir / f"method_cmp_overall_{filter_label}.png",
            title=f"Desc-Aug — Overall Mean Judged F1 ({filter_label})",
        )

    plot_comparison(baseline_per_model, augmented_per_model, output_dir, suffix=filter_label)
    plot_task_comparison(baseline_per_model, augmented_per_model, ds_to_task, output_dir, suffix=filter_label)
    plot_overall_comparison(baseline_per_model, augmented_per_model, output_dir, suffix=filter_label)
    plot_delta_per_task(baseline_rows_by_model, augmented_rows_by_model, output_dir,
                        suffix=filter_label, config_path=Path(args.config))
    plot_transitions(vis_dir, output_dir, ds_to_task, output_dir, args.models,
                     filter_label=filter_label)
    plot_aug_distribution(args.models, summary, output_dir, filter_label, ds_to_task, vis_dir=vis_dir)

    # Difficulty plots (per model — typically one model, but iterate for generality)
    for model_id in args.models:
        if model_id not in baseline_per_model or model_id not in augmented_per_model:
            continue
        bpd = baseline_per_model[model_id]
        apd = augmented_per_model[model_id]
        plot_difficulty_correlation(bpd, apd, output_dir, suffix=filter_label)
        plot_difficulty_bins(bpd, apd, output_dir, suffix=filter_label)
        clustering_per_ds = load_clustering_per_dataset(
            Path(args.clustering_metrics), backbone=args.clustering_backbone)
        if clustering_per_ds:
            plot_difficulty_bins_clustering(
                bpd, apd, clustering_per_ds, output_dir,
                suffix=filter_label, backbone=args.clustering_backbone)

        # Clustering difficulty vs judged F1 — overlaid scatter (shared axes, raw rows)
        _diff_rows = {
            "Vision Baseline": baseline_rows_by_model[model_id],
            "Desc-Aug":        augmented_rows_by_model[model_id],
        }
        plot_difficulty_scatter_comparison(
            _diff_rows,
            clustering_metrics_path=Path(args.clustering_metrics),
            output_path=output_dir / "difficulty_correlation" / f"difficulty_vs_perf_desc_aug_{filter_label}.png",
            backbone=args.clustering_backbone,
            title=f"Dataset Difficulty vs Judged F1 — Desc-Aug ({filter_label})",
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate desc_augmented_vision plots")
    parser.add_argument("--models",      nargs="+", required=True)
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/")
    parser.add_argument("--output_dir",  default="outputs/desc_augmented_vision/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--all_classes", action="store_true",
                        help="Use all-classes run (looks for aug_cls_all_*.csv)")
    parser.add_argument("--clustering_metrics", default="outputs/clustering_eval/clustering_metrics.json",
                        help="Path to clustering_metrics.json for label-free difficulty binning")
    parser.add_argument("--clustering_backbone", default="dinov3-vitb16-pretrain-lvd1689m",
                        help="Backbone key in clustering_metrics.json")
    args = parser.parse_args()
    main(args)
