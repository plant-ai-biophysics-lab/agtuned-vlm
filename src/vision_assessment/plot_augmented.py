"""
Regenerate augmented vision assessment charts from a previously saved metrics JSON.

Reads  <output_dir>/vision_assessment_metrics.json  and writes all chart PNGs
to the same directory without re-running any inference.  Also regenerates the
before/after comparison chart using the baseline metrics from --baseline_dir.

Usage:
    python main.py vision-augmented-plot \\
        --output_dir   outputs/vision_assessment_augmented/ \\
        --baseline_dir outputs/vision_assessment/ \\
        --config       src/config/datasets.yaml \\
        --multimodal_models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import datasets_by_name
from src.vision_assessment.assess import (
    plot_cls_summary_per_dataset,
    plot_cls_summary_per_task,
    plot_desc_summary_per_task,
    plot_overall_cls_summary,
    plot_overall_desc_summary,
)
from src.vision_assessment.assess_augmented import plot_before_after


def _macro_average(per_dataset: dict, metrics: list[str]) -> dict | None:
    if not per_dataset:
        return None
    return {
        m: float(np.mean([v[m] for v in per_dataset.values() if m in v]))
        for m in metrics
        if any(m in v for v in per_dataset.values())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate augmented vision assessment charts from saved metrics JSON."
    )
    parser.add_argument(
        "--output_dir", default="outputs/vision_assessment_augmented/",
        help="Directory containing vision_assessment_metrics.json (charts written here too)",
    )
    parser.add_argument(
        "--baseline_dir", default=None,
        help="Directory containing the baseline vision_assessment_metrics.json "
             "(used for before/after chart). Defaults to --output_dir/../vision_assessment/",
    )
    parser.add_argument(
        "--config", default="src/config/datasets.yaml",
        help="Path to datasets.yaml (needed for task-type groupings)",
    )
    parser.add_argument(
        "--multimodal_models", nargs="*", default=[],
        help="Model IDs that are multimodal VLMs (bars shown with diagonal hatching)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "vision_assessment_metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        sys.exit(1)

    if args.baseline_dir is None:
        args.baseline_dir = str(output_dir.parent / "vision_assessment")

    results = json.loads(metrics_path.read_text())
    augment = results.get("augment", "both")

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task = {name: ds.task_type for name, ds in datasets_cfg.items()}
    multimodal_set = set(args.multimodal_models)

    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics:  dict[str, dict[str, dict]] = {}
    all_judge_metrics:      dict[str, dict] = {}
    summary_metrics:        dict[str, dict] = {}

    for model_id, m in results.get("models", {}).items():
        all_model_desc_metrics[model_id] = m.get("description_per_dataset", {})
        all_model_cls_metrics[model_id]  = m.get("classification_per_dataset", {})
        summary_metrics[model_id] = m
        if m.get("judge"):
            all_judge_metrics[model_id] = m["judge"]

    if not summary_metrics:
        print("ERROR: no model results found in metrics JSON.")
        sys.exit(1)

    naive = results.get("naive", {})
    naive_ov_desc = (
        naive.get("description_overall")
        or _macro_average(naive.get("description", {}), ["rouge1", "rouge2", "rougeL", "rouge_ss"])
    )
    naive_cls_overall = (
        naive.get("classification_overall")
        or _macro_average(naive.get("classification", {}), ["accuracy", "contains_accuracy", "f1_macro"])
    )

    active_datasets = {name: ds for name, ds in datasets_cfg.items() if not ds.skip}
    per_ds_f1 = [1.0 / max(len(ds.clean_classes), 1) for ds in active_datasets.values()]
    uniform_cls_overall = (
        {"f1_macro": round(float(np.mean(per_ds_f1)), 4),
         "accuracy": round(float(np.mean(per_ds_f1)), 4)}
        if per_ds_f1 else None
    )

    print(f"Loaded metrics for {len(summary_metrics)} model(s): {', '.join(summary_metrics)}")
    print(f"Augment mode: {augment}")
    print(f"Naive desc baseline:  { {k: round(v,3) for k,v in (naive_ov_desc or {}).items()} }")
    print(f"Naive cls  baseline:  { {k: round(v,3) for k,v in (naive_cls_overall or {}).items()} }")
    if multimodal_set:
        print(f"Multimodal (hatched): {', '.join(multimodal_set)}")

    print("\n=== Generating plots ===")
    plot_cls_summary_per_dataset(all_model_cls_metrics, output_dir, multimodal_set, naive_cls_overall, uniform_cls_overall)
    plot_cls_summary_per_task(all_model_cls_metrics, ds_to_task, output_dir, multimodal_set, naive_cls_overall, all_judge_metrics, uniform_cls_overall)
    plot_desc_summary_per_task(all_model_desc_metrics, ds_to_task, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_cls_summary(all_model_cls_metrics, output_dir, multimodal_set, naive_cls_overall, all_judge_metrics, uniform_cls_overall)

    baseline_metrics_path = Path(args.baseline_dir) / "vision_assessment_metrics.json"
    plot_before_after(baseline_metrics_path, summary_metrics, augment, output_dir, multimodal_set)

    print(f"\nDone. Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
