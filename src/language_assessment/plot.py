"""
Regenerate language assessment charts from a previously saved metrics JSON.

Reads  <output_dir>/language_assessment_metrics.json  and writes all chart PNGs
to the same directory without re-running any inference.

Usage:
    python -m src.language_assessment.plot \
        --output_dir outputs/language_assessment/ \
        --config     src/config/datasets.yaml \
        --multimodal_models google/gemma-4-26B-A4B-it Qwen/Qwen3.5-VL-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import datasets_by_name
from src.language_assessment.assess import (
    plot_cls_summary_per_dataset,
    plot_cls_summary_per_task,
    plot_desc_summary_per_task,
    plot_overall_cls_summary,
    plot_overall_desc_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate language assessment charts from saved metrics JSON."
    )
    parser.add_argument(
        "--output_dir", default="outputs/language_assessment/",
        help="Directory that contains language_assessment_metrics.json (charts written here too)",
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
    metrics_path = output_dir / "language_assessment_metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        sys.exit(1)

    results = json.loads(metrics_path.read_text())
    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task = {name: ds.task_type for name, ds in datasets_cfg.items()}
    multimodal_set = set(args.multimodal_models)

    # Reconstruct per-model per-dataset dicts from saved JSON
    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics: dict[str, dict[str, dict]] = {}

    for model_id, m in results.get("models", {}).items():
        all_model_desc_metrics[model_id] = m.get("description_per_dataset", {})
        all_model_cls_metrics[model_id] = m.get("classification_per_dataset", {})

    if not all_model_desc_metrics:
        print("ERROR: no model results found in metrics JSON.")
        sys.exit(1)

    print(f"Loaded metrics for {len(all_model_desc_metrics)} model(s): "
          f"{', '.join(all_model_desc_metrics)}")
    if multimodal_set:
        print(f"Multimodal (hatched): {', '.join(multimodal_set)}")

    print("\n=== Generating plots ===")
    plot_cls_summary_per_dataset(all_model_cls_metrics, output_dir, multimodal_set)
    plot_cls_summary_per_task(all_model_cls_metrics, ds_to_task, output_dir, multimodal_set)
    plot_desc_summary_per_task(all_model_desc_metrics, ds_to_task, output_dir, multimodal_set)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir, multimodal_set)
    plot_overall_cls_summary(all_model_cls_metrics, output_dir, multimodal_set)

    print(f"\nDone. Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
