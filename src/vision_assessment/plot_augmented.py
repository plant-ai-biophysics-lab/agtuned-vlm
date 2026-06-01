"""
Regenerate augmented vision assessment before/after chart from saved metrics JSON.

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
        description="Regenerate augmented vision assessment before/after chart from saved metrics JSON."
    )
    parser.add_argument(
        "--output_dir", default="outputs/vision_assessment_augmented/",
    )
    parser.add_argument(
        "--baseline_dir", default=None,
        help="Directory containing the baseline vision_assessment_metrics.json. "
             "Defaults to --output_dir/../vision_assessment/",
    )
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--multimodal_models", nargs="*", default=[])
    args = parser.parse_args()

    output_dir   = Path(args.output_dir)
    metrics_path = output_dir / "vision_assessment_metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        sys.exit(1)

    if args.baseline_dir is None:
        args.baseline_dir = str(output_dir.parent / "vision_assessment")

    results        = json.loads(metrics_path.read_text())
    augment        = results.get("augment", "both")
    multimodal_set = set(args.multimodal_models)
    summary_metrics: dict[str, dict] = {}

    for model_id, m in results.get("models", {}).items():
        summary_metrics[model_id] = m

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

    print(f"Loaded metrics for {len(summary_metrics)} model(s): {', '.join(summary_metrics)}")
    print(f"Augment mode: {augment}")

    baseline_metrics_path = Path(args.baseline_dir) / "vision_assessment_metrics.json"
    plot_before_after(
        baseline_metrics_path, summary_metrics, augment, output_dir,
        multimodal_set=multimodal_set,
        naive_cls_overall=naive_cls_overall,
        naive_ov_desc=naive_ov_desc,
    )

    print(f"\nDone. Chart saved to {output_dir}/")


if __name__ == "__main__":
    main()
