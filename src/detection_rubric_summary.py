"""Stage 2.5-report — Detection-Baseline vs. Detection-Baseline-with-Rubric
comparison (no GPU, no model).

Same idea as vision_baseline.py's own "Baseline with rubric" ablation
(docs/stage2_baseline_rubric.md), applied to the spatial task: does
answering a fill-in-the-blank localization checklist (object/class,
location, occlusion, size — src/config/verifier_prompts_detection.yaml's
structured_templates["object_detection"], see detection_baseline.py's
--rubric flag) before printing boxes help or hurt IoU-based
precision/recall/F1, at an equal image budget and no tournament/verifier
machinery at all — just two single-shot prompts.

Thin wrapper over src/common/bbox_summary.py's condition-generic
comparison core — see src/detection_summary.py (Stage 5-report, Baseline
vs Verifier) for the other script sharing that same core.

Every condition is filtered to the EXACT (agml_dataset, image_name) pairs
BOTH runs scored, not merely the same datasets, since the two runs may
have been made with different --n_samples or against a still-growing
manifest.

Usage:
    uv run python main.py detection-rubric-summary \\
        --baseline_dir outputs_final/detection_baseline/ \\
        --rubric_dir outputs_final/detection_baseline_rubric/ \\
        --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_summary import (  # noqa: E402
    collect, keys_intersection, load_rows, plot_bar, plot_by_dataset, plot_iou_thresholds,
    write_iou_thresholds_csv, write_summary_csv,
)
from src.common.vision_eval import safe_model_name  # noqa: E402

_TITLE = "Baseline vs. Rubric"


def main(args: argparse.Namespace) -> None:
    model_safe = safe_model_name(args.model)
    baseline_ckpt = Path(args.baseline_dir) / model_safe / f"ckpt_{model_safe}.csv"
    rubric_ckpt = Path(args.rubric_dir) / model_safe / f"ckpt_{model_safe}.csv"

    baseline_rows = load_rows(baseline_ckpt)
    rubric_rows = load_rows(rubric_ckpt)
    if not baseline_rows:
        raise SystemExit(f"No baseline checkpoint at {baseline_ckpt}")
    if not rubric_rows:
        print(f"  [skip] no rubric checkpoint yet at {rubric_ckpt} — "
              f"run `detection-baseline --model {args.model} --rubric "
              f"--output_dir {args.rubric_dir}` first")
        return

    conditions = [("Baseline", "#1f77b4", baseline_rows), ("Rubric", "#2ca02c", rubric_rows)]
    keys = keys_intersection([baseline_rows, rubric_rows])
    if not keys:
        print("  [skip] baseline and rubric share no overlapping (dataset, image) pairs")
        return
    print(f"  Filtering both conditions to exactly {len(keys)} overlapping image(s)")

    results = collect(conditions, keys, args.n_boot, args.seed)

    out_dir = Path(args.output_dir) / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bar(results, conditions, len(keys), model_safe, out_dir / "detection_rubric_summary.png", _TITLE)
    plot_by_dataset(conditions, keys, model_safe, out_dir / "detection_rubric_summary_by_dataset.png", _TITLE)
    plot_iou_thresholds(conditions, keys, model_safe, out_dir / "detection_rubric_summary_iou_thresholds.png")

    write_summary_csv(results, conditions, out_dir / "detection_rubric_summary.csv")
    write_iou_thresholds_csv(conditions, keys, out_dir / "detection_rubric_summary_iou_thresholds.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline_dir", default="outputs_final/detection_baseline/")
    p.add_argument("--rubric_dir", default="outputs_final/detection_baseline_rubric/")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--output_dir", default="outputs_final/detection_rubric_summary/")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
