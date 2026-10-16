"""Stage 5-report — Detection-Baseline vs. Detection-Verifier comparison
(no GPU, no model).

Mirrors verifier_summary.py's role for classification: a quick,
apples-to-apples check of how the single-round K-candidate + PPT-tournament
verifier (detection_verifier.py) compares against the no-reasoning
detection_baseline.py, using IoU-based precision/recall/F1 (mean_iou_matched
too) since there is no per-box confidence score to build a real mAP curve
from (see detection_baseline.py's module docstring / bbox_metrics.py).

Thin wrapper over src/common/bbox_summary.py's condition-generic
comparison core — see src/detection_rubric_summary.py (Stage 2.5-report,
Baseline vs Rubric) for the other script sharing that same core.

Every condition is filtered to the EXACT (agml_dataset, image_name) pairs
BOTH runs scored, not merely the same datasets, since the two scripts may
have been run with different --n_samples or against a still-growing
manifest.

Usage:
    uv run python main.py detection-summary \\
        --baseline_dir outputs_final/detection_baseline/ \\
        --verifier_dir outputs_final/detection_verifier/ \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic
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

_TITLE = "Baseline vs. Verifier"


def main(args: argparse.Namespace) -> None:
    model_safe = safe_model_name(args.model)
    baseline_ckpt = Path(args.baseline_dir) / model_safe / f"ckpt_{model_safe}.csv"
    verifier_ckpt = Path(args.verifier_dir) / model_safe / f"ckpt_{model_safe}.csv"

    baseline_rows = load_rows(baseline_ckpt)
    verifier_rows = load_rows(verifier_ckpt)
    if not baseline_rows:
        raise SystemExit(f"No baseline checkpoint at {baseline_ckpt}")
    if not verifier_rows:
        print(f"  [skip] no verifier checkpoint yet at {verifier_ckpt} — "
              f"run `detection-verifier --model {args.model}` first")
        return

    conditions = [("Baseline", "#1f77b4", baseline_rows), ("Verifier", "#e06b00", verifier_rows)]
    keys = keys_intersection([baseline_rows, verifier_rows])
    if not keys:
        print("  [skip] baseline and verifier share no overlapping (dataset, image) pairs")
        return
    print(f"  Filtering both conditions to exactly {len(keys)} overlapping image(s)")

    results = collect(conditions, keys, args.n_boot, args.seed)

    out_dir = Path(args.output_dir) / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bar(results, conditions, len(keys), model_safe, out_dir / "detection_summary.png", _TITLE)
    plot_by_dataset(conditions, keys, model_safe, out_dir / "detection_summary_by_dataset.png", _TITLE)
    plot_iou_thresholds(conditions, keys, model_safe, out_dir / "detection_summary_iou_thresholds.png")

    write_summary_csv(results, conditions, out_dir / "detection_summary.csv")
    write_iou_thresholds_csv(conditions, keys, out_dir / "detection_summary_iou_thresholds.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline_dir", default="outputs_final/detection_baseline/")
    p.add_argument("--verifier_dir", default="outputs_final/detection_verifier/")
    p.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    p.add_argument("--output_dir", default="outputs_final/detection_summary/")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
