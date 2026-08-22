"""
Standalone plotting for Verifier-CoT — regenerate comparison charts from
already-generated checkpoint CSVs, without loading vLLM or a model.

Useful when the comparison CSVs (vision_assessment/cot_vision/
desc_augmented_vision outputs) are more complete in one place than another —
e.g. a local copy of outputs/ has cot_cls_{model}.csv but a remote run's
outputs/ directory is missing it, so run_verifier_cot.py's own Phase 3
(--plot_only) can't find it there. This script takes the judged CSV
(ckpt_judged.csv, itself already produced by run_verifier_cot.py) and lets
you point the comparison directories anywhere, independent of wherever
inference for the Verifier-CoT condition itself actually ran.

Usage:
  uv run python experiments/verifier_cot/plot_verifier_cot.py \\
    --judged_csv outputs/verifier_cot_smoketest/ckpt_judged.csv \\
    --vis_dir    outputs/vision_assessment/ \\
    --cot_dir    outputs/cot_vision/ \\
    --aug_dir    outputs/desc_augmented_vision/ \\
    --model      google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_cot_smoketest/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_verifier_cot import _read_csv, plot_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Verifier-CoT comparison charts from existing CSVs (no model/GPU needed)"
    )
    parser.add_argument("--judged_csv", required=True,
                        help="Path to ckpt_judged.csv from a run_verifier_cot.py run")
    parser.add_argument("--vis_dir", default="outputs/vision_assessment/")
    parser.add_argument("--cot_dir", default="outputs/cot_vision/")
    parser.add_argument("--aug_dir", default="outputs/desc_augmented_vision/")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID — must match the model_safe used in the "
                             "comparison CSVs' filenames (cls_{model_safe}.csv etc.)")
    parser.add_argument("--output_dir", default="outputs/verifier_cot/")
    args = parser.parse_args()

    from src.verifier_vision import load_existing_rows

    judged_path = Path(args.judged_csv)
    if not judged_path.exists():
        print(f"ERROR: judged CSV not found: {judged_path}")
        return
    judged = _read_csv(judged_path)
    print(f"Loaded {len(judged)} judged Verifier-CoT rows from {judged_path}")

    model_safe = args.model.replace("/", "_")
    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    print(f"Unique images: {len(image_keys)}")

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        model_safe=model_safe,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_all(judged, existing, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
