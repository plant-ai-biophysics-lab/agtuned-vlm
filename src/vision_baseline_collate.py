"""
Stage 2c — Collate Vision Baseline + Description-Augmented Vision together.

vision_baseline.py's own --collate and desc_augmented_vision_v2.py's own
--collate each summarize one script's output_dir alone. This script instead
reads BOTH output_dirs and puts the two conditions (lower bound / upper
bound) side by side for every model, with 95% confidence intervals
bootstrap-resampled over classes (see bootstrap_f1_ci in
src/common/vision_eval.py) — both overall and per task type.

Outputs (into --output_dir):
  stage2_summary.csv        — model, condition, scope, key, f1_macro, ci_low, ci_high, n_classes, n
  stage2_summary.json       — same, structured
  stage2_f1_by_task.png     — grouped bar chart: task types (+ overall), bars
                               per (model, condition), with 95% CI error bars

Usage (run once after all --model runs of BOTH scripts have finished):
    uv run python main.py vision-baseline-collate \\
        --baseline_dir outputs_final/vision_baseline/ \\
        --aug_dir      outputs_final/desc_augmented_vision_v2/ \\
        --output_dir   outputs_final/vision_baseline_collated/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import colorsys
import csv
import json
import re
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.common.vision_eval import compute_condition_stats, load_all_rows

_CONDITIONS = [("baseline", "Vision Baseline"), ("desc_aug", "Desc-Augmented")]

# Same family-shading scheme as src/linear_probe/probe.py's summary_by_task.png
# (same-family variants get shades of the same hue, more parameters = darker)
# — duplicated locally rather than imported, since probe.py pulls in the full
# torch/transformers stack at module load and this script has no need for it.
_FAMILY_BASE_COLORS = {
    "dinov3": "#1f77b4",
    "yolo": "#ff7f0e",
    "qwen": "#2ca02c",
    "gemma": "#9467bd",
}
_DEFAULT_FAMILY_COLOR = "#7f7f7f"


def _backbone_family(name: str) -> str:
    n = name.lower()
    for family in _FAMILY_BASE_COLORS:
        if family in n:
            return family
    return n


def _param_count_b(name: str) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)b", name.lower())
    return max((float(m) for m in matches), default=0.0)


def _family_shaded_colors(models: list[str]) -> dict[str, tuple]:
    families: dict[str, list[str]] = defaultdict(list)
    for m in models:
        families[_backbone_family(m)].append(m)
    colors: dict[str, tuple] = {}
    for family, members in families.items():
        base_hex = _FAMILY_BASE_COLORS.get(family, _DEFAULT_FAMILY_COLOR)
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
        params = {m: _param_count_b(m) for m in members}
        lo, hi = min(params.values()), max(params.values())
        for m in members:
            frac = (params[m] - lo) / (hi - lo) if hi > lo else 0.5
            lightness = 0.68 - frac * 0.38
            colors[m] = colorsys.hls_to_rgb(h, lightness, s)
    return colors


def collate(baseline_dir: Path, aug_dir: Path, n_boot: int, seed: int) -> dict[str, dict[str, dict]]:
    """{model: {"baseline": {scope: stats}, "desc_aug": {scope: stats}}}"""
    baseline_rows = load_all_rows(baseline_dir)
    aug_rows = load_all_rows(aug_dir)
    models = sorted(set(baseline_rows) | set(aug_rows))

    combined: dict[str, dict[str, dict]] = {}
    for model in models:
        combined[model] = {}
        if model in baseline_rows:
            combined[model]["baseline"] = compute_condition_stats(baseline_rows[model], n_boot, seed)
        if model in aug_rows:
            combined[model]["desc_aug"] = compute_condition_stats(aug_rows[model], n_boot, seed)
    return combined


def write_summary(output_dir: Path, combined: dict[str, dict[str, dict]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage2_summary.json").write_text(json.dumps(combined, indent=2))

    fieldnames = ["model", "condition", "scope", "f1_macro", "ci_low", "ci_high", "n_classes", "n"]
    with open(output_dir / "stage2_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, by_cond in combined.items():
            for cond, by_scope in by_cond.items():
                for scope, stats in by_scope.items():
                    writer.writerow({"model": model, "condition": cond, "scope": scope, **stats})
    print(f"  Saved: {output_dir / 'stage2_summary.csv'}")
    print(f"  Saved: {output_dir / 'stage2_summary.json'}")


def plot_f1_by_task(output_dir: Path, combined: dict[str, dict[str, dict]]) -> None:
    models = sorted(combined.keys())
    if not models:
        print("  No models to plot.")
        return

    task_types = sorted({
        scope for by_cond in combined.values() for by_scope in by_cond.values()
        for scope in by_scope if scope != "overall"
    })
    scopes = task_types + ["overall"]  # overall last, everything else alphabetical

    model_colors = _family_shaded_colors(models)
    n_bars = len(models) * len(_CONDITIONS)
    width = 0.8 / n_bars
    x = np.arange(len(scopes))

    fig, ax = plt.subplots(figsize=(max(9, len(scopes) * 2.2), 5.5))
    bar_i = 0
    for model in models:
        for cond_key, cond_label in _CONDITIONS:
            by_scope = combined[model].get(cond_key, {})
            means, lo_err, hi_err = [], [], []
            for scope in scopes:
                stats = by_scope.get(scope)
                m = stats["f1_macro"] if stats else 0.0
                lo = stats["ci_low"] if stats else 0.0
                hi = stats["ci_high"] if stats else 0.0
                means.append(m)
                lo_err.append(max(0.0, m - lo))
                hi_err.append(max(0.0, hi - m))

            offset = (bar_i - (n_bars - 1) / 2) * width
            label = f"{model.split('/')[-1]} ({cond_label})"
            ax.bar(
                x + offset, means, width, label=label,
                color=model_colors[model],
                hatch="//" if cond_key == "desc_aug" else None,
                edgecolor="white" if cond_key == "desc_aug" else "black",
                linewidth=0.5,
                yerr=[lo_err, hi_err], capsize=2,
                error_kw={"elinewidth": 0.8, "alpha": 0.7},
            )
            bar_i += 1

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scopes], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Macro (95% CI over classes)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(models), 4), fontsize=7)
    plt.tight_layout()
    path = output_dir / "stage2_f1_by_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def main(args: argparse.Namespace) -> None:
    combined = collate(Path(args.baseline_dir), Path(args.aug_dir), args.n_boot, args.seed)
    if not combined:
        print(f"No checkpointed models found under {args.baseline_dir} or {args.aug_dir}")
        return
    output_dir = Path(args.output_dir)
    write_summary(output_dir, combined)
    plot_f1_by_task(output_dir, combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_dir", default="outputs_final/vision_baseline/")
    parser.add_argument("--aug_dir", default="outputs_final/desc_augmented_vision_v2/")
    parser.add_argument("--output_dir", default="outputs_final/vision_baseline_collated/")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CI.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
