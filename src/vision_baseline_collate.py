"""
Stage 2c — Collate Vision Baseline + Description-Augmented Vision together.

vision_baseline.py's own --collate and desc_augmented_vision_v2.py's own
--collate each summarize one script's output_dir alone. This script instead
reads BOTH output_dirs and puts the two conditions (lower bound / upper
bound) side by side for every model, with 95% confidence intervals
bootstrap-resampled over classes (see bootstrap_f1_ci in
src/common/vision_eval.py) — both overall and per task type.

Also produces a supplementary per-class consistency plot: a box plot of each
model/condition's raw per-class F1 distribution (no resampling) — the CI bar
chart tells you how confident we are in the average; this tells you how
consistent the model is across classes (tight box = uniform performance,
wide box/long whiskers = some classes much harder than others).

Outputs (into --output_dir):
  stage2_summary.csv          — model, condition, scope, key, f1_macro, ci_low, ci_high, n_classes, n
  stage2_summary.json         — same, structured
  stage2_f1_by_task.png       — grouped bar chart: task types (+ overall), bars
                                 per (model, condition), with 95% CI error bars
  stage2_f1_percentiles.png   — supplementary: box plot of each model/condition's
                                 raw per-class F1 scores, one panel per task type + overall

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
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.common.plot_style import family_shaded_colors as _family_shaded_colors
from src.common.vision_eval import compute_condition_stats, load_all_rows, per_class_f1_values

_CONDITIONS = [("baseline", "Vision Baseline"), ("desc_aug", "Desc-Augmented")]


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


def collect_per_class_f1(baseline_dir: Path, aug_dir: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """{model: {"baseline"/"desc_aug": {scope: [per-class F1, ...]}}} — the
    raw (unresampled) per-class F1 array behind bootstrap_f1_ci's mean/CI,
    scoped the same way (overall + each task type) so it lines up 1:1 with
    plot_f1_by_task's bars."""
    baseline_rows = load_all_rows(baseline_dir)
    aug_rows = load_all_rows(aug_dir)
    models = sorted(set(baseline_rows) | set(aug_rows))

    def _scoped_arrays(rows: list[dict]) -> dict[str, list[float]]:
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_task[r["task_type"]].append(r)
        _, overall_arr = per_class_f1_values(rows)
        scoped = {"overall": overall_arr.tolist()}
        for tt, trows in by_task.items():
            _, arr = per_class_f1_values(trows)
            scoped[tt] = arr.tolist()
        return scoped

    result: dict[str, dict[str, dict[str, list[float]]]] = {}
    for model in models:
        result[model] = {}
        if model in baseline_rows:
            result[model]["baseline"] = _scoped_arrays(baseline_rows[model])
        if model in aug_rows:
            result[model]["desc_aug"] = _scoped_arrays(aug_rows[model])
    return result


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


def plot_f1_percentiles(output_dir: Path, per_class_arrays: dict[str, dict[str, dict[str, list[float]]]]) -> None:
    """Supplementary: box plot of each model/condition's RAW per-class F1
    scores (no resampling) — one panel per task type + overall. Shows
    consistency across classes (tight box = uniform performance, long
    whiskers/wide IQR = some classes much harder than others), which the CI
    bar chart doesn't — that chart's error bars describe uncertainty in the
    mean, not the spread of the underlying per-class scores."""
    models = sorted(per_class_arrays.keys())
    if not models:
        print("  No models to plot.")
        return

    task_types = sorted({
        scope for by_cond in per_class_arrays.values() for by_scope in by_cond.values()
        for scope in by_scope if scope != "overall"
    })
    scopes = task_types + ["overall"]

    model_colors = _family_shaded_colors(models)
    n_boxes = len(models) * len(_CONDITIONS)

    fig, axes = plt.subplots(1, len(scopes), figsize=(max(4.5, n_boxes * 0.7) * len(scopes), 5.5), sharey=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope in zip(axes, scopes):
        positions, data, colors, hatches = [], [], [], []
        pos = 0
        for model in models:
            for cond_key, _ in _CONDITIONS:
                arr = per_class_arrays.get(model, {}).get(cond_key, {}).get(scope, [])
                pos += 1
                if not arr:
                    continue
                data.append(arr)
                positions.append(pos)
                colors.append(model_colors[model])
                hatches.append(cond_key == "desc_aug")

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                         showfliers=True, medianprops={"color": "black"}, flierprops={"markersize": 3})
        for patch, color, hatch in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(color)
            patch.set_edgecolor("white" if hatch else "black")
            if hatch:
                patch.set_hatch("//")

        ax.set_xticks([])
        ax.set_title(scope.replace("_", " "), fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Per-class F1")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[model],
                       hatch="//" if cond_key == "desc_aug" else None,
                       edgecolor="white" if cond_key == "desc_aug" else "black",
                       label=f"{model.split('/')[-1]} ({cond_label})")
        for model in models for cond_key, cond_label in _CONDITIONS
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.1),
               ncol=min(len(models), 4), fontsize=7)
    fig.suptitle("Per-class F1 distribution — consistency across classes", fontsize=10)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    path = output_dir / "stage2_f1_percentiles.png"
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

    per_class_arrays = collect_per_class_f1(Path(args.baseline_dir), Path(args.aug_dir))
    plot_f1_percentiles(output_dir, per_class_arrays)


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--vision_baseline_config", default="src/config/vision_baseline_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vision_baseline_config", default=_pre_args.vision_baseline_config,
        help="YAML file of defaults, shared with vision_baseline.py / desc_augmented_vision_v2.py "
             "— edit it directly for routine changes instead of passing flags every time. Any "
             "flag also passed on the command line overrides this file.",
    )
    parser.add_argument("--baseline_dir", default="outputs_final/vision_baseline/")
    parser.add_argument("--aug_dir", default="outputs_final/desc_augmented_vision_v2/")
    parser.add_argument("--output_dir", default="outputs_final/vision_baseline_collated/")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CI.")
    parser.add_argument("--seed", type=int, default=42)

    # Keys this script doesn't use but the shared config file also carries
    # for vision_baseline.py / desc_augmented_vision_v2.py.
    _FOREIGN_KEYS = {
        "model", "config", "prompts", "classification_jsonl", "manifest", "images_root",
        "n_samples", "quantize", "gpu_memory_utilization", "max_model_len", "vision_token_budget",
        "max_num_seqs", "enforce_eager", "judge_model", "judge_quantize",
        "judge_confidence_threshold", "judge_chunk_size",
    }

    if _pre_args.vision_baseline_config and Path(_pre_args.vision_baseline_config).exists():
        import yaml as _yaml
        with open(_pre_args.vision_baseline_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        if "baseline_output_dir" in _cfg:
            _cfg["baseline_dir"] = _cfg.pop("baseline_output_dir")
        if "aug_output_dir" in _cfg:
            _cfg["aug_dir"] = _cfg.pop("aug_output_dir")
        if "collated_output_dir" in _cfg:
            _cfg["output_dir"] = _cfg.pop("collated_output_dir")
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests - _FOREIGN_KEYS
        if _unknown:
            print(f"[warn] {_pre_args.vision_baseline_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
