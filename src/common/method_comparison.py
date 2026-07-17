"""
Shared utilities for comparing methods (Vision Baseline, Desc-Aug, CoT, future fine-tuning).

Usage
-----
from src.common.method_comparison import plot_per_task_judge_comparison, CONDITION_STYLES

rows_by_condition = {
    "Vision Baseline": vis_rows,   # list[dict] with judged_correct column
    "Desc-Aug":        aug_rows,
    "Self-CoT":        self_cot_rows,
    "CoT YAML":        cot_yaml_rows,
}
plot_per_task_judge_comparison(rows_by_condition, output_dir / "comparison.png")

Design
------
All experiments share the same computation pipeline:
    gather_judge_per_class_by_dataset  →  {condition: {(ds, task): [per_class_f1s]}}
    gather_cls_values_by_task_from_per_class  →  {condition: {task: [ds_means]}}

This matches cls_per_task_all_errorbar.png from desc_augmented_vision exactly.
Adding a new method means adding one key to rows_by_condition — no other changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Canonical condition ordering and colors — extend here when adding new methods.
CONDITION_STYLES: list[tuple[str, str]] = [
    ("Vision Baseline", "#1f77b4"),
    ("Self-CoT",        "#ff7f0e"),
    ("CoT YAML",        "#2ca02c"),
    ("Desc-Aug",        "#9467bd"),
    ("Majority Vote",   "#e15759"),
    ("Fine-tuned",      "#17becf"),
]

CONDITION_COLOR: dict[str, str] = dict(CONDITION_STYLES)

TASK_TYPES_DEFAULT  = ["disease", "pest_damage", "weed_identification"]
TASK_LABEL: dict[str, str] = {
    "disease":             "Disease",
    "pest_damage":         "Pest Damage",
    "weed_identification": "Weed ID",
}


def compute_per_task_judge_f1(
    rows_by_condition: dict[str, list[dict]],
    task_types: list[str] = TASK_TYPES_DEFAULT,
) -> dict[str, dict[str, list[float]]]:
    """Return {condition: {task: [per_dataset_mean_judge_f1]}} using the canonical pipeline.

    Uses gather_judge_per_class_by_dataset → gather_cls_values_by_task_from_per_class,
    identical to cls_per_task_all_errorbar.png in desc_augmented_vision.
    Conditions with no judged rows are silently excluded.
    """
    from src.common.distribution_plots import (
        gather_cls_values_by_task_from_per_class,
        gather_judge_per_class_by_dataset,
    )

    judge_by_cls  = gather_judge_per_class_by_dataset(rows_by_condition)
    judge_by_task = gather_cls_values_by_task_from_per_class(judge_by_cls)
    return {
        cond: {tt: vals for tt, vals in by_task.items() if tt in task_types}
        for cond, by_task in judge_by_task.items()
        if any(by_task.get(tt) for tt in task_types)
    }


def compute_per_task_direct_f1(
    rows_by_condition: dict[str, list[dict]],
    task_types: list[str] = TASK_TYPES_DEFAULT,
    pred_key: str = "resolved_class",
) -> dict[str, dict[str, list[float]]]:
    """Return {condition: {task: [per_dataset_mean_direct_f1]}} using resolved_class.

    Uses gather_cls_per_class_by_dataset → gather_cls_values_by_task_from_per_class.
    """
    from src.common.distribution_plots import (
        gather_cls_per_class_by_dataset,
        gather_cls_values_by_task_from_per_class,
    )

    cls_by_cls  = gather_cls_per_class_by_dataset(rows_by_condition, "class_label", pred_key)
    cls_by_task = gather_cls_values_by_task_from_per_class(cls_by_cls)
    return {
        cond: {tt: vals for tt, vals in by_task.items() if tt in task_types}
        for cond, by_task in cls_by_task.items()
        if any(by_task.get(tt) for tt in task_types)
    }


def plot_baseline_vs_condition_scatter(
    rows_by_condition: dict[str, list[dict]],
    output_path: "Path",
    title: str = "Baseline vs Condition Performance (per dataset)",
    baseline_label: str = "Vision Baseline",
    condition_order: list[str] | None = None,
    separate: bool = False,
) -> None:
    """Scatter: baseline judged F1 (x) vs each other condition's judged F1 (y).

    Parameters
    ----------
    rows_by_condition:
        {condition_label: list[dict]} with judged_correct populated.
    output_path:
        Where to save the PNG.
    title:
        Chart title (used as suptitle when separate=True).
    baseline_label:
        Which condition to use as the x-axis.
    condition_order:
        Canonical order (defaults to CONDITION_STYLES order).
    separate:
        If True, one subplot per condition side-by-side.
        If False (default), all conditions overlaid on a single axes.
    """
    import matplotlib.pyplot as plt
    from src.common.distribution_plots import gather_judge_per_class_by_dataset

    output_path = Path(output_path)

    def _per_ds(rows: list[dict]) -> dict[str, float]:
        judge_by_cls = gather_judge_per_class_by_dataset({"_": rows})
        per_dt = judge_by_cls.get("_", {})
        return {ds: float(np.mean(f1s)) for (ds, _task), f1s in per_dt.items() if f1s}

    if baseline_label not in rows_by_condition:
        print(f"  [warn] plot_baseline_vs_condition_scatter: '{baseline_label}' not in rows_by_condition")
        return

    baseline_per_ds = _per_ds(rows_by_condition[baseline_label])
    if not baseline_per_ds:
        print(f"  [warn] plot_baseline_vs_condition_scatter: no judged rows for baseline")
        return

    order   = condition_order or [lbl for lbl, _ in CONDITION_STYLES]
    present = [lbl for lbl in order if lbl in rows_by_condition and lbl != baseline_label]
    if not present:
        return

    # Pre-compute per-ds F1 for each condition
    cond_data: dict[str, tuple[list, list, list]] = {}
    for cond in present:
        cond_per_ds = _per_ds(rows_by_condition[cond])
        xs, ys, names = [], [], []
        for ds, bval in baseline_per_ds.items():
            if ds not in cond_per_ds:
                continue
            xs.append(bval)
            ys.append(cond_per_ds[ds])
            names.append(ds.replace("_classification", "").replace("_", " "))
        cond_data[cond] = (xs, ys, names)

    def _draw_ax(ax, cond: str, xs, ys, names) -> None:
        color = CONDITION_COLOR.get(cond, "#888888")
        ax.scatter(xs, ys, color=color, alpha=0.8, s=60, edgecolors="black", linewidths=0.3)
        for x, y, name in zip(xs, ys, names):
            ax.annotate(name, (x, y), fontsize=5.5, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"{baseline_label} Judged F1  (← harder)", fontsize=9)
        ax.grid(True, alpha=0.2)

    if separate:
        n = len(present)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
        for i, cond in enumerate(present):
            ax = axes[0][i]
            _draw_ax(ax, cond, *cond_data[cond])
            ax.set_title(cond, fontsize=11, color=CONDITION_COLOR.get(cond, "#333333"), fontweight="bold")
            ax.set_ylabel("Judged F1" if i == 0 else "", fontsize=9)
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        for cond in present:
            xs, ys, names = cond_data[cond]
            color = CONDITION_COLOR.get(cond, "#888888")
            ax.scatter(xs, ys, label=cond, color=color, alpha=0.8, s=60,
                       edgecolors="black", linewidths=0.3)
            for x, y, name in zip(xs, ys, names):
                ax.annotate(name, (x, y), fontsize=5.5, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="no change")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"{baseline_label} Judged F1  (← harder)", fontsize=10)
        ax.set_ylabel("Condition Judged F1", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_difficulty_scatter_comparison(
    rows_by_condition: dict[str, list[dict]],
    clustering_metrics_path: "Path",
    output_path: "Path",
    title: str = "Dataset Difficulty vs Judged F1 — Method Comparison",
    backbone: str = "dinov3-vitb16-pretrain-lvd1689m",
    condition_order: list[str] | None = None,
) -> None:
    """Overlaid scatter: all conditions on the same 3 axes (one per clustering metric).

    Each condition is a different color series. A regression line per condition
    shows the trend. Replaces the per-row grid from plot_difficulty_correlation_conditions
    which was hard to compare across conditions.

    Parameters
    ----------
    rows_by_condition:
        {condition_label: list[dict]}. Rows must have judged_correct.
    clustering_metrics_path:
        Path to clustering_metrics.json.
    output_path:
        Where to save the PNG.
    title:
        Figure suptitle.
    backbone:
        Which backbone's clustering scores to use.
    condition_order:
        Canonical order (defaults to CONDITION_STYLES order).
    """
    import json
    import matplotlib.pyplot as plt
    from src.common.distribution_plots import gather_judge_per_class_by_dataset
    from src.common.difficulty_plots import _DIFF_METRICS

    clustering_metrics_path = Path(clustering_metrics_path)
    output_path = Path(output_path)

    if not clustering_metrics_path.exists():
        print(f"  [warn] plot_difficulty_scatter_comparison: clustering metrics not found")
        return

    with open(clustering_metrics_path) as f:
        clust = json.load(f)
    ds_clust = clust.get(backbone)
    if ds_clust is None:
        print(f"  [warn] plot_difficulty_scatter_comparison: backbone '{backbone}' not in clustering metrics")
        return

    # Compute per-dataset judged F1 per condition from raw rows
    order = condition_order or [lbl for lbl, _ in CONDITION_STYLES]
    cond_per_ds: dict[str, dict[str, float]] = {}
    for label, rows in rows_by_condition.items():
        judge_by_cls = gather_judge_per_class_by_dataset({"_": rows})
        per_dt = judge_by_cls.get("_", {})
        ds_means = {ds: float(np.mean(f1s)) for (ds, _task), f1s in per_dt.items() if f1s}
        if ds_means:
            cond_per_ds[label] = ds_means

    present = [lbl for lbl in order if lbl in cond_per_ds]
    if not present:
        print(f"  [warn] plot_difficulty_scatter_comparison: no judged rows — skipping")
        return

    n_cols = len(_DIFF_METRICS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), squeeze=False)

    for ci, (dm, dl) in enumerate(_DIFF_METRICS):
        ax = axes[0][ci]
        for cond in present:
            color = CONDITION_COLOR.get(cond, "#888888")
            perf  = cond_per_ds[cond]
            xs, ys = [], []
            for ds, f1 in perf.items():
                diff_val = ds_clust.get(ds, {}).get(dm)
                if diff_val is not None:
                    xs.append(diff_val)
                    ys.append(f1)

            if len(xs) < 2:
                continue

            r = float(np.corrcoef(xs, ys)[0, 1])
            ax.scatter(xs, ys, color=color, alpha=0.7, s=35,
                       edgecolors="black", linewidths=0.4, label=f"{cond} (r={r:.2f})", zorder=3)
            x_line = np.linspace(min(xs), max(xs), 100)
            m_, b_ = np.polyfit(xs, ys, 1)
            ax.plot(x_line, m_ * x_line + b_, color=color, linewidth=1.4, linestyle="--", alpha=0.8)

        ax.set_xlabel(dl, fontsize=9)
        ax.set_ylabel("Mean Judged F1" if ci == 0 else "", fontsize=9)
        ax.set_title(dl.split(" (")[0], fontsize=10)
        ax.legend(fontsize=7)
        ax.yaxis.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_overall_judge_comparison(
    rows_by_condition: dict[str, list[dict]],
    output_path: Path,
    title: str = "Method Comparison — Overall Mean Judged F1",
    condition_order: list[str] | None = None,
) -> None:
    """Single bar per condition: mean judged F1 averaged across all datasets.

    Parameters
    ----------
    rows_by_condition:
        {condition_label: list[dict]}. Rows must have agml_dataset, task_type,
        class_label, and judged_correct columns.
    output_path:
        Where to save the PNG.
    title:
        Chart title.
    condition_order:
        Canonical order for conditions (defaults to CONDITION_STYLES order).
    """
    import matplotlib.pyplot as plt
    from src.common.distribution_plots import gather_judge_per_class_by_dataset

    order = condition_order or [lbl for lbl, _ in CONDITION_STYLES]

    cond_means: dict[str, float] = {}
    for label, rows in rows_by_condition.items():
        judge_by_cls = gather_judge_per_class_by_dataset({"_": rows})
        per_dt = judge_by_cls.get("_", {})
        ds_means = [float(np.mean(f1s)) for f1s in per_dt.values() if f1s]
        if ds_means:
            cond_means[label] = float(np.mean(ds_means))

    present = [lbl for lbl in order if lbl in cond_means]
    if not present:
        print(f"  [warn] plot_overall_judge_comparison: no judged rows — skipping {output_path.name}")
        return

    means  = [cond_means[lbl] for lbl in present]
    colors = [CONDITION_COLOR.get(lbl, "#888888") for lbl in present]

    fig, ax = plt.subplots(figsize=(max(5, len(present) * 1.2), 5))
    bars = ax.bar(present, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Judged F1 (all datasets)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_task_judge_comparison(
    rows_by_condition: dict[str, list[dict]],
    output_path: Path,
    title: str = "Method Comparison — Mean Judged F1 by Task Type",
    task_types: list[str] = TASK_TYPES_DEFAULT,
    condition_order: list[str] | None = None,
) -> None:
    """Grouped bar chart: mean judged F1 per condition × task type.

    Parameters
    ----------
    rows_by_condition:
        {condition_label: list[dict]}. Rows must have agml_dataset, task_type,
        class_label, and judged_correct columns. Rows with empty judged_correct
        are skipped automatically.
    output_path:
        Where to save the PNG.
    title:
        Chart title.
    task_types:
        Which task types to include (x-axis groups), in order.
    condition_order:
        Canonical order for conditions (defaults to CONDITION_STYLES order).
    """
    import matplotlib.pyplot as plt

    judge_by_task = compute_per_task_judge_f1(rows_by_condition, task_types)
    if not judge_by_task:
        print(f"  [warn] plot_per_task_judge_comparison: no judged rows — skipping {output_path.name}")
        return

    order   = condition_order or [lbl for lbl, _ in CONDITION_STYLES]
    present = [lbl for lbl in order if lbl in judge_by_task]
    if not present:
        print(f"  [warn] plot_per_task_judge_comparison: no conditions with data — skipping")
        return

    x     = np.arange(len(task_types))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * len(present) * 0.5 + 2), 5))

    for i, cond in enumerate(present):
        by_task = judge_by_task[cond]
        vals_list = [by_task.get(tt, []) for tt in task_types]
        means  = [float(np.mean(v)) if v else 0.0 for v in vals_list]
        stds   = [float(np.std(v))  if len(v) > 1 else 0.0 for v in vals_list]
        ns     = [len(v) for v in vals_list]
        color  = CONDITION_COLOR.get(cond, "#888888")
        offset = (i - len(present) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width, label=cond,
                        color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
                        yerr=stds, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "black"})
        for bar, v, n in zip(bars, means, ns):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL.get(tt, tt) for tt in task_types], fontsize=10)
    ax.set_ylabel("Mean Judged F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_task_full_comparison(
    rows_by_condition: dict[str, list[dict]],
    output_path: Path,
    title: str = "Method Comparison — Direct vs Judged F1 by Task Type",
    task_types: list[str] = TASK_TYPES_DEFAULT,
    condition_order: list[str] | None = None,
) -> None:
    """Grouped errorbar chart showing both direct F1 (solid) and judged F1 (lighter)
    per condition × task type — mirrors cls_per_task_all_errorbar.png style.

    For each condition, two adjacent bars are drawn:
      - solid color  → direct F1 (resolved_class vs class_label)
      - lighter tint → judged F1 (judged_correct adjudicated)

    Conditions with no rows at all are skipped. Judged bars are omitted per
    condition when no judged_correct data is available.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    direct_by_task = compute_per_task_direct_f1(rows_by_condition, task_types)
    judge_by_task  = compute_per_task_judge_f1(rows_by_condition, task_types)

    order   = condition_order or [lbl for lbl, _ in CONDITION_STYLES]
    present = [lbl for lbl in order if lbl in direct_by_task or lbl in judge_by_task]
    if not present:
        print(f"  [warn] plot_per_task_full_comparison: no conditions with data — skipping")
        return

    # Each condition occupies two bar slots (direct + judged) unless judged is missing
    slots: list[tuple[str, str]] = []  # (condition_label, "direct"|"judged")
    for cond in present:
        slots.append((cond, "direct"))
        if cond in judge_by_task:
            slots.append((cond, "judged"))

    x     = np.arange(len(task_types))
    width = min(0.8 / len(slots), 0.15)
    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * len(slots) * 0.4 + 2), 5))

    def _tint(hex_color: str, factor: float = 0.5) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        return mcolors.to_hex((r + (1 - r) * factor,
                                g + (1 - g) * factor,
                                b + (1 - b) * factor))

    added_legends: set[str] = set()
    for i, (cond, kind) in enumerate(slots):
        base_color = CONDITION_COLOR.get(cond, "#888888")
        color      = base_color if kind == "direct" else _tint(base_color, 0.55)
        by_task    = direct_by_task.get(cond, {}) if kind == "direct" else judge_by_task.get(cond, {})
        vals_list  = [by_task.get(tt, []) for tt in task_types]
        means      = [float(np.mean(v)) if v else 0.0 for v in vals_list]
        stds       = [float(np.std(v))  if len(v) > 1 else 0.0 for v in vals_list]
        offset     = (i - len(slots) / 2 + 0.5) * width

        legend_label = f"{cond} (judged)" if kind == "judged" else cond
        show_legend  = legend_label not in added_legends
        added_legends.add(legend_label)

        hatch = "//" if kind == "judged" else None
        bars  = ax.bar(x + offset, means, width, label=legend_label if show_legend else "_",
                       color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
                       hatch=hatch,
                       yerr=stds, capsize=2, error_kw={"elinewidth": 1.0, "ecolor": "black"})
        for bar, v in zip(bars, means):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL.get(tt, tt) for tt in task_types], fontsize=10)
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
