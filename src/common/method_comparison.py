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
