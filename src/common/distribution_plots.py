"""
Shared strip-plot and error-bar renderers for assessment and clustering charts.

All public functions return nothing; they save a PNG to `output_path`.

Data conventions
----------------
values_by_model   : {model_key: [float, ...]}          — one value per dataset
values_by_group   : {model_key: {group_key: [float, ...]}}  — values per (model, group)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score as _sk_f1


# ── Low-level renderers ────────────────────────────────────────────────────────

def _jitter(n: int, width: float = 0.12, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _render_violin(
    ax,
    x_positions: np.ndarray,
    values_list: list[list[float]],
    colors: list,
    labels: list[str],
    width_per_group: float,
    hatches: list[str],
) -> None:
    """Violin per model per x position; falls back to a single diamond for n<3."""
    n_models = len(values_list)
    grouped = len(x_positions) == 1  # multiple models sharing one x tick
    slot = width_per_group / max(n_models, 1)
    vw = slot * 0.85 if grouped else width_per_group * 0.7

    for i, (vals, color, label, hatch) in enumerate(zip(values_list, colors, labels, hatches)):
        if not vals:
            continue
        offset = (i - (n_models - 1) / 2) * slot
        center = float(x_positions[0]) + offset if grouped else float(x_positions[i])

        if len(vals) < 3:
            ax.scatter([center], [float(np.mean(vals))], color=color, s=50, zorder=4,
                       marker="D", edgecolors="black", linewidths=0.5, label=label)
            continue

        parts = ax.violinplot(vals, positions=[center], widths=vw,
                              showmeans=True, showmedians=False, showextrema=True)
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.72)
        body.set_linewidth(0.5)
        if hatch:
            body.set_hatch(hatch)
        body.set_label(label)
        for part in ("cbars", "cmins", "cmaxes", "cmeans"):
            if part in parts:
                parts[part].set_color(color)
                parts[part].set_linewidth(1.2)


def _render_errorbar(
    ax,
    x_positions: np.ndarray,
    values_list: list[list[float]],
    colors: list,
    labels: list[str],
    bar_width: float,
    hatches: list[str],
) -> None:
    """Bar = mean, error = ±1 std."""
    n_models = len(values_list)
    w = bar_width / max(n_models, 1)
    for i, (vals, color, label, hatch) in enumerate(zip(values_list, colors, labels, hatches)):
        if not vals:
            continue
        offset = (i - (n_models - 1) / 2) * w
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        ax.bar(x_positions + offset, mean, w, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.5, label=label, hatch=hatch)
        ax.errorbar(x_positions + offset, mean, yerr=std,
                    fmt="none", color="black", capsize=3, linewidth=1.2, zorder=4)


# ── Data gathering helpers ─────────────────────────────────────────────────────

def gather_cls_values_by_dataset(
    all_cls_results: dict[str, dict[str, dict]],
    metric: str = "f1_macro",
) -> dict[str, dict[str, float]]:
    """Returns {model: {dataset: value}}."""
    return {
        model: {ds: m.get(metric, 0.0) for ds, m in per_ds.items()}
        for model, per_ds in all_cls_results.items()
    }


def gather_cls_values_by_task(
    all_cls_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    metric: str = "f1_macro",
) -> dict[str, dict[str, list[float]]]:
    """Returns {model: {task: [value per dataset in task]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for model, per_ds in all_cls_results.items():
        by_task: dict[str, list[float]] = {}
        for ds, m in per_ds.items():
            task = ds_to_task.get(ds)
            if task:
                by_task.setdefault(task, []).append(m.get(metric, 0.0))
        out[model] = by_task
    return out


def gather_cls_per_class_by_dataset(
    all_cls_rows: dict[str, list[dict]],
    ref_key: str = "class_label",
    pred_key: str = "resolved_class",
) -> dict[str, dict[str, list[float]]]:
    """Returns {model: {dataset: [per-class F1 values]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for model, rows in all_cls_rows.items():
        by_ds: dict[str, list[dict]] = {}
        for r in rows:
            by_ds.setdefault(r.get("agml_dataset", ""), []).append(r)
        per_ds: dict[str, list[float]] = {}
        for ds, ds_rows in by_ds.items():
            if not ds:
                continue
            refs  = [r[ref_key]  for r in ds_rows if ref_key  in r and pred_key in r]
            preds = [r[pred_key] for r in ds_rows if ref_key  in r and pred_key in r]
            if not refs:
                continue
            labels = sorted(set(refs))
            scores = _sk_f1(refs, preds, average=None, labels=labels, zero_division=0)
            per_ds[ds] = [float(s) for s in scores]
        out[model] = per_ds
    return out


def gather_desc_values_by_dataset(
    all_desc_results: dict[str, dict[str, dict]],
    metric: str = "sem_sim",
) -> dict[str, dict[str, float]]:
    """Returns {model: {dataset: value}}."""
    return {
        model: {ds: m.get(metric, 0.0) for ds, m in per_ds.items()}
        for model, per_ds in all_desc_results.items()
    }


def gather_desc_values_by_task(
    all_desc_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    metric: str = "sem_sim",
) -> dict[str, dict[str, list[float]]]:
    """Returns {model: {task: [value per dataset in task]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for model, per_ds in all_desc_results.items():
        by_task: dict[str, list[float]] = {}
        for ds, m in per_ds.items():
            task = ds_to_task.get(ds)
            if task:
                by_task.setdefault(task, []).append(m.get(metric, 0.0))
        out[model] = by_task
    return out


def gather_desc_per_sample_by_dataset(
    all_desc_rows: dict[str, list[dict]],
    metric: str = "sem_sim",
) -> dict[str, dict[str, list[float]]]:
    """Returns {model: {dataset: [per-sample metric values]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for model, rows in all_desc_rows.items():
        by_ds: dict[str, list[float]] = {}
        for r in rows:
            ds = r.get("agml_dataset", "")
            v  = r.get(metric)
            if ds and v is not None:
                by_ds.setdefault(ds, []).append(float(v))
        out[model] = by_ds
    return out


def gather_judge_values_by_dataset(
    judge_by_dataset: dict[str, dict[str, dict]],
    metric: str = "judge_f1_macro",
) -> dict[str, dict[str, float]]:
    """Returns {model: {dataset: judge_f1_macro}}."""
    return {
        model: {ds: m.get(metric, 0.0) for ds, m in per_ds.items() if metric in m}
        for model, per_ds in judge_by_dataset.items()
    }


def gather_judge_values_by_task(
    judge_by_dataset: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    metric: str = "judge_f1_macro",
) -> dict[str, dict[str, list[float]]]:
    """Returns {model: {task: [judge_f1_macro per dataset in task]}}."""
    out: dict[str, dict[str, list[float]]] = {}
    for model, per_ds in judge_by_dataset.items():
        by_task: dict[str, list[float]] = {}
        for ds, m in per_ds.items():
            task = ds_to_task.get(ds)
            if task and metric in m:
                by_task.setdefault(task, []).append(m[metric])
        out[model] = by_task
    return out


# ── Public plot functions ──────────────────────────────────────────────────────

def plot_overall_distribution(
    values_by_model: dict[str, dict[str, float]],
    models: list[str],
    color_map: dict[str, tuple],
    multimodal_models: set[str],
    metric_label: str,
    title: str,
    output_path: Path,
    chart_type: str = "strip",
    naive_line: float | None = None,
    judge_values_by_model: dict[str, dict[str, float]] | None = None,
) -> None:
    """Strip or errorbar: x = models, distribution = per-dataset values.
    If judge_values_by_model provided, renders judge as a second faded series."""
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 5))

    values_list = [list(values_by_model.get(m, {}).values()) for m in models]
    labels      = [m.split("/")[-1] for m in models]
    colors      = [color_map[m] for m in models]
    hatches     = ["///" if m in multimodal_models else "" for m in models]

    judge_list = [list(judge_values_by_model.get(m, {}).values()) for m in models] \
        if judge_values_by_model else []

    if chart_type == "strip":
        _render_violin(ax, x, values_list, colors, labels, width_per_group=0.6, hatches=hatches)
        judge_drawn = False
        for xi, (jvals, color) in enumerate(zip(judge_list, colors)):
            if jvals:
                jxs = np.full(len(jvals), xi) + _jitter(len(jvals), width=0.08)
                ax.scatter(jxs, jvals, color=color, alpha=0.35, s=22, marker="x",
                           linewidths=1.2, zorder=3,
                           label="judge" if not judge_drawn else "_nolegend_")
                judge_drawn = True
    else:
        if judge_list:
            slot_w = 0.7 / max(len(models), 1)
            sub_w  = slot_w * 0.46
            judge_drawn = False
            for i, (vals, jvals, color, label, hatch) in enumerate(
                    zip(values_list, judge_list, colors, labels, hatches)):
                offset = (i - (len(models) - 1) / 2) * slot_w
                if vals:
                    mean, std = float(np.mean(vals)), float(np.std(vals))
                    ax.bar(x + offset - sub_w / 2, mean, sub_w, color=color, alpha=0.85,
                           edgecolor="black", linewidth=0.5, label=label, hatch=hatch)
                    ax.errorbar(x + offset - sub_w / 2, mean, yerr=std, fmt="none",
                                color="black", capsize=3, linewidth=1.2, zorder=4)
                if jvals:
                    jmean, jstd = float(np.mean(jvals)), float(np.std(jvals))
                    ax.bar(x + offset + sub_w / 2, jmean, sub_w, color=color, alpha=0.4,
                           edgecolor="black", linewidth=0.5,
                           label="judge" if not judge_drawn else "_nolegend_")
                    ax.errorbar(x + offset + sub_w / 2, jmean, yerr=jstd, fmt="none",
                                color="black", capsize=3, linewidth=1.2, zorder=4)
                    judge_drawn = True
        else:
            _render_errorbar(ax, x, values_list, colors, labels, bar_width=0.7, hatches=hatches)

    if naive_line is not None:
        ax.axhline(naive_line, color="red", linestyle="--", linewidth=1.2,
                   label="naive (1/k classes)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=11)
    handles, lbls = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_task_distribution(
    values_by_model_by_task: dict[str, dict[str, list[float]]],
    task_types: list[str],
    models: list[str],
    color_map: dict[str, tuple],
    multimodal_models: set[str],
    task_label_fn,
    metric_label: str,
    title: str,
    output_path: Path,
    chart_type: str = "strip",
    naive_line: float | None = None,
    judge_values_by_model_by_task: dict[str, dict[str, list[float]]] | None = None,
) -> None:
    """Strip or errorbar: x = task types, distribution = per-dataset values within task."""
    x = np.arange(len(task_types))
    subplot_w = max(5, len(task_types) * (len(models) * 0.5 + 0.4))
    fig, ax = plt.subplots(figsize=(subplot_w, 5))

    colors  = [color_map[m] for m in models]
    labels  = [m.split("/")[-1] for m in models]
    hatches = ["///" if m in multimodal_models else "" for m in models]

    judge_drawn = False
    if chart_type == "strip":
        for xi, task in enumerate(task_types):
            vals_list = [values_by_model_by_task.get(m, {}).get(task, []) for m in models]
            _render_violin(ax, np.array([xi]), vals_list, colors,
                          [l if xi == 0 else "_nolegend_" for l in labels],
                          width_per_group=0.7, hatches=hatches)
            if judge_values_by_model_by_task:
                for mi, (m, color) in enumerate(zip(models, colors)):
                    jvals = judge_values_by_model_by_task.get(m, {}).get(task, [])
                    if jvals:
                        jxs = np.full(len(jvals), xi) + _jitter(len(jvals), width=0.07)
                        ax.scatter(jxs, jvals, color=color, alpha=0.35, s=22, marker="x",
                                   linewidths=1.2, zorder=3,
                                   label="judge" if not judge_drawn else "_nolegend_")
                        judge_drawn = True
    else:
        for xi, task in enumerate(task_types):
            vals_list = [values_by_model_by_task.get(m, {}).get(task, []) for m in models]
            _render_errorbar(ax, np.array([xi]), vals_list, colors,
                             [l if xi == 0 else "_nolegend_" for l in labels],
                             bar_width=0.8, hatches=hatches)
            if judge_values_by_model_by_task:
                slot_w = 0.8 / max(len(models), 1)
                sub_w  = slot_w * 0.46
                for mi, (m, color) in enumerate(zip(models, colors)):
                    jvals = judge_values_by_model_by_task.get(m, {}).get(task, [])
                    if not jvals:
                        continue
                    offset = (mi - (len(models) - 1) / 2) * slot_w
                    mean = float(np.mean(jvals))
                    std  = float(np.std(jvals))
                    ax.bar(xi + offset + sub_w / 2, mean, sub_w, color=color, alpha=0.4,
                           edgecolor="black", linewidth=0.5,
                           label="judge" if not judge_drawn else "_nolegend_")
                    ax.errorbar(xi + offset + sub_w / 2, mean, yerr=std, fmt="none",
                                color="black", capsize=3, linewidth=1.2, zorder=4)
                    judge_drawn = True

    if naive_line is not None:
        ax.axhline(naive_line, color="red", linestyle="--", linewidth=1.2,
                   label="naive (1/k classes)")

    ax.set_xticks(x)
    ax.set_xticklabels([task_label_fn(t) for t in task_types], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=11)
    handles, lbls = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_dataset_distribution(
    values_by_model_by_dataset: dict[str, dict[str, list[float]]],
    datasets: list[str],
    models: list[str],
    color_map: dict[str, tuple],
    multimodal_models: set[str],
    metric_label: str,
    title: str,
    output_path: Path,
    chart_type: str = "strip",
    naive_line: float | None = None,
) -> None:
    """Strip or errorbar: x = datasets, distribution = per-class/per-sample values."""
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * len(models) * 0.45), 5))

    colors  = [color_map[m] for m in models]
    labels  = [m.split("/")[-1] for m in models]
    hatches = ["///" if m in multimodal_models else "" for m in models]

    if chart_type == "strip":
        for xi, ds in enumerate(datasets):
            vals_list = [values_by_model_by_dataset.get(m, {}).get(ds, []) for m in models]
            _render_violin(ax, np.array([xi]), vals_list, colors,
                          [l if xi == 0 else "_nolegend_" for l in labels],
                          width_per_group=0.7, hatches=hatches)
    else:
        for xi, ds in enumerate(datasets):
            vals_list = [values_by_model_by_dataset.get(m, {}).get(ds, []) for m in models]
            _render_errorbar(ax, np.array([xi]), vals_list, colors,
                             [l if xi == 0 else "_nolegend_" for l in labels],
                             bar_width=0.8, hatches=hatches)

    if naive_line is not None:
        ax.axhline(naive_line, color="red", linestyle="--", linewidth=1.2,
                   label="naive (1/k classes)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_classification", "").replace("_", " ") for d in datasets],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=11)
    handles, lbls = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_clustering_overall_distribution(
    results: dict[str, dict[str, dict]],
    backbones: list[str],
    metrics: list[str],
    metric_labels: dict[str, str],
    colors: dict[str, tuple],
    display_fn,
    title: str,
    output_path: Path,
    chart_type: str = "strip",
) -> None:
    """Strip or errorbar for clustering overall: x = metrics, distribution = per-dataset values."""
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * len(backbones) * 0.9), 5))

    b_colors = [colors[b] for b in backbones]
    b_labels = [display_fn(b) for b in backbones]
    hatches  = [""] * len(backbones)

    for mi, metric in enumerate(metrics):
        xi = np.array([mi])
        vals_list = [
            [m[metric] for m in results.get(b, {}).values() if metric in m]
            for b in backbones
        ]
        labels_i = [l if mi == 0 else "_nolegend_" for l in b_labels]
        if chart_type == "strip":
            _render_violin(ax, xi, vals_list, b_colors, labels_i,
                          width_per_group=0.7, hatches=hatches)
        else:
            _render_errorbar(ax, xi, vals_list, b_colors, labels_i,
                             bar_width=0.8, hatches=hatches)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=9)
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    handles, lbls = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_clustering_per_task_distribution(
    results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    backbones: list[str],
    metrics: list[str],
    metric_labels: dict[str, str],
    colors: dict[str, tuple],
    display_fn,
    title: str,
    output_path: Path,
    chart_type: str = "strip",
) -> None:
    """Strip or errorbar for clustering per-task: one panel per metric, x = task types."""
    from collections import defaultdict
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    x = np.arange(len(task_types))
    subplot_w = max(4, len(task_types) * len(backbones) * 0.7 + 2)
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    b_colors = [colors[b] for b in backbones]
    b_labels = [display_fn(b) for b in backbones]
    hatches  = [""] * len(backbones)

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        by_task_per_backbone: list[dict[str, list[float]]] = []
        for b in backbones:
            by_task: dict[str, list[float]] = defaultdict(list)
            for ds, m in results.get(b, {}).items():
                tt = ds_to_task.get(ds)
                if tt and metric in m:
                    by_task[tt].append(m[metric])
            by_task_per_backbone.append(by_task)

        for xi, task in enumerate(task_types):
            vals_list = [bt.get(task, []) for bt in by_task_per_backbone]
            labels_i  = [l if xi == 0 else "_nolegend_" for l in b_labels]
            if chart_type == "strip":
                _render_violin(ax, np.array([xi]), vals_list, b_colors, labels_i,
                              width_per_group=0.7, hatches=hatches)
            else:
                _render_errorbar(ax, np.array([xi]), vals_list, b_colors, labels_i,
                                 bar_width=0.8, hatches=hatches)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ").title() for t in task_types],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(metric_labels[metric])
        ax.set_ylabel("Score")
        handles, lbls = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, lbls):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        ax.legend(h2, l2, fontsize=6)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
