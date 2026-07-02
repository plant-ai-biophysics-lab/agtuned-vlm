"""Shared difficulty-correlation scatter plots used by multiple stages."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_DIFF_METRICS = [
    ("silhouette",          "Silhouette (↑ easier)"),
    ("inter_centroid_dist", "Inter-Centroid Dist (↑ easier)"),
    ("fisher_dr",           "Fisher DR (↑ easier)"),
]


def plot_difficulty_correlation_conditions(
    conditions_per_ds: dict[str, dict[str, float]],
    clustering_metrics_path: Path,
    output_dir: Path,
    backbone: str = "dinov3-vitb16-pretrain-lvd1689m",
    filename: str = "difficulty_vs_perf_conditions.png",
    title: str = "Dataset Difficulty vs Judged F1",
) -> None:
    """Scatter grid: clustering difficulty (cols) vs judged F1 per condition (rows).

    Args:
        conditions_per_ds: {condition_label: {dataset_name: judge_f1_macro}}
        clustering_metrics_path: path to clustering_metrics.json
        output_dir: where to save the PNG
        backbone: which backbone's clustering scores to use
        filename: output filename
        title: figure suptitle
    """
    if not clustering_metrics_path.exists():
        print(f"  [warn] Clustering metrics not found: {clustering_metrics_path}")
        return

    with open(clustering_metrics_path) as f:
        clust = json.load(f)

    ds_clust = clust.get(backbone)
    if ds_clust is None:
        print(f"  [warn] Backbone '{backbone}' not in clustering metrics")
        return

    conditions = list(conditions_per_ds.keys())
    n_rows = len(conditions)
    n_cols = len(_DIFF_METRICS)

    condition_colors = [
        "#e67e22",  # orange  — baseline / first condition
        "#2ca02c",  # green
        "#1f77b4",  # blue
        "#9467bd",  # purple
        "#d62728",  # red
    ]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for ri, cond in enumerate(conditions):
        perf = conditions_per_ds[cond]  # {ds: judge_f1}
        color = condition_colors[ri % len(condition_colors)]

        for ci, (dm, dl) in enumerate(_DIFF_METRICS):
            ax = axes[ri][ci]
            xs, ys, names = [], [], []
            for ds, f1 in perf.items():
                diff_val = ds_clust.get(ds, {}).get(dm)
                if diff_val is not None and f1 is not None:
                    xs.append(diff_val)
                    ys.append(f1)
                    names.append(ds.replace("_classification", "").replace("_", " "))

            if len(xs) < 2:
                ax.set_visible(False)
                continue

            r = float(np.corrcoef(xs, ys)[0, 1])
            ax.scatter(xs, ys, color=color, alpha=0.75, s=40,
                       edgecolors="black", linewidths=0.4)
            for x, y, name in zip(xs, ys, names):
                ax.annotate(name, (x, y), fontsize=5, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

            x_line = np.linspace(min(xs), max(xs), 100)
            m_, b_ = np.polyfit(xs, ys, 1)
            ax.plot(x_line, m_ * x_line + b_, color="red", linewidth=1.2,
                    label=f"r={r:.2f}")

            # F1>0 trendline
            nz = [(x, y) for x, y in zip(xs, ys) if y > 0]
            if len(nz) >= 2:
                nzx, nzy = zip(*nz)
                r_nz = float(np.corrcoef(nzx, nzy)[0, 1])
                m2, b2 = np.polyfit(nzx, nzy, 1)
                x2 = np.linspace(min(nzx), max(nzx), 100)
                ax.plot(x2, m2 * x2 + b2, color="orange", linewidth=1.2,
                        linestyle="--", label=f"F1>0 (r={r_nz:.2f})")

            ax.legend(fontsize=6)
            ax.set_xlabel(dl, fontsize=8)
            ax.set_ylabel("Judged F1", fontsize=8)
            ax.set_title(f"{dl.split(' (')[0]} vs Judged F1", fontsize=9)

            # Row label on leftmost column
            if ci == 0:
                ax.set_ylabel(f"{cond}\n\nJudged F1", fontsize=8)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
