"""
Clustering Evaluation — unsupervised quality of frozen visual features.

For each backbone × dataset found in the NPZ embedding cache:
  1. Load frozen embeddings (no labels used for clustering)
  2. Reduce to 50 dims via PCA (stabilises k-means on high-dim VLM features)
  3. Run k-means with k = n_true_classes, n_init=20
  4. Score against ground-truth labels:
       NMI  — normalised mutual information  (0=random, 1=perfect)
       ARI  — adjusted rand index            (0=random, 1=perfect, <0=worse than random)
       Sil  — silhouette score on raw feats  (−1 to 1, measures compactness vs separation)

Labels are NEVER used for training — only for scoring.

Usage:
    python main.py cluster \
      --cache_dir  outputs/linear_probe/embedding_cache/ \
      --output_dir outputs/clustering_eval/ \
      --config     src/config/datasets.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import re
from collections import defaultdict

from src.common.distribution_plots import (
    plot_clustering_overall_distribution,
    plot_clustering_per_task_distribution,
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

from src.common.dataset_config import datasets_by_name
from src.common.io import save_json

# ── Backbone display names ─────────────────────────────────────────────────────

_BACKBONE_DISPLAY: dict[str, str] = {}

def _display(backbone_key: str) -> str:
    if backbone_key in _BACKBONE_DISPLAY:
        return _BACKBONE_DISPLAY[backbone_key]
    k = backbone_key
    # postproj suffix
    if k.endswith("_postproj"):
        base = k[: -len("_postproj")]
        label = f"{base.split('/')[-1].split('--')[-1]} (post-proj)"
    elif "__" in k:
        label = k.split("__")[0]
    else:
        short = k.replace("--", "/").split("/")[-1]
        label = short
    _BACKBONE_DISPLAY[backbone_key] = label
    return label


# ── NPZ discovery ──────────────────────────────────────────────────────────────

_NPZ_RE = re.compile(r"^(.+?)__(.+?)__(emb|pre|post)__(.+?)\.npz$")


def discover_cache(cache_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {backbone_key: {dataset_name: npz_path}}."""
    result: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in sorted(cache_dir.glob("*.npz")):
        m = _NPZ_RE.match(p.name)
        if not m:
            continue
        backbone, dataset = m.group(1), m.group(2)
        result[backbone][dataset] = p
    return dict(result)


# ── Clustering ─────────────────────────────────────────────────────────────────

_PCA_DIMS = 50       # reduce before k-means
_KMEANS_INIT = 20    # restarts for stability
_SILHOUETTE_MAX = 5_000  # subsample for silhouette (O(n²) otherwise)
_MINIBATCH_THRESHOLD = 20_000  # use MiniBatchKMeans above this


def _pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    if X.shape[1] <= n_components:
        return X
    # randomized SVD is orders of magnitude faster than full SVD on large n
    return PCA(n_components=n_components, svd_solver="randomized", random_state=42).fit_transform(X)


def _inter_centroid_dist(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean pairwise cosine distance between per-class centroids, normalised to [0, 1].
    Cosine distance ∈ [0, 2], so divide by 2 to keep the same scale as silhouette."""
    centroids = np.array([X[y == i].mean(axis=0) for i in range(k)])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_n = centroids / (norms + 1e-10)
    cos_sim = centroids_n @ centroids_n.T
    triu = np.triu_indices(k, k=1)
    raw = float(np.mean(1.0 - cos_sim[triu])) if len(triu[0]) > 0 else 0.0
    return raw / 2.0


def _fisher_dr(X: np.ndarray, y: np.ndarray, k: int, n: int) -> float:
    """Simplified Fisher's discriminant ratio: between-class / within-class scatter."""
    mu = X.mean(axis=0)
    between = sum(
        np.sum(y == i) * float(np.linalg.norm(X[y == i].mean(axis=0) - mu) ** 2)
        for i in range(k)
    ) / n
    within = sum(
        float(np.mean(np.sum((X[y == i] - X[y == i].mean(axis=0)) ** 2, axis=1)))
        * np.sum(y == i)
        for i in range(k)
    ) / n
    return between / (within + 1e-10)


def cluster_dataset(embeddings: np.ndarray, labels: list[str]) -> dict:
    """Run k-means and return NMI, ARI, silhouette + inter-centroid dist + Fisher DR."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    k = len(le.classes_)
    n = len(y)

    if k < 2 or n < k:
        return {}

    with tqdm(total=5, desc="    pca", leave=False, ncols=60) as pbar:
        X_red = _pca_reduce(embeddings.astype(np.float32), _PCA_DIMS)
        pbar.update(1)

        pbar.set_description("    kmeans")
        KM = MiniBatchKMeans if n > _MINIBATCH_THRESHOLD else KMeans
        km = KM(n_clusters=k, n_init=_KMEANS_INIT, random_state=42)
        pred = km.fit_predict(X_red)
        pbar.update(1)

        pbar.set_description("    silhouette")
        nmi = float(normalized_mutual_info_score(y, pred, average_method="arithmetic"))
        ari = float(adjusted_rand_score(y, pred))
        if n > _SILHOUETTE_MAX:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, _SILHOUETTE_MAX, replace=False)
            sil = float(silhouette_score(X_red[idx], y[idx]))
        else:
            sil = float(silhouette_score(X_red, y))
        pbar.update(1)

        pbar.set_description("    inter-centroid")
        icd = _inter_centroid_dist(X_red, y, k)
        pbar.update(1)

        pbar.set_description("    fisher")
        fdr = _fisher_dr(X_red, y, k, n)
        pbar.update(1)

    return {
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "silhouette": round(sil, 4),
        "inter_centroid_dist": round(icd, 4),
        "fisher_dr": round(min(fdr, 1e6), 4),
        "n_samples": n,
        "n_classes": k,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

_METRIC_GROUPS = {
    "unsupervised": ["nmi", "ari"],
    "separability": ["silhouette", "inter_centroid_dist", "fisher_dr"],
}
_METRICS = ["nmi", "ari", "silhouette", "inter_centroid_dist", "fisher_dr"]
_METRIC_LABELS = {
    "nmi": "NMI", "ari": "ARI", "silhouette": "Silhouette",
    "inter_centroid_dist": "Inter-Centroid Dist (norm)", "fisher_dr": "Fisher DR",
}
_GROUP_TITLES = {
    "unsupervised": "Unsupervised Recovery (k-means vs ground truth)",
    "separability": "Geometric Separability (ground-truth class boundaries)",
}
_COLORS = plt.cm.tab10.colors


def _backbone_color(backbones: list[str]) -> dict[str, tuple]:
    return {b: _COLORS[i % len(_COLORS)] for i, b in enumerate(backbones)}


def _plot_overall_group(results: dict, metrics: list[str], title: str, path: Path) -> None:
    backbones = sorted(results)
    colors = _backbone_color(backbones)
    x = np.arange(len(metrics))
    width = 0.8 / len(backbones)

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * len(backbones) * 0.9), 5))
    for i, b in enumerate(backbones):
        ds_metrics = results[b]
        vals = [
            float(np.mean([m[metric] for m in ds_metrics.values() if metric in m]))
            if ds_metrics else 0.0
            for metric in metrics
        ]
        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=_display(b),
                      color=colors[b], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(f"Clustering Quality — {title} (mean across datasets)", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall(results: dict, output_dir: Path) -> None:
    for group, metrics in _METRIC_GROUPS.items():
        _plot_overall_group(results, metrics, _GROUP_TITLES[group],
                            output_dir / f"clustering_overall_{group}.png")


def _plot_per_task_group(results: dict, ds_to_task: dict[str, str],
                         metrics: list[str], title: str, path: Path) -> None:
    backbones = sorted(results)
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return
    colors = _backbone_color(backbones)
    x = np.arange(len(task_types))
    width = 0.8 / len(backbones)
    subplot_w = max(4, len(task_types) * len(backbones) * 0.7 + 2)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, b in enumerate(backbones):
            by_task: dict[str, list[float]] = defaultdict(list)
            for ds, m in results[b].items():
                tt = ds_to_task.get(ds)
                if tt and metric in m:
                    by_task[tt].append(m[metric])
            vals = [float(np.mean(by_task[tt])) if by_task[tt] else 0.0 for tt in task_types]
            offset = (i - (len(backbones) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=_display(b),
                          color=colors[b], alpha=0.85, edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([tt.replace("_", " ").title() for tt in task_types],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_ylabel("Score")
        ax.legend(fontsize=6)
    fig.suptitle(f"Clustering Quality — {title} — By Task Type", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_task(results: dict, ds_to_task: dict[str, str], output_dir: Path) -> None:
    for group, metrics in _METRIC_GROUPS.items():
        _plot_per_task_group(results, ds_to_task, metrics, _GROUP_TITLES[group],
                             output_dir / f"clustering_per_task_{group}.png")


def plot_per_dataset_bars(results: dict, output_dir: Path) -> None:
    """Grouped bar charts: one subplot per metric, x-axis = datasets, bars = backbones."""
    backbones = sorted(results)
    all_datasets = sorted(set(ds for b in backbones for ds in results[b]))
    if not all_datasets or not backbones:
        return
    colors = _backbone_color(backbones)
    ds_labels = [d.replace("_classification", "").replace("_", " ") for d in all_datasets]
    x = np.arange(len(all_datasets))
    width = 0.8 / len(backbones)

    for group, metrics in _METRIC_GROUPS.items():
        fig, axes = plt.subplots(len(metrics), 1,
                                 figsize=(max(14, len(all_datasets) * len(backbones) * 0.35),
                                          4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for mi, metric in enumerate(metrics):
            ax = axes[mi]
            for i, b in enumerate(backbones):
                vals = [results[b].get(ds, {}).get(metric, 0.0) for ds in all_datasets]
                offset = (i - (len(backbones) - 1) / 2) * width
                ax.bar(x + offset, vals, width, label=_display(b),
                       color=colors[b], alpha=0.85, edgecolor="black", linewidth=0.4)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(-0.15, 1.05)
            ax.set_ylabel(_METRIC_LABELS[metric])
            ax.legend(fontsize=6)
        fig.suptitle(f"Clustering Quality — {_GROUP_TITLES[group]} — Per Dataset", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"clustering_per_dataset_{group}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_per_dataset(results: dict, output_dir: Path) -> None:
    """Heatmap: backbone × dataset for each metric."""
    backbones = sorted(results)
    all_datasets = sorted(set(ds for b in backbones for ds in results[b]))
    if not all_datasets or not backbones:
        return

    ds_labels = [d.replace("_classification", "").replace("_", " ") for d in all_datasets]

    for metric in _METRICS:
        matrix = np.full((len(backbones), len(all_datasets)), np.nan)
        for bi, b in enumerate(backbones):
            for di, ds in enumerate(all_datasets):
                v = results[b].get(ds, {}).get(metric)
                if v is not None:
                    matrix[bi, di] = v

        fig, ax = plt.subplots(figsize=(max(14, len(all_datasets) * 0.7 + 2),
                                        max(4, len(backbones) * 0.6 + 1.5)))
        vmin = -0.1 if metric in ("ari",) else 0.0
        cmap = "RdYlGn"
        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=1.0, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        for bi in range(len(backbones)):
            for di in range(len(all_datasets)):
                v = matrix[bi, di]
                if not np.isnan(v):
                    color = "black" if 0.2 < v < 0.8 else "white"
                    ax.text(di, bi, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        ax.set_xticks(range(len(all_datasets)))
        ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(backbones)))
        ax.set_yticklabels([_display(b) for b in backbones], fontsize=8)
        ax.set_title(f"Clustering {_METRIC_LABELS[metric]} — Per Dataset", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"clustering_heatmap_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_metric_scatter(results: dict, output_dir: Path) -> None:
    """Scatter NMI vs Silhouette per (backbone, dataset), coloured by backbone."""
    backbones = sorted(results)
    colors = _backbone_color(backbones)
    fig, ax = plt.subplots(figsize=(7, 6))
    for b in backbones:
        nmis = [m["nmi"] for m in results[b].values() if "nmi" in m and "silhouette" in m]
        sils = [m["silhouette"] for m in results[b].values() if "nmi" in m and "silhouette" in m]
        if nmis:
            ax.scatter(nmis, sils, label=_display(b), color=colors[b],
                       alpha=0.75, s=50, edgecolors="black", linewidths=0.4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("NMI")
    ax.set_ylabel("Silhouette")
    ax.set_title("NMI vs Silhouette per (backbone, dataset)", fontsize=10)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = output_dir / "clustering_scatter_nmi_vs_sil.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cache_dir  = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        print(f"ERROR: cache_dir not found: {cache_dir}")
        print("Run 'python main.py probe --cache_embeddings ...' first.")
        raise SystemExit(1)

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task   = {k: v.task_type for k, v in datasets_cfg.items()}

    cache_index = discover_cache(cache_dir)
    if not cache_index:
        print(f"No .npz files found in {cache_dir}")
        raise SystemExit(1)

    print(f"Found {len(cache_index)} backbone(s) across "
          f"{sum(len(v) for v in cache_index.values())} (backbone, dataset) pairs\n")

    # backbone → dataset → metrics dict
    results: dict[str, dict[str, dict]] = {}

    for backbone, ds_paths in sorted(cache_index.items()):
        print(f"Backbone: {_display(backbone)}")
        results[backbone] = {}
        for dataset, npz_path in sorted(ds_paths.items()):
            d = np.load(npz_path, allow_pickle=True)
            embeddings = d["embeddings"]
            labels     = list(d["labels"])
            if args.max_samples and len(labels) > args.max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(labels), args.max_samples, replace=False)
                idx.sort()
                embeddings = embeddings[idx]
                labels     = [labels[i] for i in idx]
            n_classes  = len(set(labels))
            print(f"  {dataset:<55} n={len(labels):>5}  k={n_classes}", end=" ", flush=True)
            m = cluster_dataset(embeddings, labels)
            if m:
                results[backbone][dataset] = m
                print(f"  NMI={m['nmi']:.3f}  ARI={m['ari']:.3f}  Sil={m['silhouette']:.3f}")
            else:
                print("  skipped (k<2 or n<k)")
        print()

    # Save JSON
    save_json(output_dir / "clustering_metrics.json", results)
    print(f"Metrics saved to {output_dir / 'clustering_metrics.json'}\n")

    # Summary table
    print(f"{'Backbone':<45} {'NMI':>6} {'ARI':>6} {'Sil':>6} {'ICD':>6} {'FDR':>8}")
    print("-" * 80)
    for b in sorted(results):
        ms = [m for m in results[b].values() if m]
        if not ms:
            continue
        nmi = np.mean([m["nmi"] for m in ms])
        ari = np.mean([m["ari"] for m in ms])
        sil = np.mean([m["silhouette"] for m in ms])
        icd = np.mean([m.get("inter_centroid_dist", 0) for m in ms])
        fdr = np.mean([m.get("fisher_dr", 0) for m in ms])
        print(f"  {_display(b):<43} {nmi:>6.3f} {ari:>6.3f} {sil:>6.3f} {icd:>6.3f} {fdr:>8.3f}")

    backbones = sorted(results)
    colors    = {b: _COLORS[i % len(_COLORS)] for i, b in enumerate(backbones)}

    print("\n=== Generating plots ===")
    plot_overall(results, output_dir)
    plot_per_task(results, ds_to_task, output_dir)
    plot_per_dataset_bars(results, output_dir)
    plot_per_dataset(results, output_dir)
    plot_metric_scatter(results, output_dir)

    for group, metrics in _METRIC_GROUPS.items():
        group_title = _GROUP_TITLES[group]
        for chart_type, suffix in [("strip", "strip"), ("errorbar", "errorbar")]:
            plot_clustering_overall_distribution(
                results, backbones, metrics, _METRIC_LABELS, colors, _display,
                f"Clustering — {group_title}",
                output_dir / f"clustering_overall_{group}_{suffix}.png",
                chart_type=chart_type)
            plot_clustering_per_task_distribution(
                results, ds_to_task, backbones, metrics, _METRIC_LABELS, colors, _display,
                f"Clustering — {group_title} — By Task Type",
                output_dir / f"clustering_per_task_{group}_{suffix}.png",
                chart_type=chart_type)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised clustering eval of frozen embeddings")
    parser.add_argument("--cache_dir",  default="outputs/linear_probe/embedding_cache/",
                        help="Directory containing .npz embedding cache files from linear probe")
    parser.add_argument("--output_dir", default="outputs/clustering_eval/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample each dataset to this many images before clustering (for local testing)")
    args = parser.parse_args()
    main(args)
