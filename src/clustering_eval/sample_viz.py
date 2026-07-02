"""
Visualise top-3 easiest and top-3 hardest datasets per task type
using silhouette score and Fisher discriminant ratio from clustering_metrics.json.

For each selected dataset a panel is produced showing one random sample image
per class. Panels are arranged in a grid: easiest on the right, hardest on the
left, rows = datasets within a task.

Usage:
    python main.py cluster-viz \\
      --clustering_metrics outputs/clustering_eval/clustering_metrics.json \\
      --backbone           dinov3-vitb16-pretrain-lvd1689m \\
      --agml_root          ~/.agml/datasets \\
      --output_dir         outputs/clustering_eval/ \\
      --config             src/config/datasets.yaml \\
      --n_per_class        2
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

from src.common.dataset_config import datasets_by_name

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _load_image(path: Path, size: int = 128) -> np.ndarray | None:
    try:
        img = PILImage.open(path).convert("RGB").resize((size, size), PILImage.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def _sample_images(ds_root: Path, classes: list[str], n: int, seed: int = 42) -> dict[str, list[np.ndarray]]:
    """Return {class: [image_array, ...]} with up to n images per class."""
    rng = random.Random(seed)
    result: dict[str, list[np.ndarray]] = {}
    for cls in classes:
        cls_dir = ds_root / cls
        if not cls_dir.exists():
            result[cls] = []
            continue
        paths = _find_images(cls_dir)
        chosen = rng.sample(paths, min(n, len(paths)))
        imgs = [a for p in chosen if (a := _load_image(p)) is not None]
        result[cls] = imgs
    return result


def _dataset_panel(
    ax_row: list,
    ds_name: str,
    classes: list[str],
    images: dict[str, list[np.ndarray]],
    score_sil: float,
    score_fdr: float,
    n_per_class: int,
    label_color: str,
) -> None:
    """Fill one row of axes with class label + sample images."""
    col = 0
    max_cols = len(ax_row)
    for cls in classes:
        imgs = images.get(cls, [])
        for img in imgs[:n_per_class]:
            if col >= max_cols:
                break
            ax_row[col].imshow(img)
            ax_row[col].set_title(cls.replace("_", " "), fontsize=5.5,
                                  color="black", pad=2)
            ax_row[col].axis("off")
            col += 1
        # blank if fewer images than n_per_class
        while col < (classes.index(cls) + 1) * n_per_class and col < max_cols:
            ax_row[col].axis("off")
            col += 1

    # blank remaining columns
    while col < max_cols:
        ax_row[col].axis("off")
        col += 1

    # Row label on left-most axis
    short = ds_name.replace("_classification", "").replace("_", " ")
    ax_row[0].set_ylabel(
        f"{short}\nsil={score_sil:.3f}  fdr={score_fdr:.3f}",
        fontsize=5.5, rotation=0, labelpad=60, va="center", color=label_color,
    )


def plot_difficulty_samples(
    metrics_by_ds: dict[str, dict],
    ds_to_task: dict[str, str],
    agml_root: Path,
    output_dir: Path,
    n_top: int = 3,
    n_per_class: int = 2,
    metric: str = "silhouette",
) -> None:
    tasks = sorted({t for t in ds_to_task.values()
                    if t in ("disease", "pest_damage", "weed_identification")})

    # Select top-n easiest and hardest per task (by chosen metric)
    selected: dict[str, dict[str, list[str]]] = {}  # task -> {"hard": [...], "easy": [...]}
    for task in tasks:
        task_ds = [(ds, m) for ds, m in metrics_by_ds.items()
                   if ds_to_task.get(ds) == task and metric in m]
        task_ds.sort(key=lambda x: x[1][metric])
        n = min(n_top, len(task_ds) // 2) if len(task_ds) > 1 else 1
        selected[task] = {
            "hard": [ds for ds, _ in task_ds[:n]],
            "easy": [ds for ds, _ in reversed(task_ds[-n:])],
        }

    # Build flat list of (task, difficulty, ds_name, score_sil, score_fdr)
    rows: list[tuple[str, str, str, float, float]] = []
    for task in tasks:
        for diff in ("hard", "easy"):
            for ds in selected[task][diff]:
                m = metrics_by_ds[ds]
                rows.append((task, diff, ds, m["silhouette"], m["fisher_dr"]))

    if not rows:
        print("  [warn] No datasets to visualise")
        return

    # Determine max classes across all datasets (for column count)
    all_classes: dict[str, list[str]] = {}
    for _, _, ds, _, _ in rows:
        ds_root = agml_root / ds
        if ds_root.exists():
            cls_list = sorted(p.name for p in ds_root.iterdir() if p.is_dir())
        else:
            cls_list = []
        all_classes[ds] = cls_list

    max_cls = max((len(c) for c in all_classes.values()), default=1)
    n_cols = max_cls * n_per_class
    n_rows = len(rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.4, n_rows * 1.6),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.6, wspace=0.05, left=0.18)

    # Section dividers: track task+difficulty changes
    prev_section = None
    for row_idx, (task, diff, ds, sil, fdr) in enumerate(rows):
        section = (task, diff)
        if section != prev_section:
            label_color = "#c0392b" if diff == "hard" else "#27ae60"
            section_title = f"{task.replace('_', ' ').title()} — {'Hardest' if diff=='hard' else 'Easiest'}"
            axes[row_idx][0].annotate(
                section_title,
                xy=(0, 1.5), xycoords="axes fraction",
                fontsize=7, fontweight="bold", color=label_color,
                annotation_clip=False,
            )
            prev_section = section
        else:
            label_color = "#c0392b" if diff == "hard" else "#27ae60"

        classes = all_classes.get(ds, [])
        if classes and (agml_root / ds).exists():
            images = _sample_images(agml_root / ds, classes, n_per_class)
        else:
            images = {}
            print(f"  [warn] Dataset not found locally: {agml_root / ds}")

        _dataset_panel(
            axes[row_idx].tolist(), ds, classes, images,
            sil, fdr, n_per_class, label_color,
        )

    # Legend
    hard_patch = mpatches.Patch(color="#c0392b", label="Hardest (low score)")
    easy_patch = mpatches.Patch(color="#27ae60", label="Easiest (high score)")
    fig.legend(handles=[hard_patch, easy_patch], loc="upper right", fontsize=7,
               bbox_to_anchor=(1.0, 1.0))

    metric_label = {"silhouette": "Silhouette", "fisher_dr": "Fisher DR"}.get(metric, metric)
    fig.suptitle(
        f"Top-{n_top} Easiest & Hardest Datasets per Task — {metric_label} (DINOv3)",
        fontsize=9, fontweight="bold", y=1.01,
    )

    path = output_dir / f"clustering_difficulty_samples_{metric}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agml_root = Path(args.agml_root).expanduser()

    with open(args.clustering_metrics) as f:
        data = json.load(f)

    backbone_data = data.get(args.backbone)
    if backbone_data is None:
        print(f"Backbone '{args.backbone}' not found. Available: {list(data.keys())}")
        raise SystemExit(1)

    cfg = datasets_by_name(Path(args.config))
    ds_to_task = {k: v.task_type for k, v in cfg.items()}

    print(f"Generating sample visualisations (n_top={args.n_top}, n_per_class={args.n_per_class})")
    for metric in ("silhouette", "fisher_dr"):
        print(f"\n  Metric: {metric}")
        plot_difficulty_samples(
            backbone_data, ds_to_task, agml_root, output_dir,
            n_top=args.n_top, n_per_class=args.n_per_class, metric=metric,
        )
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering_metrics", default="outputs/clustering_eval/clustering_metrics.json")
    parser.add_argument("--backbone", default="dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--agml_root", default="~/.agml/datasets")
    parser.add_argument("--output_dir", default="outputs/clustering_eval/")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--n_top", type=int, default=3,
                        help="Number of easiest and hardest datasets to show per task")
    parser.add_argument("--n_per_class", type=int, default=2,
                        help="Sample images per class per dataset")
    args = parser.parse_args()
    main(args)
