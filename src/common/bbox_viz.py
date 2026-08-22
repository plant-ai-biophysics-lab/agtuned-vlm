"""Bounding-box QC plotting — Stage 5. Generalized out of
experiments/alignment_test/check_alignment.py's draw_boxes() closure (which
was duplicated identically in two functions there, and uses pixel [x,y,w,h]
boxes since it reads the pip `agml` package's COCO export). This module's
boxes are [x1,y1,x2,y2] (pixel) instead — the convention every other Stage 5
module (bbox_parse.py, bbox_metrics.py) already uses, since that's what
models are prompted to output and what IoU math wants directly."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage


def draw_boxes(ax, boxes: list[dict], color: str = "lime", linestyle: str = "-",
                label_key: str = "category_name") -> None:
    """Draw every box in `boxes` (each {"bbox": [x1,y1,x2,y2], ...}) onto a
    matplotlib Axes already showing an image. `label_key`, if present and
    non-empty on a box, annotates it just above the box's top-left corner."""
    import matplotlib.patches as mpatches

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        label = box.get(label_key)
        if label:
            ax.text(x1, max(0, y1 - 3), str(label), color=color, fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))


def plot_predictions(
    image: PILImage.Image, pred_boxes: list[dict], gt_boxes: list[dict] | None = None,
    out_path: str | Path | None = None, title: str = "",
) -> None:
    """Draw predicted boxes (solid lime) and, if given, ground-truth boxes
    (dashed red) on the same image for a direct visual pred-vs-GT QC check —
    exactly what detection_qc_plot.py needs to catch a coordinate-alignment
    bug (predicted boxes plotted against the ORIGINAL image; a systematic
    offset means the model's coordinates are relative to an internally-
    rescaled view, not this image) or a parser bug, in one glance."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8 * image.height / max(image.width, 1)))
    ax.imshow(image)
    draw_boxes(ax, pred_boxes, color="lime", linestyle="-")
    if gt_boxes:
        draw_boxes(ax, gt_boxes, color="red", linestyle="--")

    handles = [plt.Line2D([0], [0], color="lime", lw=2, label=f"predicted ({len(pred_boxes)})")]
    if gt_boxes is not None:
        handles.append(plt.Line2D([0], [0], color="red", lw=2, linestyle="--", label=f"ground truth ({len(gt_boxes)})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
