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

# detection-multiclass (Stage 4, docs/stage4_detection_verifier.md) QC —
# a fixed categorical palette for coloring predicted boxes by their own
# sub-class/condition label (category_name, e.g. "flower: open flower" vs
# "flower: bud") instead of one flat color. Red/crimson deliberately
# excluded — reserved for ground truth (draw_boxes/plot_predictions's
# existing dashed-red convention), so a sub-class box is never confused
# with a GT box by color alone. 18 colors, not a round number — just
# "every visually-distinct named matplotlib color" after excluding
# red/GT's own hue family; widened from an initial smaller set after
# confirming empirically that 12 slots collided on realistic label pairs.
#
# KNOWN LIMITATION: category_name is FREE TEXT (detection_multiclass's own
# prompt explicitly asks for "whatever specific term best describes what
# you see," not a fixed vocabulary) — a full dataset run can produce far
# more distinct label strings than any fixed palette can guarantee unique
# colors for. This only matters for labels that CO-OCCUR in the same
# image's own legend (plot_predictions only lists labels actually present
# in that one image) — an unrelated collision between, say, some "pod"
# label on one image and some "flower" label on a totally different image
# is irrelevant, since they're never shown together. A collision between
# two sub-classes that genuinely appear together in the same image is
# still possible with enough distinct labels, just less likely at 18 slots
# than it was at 12 — if you see two visually-identical-colored boxes with
# different text labels in one plot, that's this limitation, not a bug.
_SUBCLASS_PALETTE = [
    "dodgerblue", "orange", "gold", "mediumorchid", "deeppink", "cyan",
    "chartreuse", "saddlebrown", "turquoise", "navy", "coral", "yellow",
    "slateblue", "limegreen", "hotpink", "darkorange", "mediumblue", "olive",
]


def _label_color(label: str) -> str:
    """Deterministic label -> palette color, stable across calls/processes
    (NOT Python's built-in hash(), which is randomized per-process for
    strings) — so the same sub-class label always gets the same color
    across every image in a QC run, not just within one image. zlib.crc32
    (not sum(ord(c)) — confirmed empirically to collide on real labels
    sharing most characters, e.g. "flower: open flower" vs "flower: dead
    flower" landed on the same color) — a real checksum, not just a sum,
    so two strings differing by only a few characters still land far
    apart in the output range. See _SUBCLASS_PALETTE's own comment for
    the residual collision risk this doesn't eliminate."""
    import zlib
    idx = zlib.crc32(label.encode("utf-8")) % len(_SUBCLASS_PALETTE)
    return _SUBCLASS_PALETTE[idx]


def draw_boxes(ax, boxes: list[dict], color: str = "lime", linestyle: str = "-",
                label_key: str = "category_name", color_by_label: bool = False) -> None:
    """Draw every box in `boxes` (each {"bbox": [x1,y1,x2,y2], ...}) onto a
    matplotlib Axes already showing an image. `label_key`, if present and
    non-empty on a box, annotates it just above the box's top-left corner.

    color_by_label (default False, zero behavior change for every existing
    caller): when True, each box's own `color` is derived from its
    `label_key` value via _label_color() instead of the flat `color`
    argument — e.g. detection-multiclass's QC plots, where every box's
    category_name already carries its own sub-class ("flower: bud" vs
    "flower: open flower"), not just the dataset's one coarse class. A box
    with no label (falsy label_key value) still falls back to `color`."""
    import matplotlib.patches as mpatches

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        label = box.get(label_key)
        box_color = _label_color(str(label)) if (color_by_label and label) else color
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=box_color, facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        if label:
            ax.text(x1, max(0, y1 - 3), str(label), color=box_color, fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))


def plot_predictions(
    image: PILImage.Image, pred_boxes: list[dict], gt_boxes: list[dict] | None = None,
    out_path: str | Path | None = None, title: str = "", color_by_subclass: bool = False,
) -> None:
    """Draw predicted boxes (solid lime, or one color per sub-class if
    color_by_subclass) and, if given, ground-truth boxes (dashed red,
    NEVER multi-colored — GT only ever carries one coarse class) on the
    same image for a direct visual pred-vs-GT QC check — exactly what
    detection_qc_plot.py needs to catch a coordinate-alignment bug
    (predicted boxes plotted against the ORIGINAL image; a systematic
    offset means the model's coordinates are relative to an internally-
    rescaled view, not this image) or a parser bug, in one glance.

    color_by_subclass (default False, zero behavior change for Stage 2's
    plain baseline/--rubric QC): detection-multiclass's own QC — each
    predicted box's category_name already carries its own sub-class/
    condition (e.g. "flower: bud" vs "flower: open flower"), so coloring
    every box the same flat lime hides exactly the thing that stage is
    trying to show. Legend lists each unique sub-class actually present
    among pred_boxes, colored via bbox_viz._label_color (stable across
    every image in a run, not just within one)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8 * image.height / max(image.width, 1)))
    ax.imshow(image)
    draw_boxes(ax, pred_boxes, color="lime", linestyle="-", color_by_label=color_by_subclass)
    if gt_boxes:
        draw_boxes(ax, gt_boxes, color="red", linestyle="--")

    if color_by_subclass:
        unique_labels = sorted({str(b["category_name"]) for b in pred_boxes if b.get("category_name")})
        handles = [plt.Line2D([0], [0], color=_label_color(lbl), lw=2, label=lbl) for lbl in unique_labels]
        if not handles:
            handles = [plt.Line2D([0], [0], color="lime", lw=2, label=f"predicted ({len(pred_boxes)})")]
    else:
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


def plot_predictions_comparison(
    image: PILImage.Image,
    pred_boxes_a: list[dict], pred_boxes_b: list[dict], gt_boxes: list[dict] | None,
    label_a: str, label_b: str,
    out_path: str | Path | None = None, title: str = "", color_by_subclass: bool = False,
) -> None:
    """Stage 2.5 QC — same image, two side-by-side panels: condition A's
    predictions (left) vs condition B's (right), each against the SAME
    ground truth, for a direct "did the rubric change what the model saw"
    visual check (detection_qc_plot.py's --compare_ckpt). Reuses
    draw_boxes/plot_predictions's own color convention (predicted=lime, or
    per-sub-class if color_by_subclass; ground truth=dashed red) so a
    QC-plots reader doesn't have to learn a second legend."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8 * image.height / max(image.width, 1)))
    for ax, label, pred_boxes in ((axes[0], label_a, pred_boxes_a), (axes[1], label_b, pred_boxes_b)):
        ax.imshow(image)
        draw_boxes(ax, pred_boxes, color="lime", linestyle="-", color_by_label=color_by_subclass)
        if gt_boxes:
            draw_boxes(ax, gt_boxes, color="red", linestyle="--")
        if color_by_subclass:
            unique_labels = sorted({str(b["category_name"]) for b in pred_boxes if b.get("category_name")})
            handles = [plt.Line2D([0], [0], color=_label_color(lbl), lw=2, label=lbl) for lbl in unique_labels]
            if not handles:
                handles = [plt.Line2D([0], [0], color="lime", lw=2, label=f"predicted ({len(pred_boxes)})")]
        else:
            handles = [plt.Line2D([0], [0], color="lime", lw=2, label=f"predicted ({len(pred_boxes)})")]
        if gt_boxes is not None:
            handles.append(plt.Line2D([0], [0], color="red", lw=2, linestyle="--",
                                       label=f"ground truth ({len(gt_boxes)})"))
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
