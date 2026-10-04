"""
Paper 2 Stage 5, Step 1-report — confidence/IoU threshold sweep over an
existing detection-box-verifier checkpoint (no GPU, no model).

detection-box-verifier already grades every predicted box's confidence
ONCE and stores it on the box itself (pred_boxes_json's "confidence"
field) — --confidence_threshold at run time only decides where that
checkpoint's OWN pred_boxes_filtered_json/metrics.json drew the line.
This script re-derives precision/recall/F1 at MANY different
(confidence_threshold, iou_threshold) combinations directly from the
checkpoint's already-stored per-box confidence scores — pure CPU
re-filtering + re-matching, no new grading calls needed — to answer
"what threshold combination actually maximizes F1," instead of trusting
whatever --confidence_threshold the run happened to use.

Same macro-averaging convention as src/common/bbox_summary.py's own
compute_at_thresholds (per-image precision/recall/F1 first, then
averaged across images) — not pooled/micro-averaged — so numbers at the
checkpoint's own --confidence_threshold/--iou_threshold here match its
own metrics.json exactly, letting you trust the rest of the sweep.

Usage:
    uv run python main.py detection-box-verifier-summary \\
        --ckpt outputs_final/detection_box_verifier_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.bbox_metrics import match_boxes, precision_recall_f1  # noqa: E402
from src.detection_box_verifier import _VERIFIER_PROMPTS_DETECTION_PATH, crop_box, load_yaml  # noqa: E402

# Distinct colors for each box_rubric criterion's own bars — same spirit
# as bbox_viz.py's _SUBCLASS_PALETTE (a fixed, deterministic-by-position
# categorical set) but kept local/short since there are only ever as many
# criteria as box_rubric defines (4 as of this writing), never free text.
_CRITERION_COLORS = ["dodgerblue", "orange", "mediumorchid", "chartreuse", "deeppink", "gold"]

# Same 3-point IoU grid src/common/bbox_summary.py's own threshold sweep
# uses (0.3/0.5/0.75) — kept consistent rather than inventing a different
# IoU grid just for this script. Confidence gets a finer grid (0.0-0.9 in
# steps of 0.1) since that's the genuinely new dimension this checkpoint
# can actually sweep (bbox_summary's own sweep has no confidence score to
# sweep at all — detection-baseline/--rubric checkpoints never had one).
IOU_THRESHOLDS = [0.3, 0.5, 0.75]
CONFIDENCE_THRESHOLDS = [round(0.1 * i, 1) for i in range(10)]


def load_rows(ckpt_path: Path) -> list[dict]:
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_grid(
    rows: list[dict], conf_thresholds: list[float] = CONFIDENCE_THRESHOLDS,
    iou_thresholds: list[float] = IOU_THRESHOLDS,
) -> dict[tuple[float, float], dict[str, float]]:
    """{(confidence_threshold, iou_threshold): {"precision","recall","f1"}},
    macro-averaged per-row, recomputed directly from each row's stored
    pred_boxes_json (filtered by confidence) and gt_boxes_json."""
    parsed = []
    for r in rows:
        pred_boxes = json.loads(r.get("pred_boxes_json") or "[]")
        gt_boxes = json.loads(r.get("gt_boxes_json") or "[]")
        parsed.append((pred_boxes, gt_boxes))

    out: dict[tuple[float, float], dict[str, float]] = {}
    for conf_thr in conf_thresholds:
        for iou_thr in iou_thresholds:
            precisions, recalls, f1s = [], [], []
            for pred_boxes, gt_boxes in parsed:
                filtered = [b for b in pred_boxes if b.get("confidence", 0.0) >= conf_thr]
                matches, unmatched_pred, unmatched_gt = match_boxes(filtered, gt_boxes, iou_thr)
                m = precision_recall_f1(matches, unmatched_pred, unmatched_gt)
                precisions.append(m["precision"])
                recalls.append(m["recall"])
                f1s.append(m["f1"])
            n = len(precisions)
            out[(conf_thr, iou_thr)] = {
                "precision": round(sum(precisions) / n, 4) if n else 0.0,
                "recall": round(sum(recalls) / n, 4) if n else 0.0,
                "f1": round(sum(f1s) / n, 4) if n else 0.0,
            }
    return out


def plot_sweep(grid: dict[tuple[float, float], dict[str, float]], out_path: Path, title: str = "") -> tuple[float, float, dict]:
    """F1 vs. confidence_threshold, one line per IoU threshold — matches
    bbox_summary.plot_iou_thresholds's own "one line per condition" style
    rather than introducing a new heatmap convention. Each line's own
    peak is marked; the single best (confidence_threshold, iou_threshold)
    across the whole grid gets a star + printed summary. Returns
    (best_conf_thr, best_iou_thr, best_metrics) for the caller to print/
    write out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iou_thrs = sorted({k[1] for k in grid})
    conf_thrs = sorted({k[0] for k in grid})
    colors = plt.cm.viridis([i / max(len(iou_thrs) - 1, 1) for i in range(len(iou_thrs))])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    best_key = max(grid, key=lambda k: grid[k]["f1"])
    best_conf_thr, best_iou_thr = best_key
    best_metrics = grid[best_key]

    for iou_thr, color in zip(iou_thrs, colors):
        f1s = [grid[(c, iou_thr)]["f1"] for c in conf_thrs]
        ax.plot(conf_thrs, f1s, marker="o", markersize=5, linewidth=1.8, color=color,
                 label=f"IoU={iou_thr}")
        peak_idx = max(range(len(f1s)), key=lambda i: f1s[i])
        ax.annotate(f"{f1s[peak_idx]:.3f}", (conf_thrs[peak_idx], f1s[peak_idx]),
                     ha="center", va="bottom", fontsize=7, xytext=(0, 5), textcoords="offset points")

    ax.scatter([best_conf_thr], [best_metrics["f1"]], marker="*", s=300, color="crimson",
                zorder=5, label=f"best: conf≥{best_conf_thr}, IoU={best_iou_thr} (F1={best_metrics['f1']:.3f})")

    ax.set_xlabel("confidence_threshold")
    ax.set_ylabel("F1")
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Confidence/IoU threshold sweep{subtitle}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best_conf_thr, best_iou_thr, best_metrics


def plot_confidence_distribution(rows: list[dict], out_path: Path, title: str = "") -> None:
    """Histogram of every predicted box's own confidence score, pooled
    across the whole checkpoint — a bimodal distribution (a cluster near
    1.0 and another near 0.0) is the visual signature of a clean
    separation between real/plausible boxes and over-generated junk,
    meaning confidence_threshold has an obvious cut point; a smeared,
    unimodal distribution means no single threshold cleanly separates
    them and the rubric/grading itself may need another pass. Plots the
    same pooled set plot_sweep's own "best" confidence_threshold would
    cut at, as a vertical reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = [
        b["confidence"] for r in rows for b in json.loads(r.get("pred_boxes_json") or "[]")
        if "confidence" in b
    ]
    if not confidences:
        print("  [warn] no per-box confidence scores found — skipping confidence distribution plot "
              "(is this checkpoint from detection-box-verifier, not detection-multiclass directly?)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidences, bins=20, range=(0, 1), color="dodgerblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("confidence")
    ax.set_ylabel("box count")
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Per-box confidence distribution{subtitle} (n={len(confidences)})", fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_criterion_score_distribution(rows: list[dict], out_path: Path, title: str = "") -> None:
    """Grouped bar chart — one color per box_rubric criterion, score
    (1/2/3) on the x-axis, box count on the y-axis — pooled across every
    predicted box's own criterion_scores in the checkpoint. Unlike
    plot_confidence_distribution's continuous histogram (confidence is a
    normalized float), each criterion's own raw score is one of exactly
    {1, 2, 3}, so a grouped bar chart (not a histogram with bins) is the
    honest representation — bins would imply a continuum that isn't
    there. A criterion that's mostly 3s across the board but another
    that's mixed 1/2/3 tells you WHICH part of the rubric is actually
    doing the discriminating work (and which one might be redundant or
    too lenient), information the single pooled confidence number can't
    show on its own."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vprompts = load_yaml(_VERIFIER_PROMPTS_DETECTION_PATH)
    rubric_lines = [ln.strip() for ln in vprompts["box_rubric"].strip().splitlines() if ln.strip()]

    # {criterion_number: {1: count, 2: count, 3: count}}
    by_criterion: dict[int, dict[int, int]] = {}
    for r in rows:
        for b in json.loads(r.get("pred_boxes_json") or "[]"):
            scores = b.get("criterion_scores")
            if not scores:
                continue
            for n_str, score in scores.items():
                n = int(n_str)
                score = int(score)
                by_criterion.setdefault(n, {1: 0, 2: 0, 3: 0})
                by_criterion[n][score] = by_criterion[n].get(score, 0) + 1

    if not by_criterion:
        print("  [warn] no per-box criterion_scores found — skipping criterion score distribution "
              "plot (is this checkpoint from detection-box-verifier, not detection-multiclass directly?)")
        return

    criteria = sorted(by_criterion)
    n_criteria = len(criteria)
    x = [1, 2, 3]
    bar_width = 0.8 / n_criteria

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, n in enumerate(criteria):
        counts = [by_criterion[n].get(s, 0) for s in x]
        offsets = [xi + (i - (n_criteria - 1) / 2) * bar_width for xi in x]
        color = _CRITERION_COLORS[(n - 1) % len(_CRITERION_COLORS)]
        # Legend label: the rubric's own criterion text, truncated — far
        # more informative than a bare "criterion 1".
        line = next((ln for ln in rubric_lines if ln.startswith(f"{n}.")), f"{n}.")
        label_text = line.split(".", 1)[1].strip() if "." in line else line
        if len(label_text) > 55:
            label_text = label_text[:52] + "..."
        ax.bar(offsets, counts, width=bar_width, color=color, label=f"{n}. {label_text}")

    ax.set_xticks(x)
    ax.set_xlabel("score")
    ax.set_ylabel("box count")
    subtitle = f": {title}" if title else ""
    total_boxes = sum(sum(c.values()) for c in by_criterion.values()) // n_criteria
    ax.set_title(f"Per-criterion score distribution{subtitle} (n={total_boxes} boxes)", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _numbered_path(out_path: Path, part: int, n_parts: int) -> Path:
    """out_path unchanged if n_parts == 1 (no pagination needed); else
    out_path with "_2", "_3", ... inserted before the suffix (part 1 is
    always the bare out_path, never "_1" — keeps the single-figure case's
    filename identical to before pagination existed)."""
    if n_parts <= 1 or part == 1:
        return out_path
    return out_path.with_name(f"{out_path.stem}_{part}{out_path.suffix}")


def plot_refinement_timeline(
    rows: list[dict], images_root: Path, out_path: Path, n_samples: int = 8,
    crop_pad_frac: float = 0.15, title: str = "", first_action: str | None = None,
    max_rows_per_figure: int = 6,
) -> list[Path]:
    """One row per sampled box, one column per round in its own
    `history` (detection_box_refine.py's own per-round snapshot list —
    bbox/label/scores/confidence/action AT that round) — shows how a
    single box's crop, label, and scores actually changed step by step,
    not just its final state. A split box's two+ children share the same
    early columns (the pre-split lineage) and diverge from the split
    round onward — exactly the shared-then-diverging history
    detection_box_refine.py's own history-inheritance produces.

    first_action (default None = no filter): restrict to boxes whose
    FIRST correction (history[0]["action"]) was exactly this — "split",
    "tighten", or "reclassify" — so a dataset's split/tighten/reclassify
    cases can be plotted as separate figures instead of one mixed-type
    gallery. A box included here always has a real first action in one
    of those three (boxes that passed outright in round 1 never reach
    this function at all — see the len(history) > 1 filter below).

    Only boxes detection-box-refine actually ran through refinement have
    a `history` key at all (detection-box-verifier's own checkpoints
    never do — it only grades once, never corrects) — rows without it
    are silently skipped, not an error. Boxes with history length 1
    (passed on the very first grade, no correction ever applied) are
    also skipped — there's no "change" to show for them; this plot is
    specifically for the interesting cases. Sampled longest-history-first
    (the most corrections applied = the most to show), not randomly.

    max_rows_per_figure (default 6): once `sampled` exceeds this, output
    is PAGINATED into multiple files ("<name>.png", "<name>_2.png",
    "<name>_3.png", ...) instead of one arbitrarily tall image. Returns
    the list of paths actually written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = []  # (history, agml_dataset, image_name, box)
    for r in rows:
        for b in json.loads(r.get("pred_boxes_json") or "[]"):
            history = b.get("history")
            if not history or len(history) <= 1:
                continue
            if first_action is not None and history[0].get("action") != first_action:
                continue
            candidates.append((history, r["agml_dataset"], r["image_name"], b))

    if not candidates:
        label = f" with first action {first_action!r}" if first_action else ""
        print(f"  [warn] no boxes{label} with a multi-round refinement history found — skipping "
              f"timeline (is this checkpoint from detection-box-refine, not detection-box-verifier?)")
        return []

    candidates.sort(key=lambda c: -len(c[0]))
    sampled = candidates[:n_samples]
    max_cols = max(len(h) for h, *_ in sampled)

    chunks = [sampled[i:i + max_rows_per_figure] for i in range(0, len(sampled), max_rows_per_figure)]
    written: list[Path] = []
    n_skipped_total = 0

    for part, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), max_cols, figsize=(3.6 * max_cols, 3.8 * len(chunk)),
                                  squeeze=False)

        for row_i, (history, dataset, image_name, box) in enumerate(chunk):
            img_path = images_root / dataset / image_name
            image = None
            if img_path.exists():
                image = PILImage.open(img_path).convert("RGB")
            else:
                n_skipped_total += 1

            for col_i in range(max_cols):
                ax = axes[row_i][col_i]
                ax.axis("off")
                if col_i >= len(history):
                    continue
                step = history[col_i]
                if image is None:
                    ax.set_title("[image not found]", fontsize=7, color="crimson")
                    continue
                crop = crop_box(image, step["bbox"], pad_frac=crop_pad_frac)
                ax.imshow(crop)
                crit = step.get("criterion_scores") or {}
                crit_str = "  ".join(f"c{n}={s}" for n, s in sorted(crit.items(), key=lambda kv: int(kv[0])))
                action = step.get("action") or "?"
                ax.set_title(
                    f"round {step['round']}: {step['category_name']}\n"
                    f"conf={step['confidence']:.2f}  ({crit_str})\n"
                    f"-> {action}",
                    fontsize=7.5,
                )

        subtitle = f": {title}" if title else ""
        page_str = f" (page {part}/{len(chunks)})" if len(chunks) > 1 else ""
        fig.suptitle(f"Box refinement timeline{subtitle}{page_str} ({len(sampled)}/{len(candidates)} "
                     f"multi-round boxes shown)", fontsize=11)
        plt.tight_layout()
        page_path = _numbered_path(out_path, part, len(chunks))
        page_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(page_path)

    if n_skipped_total:
        print(f"  [warn] {n_skipped_total}/{len(sampled)} sampled timeline row(s) had no source "
              f"image found under {images_root}")
    return written


def plot_crop_gallery(
    rows: list[dict], images_root: Path, out_path: Path, n_samples: int = 20,
    crop_pad_frac: float = 0.15, title: str = "", max_rows_per_figure: int = 5,
) -> list[Path]:
    """A grid of actual box crops (re-cropped from the source image via
    the SAME crop_box() the verifier itself graded — not re-derived or
    approximated) paired with their own confidence + per-criterion
    scores, so you can visually sanity-check whether a given score
    actually matches what the crop shows — exactly what detection-qc-plot
    does for whole-image boxes, applied to individual crops instead.

    Stratified by confidence, NOT random: boxes are sorted by confidence
    and sampled at even rank intervals across the full sorted list, so
    the gallery always spans the low-to-high range even when the
    underlying distribution skews heavily toward one end (confirmed on
    real data — see plot_confidence_distribution) — a plain random sample
    would mostly just show more examples from the dominant end and leave
    the (usually more diagnostically interesting) low-confidence tail
    completely unrepresented.

    max_rows_per_figure (default 5, i.e. 20 crops at the fixed 4-column
    grid): once the sampled set needs more GRID rows than this, output is
    PAGINATED into multiple files ("<name>.png", "<name>_2.png", ...)
    instead of one arbitrarily tall image. Returns the list of paths
    actually written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = []  # (confidence, agml_dataset, image_name, box)
    for r in rows:
        for b in json.loads(r.get("pred_boxes_json") or "[]"):
            if "confidence" not in b:
                continue
            candidates.append((b["confidence"], r["agml_dataset"], r["image_name"], b))

    if not candidates:
        print("  [warn] no per-box confidence scores found — skipping crop gallery (is this "
              "checkpoint from detection-box-verifier, not detection-multiclass directly?)")
        return []

    candidates.sort(key=lambda c: c[0])
    n = min(n_samples, len(candidates))
    if n < len(candidates):
        idxs = [round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    else:
        idxs = list(range(len(candidates)))
    sampled = [candidates[i] for i in idxs]

    n_cols = 4
    items_per_figure = max_rows_per_figure * n_cols
    chunks = [sampled[i:i + items_per_figure] for i in range(0, len(sampled), items_per_figure)]
    written: list[Path] = []
    n_skipped_total = 0

    for part, chunk in enumerate(chunks, start=1):
        n_rows = (len(chunk) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))
        # n_cols is always 4 (fixed above), so plt.subplots() here NEVER
        # returns a bare scalar Axes (that only happens at nrows=ncols=1) —
        # .flatten() is always safe and correct, regardless of len(chunk).
        # Confirmed bug (2026-10-02): an old `if len(sampled) > 1` branch
        # wrapped the whole 4-element axes ARRAY in a single-element list
        # for a 1-item chunk, so ax.imshow() below was called on a numpy
        # array instead of an Axes.
        axes = axes.flatten()

        for ax, (confidence, dataset, image_name, box) in zip(axes, chunk):
            img_path = images_root / dataset / image_name
            if not img_path.exists():
                ax.axis("off")
                ax.set_title(f"[image not found]\n{dataset}/{image_name}", fontsize=7, color="crimson")
                n_skipped_total += 1
                continue
            image = PILImage.open(img_path).convert("RGB")
            crop = crop_box(image, box["bbox"], pad_frac=crop_pad_frac)
            ax.imshow(crop)
            ax.axis("off")

            criterion_scores = box.get("criterion_scores") or {}
            crit_str = "  ".join(f"c{n}={s}" for n, s in sorted(criterion_scores.items(), key=lambda kv: int(kv[0])))
            label = str(box.get("category_name") or "object")
            ax.set_title(f"{label}\nconf={confidence:.2f}  ({crit_str})", fontsize=7.5)

        for ax in axes[len(chunk):]:
            ax.axis("off")

        subtitle = f": {title}" if title else ""
        page_str = f" (page {part}/{len(chunks)})" if len(chunks) > 1 else ""
        fig.suptitle(f"Box crop gallery, stratified by confidence{subtitle}{page_str} "
                     f"({len(sampled)}/{len(candidates)} boxes shown)", fontsize=11)
        plt.tight_layout()
        page_path = _numbered_path(out_path, part, len(chunks))
        page_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(page_path)

    if n_skipped_total:
        print(f"  [warn] {n_skipped_total}/{len(sampled)} sampled crop(s) skipped — source image "
              f"not found under {images_root}")
    return written


def write_grid_csv(grid: dict[tuple[float, float], dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence_threshold", "iou_threshold", "precision", "recall", "f1"])
        for (conf_thr, iou_thr), m in sorted(grid.items()):
            w.writerow([conf_thr, iou_thr, m["precision"], m["recall"], m["f1"]])
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    if not any(r.get("pred_boxes_json") for r in rows):
        raise SystemExit(f"ERROR: {ckpt_path} has no pred_boxes_json — is this a "
                          f"detection-box-verifier checkpoint, not a plain detection-baseline one?")

    grid = compute_grid(rows)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "threshold_sweep"
    plot_path = out_dir / "f1_vs_confidence_threshold.png"
    dist_path = out_dir / "confidence_distribution.png"
    criterion_path = out_dir / "criterion_score_distribution.png"
    csv_path = out_dir / "threshold_sweep.csv"

    best_conf_thr, best_iou_thr, best_metrics = plot_sweep(grid, plot_path, title=args.title)
    plot_confidence_distribution(rows, dist_path, title=args.title)
    plot_criterion_score_distribution(rows, criterion_path, title=args.title)
    write_grid_csv(grid, csv_path)

    print(f"  Saved: {plot_path}")
    print(f"  Saved: {dist_path}")
    print(f"  Saved: {criterion_path}")

    # refine_samples/ — a SEPARATE folder from threshold_sweep/ (sibling,
    # at the checkpoint's own directory level), for the per-box visual
    # inspection plots (crop gallery + per-dataset timelines) rather than
    # the aggregate statistics plots above.
    samples_dir = Path(args.samples_dir) if args.samples_dir else ckpt_path.parent / "refine_samples"
    gallery_path = samples_dir / "crop_gallery.png"
    for p in plot_crop_gallery(rows, Path(args.images_root), gallery_path, n_samples=args.n_crop_samples,
                                crop_pad_frac=args.crop_pad_frac, title=args.title,
                                max_rows_per_figure=args.max_rows_per_figure):
        print(f"  Saved: {p}")

    # One subfolder per dataset, one figure per correction TYPE within it
    # (split/tighten/reclassify, grouped by each box's FIRST correction —
    # see plot_refinement_timeline's first_action param) — keeps each
    # figure to a manageable, actually-readable height instead of one
    # giant mixed-type gallery per dataset.
    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["agml_dataset"], []).append(r)
    for dataset, ds_rows in sorted(by_dataset.items()):
        ds_dir = samples_dir / dataset
        ds_title = f"{args.title} / {dataset}" if args.title else dataset
        for action in ("split", "tighten", "reclassify"):
            timeline_path = ds_dir / f"{action}_timeline.png"
            for p in plot_refinement_timeline(ds_rows, Path(args.images_root), timeline_path,
                                               n_samples=args.n_crop_samples, crop_pad_frac=args.crop_pad_frac,
                                               title=ds_title, first_action=action,
                                               max_rows_per_figure=args.max_rows_per_figure):
                print(f"  Saved: {p}")

    print(f"\n  Best: confidence_threshold={best_conf_thr}, iou_threshold={best_iou_thr} -> "
          f"precision={best_metrics['precision']:.3f} recall={best_metrics['recall']:.3f} "
          f"f1={best_metrics['f1']:.3f} (n={len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                     help="A detection-box-verifier (BOX_VERIFIER_CKPT_FIELDS-shaped) checkpoint "
                          "CSV — needs per-box \"confidence\" in pred_boxes_json.")
    ap.add_argument("--images_root", default="dataset/images_detection",
                     help="For the crop gallery only — same convention as every other detection "
                          "script's --images_root.")
    ap.add_argument("--n_crop_samples", type=int, default=20,
                     help="How many box crops to show in crop_gallery.png, stratified by confidence "
                          "(see plot_crop_gallery's docstring for why not random).")
    ap.add_argument("--max_rows_per_figure", type=int, default=5,
                     help="Caps how tall a single crop_gallery/*_timeline.png can get — once the "
                          "sampled set needs more grid rows (crop_gallery) or boxes "
                          "(*_timeline.png) than this, output is paginated into multiple numbered "
                          "files (<name>.png, <name>_2.png, ...) instead of one arbitrarily tall "
                          "image.")
    ap.add_argument("--crop_pad_frac", type=float, default=0.15,
                     help="Must match whatever --crop_pad_frac detection-box-verifier used to "
                          "generate --ckpt, or the gallery's crops won't match what the verifier "
                          "actually saw.")
    ap.add_argument("--out_dir", default=None,
                     help="Aggregate statistics plots (sweep/distributions). Default: <ckpt's own "
                          "directory>/threshold_sweep/")
    ap.add_argument("--samples_dir", default=None,
                     help="Per-box visual inspection plots (crop gallery + one refinement_timeline "
                          "per dataset). Default: <ckpt's own directory>/refine_samples/")
    ap.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(ap.parse_args())
