"""
Per-class error overlap analysis between language and vision assessments.

Runs two parallel analyses:

  Classification overlap  (cls_*.csv)
    Pass/fail = per-class accuracy >= threshold (default 0.5).

  Description overlap  (desc_*.csv)
    Per-class mean ROUGE-SS score >= naive baseline ROUGE-SS → pass.
    Threshold is read from the assessment metrics JSON files; falls back
    to --rouge_threshold if the JSON is unavailable.

Four quadrants per analysis:
  Both pass   — language ✓  vision ✓
  Lang only   — language ✓  vision ✗
  Vision only — language ✗  vision ✓
  Both fail   — language ✗  vision ✗

Outputs (per model, per analysis type):
  error_overlap_{cls|desc}_<model>.csv
  error_overlap_{cls|desc}_<model>.png
  error_overlap_{cls|desc}_summary.json
  error_overlap_{cls|desc}_heatmap.png

Usage:
    python main.py error-overlap \\
        --language_dir  outputs/language_assessment/ \\
        --vision_dir    outputs/vision_assessment/ \\
        --output_dir    outputs/error_overlap/
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Shared constants ───────────────────────────────────────────────────────────

QUADRANT_LABELS = {
    "both_pass":   "Both pass",
    "lang_only":   "Lang only (textual knowledge compensates)",
    "vision_only": "Vision only (visual knowledge compensates)",
    "both_fail":   "Both fail",
}

QUADRANT_COLORS = {
    "both_pass":   "#2ecc71",
    "lang_only":   "#e74c3c",
    "vision_only": "#3498db",
    "both_fail":   "#95a5a6",
}

_CLS_THRESHOLD   = 0.5
_ROUGE_THRESHOLD = 0.15   # fallback if metrics JSON unavailable


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


# ── Classification helpers ─────────────────────────────────────────────────────

def _correct(row: dict) -> bool:
    if "judged_correct" in row and row["judged_correct"] != "":
        return str(row["judged_correct"]) == "1"
    return str(row.get("correct", "0")) == "1"


def per_class_accuracy(rows: list[dict]) -> dict[tuple[str, str], dict]:
    by_class: dict[tuple, list[bool]] = defaultdict(list)
    for row in rows:
        key = (row["agml_dataset"], row["class_label"])
        by_class[key].append(_correct(row))
    return {
        key: {"score": round(sum(hits) / len(hits), 4), "n": len(hits)}
        for key, hits in by_class.items()
    }


# ── Description helpers ────────────────────────────────────────────────────────

def _rouge_score(row: dict) -> float | None:
    """Return best available ROUGE score for a description row."""
    for key in ("rouge_ss", "rouge1", "rougeL", "rouge2"):
        v = row.get(key, "")
        if v not in ("", None):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def per_class_rouge(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Mean ROUGE-SS (or best available) per (dataset, class_label)."""
    by_class: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["agml_dataset"], row["class_label"])
        s = _rouge_score(row)
        if s is not None:
            by_class[key].append(s)
    return {
        key: {"score": round(sum(vals) / len(vals), 4), "n": len(vals)}
        for key, vals in by_class.items()
        if vals
    }


def _naive_rouge_from_json(metrics_path: Path) -> float | None:
    """Extract naive baseline ROUGE-SS overall from an assessment metrics JSON."""
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
        naive = data.get("naive", {})
        # Prefer pre-computed overall
        ov = naive.get("description_overall", {})
        if ov.get("rouge_ss") is not None:
            return float(ov["rouge_ss"])
        # Fall back to macro-averaging per-dataset naive values
        per_ds = naive.get("description", {})
        vals = [v["rouge_ss"] for v in per_ds.values() if "rouge_ss" in v]
        if vals:
            return round(float(np.mean(vals)), 4)
    except Exception:
        pass
    return None


# ── Quadrant logic (shared) ────────────────────────────────────────────────────

def assign_quadrant(lang_score: float, vis_score: float, threshold: float) -> str:
    lang_pass = lang_score >= threshold
    vis_pass  = vis_score  >= threshold
    if lang_pass and vis_pass:
        return "both_pass"
    if lang_pass and not vis_pass:
        return "lang_only"
    if not lang_pass and vis_pass:
        return "vision_only"
    return "both_fail"


# ── Per-model analysis ─────────────────────────────────────────────────────────

def analyze_model(
    lang_scores: dict[tuple, dict],
    vis_scores:  dict[tuple, dict],
    model_id: str,
    output_dir: Path,
    threshold: float,
    prefix: str,          # "cls" or "desc"
    x_label: str,
    y_label: str,
    score_label: str,
) -> dict:
    shared_keys = set(lang_scores) & set(vis_scores)
    if not shared_keys:
        print(f"  [{model_id}] No overlapping (dataset, class) pairs — skipping.")
        return {}

    records = []
    quadrant_counts: dict[str, int] = defaultdict(int)
    for key in sorted(shared_keys):
        dataset, cls = key
        ls = lang_scores[key]["score"]
        vs = vis_scores[key]["score"]
        q  = assign_quadrant(ls, vs, threshold)
        quadrant_counts[q] += 1
        records.append({
            "agml_dataset":  dataset,
            "class_label":   cls,
            "lang_score":    ls,
            "vis_score":     vs,
            "quadrant":      q,
            "lang_n":        lang_scores[key]["n"],
            "vis_n":         vis_scores[key]["n"],
        })

    total = len(records)
    summary = {
        "model":    model_id,
        "n_classes": total,
        "threshold": threshold,
        "quadrants": {q: {"n": n, "pct": round(n / total, 3)} for q, n in quadrant_counts.items()},
    }

    csv_path = output_dir / f"error_overlap_{prefix}_{_safe_name(model_id)}.csv"
    _write_csv(records, csv_path, prefix)

    print(f"  [{model_id}] ({prefix})  n={total}  threshold={threshold:.3f}  "
          + "  ".join(f"{q}={n}" for q, n in quadrant_counts.items()))

    return summary, records


# ── Output helpers ─────────────────────────────────────────────────────────────

def _write_csv(records: list[dict], path: Path, prefix: str) -> None:
    fields = ["agml_dataset", "class_label", "lang_score", "vis_score", "quadrant", "lang_n", "vis_n"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {path}")


def _draw_scatter_ax(
    ax: plt.Axes,
    records: list[dict],
    threshold: float,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Draw a single quadrant scatter into *ax* (no legend — caller adds it)."""
    quadrant_counts: dict[str, int] = defaultdict(int)
    for r in records:
        ax.scatter(r["lang_score"], r["vis_score"],
                   color=QUADRANT_COLORS[r["quadrant"]], alpha=0.75, s=60, zorder=3)
        quadrant_counts[r["quadrant"]] += 1

    t = threshold
    ax.axvline(t, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(t, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.axhspan(0, t, xmin=0, xmax=t, color=QUADRANT_COLORS["both_fail"],   alpha=0.06)
    ax.axhspan(0, t, xmin=t, xmax=1, color=QUADRANT_COLORS["lang_only"],   alpha=0.06)
    ax.axhspan(t, 1, xmin=0, xmax=t, color=QUADRANT_COLORS["vision_only"], alpha=0.06)
    ax.axhspan(t, 1, xmin=t, xmax=1, color=QUADRANT_COLORS["both_pass"],   alpha=0.06)

    pad = 0.04
    style = dict(fontsize=8, fontweight="bold", alpha=0.8, zorder=4)
    ax.text(pad,     pad,     f"n={quadrant_counts['both_fail']}",   color=QUADRANT_COLORS["both_fail"],   ha="left",  va="bottom", transform=ax.transAxes, **style)
    ax.text(1 - pad, pad,     f"n={quadrant_counts['lang_only']}",   color=QUADRANT_COLORS["lang_only"],   ha="right", va="bottom", transform=ax.transAxes, **style)
    ax.text(pad,     1 - pad, f"n={quadrant_counts['vision_only']}", color=QUADRANT_COLORS["vision_only"], ha="left",  va="top",    transform=ax.transAxes, **style)
    ax.text(1 - pad, 1 - pad, f"n={quadrant_counts['both_pass']}",   color=QUADRANT_COLORS["both_pass"],   ha="right", va="top",    transform=ax.transAxes, **style)

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    max_val = max(
        max((r["lang_score"] for r in records), default=1.0),
        max((r["vis_score"]  for r in records), default=1.0),
    )
    lim = min(max_val * 1.05 + 0.02, 1.02)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.annotate(f"threshold = {t:.3f} (naive baseline)",
                xy=(t, -0.02), xycoords="data",
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color="grey")


def _plot_combined_scatter(
    cls_records: list[dict],
    desc_records: list[dict],
    model_id: str,
    path: Path,
    cls_thr: float,
    rouge_thr: float,
) -> None:
    """Side-by-side classification + description scatter with a single shared legend below."""
    fig, (ax_cls, ax_desc) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Error overlap — {model_id.split('/')[-1]}", fontsize=13, fontweight="bold")

    _draw_scatter_ax(
        ax_cls, cls_records, cls_thr,
        x_label="Language accuracy (per class)",
        y_label="Vision accuracy (per class)",
        title="Classification",
    )
    _draw_scatter_ax(
        ax_desc, desc_records, rouge_thr,
        x_label="Language mean ROUGE-SS (per class)",
        y_label="Vision mean ROUGE-SS (per class)",
        title="Description (ROUGE-SS)",
    )

    legend_patches = [
        mpatches.Patch(color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        for q in ["both_pass", "lang_only", "vision_only", "both_fail"]
    ]
    fig.legend(
        handles=legend_patches, fontsize=9,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _average_summary(summaries: list[dict]) -> dict:
    """
    Aggregate summary by summing raw class counts across models, then re-computing
    fractions from the total. This weights every class observation equally regardless
    of how many classes each model was evaluated on — consistent with the interaction
    matrix which also sums counts before normalising.
    """
    quadrant_order = ["both_pass", "lang_only", "vision_only", "both_fail"]
    total_counts = {q: sum(s["quadrants"].get(q, {}).get("n", 0) for s in summaries)
                    for q in quadrant_order}
    total_n = sum(total_counts.values())
    avg_pcts = {
        q: round(total_counts[q] / total_n, 4) if total_n > 0 else 0.0
        for q in quadrant_order
    }
    return {
        "model": "Average",
        "n_classes": total_n,
        "quadrants": {q: {"pct": v, "n": total_counts[q]} for q, v in avg_pcts.items()},
    }


def _draw_heatmap_ax(ax: plt.Axes, summaries: list[dict], title: str) -> None:
    """Draw a stacked bar heatmap into *ax* (no legend — caller adds it).
    Appends an average bar at the right, separated by a small gap."""
    quadrant_order = ["both_pass", "lang_only", "vision_only", "both_fail"]

    avg = _average_summary(summaries)
    all_summaries = summaries + [avg]

    models = [s["model"].split("/")[-1] for s in all_summaries]
    n = len(all_summaries)
    x = np.arange(n)
    bottoms = np.zeros(n)

    for q in quadrant_order:
        vals = np.array([s["quadrants"].get(q, {}).get("pct", 0.0) for s in all_summaries])
        # Slightly narrower bar for the average to visually separate it
        widths = np.array([0.3] * (n - 1) + [0.25])
        bars = ax.bar(x, vals, widths, bottom=bottoms, color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        for bar, v in zip(bars, vals):
            if v >= 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + v / 2,
                    f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold",
                )
        bottoms += vals

    # Vertical separator before the average bar
    ax.axvline(n - 1.6, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    # Bold the "Average" tick label
    for tick, label in zip(ax.get_xticklabels(), models):
        if label == "Average":
            tick.set_fontweight("bold")
    ax.set_ylabel("Fraction of classes")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)


def _plot_combined_heatmap(
    cls_summaries: list[dict],
    desc_summaries: list[dict],
    path: Path,
) -> None:
    """Side-by-side classification + description heatmaps with a single shared legend below."""
    n_models = max(len(cls_summaries), len(desc_summaries), 1)
    panel_w = max(4, n_models * 1.1)
    fig, (ax_cls, ax_desc) = plt.subplots(1, 2, figsize=(panel_w * 2, 5.5))
    fig.suptitle("Per-class error overlap across models", fontsize=12, fontweight="bold")

    _draw_heatmap_ax(ax_cls,  cls_summaries,  "Classification")
    _draw_heatmap_ax(ax_desc, desc_summaries, "Description (ROUGE-SS)")

    legend_patches = [
        mpatches.Patch(color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        for q in ["both_pass", "lang_only", "vision_only", "both_fail"]
    ]
    fig.legend(
        handles=legend_patches, fontsize=9,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Classification × Description interaction matrix ───────────────────────────

_QUADRANT_ORDER = ["both_pass", "lang_only", "vision_only", "both_fail"]
_QUADRANT_SHORT = {
    "both_pass":   "Both\npass",
    "lang_only":   "Lang\nonly",
    "vision_only": "Vision\nonly",
    "both_fail":   "Both\nfail",
}


def interaction_matrix(
    cls_records: list[dict],
    desc_records: list[dict],
) -> np.ndarray:
    """
    4×4 count matrix  [cls_quadrant, desc_quadrant].
    Rows = classification quadrant, columns = description quadrant.
    Only classes present in both lists are counted.
    """
    cls_map  = {(r["agml_dataset"], r["class_label"]): r["quadrant"] for r in cls_records}
    desc_map = {(r["agml_dataset"], r["class_label"]): r["quadrant"] for r in desc_records}
    shared   = set(cls_map) & set(desc_map)

    mat = np.zeros((4, 4), dtype=int)
    for key in shared:
        row = _QUADRANT_ORDER.index(cls_map[key])
        col = _QUADRANT_ORDER.index(desc_map[key])
        mat[row, col] += 1
    return mat


def _plot_interaction_heatmap(
    cls_records: list[dict],
    desc_records: list[dict],
    model_id: str,
    path: Path,
) -> None:
    """4×4 row-normalised interaction heatmap for a single model."""
    mat = interaction_matrix(cls_records, desc_records)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _draw_interaction_ax(
        ax, mat,
        title=(
            f"Cls × Desc interaction — {model_id.split('/')[-1]}\n"
            f"(row = cls quadrant  →  col = desc quadrant  |  n={mat.sum()} classes)"
        ),
        show_counts=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _draw_interaction_ax(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    show_counts: bool = True,
) -> None:
    """Draw a single row-normalised 4×4 interaction matrix into *ax*."""
    row_totals = mat.sum(axis=1, keepdims=True).clip(min=1)
    norm = mat / row_totals

    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [_QUADRANT_SHORT[q] for q in _QUADRANT_ORDER]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Description quadrant", fontsize=9)
    ax.set_ylabel("Classification quadrant", fontsize=9)
    ax.set_title(title, fontsize=9)

    for r in range(4):
        for c in range(4):
            frac = norm[r, c]
            text_color = "white" if frac > 0.55 else "black"
            cell_text = f"{frac:.0%}\n({mat[r,c]:.0f})" if show_counts else f"{frac:.0%}"
            ax.text(c, r, cell_text,
                    ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    for tick, q in zip(ax.get_xticklabels(), _QUADRANT_ORDER):
        tick.set_color(QUADRANT_COLORS[q])
    for tick, q in zip(ax.get_yticklabels(), _QUADRANT_ORDER):
        tick.set_color(QUADRANT_COLORS[q])


def _plot_interaction_heatmaps_combined(
    cls_records_by_model: dict[str, list[dict]],
    desc_records_by_model: dict[str, list[dict]],
    path: Path,
) -> None:
    """
    One panel per model + one averaged panel at the end.

    The average matrix sums raw counts across all models before row-normalising,
    so every class observation is weighted equally (not every model equally).
    """
    shared = sorted(set(cls_records_by_model) & set(desc_records_by_model))
    if not shared:
        return

    n_panels = len(shared) + 1   # +1 for the average
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle("Classification × Description quadrant interaction",
                 fontsize=12, fontweight="bold")

    avg_mat = np.zeros((4, 4), dtype=float)
    for ax, model_id in zip(axes, shared):
        mat = interaction_matrix(cls_records_by_model[model_id], desc_records_by_model[model_id])
        avg_mat += mat
        _draw_interaction_ax(
            ax, mat,
            title=f"{model_id.split('/')[-1]}\n(n={mat.sum()} classes)",
            show_counts=True,
        )

    # Average panel — counts are summed across models, show fractions only
    _draw_interaction_ax(
        axes[-1], avg_mat,
        title=f"Average across {len(shared)} models\n(n={avg_mat.sum():.0f} total observations)",
        show_counts=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── File discovery ─────────────────────────────────────────────────────────────

def _discover_pairs(lang_dir: Path, vis_dir: Path, prefix: str) -> list[tuple[str, Path, Path]]:
    lang_files = {p.name: p for p in lang_dir.glob(f"{prefix}_*.csv")}
    vis_files  = {p.name: p for p in vis_dir.glob(f"{prefix}_*.csv")}
    shared = sorted(set(lang_files) & set(vis_files))
    return [
        (name[len(prefix) + 1:-len(".csv")], lang_files[name], vis_files[name])
        for name in shared
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error overlap analysis (classification + description) between assessments."
    )
    parser.add_argument("--language_dir",  default="outputs/language_assessment/",
                        help="Directory with cls_*.csv and desc_*.csv from language assessment")
    parser.add_argument("--vision_dir",    default="outputs/vision_assessment/",
                        help="Directory with cls_*.csv and desc_*.csv from vision assessment")
    parser.add_argument("--output_dir",    default="outputs/error_overlap/",
                        help="Where to write outputs")
    parser.add_argument("--threshold",     type=float, default=_CLS_THRESHOLD,
                        help=f"Per-class accuracy threshold for classification pass/fail (default: {_CLS_THRESHOLD})")
    parser.add_argument("--rouge_threshold", type=float, default=None,
                        help="Override ROUGE-SS threshold (default: naive baseline from metrics JSON)")
    args = parser.parse_args()

    lang_dir   = Path(args.language_dir)
    vis_dir    = Path(args.vision_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine ROUGE threshold ──────────────────────────────────────────────
    if args.rouge_threshold is not None:
        rouge_thr = args.rouge_threshold
        print(f"ROUGE-SS threshold (override): {rouge_thr:.3f}")
    else:
        # Average naive baseline across both assessments
        lang_rouge = _naive_rouge_from_json(lang_dir / "language_assessment_metrics.json")
        vis_rouge  = _naive_rouge_from_json(vis_dir  / "vision_assessment_metrics.json")
        candidates = [v for v in (lang_rouge, vis_rouge) if v is not None]
        if candidates:
            rouge_thr = round(float(np.mean(candidates)), 4)
            print(f"ROUGE-SS threshold (naive baseline): {rouge_thr:.3f}"
                  + (f"  (lang={lang_rouge:.3f}, vis={vis_rouge:.3f})" if len(candidates) == 2 else ""))
        else:
            rouge_thr = _ROUGE_THRESHOLD
            print(f"ROUGE-SS threshold (fallback default): {rouge_thr:.3f}")

    # ── Classification overlap ─────────────────────────────────────────────────
    cls_pairs = _discover_pairs(lang_dir, vis_dir, "cls")
    cls_summaries = []
    cls_records_by_model: dict[str, list[dict]] = {}
    if cls_pairs:
        print(f"\n=== Classification overlap ({len(cls_pairs)} model(s)) ===")
        for safe_name, lang_csv, vis_csv in cls_pairs:
            lang_scores = per_class_accuracy(_load_csv(lang_csv))
            vis_scores  = per_class_accuracy(_load_csv(vis_csv))
            result = analyze_model(
                lang_scores, vis_scores, safe_name, output_dir,
                threshold=args.threshold,
                prefix="cls",
                x_label="Language accuracy (per class)",
                y_label="Vision accuracy (per class)",
                score_label="accuracy",
            )
            if result:
                s, records = result
                cls_summaries.append(s)
                cls_records_by_model[safe_name] = records
    else:
        print("No matching cls_*.csv files found — skipping classification overlap.")

    # ── Description overlap ────────────────────────────────────────────────────
    desc_pairs = _discover_pairs(lang_dir, vis_dir, "desc")
    desc_summaries = []
    desc_records_by_model: dict[str, list[dict]] = {}
    if desc_pairs:
        print(f"\n=== Description overlap ({len(desc_pairs)} model(s)) ===")
        for safe_name, lang_csv, vis_csv in desc_pairs:
            lang_scores = per_class_rouge(_load_csv(lang_csv))
            vis_scores  = per_class_rouge(_load_csv(vis_csv))
            result = analyze_model(
                lang_scores, vis_scores, safe_name, output_dir,
                threshold=rouge_thr,
                prefix="desc",
                x_label="Language mean ROUGE-SS (per class)",
                y_label="Vision mean ROUGE-SS (per class)",
                score_label="ROUGE-SS",
            )
            if result:
                s, records = result
                desc_summaries.append(s)
                desc_records_by_model[safe_name] = records
    else:
        print("No matching desc_*.csv files found — skipping description overlap.")

    # ── Combined scatter plots (one per model that has both) ───────────────────
    shared_models = set(cls_records_by_model) & set(desc_records_by_model)
    if shared_models:
        print(f"\n=== Combined scatter plots ({len(shared_models)} model(s)) ===")
        for safe_name in sorted(shared_models):
            path = output_dir / f"error_overlap_{safe_name}.png"
            _plot_combined_scatter(
                cls_records_by_model[safe_name],
                desc_records_by_model[safe_name],
                safe_name, path,
                cls_thr=args.threshold,
                rouge_thr=rouge_thr,
            )

    # ── Classification × Description interaction ──────────────────────────────
    if shared_models:
        print(f"\n=== Cls × Desc interaction ({len(shared_models)} model(s)) ===")
        for safe_name in sorted(shared_models):
            path = output_dir / f"error_overlap_interaction_{safe_name}.png"
            _plot_interaction_heatmap(
                cls_records_by_model[safe_name],
                desc_records_by_model[safe_name],
                safe_name, path,
            )
        if len(shared_models) > 1:
            _plot_interaction_heatmaps_combined(
                cls_records_by_model, desc_records_by_model,
                output_dir / "error_overlap_interaction_combined.png",
            )

    # ── Summaries + heatmaps ───────────────────────────────────────────────────
    if cls_summaries:
        p = output_dir / "error_overlap_cls_summary.json"
        p.write_text(json.dumps(cls_summaries, indent=2), encoding="utf-8")
        print(f"\n  Saved: {p}")

    if desc_summaries:
        p = output_dir / "error_overlap_desc_summary.json"
        p.write_text(json.dumps(desc_summaries, indent=2), encoding="utf-8")
        print(f"  Saved: {p}")

    if cls_summaries or desc_summaries:
        _plot_combined_heatmap(
            cls_summaries, desc_summaries,
            output_dir / "error_overlap_heatmap.png",
        )

    # ── Print tables ───────────────────────────────────────────────────────────
    for label, summaries in [("Classification", cls_summaries), ("Description (ROUGE-SS)", desc_summaries)]:
        if not summaries:
            continue
        print(f"\n── {label} ──")
        print(f"  {'Model':<38} {'Both pass':>10} {'Lang only':>10} {'Vision only':>12} {'Both fail':>10}")
        print(f"  {'-'*82}")
        for s in summaries:
            q = s["quadrants"]
            print(
                f"  {s['model']:<38}"
                f" {q.get('both_pass',  {}).get('pct', 0):>10.1%}"
                f" {q.get('lang_only',  {}).get('pct', 0):>10.1%}"
                f" {q.get('vision_only',{}).get('pct', 0):>12.1%}"
                f" {q.get('both_fail',  {}).get('pct', 0):>10.1%}"
            )


if __name__ == "__main__":
    main()
