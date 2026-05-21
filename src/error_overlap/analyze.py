"""
Per-class error overlap analysis between language and vision assessments.

For each model that was evaluated in both assessments, computes per-class
accuracy in each modality and places every class into one of four quadrants:

  Both pass   — language correct  + vision correct    → model is aligned
  Lang only   — language correct  + vision wrong      → alignment broken
  Vision only — language wrong    + vision correct    → visual knowledge compensates
  Both fail   — language wrong    + vision wrong      → knowledge bottleneck

Outputs (per model):
  error_overlap_<model>.csv        — per-class quadrant table
  error_overlap_<model>.png        — scatter + quadrant chart
  error_overlap_summary.json       — aggregate quadrant counts across models
  error_overlap_heatmap.png        — heatmap of quadrant fractions per model

Usage:
    python main.py error-overlap \
        --language_dir  outputs/language_assessment/ \
        --vision_dir    outputs/vision_assessment/ \
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

# ── CSV loading ────────────────────────────────────────────────────────────────

def _load_cls_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _correct(row: dict) -> bool:
    """Use judged_correct when available, fall back to correct."""
    if "judged_correct" in row and row["judged_correct"] != "":
        return str(row["judged_correct"]) == "1"
    return str(row.get("correct", "0")) == "1"


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


# ── Per-class accuracy ─────────────────────────────────────────────────────────

def per_class_accuracy(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Returns {(agml_dataset, class_label): {accuracy, n}} for each class."""
    by_class: dict[tuple, list[bool]] = defaultdict(list)
    for row in rows:
        key = (row["agml_dataset"], row["class_label"])
        by_class[key].append(_correct(row))
    return {
        key: {"accuracy": round(sum(hits) / len(hits), 4), "n": len(hits)}
        for key, hits in by_class.items()
    }


# ── Quadrant assignment ────────────────────────────────────────────────────────

_THRESHOLD = 0.5  # accuracy above this = "pass"

QUADRANT_LABELS = {
    "both_pass":   "Both pass",
    "lang_only":   "Lang only (alignment broken)",
    "vision_only": "Vision only (visual knowledge compensates)",
    "both_fail":   "Both fail (knowledge bottleneck)",
}

QUADRANT_COLORS = {
    "both_pass":   "#2ecc71",
    "lang_only":   "#e74c3c",
    "vision_only": "#3498db",
    "both_fail":   "#95a5a6",
}


def assign_quadrant(lang_acc: float, vis_acc: float, threshold: float = _THRESHOLD) -> str:
    lang_pass = lang_acc >= threshold
    vis_pass  = vis_acc  >= threshold
    if lang_pass and vis_pass:
        return "both_pass"
    if lang_pass and not vis_pass:
        return "lang_only"
    if not lang_pass and vis_pass:
        return "vision_only"
    return "both_fail"


# ── Per-model analysis ─────────────────────────────────────────────────────────

def analyze_model(
    lang_rows: list[dict],
    vis_rows: list[dict],
    model_id: str,
    output_dir: Path,
) -> dict:
    lang_acc = per_class_accuracy(lang_rows)
    vis_acc  = per_class_accuracy(vis_rows)

    shared_keys = set(lang_acc) & set(vis_acc)
    if not shared_keys:
        print(f"  [{model_id}] No overlapping (dataset, class) pairs — skipping.")
        return {}

    records = []
    quadrant_counts = defaultdict(int)
    for key in sorted(shared_keys):
        dataset, cls = key
        la = lang_acc[key]["accuracy"]
        va = vis_acc[key]["accuracy"]
        q  = assign_quadrant(la, va)
        quadrant_counts[q] += 1
        records.append({
            "agml_dataset": dataset,
            "class_label":  cls,
            "task_type":    lang_rows[0].get("task_type", "") if lang_rows else "",
            "lang_accuracy": la,
            "vis_accuracy":  va,
            "quadrant":      q,
            "lang_n":        lang_acc[key]["n"],
            "vis_n":         vis_acc[key]["n"],
        })

    # Attach task_type per class from rows
    task_map: dict[tuple, str] = {}
    for row in lang_rows:
        task_map[(row["agml_dataset"], row["class_label"])] = row.get("task_type", "")
    for r in records:
        r["task_type"] = task_map.get((r["agml_dataset"], r["class_label"]), "")

    total = len(records)
    summary = {
        "model":  model_id,
        "n_classes": total,
        "quadrants": {q: {"n": n, "pct": round(n / total, 3)} for q, n in quadrant_counts.items()},
    }

    _write_csv(records, output_dir / f"error_overlap_{_safe_name(model_id)}.csv")
    _plot_scatter(records, model_id, output_dir / f"error_overlap_{_safe_name(model_id)}.png")

    print(f"  [{model_id}]  n={total}  "
          + "  ".join(f"{q}={n}" for q, n in quadrant_counts.items()))

    return summary


# ── Output helpers ─────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "agml_dataset", "class_label", "task_type",
    "lang_accuracy", "vis_accuracy", "quadrant", "lang_n", "vis_n",
]


def _write_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {path}")


def _plot_scatter(records: list[dict], model_id: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for r in records:
        color = QUADRANT_COLORS[r["quadrant"]]
        ax.scatter(r["lang_accuracy"], r["vis_accuracy"],
                   color=color, alpha=0.75, s=60, zorder=3)

    # Quadrant dividers
    ax.axvline(_THRESHOLD, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(_THRESHOLD, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Quadrant shading
    ax.axhspan(0, _THRESHOLD, xmin=0, xmax=_THRESHOLD, color=QUADRANT_COLORS["both_fail"],   alpha=0.06)
    ax.axhspan(0, _THRESHOLD, xmin=_THRESHOLD, xmax=1,  color=QUADRANT_COLORS["lang_only"],   alpha=0.06)
    ax.axhspan(_THRESHOLD, 1, xmin=0, xmax=_THRESHOLD,  color=QUADRANT_COLORS["vision_only"], alpha=0.06)
    ax.axhspan(_THRESHOLD, 1, xmin=_THRESHOLD, xmax=1,  color=QUADRANT_COLORS["both_pass"],   alpha=0.06)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Language accuracy (per class)", fontsize=11)
    ax.set_ylabel("Vision accuracy (per class)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Error overlap — {model_id.split('/')[-1]}", fontsize=11)

    legend_patches = [
        mpatches.Patch(color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        for q in ["both_pass", "lang_only", "vision_only", "both_fail"]
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_heatmap(summaries: list[dict], path: Path) -> None:
    """Stacked bar chart of quadrant fractions per model."""
    if not summaries:
        return

    quadrant_order = ["both_pass", "lang_only", "vision_only", "both_fail"]
    models = [s["model"].split("/")[-1] for s in summaries]
    x = np.arange(len(models))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.8), 5))
    bottoms = np.zeros(len(models))

    for q in quadrant_order:
        vals = np.array([s["quadrants"].get(q, {}).get("pct", 0.0) for s in summaries])
        ax.bar(x, vals, width, bottom=bottoms,
               color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of classes")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class error overlap across models")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Model discovery ────────────────────────────────────────────────────────────

def _discover_model_pairs(lang_dir: Path, vis_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Find cls_*.csv files present in both dirs with matching suffixes.
    Returns [(model_safe_name, lang_csv, vis_csv), ...].
    """
    lang_files = {p.name: p for p in lang_dir.glob("cls_*.csv")}
    vis_files  = {p.name: p for p in vis_dir.glob("cls_*.csv")}
    shared = sorted(set(lang_files) & set(vis_files))
    if not shared:
        return []
    # Recover a display model_id from the safe filename
    pairs = []
    for name in shared:
        safe = name[len("cls_"):-len(".csv")]
        pairs.append((safe, lang_files[name], vis_files[name]))
    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-class error overlap analysis between language and vision assessments."
    )
    parser.add_argument("--language_dir", default="outputs/language_assessment/",
                        help="Directory containing cls_*.csv files from language assessment")
    parser.add_argument("--vision_dir",   default="outputs/vision_assessment/",
                        help="Directory containing cls_*.csv files from vision assessment")
    parser.add_argument("--output_dir",   default="outputs/error_overlap/",
                        help="Where to write per-model CSVs, scatter plots, and summary JSON")
    parser.add_argument("--threshold",    type=float, default=_THRESHOLD,
                        help=f"Per-class accuracy threshold for pass/fail (default: {_THRESHOLD})")
    args = parser.parse_args()

    lang_dir   = Path(args.language_dir)
    vis_dir    = Path(args.vision_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_model_pairs(lang_dir, vis_dir)
    if not pairs:
        print("ERROR: no matching cls_*.csv files found in both directories.")
        sys.exit(1)

    print(f"Found {len(pairs)} model(s) evaluated in both assessments:")
    for safe, lp, vp in pairs:
        print(f"  {safe}")

    global _THRESHOLD
    _THRESHOLD = args.threshold

    summaries = []
    for safe_name, lang_csv, vis_csv in pairs:
        print(f"\n=== {safe_name} ===")
        lang_rows = _load_cls_csv(lang_csv)
        vis_rows  = _load_cls_csv(vis_csv)
        summary = analyze_model(lang_rows, vis_rows, safe_name, output_dir)
        if summary:
            summaries.append(summary)

    if not summaries:
        print("ERROR: no overlapping classes found for any model.")
        sys.exit(1)

    summary_path = output_dir / "error_overlap_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n  Saved: {summary_path}")

    _plot_heatmap(summaries, output_dir / "error_overlap_heatmap.png")

    # Print aggregate table
    print(f"\n{'Model':<40} {'Both pass':>10} {'Lang only':>10} {'Vision only':>12} {'Both fail':>10}")
    print(f"  {'-'*86}")
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
