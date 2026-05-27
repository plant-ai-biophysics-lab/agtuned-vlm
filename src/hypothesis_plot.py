"""
Hypothesis matrix visualisation for the agtuned-vlm results table.

Reads  <input_dir>/results_table.csv  and produces one PNG:
  hypothesis_matrix.png — LP F1 | Lang Cls F1 | Lang ROUGE-SS | Vision Cls F1 | Vision ROUGE-SS | Case

Each cell is coloured green (≥ threshold) or red (< threshold).
Thresholds default to the naive baseline values from the CSV.
CLI args override individual columns when provided.

Case is determined per-assessment: an assessment is "high" only if BOTH its
classification AND description metrics are high. If only one metric is available
for an assessment, that metric alone determines high/low.

  Case 1 — LP ✓, Vision ✓          → No bottleneck
  Case 2 — LP ✓, Lang ✓, Vision ✗  → H3 Alignment bottleneck
  Case 3 — LP ✓, Lang ✗, Vision ✗  → H2 + H3  (Language + Alignment)
  Case 4 — LP ✗                     → H1  Vision encoder bottleneck

Usage:
    python main.py hypothesis \\
        --input_dir  outputs/ \\
        --output_dir outputs/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ── Defaults (only used when no naive baseline row exists in CSV) ──────────────

_LP_THRESHOLD        = 0.60
_LANG_CLS_THRESHOLD  = 0.10
_LANG_DESC_THRESHOLD = 0.10
_VIS_CLS_THRESHOLD   = 0.10
_VIS_DESC_THRESHOLD  = 0.10

_HIGH_COLOR = "#4caf50"
_LOW_COLOR  = "#f44336"
_NA_COLOR   = "#e0e0e0"
_TEXT_HIGH  = "white"
_TEXT_LOW   = "white"
_TEXT_NA    = "#757575"

_TYPE_ORDER = {"Baseline": 0, "Vision": 1, "Language": 2, "Multimodal": 3}


# ── CSV loading ────────────────────────────────────────────────────────────────

def _fv(s: str) -> float | None:
    s = s.strip()
    if not s or s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_table(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw.get("Type") == "Type":
                continue
            rows.append({
                "type":          raw.get("Type") or raw.get("type", ""),
                "model":         raw.get("Model") or raw.get("model", ""),
                "params":        raw.get("Params") or raw.get("params", "--"),
                "lp_f1":         _fv(raw.get("Linear Probe F1") or raw.get("lp_f1", "")),
                "lang_cls_f1":   _fv(raw.get("Lang F1") or raw.get("lang_cls_f1", "")),
                "lang_desc_rss": _fv(raw.get("Lang ROUGE-SS") or raw.get("lang_desc_rss", "")),
                "vis_cls_f1":    _fv(raw.get("Vision F1") or raw.get("vis_cls_f1", "")),
                "vis_desc_rss":  _fv(raw.get("Vision ROUGE-SS") or raw.get("vis_desc_rss", "")),
            })
    rows.sort(key=lambda r: (_TYPE_ORDER.get(r["type"], 99), r["model"]))
    return rows


# ── Threshold auto-detection ───────────────────────────────────────────────────

def naive_thresholds(rows: list[dict], overrides: dict[str, float | None]) -> dict[str, float]:
    """Per-column thresholds from the Baseline row; overrides take precedence."""
    baseline = next((r for r in rows if r["type"] == "Baseline"), None)

    def _pick(col: str, default: float) -> float:
        ov = overrides.get(col)
        if ov is not None:
            return ov
        if baseline and baseline.get(col) is not None:
            return baseline[col]
        return default

    return {
        "lp_f1":         _pick("lp_f1",         _LP_THRESHOLD),
        "lang_cls_f1":   _pick("lang_cls_f1",   _LANG_CLS_THRESHOLD),
        "lang_desc_rss": _pick("lang_desc_rss", _LANG_DESC_THRESHOLD),
        "vis_cls_f1":    _pick("vis_cls_f1",    _VIS_CLS_THRESHOLD),
        "vis_desc_rss":  _pick("vis_desc_rss",  _VIS_DESC_THRESHOLD),
    }


# ── Case classification ────────────────────────────────────────────────────────

def _hi(val: float | None, threshold: float) -> bool | None:
    if val is None:
        return None
    return val >= threshold


def _assessment_hi(row: dict, cls_key: str, desc_key: str, thr: dict[str, float]) -> bool | None:
    """
    An assessment is "high" only if ALL available metrics for it are >= threshold.
    Returns None when no metrics exist for this assessment.
    """
    results = []
    for key in (cls_key, desc_key):
        h = _hi(row[key], thr[key])
        if h is not None:
            results.append(h)
    if not results:
        return None
    return all(results)


def classify_case(row: dict, thr: dict[str, float]) -> str:
    if row["type"] == "Baseline":
        return "Baseline"

    lp_hi   = _hi(row["lp_f1"], thr["lp_f1"])
    lang_hi = _assessment_hi(row, "lang_cls_f1", "lang_desc_rss", thr)
    vis_hi  = _assessment_hi(row, "vis_cls_f1",  "vis_desc_rss",  thr)

    # Vision-encoder only
    if lp_hi is not None and lang_hi is None and vis_hi is None:
        return "LP only"

    # Language-only (no LP, no vision)
    if lp_hi is None and vis_hi is None:
        if lang_hi is True:
            return "Lang ✓"
        elif lang_hi is False:
            return "Lang ✗"
        return "—"

    # Multimodal without LP
    if lp_hi is None:
        if vis_hi is True:
            return "Vision ✓"
        elif vis_hi is False:
            return "Vision ✗"
        return "—"

    # Full multimodal (LP present)
    if lp_hi is False:
        return "Case 4\nH1 bottleneck"
    if vis_hi is True:
        return "Case 1\nNo bottleneck"
    if lang_hi is True:
        return "Case 2\nH3 Alignment"
    return "Case 3\nH2+H3"


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _cell(val: float | None, threshold: float) -> tuple[str, str]:
    if val is None:
        return _NA_COLOR, _TEXT_NA
    return (_HIGH_COLOR, _TEXT_HIGH) if val >= threshold else (_LOW_COLOR, _TEXT_LOW)


def _short(model_id: str, max_len: int = 28) -> str:
    base = model_id.split("/")[-1]
    return base if len(base) <= max_len else base[:max_len - 1] + "…"


# ── Main chart ────────────────────────────────────────────────────────────────

_COL_KEYS   = ["lp_f1", "lang_cls_f1", "lang_desc_rss", "vis_cls_f1", "vis_desc_rss"]
_COL_LABELS = ["LP\nF1", "Lang\nCls F1", "Lang\nROUGE-SS", "Vision\nCls F1", "Vision\nROUGE-SS"]
# Column widths in relative units (model col = 2.5, metric cols = 1, case col = 1.6)
_MODEL_W = 2.5
_METRIC_W = 1.0
_CASE_W = 1.6


def plot_matrix(rows: list[dict], output_dir: Path, thr: dict[str, float]) -> None:
    n_rows  = len(rows)
    n_metrics = len(_COL_KEYS)
    total_w = _MODEL_W + n_metrics * _METRIC_W + _CASE_W

    fig_w = total_w * 1.1
    fig_h = max(4.5, 0.6 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")
    ax.set_title("Hypothesis Matrix", fontsize=13, fontweight="bold", pad=10)

    # Column x-positions (left edge of each column)
    col_x = [0.0]
    col_x.append(col_x[-1] + _MODEL_W)
    for _ in _COL_KEYS:
        col_x.append(col_x[-1] + _METRIC_W)
    case_x = col_x[-1]  # left edge of case column

    # Column centres for header text
    model_cx  = _MODEL_W / 2
    metric_cx = [col_x[i + 1] + _METRIC_W / 2 for i in range(n_metrics)]
    case_cx   = case_x + _CASE_W / 2

    # ── Column group separators in header ─────────────────────────────────────
    # Draw header background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, n_rows), total_w, 1,
        boxstyle="square,pad=0", facecolor="#37474f", edgecolor="white", linewidth=0.5, zorder=1,
    ))

    # Group label row (top half of header)
    ax.text(model_cx,  n_rows + 0.72, "Model",             ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=5)
    ax.text(metric_cx[0], n_rows + 0.72, "Linear Probe",   ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=5)
    # Lang group spans cols 1-2
    lang_span_cx = (col_x[2] + col_x[4]) / 2
    ax.text(lang_span_cx, n_rows + 0.72, "Language Assessment", ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=5)
    # Vision group spans cols 3-4
    vis_span_cx = (col_x[4] + col_x[6]) / 2
    ax.text(vis_span_cx,  n_rows + 0.72, "Vision Assessment",   ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=5)
    ax.text(case_cx,      n_rows + 0.72, "Case",                ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=5)

    # Sub-label row (bottom half of header)
    for i, label in enumerate(_COL_LABELS):
        ax.text(metric_cx[i], n_rows + 0.28, label, ha="center", va="center", fontsize=6.5, color="#cfd8dc", zorder=5)

    # Thin separator line between group label and sub-label rows
    ax.axhline(n_rows + 0.48, color="#546e7a", linewidth=0.5, zorder=4)
    # Vertical dividers between groups inside header
    for xd in (col_x[2], col_x[4], case_x):
        ax.plot([xd, xd], [n_rows, n_rows + 1], color="#546e7a", linewidth=0.8, zorder=4)

    # ── Data rows ─────────────────────────────────────────────────────────────
    prev_type = None
    for ri, row in enumerate(rows):
        y = n_rows - 1 - ri

        if row["type"] != prev_type and prev_type is not None:
            ax.axhline(y + 1, color="#90a4ae", linewidth=0.8, zorder=3)
        prev_type = row["type"]

        row_bg = "#fafafa" if ri % 2 == 0 else "#f0f4f8"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y), total_w, 1,
            boxstyle="square,pad=0", facecolor=row_bg, edgecolor="none", zorder=0,
        ))

        # Model label
        label = _short(row["model"])
        if row["params"] and row["params"] not in ("--", ""):
            label += f"\n({row['params']})"
        ax.text(0.1, y + 0.5, label, ha="left", va="center", fontsize=6.5, color="#212121", zorder=5)

        # Metric cells
        for i, key in enumerate(_COL_KEYS):
            val = row[key]
            fc, tc = _cell(val, thr[key])
            cx = metric_cx[i]
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - _METRIC_W * 0.42, y + 0.12), _METRIC_W * 0.84, 0.76,
                boxstyle="round,pad=0.02", facecolor=fc, edgecolor="white", linewidth=0.5, zorder=2,
            ))
            ax.text(cx, y + 0.5, f"{val:.3f}" if val is not None else "—",
                    ha="center", va="center", fontsize=7, color=tc, fontweight="bold", zorder=5)

        # Case cell
        case_text = classify_case(row, thr)
        if "H1" in case_text:
            case_bg = "#fff3e0"
        elif "H2" in case_text or "H3" in case_text:
            case_bg = "#fce4ec"
        elif "Case 1" in case_text or "No bottle" in case_text:
            case_bg = "#e8f5e9"
        else:
            case_bg = row_bg
        ax.add_patch(mpatches.FancyBboxPatch(
            (case_x + 0.06, y + 0.08), _CASE_W - 0.12, 0.84,
            boxstyle="round,pad=0.02", facecolor=case_bg, edgecolor="#b0bec5", linewidth=0.5, zorder=2,
        ))
        ax.text(case_cx, y + 0.5, case_text, ha="center", va="center",
                fontsize=6, color="#212121", zorder=5)

    # Outer border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), total_w, n_rows + 1,
        boxstyle="square,pad=0", fill=False, edgecolor="#37474f", linewidth=1.2, zorder=6,
    ))

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_HIGH_COLOR, label="High (≥ naive baseline)"),
        mpatches.Patch(facecolor=_LOW_COLOR,  label="Low  (< naive baseline)"),
        mpatches.Patch(facecolor=_NA_COLOR,   label="N/A"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
              ncol=3, fontsize=7, frameon=False)

    # Threshold footnote
    fig.text(
        0.5, 0.005,
        f"Naive baseline thresholds — LP: {thr['lp_f1']:.3f}  |  "
        f"Lang Cls: {thr['lang_cls_f1']:.3f}  Lang ROUGE-SS: {thr['lang_desc_rss']:.3f}  |  "
        f"Vision Cls: {thr['vis_cls_f1']:.3f}  Vision ROUGE-SS: {thr['vis_desc_rss']:.3f}",
        ha="center", fontsize=6.5, color="#616161",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "hypothesis_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hypothesis matrix chart from results_table.csv."
    )
    parser.add_argument("--input_dir",  default="outputs/", help="Directory containing results_table.csv")
    parser.add_argument("--output_dir", default="outputs/", help="Directory to write hypothesis_matrix.png")
    parser.add_argument("--lp_threshold",        type=float, default=None, help="Override LP F1 threshold")
    parser.add_argument("--lang_cls_threshold",  type=float, default=None, help="Override Lang Cls F1 threshold")
    parser.add_argument("--lang_desc_threshold", type=float, default=None, help="Override Lang ROUGE-SS threshold")
    parser.add_argument("--vis_cls_threshold",   type=float, default=None, help="Override Vision Cls F1 threshold")
    parser.add_argument("--vis_desc_threshold",  type=float, default=None, help="Override Vision ROUGE-SS threshold")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "results_table.csv"
    if not csv_path.exists():
        print(f"ERROR: results_table.csv not found in {input_dir}")
        sys.exit(1)

    rows = load_table(csv_path)
    if not rows:
        print("ERROR: results_table.csv is empty or could not be parsed.")
        sys.exit(1)

    thr = naive_thresholds(rows, {
        "lp_f1":         args.lp_threshold,
        "lang_cls_f1":   args.lang_cls_threshold,
        "lang_desc_rss": args.lang_desc_threshold,
        "vis_cls_f1":    args.vis_cls_threshold,
        "vis_desc_rss":  args.vis_desc_threshold,
    })

    print(f"Loaded {len(rows)} row(s) from {csv_path}")
    print("Thresholds (naive baseline):")
    print(f"  LP F1:           {thr['lp_f1']:.3f}")
    print(f"  Lang Cls F1:     {thr['lang_cls_f1']:.3f}")
    print(f"  Lang ROUGE-SS:   {thr['lang_desc_rss']:.3f}")
    print(f"  Vision Cls F1:   {thr['vis_cls_f1']:.3f}")
    print(f"  Vision ROUGE-SS: {thr['vis_desc_rss']:.3f}")
    print()

    plot_matrix(rows, output_dir, thr)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
