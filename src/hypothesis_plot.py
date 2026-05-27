"""
Hypothesis matrix visualisation for the agtuned-vlm results table.

Reads  <input_dir>/results_table.csv  and produces two PNGs:
  hypothesis_cls.png   — Classification view  (LP | Lang Cls F1 | Vision Cls F1)
  hypothesis_desc.png  — Description view     (Lang ROUGE-SS | Vision ROUGE-SS)

Each cell is coloured green (≥ threshold) or red (< threshold).
Thresholds default to the naive baseline values read from the CSV (one per column).
CLI args override individual columns when provided.

Classification chart annotates a case label per model row:
  Case 1 — LP ✓, Vision ✓          → No bottleneck
  Case 2 — LP ✓, Lang ✓, Vision ✗  → H3 Alignment bottleneck
  Case 3 — LP ✓, Lang ✗, Vision ✗  → H2 + H3  (Language + Alignment)
  Case 4 — LP ✗                     → H1  Vision encoder bottleneck
  (Language-only models skip the LP column — case based on Lang alone.)

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
import numpy as np


# ── Defaults (only used when no naive baseline row exists in CSV) ──────────────

_LP_THRESHOLD        = 0.60
_LANG_CLS_THRESHOLD  = 0.10
_LANG_DESC_THRESHOLD = 0.10
_VIS_CLS_THRESHOLD   = 0.10
_VIS_DESC_THRESHOLD  = 0.10

_HIGH_COLOR = "#4caf50"   # green
_LOW_COLOR  = "#f44336"   # red
_NA_COLOR   = "#e0e0e0"   # light grey — metric not available
_TEXT_HIGH  = "white"
_TEXT_LOW   = "white"
_TEXT_NA    = "#757575"

_TYPE_ORDER = {"Baseline": 0, "Vision": 1, "Language": 2, "Multimodal": 3}


# ── CSV loading ────────────────────────────────────────────────────────────────

def _fv(s: str) -> float | None:
    """Parse a CSV cell to float, returning None on empty/dash."""
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
            # Skip the header-alias row that DictWriter writes as values
            if raw.get("Type") == "Type":
                continue
            type_val = raw.get("Type") or raw.get("type", "")
            model    = raw.get("Model") or raw.get("model", "")
            rows.append({
                "type":          type_val,
                "model":         model,
                "params":        raw.get("Params") or raw.get("params", "--"),
                "lp_f1":         _fv(raw.get("Linear Probe F1") or raw.get("lp_f1", "")),
                "lang_cls_f1":   _fv(raw.get("Lang F1") or raw.get("lang_cls_f1", "")),
                "lang_desc_rss": _fv(raw.get("Lang ROUGE-SS") or raw.get("lang_desc_rss", "")),
                "vis_cls_f1":    _fv(raw.get("Vision F1") or raw.get("vis_cls_f1", "")),
                "vis_desc_rss":  _fv(raw.get("Vision ROUGE-SS") or raw.get("vis_desc_rss", "")),
            })
    rows.sort(key=lambda r: (_TYPE_ORDER.get(r["type"], 99), r["model"]))
    return rows


# ── Threshold auto-detection ──────────────────────────────────────────────────

def naive_thresholds(rows: list[dict], overrides: dict[str, float | None]) -> dict[str, float]:
    """
    Build per-column thresholds from the Baseline row in *rows*.
    Each key in *overrides* takes precedence when not None.
    Falls back to module-level defaults if the baseline has no value for a column.
    """
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
    """Return True/False/None (None = data absent)."""
    if val is None:
        return None
    return val >= threshold


def classify_case(
    row: dict,
    thr: dict[str, float],
) -> str:
    """
    Classify the model into a hypothesis case.
    Returns a short label and a brief description.
    """
    lp_hi   = _hi(row["lp_f1"],      thr["lp_f1"])
    lang_hi = _hi(row["lang_cls_f1"], thr["lang_cls_f1"])
    vis_hi  = _hi(row["vis_cls_f1"],  thr["vis_cls_f1"])

    # Baseline row
    if row["type"] == "Baseline":
        return "Baseline"

    # Vision-encoder-only rows (no language metrics)
    if lp_hi is not None and lang_hi is None and vis_hi is None:
        return "LP only"

    # Language-only rows (no LP, no vision)
    if lp_hi is None and vis_hi is None:
        if lang_hi is True:
            return "Lang ✓"
        elif lang_hi is False:
            return "Lang ✗"
        return "—"

    # Multimodal rows
    if lp_hi is None:
        # No LP data — just show vision result
        if vis_hi is True:
            return "Vision ✓"
        elif vis_hi is False:
            return "Vision ✗"
        return "—"

    # Full multimodal
    if lp_hi is False:
        return "Case 4\nH1 bottleneck"
    # lp_hi is True
    if vis_hi is True:
        return "Case 1\nNo bottleneck"
    # lp_hi True, vis low
    if lang_hi is True:
        return "Case 2\nH3 Alignment"
    return "Case 3\nH2+H3"


# ── Cell colour helper ─────────────────────────────────────────────────────────

def _cell(val: float | None, threshold: float) -> tuple[str, str]:
    """Return (face_color, text_color) for a metric cell."""
    if val is None:
        return _NA_COLOR, _TEXT_NA
    if val >= threshold:
        return _HIGH_COLOR, _TEXT_HIGH
    return _LOW_COLOR, _TEXT_LOW


# ── Short model label ──────────────────────────────────────────────────────────

def _short(model_id: str, max_len: int = 30) -> str:
    base = model_id.split("/")[-1]
    return base if len(base) <= max_len else base[:max_len - 1] + "…"


# ── Shared drawing helper ──────────────────────────────────────────────────────

def _draw_table(
    ax: plt.Axes,
    rows: list[dict],
    col_keys: list[str],
    col_labels: list[str],
    thr: dict[str, float],
    extra_cols: list[tuple[str, str]] | None = None,  # [(label, text_per_row)]
    title: str = "",
    show_case: bool = False,
) -> None:
    """Draw a colour-coded table into *ax*."""
    n_rows = len(rows)
    n_data = len(col_keys)
    n_extra = len(extra_cols) if extra_cols else 0
    n_cols_total = 1 + n_data + n_extra + (1 if show_case else 0)  # model + data + extra + case

    ax.set_xlim(0, n_cols_total)
    ax.set_ylim(0, n_rows + 1)  # +1 for header
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    col_widths = [0] * n_cols_total  # unused (cells span 1 each)

    # ── Header row ─────────────────────────────────────────────────────────────
    header_cells = ["Model"] + col_labels + ([e[0] for e in extra_cols] if extra_cols else [])
    if show_case:
        header_cells.append("Case")

    for ci, label in enumerate(header_cells):
        ax.text(
            ci + 0.5, n_rows + 0.5, label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white",
        )
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, n_rows), n_cols_total, 1,
        boxstyle="square,pad=0", facecolor="#37474f", edgecolor="white", linewidth=0.5,
    ))
    for ci, label in enumerate(header_cells):
        ax.text(
            ci + 0.5, n_rows + 0.5, label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=5,
        )

    # ── Type separator tracking ────────────────────────────────────────────────
    prev_type = None

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        y = n_rows - 1 - ri  # draw top-to-bottom

        # Separator line between type groups
        if row["type"] != prev_type and prev_type is not None:
            ax.axhline(y + 1, color="#37474f", linewidth=1.0, zorder=3)
        prev_type = row["type"]

        # Alternating row background
        row_bg = "#fafafa" if ri % 2 == 0 else "#f0f0f0"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y), n_cols_total, 1,
            boxstyle="square,pad=0", facecolor=row_bg, edgecolor="none", zorder=0,
        ))

        # Model name cell (column 0)
        label = f"{_short(row['model'])}"
        if row["params"] and row["params"] not in ("--", ""):
            label += f"\n({row['params']})"
        ax.text(
            0.08, y + 0.5, label,
            ha="left", va="center", fontsize=6.5, color="#212121", zorder=5,
        )

        # Metric cells
        for ci, key in enumerate(col_keys, start=1):
            val = row[key]
            fc, tc = _cell(val, thr[key])
            ax.add_patch(mpatches.FancyBboxPatch(
                (ci + 0.05, y + 0.1), 0.9, 0.8,
                boxstyle="round,pad=0.02", facecolor=fc, edgecolor="white",
                linewidth=0.5, zorder=2,
            ))
            cell_text = f"{val:.3f}" if val is not None else "—"
            ax.text(
                ci + 0.5, y + 0.5, cell_text,
                ha="center", va="center", fontsize=7.5, color=tc,
                fontweight="bold", zorder=5,
            )

        # Extra columns (plain text)
        extra_start = 1 + n_data
        if extra_cols:
            for ei, (_, texts) in enumerate(extra_cols):
                ax.text(
                    extra_start + ei + 0.5, y + 0.5,
                    texts[ri],
                    ha="center", va="center", fontsize=6.5, color="#212121", zorder=5,
                )

        # Case column
        if show_case:
            case_text = classify_case(row, thr)
            case_col = extra_start + (n_extra)
            # colour the case cell
            if "H1" in case_text:
                case_bg = "#fff3e0"
            elif "H2" in case_text or "H3" in case_text:
                case_bg = "#fce4ec"
            elif "Case 1" in case_text or "No bottle" in case_text:
                case_bg = "#e8f5e9"
            else:
                case_bg = row_bg
            ax.add_patch(mpatches.FancyBboxPatch(
                (case_col + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02", facecolor=case_bg, edgecolor="white",
                linewidth=0.5, zorder=2,
            ))
            ax.text(
                case_col + 0.5, y + 0.5, case_text,
                ha="center", va="center", fontsize=6, color="#212121", zorder=5,
            )

    # Outer border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), n_cols_total, n_rows + 1,
        boxstyle="square,pad=0", fill=False, edgecolor="#37474f", linewidth=1.0, zorder=6,
    ))

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_HIGH_COLOR, label="High (≥ threshold)"),
        mpatches.Patch(facecolor=_LOW_COLOR,  label="Low  (< threshold)"),
        mpatches.Patch(facecolor=_NA_COLOR,   label="N/A"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3, fontsize=7, frameon=False,
    )


# ── Classification chart ───────────────────────────────────────────────────────

def plot_classification(
    rows: list[dict],
    output_dir: Path,
    thr: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(rows) + 1.5)))
    fig.patch.set_facecolor("white")

    _draw_table(
        ax, rows,
        col_keys  = ["lp_f1", "lang_cls_f1", "vis_cls_f1"],
        col_labels = ["LP F1", "Lang Cls F1", "Vision Cls F1"],
        thr       = thr,
        title     = "Hypothesis Matrix — Classification",
        show_case = True,
    )

    fig.text(
        0.5, 0.01,
        f"Thresholds (naive baseline) — LP: {thr['lp_f1']:.3f}  |  Lang Cls: {thr['lang_cls_f1']:.3f}  |  Vision Cls: {thr['vis_cls_f1']:.3f}",
        ha="center", fontsize=7, color="#616161",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "hypothesis_cls.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Description chart ──────────────────────────────────────────────────────────

def plot_description(
    rows: list[dict],
    output_dir: Path,
    thr: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(rows) + 1.5)))
    fig.patch.set_facecolor("white")

    _draw_table(
        ax, rows,
        col_keys   = ["lang_desc_rss", "vis_desc_rss"],
        col_labels = ["Lang ROUGE-SS", "Vision ROUGE-SS"],
        thr        = thr,
        title      = "Hypothesis Matrix — Description (ROUGE-SS)",
        show_case  = False,
    )

    fig.text(
        0.5, 0.01,
        f"Thresholds (naive baseline) — Lang ROUGE-SS: {thr['lang_desc_rss']:.3f}  |  Vision ROUGE-SS: {thr['vis_desc_rss']:.3f}",
        ha="center", fontsize=7, color="#616161",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "hypothesis_desc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hypothesis matrix charts from results_table.csv."
    )
    parser.add_argument(
        "--input_dir", default="outputs/",
        help="Directory containing results_table.csv",
    )
    parser.add_argument(
        "--output_dir", default="outputs/",
        help="Directory to write hypothesis_cls.png and hypothesis_desc.png",
    )
    parser.add_argument(
        "--lp_threshold", type=float, default=None,
        help="Override linear probe F1 threshold (default: naive baseline value)",
    )
    parser.add_argument(
        "--lang_cls_threshold", type=float, default=None,
        help="Override language classification F1 threshold (default: naive baseline value)",
    )
    parser.add_argument(
        "--lang_desc_threshold", type=float, default=None,
        help="Override language ROUGE-SS threshold (default: naive baseline value)",
    )
    parser.add_argument(
        "--vis_cls_threshold", type=float, default=None,
        help="Override vision classification F1 threshold (default: naive baseline value)",
    )
    parser.add_argument(
        "--vis_desc_threshold", type=float, default=None,
        help="Override vision ROUGE-SS threshold (default: naive baseline value)",
    )
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

    overrides = {
        "lp_f1":         args.lp_threshold,
        "lang_cls_f1":   args.lang_cls_threshold,
        "lang_desc_rss": args.lang_desc_threshold,
        "vis_cls_f1":    args.vis_cls_threshold,
        "vis_desc_rss":  args.vis_desc_threshold,
    }
    thr = naive_thresholds(rows, overrides)

    print(f"Loaded {len(rows)} row(s) from {csv_path}")
    print(f"Thresholds (from naive baseline):")
    print(f"  LP F1:          {thr['lp_f1']:.3f}")
    print(f"  Lang Cls F1:    {thr['lang_cls_f1']:.3f}")
    print(f"  Lang ROUGE-SS:  {thr['lang_desc_rss']:.3f}")
    print(f"  Vision Cls F1:  {thr['vis_cls_f1']:.3f}")
    print(f"  Vision ROUGE-SS:{thr['vis_desc_rss']:.3f}")
    print()

    print("=== Classification chart ===")
    plot_classification(rows, output_dir, thr)

    print("=== Description chart ===")
    plot_description(rows, output_dir, thr)

    print(f"\nDone. Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
