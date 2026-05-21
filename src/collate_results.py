"""
Collate metrics from linear probe, language assessment, and vision assessment
into a single comparison table.

Outputs:
  <output_dir>/results_table.csv    — flat CSV
  <output_dir>/results_table.tex    — LaTeX booktabs table

Columns:
  Linear Probe      : F1 Macro
  Language Assess.  : Cls F1 Macro | Desc ROUGE-SS
  Vision Assess.    : Cls F1 Macro | Desc ROUGE-SS

Rows are grouped by which stages a model participated in. Models that appear
in multiple stages are merged into a single row.

Stage groups (in order):
  Linear Probe only
  Language only
  Vision only
  Language + Vision
  Linear Probe + Language
  Linear Probe + Vision
  Linear Probe + Language + Vision

Usage:
    python main.py collate \
        --linear_probe_dir   outputs/linear_probe/ \
        --language_dir       outputs/language_assessment/ \
        --vision_dir         outputs/vision_assessment/ \
        --output_dir         outputs/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _basename(model_id: str) -> str:
    """Normalize model ID to just the name part for cross-source matching."""
    return model_id.split("/")[-1]


def _empty_row(model: str) -> dict:
    return {
        "model":         model,
        "lp_f1":         None,
        "lang_cls_f1":   None,
        "lang_desc_rss": None,
        "vis_cls_f1":    None,
        "vis_desc_rss":  None,
    }


def collate(lp_dir: Path, lang_dir: Path, vis_dir: Path) -> list[dict]:
    """Merge all three sources into one dict per model, then assign stage groups.

    Matching is done on the basename (part after '/') so that short names used
    by linear probe (e.g. 'Qwen3.6-27B') merge with full IDs used by language/
    vision assessment (e.g. 'Qwen/Qwen3.6-27B'). The full ID is preferred for
    display when available.
    """
    # key: basename → row dict
    by_model: dict[str, dict] = {}

    # Linear probe — keyed by basename
    lp_data = _load_json(lp_dir / "linear_probe_results.json")
    if lp_data:
        for backbone, m in lp_data.get("overall", {}).items():
            key = _basename(backbone)
            row = by_model.setdefault(key, _empty_row(backbone))
            row["lp_f1"] = round(m.get("f1_macro", float("nan")), 4)

    # Language assessment — prefer full model ID for display
    lang_data = _load_json(lang_dir / "language_assessment_metrics.json")
    if lang_data:
        for model_id, m in lang_data.get("models", {}).items():
            key = _basename(model_id)
            row = by_model.setdefault(key, _empty_row(model_id))
            row["model"] = model_id          # upgrade to full ID
            cls_f1  = m.get("classification_overall", {}).get("f1_macro")
            desc_ss = m.get("description_overall", {}).get("rouge_ss")
            row["lang_cls_f1"]   = round(cls_f1,  4) if cls_f1  is not None else None
            row["lang_desc_rss"] = round(desc_ss, 4) if desc_ss is not None else None

    # Vision assessment — prefer full model ID for display
    vis_data = _load_json(vis_dir / "vision_assessment_metrics.json")
    if vis_data:
        for model_id, m in vis_data.get("models", {}).items():
            key = _basename(model_id)
            row = by_model.setdefault(key, _empty_row(model_id))
            row["model"] = model_id          # upgrade to full ID
            cls_f1  = m.get("classification_overall", {}).get("f1_macro")
            desc_ss = m.get("description_overall", {}).get("rouge_ss")
            row["vis_cls_f1"]    = round(cls_f1,  4) if cls_f1  is not None else None
            row["vis_desc_rss"]  = round(desc_ss, 4) if desc_ss is not None else None

    # Assign stage group and sort
    def _stage(row: dict) -> str:
        has_lp   = row["lp_f1"]         is not None
        has_lang = row["lang_cls_f1"]    is not None or row["lang_desc_rss"] is not None
        has_vis  = row["vis_cls_f1"]     is not None or row["vis_desc_rss"]  is not None
        if has_lp and has_lang and has_vis: return "Linear Probe + Language + Vision"
        if has_lp and has_lang:             return "Linear Probe + Language"
        if has_lp and has_vis:             return "Linear Probe + Vision"
        if has_lang and has_vis:           return "Language + Vision"
        if has_lp:                         return "Linear Probe"
        if has_lang:                       return "Language Assessment"
        if has_vis:                        return "Vision Assessment"
        return "Unknown"

    _STAGE_ORDER = {
        "Linear Probe": 0,
        "Language Assessment": 1,
        "Vision Assessment": 2,
        "Language + Vision": 3,
        "Linear Probe + Language": 4,
        "Linear Probe + Vision": 5,
        "Linear Probe + Language + Vision": 6,
    }

    for row in by_model.values():
        row["stage"] = _stage(row)

    return sorted(
        by_model.values(),
        key=lambda r: (_STAGE_ORDER.get(r["stage"], 99), -(r["lp_f1"] or 0)),
    )


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS  = ["stage", "model", "lp_f1", "lang_cls_f1", "lang_desc_rss", "vis_cls_f1", "vis_desc_rss"]
HEADERS = {
    "stage":         "Stage",
    "model":         "Model",
    "lp_f1":         "Linear Probe F1",
    "lang_cls_f1":   "Lang Cls F1",
    "lang_desc_rss": "Lang Desc ROUGE-SS",
    "vis_cls_f1":    "Vision Cls F1",
    "vis_desc_rss":  "Vision Desc ROUGE-SS",
}


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(HEADERS)
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in FIELDS})
    print(f"  Saved: {path}")


# ── LaTeX ──────────────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    return "--" if val is None else f"{val:.4f}"


def _short(model_id: str) -> str:
    return model_id.split("/")[-1].replace("_", r"\_")


def write_latex(rows: list[dict], path: Path) -> None:
    from itertools import groupby

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        (
            r" & "
            r" & \multicolumn{1}{c}{Linear Probe}"
            r" & \multicolumn{2}{c}{Language Assessment}"
            r" & \multicolumn{2}{c}{Vision Assessment} \\"
        ),
        r"\cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        (
            r"Stage & Model"
            r" & F1"
            r" & Cls F1 & Desc ROUGE-SS"
            r" & Cls F1 & Desc ROUGE-SS \\"
        ),
        r"\midrule",
    ]

    stage_groups = [(s, list(g)) for s, g in groupby(rows, key=lambda r: r["stage"])]

    for gi, (stage, grp) in enumerate(stage_groups):
        n = len(grp)
        escaped_stage = stage.replace("+", r"\texttt{+}")
        for ri, row in enumerate(grp):
            stage_cell = (
                rf"\multirow{{{n}}}{{*}}{{\textit{{{escaped_stage}}}}}" if ri == 0 else ""
            )
            lines.append(
                f"{stage_cell} & {_short(row['model'])}"
                f" & {_fmt(row['lp_f1'])}"
                f" & {_fmt(row['lang_cls_f1'])} & {_fmt(row['lang_desc_rss'])}"
                f" & {_fmt(row['vis_cls_f1'])} & {_fmt(row['vis_desc_rss'])} \\\\"
            )
        if gi < len(stage_groups) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Comparison of model performance across pipeline stages. "
        r"Linear Probe reports macro F1 on frozen embeddings. "
        r"Language and Vision Assessment report macro F1 (classification) "
        r"and ROUGE-SS (description). -- indicates the model was not evaluated at that stage.}",
        r"\label{tab:results}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate linear probe, language, and vision metrics into a results table."
    )
    parser.add_argument(
        "--linear_probe_dir", default="outputs/linear_probe/",
        help="Directory containing linear_probe_results.json",
    )
    parser.add_argument(
        "--language_dir", default="outputs/language_assessment/",
        help="Directory containing language_assessment_metrics.json",
    )
    parser.add_argument(
        "--vision_dir", default="outputs/vision_assessment/",
        help="Directory containing vision_assessment_metrics.json",
    )
    parser.add_argument(
        "--output_dir", default="outputs/",
        help="Directory to write results_table.csv and results_table.tex",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collate(
        Path(args.linear_probe_dir),
        Path(args.language_dir),
        Path(args.vision_dir),
    )

    if not rows:
        print("ERROR: no data found in any of the provided directories.")
        sys.exit(1)

    print(f"Collating {len(rows)} model(s) across "
          f"{len(set(r['stage'] for r in rows))} stage group(s)...")

    write_csv(rows, output_dir / "results_table.csv")
    write_latex(rows, output_dir / "results_table.tex")

    print()
    print(f"  {'Stage':<35} {'Model':<40} {'LP':>6} {'LCls':>6} {'LDesc':>6} {'VCls':>6} {'VDesc':>6}")
    print(f"  {'-'*105}")
    for row in rows:
        print(
            f"  {row['stage']:<35} {row['model']:<40}"
            f" {_fmt(row['lp_f1']):>6}"
            f" {_fmt(row['lang_cls_f1']):>6}"
            f" {_fmt(row['lang_desc_rss']):>6}"
            f" {_fmt(row['vis_cls_f1']):>6}"
            f" {_fmt(row['vis_desc_rss']):>6}"
        )


if __name__ == "__main__":
    main()
