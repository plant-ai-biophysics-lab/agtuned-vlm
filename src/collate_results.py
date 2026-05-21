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

Rows are grouped by stage (linear probe → language → vision), one row per model.

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


def load_linear_probe(results_dir: Path) -> list[dict]:
    """Returns [{stage, model, lp_f1}] sorted by f1 desc."""
    data = _load_json(results_dir / "linear_probe_results.json")
    if not data:
        return []
    rows = []
    for backbone, m in data.get("overall", {}).items():
        rows.append({
            "stage": "Linear Probe",
            "model": backbone,
            "lp_f1":     round(m.get("f1_macro", float("nan")), 4),
            "lang_cls_f1":  None,
            "lang_desc_rss": None,
            "vis_cls_f1":   None,
            "vis_desc_rss": None,
        })
    return sorted(rows, key=lambda r: r["lp_f1"], reverse=True)


def load_language(results_dir: Path) -> list[dict]:
    data = _load_json(results_dir / "language_assessment_metrics.json")
    if not data:
        return []
    rows = []
    for model_id, m in data.get("models", {}).items():
        cls_f1  = m.get("classification_overall", {}).get("f1_macro")
        desc_ss = m.get("description_overall", {}).get("rouge_ss")
        rows.append({
            "stage": "Language Assessment",
            "model": model_id,
            "lp_f1":      None,
            "lang_cls_f1":   round(cls_f1,  4) if cls_f1  is not None else None,
            "lang_desc_rss": round(desc_ss, 4) if desc_ss is not None else None,
            "vis_cls_f1":    None,
            "vis_desc_rss":  None,
        })
    return rows


def load_vision(results_dir: Path) -> list[dict]:
    data = _load_json(results_dir / "vision_assessment_metrics.json")
    if not data:
        return []
    rows = []
    for model_id, m in data.get("models", {}).items():
        cls_f1  = m.get("classification_overall", {}).get("f1_macro")
        desc_ss = m.get("description_overall", {}).get("rouge_ss")
        rows.append({
            "stage": "Vision Assessment",
            "model": model_id,
            "lp_f1":      None,
            "lang_cls_f1":   None,
            "lang_desc_rss": None,
            "vis_cls_f1":    round(cls_f1,  4) if cls_f1  is not None else None,
            "vis_desc_rss":  round(desc_ss, 4) if desc_ss is not None else None,
        })
    return rows


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS = ["stage", "model", "lp_f1", "lang_cls_f1", "lang_desc_rss", "vis_cls_f1", "vis_desc_rss"]
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
    if val is None:
        return "--"
    return f"{val:.4f}"


def _short(model_id: str) -> str:
    """Strip org prefix for display: org/model-name → model-name."""
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

    groups = groupby(rows, key=lambda r: r["stage"])
    stage_list = [(stage, list(grp)) for stage, grp in groups]

    for gi, (stage, grp) in enumerate(stage_list):
        n = len(grp)
        for ri, row in enumerate(grp):
            stage_cell = (
                rf"\multirow{{{n}}}{{*}}{{\textit{{{stage}}}}}" if ri == 0 else ""
            )
            lines.append(
                f"{stage_cell} & {_short(row['model'])}"
                f" & {_fmt(row['lp_f1'])}"
                f" & {_fmt(row['lang_cls_f1'])} & {_fmt(row['lang_desc_rss'])}"
                f" & {_fmt(row['vis_cls_f1'])} & {_fmt(row['vis_desc_rss'])} \\\\"
            )
        if gi < len(stage_list) - 1:
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

    rows = (
        load_linear_probe(Path(args.linear_probe_dir))
        + load_language(Path(args.language_dir))
        + load_vision(Path(args.vision_dir))
    )

    if not rows:
        print("ERROR: no data found in any of the provided directories.")
        sys.exit(1)

    print(f"Collating {len(rows)} rows across "
          f"{len(set(r['stage'] for r in rows))} stage(s)...")

    write_csv(rows, output_dir / "results_table.csv")
    write_latex(rows, output_dir / "results_table.tex")

    # Print a quick summary to stdout
    print()
    print(f"  {'Stage':<22} {'Model':<45} {'LP':>6} {'LCls':>6} {'LDesc':>6} {'VCls':>6} {'VDesc':>6}")
    print(f"  {'-'*95}")
    prev_stage = None
    for row in rows:
        if row["stage"] != prev_stage:
            prev_stage = row["stage"]
        print(
            f"  {row['stage']:<22} {row['model']:<45}"
            f" {_fmt(row['lp_f1']):>6}"
            f" {_fmt(row['lang_cls_f1']):>6}"
            f" {_fmt(row['lang_desc_rss']):>6}"
            f" {_fmt(row['vis_cls_f1']):>6}"
            f" {_fmt(row['vis_desc_rss']):>6}"
        )


if __name__ == "__main__":
    main()
