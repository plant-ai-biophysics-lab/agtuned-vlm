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

Rows are grouped by model type:
  Vision      — vision-encoder backbones (linear probe only)
  Language    — text-only LLMs
  Multimodal  — VLMs evaluated on language + vision (or vision only)
  Baseline    — naive shuffled baseline row

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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _r(val) -> float | None:
    return round(val, 3) if val is not None else None


def _macro(per_dataset: dict, metric: str) -> float | None:
    vals = [v[metric] for v in per_dataset.values() if metric in v]
    return round(sum(vals) / len(vals), 3) if vals else None


def _basename(model_id: str) -> str:
    return model_id.split("/")[-1]


# Known parameter counts keyed by basename.
# MoE models: "total/active", dense models: "NB" or "NM".
_PARAMS: dict[str, str] = {
    "dinov3-vitb16-pretrain-lvd1689m": "86M",
    "yolo11x-cls":                     "30M",
    "gpt-oss-20b":                     "21B/3.6B",
    "Llama-4-Scout-17B-16E-Instruct":  "109B/17B",
    "Meta-Llama-3-70B-Instruct":       "70B",
    "gemma-4-26B-A4B-it":              "26B/4B",
    "Qwen3.6-27B":                     "27B",
}


def _empty_row(model: str) -> dict:
    return {
        "model":         model,
        "params":        _PARAMS.get(_basename(model), "--"),
        "lp_f1":         None,
        "lang_cls_f1":   None,
        "lang_desc_rss": None,
        "vis_cls_f1":    None,
        "vis_desc_rss":  None,
    }


# ── Collation ──────────────────────────────────────────────────────────────────

def collate(lp_dir: Path, lang_dir: Path, vis_dir: Path) -> list[dict]:
    by_model: dict[str, dict] = {}

    # Linear probe
    lp_data = _load_json(lp_dir / "linear_probe_results.json")
    if lp_data:
        for backbone, m in lp_data.get("overall", {}).items():
            key = _basename(backbone)
            row = by_model.setdefault(key, _empty_row(backbone))
            row["lp_f1"] = _r(m.get("f1_macro"))

    # Language assessment
    lang_data = _load_json(lang_dir / "language_assessment_metrics.json")
    if lang_data:
        for model_id, m in lang_data.get("models", {}).items():
            key = _basename(model_id)
            row = by_model.setdefault(key, _empty_row(model_id))
            row["model"] = model_id
            judge_f1 = m.get("judge", {}).get("judge_f1_macro")
            row["lang_cls_f1"]   = _r(judge_f1 if judge_f1 is not None
                                       else m.get("classification_overall", {}).get("f1_macro"))
            row["lang_desc_rss"] = _r(m.get("description_overall", {}).get("rouge_ss"))

    # Vision assessment
    vis_data = _load_json(vis_dir / "vision_assessment_metrics.json")
    if vis_data:
        for model_id, m in vis_data.get("models", {}).items():
            key = _basename(model_id)
            row = by_model.setdefault(key, _empty_row(model_id))
            row["model"] = model_id
            judge_f1 = m.get("judge", {}).get("judge_f1_macro")
            row["vis_cls_f1"]   = _r(judge_f1 if judge_f1 is not None
                                      else m.get("classification_overall", {}).get("f1_macro"))
            row["vis_desc_rss"] = _r(m.get("description_overall", {}).get("rouge_ss"))

    # Assign model type
    def _model_type(row: dict) -> str:
        has_lp   = row["lp_f1"]        is not None
        has_lang = row["lang_cls_f1"]  is not None or row["lang_desc_rss"] is not None
        has_vis  = row["vis_cls_f1"]   is not None or row["vis_desc_rss"]  is not None
        if has_vis or (has_lp and has_lang):
            return "Multimodal"
        if has_lang:
            return "Language"
        return "Vision"

    _TYPE_ORDER = {"Vision": 0, "Language": 1, "Multimodal": 2, "Baseline": 3}

    for row in by_model.values():
        row["type"] = _model_type(row)

    rows = sorted(
        by_model.values(),
        key=lambda r: (_TYPE_ORDER.get(r["type"], 99), -(r["lp_f1"] or 0)),
    )

    # Naive baseline row — placed at top
    naive_row = _empty_row("Naive Baseline")
    naive_row["type"] = "Baseline"
    if lang_data:
        naive = lang_data.get("naive", {})
        naive_row["lang_cls_f1"]   = _r(naive.get("classification_overall", {}).get("f1_macro"))
        naive_row["lang_desc_rss"] = _r(naive.get("description_overall", {}).get("rouge_ss"))
    if vis_data:
        naive = vis_data.get("naive", {})
        naive_row["vis_cls_f1"]  = _macro(naive.get("classification", {}), "f1_macro")
        naive_row["vis_desc_rss"] = _macro(naive.get("description", {}), "rouge_ss")
    rows.insert(0, naive_row)

    return rows


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS  = ["type", "model", "params", "lp_f1", "lang_cls_f1", "lang_desc_rss", "vis_cls_f1", "vis_desc_rss"]
HEADERS = {
    "type":          "Type",
    "model":         "Model",
    "params":        "Params",
    "lp_f1":         "Linear Probe F1",
    "lang_cls_f1":   "Lang F1",
    "lang_desc_rss": "Lang ROUGE-SS",
    "vis_cls_f1":    "Vision F1",
    "vis_desc_rss":  "Vision ROUGE-SS",
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
    return "--" if val is None else f"{val:.3f}"


def _short(model_id: str) -> str:
    return model_id.split("/")[-1].replace("_", r"\_")


def write_latex(rows: list[dict], path: Path) -> None:
    from itertools import groupby

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llrccccc}",
        r"\toprule",
        (
            r" & "
            r" & "
            r" & \multicolumn{1}{c}{Linear Probe}"
            r" & \multicolumn{2}{c}{Language Assessment}"
            r" & \multicolumn{2}{c}{Vision Assessment} \\"
        ),
        r"\cmidrule(lr){4-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r" &  &  &  & Classification & Description & Classification & Description \\",
        r"\cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-7} \cmidrule(lr){8-8}",
        r"Type & Model & Params & F1 & F1 & ROUGE-SS & F1 & ROUGE-SS \\",
        r"\midrule",
    ]

    stage_groups = [(t, list(g)) for t, g in groupby(rows, key=lambda r: r["type"])]

    for gi, (type_label, grp) in enumerate(stage_groups):
        n = len(grp)
        for ri, row in enumerate(grp):
            type_cell = (
                rf"\multirow{{{n}}}{{*}}{{\textit{{{type_label}}}}}" if ri == 0 else ""
            )
            lines.append(
                f"{type_cell} & {_short(row['model'])}"
                f" & {row['params']}"
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
        r"Params shows total parameters; MoE models show total/active. "
        r"Linear Probe reports macro F1 on frozen embeddings. "
        r"Language and Vision Assessment report macro F1 (classification, LLM-judged where available) "
        r"and ROUGE-SS (description). -- indicates the model was not evaluated at that stage.}",
        r"\label{tab:results}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {path}")


# ── Cross-assessment correlation ──────────────────────────────────────────────

def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_class_scores(cls_rows: list[dict], desc_rows: list[dict]) -> dict[tuple, dict]:
    """Return {(dataset, class): {judge_f1, sem_sim, task_type}} from CSV rows."""
    cls_by_dc: dict[tuple, list[dict]] = defaultdict(list)
    for r in cls_rows:
        ds, cls = r.get("agml_dataset", ""), r.get("class_label", "")
        if ds and cls:
            cls_by_dc[(ds, cls)].append(r)

    desc_by_dc: dict[tuple, list[dict]] = defaultdict(list)
    for r in desc_rows:
        ds, cls = r.get("agml_dataset", ""), r.get("class_label", "")
        if ds and cls:
            desc_by_dc[(ds, cls)].append(r)

    result: dict[tuple, dict] = {}
    for key in set(cls_by_dc) | set(desc_by_dc):
        scores: dict = {}
        if key in cls_by_dc:
            rows = cls_by_dc[key]
            scores["task_type"] = rows[0].get("task_type", "unknown")
            judged = [r for r in rows if r.get("judged_correct", "") != ""]
            if judged:
                scores["judge_f1"] = float(np.mean([int(r["judged_correct"]) for r in judged]))
        if key in desc_by_dc:
            ss_vals = [float(r["sem_sim"]) for r in desc_by_dc[key]
                       if r.get("sem_sim") not in ("", None)]
            if ss_vals:
                scores["sem_sim"] = float(np.mean(ss_vals))
        if scores:
            result[key] = scores
    return result


def _per_dataset_scores(cls_rows: list[dict], desc_rows: list[dict]) -> dict[str, dict]:
    """Return {dataset: {judge_f1, sem_sim, task_type}} aggregated across all classes."""
    from sklearn.metrics import f1_score as _sk_f1

    cls_by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in cls_rows:
        ds = r.get("agml_dataset", "")
        if ds:
            cls_by_ds[ds].append(r)

    desc_by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in desc_rows:
        ds = r.get("agml_dataset", "")
        if ds:
            desc_by_ds[ds].append(r)

    result: dict[str, dict] = {}
    for ds in set(cls_by_ds) | set(desc_by_ds):
        scores: dict = {}
        if ds in cls_by_ds:
            rows = cls_by_ds[ds]
            scores["task_type"] = rows[0].get("task_type", "unknown")
            judged = [r for r in rows if r.get("judged_correct", "") != ""]
            if judged:
                refs  = [r.get("class_label", "") for r in judged]
                preds = [r.get("class_label", "") if int(r["judged_correct"])
                         else (r.get("resolved_class") or "__") for r in judged]
                lbls  = sorted(set(refs))
                scores["judge_f1"] = float(_sk_f1(refs, preds, average="macro",
                                                   labels=lbls, zero_division=0))
        if ds in desc_by_ds:
            ss = [float(r["sem_sim"]) for r in desc_by_ds[ds]
                  if r.get("sem_sim") not in ("", None)]
            if ss:
                scores["sem_sim"] = float(np.mean(ss))
        if scores:
            result[ds] = scores
    return result


def _scatter_2x2(lang_scores, vis_scores, key_fn, output_path, title, min_pts=5):
    """Shared 2×2 scatter grid for per-class or per-dataset correlation."""
    common = set(lang_scores) & set(vis_scores)
    if len(common) < min_pts:
        print(f"  [skip] {output_path.name}: only {len(common)} common entries")
        return

    task_types = sorted(set(lang_scores[k].get("task_type", "unknown") for k in common))
    _PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628", "#f781bf"]
    task_color = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(task_types)}

    PAIRS = [
        ("judge_f1", "judge_f1", "Lang Judge F1", "Vis Judge F1"),
        ("judge_f1", "sem_sim",  "Lang Judge F1", "Vis Sem Sim"),
        ("sem_sim",  "judge_f1", "Lang Sem Sim",  "Vis Judge F1"),
        ("sem_sim",  "sem_sim",  "Lang Sem Sim",  "Vis Sem Sim"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for ax, (lm, vm, lx, vy) in zip(axes, PAIRS):
        points = [
            (lang_scores[k][lm], vis_scores[k][vm], lang_scores[k].get("task_type", "unknown"))
            for k in common
            if lm in lang_scores[k] and vm in vis_scores[k]
        ]
        if len(points) < min_pts:
            ax.set_visible(False)
            continue

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        colors = [task_color[p[2]] for p in points]

        ax.scatter(xs, ys, c=colors, alpha=0.75, s=55,
                   edgecolors="black", linewidths=0.4)

        r = float(np.corrcoef(xs, ys)[0, 1])
        z = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color="crimson", linewidth=1.2,
                linestyle="--", label=f"r = {r:.3f}")

        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(vy, fontsize=9)
        ax.set_title(f"{lx} vs {vy}  (n={len(points)})", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    for tt in task_types:
        axes[0].scatter([], [], c=task_color[tt], label=tt.replace("_", " "), s=35)
    axes[0].legend(fontsize=7, loc="upper left")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cross_assessment_correlation(
    lang_cls_rows: list[dict],
    lang_desc_rows: list[dict],
    vis_cls_rows: list[dict],
    vis_desc_rows: list[dict],
    output_dir: Path,
    model_id: str,
) -> None:
    short = model_id.split("/")[-1]
    safe  = model_id.replace("/", "_").replace(" ", "_")

    lang_cls  = _per_class_scores(lang_cls_rows, lang_desc_rows)
    vis_cls   = _per_class_scores(vis_cls_rows,  vis_desc_rows)
    lang_ds   = _per_dataset_scores(lang_cls_rows, lang_desc_rows)
    vis_ds    = _per_dataset_scores(vis_cls_rows,  vis_desc_rows)

    _scatter_2x2(
        lang_cls, vis_cls,
        key_fn=lambda k: k,
        output_path=output_dir / f"cross_correlation_per_class_{safe}.png",
        title=f"Language vs Vision — Per-Class Correlation\n{short}",
    )
    _scatter_2x2(
        lang_ds, vis_ds,
        key_fn=lambda k: k,
        output_path=output_dir / f"cross_correlation_per_dataset_{safe}.png",
        title=f"Language vs Vision — Per-Dataset Correlation\n{short}",
    )


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

    n_models = sum(1 for r in rows if r["type"] != "Baseline")
    print(f"Collating {n_models} model(s) + naive baseline across "
          f"{len(set(r['type'] for r in rows))} type(s)...")

    write_csv(rows, output_dir / "results_table.csv")
    write_latex(rows, output_dir / "results_table.tex")

    # Cross-assessment per-class correlation plots
    lang_data = _load_json(Path(args.language_dir) / "language_assessment_metrics.json")
    vis_data  = _load_json(Path(args.vision_dir)   / "vision_assessment_metrics.json")
    if lang_data and vis_data:
        common_models = set(lang_data.get("models", {})) & set(vis_data.get("models", {}))
        for model_id in sorted(common_models):
            safe = model_id.replace("/", "_").replace(" ", "_")
            lang_cls_rows  = _read_csv_rows(Path(args.language_dir) / f"cls_{safe}.csv")
            lang_desc_rows = _read_csv_rows(Path(args.language_dir) / f"desc_{safe}.csv")
            vis_cls_rows   = _read_csv_rows(Path(args.vision_dir)   / f"cls_{safe}.csv")
            vis_desc_rows  = _read_csv_rows(Path(args.vision_dir)   / f"desc_{safe}.csv")
            if not (lang_cls_rows and vis_cls_rows):
                print(f"  [skip] {model_id}: missing cls CSVs for correlation plot")
                continue
            plot_cross_assessment_correlation(
                lang_cls_rows, lang_desc_rows,
                vis_cls_rows,  vis_desc_rows,
                output_dir,
                model_id,
            )

    print()
    print(f"  {'Type':<14} {'Model':<40} {'Params':>9} {'LP F1':>6} {'L F1':>6} {'L RSS':>6} {'V F1':>6} {'V RSS':>6}")
    print(f"  {'-'*100}")
    for row in rows:
        print(
            f"  {row['type']:<14} {row['model']:<40}"
            f" {row['params']:>9}"
            f" {_fmt(row['lp_f1']):>6}"
            f" {_fmt(row['lang_cls_f1']):>6}"
            f" {_fmt(row['lang_desc_rss']):>6}"
            f" {_fmt(row['vis_cls_f1']):>6}"
            f" {_fmt(row['vis_desc_rss']):>6}"
        )


if __name__ == "__main__":
    main()
