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


# ── Analysis tables (perception → reasoning collapse) ───────────────────────────

def _f1(val) -> str:
    return "  --  " if val is None else f"{val:6.3f}"


def _class_to_dataset(vis_cls_rows: list[dict]) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    for r in vis_cls_rows:
        cls, ds = r.get("class_label", ""), r.get("agml_dataset", "")
        if cls and ds:
            m[cls].add(ds)
    return m


def _lp_per_dataset(report: dict, cls_to_ds: dict[str, set[str]]) -> dict[str, float]:
    """Support-weighted mean of per-class linear-probe F1, aggregated to dataset.

    Approximate for the few classes shared across datasets (the class is
    counted toward each dataset it appears in).
    """
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)  # ds -> [(f1, support)]
    for cls, stats in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        f1  = stats.get("f1-score")
        sup = stats.get("support", 0.0)
        if f1 is None:
            continue
        for ds in cls_to_ds.get(cls, ()):
            acc[ds].append((float(f1), float(sup)))
    out: dict[str, float] = {}
    for ds, pairs in acc.items():
        tot = sum(s for _, s in pairs)
        out[ds] = sum(f * s for f, s in pairs) / tot if tot else float(np.mean([f for f, _ in pairs]))
    return out


def build_analysis(
    lp_data: dict | None,
    lang_data: dict | None,
    vis_data: dict | None,
    aug_data: dict | None,
    vis_cls_rows: list[dict],
    model_id: str,
    cot_data: dict | None = None,
) -> dict:
    """Assemble the three analysis tables into a structured dict."""
    short = _basename(model_id)
    A: dict = {"model": short, "chain": {}, "by_task": [], "by_dataset": []}

    # ── Table 1: evidence chain (overall) ──────────────────────────────────────
    def _lp_overall(node_key: str, bb_match) -> float | None:
        for bb, m in (lp_data or {}).get(node_key, {}).items():
            if bb_match(bb) and isinstance(m, dict):
                return m.get("f1_macro")
        return None

    A["chain"]["lp_pre"]   = _lp_overall("fc_combined",        lambda b: b == short)
    A["chain"]["lp_post"]  = _lp_overall("postproj_fc_combined", lambda b: b.startswith(short))
    A["chain"]["lp_dino"]  = _lp_overall("fc_combined",        lambda b: "dinov3" in b)
    A["chain"]["lp_yolo"]  = _lp_overall("fc_combined",        lambda b: "yolo" in b)

    vm = (vis_data or {}).get("models", {}).get(model_id, {})
    lm = (lang_data or {}).get("models", {}).get(model_id, {})
    A["chain"]["vis_zs"]   = vm.get("judge", {}).get("judge_f1_macro")
    A["chain"]["lang_zs"]  = lm.get("judge", {}).get("judge_f1_macro")
    am = (aug_data or {}).get("models", {}).get(model_id, {})
    A["chain"]["aug_raw"]   = am.get("augmented_overall", {}).get("f1_macro")
    A["chain"]["aug_judge"] = am.get("judge", {}).get("judge_f1_macro")

    # CoT conditions (from cot_vision.py output — self_cot and cot_yaml only)
    cm = (cot_data or {}).get("conditions", {})
    A["chain"]["cot_self"] = cm.get("self_cot", {}).get("f1_macro")
    A["chain"]["cot_yaml"] = cm.get("cot_yaml", {}).get("f1_macro")

    # ── Table 2: per-task dissociation (LP perception vs VLM output) ────────────
    lp_task     = (lp_data or {}).get("by_task_fc_combined", {})
    vis_jtask   = vm.get("judge_by_task", {})
    tasks = sorted(set(lp_task) | set(vis_jtask))
    for t in tasks:
        lp_f1  = lp_task.get(t, {}).get(short)
        vis_f1 = vis_jtask.get(t, {}).get("judge_f1_macro")
        gap = (lp_f1 - vis_f1) if (lp_f1 is not None and vis_f1 is not None) else None
        A["by_task"].append({"task": t, "lp": lp_f1, "vis": vis_f1, "gap": gap})

    # ── Table 3: per-dataset difficulty ────────────────────────────────────────
    cls_to_ds = _class_to_dataset(vis_cls_rows)
    lp_rep = (lp_data or {}).get("fc_combined", {}).get(short, {}).get("per_class_report_dict", {})
    lp_ds  = _lp_per_dataset(lp_rep, cls_to_ds) if lp_rep else {}

    vis_jds  = vm.get("judge_by_dataset", {})
    vis_dpd  = vm.get("description_per_dataset", {})
    aug_pd   = am.get("augmented_per_dataset", {})

    all_ds = sorted(set(vis_jds) | set(lp_ds))
    for ds in all_ds:
        lp_f1   = lp_ds.get(ds)
        vis_f1  = vis_jds.get(ds, {}).get("judge_f1_macro")
        sem     = vis_dpd.get(ds, {}).get("sem_sim")
        aug_f1  = aug_pd.get(ds, {}).get("f1_macro")
        gap = (lp_f1 - vis_f1) if (lp_f1 is not None and vis_f1 is not None) else None
        A["by_dataset"].append({
            "dataset": ds, "lp": lp_f1, "vis": vis_f1,
            "sem_sim": sem, "aug": aug_f1, "gap": gap,
        })
    # sort hardest (lowest vis judged F1) first
    A["by_dataset"].sort(key=lambda r: (r["vis"] if r["vis"] is not None else 1.0))
    return A


def plot_four_condition_comparison(
    vis_judge_ds: dict,
    cot_conditions: dict,
    aug_per_ds: dict,
    output_path: Path,
) -> None:
    """Bar chart: 4-condition judged F1 per dataset.

    vis_judge_ds  — {ds: {"judge_f1_macro": float}}  from vision assessment
    cot_conditions — {"self_cot": {"per_dataset": {ds: {"judge_f1_macro": float}}}, ...}
    aug_per_ds    — {ds: {"f1_macro": float}}  from desc_augmented (non-judged)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_ds = sorted(set(vis_judge_ds) | set(aug_per_ds) |
                    set(cot_conditions.get("self_cot", {}).get("per_dataset", {})) |
                    set(cot_conditions.get("cot_yaml",  {}).get("per_dataset", {})))
    if not all_ds:
        return

    conditions = [
        ("baseline",  "Vision Zero-shot",       "#1f77b4"),
        ("self_cot",  "Self-CoT",               "#ff7f0e"),
        ("cot_yaml",  "Structured CoT",         "#2ca02c"),
        ("desc_aug",  "Desc-Aug (non-judged)",  "#9467bd"),
    ]

    def _get(ds, cond_key):
        if cond_key == "baseline":
            return (vis_judge_ds.get(ds) or {}).get("judge_f1_macro")
        if cond_key == "desc_aug":
            return (aug_per_ds.get(ds) or {}).get("f1_macro")
        return (cot_conditions.get(cond_key, {})
                .get("per_dataset", {}).get(ds, {}).get("judge_f1_macro"))

    x = np.arange(len(all_ds))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(12, len(all_ds) * 1.8), 5))

    for i, (key, label, color) in enumerate(conditions):
        f1s = [_get(ds, key) or 0.0 for ds in all_ds]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, f1s):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    clean_ds = [d.replace("_classification", "").replace("_", " ") for d in all_ds]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_ds, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1 Macro (judged where available)")
    ax.set_ylim(0, 1.15)
    ax.set_title("4-Condition Comparison — Same Images, Different Prompts\n"
                 "(Baseline / CoT = judged F1; Desc-Aug = direct F1)")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# Dataset → task_type mapping (matches cot_vision.py DATASET_TASK_TYPES)
_DS_TASK: dict[str, str] = {
    "arabica_coffee_leaf_disease_classification": "disease",
    "banana_leaf_disease_classification":         "disease",
    "bean_disease_uganda":                        "disease",
    "betel_leaf_disease_classification":          "disease",
    "blackgram_plant_leaf_disease_classification":"disease",
    "chilli_leaf_classification":                 "disease",
    "coconut_tree_disease_classification":        "disease",
    "corn_maize_leaf_disease":                    "disease",
    "crop_weeds_greece":                          "weed_identification",
    "cucumber_disease_classification":            "disease",
    "guava_disease_pakistan":                     "disease",
    "java_plum_leaf_disease_classification":      "disease",
    "onion_leaf_classification":                  "disease",
    "orange_leaf_disease_classification":         "disease",
    "paddy_disease_classification":               "disease",
    "papaya_leaf_disease_classification":         "disease",
    "plant_seedlings_aarhus":                     "weed_identification",
    "rangeland_weeds_australia":                  "weed_identification",
    "rice_leaf_disease_classification":           "disease",
    "soybean_insect_classification":              "pest_damage",
    "soybean_weed_uav_brazil":                    "weed_identification",
    "sugarcane_damage_usa":                       "pest_damage",
    "sunflower_disease_classification":           "disease",
    "tea_leaf_disease_classification":            "disease",
    "tomato_leaf_disease":                        "disease",
    "vine_virus_photo_dataset":                   "disease",
}

_FOUR_CONDITIONS = [
    ("baseline", "Vision Zero-shot",      "#1f77b4"),
    ("self_cot", "Self-CoT",              "#ff7f0e"),
    ("cot_yaml", "Structured CoT",        "#2ca02c"),
    ("desc_aug", "Desc-Aug (non-judged)", "#9467bd"),
]


def _four_cond_get(ds: str, cond_key: str, vis_judge_ds: dict,
                   cot_conditions: dict, aug_per_ds: dict) -> float | None:
    if cond_key == "baseline":
        return (vis_judge_ds.get(ds) or {}).get("judge_f1_macro")
    if cond_key == "desc_aug":
        return (aug_per_ds.get(ds) or {}).get("f1_macro")
    return (cot_conditions.get(cond_key, {})
            .get("per_dataset", {}).get(ds, {}).get("judge_f1_macro"))


def _four_cond_bars(ax, labels: list[str], data: dict[str, list[float]]) -> None:
    x = np.arange(len(labels))
    width = 0.8 / len(_FOUR_CONDITIONS)
    for i, (key, label, color) in enumerate(_FOUR_CONDITIONS):
        vals = data[key]
        offset = (i - len(_FOUR_CONDITIONS) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Macro (judged where available)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)


def plot_four_condition_overall(
    vis_judge_ds: dict, cot_conditions: dict, aug_per_ds: dict, output_path: Path,
) -> None:
    """Single grouped bar: mean F1 per condition across all datasets."""
    all_ds = sorted(set(vis_judge_ds) | set(aug_per_ds) |
                    set(cot_conditions.get("self_cot", {}).get("per_dataset", {})) |
                    set(cot_conditions.get("cot_yaml",  {}).get("per_dataset", {})))
    if not all_ds:
        return

    data: dict[str, list[float]] = {}
    for key, _, _ in _FOUR_CONDITIONS:
        vals = [v for ds in all_ds
                if (v := _four_cond_get(ds, key, vis_judge_ds, cot_conditions, aug_per_ds)) is not None]
        data[key] = [float(np.mean(vals)) if vals else 0.0]

    fig, ax = plt.subplots(figsize=(6, 5))
    _four_cond_bars(ax, ["Overall"], data)
    ax.set_title("4-Condition Comparison — Overall Mean F1\n"
                 "(Baseline / CoT = judged; Desc-Aug = direct)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_four_condition_per_task(
    vis_judge_ds: dict, cot_conditions: dict, aug_per_ds: dict,
    vis_judge_task: dict, output_path: Path,
) -> None:
    """Grouped bar per task type (disease / pest_damage / weed_identification)."""
    from collections import defaultdict

    # Group datasets by task
    by_task: dict[str, list[str]] = defaultdict(list)
    all_ds = sorted(set(vis_judge_ds) | set(aug_per_ds) |
                    set(cot_conditions.get("self_cot", {}).get("per_dataset", {})) |
                    set(cot_conditions.get("cot_yaml",  {}).get("per_dataset", {})))
    for ds in all_ds:
        task = _DS_TASK.get(ds, "disease")
        by_task[task].append(ds)

    tasks = sorted(by_task)
    if not tasks:
        return

    data: dict[str, list[float]] = {k: [] for k, _, _ in _FOUR_CONDITIONS}
    for task in tasks:
        ds_list = by_task[task]
        for key, _, _ in _FOUR_CONDITIONS:
            if key == "baseline" and task in vis_judge_task:
                # Vision assessment has per-task judged F1 directly
                val = vis_judge_task[task].get("judge_f1_macro")
            else:
                vals = [v for ds in ds_list
                        if (v := _four_cond_get(ds, key, vis_judge_ds,
                                                cot_conditions, aug_per_ds)) is not None]
                val = float(np.mean(vals)) if vals else 0.0
            data[key].append(val if val is not None else 0.0)

    task_labels = [t.replace("_", " ").title() for t in tasks]
    fig, ax = plt.subplots(figsize=(max(7, len(tasks) * 2.5), 5))
    _four_cond_bars(ax, task_labels, data)
    ax.set_title("4-Condition Comparison — Per Task Type\n"
                 "(Baseline / CoT = judged; Desc-Aug = direct)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_difficulty_bins_four_condition(
    vis_judge_ds: dict, cot_conditions: dict, aug_per_ds: dict,
    output_path: Path,
) -> None:
    """Bar chart: mean judged F1 per difficulty tercile across all 4 conditions.

    Difficulty is defined by vision baseline judged F1 (lower = harder).
    """
    from collections import defaultdict

    all_ds = sorted(set(vis_judge_ds) | set(aug_per_ds) |
                    set(cot_conditions.get("self_cot", {}).get("per_dataset", {})) |
                    set(cot_conditions.get("cot_yaml",  {}).get("per_dataset", {})))
    # Only include datasets with a vis baseline score (needed to define difficulty)
    ranked = sorted([ds for ds in all_ds if ds in vis_judge_ds],
                    key=lambda d: vis_judge_ds[d].get("judge_f1_macro") or
                                  vis_judge_ds[d].get("f1_macro", 0.0))
    if not ranked:
        return

    n = len(ranked)
    bins: dict[str, list[str]] = {
        "Hard":   ranked[:n//3],
        "Medium": ranked[n//3:2*n//3],
        "Easy":   ranked[2*n//3:],
    }

    data: dict[str, list[float]] = {k: [] for k, _, _ in _FOUR_CONDITIONS}
    for bin_ds in bins.values():
        for key, _, _ in _FOUR_CONDITIONS:
            vals = [v for ds in bin_ds
                    if (v := _four_cond_get(ds, key, vis_judge_ds,
                                            cot_conditions, aug_per_ds)) is not None]
            data[key].append(float(np.mean(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    _four_cond_bars(ax, list(bins.keys()), data)
    ax.set_title("4-Condition Comparison — By Difficulty Bin\n"
                 "(Baseline / CoT = judged; Desc-Aug = direct)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def render_analysis(A: dict) -> str:
    short = A["model"]
    L: list[str] = []
    c = A["chain"]

    L.append(f"# Perception → Reasoning Analysis — {short}\n")
    L.append("## Table 1 — Evidence Chain (overall macro F1)\n")
    L.append("| Stage | What it measures | F1 |")
    L.append("|---|---|---|")
    L.append(f"| Linear probe, pre-projector | Features decodable in vision tower | {_f1(c['lp_pre']).strip()} |")
    L.append(f"| Linear probe, post-projector | Survive vision→text projection | {_f1(c['lp_post']).strip()} |")
    L.append(f"| DINOv3 reference | Ceiling for good perception | {_f1(c['lp_dino']).strip()} |")
    L.append(f"| YOLO reference | Weak-backbone floor | {_f1(c['lp_yolo']).strip()} |")
    L.append(f"| **Vision zero-shot** (judged) | Full VLM outputs the class | **{_f1(c['vis_zs']).strip()}** |")
    L.append(f"| Language text-only (judged) | Does it know the words | {_f1(c['lang_zs']).strip()} |")
    L.append(f"| Vision + description (judged) | Hand it the bridge | **{_f1(c['aug_judge']).strip()}** |")
    if c["lp_post"] is not None and c["vis_zs"] is not None:
        L.append(f"\n**Collapse:** post-projector {c['lp_post']:.3f} → zero-shot {c['vis_zs']:.3f} "
                 f"(−{c['lp_post']-c['vis_zs']:.3f}). "
                 + (f"Recovers to {c['aug_judge']:.3f} with description.\n" if c["aug_judge"] else "\n"))

    # CoT comparison (only if any CoT data available)
    has_cot = any(c.get(k) is not None for k in ("cot_self", "cot_yaml"))
    if has_cot:
        vis_zs_f1 = c.get("vis_zs")
        L.append("\n## Table 1b — CoT Prompting Comparison\n")
        L.append("| Condition | F1 Macro | Δ vs Zero-shot |")
        L.append("|---|---|---|")

        def _delta(val):
            if val is None or vis_zs_f1 is None:
                return "--"
            return f"{val - vis_zs_f1:+.3f}"

        L.append(f"| Vision zero-shot (baseline) | {_f1(vis_zs_f1).strip()} | -- |")
        L.append(f"| Self-description CoT | {_f1(c.get('cot_self')).strip()} | {_delta(c.get('cot_self'))} |")
        L.append(f"| Structured CoT (cot_prompts.yaml) | {_f1(c.get('cot_yaml')).strip()} | {_delta(c.get('cot_yaml'))} |")

    L.append("## Table 2 — Per-Task Dissociation (perception vs output)\n")
    L.append("| Task | Linear probe F1 | VLM zero-shot judged F1 | Gap |")
    L.append("|---|---|---|---|")
    for r in A["by_task"]:
        gap = f"{r['gap']:+.3f}" if r["gap"] is not None else "--"
        L.append(f"| {r['task'].replace('_',' ')} | {_f1(r['lp']).strip()} | {_f1(r['vis']).strip()} | {gap} |")

    L.append("\n## Table 3 — Per-Dataset Difficulty (hardest → easiest by VLM judged F1)\n")
    L.append("| Dataset | LP F1 | Vis judged F1 | Desc SemSim | Aug F1 | LP−Vis gap |")
    L.append("|---|---|---|---|---|---|")
    for r in A["by_dataset"]:
        gap = f"{r['gap']:+.3f}" if r["gap"] is not None else "--"
        name = r["dataset"].replace("_classification", "").replace("_", " ")
        L.append(f"| {name} | {_f1(r['lp']).strip()} | {_f1(r['vis']).strip()} | "
                 f"{_f1(r['sem_sim']).strip()} | {_f1(r['aug']).strip()} | {gap} |")

    # easiest/hardest callouts
    ranked = [r for r in A["by_dataset"] if r["vis"] is not None]
    if ranked:
        hardest = ranked[:3]
        easiest = ranked[-3:][::-1]
        L.append("\n**Hardest (lowest VLM judged F1):** "
                 + ", ".join(f"{r['dataset'].replace('_classification','')} ({r['vis']:.3f})" for r in hardest))
        L.append("\n**Easiest (highest VLM judged F1):** "
                 + ", ".join(f"{r['dataset'].replace('_classification','')} ({r['vis']:.3f})" for r in easiest))
        # biggest perception-output dissociation
        gapped = [r for r in ranked if r["gap"] is not None]
        if gapped:
            gapped.sort(key=lambda r: -r["gap"])
            L.append("\n**Biggest perception→output gap (features clean, model fails):** "
                     + ", ".join(f"{r['dataset'].replace('_classification','')} "
                                 f"(LP {r['lp']:.2f} → Vis {r['vis']:.2f})" for r in gapped[:3]))
    return "\n".join(L) + "\n"


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
        "--aug_dir", default="outputs/desc_augmented_vision/",
        help="Directory containing desc_augmented_metrics_all.json",
    )
    parser.add_argument(
        "--cot_dir", default="outputs/cot_vision/",
        help="Directory containing cot_vision_metrics.json",
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

    # Perception → reasoning analysis tables (markdown)
    lp_data  = _load_json(Path(args.linear_probe_dir) / "linear_probe_results.json")
    aug_data = (_load_json(Path(args.aug_dir) / "desc_augmented_metrics_all.json")
                or _load_json(Path(args.aug_dir) / "desc_augmented_metrics_filtered.json"))
    cot_data = _load_json(Path(args.cot_dir) / "cot_vision_metrics.json")
    if vis_data:
        analysis_sections: list[str] = []
        for model_id in sorted(vis_data.get("models", {})):
            safe = model_id.replace("/", "_").replace(" ", "_")
            vis_cls_rows = _read_csv_rows(Path(args.vision_dir) / f"cls_{safe}.csv")
            A = build_analysis(lp_data, lang_data, vis_data, aug_data, vis_cls_rows, model_id,
                               cot_data=cot_data)
            section = render_analysis(A)
            analysis_sections.append(section)
            print("\n" + section)
        analysis_path = output_dir / "analysis_summary.md"
        analysis_path.write_text("\n\n".join(analysis_sections), encoding="utf-8")
        print(f"  Saved: {analysis_path}")

        # 4-condition comparison charts (per-dataset, per-task, overall)
        for model_id in sorted(vis_data.get("models", {})):
            vm = (vis_data or {}).get("models", {}).get(model_id, {})
            am = (aug_data or {}).get("models", {}).get(model_id, {})
            vis_jds      = vm.get("judge_by_dataset", {})
            vis_jtask    = vm.get("judge_by_task",    {})
            aug_pd       = am.get("augmented_per_dataset", {})
            cot_conds    = (cot_data or {}).get("conditions", {})
            plot_four_condition_comparison(
                vis_jds, cot_conds, aug_pd,
                output_dir / "four_condition_per_dataset.png",
            )
            plot_four_condition_per_task(
                vis_jds, cot_conds, aug_pd, vis_jtask,
                output_dir / "four_condition_per_task.png",
            )
            plot_four_condition_overall(
                vis_jds, cot_conds, aug_pd,
                output_dir / "four_condition_overall.png",
            )
            plot_difficulty_bins_four_condition(
                vis_jds, cot_conds, aug_pd,
                output_dir / "four_condition_difficulty_bins.png",
            )
            break  # one model only

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
