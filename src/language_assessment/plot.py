"""
Regenerate language assessment charts from previously saved metrics JSON + CSVs.

Reads  <output_dir>/language_assessment_metrics.json  and per-model CSVs
(cls_*.csv, desc_*.csv) to regenerate all charts without re-running inference.

Usage:
    python main.py language-plot \
        --output_dir        outputs/language_assessment/ \
        --multimodal_models google/gemma-4-26B-A4B-it \
        --config            src/config/datasets.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.language_assessment.assess import (
    _task_label,
    plot_cls_summary_per_dataset,
    plot_cls_summary_per_task,
    plot_desc_sem_sim_per_dataset,
    plot_desc_summary_per_task,
    plot_difficulty_correlation,
    plot_metric_correlation,
    plot_overall_cls_summary,
    plot_overall_desc_summary,
)
from src.common.distribution_plots import (
    gather_cls_per_class_by_dataset,
    gather_cls_values_by_dataset,
    gather_cls_values_by_task_from_per_class,
    gather_desc_per_sample_by_dataset,
    gather_desc_values_by_dataset,
    gather_desc_values_by_task_from_per_sample,
    gather_judge_per_class_by_dataset,
    gather_judge_values_by_dataset,
    model_color_map,
    plot_overall_distribution,
    plot_per_dataset_distribution,
    plot_per_task_distribution,
)
from src.common.io import read_csv


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def _macro_average(per_dataset: dict, metrics: list[str]) -> dict | None:
    if not per_dataset:
        return None
    return {
        m: float(np.mean([v[m] for v in per_dataset.values() if m in v]))
        for m in metrics
        if any(m in v for v in per_dataset.values())
    }


def _parse_summary_naive(summary_path: Path) -> tuple[dict | None, dict | None]:
    if not summary_path.exists():
        return None, None
    import re
    text = summary_path.read_text()
    desc_naive = cls_naive = None
    m = re.search(
        r"Description ROUGE.*?naive\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        desc_naive = {
            "rouge1": float(m.group(1)), "rouge2": float(m.group(2)),
            "rougeL": float(m.group(3)), "rouge_ss": float(m.group(4)),
        }
    m = re.search(
        r"Classification.*?naive\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+/\d+",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        cls_naive = {
            "accuracy": float(m.group(1)),
            "contains_accuracy": float(m.group(2)),
            "f1_macro": float(m.group(3)),
        }
    return desc_naive, cls_naive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate language assessment charts from saved metrics JSON + CSVs."
    )
    parser.add_argument("--output_dir", default="outputs/language_assessment/")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--multimodal_models", nargs="*", default=[])
    parser.add_argument(
        "--clustering_metrics",
        default="outputs/clustering_eval/clustering_metrics.json",
        help="Path to clustering_metrics.json for difficulty correlation plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "language_assessment_metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}")
        sys.exit(1)

    results = json.loads(metrics_path.read_text())
    multimodal_set = set(args.multimodal_models)

    # ── Load aggregated metrics from JSON ──────────────────────────────────────
    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics:  dict[str, dict[str, dict]] = {}
    all_model_cls_by_task:  dict[str, dict[str, dict]] = {}
    all_model_desc_by_task: dict[str, dict[str, dict]] = {}
    all_judge_metrics:      dict[str, dict] = {}
    all_judge_by_task:      dict[str, dict] = {}
    all_judge_by_dataset:   dict[str, dict] = {}

    for model_id, m in results.get("models", {}).items():
        all_model_desc_metrics[model_id] = m.get("description_per_dataset", {})
        all_model_cls_metrics[model_id]  = m.get("classification_per_dataset", {})
        all_model_cls_by_task[model_id]  = m.get("classification_per_task", {})
        all_model_desc_by_task[model_id] = m.get("description_per_task", {})
        if m.get("judge"):
            all_judge_metrics[model_id] = m["judge"]
        if m.get("judge_by_task"):
            all_judge_by_task[model_id] = m["judge_by_task"]
        if m.get("judge_by_dataset"):
            all_judge_by_dataset[model_id] = m["judge_by_dataset"]

    if not all_model_desc_metrics:
        print("ERROR: no model results found in metrics JSON.")
        sys.exit(1)

    # ── Load per-row CSVs ──────────────────────────────────────────────────────
    from src.common.dataset_config import datasets_by_name
    datasets_cfg = datasets_by_name(Path(args.config))
    skip_desc_set = {name for name, ds in datasets_cfg.items() if ds.skip_description}
    ds_to_task = {k: v.task_type for k, v in datasets_cfg.items()}

    all_model_cls_rows:  dict[str, list[dict]] = {}
    all_model_desc_rows: dict[str, list[dict]] = {}

    for model_id in all_model_cls_metrics:
        safe = _safe_name(model_id)
        cls_csv  = output_dir / f"cls_{safe}.csv"
        desc_csv = output_dir / f"desc_{safe}.csv"
        if cls_csv.exists():
            all_model_cls_rows[model_id] = read_csv(cls_csv)
            print(f"  Loaded {cls_csv.name} ({len(all_model_cls_rows[model_id])} rows)")
        else:
            print(f"  WARNING: missing {cls_csv.name} — correlation/distribution plots may be incomplete")
            all_model_cls_rows[model_id] = []
        if desc_csv.exists():
            all_model_desc_rows[model_id] = [
                r for r in read_csv(desc_csv)
                if r.get("agml_dataset") not in skip_desc_set
            ]
            print(f"  Loaded {desc_csv.name} ({len(all_model_desc_rows[model_id])} rows)")
        else:
            print(f"  WARNING: missing {desc_csv.name} — correlation/distribution plots may be incomplete")
            all_model_desc_rows[model_id] = []

    # ── Naive baselines ────────────────────────────────────────────────────────
    naive = results.get("naive", {})
    summary_desc, summary_cls = _parse_summary_naive(output_dir / "summary.txt")
    naive_ov_desc = (
        naive.get("description_overall")
        or summary_desc
        or _macro_average(naive.get("description", {}), ["rouge1", "rouge2", "rougeL", "rouge_ss"])
    )
    naive_cls_overall = (
        naive.get("classification_overall")
        or summary_cls
        or _macro_average(naive.get("classification", {}), ["accuracy", "contains_accuracy", "f1_macro"])
    )

    active_datasets = {name: ds for name, ds in datasets_cfg.items() if not ds.skip}
    per_ds_f1 = [1.0 / max(len(ds.clean_classes), 1) for ds in active_datasets.values()]
    uniform_cls_overall = (
        {"f1_macro": round(float(np.mean(per_ds_f1)), 4),
         "accuracy": round(float(np.mean(per_ds_f1)), 4)}
        if per_ds_f1 else None
    )

    print(f"\nLoaded metrics for {len(all_model_desc_metrics)} model(s): "
          f"{', '.join(all_model_desc_metrics)}")
    print(f"Naive desc baseline:  { {k: round(v,3) for k,v in (naive_ov_desc or {}).items()} }")
    print(f"Naive cls  baseline:  { {k: round(v,3) for k,v in (naive_cls_overall or {}).items()} }")
    print(f"Uniform cls baseline: { {k: round(v,3) for k,v in (uniform_cls_overall or {}).items()} }")

    print("\n=== Generating plots ===")

    # ── Summary bar charts (JSON-based) ────────────────────────────────────────
    plot_cls_summary_per_dataset(
        all_model_cls_metrics, output_dir, multimodal_set,
        naive_cls_overall, uniform_cls_overall, all_judge_by_dataset)
    plot_cls_summary_per_task(
        all_model_cls_by_task, output_dir, multimodal_set,
        naive_cls_overall, all_judge_metrics, uniform_cls_overall, all_judge_by_task)
    plot_desc_summary_per_task(all_model_desc_by_task, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc)
    plot_desc_sem_sim_per_dataset(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc, ds_to_task)
    plot_overall_cls_summary(
        all_model_cls_metrics, output_dir, multimodal_set,
        naive_cls_overall, all_judge_metrics, uniform_cls_overall)

    # ── Correlation plots (CSV-based, uses judged_correct from cls CSVs) ───────
    plot_metric_correlation(all_model_desc_rows, all_model_cls_rows, output_dir)
    plot_difficulty_correlation(
        all_model_cls_metrics, all_model_desc_metrics,
        Path(args.clustering_metrics), output_dir)

    # ── Distribution charts: strip + errorbar (CSV-based) ─────────────────────
    models    = list(all_model_cls_metrics.keys())
    color_map = model_color_map(models)
    all_datasets  = sorted(set(ds for m in all_model_cls_metrics.values() for ds in m))
    task_types    = sorted(set(ds_to_task.get(ds, "") for ds in all_datasets) - {""})

    cls_by_ds         = gather_cls_values_by_dataset(all_model_cls_metrics)
    cls_by_cls        = gather_cls_per_class_by_dataset(all_model_cls_rows, "ground_truth", "resolved_class", ds_to_task)
    judge_by_cls      = gather_judge_per_class_by_dataset(all_model_cls_rows, "ground_truth", ds_to_task)
    cls_by_task       = gather_cls_values_by_task_from_per_class(cls_by_cls)
    _dt_key           = lambda k: (k[1], k[0]) if isinstance(k, tuple) else ("", k)
    cls_dataset_tasks = sorted(set(k for m in cls_by_cls.values() for k in m), key=_dt_key)
    desc_by_ds        = gather_desc_values_by_dataset(all_model_desc_metrics)
    desc_by_smp       = gather_desc_per_sample_by_dataset(all_model_desc_rows, ds_to_task=ds_to_task)
    desc_by_task      = gather_desc_values_by_task_from_per_sample(desc_by_smp)
    desc_dataset_tasks = sorted(set(k for m in desc_by_smp.values() for k in m), key=_dt_key)
    judge_by_ds_dist  = gather_judge_values_by_dataset(all_judge_by_dataset)
    judge_by_task_dist = gather_cls_values_by_task_from_per_class(judge_by_cls)

    uniform_f1    = uniform_cls_overall.get("f1_macro") if uniform_cls_overall else None
    naive_sem_sim = naive_ov_desc.get("sem_sim") if naive_ov_desc else None

    for chart_type, suffix in [("strip", "strip"), ("errorbar", "errorbar")]:
        plot_overall_distribution(
            cls_by_ds, models, color_map, multimodal_set, "F1 Macro",
            "Language Assessment — Classification Overall",
            output_dir / f"cls_overall_{suffix}.png",
            chart_type=chart_type, naive_line=uniform_f1,
            judge_values_by_model=judge_by_ds_dist)
        plot_per_task_distribution(
            cls_by_task, task_types, models, color_map, multimodal_set, _task_label,
            "F1 Macro", "Language Assessment — Classification Per Task",
            output_dir / f"cls_per_task_{suffix}.png",
            chart_type=chart_type, naive_line=uniform_f1,
            judge_values_by_model_by_task=judge_by_task_dist)
        plot_per_dataset_distribution(
            cls_by_cls, cls_dataset_tasks, models, color_map, multimodal_set,
            "F1 per Class", "Language Assessment — Classification Per Dataset (per-class spread)",
            output_dir / f"cls_per_dataset_{suffix}.png",
            chart_type=chart_type, naive_line=uniform_f1,
            judge_values_by_model_by_dataset=judge_by_cls)
        plot_overall_distribution(
            desc_by_ds, models, color_map, multimodal_set, "Sem Sim",
            "Language Assessment — Description Overall",
            output_dir / f"desc_overall_{suffix}.png",
            chart_type=chart_type, naive_line=naive_sem_sim)
        plot_per_task_distribution(
            desc_by_task, task_types, models, color_map, multimodal_set, _task_label,
            "Sem Sim", "Language Assessment — Description Per Task",
            output_dir / f"desc_per_task_{suffix}.png",
            chart_type=chart_type, naive_line=naive_sem_sim)
        plot_per_dataset_distribution(
            desc_by_smp, desc_dataset_tasks, models, color_map, multimodal_set,
            "Sem Sim per Sample",
            "Language Assessment — Description Per Dataset (per-sample spread)",
            output_dir / f"desc_per_dataset_{suffix}.png",
            chart_type=chart_type, naive_line=naive_sem_sim)

    print(f"\nDone. Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
