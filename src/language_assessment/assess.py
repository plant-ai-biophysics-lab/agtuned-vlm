"""
Stage 3: Language Assessment — Text-only agricultural knowledge evaluation.

Evaluates language models on two JSONL datasets produced by Stage 1a:
  - agml_descriptions.jsonl: open-ended visual description Q&A (ROUGE-SS metric)
  - agml_classification.jsonl: short-answer classification Q&A (F1-macro metric)

Models are loaded one at a time, evaluated on both tasks, and unloaded before
the next model. A naive shuffled baseline is computed for each task.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import json
import multiprocessing
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from vllm import LLM, SamplingParams

from src.common.dataset_config import datasets_by_name
from src.common.io import (
    Tee,
    load_jsonl,
    load_prompts,
    save_json,
    update_pipeline_results,
    write_csv,
)
from src.common.metrics import (
    aggregate_by_task,
    aggregate_rouge,
    apply_judge,
    compute_classification_metrics,
    contains_match,
    judged_summary,
    naive_classification_baseline,
    naive_rouge_baseline,
    resolve_class,
    rouge_scores_per_sample,
)

_SAMPLING = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    repetition_penalty=1.1,
    stop=["Question:", "\n\nQ:", "\n\nAnswer:", "[Answer"],
)

_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_TASK_LABELS = {"pest_damage": "pest/damage"}


def _task_label(tt: str) -> str:
    return _TASK_LABELS.get(tt, tt.replace("_", " "))


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def _clean_response(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    think_start = text.lower().find("<think>")
    if think_start != -1:
        text = text[:think_start].strip()
    bracket = text.find("[Answer")
    if bracket != -1:
        text = text[:bracket].strip()
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text


def _apply_chat_template(tokenizer, question: str, prompt_template: str) -> str:
    messages = [{"role": "user", "content": prompt_template.format(question=question)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "/no_think"}, *messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def load_llm(
    model_id: str,
    quantization: str | None = None,
    max_num_seqs: int = 32,
) -> tuple[LLM, object]:
    from transformers import AutoTokenizer
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        max_num_seqs=max_num_seqs,
        quantization=quantization,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return llm, tokenizer


def destroy_llm(llm) -> None:
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_inference(
    llm: LLM,
    prompts: list[str],
    label: str,
    tokenizer,
    prompt_template: str,
) -> tuple[list[str], float, int]:
    formatted = [_apply_chat_template(tokenizer, p, prompt_template) for p in prompts]
    print(f"\nRunning inference [{label}]...")
    t_start = time.perf_counter()
    outputs = llm.generate(formatted, sampling_params=_SAMPLING)
    elapsed = time.perf_counter() - t_start
    responses = [_clean_response(o.outputs[0].text) for o in outputs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {elapsed:.1f}s | {total_tokens / elapsed:.0f} tok/s")
    return responses, elapsed, total_tokens


def run_descriptions(
    llm: LLM,
    tokenizer,
    records: list[dict],
    prompt_template: str,
) -> list[dict]:
    """Run description inference and compute per-record ROUGE scores."""
    questions = [r["messages"][0]["content"] for r in records]
    ground_truths = [r["messages"][1]["content"] for r in records]

    responses, elapsed, _ = run_inference(llm, questions, "descriptions", tokenizer, prompt_template)

    rows = []
    for record, resp, gt in zip(records, responses, ground_truths):
        meta = record.get("metadata", {})
        scores = rouge_scores_per_sample(resp, gt)
        rows.append({
            "agml_dataset": meta.get("agml_dataset", ""),
            "class_label": meta.get("class_label", ""),
            "task_type": meta.get("task_type", ""),
            "question": record["messages"][0]["content"],
            "ground_truth": gt,
            "prediction": resp,
            **scores,
        })
    return rows


def run_classifications(
    llm: LLM,
    tokenizer,
    records: list[dict],
    prompt_template: str,
    classification_suffix: str,
    datasets_cfg: dict,
) -> list[dict]:
    """Run classification inference and compute per-record resolve/match scores."""
    questions = [r["messages"][0]["content"] + classification_suffix for r in records]
    ground_truths = [r["messages"][1]["content"] for r in records]

    responses, elapsed, _ = run_inference(llm, questions, "classification", tokenizer, prompt_template)

    rows = []
    for record, resp, gt in zip(records, responses, ground_truths):
        meta = record.get("metadata", {})
        agml_name = meta.get("agml_dataset", "")
        ds = datasets_cfg.get(agml_name)
        classes = ds.clean_classes if ds else []
        resolved = resolve_class(resp, classes) if classes else None
        rows.append({
            "agml_dataset": agml_name,
            "class_label": gt,
            "task_type": meta.get("task_type", ""),
            "question": record["messages"][0]["content"],
            "ground_truth": gt,
            "prediction": resp,
            "resolved_class": resolved or "",
            "resolvable": int(resolved is not None),
            "contains_match": int(contains_match(resp, gt)),
        })
    return rows


def compute_desc_metrics_per_dataset(
    desc_rows: list[dict],
) -> dict[str, dict]:
    """Returns {agml_name: rouge_summary} for each dataset."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in desc_rows:
        by_ds[row["agml_dataset"]].append(row)
    return {ds: aggregate_rouge(rows) for ds, rows in by_ds.items()}


def compute_cls_metrics_per_dataset(
    cls_rows: list[dict],
) -> dict[str, dict]:
    """Returns {agml_name: classification_metrics} for each dataset."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in cls_rows:
        by_ds[row["agml_dataset"]].append(row)

    result = {}
    for ds, rows in by_ds.items():
        refs = [r["ground_truth"] for r in rows]
        preds = [r["resolved_class"] for r in rows]
        result[ds] = compute_classification_metrics(refs, preds)
    return result


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_cls_summary_per_dataset(
    all_cls_results: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Grouped bar chart: classification metrics per AgML dataset, one group per model."""
    all_datasets = []
    for model_results in all_cls_results.values():
        for ds in model_results:
            if ds not in all_datasets:
                all_datasets.append(ds)
    if not all_datasets:
        return

    models = list(all_cls_results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(models)))
    x = np.arange(len(all_datasets))
    width = 0.8 / (len(models) * len(metrics))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(10, len(all_datasets) * 0.8), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            per_ds = all_cls_results[model_key]
            vals = [per_ds.get(ds, {}).get(metric, 0.0) for ds in all_datasets]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Language Assessment — Classification Per Dataset", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cls_summary_per_task(
    all_cls_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: classification metrics per task_type, one group per model."""
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = list(all_cls_results.keys())
    metrics = ["accuracy", "contains_accuracy", "f1_macro"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(task_types))
    width = 0.8 / (len(models) * len(metrics))

    task_scores: dict[str, dict[str, dict[str, list]]] = {
        model_key: {tt: {m: [] for m in metrics} for tt in task_types}
        for model_key in models
    }
    for model_key, per_ds in all_cls_results.items():
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                for metric in metrics:
                    task_scores[model_key][tt][metric].append(m.get(metric, 0.0))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(10, len(task_types) * 2), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            vals = [
                float(np.mean(task_scores[model_key][tt][metric]))
                if task_scores[model_key][tt][metric] else 0.0
                for tt in task_types
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Language Assessment — Classification Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_desc_summary_per_task(
    all_desc_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: ROUGE metrics per task_type for the description task."""
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = list(all_desc_results.keys())
    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(task_types))
    width = 0.8 / (len(models) * len(metrics))

    task_scores: dict[str, dict[str, dict[str, list]]] = {
        mk: {tt: {m: [] for m in metrics} for tt in task_types}
        for mk in models
    }
    for model_key, per_ds in all_desc_results.items():
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                for metric in metrics:
                    task_scores[model_key][tt][metric].append(m.get(metric, 0.0))

    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(12, len(task_types) * 2.5), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            vals = [
                float(np.mean(task_scores[model_key][tt][metric]))
                if task_scores[model_key][tt][metric] else 0.0
                for tt in task_types
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Language Assessment — Description ROUGE Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "desc_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_desc_summary(
    all_desc_results: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Bar chart: overall ROUGE scores per model averaged across all datasets."""
    models = list(all_desc_results.keys())
    if not models:
        return

    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss"]
    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS"}
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(8, len(models) * 2), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_desc_results[mk].values()]))
            if all_desc_results[mk] else 0.0
            for mk in models
        ]
        bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=20, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score" if mi == 0 else "")

    fig.suptitle("Language Assessment — Overall Description ROUGE", fontsize=11)
    plt.tight_layout()
    path = output_dir / "overall_desc_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_cls_summary(
    all_cls_results: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Bar chart: overall classification scores per model averaged across all datasets."""
    models = list(all_cls_results.keys())
    if not models:
        return

    metrics = ["accuracy", "contains_accuracy", "f1_macro"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(8, len(models) * 2), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_cls_results[mk].values()]))
            if all_cls_results[mk] else 0.0
            for mk in models
        ]
        bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=20, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")

    fig.suptitle("Language Assessment — Overall Classification", fontsize=11)
    plt.tight_layout()
    path = output_dir / "overall_cls_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _worker_language(kwargs: dict) -> None:
    """Runs in a subprocess — loads one model, runs inference, saves results, exits."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.common.dataset_config import DatasetConfig

    model_id      = kwargs["model_id"]
    quantization  = kwargs["quantization"]
    max_num_seqs  = kwargs.get("max_num_seqs", 128)
    out_path      = kwargs["out_path"]

    # Reconstruct DatasetConfig objects from plain dicts
    datasets_cfg = {k: DatasetConfig(**v) for k, v in kwargs["datasets_cfg"].items()}

    llm, tokenizer = load_llm(model_id, quantization=quantization, max_num_seqs=max_num_seqs)
    desc_rows = run_descriptions(llm, tokenizer, kwargs["desc_records"], kwargs["prompt_template"])
    cls_rows  = run_classifications(
        llm, tokenizer, kwargs["cls_records"],
        kwargs["prompt_template"], kwargs["classification_suffix"],
        datasets_cfg,
    )
    del tokenizer, llm

    with open(out_path, "w") as f:
        json.dump({"desc_rows": desc_rows, "cls_rows": cls_rows}, f)


def _run_in_subprocess(fn, kwargs: dict) -> None:
    """Spawns fn(kwargs) in a fresh process and waits for it to finish."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=fn, args=(kwargs,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Worker process exited with code {p.exitcode}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    prompt_template = prompts["assessment_descriptive"]
    classification_suffix = prompts["classification_suffix"]

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task = {name: ds.task_type for name, ds in datasets_cfg.items()}

    desc_records = load_jsonl(Path(args.descriptions_jsonl), args.n_samples)
    cls_records = load_jsonl(Path(args.classification_jsonl), args.n_samples)
    print(f"Description records: {len(desc_records)}")
    print(f"Classification records: {len(cls_records)}")

    # Naive baselines
    naive_desc_rows = [
        {**{"ground_truth": r["messages"][1]["content"]},
         **rouge_scores_per_sample(r["messages"][1]["content"], r["messages"][1]["content"])}
        for r in desc_records
    ]
    naive_desc_rows_shuffled_pairs = list(zip(
        [r["messages"][1]["content"] for r in desc_records],
        np.random.default_rng(42).permutation([r["messages"][1]["content"] for r in desc_records]).tolist()
    ))
    naive_desc_per_ds: dict[str, list] = defaultdict(list)
    for record, (gt, shuffled) in zip(desc_records, naive_desc_rows_shuffled_pairs):
        agml_name = record.get("metadata", {}).get("agml_dataset", "")
        naive_desc_per_ds[agml_name].append({**rouge_scores_per_sample(shuffled, gt), "ground_truth": gt})
    naive_desc_metrics = {ds: aggregate_rouge(rows) for ds, rows in naive_desc_per_ds.items()}

    naive_cls_refs_by_ds: dict[str, list[str]] = defaultdict(list)
    for r in cls_records:
        agml_name = r.get("metadata", {}).get("agml_dataset", "")
        naive_cls_refs_by_ds[agml_name].append(r["messages"][1]["content"])
    naive_cls_metrics: dict[str, dict] = {}
    for ds, refs in naive_cls_refs_by_ds.items():
        naive_cls_metrics[ds] = naive_classification_baseline(refs)

    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics: dict[str, dict[str, dict]] = {}
    summary_metrics: dict[str, dict] = {}

    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        # Run each model in a subprocess so the OS fully reclaims GPU memory on exit
        worker_out = output_dir / f"_worker_{_safe_name(model_id)}.json"
        _run_in_subprocess(_worker_language, dict(
            model_id=model_id,
            quantization=args.quantization,
            max_num_seqs=args.max_num_seqs,
            desc_records=desc_records,
            cls_records=cls_records,
            prompt_template=prompt_template,
            classification_suffix=classification_suffix,
            datasets_cfg={k: vars(v) for k, v in datasets_cfg.items()},
            out_path=str(worker_out),
        ))
        worker_result = json.loads(worker_out.read_text())
        worker_out.unlink()
        desc_rows = worker_result["desc_rows"]
        cls_rows  = worker_result["cls_rows"]

        desc_per_ds = compute_desc_metrics_per_dataset(desc_rows)
        cls_per_ds = compute_cls_metrics_per_dataset(cls_rows)

        desc_overall = aggregate_rouge(desc_rows)
        cls_refs = [r["ground_truth"] for r in cls_rows]
        cls_preds = [r["resolved_class"] for r in cls_rows]
        cls_overall = compute_classification_metrics(cls_refs, cls_preds)

        all_model_desc_metrics[model_id] = desc_per_ds
        all_model_cls_metrics[model_id] = cls_per_ds
        summary_metrics[model_id] = {
            "description_overall": desc_overall,
            "description_per_dataset": desc_per_ds,
            "classification_overall": cls_overall,
            "classification_per_dataset": cls_per_ds,
        }

        write_csv(output_dir / f"desc_{_safe_name(model_id)}.csv", desc_rows)
        write_csv(output_dir / f"cls_{_safe_name(model_id)}.csv", cls_rows)

    # LLM judge (classification only)
    if args.judge_model:
        print(f"\n=== LLM Judge: {args.judge_model} ===")
        judge_llm, judge_tok = load_llm(args.judge_model)

        for model_id in args.models:
            cls_csv = output_dir / f"cls_{_safe_name(model_id)}.csv"
            if not cls_csv.exists():
                continue
            import csv as _csv
            with open(cls_csv, newline="", encoding="utf-8") as f:
                cls_rows = list(_csv.DictReader(f))

            print(f"  Judging {model_id} ({len(cls_rows)} rows)...")
            cls_rows = apply_judge(
                rows=cls_rows,
                llm=judge_llm,
                tokenizer=judge_tok,
                system_prompt=prompts["judge_system_language"],
                judge_template=prompts["judge_classification_template"],
                gt_key="ground_truth",
                pred_key="prediction",
                confidence_threshold=args.judge_confidence_threshold,
            )
            write_csv(cls_csv, cls_rows)
            summary_metrics[model_id]["judge"] = judged_summary(cls_rows)

        del judge_llm, judge_tok
        gc.collect()

    # Save metrics JSON
    full_results = {
        "naive": {
            "description": naive_desc_metrics,
            "classification": naive_cls_metrics,
        },
        "models": summary_metrics,
    }
    save_json(output_dir / "language_assessment_metrics.json", full_results)

    # Summary output
    tee = Tee(output_dir / "summary.txt")

    tee.print("\n=== Language Assessment — Description ROUGE ===")
    tee.print(f"  {'Model':<45} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'R-SS':>6}")
    tee.print(f"  {'-'*66}")
    naive_ov_desc = aggregate_rouge([
        row for rows in naive_desc_per_ds.values() for row in rows
    ]) if naive_desc_per_ds else {}
    tee.print(f"  {'naive':<45} "
              f"{naive_ov_desc.get('rouge1', 0):>6.3f} {naive_ov_desc.get('rouge2', 0):>6.3f} "
              f"{naive_ov_desc.get('rougeL', 0):>6.3f} {naive_ov_desc.get('rouge_ss', 0):>6.3f}")
    for model_id, m in summary_metrics.items():
        ov = m["description_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('rouge1', 0):>6.3f} {ov.get('rouge2', 0):>6.3f} "
                  f"{ov.get('rougeL', 0):>6.3f} {ov.get('rouge_ss', 0):>6.3f}")

    tee.print("\n=== Language Assessment — Classification ===")
    tee.print(f"  {'Model':<45} {'Acc':>6} {'Cont':>6} {'F1-mac':>8} {'Res/Tot':>9}")
    tee.print(f"  {'-'*77}")
    for ds, m in naive_cls_metrics.items():
        pass
    naive_cls_all_refs = [r["messages"][1]["content"] for r in cls_records]
    naive_cls_overall = naive_classification_baseline(naive_cls_all_refs)
    tee.print(f"  {'naive':<45} "
              f"{naive_cls_overall.get('accuracy', 0):>6.3f} "
              f"{naive_cls_overall.get('contains_accuracy', 0):>6.3f} "
              f"{naive_cls_overall.get('f1_macro', 0):>8.3f} "
              f"{naive_cls_overall.get('n_resolvable', 0):>4}/{naive_cls_overall.get('n_total', 0):<4}")
    for model_id, m in summary_metrics.items():
        ov = m["classification_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('accuracy', 0):>6.3f} "
                  f"{ov.get('contains_accuracy', 0):>6.3f} "
                  f"{ov.get('f1_macro', 0):>8.3f} "
                  f"{ov.get('n_resolvable', 0):>4}/{ov.get('n_total', 0):<4}")

    if args.judge_model and any("judge" in m for m in summary_metrics.values()):
        tee.print(f"\n=== LLM Judge — {args.judge_model} ===")
        tee.print(f"  {'Model':<45} {'J-Acc':>7} {'J-F1':>7} {'N':>6}")
        tee.print(f"  {'-'*68}")
        for model_id, m in summary_metrics.items():
            jm = m.get("judge", {})
            if jm:
                tee.print(f"  {model_id:<45} "
                          f"{jm.get('judge_accuracy', 0):>7.3f} "
                          f"{jm.get('judge_f1_macro', 0):>7.3f} "
                          f"{jm.get('n_judged', 0):>6}")

    tee.close()

    # Plots
    print("\n=== Generating plots ===")
    plot_cls_summary_per_dataset(all_model_cls_metrics, output_dir)
    plot_cls_summary_per_task(all_model_cls_metrics, ds_to_task, output_dir)
    plot_desc_summary_per_task(all_model_desc_metrics, ds_to_task, output_dir)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir)
    plot_overall_cls_summary(all_model_cls_metrics, output_dir)

    # Update pipeline results CSV
    pipeline_csv = output_dir.parent / "pipeline_results.csv"
    for model_id, m in summary_metrics.items():
        update_pipeline_results(
            csv_path=pipeline_csv,
            model=model_id,
            model_type="language_model",
            stage="language_assessment",
            metrics={
                "description_rouge_ss": m["description_overall"].get("rouge_ss", ""),
                "classification_f1_macro": m["classification_overall"].get("f1_macro", ""),
            },
        )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Language assessment: text-only VQA evaluation on AgML datasets."
    )
    parser.add_argument(
        "--descriptions_jsonl", default="outputs/data_generation/agml_descriptions.jsonl",
        help="Path to agml_descriptions.jsonl",
    )
    parser.add_argument(
        "--classification_jsonl", default="outputs/data_generation/agml_classification.jsonl",
        help="Path to agml_classification.jsonl",
    )
    parser.add_argument(
        "--config", default="src/config/datasets.yaml",
        help="Path to datasets.yaml",
    )
    parser.add_argument(
        "--prompts", default="src/config/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Space-separated list of model IDs to evaluate",
    )
    parser.add_argument(
        "--judge_model", default=None,
        help="LLM-as-a-judge model (runs after all eval models; classification only)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/language_assessment/",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit number of records per JSONL (default: all)",
    )
    parser.add_argument(
        "--judge_reasoning", default="medium", choices=["low", "medium", "high"],
    )
    parser.add_argument(
        "--judge_confidence_threshold", type=int, default=1,
        help="Minimum confidence (0-2) required to count as correct (default: 1)",
    )
    parser.add_argument(
        "--quantization", default=None, choices=["awq", "gptq", "fp8", "bitsandbytes"],
        help="vLLM quantization method for loading models (use 'bitsandbytes' for large MoE models on 80GB)",
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=32,
        help="Max concurrent sequences for vLLM (default 32 is safe for Mamba models; raise for throughput)",
    )
    args = parser.parse_args()
    main(args)
