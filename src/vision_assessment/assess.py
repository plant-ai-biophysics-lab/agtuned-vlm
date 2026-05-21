"""
Stage 4: Vision Assessment — Multimodal VQA evaluation on AgML datasets.

Evaluates vision-language models on two tasks using the description JSONL
produced by Stage 1a:
  - Description task: image + visual question → ROUGE-SS vs synthesized reference
  - Classification task: image + classification question → F1-macro vs class label

Models are loaded one at a time, evaluated on both tasks, and unloaded before
the next. A naive shuffled baseline is computed for each task.
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
from PIL import Image as PILImage
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

DESC_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
CLS_PARAMS = SamplingParams(
    temperature=0.0, max_tokens=32, repetition_penalty=1.1,
    stop=["\n", ".", ",", ";"],
)

_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*{1,3}([^*]*)\*{1,3}|^#{1,6}\s*|^[-*+]\s+", re.MULTILINE)
_GREETING_RE = re.compile(
    r"^(what (a )?(great|good|excellent|interesting|wonderful|nice)|"
    r"sure[!,.]?|great[!,.]?|of course[!,.]?|absolutely[!,.]?|"
    r"certainly[!,.]?|hello[!,.]?|hi[!,.]?|yes[!,.]?)\s*$",
    re.IGNORECASE,
)

_TASK_LABELS = {"pest_damage": "pest/damage"}

_HEALTHY_DESCRIPTION = "This leaf/plant is healthy."

_JUDGE_METRIC_MAP = {"f1_macro": "judge_f1_macro", "accuracy": "judge_accuracy"}


def _apply_healthy_override(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        label = r.get("metadata", {}).get("class_label", "")
        if "healthy" in label.lower():
            r = {**r, "messages": [r["messages"][0],
                                   {**r["messages"][1], "content": _HEALTHY_DESCRIPTION}]}
        out.append(r)
    return out


def _task_label(tt: str) -> str:
    return _TASK_LABELS.get(tt, tt.replace("_", " "))


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def _parse_model_quantization(entries: list[str]) -> dict[str, str]:
    """Parse ['model_id:method', ...] into {model_id: method}."""
    result = {}
    valid = {"awq", "gptq", "fp8", "bitsandbytes"}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"--model_quantization entry {entry!r} must be in 'model_id:method' format"
            )
        model_id, _, method = entry.rpartition(":")
        if method not in valid:
            raise ValueError(f"Unknown quantization {method!r} for {model_id!r}. Choose from: {valid}")
        result[model_id] = method
    return result


def _clean_response(text: str) -> str:
    # Strip think blocks
    text = _THINK_RE.sub("", text).strip()
    think_start = text.lower().find("<think>")
    if think_start != -1:
        text = text[:think_start].strip()

    # Strip markdown (headers, bold/italic, bullet markers) — keep inner text
    text = _MARKDOWN_RE.sub(r"\1", text)

    # Remove greeting-only lines at the start
    lines = text.splitlines()
    while lines and _GREETING_RE.match(lines[0].strip()):
        lines.pop(0)
    text = "\n".join(lines)

    # Take first non-empty paragraph (stop at blank line)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        text = paragraphs[0]

    # Collapse internal whitespace / newlines within the paragraph
    text = " ".join(text.split())

    # Cut repetition loops
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()

    return text.strip()


def _apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "/no_think"}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


_MAMBA_MAX_NUM_SEQS = 32
_MAMBA_MODEL_PREFIXES = {"mamba", "jamba", "zamba", "falcon_mamba", "bamba", "qwen"}


def _is_mamba(model_id: str) -> bool:
    name = model_id.lower().split("/")[-1]
    return any(name.startswith(p) for p in _MAMBA_MODEL_PREFIXES)


def load_llm(
    model_id: str,
    quantization: str | None = None,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.7,
) -> tuple[LLM, object]:
    if _is_mamba(model_id):
        effective_seqs = min(max_num_seqs, _MAMBA_MAX_NUM_SEQS)
        if effective_seqs < max_num_seqs:
            print(f"  [load_llm] Mamba architecture detected — capping max_num_seqs at {effective_seqs}")
        max_num_seqs = effective_seqs

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        quantization=quantization,
    )
    return llm, llm.get_tokenizer()


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


def build_multimodal_prompt(tokenizer, image: PILImage.Image, question: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def run_batch(
    llm: LLM,
    tokenizer,
    images: list[PILImage.Image],
    question: str,
    params: SamplingParams,
) -> list[str]:
    prompts = [build_multimodal_prompt(tokenizer, img, question) for img in images]
    outputs = llm.generate(prompts, sampling_params=params)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def build_name_map(agml_name: str) -> dict[str, str]:
    """Returns {filename: full_path} for all images in an AgML dataset."""
    import agml
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    return {Path(str(p)).name: str(p) for p in contents.keys()}


def run_model(
    model_id: str,
    records: list[dict],
    datasets_cfg: dict,
    classification_suffix: str,
    n_samples: int | None,
    output_dir: Path,
    quantization: str | None = None,
    max_num_seqs: int = 64,
) -> tuple[list[dict], list[dict]]:
    """Load model, run both tasks on all records, unload, return (desc_rows, cls_rows)."""
    llm, tokenizer = load_llm(model_id, quantization=quantization, max_num_seqs=max_num_seqs)

    desc_rows: list[dict] = []
    cls_rows: list[dict] = []
    agml_cache: dict[str, dict[str, str]] = {}

    t_start = time.perf_counter()

    for record in records:
        meta = record["metadata"]
        agml_name = meta["agml_dataset"]
        class_label = meta["class_label"]
        task_type = meta["task_type"]
        crop = meta["crop"]
        image_names = meta["image_names"]
        synth_desc = record["messages"][1]["content"]
        visual_desc_q = meta["visual_question"]
        visual_cls_q = meta.get("visual_cls_question")

        if n_samples:
            image_names = image_names[:n_samples]

        if agml_name not in agml_cache:
            print(f"\n  Building image map: {agml_name}...")
            agml_cache[agml_name] = build_name_map(agml_name)
        name_map = agml_cache[agml_name]

        images, valid_names = [], []
        for name in image_names:
            if name in name_map:
                images.append(PILImage.open(name_map[name]).convert("RGB"))
                valid_names.append(name)
            else:
                print(f"    [warn] {name} not in {agml_name}")
        if not images:
            continue

        print(f"  {agml_name} / {class_label}  ({len(images)} images)")

        # Description task
        desc_preds = run_batch(llm, tokenizer, images, visual_desc_q, DESC_PARAMS)
        for pred, img_name in zip(desc_preds, valid_names):
            scores = rouge_scores_per_sample(pred, synth_desc)
            desc_rows.append({
                "agml_dataset": agml_name,
                "class_label": class_label,
                "task_type": task_type,
                "crop": crop,
                "image_name": img_name,
                "prediction": pred,
                "ground_truth": synth_desc,
                **scores,
            })

        # Classification task
        if visual_cls_q:
            ds_cfg = datasets_cfg.get(agml_name)
            clean_classes = ds_cfg.clean_classes if ds_cfg else []
            cls_question = visual_cls_q + classification_suffix

            cls_preds = run_batch(llm, tokenizer, images, cls_question, CLS_PARAMS)
            resolved_list = [resolve_class(p, clean_classes) for p in cls_preds]
            for pred, img_name, resolved in zip(cls_preds, valid_names, resolved_list):
                correct = resolved is not None and resolved.lower() == class_label.lower()
                c_match = contains_match(resolved or pred, class_label)
                cls_rows.append({
                    "agml_dataset": agml_name,
                    "class_label": class_label,
                    "task_type": task_type,
                    "crop": crop,
                    "image_name": img_name,
                    "prediction": pred,
                    "resolved_class": resolved or "",
                    "ground_truth": class_label,
                    "correct": int(correct),
                    "contains_match": int(c_match),
                    "resolvable": int(resolved is not None),
                })

    elapsed = time.perf_counter() - t_start
    print(f"\nInference time: {elapsed:.1f}s")

    destroy_llm(llm)
    del tokenizer

    return desc_rows, cls_rows


def compute_desc_metrics_per_dataset(desc_rows: list[dict]) -> dict[str, dict]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in desc_rows:
        by_ds[row["agml_dataset"]].append(row)
    return {ds: aggregate_rouge(rows) for ds, rows in by_ds.items()}


def compute_cls_metrics_per_dataset(cls_rows: list[dict]) -> dict[str, dict]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in cls_rows:
        by_ds[row["agml_dataset"]].append(row)

    result = {}
    for ds, rows in by_ds.items():
        refs = [r["class_label"] for r in rows]
        preds = [r["resolved_class"] for r in rows]
        result[ds] = compute_classification_metrics(refs, preds)
    return result


# ── Plots ──────────────────────────────────────────────────────────────────────

def _hatch(model_key: str, multimodal_models: set[str]) -> str:
    return "///" if model_key in multimodal_models else ""


def _draw_naive_line(ax, naive_overall: dict | None, metric: str, label: bool = False) -> None:
    if not naive_overall:
        return
    value = naive_overall.get(metric)
    if value is None:
        return
    ax.axhline(value, color="red", linestyle="--", linewidth=1.2,
               label="naive baseline" if label else "_nolegend_")


_JUDGE_METRIC_MAP = {"accuracy": "judge_accuracy", "f1_macro": "judge_f1_macro"}


def _draw_judge_marker(ax, bar, value: float, label: bool = False) -> None:
    """Draw a thick horizontal tick across a bar at the judge score, with value label."""
    x0 = bar.get_x()
    w = bar.get_width()
    ax.plot([x0, x0 + w], [value, value], color="black", linewidth=2.5,
            solid_capstyle="butt", zorder=5,
            label="judge score" if label else "_nolegend_")
    ax.text(x0 + w / 2, value + 0.005, f"{value:.2f}",
            ha="center", va="bottom", fontsize=6, color="black", zorder=6)


def plot_cls_summary_per_dataset(
    all_cls_results: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_cls_metrics: dict | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
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
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(12, len(all_datasets) * len(models) * 0.5), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            per_ds = all_cls_results[model_key]
            vals = [per_ds.get(ds, {}).get(metric, 0.0) for ds in all_datasets]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85,
                   hatch=_hatch(model_key, multimodal_models), edgecolor="black", linewidth=0.5)
        if naive_cls_metrics:
            _draw_naive_line(ax, naive_cls_metrics, metric, label=(mi == 0))
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

    fig.suptitle("Vision Assessment — Classification Per Dataset", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cls_summary_per_task(
    all_cls_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_cls_metrics: dict | None = None,
    all_judge_metrics: dict[str, dict] | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    all_judge_metrics = all_judge_metrics or {}
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = list(all_cls_results.keys())
    metrics = ["accuracy", "contains_accuracy", "f1_macro"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(task_types))
    width = 0.8 / len(models)

    task_scores: dict[str, dict[str, dict[str, list]]] = {
        mk: {tt: {m: [] for m in metrics} for tt in task_types} for mk in models
    }
    for model_key, per_ds in all_cls_results.items():
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                for metric in metrics:
                    task_scores[model_key][tt][metric].append(m.get(metric, 0.0))

    subplot_w = max(4, len(task_types) * (len(models) * 0.6 + 0.4))
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=True)
    judge_label_drawn = False
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        judge_key = _JUDGE_METRIC_MAP.get(metric)
        for i, (model_key, color) in enumerate(zip(models, colors)):
            vals = [
                float(np.mean(task_scores[model_key][tt][metric]))
                if task_scores[model_key][tt][metric] else 0.0
                for tt in task_types
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85,
                          hatch=_hatch(model_key, multimodal_models), edgecolor="black", linewidth=0.5)
            jval = all_judge_metrics.get(model_key, {}).get(judge_key) if judge_key else None
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
                if jval is not None:
                    _draw_judge_marker(ax, bar, jval, label=(not judge_label_drawn))
                    judge_label_drawn = True
        if naive_cls_metrics:
            _draw_naive_line(ax, naive_cls_metrics, metric, label=(mi == 0))
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Classification Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_desc_summary_per_task(
    all_desc_results: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_desc_metrics: dict | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = list(all_desc_results.keys())
    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(task_types))
    width = 0.8 / len(models)

    task_scores: dict[str, dict[str, dict[str, list]]] = {
        mk: {tt: {m: [] for m in metrics} for tt in task_types} for mk in models
    }
    for model_key, per_ds in all_desc_results.items():
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                for metric in metrics:
                    task_scores[model_key][tt][metric].append(m.get(metric, 0.0))

    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS"}
    subplot_w = max(4, len(task_types) * (len(models) * 0.6 + 0.4))
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            vals = [
                float(np.mean(task_scores[model_key][tt][metric]))
                if task_scores[model_key][tt][metric] else 0.0
                for tt in task_types
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85,
                          hatch=_hatch(model_key, multimodal_models), edgecolor="black", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        if naive_desc_metrics:
            _draw_naive_line(ax, naive_desc_metrics, metric, label=(mi == 0))
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Description ROUGE Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "desc_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_desc_summary(
    all_desc_results: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_desc_metrics: dict | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    models = list(all_desc_results.keys())
    if not models:
        return

    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss"]
    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS"}
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(models))
    bar_w = 0.7
    subplot_w = max(3.5, len(models) * 1.5)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_desc_results[mk].values()]))
            if all_desc_results[mk] else 0.0
            for mk in models
        ]
        for xi, (mk, color, v) in enumerate(zip(models, colors, vals)):
            ax.bar(xi, v, bar_w, color=color, alpha=0.85,
                   hatch=_hatch(mk, multimodal_models), edgecolor="black", linewidth=0.5,
                   label=mk)
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        if naive_desc_metrics:
            _draw_naive_line(ax, naive_desc_metrics, metric, label=(mi == 0))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Overall Description ROUGE", fontsize=11)
    plt.tight_layout()
    path = output_dir / "overall_desc_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_cls_summary(
    all_cls_results: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_cls_metrics: dict | None = None,
    all_judge_metrics: dict[str, dict] | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    all_judge_metrics = all_judge_metrics or {}
    models = list(all_cls_results.keys())
    if not models:
        return

    metrics = ["accuracy", "contains_accuracy", "f1_macro"]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    x = np.arange(len(models))
    bar_w = 0.7
    subplot_w = max(3.5, len(models) * 1.5)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=True)
    judge_label_drawn = False
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        judge_key = _JUDGE_METRIC_MAP.get(metric)
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_cls_results[mk].values()]))
            if all_cls_results[mk] else 0.0
            for mk in models
        ]
        for xi, (mk, color, v) in enumerate(zip(models, colors, vals)):
            bar = ax.bar(xi, v, bar_w, color=color, alpha=0.85,
                         hatch=_hatch(mk, multimodal_models), edgecolor="black", linewidth=0.5,
                         label=mk)[0]
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            if judge_key:
                jval = all_judge_metrics.get(mk, {}).get(judge_key)
                if jval is not None:
                    _draw_judge_marker(ax, bar, jval, label=(not judge_label_drawn))
                    judge_label_drawn = True
        if naive_cls_metrics:
            _draw_naive_line(ax, naive_cls_metrics, metric, label=(mi == 0))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Overall Classification", fontsize=11)
    plt.tight_layout()
    path = output_dir / "overall_cls_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _worker_vision(kwargs: dict) -> None:
    """Runs in a subprocess — loads one model, runs inference, saves results, exits."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.common.dataset_config import DatasetConfig

    datasets_cfg = {k: DatasetConfig(**v) for k, v in kwargs["datasets_cfg"].items()}
    desc_rows, cls_rows = run_model(
        model_id=kwargs["model_id"],
        records=kwargs["records"],
        datasets_cfg=datasets_cfg,
        classification_suffix=kwargs["classification_suffix"],
        n_samples=kwargs["n_samples"],
        output_dir=Path(kwargs["output_dir"]),
        quantization=kwargs["quantization"],
        max_num_seqs=kwargs.get("max_num_seqs", 64),
    )
    with open(kwargs["out_path"], "w") as f:
        json.dump({"desc_rows": desc_rows, "cls_rows": cls_rows}, f)


def _run_in_subprocess(fn, kwargs: dict) -> None:
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
    classification_suffix = prompts["classification_suffix"]

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task = {name: ds.task_type for name, ds in datasets_cfg.items()}

    active_set = {name for name, ds in datasets_cfg.items() if not ds.skip}
    records = _apply_healthy_override([
        r for r in load_jsonl(Path(args.descriptions_jsonl), args.n_samples)
        if r.get("metadata", {}).get("agml_dataset") in active_set
    ])
    print(f"Records: {len(records)}  |  Active datasets: {len(active_set)}")

    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics: dict[str, dict[str, dict]] = {}
    summary_metrics: dict[str, dict] = {}

    per_model_quant = _parse_model_quantization(args.model_quantization or [])

    for model_id in args.models:
        quant = per_model_quant.get(model_id, args.quantization)
        print(f"\n{'='*60}")
        print(f"Model: {model_id}" + (f"  [quantization={quant}]" if quant else ""))
        print(f"{'='*60}")

        worker_out = output_dir / f"_worker_{_safe_name(model_id)}.json"
        _run_in_subprocess(_worker_vision, dict(
            model_id=model_id,
            records=records,
            datasets_cfg={k: vars(v) for k, v in datasets_cfg.items()},
            classification_suffix=classification_suffix,
            n_samples=args.n_samples,
            output_dir=str(output_dir),
            quantization=quant,
            max_num_seqs=args.max_num_seqs,
            out_path=str(worker_out),
        ))
        worker_result = json.loads(worker_out.read_text())
        worker_out.unlink()
        desc_rows = worker_result["desc_rows"]
        cls_rows  = worker_result["cls_rows"]

        desc_per_ds = compute_desc_metrics_per_dataset(desc_rows)
        cls_per_ds = compute_cls_metrics_per_dataset(cls_rows)

        desc_overall = aggregate_rouge(desc_rows)
        cls_refs = [r["class_label"] for r in cls_rows]
        cls_preds = [r["resolved_class"] for r in cls_rows]
        cls_overall = compute_classification_metrics(cls_refs, cls_preds)

        desc_by_task = aggregate_by_task(desc_per_ds, ds_to_task)
        cls_by_task = aggregate_by_task(cls_per_ds, ds_to_task)

        all_model_desc_metrics[model_id] = desc_per_ds
        all_model_cls_metrics[model_id] = cls_by_task
        summary_metrics[model_id] = {
            "description_overall": desc_overall,
            "description_per_dataset": desc_per_ds,
            "description_by_task": desc_by_task,
            "classification_overall": cls_overall,
            "classification_per_dataset": cls_per_ds,
            "classification_by_task": cls_by_task,
        }

        write_csv(output_dir / f"desc_{_safe_name(model_id)}.csv", desc_rows)
        write_csv(output_dir / f"cls_{_safe_name(model_id)}.csv", cls_rows)

    # Naive baselines
    naive_desc_per_ds: dict[str, list] = defaultdict(list)
    for record in records:
        agml_name = record.get("metadata", {}).get("agml_dataset", "")
        gt = record["messages"][1]["content"]
        naive_desc_per_ds[agml_name].append({"ground_truth": gt})
    naive_desc_metrics = {
        ds: naive_rouge_baseline(rows) for ds, rows in naive_desc_per_ds.items()
    }

    naive_cls_by_ds: dict[str, list[str]] = defaultdict(list)
    for r in records:
        agml_name = r.get("metadata", {}).get("agml_dataset", "")
        naive_cls_by_ds[agml_name].append(r.get("metadata", {}).get("class_label", ""))
    naive_cls_per_ds = {ds: naive_classification_baseline(refs) for ds, refs in naive_cls_by_ds.items()}
    naive_cls_by_task = aggregate_by_task(naive_cls_per_ds, ds_to_task)

    # LLM judge (classification only)
    if args.judge_model:
        print(f"\n=== LLM Judge: {args.judge_model} ===")
        judge_llm, judge_tok = load_llm(
            args.judge_model,
            quantization=args.judge_quantization,
            gpu_memory_utilization=0.9,
        )

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
                system_prompt=prompts["judge_system_vision"],
                judge_template=prompts["judge_classification_template"],
                gt_key="class_label",
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
            "classification": naive_cls_per_ds,
        },
        "models": summary_metrics,
    }
    save_json(output_dir / "vision_assessment_metrics.json", full_results)

    # Summary output
    tee = Tee(output_dir / "summary.txt")

    tee.print("\n=== Vision Assessment — Description ROUGE ===")
    tee.print(f"  {'Model':<45} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'R-SS':>6}")
    tee.print(f"  {'-'*66}")
    for model_id, m in summary_metrics.items():
        ov = m["description_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('rouge1', 0):>6.3f} {ov.get('rouge2', 0):>6.3f} "
                  f"{ov.get('rougeL', 0):>6.3f} {ov.get('rouge_ss', 0):>6.3f}")

    tee.print("\n=== Vision Assessment — Classification ===")
    tee.print(f"  {'Model':<45} {'Acc':>6} {'Cont':>6} {'F1-mac':>8} {'Res/Tot':>9}")
    tee.print(f"  {'-'*77}")
    for model_id, m in summary_metrics.items():
        ov = m["classification_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('accuracy', 0):>6.3f} "
                  f"{ov.get('contains_accuracy', 0):>6.3f} "
                  f"{ov.get('f1_macro', 0):>8.3f} "
                  f"{ov.get('n_resolvable', 0):>4}/{ov.get('n_total', 0):<4}")

    tee.print("\n=== Vision Assessment — Classification by Task Type ===")
    tee.print(f"  {'Task':<25} ", end="")
    for model_id in args.models:
        tee.print(f"  {model_id[:20]:<20}", end="")
    tee.print()
    for tt in sorted(set(tt for m in all_model_cls_metrics.values() for tt in m)):
        tee.print(f"  {_task_label(tt):<25} ", end="")
        for model_id in args.models:
            f1 = all_model_cls_metrics.get(model_id, {}).get(tt, {}).get("f1_macro", 0.0)
            tee.print(f"  {f1:<20.3f}", end="")
        tee.print()

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
    multimodal_set = set(args.multimodal_models or [])
    all_model_cls_per_ds = {mid: m["classification_per_dataset"] for mid, m in summary_metrics.items()}
    naive_ov_desc = {
        metric: float(np.mean([v[metric] for v in naive_desc_metrics.values() if metric in v]))
        for metric in ["rouge1", "rouge2", "rougeL", "rouge_ss"]
        if any(metric in v for v in naive_desc_metrics.values())
    } if naive_desc_metrics else {}
    naive_cls_all_refs = [r.get("metadata", {}).get("class_label", "") for r in records]
    naive_cls_overall = naive_classification_baseline(naive_cls_all_refs)
    judge_metrics = {mid: m["judge"] for mid, m in summary_metrics.items() if m.get("judge")}

    print("\n=== Generating plots ===")
    plot_cls_summary_per_dataset(all_model_cls_per_ds, output_dir, multimodal_set, naive_cls_overall)
    plot_cls_summary_per_task(all_model_cls_per_ds, ds_to_task, output_dir, multimodal_set, naive_cls_overall, judge_metrics)
    plot_desc_summary_per_task(all_model_desc_metrics, ds_to_task, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_cls_summary(all_model_cls_per_ds, output_dir, multimodal_set, naive_cls_overall, judge_metrics)

    # Update pipeline results CSV
    pipeline_csv = output_dir.parent / "pipeline_results.csv"
    for model_id, m in summary_metrics.items():
        update_pipeline_results(
            csv_path=pipeline_csv,
            model=model_id,
            model_type="vlm",
            stage="vision_assessment",
            metrics={
                "description_rouge_ss": m["description_overall"].get("rouge_ss", ""),
                "classification_f1_macro": m["classification_overall"].get("f1_macro", ""),
            },
        )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision assessment: multimodal VQA evaluation on AgML datasets."
    )
    parser.add_argument(
        "--descriptions_jsonl", default="outputs/data_generation/agml_descriptions.jsonl",
        help="Path to agml_descriptions.jsonl",
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
        help="Space-separated list of VLM IDs to evaluate",
    )
    parser.add_argument(
        "--multimodal_models", nargs="*", default=[],
        help="Subset of --models that are VLMs — shown with diagonal hatching in charts",
    )
    parser.add_argument(
        "--judge_model", default=None,
        help="LLM-as-a-judge model (runs after all eval models; classification only)",
    )
    parser.add_argument(
        "--judge_quantization", default=None, choices=["awq", "gptq", "fp8", "bitsandbytes"],
        help="Quantization method for the judge model (optional)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/vision_assessment/",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Max images per class to evaluate (default: all)",
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
        help="vLLM quantization method applied to all models (overridden per model by --model_quantization)",
    )
    parser.add_argument(
        "--model_quantization", nargs="*", default=[],
        metavar="MODEL_ID:METHOD",
        help="Per-model quantization overrides, e.g. meta-llama/Meta-Llama-3-70B-Instruct:bitsandbytes",
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=64,
        help="Max concurrent sequences for vLLM (lower to ≤35 for Mamba-architecture models)",
    )
    args = parser.parse_args()
    main(args)
