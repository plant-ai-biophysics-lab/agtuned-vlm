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

from src.common.dataset_config import datasets_by_name, prefetch_agml_datasets
from src.common.distribution_plots import (
    gather_cls_per_class_by_dataset,
    gather_cls_values_by_dataset,
    gather_cls_values_by_task,
    gather_desc_per_sample_by_dataset,
    gather_desc_values_by_dataset,
    gather_desc_values_by_task,
    gather_judge_values_by_dataset,
    gather_judge_values_by_task,
    plot_overall_distribution,
    plot_per_dataset_distribution,
    plot_per_task_distribution,
)
from src.common.io import (
    Tee,
    load_jsonl,
    load_prompts,
    read_csv,
    save_json,
    update_pipeline_results,
    write_csv,
)
from src.common.metrics import (
    THINK_RE,
    aggregate_by_task,
    aggregate_rouge,
    apply_judge,
    compute_classification_metrics,
    contains_match,
    judged_summary,
    model_color_map,
    naive_classification_baseline,
    naive_rouge_baseline,
    naive_uniform_baseline_per_dataset,
    resolve_class,
    rouge_scores_per_sample,
    sample_class_images,
)

DESC_PARAMS = SamplingParams(temperature=0.0, max_tokens=384, repetition_penalty=1.1)
CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)

_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*{1,3}([^*]*)\*{1,3}|^#{1,6}\s*|^[-*+]\s+", re.MULTILINE)
_REASONING_RE = re.compile(
    r"\b(we need to|the user wants|the instruction|let me|i need to|i should|i will|"
    r"to answer this|the question (is|asks)|so (just|the answer)|"
    r"the answer (is|would be|should)|first[,\s]+(let|i)|step \d)\b",
    re.IGNORECASE,
)
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
    # Strip complete think/thought blocks, then discard any unclosed tag onward
    text = THINK_RE.sub("", text).strip()
    for tag in ("<think>", "<thought>"):
        idx = text.lower().find(tag)
        if idx != -1:
            text = text[:idx].strip()

    # Strip markdown (headers, bold/italic, bullet markers) — keep inner text
    text = _MARKDOWN_RE.sub(r"\1", text)

    # Remove greeting-only lines at the start
    lines = text.splitlines()
    while lines and _GREETING_RE.match(lines[0].strip()):
        lines.pop(0)
    text = "\n".join(lines)

    # Take first non-reasoning paragraph; fall back to last paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    text = ""
    for para in paragraphs:
        collapsed = " ".join(para.split())
        if not _REASONING_RE.search(collapsed):
            text = collapsed
            break
    if not text and paragraphs:
        # Only fall back if the last paragraph doesn't look like pure reasoning either
        last = " ".join(paragraphs[-1].split())
        if not _REASONING_RE.search(last):
            text = last

    # Collapse internal whitespace / newlines within the paragraph
    text = " ".join(text.split())

    # Cut repetition loops
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()

    return text.strip()


def _apply_chat_template(tokenizer, prompt: str, system_prompt: str = "") -> str:
    messages = [{"role": "user", "content": prompt}]
    sys_msg = system_prompt or ""
    try:
        full = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + messages
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        # Qwen fallback: merge system prompt with /no_think
        sys_content = " ".join(filter(None, [sys_msg, "/no_think"]))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_content}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        full = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + messages
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=True
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
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    enforce_eager: bool = False,
) -> tuple[LLM, object]:
    if _is_mamba(model_id):
        effective_seqs = min(max_num_seqs, _MAMBA_MAX_NUM_SEQS)
        if effective_seqs < max_num_seqs:
            print(f"  [load_llm] Mamba architecture detected — capping max_num_seqs at {effective_seqs}")
        max_num_seqs = effective_seqs

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        quantization=quantization,
        enforce_eager=enforce_eager,
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


def build_multimodal_prompt(tokenizer, image: PILImage.Image, question: str, system_prompt: str = "") -> dict:
    sys_msg = system_prompt or ""
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [user_msg]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
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
    system_prompt: str = "",
) -> list[str]:
    prompts = [build_multimodal_prompt(tokenizer, img, question, system_prompt=system_prompt) for img in images]
    outputs = llm.generate(prompts, sampling_params=params)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def build_name_map(agml_name: str, images_dir: Path | None = None) -> dict[str, str]:
    """Returns {filename: full_path} for all images in a dataset.
    For digital_green (or any dataset with images_dir set), scans the local directory
    instead of using the AgML data loader.
    """
    if images_dir is not None:
        return {p.name: str(p) for p in Path(images_dir).iterdir() if p.is_file()}
    import agml
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    return {Path(str(p)).name: str(p) for p in contents.keys()}


def run_model(
    model_id: str,
    records: list[dict],
    datasets_cfg: dict,
    classification_suffix: str,
    desc_prompt_template: str,
    cls_prompt_template: str,
    n_samples: int | None,
    output_dir: Path,
    quantization: str | None = None,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    enforce_eager: bool = False,
    images_dir: Path | None = None,
    system_prompt: str = "",
) -> tuple[list[dict], list[dict]]:
    """Load model, run both tasks on all records, unload, return (desc_rows, cls_rows)."""
    llm, tokenizer = load_llm(
        model_id, quantization=quantization, max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    from tqdm import tqdm

    desc_rows: list[dict] = []
    cls_rows: list[dict] = []
    agml_cache: dict[str, dict[str, str]] = {}

    t_start = time.perf_counter()

    pbar = tqdm(records, desc="  Classes", unit="class", dynamic_ncols=True)
    for record in pbar:
        meta = record["metadata"]
        agml_name = meta["agml_dataset"]
        class_label = meta["class_label"]
        ds_cfg_eff = datasets_cfg.get(agml_name)
        task_type = ds_cfg_eff.effective_task_type(class_label) if ds_cfg_eff else meta["task_type"]
        crop = meta["crop"]
        image_names = meta["image_names"]
        synth_desc = record["messages"][1]["content"]
        visual_desc_q = meta["visual_question"]
        visual_cls_q = meta.get("visual_cls_question")

        image_names = sample_class_images(image_names, n_samples)

        pbar.set_postfix_str(f"{agml_name} / {class_label}", refresh=False)

        if agml_name not in agml_cache:
            tqdm.write(f"\n  Building image map: {agml_name}...")
            dg_dir = images_dir if agml_name == "digital_green" else None
            agml_cache[agml_name] = build_name_map(agml_name, images_dir=dg_dir)
        name_map = agml_cache[agml_name]

        images, valid_names = [], []
        for name in image_names:
            if name in name_map:
                images.append(PILImage.open(name_map[name]).convert("RGB"))
                valid_names.append(name)
            else:
                tqdm.write(f"    [warn] {name} not in {agml_name}")
        if not images:
            continue

        tqdm.write(f"  {agml_name} / {class_label}  ({len(images)} images)")

        # Description task
        wrapped_desc_q = desc_prompt_template.format(question=visual_desc_q)
        desc_preds = run_batch(llm, tokenizer, images, wrapped_desc_q, DESC_PARAMS, system_prompt=system_prompt)
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
            wrapped_cls_q = cls_prompt_template.format(question=visual_cls_q)

            cls_preds = run_batch(llm, tokenizer, images, wrapped_cls_q, CLS_PARAMS, system_prompt=system_prompt)
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


def compute_cls_metrics_per_task(cls_rows: list[dict]) -> dict[str, dict]:
    """Returns {task_type: classification_metrics} grouped by class-level task_type."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in cls_rows:
        tt = row.get("task_type", "")
        if tt:
            by_task[tt].append(row)
    result = {}
    for tt, rows in by_task.items():
        refs = [r["class_label"] for r in rows]
        preds = [r["resolved_class"] for r in rows]
        result[tt] = compute_classification_metrics(refs, preds)
    return result


def compute_desc_metrics_per_task(desc_rows: list[dict]) -> dict[str, dict]:
    """Returns {task_type: rouge_summary} grouped by class-level task_type."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in desc_rows:
        tt = row.get("task_type", "")
        if tt:
            by_task[tt].append(row)
    return {tt: aggregate_rouge(rows) for tt, rows in by_task.items()}


# ── Plots ──────────────────────────────────────────────────────────────────────

def _hatch(model_key: str, multimodal_models: set[str]) -> str:
    return "///" if model_key in multimodal_models else ""


def _draw_naive_line(
    ax,
    naive_overall: dict | None,
    metric: str,
    label: bool = False,
    uniform_overall: dict | None = None,
) -> None:
    if naive_overall:
        value = naive_overall.get(metric)
        if value is not None:
            ax.axhline(value, color="red", linestyle="--", linewidth=1.2,
                       label="naive (shuffled)" if label else "_nolegend_")
    if uniform_overall:
        value = uniform_overall.get(metric)
        if value is not None:
            ax.axhline(value, color="red", linestyle="-.", linewidth=1.2,
                       label="naive (1/k classes)" if label else "_nolegend_")


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
    uniform_cls_metrics: dict | None = None,
    all_judge_by_dataset: dict[str, dict] | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    all_judge_by_dataset = all_judge_by_dataset or {}
    all_datasets = []
    for model_results in all_cls_results.values():
        for ds in model_results:
            if ds not in all_datasets:
                all_datasets.append(ds)
    if not all_datasets:
        return

    models = list(all_cls_results.keys())
    color_map = model_color_map(models)
    x = np.arange(len(all_datasets))
    width = 0.8 / len(models)

    judge_label_drawn = False
    fig, ax = plt.subplots(figsize=(max(10, len(all_datasets) * len(models) * 0.5), 5))
    for i, model_key in enumerate(models):
        per_ds = all_cls_results[model_key]
        offset = (i - (len(models) - 1) / 2) * width
        color  = color_map[model_key]
        hatch  = _hatch(model_key, multimodal_models)
        has_judge = bool(all_judge_by_dataset.get(model_key))
        sub_w = width * 0.46 if has_judge else width
        for xi, ds in enumerate(all_datasets):
            val = per_ds.get(ds, {}).get("f1_macro", 0.0)
            lbl = model_key.split("/")[-1] if xi == 0 else "_nolegend_"
            ax.bar(xi + offset - (sub_w / 2 if has_judge else 0), val, sub_w,
                   color=color, alpha=0.85, hatch=hatch,
                   edgecolor="black", linewidth=0.5, label=lbl)
            if has_judge:
                jval = all_judge_by_dataset.get(model_key, {}).get(ds, {}).get("judge_f1_macro")
                if jval is not None:
                    jlbl = "judge" if not judge_label_drawn else "_nolegend_"
                    ax.bar(xi + offset + sub_w / 2, jval, sub_w,
                           color=color, alpha=0.4, edgecolor="black", linewidth=0.5, label=jlbl)
                    judge_label_drawn = True
    _draw_naive_line(ax, None, "f1_macro", label=True, uniform_overall=uniform_cls_metrics)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 Macro")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Classification Per Dataset", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_desc_sem_sim_per_dataset(
    all_desc_results: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_desc_metrics: dict | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    all_datasets: list[str] = []
    for model_results in all_desc_results.values():
        for ds in model_results:
            if ds not in all_datasets:
                all_datasets.append(ds)
    if not all_datasets:
        return

    models = list(all_desc_results.keys())
    color_map = model_color_map(models)
    x = np.arange(len(all_datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(10, len(all_datasets) * len(models) * 0.5), 5))
    for i, model_key in enumerate(models):
        per_ds = all_desc_results[model_key]
        vals = [per_ds.get(ds, {}).get("sem_sim", 0.0) for ds in all_datasets]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model_key.split("/")[-1],
               color=color_map[model_key], alpha=0.85,
               hatch=_hatch(model_key, multimodal_models), edgecolor="black", linewidth=0.5)
    _draw_naive_line(ax, naive_desc_metrics, "sem_sim", label=True)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylim(0, 1.15)
    ax.set_title("Sem Sim")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Description Sem Sim Per Dataset", fontsize=11)
    plt.tight_layout()
    path = output_dir / "desc_sem_sim_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cls_summary_per_task(
    all_cls_by_task: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_cls_metrics: dict | None = None,
    all_judge_metrics: dict[str, dict] | None = None,
    uniform_cls_metrics: dict | None = None,
    all_judge_by_task: dict[str, dict] | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    all_judge_metrics = all_judge_metrics or {}
    task_types = sorted(set(tt for per_task in all_cls_by_task.values() for tt in per_task))
    if not task_types:
        return

    models = list(all_cls_by_task.keys())
    metrics = ["accuracy", "f1_macro"]
    color_map = model_color_map(models)
    x = np.arange(len(task_types))
    width = 0.8 / len(models)

    subplot_w = max(4, len(task_types) * (len(models) * 0.6 + 0.4))
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=False)
    judge_label_drawn = False
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        judge_key = _JUDGE_METRIC_MAP.get(metric)
        has_judge_any = bool(judge_key and all_judge_by_task)
        sub_w = width * 0.46 if has_judge_any else width
        for i, model_key in enumerate(models):
            per_task = all_cls_by_task.get(model_key, {})
            offset = (i - (len(models) - 1) / 2) * width
            color  = color_map[model_key]
            hatch  = _hatch(model_key, multimodal_models)
            for ti, tt in enumerate(task_types):
                val = per_task.get(tt, {}).get(metric, 0.0)
                lbl = model_key if ti == 0 else "_nolegend_"
                bx = ti + offset - (sub_w / 2 if has_judge_any else 0)
                ax.bar(bx, val, sub_w, color=color, alpha=0.85, hatch=hatch,
                       edgecolor="black", linewidth=0.5, label=lbl)
                ax.text(bx + sub_w / 2, val + 0.005, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=6)
                if has_judge_any:
                    jval = all_judge_by_task.get(model_key, {}).get(tt, {}).get(judge_key)
                    if jval is not None:
                        jlbl = "judge" if not judge_label_drawn else "_nolegend_"
                        ax.bar(ti + offset + sub_w / 2, jval, sub_w,
                               color=color, alpha=0.4, edgecolor="black", linewidth=0.5, label=jlbl)
                        judge_label_drawn = True
        _draw_naive_line(ax, None, metric, label=(mi == 0), uniform_overall=uniform_cls_metrics)
        ax.set_xticks(np.arange(len(task_types)))
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Classification Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_desc_summary_per_task(
    all_desc_by_task: dict[str, dict[str, dict]],
    output_dir: Path,
    multimodal_models: set[str] | None = None,
    naive_desc_metrics: dict | None = None,
) -> None:
    multimodal_models = multimodal_models or set()
    task_types = sorted(set(tt for per_task in all_desc_by_task.values() for tt in per_task))
    if not task_types:
        return

    models = list(all_desc_by_task.keys())
    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim"]
    color_map = model_color_map(models)
    x = np.arange(len(task_types))
    width = 0.8 / len(models)

    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS", "sem_sim": "Sem Sim"}
    subplot_w = max(4, len(task_types) * (len(models) * 0.6 + 0.4))
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=False)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, model_key in enumerate(models):
            per_task = all_desc_by_task.get(model_key, {})
            vals = [per_task.get(tt, {}).get(metric, 0.0) for tt in task_types]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color_map[model_key], alpha=0.85,
                          hatch=_hatch(model_key, multimodal_models), edgecolor="black", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        _draw_naive_line(ax, naive_desc_metrics, metric, label=True)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score")
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

    metrics = ["rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim"]
    metric_labels = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rouge_ss": "ROUGE-SS", "sem_sim": "Sem Sim"}
    color_map = model_color_map(models)
    x = np.arange(len(models))
    bar_w = 0.7
    subplot_w = max(3.5, len(models) * 1.5)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=False)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_desc_results[mk].values()]))
            if all_desc_results[mk] else 0.0
            for mk in models
        ]
        for xi, (mk, v) in enumerate(zip(models, vals)):
            ax.bar(xi, v, bar_w, color=color_map[mk], alpha=0.85,
                   hatch=_hatch(mk, multimodal_models), edgecolor="black", linewidth=0.5,
                   label=mk)
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _draw_naive_line(ax, naive_desc_metrics, metric, label=True)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Score")
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
    uniform_cls_metrics: dict | None = None,
    filename: str = "overall_cls_summary.png",
) -> None:
    multimodal_models = multimodal_models or set()
    all_judge_metrics = all_judge_metrics or {}
    models = list(all_cls_results.keys())
    if not models:
        return

    metrics = ["accuracy", "f1_macro"]
    color_map = model_color_map(models)
    x = np.arange(len(models))
    bar_w = 0.7
    subplot_w = max(3.5, len(models) * 1.5)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(subplot_w * len(metrics), 6), sharey=False)
    judge_label_drawn = False
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        judge_key = _JUDGE_METRIC_MAP.get(metric)
        has_judge_any = bool(judge_key and all_judge_metrics)
        sub_w = bar_w * 0.46 if has_judge_any else bar_w
        vals = [
            float(np.mean([m.get(metric, 0.0) for m in all_cls_results[mk].values()]))
            if all_cls_results[mk] else 0.0
            for mk in models
        ]
        for xi, (mk, v) in enumerate(zip(models, vals)):
            bx = xi - (sub_w / 2 if has_judge_any else 0)
            ax.bar(bx, v, sub_w, color=color_map[mk], alpha=0.85,
                   hatch=_hatch(mk, multimodal_models), edgecolor="black", linewidth=0.5, label=mk)
            ax.text(bx + sub_w / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            if has_judge_any:
                jval = all_judge_metrics.get(mk, {}).get(judge_key)
                if jval is not None:
                    jlbl = "judge" if not judge_label_drawn else "_nolegend_"
                    ax.bar(xi + sub_w / 2, jval, sub_w,
                           color=color_map[mk], alpha=0.4, edgecolor="black", linewidth=0.5, label=jlbl)
                    judge_label_drawn = True
        _draw_naive_line(ax, None, metric, label=(mi == 0), uniform_overall=uniform_cls_metrics)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [mk.split("/")[-1] for mk in models], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Vision Assessment — Overall Classification", fontsize=11)
    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metric_correlation(
    all_model_desc_rows: dict[str, list[dict]],
    all_model_cls_rows: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Per-sample desc metric scatter plots and per-class desc quality vs classification F1."""
    from sklearn.metrics import f1_score as sk_f1
    from collections import defaultdict as _dd

    corr_dir = output_dir / "metric_correlations"
    corr_dir.mkdir(exist_ok=True)

    for model_id, desc_rows in all_model_desc_rows.items():
        cls_rows = all_model_cls_rows.get(model_id, [])
        safe = _safe_name(model_id)

        # ── 1. Per-sample description metric correlations ──
        pairs = [
            ("rouge_ss", "sem_sim",  "ROUGE-SS", "Sem Sim"),
            ("rougeL",   "sem_sim",  "ROUGE-L",  "Sem Sim"),
            ("rouge_ss", "rougeL",   "ROUGE-SS", "ROUGE-L"),
        ]
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
        for ax, (xk, yk, xl, yl) in zip(axes, pairs):
            xs = [float(r.get(xk, 0) or 0) for r in desc_rows]
            ys = [float(r.get(yk, 0) or 0) for r in desc_rows]
            if not xs:
                continue
            r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else 0.0
            ax.scatter(xs, ys, alpha=0.3, s=10)
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, m * x_line + b, color="red", linewidth=1.2)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{xl} vs {yl}\nr = {r:.3f}")
        fig.suptitle(f"Per-Sample Metric Correlation — {model_id.split('/')[-1]}", fontsize=10)
        plt.tight_layout()
        fig.savefig(corr_dir / f"per_sample_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {corr_dir / f'per_sample_{safe}.png'}")

        # ── 2. Per-class: desc quality vs classification F1 ──
        if not cls_rows:
            continue
        refs  = [r.get("class_label", "") for r in cls_rows]
        preds = [r.get("resolved_class") or "" for r in cls_rows]
        classes = sorted(set(refs))
        try:
            per_class_f1 = dict(zip(
                classes,
                sk_f1(refs, preds, average=None, labels=classes, zero_division=0),
            ))
        except Exception:
            continue

        cls_desc: dict = _dd(lambda: {"rouge_ss": [], "sem_sim": [], "rougeL": []})
        for row in desc_rows:
            lbl = row.get("class_label", "")
            for k in ("rouge_ss", "sem_sim", "rougeL"):
                v = row.get(k)
                if v not in (None, ""):
                    try:
                        cls_desc[lbl][k].append(float(v))
                    except (TypeError, ValueError):
                        pass

        pairs_cross = [
            ("rouge_ss", "ROUGE-SS"),
            ("sem_sim",  "Sem Sim"),
            ("rougeL",   "ROUGE-L"),
        ]
        fig, axes = plt.subplots(1, len(pairs_cross), figsize=(5 * len(pairs_cross), 4))
        for ax, (dk, dl) in zip(axes, pairs_cross):
            xs, ys = [], []
            for cls in classes:
                vals = cls_desc[cls][dk]
                if vals and cls in per_class_f1:
                    xs.append(float(np.mean(vals)))
                    ys.append(per_class_f1[cls])
            if not xs:
                continue
            r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else 0.0
            ax.scatter(xs, ys, alpha=0.6, s=20)
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, m * x_line + b, color="red", linewidth=1.2)
            ax.set_xlabel(f"Mean {dl} (description)"); ax.set_ylabel("F1 (classification)")
            ax.set_title(f"{dl} vs F1\nr = {r:.3f}")
        fig.suptitle(f"Per-Class Desc Quality vs Classification F1 — {model_id.split('/')[-1]}", fontsize=10)
        plt.tight_layout()
        fig.savefig(corr_dir / f"per_class_f1_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {corr_dir / f'per_class_f1_{safe}.png'}")


def plot_difficulty_correlation(
    all_cls_results: dict[str, dict[str, dict]],
    all_desc_results: dict[str, dict[str, dict]],
    clustering_metrics_path: Path,
    output_dir: Path,
) -> None:
    """Scatter: per-dataset class separability (difficulty) vs assessment F1/sem_sim."""
    if not clustering_metrics_path.exists():
        print(f"  Skipping difficulty correlation — not found: {clustering_metrics_path}")
        return

    with open(clustering_metrics_path) as f:
        clust: dict[str, dict[str, dict]] = json.load(f)

    diff_dir = output_dir / "difficulty_correlation"
    diff_dir.mkdir(parents=True, exist_ok=True)

    diff_metrics = [
        ("silhouette",        "Silhouette"),
        ("inter_centroid_dist", "Inter-Centroid Dist"),
        ("fisher_dr",         "Fisher DR"),
    ]
    perf_metrics = [("f1_macro", "F1 Macro"), ("sem_sim", "Sem Sim")]

    # mean performance across all models per dataset
    all_datasets = sorted(set(
        ds for m in list(all_cls_results.values()) + list(all_desc_results.values())
        for ds in m
    ))

    def _mean_perf(results: dict[str, dict[str, dict]], metric: str, ds: str) -> float | None:
        vals = [m[ds][metric] for m in results.values() if ds in m and metric in m[ds]]
        return float(np.mean(vals)) if vals else None

    for backbone, ds_clust in clust.items():
        short = backbone.split("/")[-1]
        n_diff = len(diff_metrics)
        n_perf = len(perf_metrics)
        fig, axes = plt.subplots(n_perf, n_diff,
                                 figsize=(4 * n_diff, 4 * n_perf), squeeze=False)

        for pi, (pm, pl) in enumerate(perf_metrics):
            results_src = all_cls_results if pm == "f1_macro" else all_desc_results
            for di, (dm, dl) in enumerate(diff_metrics):
                ax = axes[pi][di]
                xs, ys, ds_names = [], [], []
                for ds in all_datasets:
                    diff_val = ds_clust.get(ds, {}).get(dm)
                    perf_val = _mean_perf(results_src, pm, ds)
                    if diff_val is not None and perf_val is not None:
                        xs.append(diff_val)
                        ys.append(perf_val)
                        ds_names.append(ds.replace("_classification", "").replace("_", " "))
                if len(xs) < 2:
                    ax.set_visible(False)
                    continue
                r = float(np.corrcoef(xs, ys)[0, 1])
                ax.scatter(xs, ys, alpha=0.75, s=40, edgecolors="black", linewidths=0.4)
                for x, y, name in zip(xs, ys, ds_names):
                    ax.annotate(name, (x, y), fontsize=5, alpha=0.7,
                                xytext=(3, 3), textcoords="offset points")
                m_, b_ = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line, m_ * x_line + b_, color="red", linewidth=1.2)
                ax.set_xlabel(f"{dl} (↑ easier)"); ax.set_ylabel(pl)
                ax.set_title(f"{dl} vs {pl}\nr = {r:.3f}", fontsize=9)

        fig.suptitle(f"Dataset Difficulty vs Performance — {short}", fontsize=11)
        plt.tight_layout()
        safe = short.replace("/", "_")
        path = diff_dir / f"difficulty_vs_perf_{safe}.png"
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
        desc_prompt_template=kwargs["desc_prompt_template"],
        cls_prompt_template=kwargs["cls_prompt_template"],
        n_samples=kwargs["n_samples"],
        output_dir=Path(kwargs["output_dir"]),
        quantization=kwargs["quantization"],
        max_num_seqs=kwargs.get("max_num_seqs", 64),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        max_model_len=kwargs.get("max_model_len", 8192),
        enforce_eager=kwargs.get("enforce_eager", False),
        images_dir=Path(kwargs["images_dir"]) if kwargs.get("images_dir") else None,
        system_prompt=kwargs.get("system_prompt", ""),
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
    desc_prompt_template = prompts["assessment_descriptive"]
    cls_prompt_template = prompts["assessment_classification"]
    assessment_system = prompts.get("assessment_system", "")

    use_dg = getattr(args, "dataset_source", "agml") == "digital_green"

    if use_dg:
        dg_config = getattr(args, "dg_config", None) or "outputs/data_generation/digital_green_datasets.yaml"
        dg_jsonl  = getattr(args, "dg_descriptions_jsonl", None) or "outputs/data_generation/dg_descriptions.jsonl"
        images_dir_val = getattr(args, "images_dir", None)
        datasets_cfg = datasets_by_name(Path(dg_config))
        active_set = {name for name, ds in datasets_cfg.items() if not ds.skip}
        records = _apply_healthy_override(list(load_jsonl(Path(dg_jsonl), args.n_samples)))
    else:
        images_dir_val = None
        datasets_cfg = datasets_by_name(Path(args.config))
        active_set = {name for name, ds in datasets_cfg.items() if not ds.skip}
        records = _apply_healthy_override([
            r for r in load_jsonl(Path(args.descriptions_jsonl), args.n_samples)
            if r.get("metadata", {}).get("agml_dataset") in active_set
        ])

    ds_to_task = {name: ds.task_type for name, ds in datasets_cfg.items()}

    if not use_dg:
        prefetch_agml_datasets([ds for ds in datasets_cfg.values() if not ds.skip])

    # Override per-class image_names with linear probe test split if provided
    test_split_path = getattr(args, "test_split", None)
    if test_split_path and not Path(test_split_path).exists():
        print(f"[warn] test_split not found at {test_split_path} — using all available images per class")
        test_split_path = None
    if test_split_path:
        test_split = json.loads(Path(test_split_path).read_text())
        overridden, skipped = 0, 0
        filtered = []
        for r in records:
            meta = r.get("metadata", {})
            ds = meta.get("agml_dataset", "")
            cls = meta.get("class_label", "")
            fnames = test_split.get(ds, {}).get(cls)
            if fnames:
                r = {**r, "metadata": {**meta, "image_names": fnames}}
                filtered.append(r)
                overridden += 1
            else:
                skipped += 1
        records = filtered
        print(f"Test split loaded from {test_split_path}: {overridden} classes overridden, {skipped} skipped (not in split)")

    print(f"Records: {len(records)}  |  Active datasets: {len(active_set)}")

    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics: dict[str, dict[str, dict]] = {}
    all_model_desc_rows: dict[str, list[dict]] = {}
    all_model_cls_rows: dict[str, list[dict]] = {}
    summary_metrics: dict[str, dict] = {}

    per_model_quant = _parse_model_quantization(args.model_quantization or [])
    n_models = len(args.models)

    for model_idx, model_id in enumerate(args.models):
        quant = per_model_quant.get(model_id, args.quantization)
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{n_models}: {model_id}" + (f"  [quantization={quant}]" if quant else ""))
        print(f"{'='*60}")

        cls_csv  = output_dir / f"cls_{_safe_name(model_id)}.csv"
        desc_csv = output_dir / f"desc_{_safe_name(model_id)}.csv"

        if args.skip_existing and cls_csv.exists() and desc_csv.exists():
            print(f"  [skip] loading existing CSVs for {model_id}")
            cls_rows  = read_csv(cls_csv)
            desc_rows = read_csv(desc_csv)
        else:
            worker_out = output_dir / f"_worker_{_safe_name(model_id)}.json"
            _run_in_subprocess(_worker_vision, dict(
                model_id=model_id,
                records=records,
                datasets_cfg={k: vars(v) for k, v in datasets_cfg.items()},
                classification_suffix=classification_suffix,
                desc_prompt_template=desc_prompt_template,
                cls_prompt_template=cls_prompt_template,
                n_samples=args.n_samples,
                output_dir=str(output_dir),
                quantization=quant,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                enforce_eager=args.enforce_eager,
                images_dir=str(images_dir_val) if images_dir_val else None,
                system_prompt=assessment_system,
                out_path=str(worker_out),
            ))
            worker_result = json.loads(worker_out.read_text())
            worker_out.unlink()
            desc_rows = worker_result["desc_rows"]
            cls_rows  = worker_result["cls_rows"]
            write_csv(cls_csv,  cls_rows)
            write_csv(desc_csv, desc_rows)

        desc_per_ds   = compute_desc_metrics_per_dataset(desc_rows)
        cls_per_ds    = compute_cls_metrics_per_dataset(cls_rows)
        desc_per_task = compute_desc_metrics_per_task(desc_rows)
        cls_per_task  = compute_cls_metrics_per_task(cls_rows)

        desc_overall = aggregate_rouge(desc_rows)
        cls_refs = [r["class_label"] for r in cls_rows]
        cls_preds = [r["resolved_class"] for r in cls_rows]
        cls_overall = compute_classification_metrics(cls_refs, cls_preds)

        all_model_desc_metrics[model_id] = desc_per_ds
        all_model_cls_metrics[model_id]  = cls_per_task
        all_model_desc_rows[model_id]    = desc_rows
        all_model_cls_rows[model_id]     = cls_rows
        summary_metrics[model_id] = {
            "description_overall": desc_overall,
            "description_per_dataset": desc_per_ds,
            "description_per_task": desc_per_task,
            "classification_overall": cls_overall,
            "classification_per_dataset": cls_per_ds,
            "classification_per_task": cls_per_task,
        }

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
            cls_rows = read_csv(cls_csv)
            already_judged = bool(cls_rows and cls_rows[0].get("judged_correct", "") != "")
            if args.skip_existing and already_judged:
                print(f"  [skip] already judged: {model_id}")
            else:
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
            by_task_judge: dict[str, dict] = {}
            for tt in sorted(set(r.get("task_type", "") for r in cls_rows)):
                task_rows = [r for r in cls_rows if r.get("task_type") == tt]
                by_task_judge[tt] = judged_summary(task_rows)
            summary_metrics[model_id]["judge_by_task"] = by_task_judge
            by_dataset_judge: dict[str, dict] = {}
            for ds in sorted(set(r.get("agml_dataset", "") for r in cls_rows)):
                ds_rows = [r for r in cls_rows if r.get("agml_dataset") == ds]
                by_dataset_judge[ds] = judged_summary(ds_rows)
            summary_metrics[model_id]["judge_by_dataset"] = by_dataset_judge

        del judge_llm, judge_tok
        gc.collect()

    # Save metrics JSON
    full_results = {
        "naive": {
            "description": naive_desc_metrics,
            "description_overall": naive_ov_desc,
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
        for metric in ["rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim"]
        if any(metric in v for v in naive_desc_metrics.values())
    } if naive_desc_metrics else {}
    naive_cls_all_refs = [r.get("metadata", {}).get("class_label", "") for r in records]
    naive_cls_overall = naive_classification_baseline(naive_cls_all_refs)
    uniform_cls_overall = naive_uniform_baseline_per_dataset([
        {"agml_dataset": r.get("metadata", {}).get("agml_dataset", ""),
         "class_label":  r.get("metadata", {}).get("class_label", "")}
        for r in records
    ])
    judge_metrics = {mid: m["judge"] for mid, m in summary_metrics.items() if m.get("judge")}
    judge_by_task = {mid: m["judge_by_task"] for mid, m in summary_metrics.items() if m.get("judge_by_task")}
    judge_by_dataset = {mid: m["judge_by_dataset"] for mid, m in summary_metrics.items() if m.get("judge_by_dataset")}

    print("\n=== Generating plots ===")
    plot_cls_summary_per_dataset(all_model_cls_per_ds, output_dir, multimodal_set, naive_cls_overall, uniform_cls_overall, judge_by_dataset)
    plot_cls_summary_per_task(all_model_cls_metrics, output_dir, multimodal_set, naive_cls_overall, judge_metrics, uniform_cls_overall, judge_by_task)
    plot_desc_summary_per_task({mid: m.get("description_per_task", {}) for mid, m in summary_metrics.items()}, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_desc_summary(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc)
    plot_desc_sem_sim_per_dataset(all_model_desc_metrics, output_dir, multimodal_set, naive_ov_desc)
    plot_overall_cls_summary(all_model_cls_per_ds, output_dir, multimodal_set, naive_cls_overall, judge_metrics, uniform_cls_overall)
    plot_metric_correlation(all_model_desc_rows, all_model_cls_rows, output_dir)
    plot_difficulty_correlation(all_model_cls_per_ds, all_model_desc_metrics,
                                Path(args.clustering_metrics), output_dir)

    # ── Distribution charts (strip + errorbar) ──────────────────────────────
    models        = list(all_model_cls_per_ds.keys())
    color_map     = model_color_map(models)
    all_datasets  = sorted(set(ds for m in all_model_cls_per_ds.values() for ds in m))
    task_types    = sorted(set(ds_to_task.get(ds, "") for ds in all_datasets) - {""})

    cls_by_ds    = gather_cls_values_by_dataset(all_model_cls_per_ds)
    cls_by_task  = gather_cls_values_by_task(all_model_cls_per_ds, ds_to_task)
    cls_by_cls   = gather_cls_per_class_by_dataset(all_model_cls_rows, "class_label", "resolved_class")
    desc_by_ds   = gather_desc_values_by_dataset(all_model_desc_metrics)
    desc_by_task = gather_desc_values_by_task(all_model_desc_metrics, ds_to_task)
    desc_by_smp  = gather_desc_per_sample_by_dataset(all_model_desc_rows)
    judge_by_ds_dist   = gather_judge_values_by_dataset(judge_by_dataset)
    judge_by_task_dist = gather_judge_values_by_task(judge_by_dataset, ds_to_task)

    uniform_f1   = uniform_cls_overall.get("f1_macro") if uniform_cls_overall else None
    naive_sem_sim = naive_ov_desc.get("sem_sim") if naive_ov_desc else None

    for chart_type, suffix in [("strip", "strip"), ("errorbar", "errorbar")]:
        plot_overall_distribution(
            cls_by_ds, models, color_map, multimodal_set, "F1 Macro",
            "Vision Assessment — Classification Overall", output_dir / f"cls_overall_{suffix}.png",
            chart_type=chart_type, naive_line=uniform_f1,
            judge_values_by_model=judge_by_ds_dist)
        plot_per_task_distribution(
            cls_by_task, task_types, models, color_map, multimodal_set, _task_label,
            "F1 Macro", "Vision Assessment — Classification Per Task",
            output_dir / f"cls_per_task_{suffix}.png", chart_type=chart_type, naive_line=uniform_f1,
            judge_values_by_model_by_task=judge_by_task_dist)
        plot_per_dataset_distribution(
            cls_by_cls, all_datasets, models, color_map, multimodal_set,
            "F1 per Class", "Vision Assessment — Classification Per Dataset (per-class spread)",
            output_dir / f"cls_per_dataset_{suffix}.png", chart_type=chart_type,
            naive_line=uniform_f1, ds_to_task=ds_to_task)
        plot_overall_distribution(
            desc_by_ds, models, color_map, multimodal_set, "Sem Sim",
            "Vision Assessment — Description Overall", output_dir / f"desc_overall_{suffix}.png",
            chart_type=chart_type, naive_line=naive_sem_sim)
        plot_per_task_distribution(
            desc_by_task, task_types, models, color_map, multimodal_set, _task_label,
            "Sem Sim", "Vision Assessment — Description Per Task",
            output_dir / f"desc_per_task_{suffix}.png", chart_type=chart_type,
            naive_line=naive_sem_sim)
        plot_per_dataset_distribution(
            desc_by_smp, all_datasets, models, color_map, multimodal_set,
            "Sem Sim per Sample", "Vision Assessment — Description Per Dataset (per-sample spread)",
            output_dir / f"desc_per_dataset_{suffix}.png", chart_type=chart_type,
            naive_line=naive_sem_sim, ds_to_task=ds_to_task)

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
        "--dataset_source", default="agml", choices=["agml", "digital_green"],
        help="Source dataset: 'agml' uses --descriptions_jsonl + --config; "
             "'digital_green' uses --dg_descriptions_jsonl + --dg_config + --images_dir",
    )
    parser.add_argument(
        "--test_split", default="outputs/linear_probe/test_split.json",
        help="Path to test_split.json produced by 'probe' — restricts each class to the same "
             "images used as the linear probe test set for a controlled comparison",
    )
    parser.add_argument(
        "--dg_descriptions_jsonl", default="outputs/data_generation/dg_descriptions.jsonl",
        help="Digital Green descriptions JSONL (used when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--dg_config", default="outputs/data_generation/digital_green_datasets.yaml",
        help="Digital Green datasets YAML written by build-dg-jsonl (used when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--images_dir", default=None,
        help="Local directory of Digital Green images (used when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--descriptions_jsonl", default="dataset/agml_descriptions.jsonl",
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
        help="Max concurrent sequences for vLLM (lower to reduce KV-cache pressure on small GPUs)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="Fraction of GPU VRAM vLLM may use for weights + KV cache (default 0.9)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="Maximum sequence length (tokens) vLLM will allocate KV cache for (default 8192)",
    )
    parser.add_argument(
        "--enforce_eager", action="store_true",
        help="Disable CUDA graph capture (workaround for Triton kernel crashes with AWQ models)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip inference and judging for models whose output CSVs already exist; load from disk instead",
    )
    parser.add_argument(
        "--clustering_metrics", default="outputs/clustering_eval/clustering_metrics.json",
        help="Path to clustering_metrics.json from clustering eval (for difficulty correlation plots)",
    )
    args = parser.parse_args()
    if not args.judge_model and args.models:
        args.judge_model = args.models[0]
    main(args)
