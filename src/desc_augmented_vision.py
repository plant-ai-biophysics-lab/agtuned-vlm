"""
Description-Augmented Vision Assessment

Tests whether language assessment descriptions that produce a correctly-judged
classification answer can improve vision classification when prepended as
context to the image prompt.

Flow:
  1. Load lang cls CSV → find (dataset, class) pairs where judged_correct == 1
  2. Load lang desc CSV → get the description *response* for each winning pair
  3. Load existing vis cls CSV → filter to same pairs → recompute baseline metrics
  4. Run vision inference with augmented prompt:
       "Reference description: {desc}\n\nQuestion: {cls_q}\n\nAnswer with only the class name:"
  5. Compare baseline (filtered) vs augmented metrics
  6. Save CSVs + metrics JSON + plots to output_dir

Usage:
    uv run python src/desc_augmented_vision.py \\
      --models      google/gemma-4-26B-A4B-it \\
      --lang_dir    outputs/language_assessment/ \\
      --vis_dir     outputs/vision_assessment/ \\
      --output_dir  outputs/desc_augmented_vision/
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import gc
import json
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.common.dataset_config import datasets_by_name, prefetch_agml_datasets
from src.common.io import load_prompts, save_json, write_csv
from src.common.metrics import (
    THINK_RE,
    aggregate_by_task,
    apply_judge,
    compute_classification_metrics,
    contains_match,
    judged_summary,
    model_color_map,
    naive_classification_baseline,
    naive_uniform_baseline_per_dataset,
    resolve_class,
    sample_class_images,
)

CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)

_REPEAT_RE   = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*{1,3}([^*]*)\*{1,3}|^#{1,6}\s*|^[-*+]\s+", re.MULTILINE)
_GREETING_RE = re.compile(
    r"^(what (a )?(great|good|excellent|interesting|wonderful|nice)|"
    r"sure[!,.]?|great[!,.]?|of course[!,.]?|absolutely[!,.]?|"
    r"certainly[!,.]?|hello[!,.]?|hi[!,.]?|yes[!,.]?)\s*$",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(we need to|the user wants|the instruction|let me|i need to|i should|i will|"
    r"to answer this|the question (is|asks)|so (just|the answer)|"
    r"the answer (is|would be|should)|first[,\s]+(let|i)|step \d)\b",
    re.IGNORECASE,
)

_JUDGE_METRIC_MAP = {"f1_macro": "judge_f1_macro", "accuracy": "judge_accuracy"}
_TASK_LABELS = {"pest_damage": "pest/damage"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def _task_label(tt: str) -> str:
    return _TASK_LABELS.get(tt, tt.replace("_", " "))


def _clean_response(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    for tag in ("<think>", "<thought>"):
        idx = text.lower().find(tag)
        if idx != -1:
            text = text[:idx].strip()
    text = _MARKDOWN_RE.sub(r"\1", text)
    lines = text.splitlines()
    while lines and _GREETING_RE.match(lines[0].strip()):
        lines.pop(0)
    text = "\n".join(lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    text = ""
    for para in paragraphs:
        collapsed = " ".join(para.split())
        if not _REASONING_RE.search(collapsed):
            text = collapsed
            break
    if not text and paragraphs:
        last = " ".join(paragraphs[-1].split())
        if not _REASONING_RE.search(last):
            text = last
    text = " ".join(text.split())
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text.strip()


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(val) -> bool | None:
    if val is None or val == "":
        return None
    try:
        return bool(int(val))
    except (ValueError, TypeError):
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_winning_classes(
    lang_dir: Path, model_id: str
) -> dict[tuple[str, str], str]:
    """Return {(dataset, class): description_response} for judged-correct classes."""
    cls_file  = lang_dir / f"cls_{_safe_name(model_id)}.csv"
    desc_file = lang_dir / f"desc_{_safe_name(model_id)}.csv"

    if not cls_file.exists():
        print(f"  [warn] lang cls CSV not found: {cls_file}")
        return {}
    if not desc_file.exists():
        print(f"  [warn] lang desc CSV not found: {desc_file}")
        return {}

    # Find classes with judged_correct == 1
    winning: set[tuple[str, str]] = set()
    for row in _read_csv(cls_file):
        if _to_bool(row.get("judged_correct")) is True:
            winning.add((row["agml_dataset"], row["class_label"]))

    if not winning:
        print("  [warn] No judged-correct classes found in language assessment.")
        return {}

    # Map winning classes to their description response
    desc_map: dict[tuple[str, str], str] = {}
    for row in _read_csv(desc_file):
        key = (row.get("agml_dataset", ""), row.get("class_label", ""))
        if key in winning:
            desc_map[key] = row.get("prediction", "")

    return desc_map


def load_baseline_cls_rows(
    vis_dir: Path, model_id: str,
    winning: set[tuple[str, str]],
) -> list[dict]:
    """Load existing vision cls CSV and filter to winning (dataset, class) pairs."""
    cls_file = vis_dir / f"cls_{_safe_name(model_id)}.csv"
    if not cls_file.exists():
        print(f"  [warn] vis cls CSV not found: {cls_file}")
        return []
    rows = _read_csv(cls_file)
    return [r for r in rows if (r.get("agml_dataset", ""), r.get("class_label", "")) in winning]


def load_test_split(test_split_path: str | None) -> dict[str, dict[str, list[str]]]:
    if test_split_path and Path(test_split_path).exists():
        return json.loads(Path(test_split_path).read_text())
    return {}


# ── vLLM helpers ─────────────────────────────────────────────────────────────

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
        max_num_seqs = min(max_num_seqs, 32)
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
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [user_msg]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def run_batch(llm: LLM, tokenizer, images: list[PILImage.Image], question: str,
              params: SamplingParams, system_prompt: str = "") -> list[str]:
    prompts = [build_multimodal_prompt(tokenizer, img, question, system_prompt=system_prompt) for img in images]
    outputs = llm.generate(prompts, sampling_params=params)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def build_name_map(agml_name: str) -> dict[str, str]:
    import agml
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    return {Path(str(p)).name: str(p) for p in contents.keys()}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_cls_metrics_per_dataset(cls_rows: list[dict]) -> dict[str, dict]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in cls_rows:
        by_ds[row["agml_dataset"]].append(row)
    result = {}
    for ds, rows in by_ds.items():
        refs  = [r["class_label"] for r in rows]
        preds = [r.get("resolved_class", "") for r in rows]
        result[ds] = compute_classification_metrics(refs, preds)
    return result


# ── Inference ─────────────────────────────────────────────────────────────────

def run_augmented_model(
    model_id: str,
    desc_map: dict[tuple[str, str], str],
    datasets_cfg: dict,
    test_split: dict[str, dict[str, list[str]]],
    descriptions_jsonl_path: Path,
    cls_prompt_template: str,
    aug_prompt_template: str,
    system_prompt: str,
    n_samples: int | None,
    output_dir: Path,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
) -> list[dict]:
    """Run augmented vision classification for all winning (dataset, class) pairs."""

    # Build a lookup from JSONL for visual_cls_question per (dataset, class)
    from src.common.io import load_jsonl
    cls_q_map: dict[tuple[str, str], dict] = {}
    for record in load_jsonl(descriptions_jsonl_path):
        meta = record.get("metadata", {})
        key = (meta.get("agml_dataset", ""), meta.get("class_label", ""))
        cls_q_map[key] = meta

    llm, tokenizer = load_llm(
        model_id, quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )

    cls_rows: list[dict] = []
    agml_cache: dict[str, dict[str, str]] = {}
    winning_keys = sorted(desc_map.keys())

    t_start = time.perf_counter()
    pbar = tqdm(winning_keys, desc="  Classes", unit="class", dynamic_ncols=True)

    for key in pbar:
        ds, cls_label = key
        description = desc_map[key]
        meta = cls_q_map.get(key, {})
        task_type = meta.get("task_type", "")
        crop = meta.get("crop", "")
        visual_cls_q = meta.get("visual_cls_question")

        if not visual_cls_q:
            tqdm.write(f"  [skip] {ds} / {cls_label} — no visual_cls_question in JSONL")
            continue

        # Resolve image names: prefer test_split, fall back to all images in dataset
        if test_split and ds in test_split and cls_label in test_split[ds]:
            image_names = test_split[ds][cls_label]
        else:
            image_names = meta.get("image_names", [])
        image_names = sample_class_images(image_names, n_samples)

        pbar.set_postfix_str(f"{ds} / {cls_label}", refresh=False)

        if ds not in agml_cache:
            tqdm.write(f"\n  Building image map: {ds}...")
            agml_cache[ds] = build_name_map(ds)
        name_map = agml_cache[ds]

        images, valid_names = [], []
        for name in image_names:
            if name in name_map:
                images.append(PILImage.open(name_map[name]).convert("RGB"))
                valid_names.append(name)
            else:
                tqdm.write(f"    [warn] {name} not in {ds}")
        if not images:
            continue

        tqdm.write(f"  {ds} / {cls_label}  ({len(images)} images)")

        ds_cfg = datasets_cfg.get(ds)
        clean_classes = ds_cfg.clean_classes if ds_cfg else []

        # Augmented prompt: description context + classification question
        aug_question = aug_prompt_template.format(
            description=description,
            question=visual_cls_q,
        )

        preds = run_batch(llm, tokenizer, images, aug_question, CLS_PARAMS, system_prompt=system_prompt)
        resolved_list = [resolve_class(p, clean_classes) for p in preds]

        for pred, img_name, resolved in zip(preds, valid_names, resolved_list):
            correct = resolved is not None and resolved.lower() == cls_label.lower()
            c_match = contains_match(resolved or pred, cls_label)
            cls_rows.append({
                "agml_dataset":   ds,
                "class_label":    cls_label,
                "task_type":      task_type,
                "crop":           crop,
                "image_name":     img_name,
                "prediction":     pred,
                "resolved_class": resolved or "",
                "ground_truth":   cls_label,
                "correct":        int(correct),
                "contains_match": int(c_match),
                "resolvable":     int(resolved is not None),
                "lang_description": description,
            })

    elapsed = time.perf_counter() - t_start
    print(f"\nInference time: {elapsed:.1f}s")

    destroy_llm(llm)
    del tokenizer

    return cls_rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def _hatch(model_key: str, multimodal_models: set[str]) -> str:
    return "///" if model_key in multimodal_models else ""


def _draw_naive_line(ax, naive_overall: dict | None, metric: str,
                     label: bool = False, uniform_overall: dict | None = None) -> None:
    if naive_overall:
        value = naive_overall.get(metric)
        if value is not None:
            ax.axhline(value, color="red", linestyle="--", linewidth=1.2,
                       label="naive (shuffled)" if label else "_nolegend_")
    if uniform_overall:
        value = uniform_overall.get(metric)
        if value is not None:
            ax.axhline(value, color="steelblue", linestyle="--", linewidth=1.2,
                       label="naive (1/k classes)" if label else "_nolegend_")


def plot_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    output_dir: Path,
    naive_cls_overall: dict | None = None,
    uniform_cls_overall: dict | None = None,
) -> None:
    """Side-by-side comparison: baseline (filtered) vs description-augmented."""
    all_datasets = sorted(
        set(ds for m in baseline_per_ds.values() for ds in m)
        | set(ds for m in augmented_per_ds.values() for ds in m)
    )
    if not all_datasets:
        return

    # Build combined model keys: "model (baseline)" and "model (augmented)"
    models = sorted(baseline_per_ds.keys())
    combined: dict[str, dict[str, dict]] = {}
    colors: dict[str, str] = {}
    base_palette  = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    aug_palette   = ["#1a5276", "#1e8449", "#6c3483", "#9a6b00"]
    for i, model in enumerate(models):
        bkey = f"{model.split('/')[-1]} (baseline)"
        akey = f"{model.split('/')[-1]} (augmented)"
        combined[bkey] = baseline_per_ds.get(model, {})
        combined[akey] = augmented_per_ds.get(model, {})
        colors[bkey] = base_palette[i % len(base_palette)]
        colors[akey] = aug_palette[i % len(aug_palette)]

    all_keys = list(combined.keys())
    metrics = ["accuracy", "f1_macro"]
    x = np.arange(len(all_datasets))
    width = 0.8 / len(all_keys)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(12, len(all_datasets) * len(all_keys) * 0.4 + 2), 5))
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, key in enumerate(all_keys):
            vals = [combined[key].get(ds, {}).get(metric, 0.0) for ds in all_datasets]
            offset = (i - (len(all_keys) - 1) / 2) * width
            ls = "--" if "(augmented)" in key else "-"
            ax.bar(x + offset, vals, width, label=key, color=colors[key],
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   linestyle=ls)
            for j, v in enumerate(vals):
                ax.text(x[j] + offset, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6)
        _draw_naive_line(ax, naive_cls_overall, metric, label=True, uniform_overall=uniform_cls_overall)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Description-Augmented Vision — Baseline vs Augmented (judged-correct classes only)", fontsize=10)
    plt.tight_layout()
    path = output_dir / "cls_comparison_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
    naive_cls_overall: dict | None = None,
    uniform_cls_overall: dict | None = None,
) -> None:
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = sorted(baseline_per_ds.keys())
    combined: dict[str, dict[str, dict]] = {}
    colors: dict[str, str] = {}
    base_palette = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    aug_palette  = ["#1a5276", "#1e8449", "#6c3483", "#9a6b00"]
    for i, model in enumerate(models):
        bkey = f"{model.split('/')[-1]} (baseline)"
        akey = f"{model.split('/')[-1]} (augmented)"
        combined[bkey] = {
            tt: compute_classification_metrics(
                [r for ds, rows in baseline_per_ds.get(model, {}).items()
                 if ds_to_task.get(ds) == tt for r in ["_dummy_"]],
                []
            )
            for tt in task_types
        }
        colors[bkey] = base_palette[i % len(base_palette)]
        colors[akey] = aug_palette[i % len(aug_palette)]

    # Aggregate per task from per-dataset results
    def _by_task(per_ds: dict[str, dict]) -> dict[str, dict]:
        task_rows: dict[str, list] = defaultdict(list)
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                task_rows[tt].append(m)
        return {
            tt: {
                metric: float(np.mean([r.get(metric, 0.0) for r in rows]))
                for metric in ["accuracy", "f1_macro"]
            }
            for tt, rows in task_rows.items()
        }

    all_keys = list(models)
    metrics = ["accuracy", "f1_macro"]
    x = np.arange(len(task_types))
    width = 0.8 / (len(all_keys) * 2)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(8, len(task_types) * len(all_keys) * 1.2 + 2), 5))
    slot = 0
    for i, model in enumerate(all_keys):
        bkey = f"{model.split('/')[-1]} (baseline)"
        akey = f"{model.split('/')[-1]} (augmented)"
        b_by_task = _by_task(baseline_per_ds.get(model, {}))
        a_by_task = _by_task(augmented_per_ds.get(model, {}))

        for mi, metric in enumerate(metrics):
            ax = axes[mi]
            bvals = [b_by_task.get(tt, {}).get(metric, 0.0) for tt in task_types]
            avals = [a_by_task.get(tt, {}).get(metric, 0.0) for tt in task_types]
            b_offset = (slot * 2     - (len(all_keys) * 2 - 1) / 2) * width
            a_offset = (slot * 2 + 1 - (len(all_keys) * 2 - 1) / 2) * width
            ax.bar(x + b_offset, bvals, width, label=bkey, color=base_palette[i % len(base_palette)],
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x + a_offset, avals, width, label=akey, color=aug_palette[i % len(aug_palette)],
                   alpha=0.85, edgecolor="black", linewidth=0.5, hatch="///")
            for j, (bv, av) in enumerate(zip(bvals, avals)):
                ax.text(x[j] + b_offset, bv + 0.005, f"{bv:.2f}", ha="center", va="bottom", fontsize=6)
                ax.text(x[j] + a_offset, av + 0.005, f"{av:.2f}", ha="center", va="bottom", fontsize=6)
        slot += 1

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        _draw_naive_line(ax, naive_cls_overall, metric, label=True, uniform_overall=uniform_cls_overall)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Description-Augmented Vision — By Task Type", fontsize=10)
    plt.tight_layout()
    path = output_dir / "cls_comparison_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    output_dir: Path,
    naive_cls_overall: dict | None = None,
    uniform_cls_overall: dict | None = None,
) -> None:
    models = sorted(baseline_per_ds.keys())
    metrics = ["accuracy", "f1_macro"]
    base_palette = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    aug_palette  = ["#1a5276", "#1e8449", "#6c3483", "#9a6b00"]

    def _overall(per_ds: dict[str, dict]) -> dict[str, float]:
        all_rows = [v for rows in per_ds.values() for v in [rows]]
        if not all_rows:
            return {}
        return {
            metric: float(np.mean([r.get(metric, 0.0) for r in all_rows]))
            for metric in metrics
        }

    all_keys = []
    vals_map: dict[str, dict] = {}
    colors: dict[str, str] = {}
    for i, model in enumerate(models):
        bkey = f"{model.split('/')[-1]} (baseline)"
        akey = f"{model.split('/')[-1]} (augmented)"
        all_keys += [bkey, akey]
        vals_map[bkey] = _overall(baseline_per_ds.get(model, {}))
        vals_map[akey] = _overall(augmented_per_ds.get(model, {}))
        colors[bkey] = base_palette[i % len(base_palette)]
        colors[akey] = aug_palette[i % len(aug_palette)]

    x = np.arange(len(metrics))
    width = 0.8 / len(all_keys)

    fig, ax = plt.subplots(figsize=(max(6, len(all_keys) * 1.5), 5))
    for i, key in enumerate(all_keys):
        vals = [vals_map[key].get(m, 0.0) for m in metrics]
        offset = (i - (len(all_keys) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=key, color=colors[key],
               alpha=0.85, edgecolor="black", linewidth=0.5,
               hatch="///" if "(augmented)" in key else "")
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    _draw_naive_line(ax, naive_cls_overall, "f1_macro", label=True, uniform_overall=uniform_cls_overall)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    fig.suptitle("Description-Augmented Vision — Overall (judged-correct classes only)", fontsize=10)
    plt.tight_layout()
    path = output_dir / "cls_overall_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_dir = Path(args.lang_dir)
    vis_dir  = Path(args.vis_dir)

    prompts = load_prompts(Path(args.prompts))
    aug_prompt_template = prompts["desc_augmented_classification"]
    cls_prompt_template = prompts["assessment_classification"]
    system_prompt = prompts.get("assessment_system", "")

    datasets_cfg = datasets_by_name(Path(args.config))
    active_cfg   = {k: v for k, v in datasets_cfg.items() if not v.skip}
    ds_to_task   = {k: v.task_type for k, v in datasets_cfg.items()}

    test_split = load_test_split(args.test_split)
    if test_split:
        print(f"Test split loaded: {len(test_split)} datasets")
    else:
        print("[warn] No test split found — using all images per class")

    prefetch_agml_datasets(list(active_cfg.values()))

    descriptions_jsonl = Path(args.descriptions_jsonl)

    n_models = len(args.models)
    baseline_per_model:  dict[str, dict[str, dict]] = {}
    augmented_per_model: dict[str, dict[str, dict]] = {}
    summary: dict[str, dict] = {}

    for model_idx, model_id in enumerate(args.models):
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{n_models}: {model_id}")
        print(f"{'='*60}")

        # Step 1: find winning (judged-correct) classes from language assessment
        print("\nStep 1/3: Loading judged-correct classes from language assessment...")
        desc_map = load_winning_classes(lang_dir, model_id)
        if not desc_map:
            print(f"  [skip] No winning classes found for {model_id}")
            continue
        winning = set(desc_map.keys())
        print(f"  Found {len(winning)} judged-correct classes")

        # Step 2: load and filter existing vision baseline
        print("\nStep 2/3: Loading baseline vision scores (filtered to winning classes)...")
        baseline_rows = load_baseline_cls_rows(vis_dir, model_id, winning)
        if not baseline_rows:
            print(f"  [warn] No baseline vision rows found for {model_id}")
        else:
            print(f"  {len(baseline_rows)} baseline image rows across {len(winning)} classes")
        baseline_per_ds = compute_cls_metrics_per_dataset(baseline_rows)
        baseline_overall = compute_classification_metrics(
            [r["class_label"] for r in baseline_rows],
            [r.get("resolved_class", "") for r in baseline_rows],
        ) if baseline_rows else {}

        # Step 3: run augmented inference (skip if CSV already exists and --force not set)
        safe = _safe_name(model_id)
        aug_csv_path = output_dir / f"aug_cls_{safe}.csv"

        if aug_csv_path.exists() and args.resume:
            print(f"\nStep 3/3: Inference CSV already exists — skipping ({aug_csv_path.name})")
            import csv as _csv
            with open(aug_csv_path, newline="", encoding="utf-8") as f:
                aug_rows = list(_csv.DictReader(f))
        else:
            print(f"\nStep 3/3: Running description-augmented vision inference...")
            aug_rows = run_augmented_model(
                model_id=model_id,
                desc_map=desc_map,
                datasets_cfg=active_cfg,
                test_split=test_split,
                descriptions_jsonl_path=descriptions_jsonl,
                cls_prompt_template=cls_prompt_template,
                aug_prompt_template=aug_prompt_template,
                system_prompt=system_prompt,
                n_samples=args.n_samples,
                output_dir=output_dir,
                quantization=args.quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager,
            )
            write_csv(aug_csv_path, aug_rows)
            write_csv(output_dir / f"baseline_cls_{safe}.csv", baseline_rows)

        aug_per_ds = compute_cls_metrics_per_dataset(aug_rows)
        aug_overall = compute_classification_metrics(
            [r["class_label"] for r in aug_rows],
            [r.get("resolved_class", "") for r in aug_rows],
        ) if aug_rows else {}

        baseline_per_model[model_id]  = baseline_per_ds
        augmented_per_model[model_id] = aug_per_ds

        summary[model_id] = {
            "n_winning_classes": len(winning),
            "n_baseline_rows":   len(baseline_rows),
            "n_augmented_rows":  len(aug_rows),
            "baseline_overall":  baseline_overall,
            "augmented_overall": aug_overall,
            "baseline_per_dataset":  baseline_per_ds,
            "augmented_per_dataset": aug_per_ds,
        }

        print(f"\n  Baseline  → acc: {baseline_overall.get('accuracy', 0):.3f}  F1: {baseline_overall.get('f1_macro', 0):.3f}")
        print(f"  Augmented → acc: {aug_overall.get('accuracy', 0):.3f}  F1: {aug_overall.get('f1_macro', 0):.3f}")

    # LLM judge (on augmented rows only, shared judge model)
    if args.judge_model and summary:
        print(f"\n=== LLM Judge: {args.judge_model} ===")
        from vllm import LLM as _LLM
        judge_llm, judge_tok = load_llm(
            args.judge_model,
            quantization=args.judge_quantization,
            gpu_memory_utilization=0.9,
        )
        for model_id in args.models:
            if model_id not in summary:
                continue
            safe = _safe_name(model_id)
            aug_csv = output_dir / f"aug_cls_{safe}.csv"
            import csv as _csv
            with open(aug_csv, newline="", encoding="utf-8") as f:
                aug_rows = list(_csv.DictReader(f))
            if args.resume and aug_rows and "judged_correct" in aug_rows[0]:
                print(f"  [{model_id}] already judged — skipping")
                summary[model_id]["judge"] = judged_summary(aug_rows)
                continue
            print(f"  Judging {model_id} ({len(aug_rows)} rows)...")
            aug_rows = apply_judge(
                rows=aug_rows,
                llm=judge_llm,
                tokenizer=judge_tok,
                system_prompt=prompts["judge_system_vision"],
                judge_template=prompts["judge_classification_template"],
                gt_key="class_label",
                pred_key="prediction",
                confidence_threshold=args.judge_confidence_threshold,
            )
            write_csv(aug_csv, aug_rows)
            summary[model_id]["judge"] = judged_summary(aug_rows)
        del judge_llm, judge_tok
        gc.collect()

    # Save metrics JSON
    save_json(output_dir / "desc_augmented_metrics.json", {"models": summary})
    print(f"\nMetrics saved to {output_dir / 'desc_augmented_metrics.json'}")

    # Naive baselines over the filtered set (union of all baseline rows)
    all_baseline_rows = [
        r for model_id in args.models
        for r in _read_baseline_rows_cached(output_dir, model_id)
    ]
    if all_baseline_rows:
        naive_cls_overall   = naive_classification_baseline([r["class_label"] for r in all_baseline_rows])
        uniform_cls_overall = naive_uniform_baseline_per_dataset(all_baseline_rows)
    else:
        naive_cls_overall = uniform_cls_overall = None

    print("\n=== Generating plots ===")
    plot_comparison(baseline_per_model, augmented_per_model, output_dir, naive_cls_overall, uniform_cls_overall)
    plot_task_comparison(baseline_per_model, augmented_per_model, ds_to_task, output_dir, naive_cls_overall, uniform_cls_overall)
    plot_overall_comparison(baseline_per_model, augmented_per_model, output_dir, naive_cls_overall, uniform_cls_overall)
    print("\nDone.")


def _read_baseline_rows_cached(output_dir: Path, model_id: str) -> list[dict]:
    path = output_dir / f"baseline_cls_{_safe_name(model_id)}.csv"
    return _read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description-Augmented Vision Assessment")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--lang_dir",    default="outputs/language_assessment/")
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/")
    parser.add_argument("--output_dir",  default="outputs/desc_augmented_vision/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--prompts",     default="src/config/prompts.yaml")
    parser.add_argument("--descriptions_jsonl", default="dataset/agml_descriptions.jsonl")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max images per class (same as vision assessment)")
    parser.add_argument("--judge_model", default=None,
                        help="LLM-as-judge model (default: first model in --models)")
    parser.add_argument("--judge_quantization", default=None,
                        choices=["awq", "gptq", "fp8", "bitsandbytes"])
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--quantization", default=None,
                        choices=["awq", "gptq", "fp8", "bitsandbytes"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs",  type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip inference/judging if output CSVs already exist")

    args = parser.parse_args()
    if not args.judge_model and args.models:
        args.judge_model = args.models[0]
    main(args)
