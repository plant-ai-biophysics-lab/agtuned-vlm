"""
Description-Augmented Vision Assessment

Tests whether clean class descriptions can improve vision classification when
prepended as context to the image prompt.

Flow:
  1. Load descriptions from dataset/agml_descriptions.jsonl (synthesized_description field)
     For filtered run: restrict to judged-correct classes from lang cls CSV.
     For unfiltered (--all_classes): use all classes in JSONL.
  2. Load existing vis cls CSV → filter to same pairs → recompute baseline metrics
  3. Run vision inference with augmented prompt:
       "Reference description: {desc}\n\nQuestion: {cls_q}\n\nAnswer with only the class name:"
  4. Compare baseline (filtered) vs augmented metrics
  5. Save CSVs + metrics JSON + plots to output_dir

Descriptions always come from --descriptions_jsonl (clean, no class names in prompt).
--lang_dir is only needed for filtered run (to find judged-correct classes from cls CSV).

Usage (unfiltered — all classes from JSONL):
    uv run python src/desc_augmented_vision.py \\
      --models      google/gemma-4-26B-A4B-it \\
      --vis_dir     outputs/vision_assessment/ \\
      --output_dir  outputs/desc_augmented_vision/ \\
      --descriptions_jsonl dataset/agml_descriptions.jsonl \\
      --all_classes

Usage (filtered — judged-correct classes only):
    uv run python src/desc_augmented_vision.py \\
      --models      google/gemma-4-26B-A4B-it \\
      --lang_dir    outputs/language_assessment/ \\
      --vis_dir     outputs/vision_assessment/ \\
      --output_dir  outputs/desc_augmented_vision/ \\
      --descriptions_jsonl dataset/agml_descriptions.jsonl
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
    resolve_class,
    sample_class_images,
)

try:
    from vllm import LLM, SamplingParams
    CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)
except ImportError:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]
    CLS_PARAMS = None

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

def load_jsonl_desc_map(jsonl_path: Path) -> dict[tuple[str, str], str]:
    """Load {(dataset, class): synthesized_description} from agml_descriptions.jsonl."""
    from src.common.io import load_jsonl
    desc_map: dict[tuple[str, str], str] = {}
    for record in load_jsonl(jsonl_path):
        meta = record.get("metadata", {})
        ds   = meta.get("agml_dataset", "")
        cls  = meta.get("class_label", "")
        desc = meta.get("synthesized_description", "")
        if ds and cls and desc:
            desc_map[(ds, cls)] = desc
    return desc_map


def load_winning_classes(
    lang_dir: Path | None,
    model_id: str,
    jsonl_desc_map: dict[tuple[str, str], str],
    all_classes: bool = False,
) -> dict[tuple[str, str], str]:
    """Return {(dataset, class): description} using clean JSONL synthesized descriptions.

    all_classes=True : all classes present in JSONL.
    all_classes=False: only judged-correct classes from lang cls CSV (requires --lang_dir).
    Descriptions always come from JSONL — never from lang desc_*.csv.
    """
    if all_classes:
        return dict(jsonl_desc_map)

    if lang_dir is None:
        print("  [warn] --lang_dir required for filtered run (judged-correct cls CSV lookup)")
        return {}
    cls_file = lang_dir / f"cls_{_safe_name(model_id)}.csv"
    if not cls_file.exists():
        print(f"  [warn] lang cls CSV not found: {cls_file}")
        return {}

    winning: set[tuple[str, str]] = set()
    for row in _read_csv(cls_file):
        if _to_bool(row.get("judged_correct")) is True:
            winning.add((row["agml_dataset"], row["class_label"]))
    if not winning:
        print("  [warn] No judged-correct classes found in language assessment.")
        return {}

    return {key: desc for key, desc in jsonl_desc_map.items() if key in winning}


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


def load_vision_desc_map(
    vis_dir: Path, model_id: str,
) -> dict[tuple[str, str], dict[str, dict]]:
    """Load per-image descriptions from vision assessment desc CSV.

    Returns {(dataset, class): {image_name: {description, rouge_ss, sem_sim, ...}}}
    Only images that were actually described are included, ensuring correct alignment.
    """
    desc_file = vis_dir / f"desc_{_safe_name(model_id)}.csv"
    if not desc_file.exists():
        print(f"  [warn] vision desc CSV not found: {desc_file}")
        return {}
    result: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in _read_csv(desc_file):
        ds        = row.get("agml_dataset", "")
        cls_label = row.get("class_label", "")
        img_name  = row.get("image_name", "")
        if not (ds and cls_label and img_name):
            continue
        result[(ds, cls_label)][img_name] = {
            "description": row.get("prediction", ""),
            "rouge_ss":    row.get("rouge_ss", ""),
            "sem_sim":     row.get("sem_sim", ""),
            "rouge1":      row.get("rouge1", ""),
            "rougeL":      row.get("rougeL", ""),
        }
    return dict(result)


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

def compute_cls_metrics_per_dataset(cls_rows: list[dict], use_judged: bool = False) -> dict[str, dict]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in cls_rows:
        by_ds[row["agml_dataset"]].append(row)
    result = {}
    for ds, rows in by_ds.items():
        refs = [r["class_label"] for r in rows]
        if use_judged:
            # treat judged_correct==1 as a correct prediction (class_label), else wrong
            preds = [r["class_label"] if int(r.get("judged_correct") or 0) else "" for r in rows]
        else:
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


def run_augmented_model_vision_desc(
    model_id: str,
    vision_desc_map: dict[tuple[str, str], dict[str, dict]],
    datasets_cfg: dict,
    descriptions_jsonl_path: Path,
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
    """Run augmented vision classification using per-image vision-generated descriptions.

    Each image is matched to its own description from the vision assessment desc CSV,
    so the description and image are guaranteed to be aligned.
    """
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
    winning_keys = sorted(vision_desc_map.keys())

    t_start = time.perf_counter()
    pbar = tqdm(winning_keys, desc="  Classes", unit="class", dynamic_ncols=True)

    for key in pbar:
        ds, cls_label = key
        img_desc_map  = vision_desc_map[key]  # {image_name: {description, rouge_ss, ...}}
        meta          = cls_q_map.get(key, {})
        task_type     = meta.get("task_type", "")
        crop          = meta.get("crop", "")
        visual_cls_q  = meta.get("visual_cls_question")

        if not visual_cls_q:
            tqdm.write(f"  [skip] {ds} / {cls_label} — no visual_cls_question in JSONL")
            continue

        # Use only images that were described; respect n_samples with same seed
        image_names = sample_class_images(sorted(img_desc_map.keys()), n_samples)
        pbar.set_postfix_str(f"{ds} / {cls_label}", refresh=False)

        if ds not in agml_cache:
            tqdm.write(f"\n  Building image map: {ds}...")
            agml_cache[ds] = build_name_map(ds)
        name_map = agml_cache[ds]

        ds_cfg       = datasets_cfg.get(ds)
        clean_classes = ds_cfg.clean_classes if ds_cfg else []

        # Build one prompt per image (each has its own description)
        prompts: list[dict] = []
        valid_meta: list[dict] = []
        for img_name in image_names:
            if img_name not in name_map:
                tqdm.write(f"    [warn] {img_name} not in {ds}")
                continue
            img_info    = img_desc_map[img_name]
            description = img_info["description"]
            if not description:
                continue
            image = PILImage.open(name_map[img_name]).convert("RGB")
            aug_question = aug_prompt_template.format(
                description=description,
                question=visual_cls_q,
            )
            prompts.append(build_multimodal_prompt(tokenizer, image, aug_question, system_prompt))
            valid_meta.append({"img_name": img_name, **img_info})

        if not prompts:
            continue

        tqdm.write(f"  {ds} / {cls_label}  ({len(prompts)} images)")
        outputs = llm.generate(prompts, sampling_params=CLS_PARAMS)

        for out, vmeta in zip(outputs, valid_meta):
            pred     = _clean_response(out.outputs[0].text)
            resolved = resolve_class(pred, clean_classes)
            correct  = resolved is not None and resolved.lower() == cls_label.lower()
            cls_rows.append({
                "agml_dataset":    ds,
                "class_label":     cls_label,
                "task_type":       task_type,
                "crop":            crop,
                "image_name":      vmeta["img_name"],
                "prediction":      pred,
                "resolved_class":  resolved or "",
                "ground_truth":    cls_label,
                "correct":         int(correct),
                "contains_match":  int(contains_match(resolved or pred, cls_label)),
                "resolvable":      int(resolved is not None),
                "vis_description": vmeta["description"],
                "desc_rouge_ss":   vmeta["rouge_ss"],
                "desc_sem_sim":    vmeta["sem_sim"],
            })

    elapsed = time.perf_counter() - t_start
    print(f"\nInference time: {elapsed:.1f}s")
    destroy_llm(llm)
    del tokenizer
    return cls_rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def _hatch(model_key: str, multimodal_models: set[str]) -> str:
    return "///" if model_key in multimodal_models else ""


_SERIES_STYLES: list[dict] = [
    {"color": "#e67e22", "hatch": ""},     # vision baseline
    {"color": "#2ecc71", "hatch": "///"},  # augmented
]
_SERIES_SUFFIXES = [" (vision)", " (augmented)"]


def _build_series(
    models: list[str],
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
) -> tuple[list[str], list[dict], list[str], list[str]]:
    """Return (keys, per_ds_list, colors, hatches) in vision/augmented order."""
    keys, per_ds_list, colors, hatches = [], [], [], []
    for model in models:
        short = model.split("/")[-1]
        for suffix, src, style in zip(_SERIES_SUFFIXES, [baseline_per_ds, augmented_per_ds], _SERIES_STYLES):
            if src.get(model):
                keys.append(f"{short}{suffix}")
                per_ds_list.append(src[model])
                colors.append(style["color"])
                hatches.append(style["hatch"])
    return keys, per_ds_list, colors, hatches


def plot_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    output_dir: Path,
    suffix: str = "filtered",
) -> None:
    models = sorted(set(baseline_per_ds) | set(augmented_per_ds))
    keys, series, colors, hatches = _build_series(models, baseline_per_ds, augmented_per_ds)

    all_datasets = sorted(set(ds for src in series for ds in src))
    if not all_datasets or not keys:
        return

    metrics = ["accuracy", "f1_macro"]
    x = np.arange(len(all_datasets))
    width = 0.8 / len(keys)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(12, len(all_datasets) * len(keys) * 0.4 + 2), 5))
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (key, src, color, hatch) in enumerate(zip(keys, series, colors, hatches)):
            vals = [src.get(ds, {}).get(metric, 0.0) for ds in all_datasets]
            offset = (i - (len(keys) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=key, color=color, hatch=hatch,
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            for j, v in enumerate(vals):
                ax.text(x[j] + offset, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Description-Augmented Vision — Per Dataset (judged-correct classes only)", fontsize=10)
    plt.tight_layout()
    path = output_dir / f"cls_comparison_per_dataset_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    ds_to_task: dict[str, str],
    output_dir: Path,
    suffix: str = "filtered",
) -> None:
    task_types = sorted(set(ds_to_task.values()))
    if not task_types:
        return

    models = sorted(set(baseline_per_ds) | set(augmented_per_ds))
    keys, series, colors, hatches = _build_series(models, baseline_per_ds, augmented_per_ds)

    def _by_task(per_ds: dict[str, dict]) -> dict[str, dict]:
        task_rows: dict[str, list] = defaultdict(list)
        for ds, m in per_ds.items():
            tt = ds_to_task.get(ds)
            if tt:
                task_rows[tt].append(m)
        return {
            tt: {metric: float(np.mean([r.get(metric, 0.0) for r in rows]))
                 for metric in ["accuracy", "f1_macro"]}
            for tt, rows in task_rows.items()
        }

    by_task_list = [_by_task(src) for src in series]
    metrics = ["accuracy", "f1_macro"]
    x = np.arange(len(task_types))
    width = 0.8 / len(keys)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(8, len(task_types) * len(keys) * 0.6 + 2), 5))
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (key, bt, color, hatch) in enumerate(zip(keys, by_task_list, colors, hatches)):
            vals = [bt.get(tt, {}).get(metric, 0.0) for tt in task_types]
            offset = (i - (len(keys) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=key, color=color, hatch=hatch,
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            for j, v in enumerate(vals):
                ax.text(x[j] + offset, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_label(tt) for tt in task_types], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Description-Augmented Vision — By Task Type", fontsize=10)
    plt.tight_layout()
    path = output_dir / f"cls_comparison_per_task_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_comparison(
    baseline_per_ds: dict[str, dict[str, dict]],
    augmented_per_ds: dict[str, dict[str, dict]],
    output_dir: Path,
    suffix: str = "filtered",
) -> None:
    models = sorted(set(baseline_per_ds) | set(augmented_per_ds))
    keys, series, colors, hatches = _build_series(models, baseline_per_ds, augmented_per_ds)

    def _overall(per_ds: dict[str, dict]) -> dict[str, float]:
        rows = list(per_ds.values())
        if not rows:
            return {}
        return {metric: float(np.mean([r.get(metric, 0.0) for r in rows]))
                for metric in ["accuracy", "f1_macro"]}

    metrics = ["accuracy", "f1_macro"]
    x = np.arange(len(metrics))
    width = 0.8 / len(keys)

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.5), 5))
    for i, (key, src, color, hatch) in enumerate(zip(keys, series, colors, hatches)):
        vals = [_overall(src).get(m, 0.0) for m in metrics]
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=key, color=color, hatch=hatch,
               alpha=0.85, edgecolor="black", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    fig.suptitle("Description-Augmented Vision — Overall (judged-correct classes only)", fontsize=10)
    plt.tight_layout()
    path = output_dir / f"cls_overall_comparison_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_dir = Path(args.lang_dir) if args.lang_dir else None
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
    print(f"Loading JSONL descriptions from {descriptions_jsonl}...")
    jsonl_desc_map = load_jsonl_desc_map(descriptions_jsonl)
    print(f"  Loaded {len(jsonl_desc_map)} class descriptions from JSONL")

    n_models = len(args.models)
    baseline_per_model:  dict[str, dict[str, dict]] = {}
    augmented_per_model: dict[str, dict[str, dict]] = {}
    summary: dict[str, dict] = {}

    for model_idx, model_id in enumerate(args.models):
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{n_models}: {model_id}")
        print(f"{'='*60}")

        # Step 1: find classes to augment
        filter_label = "all" if args.all_classes else "filtered"
        if args.all_classes:
            print("\nStep 1/3: Loading all classes with descriptions...")
        else:
            print("\nStep 1/3: Loading judged-correct classes from language assessment...")
        desc_map = load_winning_classes(lang_dir, model_id, jsonl_desc_map, all_classes=args.all_classes)
        if not desc_map:
            print(f"  [skip] No classes found for {model_id}")
            continue
        winning = set(desc_map.keys())
        print(f"  Found {len(winning)} classes ({'all' if args.all_classes else 'judged-correct only'})")

        # Step 2: load and filter existing vision baseline
        print(f"\nStep 2/3: Loading baseline vision scores (filtered to {len(winning)} classes)...")
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

        # Step 3: run augmented inference (skip if CSV already exists and --resume set)
        safe = _safe_name(model_id)
        aug_csv_path      = output_dir / f"aug_cls_{filter_label}_{safe}.csv"
        baseline_csv_path = output_dir / f"baseline_cls_{filter_label}_{safe}.csv"

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
            write_csv(baseline_csv_path, baseline_rows)

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
            aug_csv = output_dir / f"aug_cls_{filter_label}_{safe}.csv"
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

    # Save metrics JSON (one per filter mode so runs don't overwrite each other)
    metrics_path = output_dir / f"desc_augmented_metrics_{filter_label}.json"
    save_json(metrics_path, {"models": summary, "filter": filter_label})
    print(f"\nMetrics saved to {metrics_path}")

    print("\n=== Generating plots ===")
    plot_comparison(baseline_per_model, augmented_per_model, output_dir, suffix=filter_label)
    plot_task_comparison(baseline_per_model, augmented_per_model, ds_to_task, output_dir, suffix=filter_label)
    plot_overall_comparison(baseline_per_model, augmented_per_model, output_dir, suffix=filter_label)
    plot_transitions(vis_dir, output_dir, ds_to_task, output_dir, args.models,
                     filter_label=filter_label)
    plot_aug_distribution(args.models, summary, output_dir, filter_label, ds_to_task)

    # ── Sub-step: vision-generated descriptions ────────────────────────────────
    if args.use_vision_descriptions:
        print("\n=== Sub-step: Vision-Generated Description Augmentation ===")
        for model_id in args.models:
            print(f"\n  Model: {model_id}")
            safe         = _safe_name(model_id)
            vis_desc_map = load_vision_desc_map(vis_dir, model_id)
            if not vis_desc_map:
                print(f"  [skip] no vision desc CSV found for {model_id}")
                continue
            n_cls = len(vis_desc_map)
            n_img = sum(len(v) for v in vis_desc_map.values())
            print(f"  Loaded {n_img} image descriptions across {n_cls} classes")

            vis_aug_csv = output_dir / f"vis_desc_aug_cls_{safe}.csv"
            if vis_aug_csv.exists() and args.resume:
                print(f"  [resume] loading existing: {vis_aug_csv.name}")
                vis_aug_rows = _read_csv(vis_aug_csv)
            else:
                vis_aug_rows = run_augmented_model_vision_desc(
                    model_id=model_id,
                    vision_desc_map=vis_desc_map,
                    datasets_cfg=active_cfg,
                    descriptions_jsonl_path=descriptions_jsonl,
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
                write_csv(vis_aug_csv, vis_aug_rows)
                print(f"  Saved: {vis_aug_csv.name}")

            print(f"  Generating correlation plot...")
            plot_desc_quality_correlation(vis_aug_rows, output_dir, suffix=f"vis_desc_{safe}")

    print("\nDone.")


def plot_aug_distribution(
    models: list[str],
    summary: dict,
    output_dir: Path,
    filter_label: str,
    ds_to_task: dict[str, str],
) -> None:
    """Strip + errorbar distribution charts for baseline vs augmented classification."""
    from src.common.distribution_plots import (
        gather_cls_per_class_by_dataset,
        gather_cls_values_by_dataset,
        gather_cls_values_by_task_from_per_class,
        gather_judge_per_class_by_dataset,
        plot_overall_distribution,
        plot_per_dataset_distribution,
        plot_per_task_distribution,
    )
    from src.common.metrics import model_color_map

    dist_rows:    dict[str, list[dict]] = {}
    dist_metrics: dict[str, dict[str, dict]] = {}
    for model_id in models:
        if model_id not in summary:
            continue
        safe  = _safe_name(model_id)
        short = model_id.split("/")[-1]
        b_rows = _read_csv(output_dir / f"baseline_cls_{filter_label}_{safe}.csv")
        a_rows = _read_csv(output_dir / f"aug_cls_{filter_label}_{safe}.csv")
        dist_rows[f"{short} (baseline)"]    = b_rows
        dist_rows[f"{short} (aug)"]         = a_rows
        dist_metrics[f"{short} (baseline)"] = compute_cls_metrics_per_dataset(b_rows)
        dist_metrics[f"{short} (aug)"]      = compute_cls_metrics_per_dataset(a_rows)

    if not dist_metrics:
        return

    dist_models = list(dist_metrics.keys())
    color_map   = model_color_map(dist_models)
    mm_set: set[str] = set()

    all_ds      = sorted(set(ds for m in dist_metrics.values() for ds in m))
    task_types  = sorted(set(ds_to_task.get(ds, "") for ds in all_ds) - {""})

    cls_by_ds   = gather_cls_values_by_dataset(dist_metrics)
    cls_by_cls  = gather_cls_per_class_by_dataset(dist_rows, "class_label", "resolved_class", ds_to_task)
    judge_by_cls = gather_judge_per_class_by_dataset(dist_rows, "class_label", ds_to_task)
    cls_by_task = gather_cls_values_by_task_from_per_class(cls_by_cls)
    _dtk        = lambda k: (k[1], k[0]) if isinstance(k, tuple) else ("", k)
    cls_ds_tasks = sorted(set(k for m in cls_by_cls.values() for k in m), key=_dtk)
    judge_by_task = gather_cls_values_by_task_from_per_class(judge_by_cls)

    for chart_type, sfx in [("strip", "strip"), ("errorbar", "errorbar")]:
        plot_overall_distribution(
            cls_by_ds, dist_models, color_map, mm_set, "F1 Macro",
            f"Aug Vision — Classification Overall ({filter_label})",
            output_dir / f"cls_overall_{filter_label}_{sfx}.png",
            chart_type=chart_type,
        )
        plot_per_task_distribution(
            cls_by_task, task_types, dist_models, color_map, mm_set, _task_label,
            "F1 Macro", f"Aug Vision — Classification Per Task ({filter_label})",
            output_dir / f"cls_per_task_{filter_label}_{sfx}.png",
            chart_type=chart_type,
            judge_values_by_model_by_task=judge_by_task,
        )
        plot_per_dataset_distribution(
            cls_by_cls, cls_ds_tasks, dist_models, color_map, mm_set,
            "F1 per Class", f"Aug Vision — Classification Per Dataset ({filter_label})",
            output_dir / f"cls_per_dataset_{filter_label}_{sfx}.png",
            chart_type=chart_type,
            judge_values_by_model_by_dataset=judge_by_cls,
        )


def plot_desc_quality_correlation(
    aug_rows: list[dict],
    output_dir: Path,
    suffix: str = "vis_desc",
) -> None:
    """Scatter: per-class accuracy vs mean rouge_ss and mean sem_sim of generated descriptions."""
    from sklearn.metrics import f1_score as sk_f1

    by_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in aug_rows:
        by_class[(row["agml_dataset"], row["class_label"])].append(row)

    points = []
    for (ds, cls), rows in by_class.items():
        refs  = [r["class_label"] for r in rows]
        preds = [r.get("resolved_class", "") for r in rows]
        f1    = sk_f1(refs, preds, average="macro",
                      labels=sorted(set(refs)), zero_division=0)

        rs_vals  = [float(r["desc_rouge_ss"]) for r in rows
                    if r.get("desc_rouge_ss") not in ("", None)]
        ss_vals  = [float(r["desc_sem_sim"])  for r in rows
                    if r.get("desc_sem_sim")  not in ("", None)]
        if not rs_vals:
            continue
        points.append({
            "ds": ds, "cls": cls,
            "task_type": rows[0].get("task_type", "unknown"),
            "f1":       f1,
            "rouge_ss": float(np.mean(rs_vals)),
            "sem_sim":  float(np.mean(ss_vals)) if ss_vals else None,
        })

    if not points:
        print("  [skip] no data for desc quality correlation plot")
        return

    task_types  = sorted(set(p["task_type"] for p in points))
    task_colors = {tt: plt.cm.tab10.colors[i % 10] for i, tt in enumerate(task_types)}

    score_keys  = [("rouge_ss", "ROUGE-SS"), ("sem_sim", "Semantic Similarity")]
    fig, axes   = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (sk, xlabel) in zip(axes, score_keys):
        valid = [p for p in points if p.get(sk) is not None]
        if not valid:
            ax.set_visible(False)
            continue
        xs     = [p[sk] for p in valid]
        ys     = [p["f1"]    for p in valid]
        colors = [task_colors[p["task_type"]] for p in valid]

        ax.scatter(xs, ys, c=colors, alpha=0.75, s=70,
                   edgecolors="black", linewidths=0.5)

        corr = float(np.corrcoef(xs, ys)[0, 1])
        z    = np.polyfit(xs, ys, 1)
        x_ln = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_ln, np.poly1d(z)(x_ln), "k--", alpha=0.5, linewidth=1)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Per-class F1", fontsize=10)
        ax.set_title(f"F1 vs {xlabel}  (r = {corr:.3f}, n = {len(valid)} classes)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for tt in task_types:
            ax.scatter([], [], c=task_colors[tt], label=_task_label(tt), s=40)
        ax.legend(fontsize=7)

    fig.suptitle("Description Quality vs Classification F1\n(vision-generated descriptions)", fontsize=11)
    plt.tight_layout()
    path = output_dir / f"desc_quality_correlation_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_transitions(
    vis_dir: Path,
    aug_dir: Path,
    ds_to_task: dict[str, str],
    output_dir: Path,
    models: list[str],
    filter_label: str = "filtered",
) -> None:
    """Stacked bar: fixed / both_right / both_wrong / broken — overall, per task, per dataset."""
    CATS   = ["fixed", "both_right", "both_wrong", "broken"]
    LABELS = ["Fixed (✗→✓)", "Both correct", "Both wrong", "Broken (✓→✗)"]
    COLORS = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c"]

    for model_id in models:
        safe     = _safe_name(model_id)
        vis_file = vis_dir / f"cls_{safe}.csv"
        aug_file = aug_dir / f"aug_cls_{filter_label}_{safe}.csv"
        if not vis_file.exists() or not aug_file.exists():
            print(f"  [skip] missing CSVs for {model_id}")
            continue

        vis_rows = _read_csv(vis_file)
        aug_rows = _read_csv(aug_file)

        vis_map = {(r["agml_dataset"], r["class_label"], r["image_name"]): r for r in vis_rows}
        aug_map = {(r["agml_dataset"], r["class_label"], r["image_name"]): r for r in aug_rows}
        common  = set(vis_map) & set(aug_map)

        # Categorise each image
        task_counts: dict[str, dict[str, int]] = defaultdict(lambda: {c: 0 for c in CATS})
        ds_counts:   dict[str, dict[str, int]] = defaultdict(lambda: {c: 0 for c in CATS})
        overall: dict[str, int] = {c: 0 for c in CATS}

        for key in common:
            ds   = key[0]
            task = ds_to_task.get(ds, "unknown")
            vr   = vis_map[key]
            ar   = aug_map[key]
            v_ok = int(vr.get("judged_correct") or vr.get("correct") or 0)
            a_ok = int(ar.get("judged_correct") or ar.get("correct") or 0)
            cat  = (
                "fixed"      if not v_ok and     a_ok else
                "broken"     if     v_ok and not a_ok else
                "both_right" if     v_ok and     a_ok else
                "both_wrong"
            )
            overall[cat]         += 1
            task_counts[task][cat] += 1
            ds_counts[ds][cat]    += 1

        # ── Figure: 3 panels (overall | per task | per dataset) ──────────────
        task_order = sorted(task_counts)
        ds_order   = sorted(ds_counts, key=lambda d: -(ds_counts[d]["fixed"] - ds_counts[d]["broken"]))

        fig, axes = plt.subplots(
            1, 3,
            figsize=(max(22, len(ds_order) * 1.1 + 8), 6),
            gridspec_kw={"width_ratios": [1, len(task_order), len(ds_order)]},
        )

        def _stacked_bar(ax, groups: list[str], counts_map: dict, title: str, xlabel: str = "") -> None:
            x      = np.arange(len(groups))
            bottom = np.zeros(len(groups))
            for cat, label, color in zip(CATS, LABELS, COLORS):
                vals = np.array([counts_map[g][cat] for g in groups], dtype=float)
                bars = ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
                for bar, v in zip(bars, vals):
                    if v > max(sum(sum(counts_map[g].values()) for g in groups) / len(groups) * 0.05, 30):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f"{int(v)}", ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                        )
                bottom += vals
            totals = [sum(counts_map[g].values()) for g in groups]
            for i, (tot, net) in enumerate(zip(totals, [counts_map[g]["fixed"] - counts_map[g]["broken"] for g in groups])):
                ax.text(i, tot + totals[0] * 0.01, f"net {net:+d}", ha="center", va="bottom", fontsize=7, color="#333")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [g.replace("_classification", "").replace("_", " ") for g in groups],
                rotation=35 if len(groups) > 3 else 0, ha="right" if len(groups) > 3 else "center", fontsize=8,
            )
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Images")
            if xlabel:
                ax.set_xlabel(xlabel)

        _stacked_bar(axes[0], ["overall"], {"overall": overall}, "Overall")
        _stacked_bar(axes[1], task_order, task_counts, "By Task Type")
        _stacked_bar(axes[2], ds_order,   ds_counts,  "By Dataset")

        axes[0].legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"Vision → Augmented Vision Transitions\n{model_id.split('/')[-1]} "
            f"(judged-correct lang classes only, n={len(common):,})",
            fontsize=11,
        )
        plt.tight_layout()
        path = output_dir / f"transitions_{filter_label}_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def _read_baseline_rows_cached(output_dir: Path, model_id: str) -> list[dict]:
    path = output_dir / f"baseline_cls_{_safe_name(model_id)}.csv"
    return _read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description-Augmented Vision Assessment")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--lang_dir",    default=None,
                        help="language assessment output dir; only needed for filtered run (judged-correct classes)")
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
    parser.add_argument("--all_classes", action="store_true",
                        help="Use all classes with descriptions, not just judged-correct ones")
    parser.add_argument("--resume", action="store_true",
                        help="Skip inference/judging if output CSVs already exist")
    parser.add_argument("--use_vision_descriptions", action="store_true",
                        help="Sub-step: run augmented inference using per-image vision-generated "
                             "descriptions (from vis_dir desc CSV) and plot F1 vs rouge_ss/sem_sim")

    args = parser.parse_args()
    if not args.judge_model and args.models:
        args.judge_model = args.models[0]
    main(args)
