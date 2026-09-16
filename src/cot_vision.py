"""
Chain-of-Thought Vision Assessment
====================================
Evaluates two CoT prompting strategies on agricultural image classification:

  self_cot   — model generates its own visual description first (2-turn history),
               then classifies using the description as context.

  cot_yaml   — structured multi-step CoT from cot_prompts.yaml; each step is its
               own conversation turn, then final classification bridges from analysis.

Neither condition uses any external descriptions — the model reasons from the image.
Aug-vision (pre-written descriptions injected) is evaluated separately via
desc_augmented_vision.py.

Uses vLLM with staged batching: all images are collected upfront, then processed
one stage at a time (all images in parallel per stage). This avoids sequential
per-image generation while preserving the full multi-turn conversation history
for each image across stages.

Outputs per model:
  cot_cls_{model}.csv      — one row per (image, condition)
  cot_vision_metrics.json  — F1 / accuracy per condition × dataset

Usage:
    uv run python src/cot_vision.py \\
        --model      google/gemma-4-26B-A4B-it \\
        --output_dir outputs/cot_vision/ \\
        --test_split outputs/linear_probe/test_split.json

    # Regenerate plots only (no inference):
    uv run python src/cot_vision.py \\
        --model      google/gemma-4-26B-A4B-it \\
        --output_dir outputs/cot_vision/ \\
        --vis_dir    outputs/vision_assessment/ \\
        --plot_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.metrics import (
    apply_judge, compute_classification_metrics, contains_match,
    judged_summary, resolve_class, sample_class_images,
)
from src.common.dataset_config import datasets_by_name, clean_class_list
from src.common.difficulty_plots import plot_difficulty_correlation_conditions

try:
    from vllm import LLM, SamplingParams
    _COT_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64,  repetition_penalty=1.1)
except ImportError:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]
    _COT_PARAMS = _CLS_PARAMS = None

# ── Dataset routing (26 datasets from test_split.json) ────────────────────────
# Maps agml dataset name → (task_type, crop)
DATASET_TASK_TYPES: dict[str, tuple[str, str | None]] = {
    "arabica_coffee_leaf_disease_classification": ("disease",            "arabica_coffee"),
    "banana_leaf_disease_classification":         ("disease",            "banana"),
    "bean_disease_uganda":                        ("disease",            "bean"),
    "betel_leaf_disease_classification":          ("disease",            "betel_leaf"),
    "blackgram_plant_leaf_disease_classification":("disease",            "blackgram"),
    "chilli_leaf_classification":                 ("disease",            "chilli"),
    "coconut_tree_disease_classification":        ("disease",            "coconut"),
    "corn_maize_leaf_disease":                    ("disease",            "corn"),
    "crop_weeds_greece":                          ("weed_identification", None),
    "cucumber_disease_classification":            ("disease",            "cucumber"),
    "guava_disease_pakistan":                     ("disease",            "guava"),
    "java_plum_leaf_disease_classification":      ("disease",            "java_plum"),
    "onion_leaf_classification":                  ("disease",            "onion"),
    "orange_leaf_disease_classification":         ("disease",            "orange"),
    "paddy_disease_classification":               ("disease",            "paddy"),
    "papaya_leaf_disease_classification":         ("disease",            "papaya"),
    "plant_seedlings_aarhus":                     ("weed_identification", None),
    "rangeland_weeds_australia":                  ("weed_identification", None),
    "rice_leaf_disease_classification":           ("disease",            "rice"),
    "soybean_insect_classification":              ("pest_damage",        "soybean"),
    "soybean_weed_uav_brazil":                    ("weed_identification", None),
    "sugarcane_damage_usa":                       ("pest_damage",        "sugarcane"),
    "sunflower_disease_classification":           ("disease",            "sunflower"),
    "tea_leaf_disease_classification":            ("disease",            "tea"),
    "tomato_leaf_disease":                        ("disease",            "tomato"),
    "vine_virus_photo_dataset":                   ("disease",            "grape"),
}

DEFAULT_DATASETS = list(DATASET_TASK_TYPES.keys())

_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)


# ── Prompt config ──────────────────────────────────────────────────────────────

_PROMPTS_YAML     = Path(__file__).parent / "config" / "prompts.yaml"
_COT_PROMPTS_YAML = Path(__file__).parent / "config" / "cot_prompts.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


_PROMPTS     = _load_yaml(_PROMPTS_YAML)
_COT_PROMPTS = _load_yaml(_COT_PROMPTS_YAML)


def _sys_msg() -> list[dict]:
    s = _PROMPTS.get("assessment_system", "")
    return [{"role": "system", "content": s}] if s else []


def _cls_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        inner = _PROMPTS["visual_classification_weed"]
    else:
        inner = _PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    return _PROMPTS["assessment_classification"].format(question=inner)


def _desc_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        return _PROMPTS["visual_description_weed"]
    q = _PROMPTS["visual_description_disease"]
    return q.format(crop=crop or "plant") if "{crop}" in q else q


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def load_llm(
    model_id: str,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
) -> tuple:
    print(f"\n[cot_vision] Loading {model_id} via vLLM")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
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
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    m = _REPEAT_RE.search(text)
    return text[:m.start()].strip() if m else text


def _build_prompt(messages: list[dict], tokenizer, image: PILImage.Image) -> dict:
    """Build a vLLM multimodal prompt dict from a conversation history."""
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return {"prompt": text, "multi_modal_data": {"image": image}}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_classes(
    ds_name: str,
    n_samples: int | None,
    test_split: dict | None = None,
) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} restricted to test_split when provided."""
    import agml
    loader = agml.data.AgMLDataLoader(ds_name)

    class_name_map: list[str] | None = None
    for src in (loader, getattr(loader, "dataset_info", None)):
        if src is None:
            continue
        for attr in ("class_names", "class_labels", "classes"):
            val = getattr(src, attr, None)
            if val is not None:
                class_name_map = list(val)
                break
        if class_name_map is not None:
            break

    contents = loader.export_contents()
    by_class: dict[str, list[Path]] = {}
    for path, label in contents.items():
        if class_name_map is not None and isinstance(label, int):
            lbl = class_name_map[label] if label < len(class_name_map) else str(label)
        else:
            lbl = str(label) if not isinstance(label, str) else label
        by_class.setdefault(lbl, []).append(Path(str(path)))

    result: dict[str, list[Path]] = {}
    split_cls = (test_split or {}).get(ds_name, {})
    for cls, paths in by_class.items():
        if split_cls and cls in split_cls:
            # Use test_split JSON order (same as vision_assessment) for deterministic sampling
            allowed_ordered = split_cls[cls]
            path_by_name = {p.name: p for p in paths}
            paths = [path_by_name[n] for n in allowed_ordered if n in path_by_name]
        else:
            paths = sorted(paths, key=lambda p: p.name)
        names = sample_class_images([p.name for p in paths], n_samples)
        name_set = set(names)
        result[cls] = [p for p in paths if p.name in name_set]
    return result


# ── Staged batched inference ────────────────────────────────────────────────────

def collect_work_items(
    datasets: list[str],
    n_samples: int | None,
    test_split: dict | None,
) -> list[dict]:
    """Collect (dataset, class, image_path) tuples upfront. PIL images are loaded lazily."""
    ds_cfg_map = datasets_by_name(Path(__file__).parent / "config" / "datasets.yaml")
    items: list[dict] = []
    for ds in tqdm(datasets, desc="Datasets", unit="ds", dynamic_ncols=True):
        try:
            by_class = load_dataset_classes(ds, n_samples, test_split)
        except Exception as e:
            tqdm.write(f"  [skip] {ds}: {e}")
            continue
        primary_task_type, crop = DATASET_TASK_TYPES.get(ds, ("disease", "plant"))
        # Use DatasetConfig.clean_classes for resolve_class — same source as vision_assessment
        cfg = ds_cfg_map.get(ds)
        clean_classes = cfg.clean_classes if cfg else sorted(by_class.keys())
        for cls, paths in by_class.items():
            # Use effective_task_type per class (respects class_overrides, same as baseline)
            task_type = cfg.effective_task_type(cls) if cfg else primary_task_type
            tqdm.write(f"  {ds} / {cls}  ({len(paths)} images)")
            for path in paths:
                items.append({
                    "ds":           ds,
                    "class_label":  cls,
                    "task_type":    task_type,
                    "crop":         crop,
                    "clean_classes": clean_classes,
                    "image_name":   path.name,
                    "image_path":   path,
                })
    print(f"  Collected {len(items)} work items across {len(datasets)} datasets")
    return items


def _load_image(item: dict) -> PILImage.Image:
    return PILImage.open(item["image_path"]).convert("RGB")


def _make_row(item: dict, condition: str, pred: str) -> dict:
    resolved = resolve_class(pred, item["clean_classes"])
    correct  = (resolved or "").lower() == item["class_label"].lower()
    return {
        "agml_dataset":   item["ds"],
        "class_label":    item["class_label"],
        "task_type":      item["task_type"],
        "crop":           item["crop"] or "",
        "image_name":     item["image_name"],
        "condition":      condition,
        "prediction":     pred,
        "resolved_class": resolved or "",
        "ground_truth":   item["class_label"],
        "correct":        int(correct),
        "contains_match": int(contains_match(resolved or pred, item["class_label"])),
    }


def run_self_cot(items: list[dict], llm, tokenizer, prompt_batch_size: int = 256) -> list[dict]:
    """
    Staged batching with mini-batches to cap peak CPU RAM.
    Stage 1 (describe) runs in chunks, then Stage 2 (classify) runs in chunks.
    PIL images are loaded lazily per chunk — never held across stages.
    """
    # Stage 1 — describe all images in mini-batches
    print(f"\n[self_cot] Stage 1/2: describing ({len(items)} images, batch={prompt_batch_size})...")
    descriptions: list[str] = []
    for start in range(0, len(items), prompt_batch_size):
        batch = items[start:start + prompt_batch_size]
        prompts = []
        for item in tqdm(batch, desc=f"  Building prompts (stage 1, chunk {start//prompt_batch_size+1})",
                         unit="img", dynamic_ncols=True):
            desc_q = _desc_question(item["task_type"], item["crop"])
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc_q},
            ]}]
            prompts.append(_build_prompt(msgs, tokenizer, _load_image(item)))
        descriptions.extend(_clean(o.outputs[0].text) for o in llm.generate(prompts, _COT_PARAMS))
        del prompts

    # Stage 2 — classify in mini-batches using stage-1 descriptions
    print(f"[self_cot] Stage 2/2: classifying ({len(items)} images, batch={prompt_batch_size})...")
    predictions: list[str] = []
    for start in range(0, len(items), prompt_batch_size):
        batch_items = items[start:start + prompt_batch_size]
        batch_descs = descriptions[start:start + prompt_batch_size]
        prompts = []
        for item, desc in tqdm(zip(batch_items, batch_descs),
                               desc=f"  Building prompts (stage 2, chunk {start//prompt_batch_size+1})",
                               unit="img", total=len(batch_items), dynamic_ncols=True):
            desc_q = _desc_question(item["task_type"], item["crop"])
            cls_q  = _cls_question(item["task_type"], item["crop"])
            msgs = _sys_msg() + [
                {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": desc_q}]},
                {"role": "assistant", "content": desc},
                {"role": "user",      "content": cls_q},
            ]
            prompts.append(_build_prompt(msgs, tokenizer, _load_image(item)))
        predictions.extend(_clean(o.outputs[0].text) for o in llm.generate(prompts, _CLS_PARAMS))
        del prompts

    return [_make_row(item, "self_cot", pred) for item, pred in zip(items, predictions)]


def run_cot_yaml(
    items: list[dict],
    llm,
    tokenizer,
    prompt_batch_size: int = 256,
    ckpt_dir: Path | None = None,
) -> list[dict]:
    """
    Staged batching split by CoT path (disease_pest vs weed_identification).
    Each group is checkpointed independently to ckpt_dir/ckpt_cot_yaml_{group}.csv
    so a rerun can skip completed groups.
    """
    groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, item in enumerate(items):
        cot_key = _COT_PROMPTS["task_type_to_cot"].get(item["task_type"], "disease_pest")
        groups.setdefault(cot_key, []).append((idx, item))

    all_rows: list[tuple[int, dict]] = []

    for cot_key, indexed_items in groups.items():
        # Per-group checkpoint
        ckpt_group = (ckpt_dir / f"ckpt_cot_yaml_{cot_key}.csv") if ckpt_dir else None
        if ckpt_group and ckpt_group.exists():
            print(f"\n[cot_yaml] {cot_key}: SKIPPED — checkpoint found ({ckpt_group.name})")
            group_rows = list(csv.DictReader(open(ckpt_group, encoding="utf-8")))
            for i, (orig_idx, _) in enumerate(indexed_items):
                if i < len(group_rows):
                    all_rows.append((orig_idx, group_rows[i]))
            continue

        steps       = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        orig_idxs   = [idx for idx, _ in indexed_items]
        group_items = [item for _, item in indexed_items]
        print(f"\n[cot_yaml] {cot_key}: {len(group_items)} images × {len(steps)} steps (batch={prompt_batch_size})")

        histories: list[list[dict]] = [
            _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": steps[0]},
            ]}]
            for _ in group_items
        ]

        for step_idx in range(len(steps)):
            print(f"  Step {step_idx + 1}/{len(steps)}: {len(group_items)} prompts...")
            responses: list[str] = []
            for start in range(0, len(group_items), prompt_batch_size):
                batch_items = group_items[start:start + prompt_batch_size]
                batch_hist  = histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in tqdm(zip(batch_hist, batch_items),
                                               desc=f"    Prompts step {step_idx+1} chunk {start//prompt_batch_size+1}",
                                               unit="img", total=len(batch_items), dynamic_ncols=True)]
                responses.extend(_clean(o.outputs[0].text) for o in llm.generate(prompts, _COT_PARAMS))
                del prompts
            for i, resp in enumerate(responses):
                histories[i].append({"role": "assistant", "content": resp})
                if step_idx < len(steps) - 1:
                    histories[i].append({"role": "user", "content": steps[step_idx + 1]})

        # Classification turn
        print(f"  Classification: {len(group_items)} prompts...")
        for i, item in enumerate(group_items):
            histories[i].append({"role": "user", "content": _cls_question(item["task_type"], item["crop"])})
        predictions: list[str] = []
        for start in range(0, len(group_items), prompt_batch_size):
            batch_items = group_items[start:start + prompt_batch_size]
            batch_hist  = histories[start:start + prompt_batch_size]
            cls_prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in tqdm(zip(batch_hist, batch_items),
                                               desc=f"    Prompts cls chunk {start//prompt_batch_size+1}",
                                               unit="img", total=len(batch_items), dynamic_ncols=True)]
            predictions.extend(_clean(o.outputs[0].text) for o in llm.generate(cls_prompts, _CLS_PARAMS))
            del cls_prompts

        # Extract intermediate step responses from histories for the reasoning column.
        # Each history at this point is: [sys, user(step0+img), asst(resp0), user(step1), asst(resp1), ...]
        # Assistant turns are every other message starting at index 2.
        group_result = []
        for item, pred, hist in zip(group_items, predictions, histories):
            asst_responses = [m["content"] for m in hist if m["role"] == "assistant"]
            reasoning = "\n\n".join(
                f"Q: {steps[i]}\nA: {asst_responses[i]}"
                for i in range(min(len(steps), len(asst_responses)))
            )
            row = _make_row(item, "cot_yaml", pred)
            row["reasoning"] = reasoning
            group_result.append(row)

        # Save group checkpoint immediately
        if ckpt_group:
            write_cls_csv(group_result, ckpt_group)
            print(f"  Checkpoint saved: {ckpt_group.name}")

        for orig_idx, row in zip(orig_idxs, group_result):
            all_rows.append((orig_idx, row))

    all_rows.sort(key=lambda x: x[0])
    return [row for _, row in all_rows]




# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """Compute direct and judged F1 per (condition, dataset) and overall per condition.
    Matches assess.py: resolved_class as prediction (not sentinel), judged_summary for judged F1.
    """
    from collections import defaultdict
    by_cond_ds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_ds[(r["condition"], r["agml_dataset"])].append(r)

    per_condition: dict[str, dict] = {}
    for (cond, ds), ds_rows in sorted(by_cond_ds.items()):
        refs  = [r["class_label"]    for r in ds_rows]
        preds = [r["resolved_class"] for r in ds_rows]
        m     = compute_classification_metrics(refs, preds)
        m.update(judged_summary(ds_rows))
        per_condition.setdefault(cond, {})[ds] = m

    overall: dict[str, dict] = {}
    for cond, per_ds in per_condition.items():
        f1s  = [v["f1_macro"]       for v in per_ds.values()]
        jf1s = [v["judge_f1_macro"] for v in per_ds.values() if "judge_f1_macro" in v]
        overall[cond] = {
            "f1_macro":       float(np.mean(f1s))  if f1s  else 0.0,
            "judge_f1_macro": float(np.mean(jf1s)) if jf1s else None,
            "per_dataset":    per_ds,
        }

    return overall


# ── CSV / JSON I/O ─────────────────────────────────────────────────────────────

_CLS_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "condition", "prediction", "reasoning", "resolved_class", "ground_truth",
    "correct", "contains_match", "judged_correct", "judge_reasoning",
]


def write_cls_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CLS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def select_hard_datasets(
    vis_per_ds: dict[str, float],
    n_per_task: int = 3,
) -> list[str]:
    """Return the n worst-performing datasets per task type by vision judged F1.

    Disease datasets with class_overrides to pest_damage are counted in BOTH
    the disease AND pest_damage groups, since they contain meaningful pest classes
    and would otherwise leave pest_damage under-represented (only 2 pure datasets).
    """
    from collections import defaultdict
    import yaml as _yaml

    datasets_yaml = Path(__file__).parent / "config" / "datasets.yaml"
    cfg = _yaml.safe_load(datasets_yaml.read_text())

    # Build {ds_name: set of task_types it belongs to}
    task_groups: dict[str, set[str]] = defaultdict(set)
    for entry in cfg.get("datasets", []):
        name = entry.get("agml_name", "")
        if name not in DATASET_TASK_TYPES:
            continue
        primary = entry.get("task_type", DATASET_TASK_TYPES[name][0])
        task_groups[name].add(primary)
        for override_task in (entry.get("class_overrides") or {}).values():
            task_groups[name].add(override_task)

    # Group datasets by task, sorted hardest first (lowest F1)
    by_task: dict[str, list[str]] = defaultdict(list)
    for ds, tasks in task_groups.items():
        if ds not in vis_per_ds:
            continue
        for task in tasks:
            by_task[task].append(ds)

    selected: set[str] = set()
    for task, ds_list in by_task.items():
        sorted_ds = sorted(ds_list, key=lambda d: vis_per_ds.get(d, 1.0))
        worst = sorted_ds[:n_per_task]
        selected.update(worst)
        f1s = [f"{d.split('_')[0]}({vis_per_ds.get(d, 0):.2f})" for d in worst]
        print(f"  [{task}] hardest {len(worst)}: {', '.join(f1s)}")

    return [ds for ds in DEFAULT_DATASETS if ds in selected]


def load_aug_per_dataset(aug_dir: Path, model_id: str) -> dict[str, float]:
    """Load per-dataset augmented judged F1 from desc_augmented_metrics_all.json.

    Uses judge_f1_macro where available, falls back to f1_macro — consistent
    with load_vis_per_dataset which also returns judged F1.
    """
    for label in ("all", "filtered"):
        path = aug_dir / f"desc_augmented_metrics_{label}.json"
        if path.exists():
            data = json.loads(path.read_text())
            model_data = data.get("models", {}).get(model_id, {})
            per_ds = model_data.get("augmented_per_dataset", {})
            return {ds: v.get("judge_f1_macro") or v.get("f1_macro", 0.0)
                    for ds, v in per_ds.items()}
    print(f"  [warn] No desc_augmented_metrics JSON found in {aug_dir}")
    return {}


def _f1_from_csv_subset(
    csv_path: Path,
    allowed: dict[str, set[tuple[str, str]]],  # {ds: {(image_name, class_label), ...}}
    judged: bool = True,
) -> dict[str, float]:
    """Recompute per-dataset macro F1 from a raw cls CSV, restricted to allowed (image_name, class_label) pairs."""
    import csv as _csv
    from collections import defaultdict
    from sklearn.metrics import f1_score

    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ds = r.get("agml_dataset", "")
        if ds in allowed and (r.get("image_name", ""), r.get("class_label", "")) in allowed[ds]:
            by_ds[ds].append(r)

    result: dict[str, float] = {}
    for ds, ds_rows in by_ds.items():
        if judged:
            ds_rows = [r for r in ds_rows if r.get("judged_correct", "") != ""]
            if not ds_rows:
                continue
            refs  = [r["class_label"] for r in ds_rows]
            preds = [r["class_label"] if int(r["judged_correct"]) else "" for r in ds_rows]
        else:
            refs  = [r.get("ground_truth", r.get("class_label", "")) for r in ds_rows]
            preds = [r.get("resolved_class", r.get("prediction", "")) for r in ds_rows]
        result[ds] = f1_score(refs, preds, average="macro", zero_division=0)
    return result


def plot_hard_four_condition(
    active_datasets: list[str],
    vis_per_ds: dict[str, float],
    aug_per_ds: dict[str, float],
    cot_metrics: dict,
    output_dir: Path,
    vis_dir: Path | None = None,
    aug_dir: Path | None = None,
    model_id: str = "",
    n_samples: int | None = None,
    mv_metrics: dict | None = None,
) -> None:
    """Grouped bar chart comparing conditions on the hard-subset datasets only.

    When n_samples is set, baseline and desc_aug scores are recomputed on the exact
    same images used by cot_vision (fair subsample comparison).
    mv_metrics adds one bar per majority-vote N value.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    datasets = [ds for ds in active_datasets if ds in vis_per_ds]
    if not datasets:
        print("  [warn] No common datasets for hard 4-condition chart")
        return

    # When subsampling, recompute baseline + desc_aug on the cot-evaluated images
    _vis_per_ds  = vis_per_ds
    _aug_per_ds  = aug_per_ds
    if n_samples is not None:
        cot_csv = output_dir / f"cot_cls_{model_id.replace('/', '_')}.csv"
        if cot_csv.exists():
            import csv as _csv
            allowed: dict[str, set[tuple[str, str]]] = defaultdict(set)
            with open(cot_csv, newline="", encoding="utf-8") as f:
                for r in _csv.DictReader(f):
                    allowed[r["agml_dataset"]].add((r["image_name"], r["class_label"]))
            if vis_dir and model_id:
                safe = model_id.replace("/", "_")
                recomputed = _f1_from_csv_subset(
                    vis_dir / f"cls_{safe}.csv", allowed, judged=True)
                if recomputed:
                    _vis_per_ds = recomputed
                    print(f"  [subsample] Recomputed baseline F1 on {sum(len(v) for v in allowed.values())} cot images")
            if aug_dir and model_id:
                safe = model_id.replace("/", "_")
                aug_csv = aug_dir / f"aug_cls_all_{safe}.csv"
                recomputed = _f1_from_csv_subset(aug_csv, allowed, judged=True)
                if recomputed:
                    _aug_per_ds = recomputed
                    print(f"  [subsample] Recomputed desc_aug F1 on subsample")

    # Base conditions
    CONDITIONS = [
        ("baseline",  "Vision Baseline",  "#1f77b4"),
        ("self_cot",  "Self-CoT",         "#ff7f0e"),
        ("cot_yaml",  "Structured CoT",   "#2ca02c"),
        ("desc_aug",  "Desc-Aug",         "#9467bd"),
    ]

    # Add MV bars: baseline, self_cot, cot_yaml — each N sorted ascending
    _mv = mv_metrics or {}
    _MV_SERIES = [
        # (condition prefix, label prefix, color palette dark→light)
        ("mv_cot_yaml_", "CoT-YAML MV N=", ["#1a7a1a", "#4db34d", "#99d699"]),
        ("mv_self_cot_", "Self-CoT MV N=", ["#7a1a7a", "#b34db3", "#d699d6"]),
        ("mv_",          "Baseline MV N=", ["#d62728", "#e8846a", "#f5c5b0"]),
    ]
    for prefix, label_prefix, palette in _MV_SERIES:
        ns = sorted(
            int(k.split("_")[-1]) for k in _mv
            if k.startswith(prefix)
            and (prefix == "mv_" and not k.startswith(("mv_self_cot_", "mv_cot_yaml_"))
                 or prefix != "mv_")
        )
        for n, color in zip(ns, palette):
            CONDITIONS.append((f"{prefix}{n}", f"{label_prefix}{n}", color))

    def _get(ds: str, key: str) -> float:
        if key == "baseline":
            return _vis_per_ds.get(ds, 0.0)
        if key == "desc_aug":
            return _aug_per_ds.get(ds, 0.0)
        src = _mv if key.startswith("mv_") else cot_metrics
        per_ds = src.get(key, {}).get("per_dataset", {})
        dm = per_ds.get(ds, {})
        return dm.get("judge_f1_macro") or dm.get("f1_macro", 0.0)

    x = np.arange(len(datasets))
    width = 0.8 / len(CONDITIONS)
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.8 + len(CONDITIONS) * 0.3), 5))

    for i, (key, label, color) in enumerate(CONDITIONS):
        vals = [_get(ds, key) for ds in datasets]
        offset = (i - len(CONDITIONS) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    n_conds = len(CONDITIONS)
    clean = [d.replace("_classification", "").replace("_", " ") for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(clean, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Judged F1  (Desc-Aug = direct F1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{n_conds}-Condition Comparison — Hard Datasets Only (3 worst per task)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "hard_four_condition_subsample.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_vis_per_dataset(vis_dir: Path, model_id: str) -> dict[str, float]:
    """Load per-dataset judged F1 from vision assessment cls CSV as difficulty proxy."""
    import csv as _csv
    from collections import defaultdict
    safe = model_id.replace("/", "_")
    cls_file = vis_dir / f"cls_{safe}.csv"
    if not cls_file.exists():
        print(f"  [warn] vis cls CSV not found: {cls_file}")
        return {}
    with open(cls_file, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r.get("agml_dataset", "")].append(r)
    result: dict[str, float] = {}
    for ds, ds_rows in by_ds.items():
        judged = [r for r in ds_rows if r.get("judged_correct", "") != ""]
        if judged:
            refs  = [r["class_label"] for r in judged]
            preds = [r["class_label"] if int(r["judged_correct"]) else "" for r in judged]
            result[ds] = compute_classification_metrics(refs, preds).get("f1_macro", 0.0)
    return result


def plot_difficulty_correlation(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    output_dir: Path,
    vis_rows: list[dict] | None = None,
    aug_rows: list[dict] | None = None,
    cot_rows: list[dict] | None = None,
) -> None:
    """Scatter: vision baseline judged F1 (x) vs CoT judged F1 (y) per dataset."""
    from src.common.method_comparison import plot_baseline_vs_condition_scatter

    rows_by_cond = _build_rows_by_cond(cot_metrics, vis_rows, aug_rows, cot_rows)
    if not rows_by_cond or "Vision Baseline" not in rows_by_cond:
        return

    plot_baseline_vs_condition_scatter(
        rows_by_cond,
        output_path=output_dir / "cot_difficulty_correlation.png",
        title="Difficulty vs CoT Performance (per dataset)",
        separate=False,
    )
    plot_baseline_vs_condition_scatter(
        rows_by_cond,
        output_path=output_dir / "cot_difficulty_correlation_separate.png",
        title="Difficulty vs CoT Performance (per dataset)",
        separate=True,
    )


def _judged_f1_per_ds_from_rows(rows: list[dict]) -> dict[str, float]:
    """Return {dataset: mean_judged_f1} using gather_judge_per_class_by_dataset.

    Mirrors the computation in plot_conditions_per_task — mean of per-class judge F1
    within each dataset. Returns only datasets that have judged rows.
    """
    from src.common.distribution_plots import gather_judge_per_class_by_dataset
    judge_by_cls = gather_judge_per_class_by_dataset({"_": rows})
    per_dt = judge_by_cls.get("_", {})
    return {ds: float(np.mean(f1s)) for (ds, _task), f1s in per_dt.items() if f1s}


def _plot_difficulty_bins_impl(
    rows_by_cond: dict[str, list[dict]],
    difficulty_per_ds: dict[str, float],
    output_dir: Path,
    title: str,
    filename: str,
    difficulty_label: str,
) -> None:
    """Bar chart: mean judged F1 per difficulty tercile, binned by difficulty_per_ds.

    Uses raw rows for all conditions so the metric is always judged F1 from
    judged_correct — never direct match F1. Conditions with no judged rows are
    silently skipped, so partial checkpoint runs only show what is available.
    """
    import matplotlib.pyplot as plt
    from src.common.method_comparison import CONDITION_COLOR, CONDITION_STYLES

    if not difficulty_per_ds:
        return

    # Compute per-dataset judged F1 for each condition from raw rows
    cond_per_ds: dict[str, dict[str, float]] = {}
    for label, rows in rows_by_cond.items():
        per_ds = _judged_f1_per_ds_from_rows(rows)
        if per_ds:
            cond_per_ds[label] = per_ds

    order   = [lbl for lbl, _ in CONDITION_STYLES if lbl in cond_per_ds]
    present = order
    if not present:
        print(f"  [warn] {filename}: no conditions with judged rows — skipping")
        return

    # Bin datasets by difficulty (ascending = hard first)
    all_ds = sorted(difficulty_per_ds, key=difficulty_per_ds.get)
    n = len(all_ds)
    bins = {"Hard": all_ds[:n//3], "Medium": all_ds[n//3:2*n//3], "Easy": all_ds[2*n//3:]}

    x     = np.arange(len(bins))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, cond in enumerate(present):
        per_ds = cond_per_ds[cond]
        means  = [float(np.mean([per_ds[d] for d in bin_ds if d in per_ds]))
                  if any(d in per_ds for d in bin_ds) else 0.0
                  for bin_ds in bins.values()]
        offset = (i - len(present) / 2 + 0.5) * width
        color  = CONDITION_COLOR.get(cond, "#888888")
        bars   = ax.bar(x + offset, means, width, label=cond,
                        color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, means):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(bins))
    ax.set_ylabel("Mean Judged F1")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xlabel(f"Difficulty bin (by {difficulty_label}, low → hard)")
    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _build_rows_by_cond(
    cot_metrics: dict,
    vis_rows: list[dict] | None,
    aug_rows: list[dict] | None,
    cot_rows: list[dict] | None,
) -> dict[str, list[dict]]:
    """Build {condition_label: rows} for all available conditions with judged data."""
    rows_by_cond: dict[str, list[dict]] = {}
    if vis_rows:
        rows_by_cond["Vision Baseline"] = vis_rows
    for cot_key, label in [("self_cot", "Self-CoT"), ("cot_yaml", "CoT YAML")]:
        if cot_key in cot_metrics:
            filtered = [r for r in (cot_rows or []) if r.get("condition") == cot_key
                        and r.get("judged_correct", "") != ""]
            if filtered:
                rows_by_cond[label] = filtered
    if aug_rows:
        rows_by_cond["Desc-Aug"] = aug_rows
    return rows_by_cond


def plot_difficulty_bins(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    output_dir: Path,
    vis_rows: list[dict] | None = None,
    aug_rows: list[dict] | None = None,
    cot_rows: list[dict] | None = None,
) -> None:
    """Bar chart binned by vision baseline judged F1 (supervised difficulty proxy)."""
    rows_by_cond = _build_rows_by_cond(cot_metrics, vis_rows, aug_rows, cot_rows)
    _plot_difficulty_bins_impl(
        rows_by_cond, vis_per_ds, output_dir,
        title="CoT Performance by Difficulty Bin (Vision F1)",
        filename="cot_difficulty_bins_f1.png",
        difficulty_label="Vision Baseline Judged F1",
    )


def load_clustering_per_dataset(
    path: Path,
    backbone: str = "dinov3-vitb16-pretrain-lvd1689m",
) -> dict[str, dict[str, float]]:
    """Load clustering metrics per dataset for a given backbone.

    Returns {dataset_name: {metric: value}} for metrics:
    nmi, ari, silhouette, inter_centroid_dist, fisher_dr.
    Higher values = more separable = easier.
    """
    import json as _json
    if not path.exists():
        print(f"  [warn] Clustering metrics not found: {path}")
        return {}
    with open(path) as f:
        data = _json.load(f)
    backbone_data = data.get(backbone)
    if backbone_data is None:
        available = list(data.keys())
        print(f"  [warn] Backbone '{backbone}' not in clustering metrics. Available: {available}")
        return {}
    return {ds: {k: v for k, v in m.items()
                 if k in ("nmi", "ari", "silhouette", "inter_centroid_dist", "fisher_dr")}
            for ds, m in backbone_data.items()}


def plot_difficulty_bins_clustering(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    clustering_per_ds: dict[str, dict[str, float]],
    output_dir: Path,
    backbone: str,
    vis_rows: list[dict] | None = None,
    aug_rows: list[dict] | None = None,
    cot_rows: list[dict] | None = None,
) -> None:
    """One difficulty-bins chart per clustering metric (label-free difficulty proxies)."""
    metric_labels = {
        "nmi":               "NMI",
        "ari":               "ARI",
        "silhouette":        "Silhouette",
        "inter_centroid_dist": "Inter-centroid Distance",
        "fisher_dr":         "Fisher Discriminant Ratio",
    }
    rows_by_cond  = _build_rows_by_cond(cot_metrics, vis_rows, aug_rows, cot_rows)
    backbone_short = backbone.split("/")[-1] if "/" in backbone else backbone
    for metric, label in metric_labels.items():
        diff = {ds: vals[metric] for ds, vals in clustering_per_ds.items() if metric in vals}
        if not diff:
            continue
        _plot_difficulty_bins_impl(
            rows_by_cond, diff, output_dir,
            title=f"CoT Performance by Difficulty Bin ({label}, {backbone_short})",
            filename=f"cot_difficulty_bins_{metric}.png",
            difficulty_label=label,
        )


from src.common.method_comparison import CONDITION_STYLES as _CONDITION_STYLE


def _f1(m: dict) -> float:
    return m.get("judge_f1_macro") or m.get("f1_macro", 0.0)


def _build_condition_map(
    metrics: dict,
    vis_per_ds: dict[str, float] | None,
    aug_per_ds: dict[str, float] | None,
) -> dict[str, dict[str, float]]:
    """Return {condition_label: {dataset: f1}} for all available conditions."""
    cond_map: dict[str, dict[str, float]] = {}
    if vis_per_ds:
        cond_map["Vision Baseline"] = dict(vis_per_ds)
    for cot_key, label in [("self_cot", "Self-CoT"), ("cot_yaml", "CoT YAML")]:
        if cot_key in metrics:
            cond_map[label] = {
                ds: _f1(m)
                for ds, m in metrics[cot_key]["per_dataset"].items()
            }
    if aug_per_ds:
        cond_map["Desc-Aug"] = dict(aug_per_ds)
    return cond_map


def _judge_f1_per_class_from_rows(rows: list[dict]) -> dict[tuple[str, str, str], float]:
    """Returns {(ds, task, class_label): judge_f1} — mirrors gather_judge_per_class_by_dataset."""
    from collections import defaultdict as _defaultdict
    from sklearn.metrics import f1_score as _sk_f1
    by_key: dict[tuple, list[dict]] = _defaultdict(list)
    for r in rows:
        ds   = r.get("agml_dataset", "")
        task = r.get("task_type", "")
        cls  = r.get("class_label", "")
        if not ds or not cls or r.get("judged_correct", "") == "":
            continue
        by_key[(ds, task, cls)].append(r)
    result: dict[tuple, float] = {}
    for (ds, task, cls), cls_rows in by_key.items():
        refs  = [r["class_label"] for r in cls_rows]
        preds = [r["class_label"] if int(r["judged_correct"]) else "" for r in cls_rows]
        result[(ds, task, cls)] = _sk_f1(refs, preds, average="macro", zero_division=0)
    return result


def _task_means_from_cls_map(
    cls_map: dict[tuple[str, str, str], float],
    task_types: list[str],
) -> tuple[list[float], list[int]]:
    """Mean judge F1 per task and n (number of classes) from a per-class map."""
    by_task: dict[str, list[float]] = {tt: [] for tt in task_types}
    for (ds, task, cls), f1 in cls_map.items():
        if task in by_task:
            by_task[task].append(f1)
    means = [float(np.mean(by_task[tt])) if by_task[tt] else 0.0 for tt in task_types]
    ns    = [len(by_task[tt]) for tt in task_types]
    return means, ns


def plot_conditions_per_task(
    metrics: dict,
    vis_rows: list[dict] | None,
    aug_rows: list[dict] | None,
    output_dir: Path,
    cot_rows: list[dict] | None = None,
) -> None:
    """Grouped bar charts: judged F1 and direct+judged F1 per condition × task type."""
    from src.common.method_comparison import (
        plot_per_task_full_comparison,
        plot_per_task_judge_comparison,
        plot_per_task_delta,
        plot_per_dataset_delta,
    )

    rows_by_cond: dict[str, list[dict]] = {}
    if vis_rows:
        rows_by_cond["Vision Baseline"] = vis_rows
    for cot_key, label in [("self_cot", "Self-CoT"), ("cot_yaml", "CoT YAML")]:
        if cot_key not in metrics:
            continue
        filtered = [r for r in (cot_rows or []) if r.get("condition") == cot_key
                    and r.get("judged_correct", "") != ""]
        if filtered:
            rows_by_cond[label] = filtered
    if aug_rows:
        rows_by_cond["Desc-Aug"] = aug_rows

    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "cot_per_task_comparison.png",
        title="CoT Assessment — Mean Judged F1 by Task Type",
    )
    plot_per_task_full_comparison(
        rows_by_cond,
        output_path=output_dir / "cot_per_task_full_comparison.png",
        title="CoT Assessment — Direct vs Judged F1 by Task Type",
    )
    plot_per_task_delta(
        rows_by_cond,
        output_path=output_dir / "cot_per_task_delta.png",
        title="CoT Assessment — ΔF1 by Task Type (vs Vision Baseline)",
    )
    plot_per_task_delta(
        rows_by_cond,
        output_path=output_dir / "cot_per_task_delta_errorbar.png",
        title="CoT Assessment — ΔF1 by Task Type (vs Vision Baseline)",
        show_error_bars=True,
    )
    plot_per_dataset_delta(
        rows_by_cond,
        output_path=output_dir / "cot_per_dataset_delta.png",
        title="CoT Assessment — ΔJudged F1 per Dataset (sorted by method score)",
    )
    save_method_stats_table(rows_by_cond, output_dir)


def save_method_stats_table(
    rows_by_cond: dict[str, list[dict]],
    output_dir: Path,
    baseline_label: str = "Vision Baseline",
    threshold: float = 0.02,
) -> None:
    """Compute per-method dataset-level stats vs baseline and save CSV + PNG table."""
    import csv as _csv
    import matplotlib.pyplot as plt
    from src.common.distribution_plots import gather_judge_per_class_by_dataset

    # Build {condition: {dataset: mean_judged_f1}}
    ds_f1: dict[str, dict[str, float]] = {}
    for cond, rows in rows_by_cond.items():
        by_cls = gather_judge_per_class_by_dataset({cond: rows})
        per_dt = by_cls.get(cond, {})
        ds_f1[cond] = {ds: float(np.mean(f1s)) for (ds, _t), f1s in per_dt.items() if f1s}

    baseline = ds_f1.get(baseline_label, {})
    conditions = [c for c in rows_by_cond if c != baseline_label]
    all_ds = sorted(baseline.keys())

    rows_out = []
    for cond in conditions:
        method = ds_f1.get(cond, {})
        deltas = [method[ds] - baseline[ds] for ds in all_ds if ds in method and ds in baseline]
        if not deltas:
            continue
        gains     = [d for d in deltas if d > 0]
        regressions = [d for d in deltas if d < 0]
        n_total   = len(deltas)
        n_wins    = len(gains)
        n_losses  = len(regressions)
        win_rate  = n_wins / n_total if n_total else 0.0
        wl_ratio  = n_wins / n_losses if n_losses else float("inf")
        mean_gain = float(np.mean(gains)) if gains else 0.0
        mean_loss = float(np.mean(regressions)) if regressions else 0.0
        updown    = mean_gain / abs(mean_loss) if mean_loss != 0 else float("inf")
        median_d  = float(np.median(deltas))
        consistency = sum(1 for d in deltas if d > threshold) / n_total if n_total else 0.0
        rows_out.append({
            "Method":             cond,
            "Datasets":           n_total,
            "Wins":               n_wins,
            "Losses":             n_losses,
            "Win Rate":           round(win_rate, 3),
            "Win/Loss Ratio":     round(wl_ratio, 2) if wl_ratio != float("inf") else "∞",
            "Mean Gain":          round(mean_gain, 4),
            "Mean Loss":          round(mean_loss, 4),
            "Upside/Downside":    round(updown, 2) if updown != float("inf") else "∞",
            "Median ΔF1":         round(median_d, 4),
            f"Consistency (>{int(threshold*100)}% gain)": round(consistency, 3),
        })

    if not rows_out:
        return

    # Save CSV
    csv_path = output_dir / "method_stats_table.csv"
    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"  Saved: {csv_path}")

    # Save PNG table
    fig, ax = plt.subplots(figsize=(14, 0.55 * (len(rows_out) + 2)))
    ax.axis("off")
    col_labels = fieldnames
    cell_text  = [[str(r[c]) for c in fieldnames] for r in rows_out]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#cccccc")
    ax.set_title(
        f"Method Comparison Stats vs {baseline_label}  (threshold={int(threshold*100)}%)",
        fontsize=11, fontweight="bold", pad=12,
    )
    png_path = output_dir / "method_stats_table.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


def plot_conditions_overall(
    metrics: dict,
    vis_per_ds: dict[str, float] | None,
    aug_per_ds: dict[str, float] | None,
    output_dir: Path,
    vis_rows: list[dict] | None = None,
    aug_rows: list[dict] | None = None,
    cot_rows: list[dict] | None = None,
) -> None:
    """Single bar per condition: mean judged F1 across all datasets."""
    from src.common.method_comparison import plot_overall_judge_comparison

    rows_by_cond = _build_rows_by_cond(metrics, vis_rows, aug_rows, cot_rows)
    if not rows_by_cond:
        return
    plot_overall_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "cot_overall_comparison.png",
        title="CoT Assessment — Overall Mean Judged F1",
    )


def plot_delta_per_dataset(
    metrics: dict,
    vis_per_ds: dict[str, float],
    aug_per_ds: dict[str, float],
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """
    For each method (desc_aug, self_cot, cot_yaml), produce two horizontal bar charts:
      - top N datasets with the largest positive delta vs baseline (judged F1)
      - top N datasets with the largest negative delta vs baseline (judged F1)
    Saves 2 * n_methods plots into output_dir.
    """
    import matplotlib.pyplot as plt

    method_f1: dict[str, dict[str, float]] = {}
    if aug_per_ds:
        method_f1["desc_aug"] = aug_per_ds
    for cond in ("self_cot", "cot_yaml"):
        if cond in metrics:
            method_f1[cond] = {
                ds: m.get("judge_f1_macro") or m.get("f1_macro", 0.0)
                for ds, m in metrics[cond].get("per_dataset", {}).items()
            }

    if not vis_per_ds or not method_f1:
        print("  [delta plot] missing baseline or method data — skipping")
        return

    labels = {"desc_aug": "Desc-Aug", "self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, per_ds in method_f1.items():
        deltas = {
            ds: per_ds[ds] - vis_per_ds[ds]
            for ds in per_ds if ds in vis_per_ds
        }
        if not deltas:
            continue

        sorted_ds = sorted(deltas, key=lambda d: deltas[d])
        top_pos = sorted_ds[-top_n:][::-1]   # largest positive first
        top_neg = sorted_ds[:top_n]           # most negative first

        for subset, ds_list, title_suffix, fname_suffix in [
            (top_pos, top_pos, f"Top {top_n} Gains",     "gains"),
            (top_neg, top_neg, f"Top {top_n} Regressions", "regressions"),
        ]:
            vals   = [deltas[ds] for ds in ds_list]
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
            ylabels = [ds.replace("_classification", "").replace("_", " ") for ds in ds_list]

            fig, ax = plt.subplots(figsize=(7, max(3, len(ds_list) * 0.65)))
            bars = ax.barh(range(len(ds_list)), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(len(ds_list)))
            ax.set_yticklabels(ylabels, fontsize=9)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Δ Judged F1 Macro (method − baseline)")
            ax.set_title(f"{labels.get(method, method)} — {title_suffix} vs Baseline")
            ax.xaxis.grid(True, alpha=0.3)
            for bar, v in zip(bars, vals):
                x_pos = v + 0.005 if v >= 0 else v - 0.005
                ha = "left" if v >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{v:+.3f}", va="center", ha=ha, fontsize=8)
            plt.tight_layout()
            out = output_dir / f"cot_delta_{method}_{fname_suffix}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")


def plot_delta_response_examples(
    metrics: dict,
    vis_per_ds: dict[str, float],
    aug_per_ds: dict[str, float],
    aug_rows: list[dict] | None,
    cot_rows: list[dict] | None,
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """
    For each method × direction (gains / regressions), render one PNG that shows
    top_n dataset cards, each with one correct and one incorrect example response.
    Saved to output_dir/delta_examples/{method}_{direction}.png.
    """
    import textwrap
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    METHOD_LABELS = {"desc_aug": "Desc-Aug", "self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}
    METHOD_COLORS = {"desc_aug": "#1f77b4", "self_cot": "#ff7f0e", "cot_yaml": "#2ca02c"}

    method_f1: dict[str, dict[str, float]] = {}
    if aug_per_ds:
        method_f1["desc_aug"] = aug_per_ds
    for cond in ("self_cot", "cot_yaml"):
        if cond in metrics:
            method_f1[cond] = {
                ds: m.get("judge_f1_macro") or m.get("f1_macro", 0.0)
                for ds, m in metrics[cond].get("per_dataset", {}).items()
            }

    if not vis_per_ds or not method_f1:
        return

    # Build per-method row lookup: {ds: [rows]}
    from collections import defaultdict

    def _build_lookup(rows, cond_filter=None):
        d = defaultdict(list)
        for r in (rows or []):
            if cond_filter and r.get("condition") != cond_filter:
                continue
            d[r.get("agml_dataset", "")].append(r)
        return dict(d)

    rows_by_method = {
        "desc_aug": _build_lookup(aug_rows),
        "self_cot": _build_lookup(cot_rows, "self_cot"),
        "cot_yaml": _build_lookup(cot_rows, "cot_yaml"),
    }

    def _wrap(text, width=88, max_lines=4):
        if not text:
            return "(no response)"
        lines = textwrap.wrap(text.strip().replace("\n", " "), width)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][: width - 3] + "..."
        return "\n".join(lines) or text[:width]

    def _format_response(row, method):
        pred = row.get("prediction", "")
        if method == "desc_aug":
            desc = row.get("lang_description", "")
            parts = []
            if desc:
                parts.append("Desc:   " + _wrap(desc, 84, 2))
            if pred:
                parts.append("Answer: " + _wrap(pred, 84, 2))
            return "\n".join(parts) if parts else _wrap(pred)
        if method == "cot_yaml":
            reasoning = row.get("reasoning", "")
            if reasoning:
                # Show each Q/A step, one line per step (question truncated, answer wrapped)
                parts = []
                for block in reasoning.split("\n\n"):
                    lines = block.split("\n", 1)
                    q = lines[0][3:].strip() if lines[0].startswith("Q:") else lines[0]
                    a = lines[1][3:].strip() if len(lines) > 1 and lines[1].startswith("A:") else ""
                    parts.append(f"Q: {q[:70]}")
                    if a:
                        parts.append("   " + _wrap(a, 85, 2))
                return "\n".join(parts) if parts else _wrap(pred, 88, 4)
        return _wrap(pred, 88, 4)

    # Layout (all units in inches, origin bottom-left)
    FIG_W      = 13.5
    MARGIN_X   = 0.45
    INNER_W    = FIG_W - 2 * MARGIN_X
    TITLE_H    = 0.75
    HDR_H      = 0.42    # dataset header bar
    CARD_H     = 2.0     # one example card
    DS_GAP     = 0.28    # gap between dataset blocks
    BOTTOM_PAD = 0.35
    LINE_H     = 0.165   # approx height of one text line at 8.5pt

    cards_per_ds = 2     # correct + incorrect
    ds_block_h   = HDR_H + cards_per_ds * CARD_H
    total_h      = TITLE_H + top_n * (ds_block_h + DS_GAP) + BOTTOM_PAD

    out_dir = output_dir / "delta_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method, per_ds in method_f1.items():
        deltas = {
            ds: per_ds[ds] - vis_per_ds[ds]
            for ds in per_ds if ds in vis_per_ds
        }
        if not deltas:
            continue

        sorted_ds   = sorted(deltas, key=lambda d: deltas[d])
        m_color     = METHOD_COLORS.get(method, "#555555")
        m_label     = METHOD_LABELS.get(method, method)
        method_rows = rows_by_method.get(method, {})

        for direction, ds_list, dir_label in [
            ("gains",       sorted_ds[-top_n:][::-1], f"Top {top_n} Gains vs Baseline"),
            ("regressions", sorted_ds[:top_n],        f"Top {top_n} Regressions vs Baseline"),
        ]:
            fig, ax = plt.subplots(figsize=(FIG_W, total_h))
            ax.set_xlim(0, FIG_W)
            ax.set_ylim(0, total_h)
            ax.axis("off")
            fig.patch.set_facecolor("#f8f8f8")

            # ── Title ──────────────────────────────────────────────────────
            y = total_h - 0.18
            ax.text(FIG_W / 2, y, f"{m_label}  —  {dir_label}  (judged F1 Δ vs baseline)",
                    ha="center", va="top", fontsize=13, fontweight="bold", color=m_color)
            y -= TITLE_H

            for ds in ds_list:
                delta      = deltas.get(ds, 0.0)
                ds_label   = ds.replace("_classification", "").replace("_", " ").title()
                ds_rows    = method_rows.get(ds, [])
                correct_ex = next((r for r in ds_rows if str(r.get("judged_correct", "")) == "1"), None)
                wrong_ex   = next((r for r in ds_rows if str(r.get("judged_correct", "")) == "0"), None)

                # ── Dataset header bar ──────────────────────────────────────
                hdr = mpatches.FancyBboxPatch(
                    (MARGIN_X, y - HDR_H), INNER_W, HDR_H,
                    boxstyle="round,pad=0.05", linewidth=0,
                    facecolor=m_color, alpha=0.18, zorder=1,
                )
                ax.add_patch(hdr)
                ax.text(MARGIN_X + 0.18, y - HDR_H / 2,
                        ds_label, ha="left", va="center",
                        fontsize=11, fontweight="bold", color="#111111", zorder=2)
                d_color = "#2ca02c" if delta >= 0 else "#d62728"
                ax.text(FIG_W - MARGIN_X - 0.18, y - HDR_H / 2,
                        f"Δ = {delta:+.3f}", ha="right", va="center",
                        fontsize=11, fontweight="bold", color=d_color, zorder=2)
                y -= HDR_H

                # ── Example cards ───────────────────────────────────────────
                for row, card_label, bg, border in [
                    (correct_ex, "✓  Correct",   "#f0faf0", "#4caf50"),
                    (wrong_ex,   "✗  Incorrect", "#fff5f5", "#ef5350"),
                ]:
                    body = _format_response(row, method) if row else "(no example found)"
                    gt   = (row or {}).get("class_label", "")
                    _r = row or {}
                    pred = _r.get("resolved_class", "") or _r.get("prediction", "")

                    card = mpatches.FancyBboxPatch(
                        (MARGIN_X, y - CARD_H + 0.06), INNER_W, CARD_H - 0.1,
                        boxstyle="round,pad=0.06", linewidth=1.4,
                        edgecolor=border, facecolor=bg, zorder=1,
                    )
                    ax.add_patch(card)

                    import textwrap
                    lbl_color = "#2ca02c" if "Correct" in card_label else "#c62828"
                    line1 = card_label + (f"     GT: {gt}" if gt else "")
                    ax.text(MARGIN_X + 0.22, y - 0.15,
                            line1, ha="left", va="top",
                            fontsize=8.5, fontweight="bold",
                            color=lbl_color, zorder=2)
                    body_y = y - 0.38  # default body start when no pred
                    if pred:
                        pred_wrapped = textwrap.fill(f"→   Pred: {pred}", width=60, max_lines=3, placeholder="…")
                        n_pred_lines = pred_wrapped.count("\n") + 1
                        ax.text(MARGIN_X + 0.22, y - 0.32,
                                pred_wrapped, ha="left", va="top",
                                fontsize=7.5, fontweight="bold",
                                color=lbl_color, zorder=2)
                        body_y = y - 0.32 - n_pred_lines * 0.155
                    ax.text(MARGIN_X + 0.22, body_y,
                            body, ha="left", va="top",
                            fontsize=8, color="#2e2e2e",
                            fontfamily="monospace", linespacing=1.45, zorder=2)
                    y -= CARD_H

                y -= DS_GAP

            out = out_dir / f"{method}_{direction}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {out}")

            # ── CSV: all rows for selected datasets ────────────────────────
            csv_rows = []
            for ds in ds_list:
                rank = ds_list.index(ds) + 1
                delta_val = deltas.get(ds, 0.0)
                for r in method_rows.get(ds, []):
                    csv_rows.append({
                        **r,
                        "delta": f"{delta_val:+.4f}",
                        "delta_rank": rank,
                    })
            if csv_rows:
                csv_out = out_dir / f"{method}_{direction}.csv"
                fieldnames = list(csv_rows[0].keys())
                with open(csv_out, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(csv_rows)
                print(f"  Saved: {csv_out}  ({len(csv_rows)} rows)")


def plot_condition_comparison(metrics: dict, output_path: Path) -> None:
    """Bar chart: F1 per condition per dataset."""
    import matplotlib.pyplot as plt

    conditions = [c for c in ("self_cot", "cot_yaml") if c in metrics]
    if not conditions:
        return

    all_ds = sorted({ds for cond in conditions for ds in metrics[cond]["per_dataset"]})
    if not all_ds:
        return

    x = np.arange(len(all_ds))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(10, len(all_ds) * 1.4), 5))

    colors = {"self_cot": "#1f77b4", "cot_yaml": "#2ca02c"}
    labels = {"self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}

    for i, cond in enumerate(conditions):
        f1s = [metrics[cond]["per_dataset"].get(ds, {}).get("f1_macro", 0.0) for ds in all_ds]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=labels[cond],
                      color=colors.get(cond, f"C{i}"), alpha=0.85)
        for bar, v in zip(bars, f1s):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    clean_ds = [d.replace("_classification", "").replace("_", " ") for d in all_ds]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_ds, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("CoT Vision Assessment — F1 Macro by Condition and Dataset")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoT Vision Assessment — self-description and structured CoT conditions"
    )
    parser.add_argument("--model",       default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--config",      default="src/config/datasets.yaml",
                         help="datasets.yaml, for each row's class list — needed since the pipeline's "
                              "default judge template (DEFAULT_JUDGE_TEMPLATE_KEY, src/common/metrics.py) "
                              "has a {class_list} placeholder.")
    parser.add_argument("--output_dir",  default="outputs/cot_vision/")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max images per class (default: all — uses test_split when available)")
    parser.add_argument("--datasets",    nargs="+", default=DEFAULT_DATASETS,
                        help="AgML dataset names to evaluate")
    parser.add_argument("--hard_only", action="store_true",
                        help="Only run the 3 worst-performing datasets per task type "
                             "(by vision judged F1). Requires --vis_dir. Useful for fast "
                             "iteration and test-time scaling experiments.")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json",
                        help="Path to test_split.json from linear probe — restricts each class "
                             "to the same images used in vision assessment")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use bitsandbytes 4-bit quantization via vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs",  type=int, default=32)
    parser.add_argument("--prompt_batch_size", type=int, default=256,
                        help="Images per prompt-build+generate chunk — reduce to lower peak CPU RAM")
    parser.add_argument("--vis_dir",    default="outputs/vision_assessment/",
                        help="Vision assessment output dir — used to load baseline judged F1 for difficulty plots")
    parser.add_argument("--aug_dir",    default="outputs/desc_augmented_vision/",
                        help="Desc-aug output dir — used to load augmented F1 for hard-subset 4-condition chart")
    parser.add_argument("--clustering_metrics", default="outputs/clustering_eval/clustering_metrics.json",
                        help="Path to clustering_metrics.json — enables label-free difficulty binning")
    parser.add_argument("--clustering_backbone", default="dinov3-vitb16-pretrain-lvd1689m",
                        help="Backbone key in clustering_metrics.json to use for difficulty binning")
    parser.add_argument("--mv_dir", default="outputs/majority_vote/",
                        help="Majority vote output dir — adds MV bars to the hard-condition chart")
    parser.add_argument("--from_ckpt", action="store_true",
                        help="Compute metrics from checkpoint CSVs (ckpt_self_cot.csv etc.) "
                             "and regenerate plots — use when inference is incomplete")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip inference; regenerate plots from existing cot_vision_metrics.json + CSV")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    test_split: dict | None = None
    split_path = Path(args.test_split)
    if split_path.exists():
        import json as _json
        test_split = _json.loads(split_path.read_text())
        print(f"[cot_vision] test_split loaded from {split_path}")
    else:
        print(f"[cot_vision] test_split not found at {split_path} — using all available images")

    # Resolve dataset list — optionally filter to 3 worst per task
    active_datasets = args.datasets
    if args.hard_only:
        print(f"\n[cot_vision] --hard_only: selecting 3 worst datasets per task type...")
        vis_per_ds_for_filter = load_vis_per_dataset(Path(args.vis_dir), args.model)
        if not vis_per_ds_for_filter:
            print("  [warn] No vision baseline found — falling back to all datasets")
        else:
            active_datasets = select_hard_datasets(vis_per_ds_for_filter, n_per_task=3)
            print(f"  Selected {len(active_datasets)} datasets: {active_datasets}")

    metrics_path = output_dir / "cot_vision_metrics.json"
    cls_csv      = output_dir / f"cot_cls_{safe_model}.csv"

    if args.from_ckpt:
        # Load from whichever checkpoint CSVs exist and compute partial metrics
        ckpt_files = {
            "self_cot":  output_dir / "ckpt_self_cot.csv",
            "cot_yaml":  output_dir / "ckpt_cot_yaml.csv",
        }
        all_rows: list[dict] = []
        for cond, path in ckpt_files.items():
            if path.exists():
                rows = list(csv.DictReader(open(path, encoding="utf-8")))
                all_rows.extend(rows)
                print(f"  Loaded {len(rows)} rows from {path.name} ({cond})")
            else:
                print(f"  [skip] {path.name} not found")
        if not all_rows:
            print("[error] No checkpoint CSVs found in output_dir")
            return
        metrics = compute_metrics(all_rows)
        print(f"[from_ckpt] Computed metrics for conditions: {list(metrics.keys())}")

    elif args.plot_only:
        if not metrics_path.exists():
            print(f"[error] --plot_only requires {metrics_path} — run inference first")
            return
        with open(metrics_path) as f:
            saved = json.load(f)
        metrics = saved.get("conditions", saved)
        print(f"[cot_vision] plot_only: loaded metrics from {metrics_path}")
    else:
        # Phase 0: collect all work items upfront
        print("\n=== Collecting work items ===")
        t0 = time.perf_counter()
        items = collect_work_items(active_datasets, args.n_samples, test_split)
        if not items:
            print("[error] No work items collected — check datasets and test_split")
            return

        # Checkpoint paths — each phase saves on completion so a rerun can skip done phases
        ckpt_self_cot  = output_dir / "ckpt_self_cot.csv"
        ckpt_cot_yaml  = output_dir / "ckpt_cot_yaml.csv"
        ckpt_judged    = output_dir / "ckpt_judged.csv"

        # Load vLLM once for all phases
        llm, tokenizer = load_llm(
            args.model,
            quantization="bitsandbytes" if args.use_4bit else None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )

        # Phase 1: self_cot
        if ckpt_self_cot.exists():
            print(f"\n=== Phase 1: self_cot [SKIPPED — checkpoint found] ===")
            self_cot_rows = list(csv.DictReader(open(ckpt_self_cot, encoding="utf-8")))
            print(f"  Loaded {len(self_cot_rows)} rows from {ckpt_self_cot.name}")
        else:
            print("\n=== Phase 1: self_cot ===")
            self_cot_rows = run_self_cot(items, llm, tokenizer, prompt_batch_size=args.prompt_batch_size)
            write_cls_csv(self_cot_rows, ckpt_self_cot)
            print(f"  Checkpoint saved: {ckpt_self_cot.name}")

        # Phase 2: cot_yaml
        if ckpt_cot_yaml.exists():
            print(f"\n=== Phase 2: cot_yaml [SKIPPED — checkpoint found] ===")
            cot_yaml_rows = list(csv.DictReader(open(ckpt_cot_yaml, encoding="utf-8")))
            print(f"  Loaded {len(cot_yaml_rows)} rows from {ckpt_cot_yaml.name}")
        else:
            print("\n=== Phase 2: cot_yaml ===")
            cot_yaml_rows = run_cot_yaml(items, llm, tokenizer,
                                         prompt_batch_size=args.prompt_batch_size,
                                         ckpt_dir=output_dir)
            write_cls_csv(cot_yaml_rows, ckpt_cot_yaml)
            print(f"  Checkpoint saved: {ckpt_cot_yaml.name}")

        all_rows = self_cot_rows + cot_yaml_rows

        # Phase 3: judge (incremental — skips rows already in ckpt_judged.csv)
        _row_key = lambda r: (
            r.get("agml_dataset", ""), r.get("image_name", ""),
            r.get("class_label", ""),  r.get("condition", ""),
        )
        already_judged: dict[tuple, dict] = {}
        if ckpt_judged.exists():
            for r in csv.DictReader(open(ckpt_judged, encoding="utf-8")):
                if r.get("judged_correct", "") != "":
                    already_judged[_row_key(r)] = r

        todo  = [r for r in all_rows if _row_key(r) not in already_judged]
        n_skip = len(all_rows) - len(todo)

        def _merge_judged(new_row: dict, judged_row: dict) -> dict:
            # Keep all fields from the new row (e.g. reasoning), overlay only the
            # judge outputs so we don't lose new columns added since the last run.
            return {**new_row,
                    "judged_correct":  judged_row.get("judged_correct", ""),
                    "judge_reasoning": judged_row.get("judge_reasoning", "")}

        if not todo:
            print(f"\n=== Phase 3: judge [SKIPPED — all {len(all_rows)} rows already judged] ===")
            all_rows = [_merge_judged(r, already_judged[_row_key(r)])
                        if _row_key(r) in already_judged else r
                        for r in all_rows]
        else:
            print(f"\n=== Phase 3: judge ({len(todo)} new rows, {n_skip} already done) ===")
            from src.common.dataset_config import class_list_extra_keys
            from src.common.metrics import DEFAULT_JUDGE_TEMPLATE_KEY
            extra_keys = class_list_extra_keys(todo, Path(args.config))
            judged_new = apply_judge(
                rows=todo,
                llm=llm,
                tokenizer=tokenizer,
                system_prompt=_PROMPTS.get("judge_system_vision", ""),
                judge_template=_PROMPTS[DEFAULT_JUDGE_TEMPLATE_KEY],
                gt_key="ground_truth",
                pred_key="prediction",
                extra_keys=extra_keys,
            )
            new_judged_map = {_row_key(r): r for r in judged_new}
            all_rows = [
                _merge_judged(r, already_judged[_row_key(r)]) if _row_key(r) in already_judged
                else new_judged_map.get(_row_key(r), r)
                for r in all_rows
            ]
            write_cls_csv(all_rows, ckpt_judged)
            print(f"  Checkpoint saved: {ckpt_judged.name}")

        elapsed = time.perf_counter() - t0
        print(f"\nTotal time: {elapsed:.1f}s  ({len(all_rows)} rows)")

        destroy_llm(llm)
        del tokenizer

        write_cls_csv(all_rows, cls_csv)

        metrics = compute_metrics(all_rows)
        with open(metrics_path, "w") as f:
            json.dump({"model": args.model, "n_samples": args.n_samples,
                       "conditions": metrics}, f, indent=2)
        print(f"  Saved: {metrics_path}")

    # Print summary
    print("\n── CoT Vision Assessment Summary ─────────────────")
    cond_labels = {"self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}
    for cond, m in metrics.items():
        jf1 = m.get('judge_f1_macro')
        jf1_str = f"  judge_F1={jf1:.3f}" if jf1 is not None else ""
        print(f"  {cond_labels.get(cond, cond):<20}  F1={m['f1_macro']:.3f}{jf1_str}")
        for ds, dm in sorted(m["per_dataset"].items()):
            short = ds.replace("_classification", "").replace("_", " ")
            print(f"    {short:<45}  F1={dm['f1_macro']:.3f}")

    # Load vision baseline for difficulty plots + hard 4-condition chart
    vis_per_ds = load_vis_per_dataset(Path(args.vis_dir), args.model)
    if not vis_per_ds:
        print("  [warn] No vision baseline found — skipping difficulty plots")

    print("\n=== Generating plots ===")
    aug_per_ds = load_aug_per_dataset(Path(args.aug_dir), args.model)

    # Load raw rows for plot_conditions_per_task — mirrors desc_augmented_vision_plot.py
    safe_model = args.model.replace("/", "_")
    _vis_cls_path      = Path(args.vis_dir) / f"cls_{safe_model}.csv"
    _baseline_cls_path = Path(args.aug_dir) / f"baseline_cls_all_{safe_model}.csv"
    _aug_cls_path      = Path(args.aug_dir) / f"aug_cls_all_{safe_model}.csv"
    _cot_cls_path      = output_dir / f"cot_cls_{safe_model}.csv"

    # Prefer the desc-aug baseline CSV (same subset as aug run) so numbers match desc-aug chart.
    # Merge judged_correct from vis CSV just like desc_augmented_vision_plot.py does.
    if _baseline_cls_path.exists():
        vis_rows = list(csv.DictReader(open(_baseline_cls_path, encoding="utf-8")))
        if _vis_cls_path.exists():
            vis_judged = {(r["agml_dataset"], r["image_name"], r["class_label"]): r.get("judged_correct", "")
                          for r in csv.DictReader(open(_vis_cls_path, encoding="utf-8"))}
            for r in vis_rows:
                key = (r.get("agml_dataset", ""), r.get("image_name", ""), r.get("class_label", ""))
                if not r.get("judged_correct") and key in vis_judged:
                    r["judged_correct"] = vis_judged[key]
            print(f"  Merged judged_correct into baseline rows from {_vis_cls_path.name}")
    elif _vis_cls_path.exists():
        vis_rows = list(csv.DictReader(open(_vis_cls_path, encoding="utf-8")))
    else:
        vis_rows = None

    aug_rows = list(csv.DictReader(open(_aug_cls_path, encoding="utf-8"))) if _aug_cls_path.exists() else None
    if _cot_cls_path.exists():
        cot_rows = list(csv.DictReader(open(_cot_cls_path, encoding="utf-8")))
    else:
        # Fall back to ckpt CSVs (used by --from_ckpt and partial runs)
        cot_rows = []
        for _ckpt in [output_dir / "ckpt_self_cot.csv", output_dir / "ckpt_cot_yaml.csv"]:
            if _ckpt.exists():
                cot_rows.extend(csv.DictReader(open(_ckpt, encoding="utf-8")))
        cot_rows = cot_rows or None

    plot_condition_comparison(metrics, output_dir / "cot_comparison.png")
    plot_delta_per_dataset(metrics, vis_per_ds, aug_per_ds, output_dir)
    plot_delta_response_examples(metrics, vis_per_ds, aug_per_ds, aug_rows, cot_rows, output_dir)
    plot_conditions_per_task(metrics, vis_rows, aug_rows, output_dir, cot_rows=cot_rows)
    plot_conditions_overall(metrics, vis_per_ds, aug_per_ds, output_dir,
                            vis_rows=vis_rows, aug_rows=aug_rows, cot_rows=cot_rows)

    if vis_per_ds:
        plot_difficulty_correlation(metrics, vis_per_ds, output_dir,
                                    vis_rows=vis_rows, aug_rows=aug_rows, cot_rows=cot_rows)
        plot_difficulty_bins(metrics, vis_per_ds, output_dir,
                             vis_rows=vis_rows, aug_rows=aug_rows, cot_rows=cot_rows)
        clustering_per_ds = load_clustering_per_dataset(
            Path(args.clustering_metrics), backbone=args.clustering_backbone)
        if clustering_per_ds:
            plot_difficulty_bins_clustering(
                metrics, vis_per_ds, clustering_per_ds, output_dir,
                backbone=args.clustering_backbone,
                vis_rows=vis_rows, aug_rows=aug_rows, cot_rows=cot_rows)

        # Clustering difficulty vs judged F1 — overlaid scatter (all conditions, shared axes)
        from src.common.method_comparison import plot_difficulty_scatter_comparison
        _diff_rows = _build_rows_by_cond(metrics, vis_rows, aug_rows, cot_rows)
        plot_difficulty_scatter_comparison(
            _diff_rows,
            clustering_metrics_path=Path(args.clustering_metrics),
            output_path=output_dir / "difficulty_correlation" / "difficulty_vs_perf_cot_overlay.png",
            backbone=args.clustering_backbone,
            title="Dataset Difficulty vs Judged F1 — CoT Conditions",
        )

    if args.hard_only and vis_per_ds:
        aug_per_ds = load_aug_per_dataset(Path(args.aug_dir), args.model)
        if not aug_per_ds:
            print("  [warn] No desc-aug metrics found — desc_aug bar will be zero")
        mv_metrics: dict | None = None
        mv_metrics_path = Path(args.mv_dir) / "majority_vote_metrics.json"
        if mv_metrics_path.exists():
            with open(mv_metrics_path) as f:
                mv_saved = json.load(f)
            mv_metrics = mv_saved.get("conditions", mv_saved)
            print(f"  Loaded MV metrics from {mv_metrics_path} ({len(mv_metrics)} conditions)")
        else:
            print(f"  [info] No MV metrics at {mv_metrics_path} — MV bars omitted")
        plot_hard_four_condition(
            active_datasets, vis_per_ds, aug_per_ds, metrics, output_dir,
            vis_dir=Path(args.vis_dir), aug_dir=Path(args.aug_dir),
            model_id=args.model, n_samples=args.n_samples,
            mv_metrics=mv_metrics,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
