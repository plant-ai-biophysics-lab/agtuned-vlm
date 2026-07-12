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
        task_type, crop = DATASET_TASK_TYPES.get(ds, ("disease", "plant"))
        # Use DatasetConfig.clean_classes for resolve_class — same source as vision_assessment
        cfg = ds_cfg_map.get(ds)
        clean_classes = cfg.clean_classes if cfg else sorted(by_class.keys())
        for cls, paths in by_class.items():
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

        group_result = [_make_row(item, "cot_yaml", pred)
                        for item, pred in zip(group_items, predictions)]

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
    "condition", "prediction", "resolved_class", "ground_truth",
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
    """Load per-dataset augmented F1 from desc_augmented_metrics_all.json."""
    for label in ("all", "filtered"):
        path = aug_dir / f"desc_augmented_metrics_{label}.json"
        if path.exists():
            data = json.loads(path.read_text())
            model_data = data.get("models", {}).get(model_id, {})
            per_ds = model_data.get("augmented_per_dataset", {})
            return {ds: v.get("f1_macro", 0.0) for ds, v in per_ds.items()}
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
) -> None:
    """Scatter: vision baseline judged F1 (x=difficulty proxy) vs CoT judged F1 per dataset."""
    import matplotlib.pyplot as plt

    conditions = [c for c in ("self_cot", "cot_yaml") if c in cot_metrics]
    if not conditions or not vis_per_ds:
        return

    colors = {"self_cot": "#1f77b4", "cot_yaml": "#2ca02c"}
    labels = {"self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in conditions:
        per_ds = cot_metrics[cond].get("per_dataset", {})
        xs, ys, names = [], [], []
        for ds, dm in per_ds.items():
            if ds not in vis_per_ds:
                continue
            xs.append(vis_per_ds[ds])
            ys.append(dm.get("judge_f1_macro") or dm.get("f1_macro", 0.0))
            names.append(ds.replace("_classification", "").replace("_", " "))
        ax.scatter(xs, ys, label=labels[cond], color=colors[cond], alpha=0.8, s=60)
        for x, y, name in zip(xs, ys, names):
            ax.annotate(name, (x, y), fontsize=5.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="no change")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Vision Baseline Judged F1  (← harder)")
    ax.set_ylabel("CoT Judged F1")
    ax.set_title("Difficulty vs CoT Performance (per dataset)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = output_dir / "cot_difficulty_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_difficulty_bins_impl(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    difficulty_per_ds: dict[str, float],
    output_dir: Path,
    title: str,
    filename: str,
    difficulty_label: str,
) -> None:
    """Bar chart: mean judged F1 per difficulty tercile, binned by difficulty_per_ds."""
    import matplotlib.pyplot as plt

    conditions = [c for c in ("self_cot", "cot_yaml") if c in cot_metrics]
    if not conditions or not difficulty_per_ds:
        return

    # Restrict to datasets present in difficulty scores; sort ascending (low = hard)
    all_ds = sorted(difficulty_per_ds, key=difficulty_per_ds.get)
    n = len(all_ds)
    bins = {"Hard": all_ds[:n//3], "Medium": all_ds[n//3:2*n//3], "Easy": all_ds[2*n//3:]}

    colors = {"vision": "#e67e22", "self_cot": "#1f77b4", "cot_yaml": "#2ca02c"}
    labels = {"vision": "Vision Baseline", "self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}
    all_conds = ["vision"] + conditions
    x = np.arange(len(bins))
    width = 0.8 / len(all_conds)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cond in enumerate(all_conds):
        means = []
        for bin_ds in bins.values():
            if cond == "vision":
                vals = [vis_per_ds[d] for d in bin_ds if d in vis_per_ds]
            else:
                per_ds = cot_metrics[cond].get("per_dataset", {})
                vals = [per_ds[d].get("judge_f1_macro") or per_ds[d].get("f1_macro", 0.0)
                        for d in bin_ds if d in per_ds]
            means.append(float(np.mean(vals)) if vals else 0.0)
        offset = (i - len(all_conds) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=labels[cond],
                      color=colors[cond], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(list(bins))
    ax.set_ylabel("Mean Judged F1"); ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)
    # Annotate difficulty axis label
    ax.set_xlabel(f"Difficulty bin (by {difficulty_label}, low → hard)")
    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_difficulty_bins(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    output_dir: Path,
) -> None:
    """Bar chart binned by vision baseline judged F1 (supervised difficulty proxy)."""
    _plot_difficulty_bins_impl(
        cot_metrics, vis_per_ds, vis_per_ds, output_dir,
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
) -> None:
    """One difficulty-bins chart per clustering metric (label-free difficulty proxies)."""
    metric_labels = {
        "nmi":               "NMI",
        "ari":               "ARI",
        "silhouette":        "Silhouette",
        "inter_centroid_dist": "Inter-centroid Distance",
        "fisher_dr":         "Fisher Discriminant Ratio",
    }
    backbone_short = backbone.split("/")[-1] if "/" in backbone else backbone
    for metric, label in metric_labels.items():
        diff = {ds: vals[metric] for ds, vals in clustering_per_ds.items() if metric in vals}
        if not diff:
            continue
        _plot_difficulty_bins_impl(
            cot_metrics, vis_per_ds, diff, output_dir,
            title=f"CoT Performance by Difficulty Bin ({label}, {backbone_short})",
            filename=f"cot_difficulty_bins_{metric}.png",
            difficulty_label=label,
        )


_CONDITION_STYLE = [
    ("Vision Baseline", "#1f77b4"),
    ("Self-CoT",        "#ff7f0e"),
    ("CoT YAML",        "#2ca02c"),
    ("Desc-Aug",        "#9467bd"),
]


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


def plot_conditions_per_task(
    metrics: dict,
    vis_per_ds: dict[str, float] | None,
    aug_per_ds: dict[str, float] | None,
    output_dir: Path,
) -> None:
    """Grouped bar chart: mean judged F1 per condition × task type."""
    import matplotlib.pyplot as plt

    cond_map = _build_condition_map(metrics, vis_per_ds, aug_per_ds)
    if not cond_map:
        return

    ds_to_task = {ds: tt for ds, (tt, _) in DATASET_TASK_TYPES.items()}
    task_types = sorted({tt for tt in ds_to_task.values()
                         if tt in ("disease", "pest_damage", "weed_identification")})

    # mean F1 per (condition, task)
    def _task_mean(per_ds: dict[str, float], task: str) -> float:
        vals = [v for ds, v in per_ds.items() if ds_to_task.get(ds) == task]
        return float(np.mean(vals)) if vals else 0.0

    conditions = [lbl for lbl, _ in _CONDITION_STYLE if lbl in cond_map]
    colors     = {lbl: c for lbl, c in _CONDITION_STYLE}

    x     = np.arange(len(task_types))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * len(conditions) * 0.5 + 2), 5))

    for i, lbl in enumerate(conditions):
        vals   = [_task_mean(cond_map[lbl], tt) for tt in task_types]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=lbl,
                        color=colors[lbl], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    task_labels = {"disease": "Disease", "pest_damage": "Pest Damage",
                   "weed_identification": "Weed ID"}
    ax.set_xticks(x)
    ax.set_xticklabels([task_labels.get(tt, tt) for tt in task_types], fontsize=10)
    ax.set_ylabel("Mean Judged F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("CoT Assessment — Mean Judged F1 by Task Type")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "cot_per_task_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_conditions_overall(
    metrics: dict,
    vis_per_ds: dict[str, float] | None,
    aug_per_ds: dict[str, float] | None,
    output_dir: Path,
) -> None:
    """Single grouped bar: mean judged F1 across all datasets per condition."""
    import matplotlib.pyplot as plt

    cond_map = _build_condition_map(metrics, vis_per_ds, aug_per_ds)
    if not cond_map:
        return

    conditions = [lbl for lbl, _ in _CONDITION_STYLE if lbl in cond_map]
    colors     = {lbl: c for lbl, c in _CONDITION_STYLE}
    means      = [float(np.mean(list(cond_map[lbl].values()))) for lbl in conditions]

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 1.2), 5))
    bars = ax.bar(conditions, means,
                  color=[colors[lbl] for lbl in conditions],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Judged F1 (all datasets)")
    ax.set_ylim(0, 1.05)
    ax.set_title("CoT Assessment — Overall Mean Judged F1")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "cot_overall_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


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

        # Phase 3: judge
        if ckpt_judged.exists():
            print(f"\n=== Phase 3: judge [SKIPPED — checkpoint found] ===")
            all_rows = list(csv.DictReader(open(ckpt_judged, encoding="utf-8")))
            print(f"  Loaded {len(all_rows)} judged rows from {ckpt_judged.name}")
        else:
            print(f"\n=== Phase 3: judge ({len(all_rows)} rows) ===")
            all_rows = apply_judge(
                rows=all_rows,
                llm=llm,
                tokenizer=tokenizer,
                system_prompt=_PROMPTS.get("judge_system_vision", ""),
                judge_template=_PROMPTS["judge_classification_template"],
                gt_key="ground_truth",
                pred_key="prediction",
            )
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

    plot_condition_comparison(metrics, output_dir / "cot_comparison.png")
    plot_conditions_per_task(metrics, vis_per_ds, aug_per_ds, output_dir)
    plot_conditions_overall(metrics, vis_per_ds, aug_per_ds, output_dir)

    if vis_per_ds:
        plot_difficulty_correlation(metrics, vis_per_ds, output_dir)
        plot_difficulty_bins(metrics, vis_per_ds, output_dir)
        clustering_per_ds = load_clustering_per_dataset(
            Path(args.clustering_metrics), backbone=args.clustering_backbone)
        if clustering_per_ds:
            plot_difficulty_bins_clustering(
                metrics, vis_per_ds, clustering_per_ds, output_dir,
                backbone=args.clustering_backbone)

        # Clustering difficulty vs judged F1 per CoT condition
        def _judged_f1_per_ds(cond_key: str) -> dict[str, float]:
            per_ds = metrics.get(cond_key, {}).get("per_dataset", {})
            return {ds: m["judge_f1_macro"] for ds, m in per_ds.items()
                    if "judge_f1_macro" in m}

        cot_conditions = {"Vision Baseline": vis_per_ds}
        for cond, label in [("self_cot", "Self-CoT"), ("cot_yaml", "CoT YAML")]:
            if cond in metrics:
                cot_conditions[label] = _judged_f1_per_ds(cond)

        plot_difficulty_correlation_conditions(
            cot_conditions,
            clustering_metrics_path=Path(args.clustering_metrics),
            output_dir=output_dir / "difficulty_correlation",
            backbone=args.clustering_backbone,
            filename="difficulty_vs_perf_cot_conditions.png",
            title="Difficulty vs Judged F1 — CoT Conditions",
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
