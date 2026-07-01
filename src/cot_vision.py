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
    resolve_class, sample_class_images,
)

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


def _cls_question(class_names: list[str], task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        inner = _PROMPTS["visual_classification_weed"]
    else:
        inner = _PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    inner += f" Choose exactly one from: {', '.join(class_names)}."
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
            # Restrict to the exact images used in the vision assessment test split
            allowed = set(split_cls[cls])
            paths = [p for p in paths if p.name in allowed]
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
    """Load all (dataset, class, image) tuples upfront for batched inference."""
    items: list[dict] = []
    for ds in datasets:
        print(f"  Loading {ds}...")
        try:
            by_class = load_dataset_classes(ds, n_samples, test_split)
        except Exception as e:
            print(f"    [skip] {e}")
            continue
        task_type, crop = DATASET_TASK_TYPES.get(ds, ("disease", "plant"))
        class_names = sorted(by_class.keys())
        for cls, paths in by_class.items():
            for path in paths:
                try:
                    image = PILImage.open(path).convert("RGB")
                except Exception:
                    continue
                items.append({
                    "ds":          ds,
                    "class_label": cls,
                    "task_type":   task_type,
                    "crop":        crop,
                    "class_names": class_names,
                    "image_name":  path.name,
                    "image":       image,
                })
    print(f"  Collected {len(items)} work items across {len(datasets)} datasets")
    return items


def _make_row(item: dict, condition: str, pred: str) -> dict:
    resolved = resolve_class(pred, item["class_names"])
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


def run_self_cot_batched(items: list[dict], llm, tokenizer) -> list[dict]:
    """
    Stage 1: batch-generate descriptions for all images.
    Stage 2: batch-generate classifications using stage-1 history.
    """
    # Stage 1 — describe
    print(f"\n[self_cot] Stage 1/2: generating descriptions ({len(items)} images)...")
    stage1_prompts = []
    for item in items:
        desc_q = _desc_question(item["task_type"], item["crop"])
        msgs = _sys_msg() + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": desc_q},
        ]}]
        stage1_prompts.append(_build_prompt(msgs, tokenizer, item["image"]))
    outputs1 = llm.generate(stage1_prompts, _COT_PARAMS)
    descriptions = [_clean(o.outputs[0].text) for o in outputs1]

    # Stage 2 — classify with description in history
    print(f"[self_cot] Stage 2/2: classifying...")
    stage2_prompts = []
    for item, desc in zip(items, descriptions):
        desc_q = _desc_question(item["task_type"], item["crop"])
        cls_q  = _cls_question(item["class_names"], item["task_type"], item["crop"])
        msgs = _sys_msg() + [
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": desc_q}]},
            {"role": "assistant", "content": desc},
            {"role": "user",      "content": cls_q},
        ]
        stage2_prompts.append(_build_prompt(msgs, tokenizer, item["image"]))
    outputs2 = llm.generate(stage2_prompts, _CLS_PARAMS)
    predictions = [_clean(o.outputs[0].text) for o in outputs2]

    return [_make_row(item, "self_cot", pred)
            for item, pred in zip(items, predictions)]


def run_cot_yaml_batched(items: list[dict], llm, tokenizer) -> list[dict]:
    """
    Multi-stage CoT: images split by CoT path (disease_pest vs weed_identification).
    Each group runs N stages in parallel, then a final classification stage.
    Full conversation history is maintained per image across stages.
    """
    cot_sfx = _COT_PROMPTS["cot_classification_suffix"].strip()

    # Split by CoT path
    groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, item in enumerate(items):
        cot_key = _COT_PROMPTS["task_type_to_cot"].get(item["task_type"], "disease_pest")
        groups.setdefault(cot_key, []).append((idx, item))

    all_rows: list[tuple[int, dict]] = []  # (original_idx, row) to preserve order

    for cot_key, indexed_items in groups.items():
        steps = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        n_steps = len(steps)
        orig_idxs = [idx for idx, _ in indexed_items]
        group_items = [item for _, item in indexed_items]
        print(f"\n[cot_yaml] {cot_key}: {len(group_items)} images × {n_steps} steps")

        # Initialize per-image conversation history (image in first user turn)
        histories: list[list[dict]] = []
        for item in group_items:
            histories.append(_sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": steps[0]},
            ]}])

        # Run each CoT step as a batch
        for step_idx in range(n_steps):
            print(f"  Step {step_idx + 1}/{n_steps}...")
            prompts = [_build_prompt(h, tokenizer, item["image"])
                       for h, item in zip(histories, group_items)]
            outputs = llm.generate(prompts, _COT_PARAMS)
            responses = [_clean(o.outputs[0].text) for o in outputs]

            for i, resp in enumerate(responses):
                histories[i].append({"role": "assistant", "content": resp})
                if step_idx < n_steps - 1:
                    histories[i].append({"role": "user", "content": steps[step_idx + 1]})

        # Final classification turn (appended to full history)
        print(f"  Classification turn...")
        for i, item in enumerate(group_items):
            cls_q = _cls_question(item["class_names"], item["task_type"], item["crop"])
            histories[i].append({"role": "user", "content": cls_q + " " + cot_sfx})

        cls_prompts = [_build_prompt(h, tokenizer, item["image"])
                       for h, item in zip(histories, group_items)]
        cls_outputs = llm.generate(cls_prompts, _CLS_PARAMS)
        predictions = [_clean(o.outputs[0].text) for o in cls_outputs]

        for orig_idx, item, pred in zip(orig_idxs, group_items, predictions):
            all_rows.append((orig_idx, _make_row(item, "cot_yaml", pred)))

    # Restore original item order
    all_rows.sort(key=lambda x: x[0])
    return [row for _, row in all_rows]




# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """Compute direct and judged F1 per (condition, dataset) and overall per condition."""
    from collections import defaultdict
    by_cond_ds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_ds[(r["condition"], r["agml_dataset"])].append(r)

    per_condition: dict[str, dict] = {}
    for (cond, ds), ds_rows in sorted(by_cond_ds.items()):
        refs  = [r["ground_truth"]   for r in ds_rows]
        preds = [r["resolved_class"] if r["correct"] else "__" for r in ds_rows]
        m     = compute_classification_metrics(refs, preds)

        judged = [r for r in ds_rows if r.get("judged_correct", "") != ""]
        if judged:
            j_refs  = [r["ground_truth"] for r in judged]
            j_preds = [r["resolved_class"] if int(r["judged_correct"]) else "__"
                       for r in judged]
            jm = compute_classification_metrics(j_refs, j_preds)
            m["judge_f1_macro"] = jm["f1_macro"]
            m["judge_accuracy"] = jm["accuracy"]

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


def plot_hard_four_condition(
    active_datasets: list[str],
    vis_per_ds: dict[str, float],
    aug_per_ds: dict[str, float],
    cot_metrics: dict,
    output_dir: Path,
) -> None:
    """Grouped bar chart comparing all 4 conditions on the hard-subset datasets only."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Only keep datasets present in all sources (or at least vis baseline)
    datasets = [ds for ds in active_datasets if ds in vis_per_ds]
    if not datasets:
        print("  [warn] No common datasets for hard 4-condition chart")
        return

    CONDITIONS = [
        ("baseline",  "Vision Baseline",  "#1f77b4"),
        ("self_cot",  "Self-CoT",         "#ff7f0e"),
        ("cot_yaml",  "Structured CoT",   "#2ca02c"),
        ("desc_aug",  "Desc-Aug",         "#9467bd"),
    ]

    def _get(ds: str, key: str) -> float:
        if key == "baseline":
            return vis_per_ds.get(ds, 0.0)
        if key == "desc_aug":
            return aug_per_ds.get(ds, 0.0)
        per_ds = cot_metrics.get(key, {}).get("per_dataset", {})
        dm = per_ds.get(ds, {})
        return dm.get("judge_f1_macro") or dm.get("f1_macro", 0.0)

    x = np.arange(len(datasets))
    width = 0.8 / len(CONDITIONS)
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.8), 5))

    for i, (key, label, color) in enumerate(CONDITIONS):
        vals = [_get(ds, key) for ds in datasets]
        offset = (i - len(CONDITIONS) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    clean = [d.replace("_classification", "").replace("_", " ") for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(clean, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Judged F1  (Desc-Aug = direct F1)")
    ax.set_ylim(0, 1.15)
    ax.set_title("4-Condition Comparison — Hard Datasets Only (3 worst per task)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "hard_four_condition.png"
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

    colors = {"self_cot": "#1f77b4", "cot_yaml": "#ff7f0e"}
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


def plot_difficulty_bins(
    cot_metrics: dict,
    vis_per_ds: dict[str, float],
    output_dir: Path,
) -> None:
    """Bar chart: mean judged F1 per difficulty tercile (hard / medium / easy)."""
    import matplotlib.pyplot as plt

    conditions = [c for c in ("self_cot", "cot_yaml") if c in cot_metrics]
    if not conditions or not vis_per_ds:
        return

    all_ds = sorted(vis_per_ds, key=vis_per_ds.get)  # hardest first
    n = len(all_ds)
    bins = {"Hard": all_ds[:n//3], "Medium": all_ds[n//3:2*n//3], "Easy": all_ds[2*n//3:]}

    colors = {"vision": "#e67e22", "self_cot": "#1f77b4", "cot_yaml": "#ff7f0e"}
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
    ax.set_title("CoT Performance by Difficulty Bin")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "cot_difficulty_bins.png"
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

    colors = {"self_cot": "#1f77b4", "cot_yaml": "#ff7f0e"}
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
    parser.add_argument("--vis_dir",    default="outputs/vision_assessment/",
                        help="Vision assessment output dir — used to load baseline judged F1 for difficulty plots")
    parser.add_argument("--aug_dir",    default="outputs/desc_augmented_vision/",
                        help="Desc-aug output dir — used to load augmented F1 for hard-subset 4-condition chart")
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

    if args.plot_only:
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

        # Load vLLM once for all phases
        llm, tokenizer = load_llm(
            args.model,
            quantization="bitsandbytes" if args.use_4bit else None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )

        # Phase 1: self_cot (2 stages)
        print("\n=== Phase 1: self_cot ===")
        self_cot_rows = run_self_cot_batched(items, llm, tokenizer)

        # Phase 2: cot_yaml (N stages per CoT path)
        print("\n=== Phase 2: cot_yaml ===")
        cot_yaml_rows = run_cot_yaml_batched(items, llm, tokenizer)

        all_rows = self_cot_rows + cot_yaml_rows

        # Phase 3: judge (all rows, one batched call via shared apply_judge)
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
    plot_condition_comparison(metrics, output_dir / "cot_comparison.png")
    if vis_per_ds:
        plot_difficulty_correlation(metrics, vis_per_ds, output_dir)
        plot_difficulty_bins(metrics, vis_per_ds, output_dir)

    if args.hard_only and vis_per_ds:
        aug_per_ds = load_aug_per_dataset(Path(args.aug_dir), args.model)
        if not aug_per_ds:
            print("  [warn] No desc-aug metrics found — desc_aug bar will be zero")
        plot_hard_four_condition(active_datasets, vis_per_ds, aug_per_ds, metrics, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
