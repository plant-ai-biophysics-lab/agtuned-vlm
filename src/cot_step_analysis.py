"""
Analyse how judged F1 changes with each CoT YAML step.

Dataset selection is delta-based: load top-N gain / regression datasets from
existing cot_vision results (cot_yaml vs baseline) instead of random sampling.
Uses full test split (no per-class subsampling), matching cot_vision.

Two inference modes per image:
  cumulative  — predict after steps 1, 1+2, 1+2+3, ... (growing context)
  independent — predict after each step alone (no prior context)

Also produces the same response-card PNGs + CSVs as cot_vision/delta_examples/
for all three methods: desc_aug, self_cot (from existing CSVs), cot_yaml (from
cumulative final-step rows from this run).

Usage:
  uv run python src/cot_step_analysis.py \\
    --model google/gemma-4-26B-A4B-it \\
    --vis_dir outputs/vision_assessment/ \\
    --aug_dir outputs/desc_augmented_vision/ \\
    --cot_dir outputs/cot_vision/ \\
    --top_n 5 \\
    --output_dir outputs/cot_step_analysis/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42

TASK_LABEL = {
    "disease":             "Disease",
    "pest_damage":         "Pest Damage",
    "weed_identification": "Weed ID",
}

CLS_CSV_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "mode", "step_idx",
    "prediction", "reasoning", "ground_truth", "correct", "contains_match",
    "judged_correct", "judge_reasoning",
]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLS_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ── Dataset selection ─────────────────────────────────────────────────────────

def build_task_datasets(config_path: str = "src/config/datasets.yaml") -> dict[str, list[str]]:
    """Build {task_type: [ds_name, ...]} from config, including class_override datasets."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    task_datasets: dict[str, set[str]] = defaultdict(set)
    for d in cfg["datasets"]:
        if d.get("skip"):
            continue
        name = d["agml_name"]
        task_datasets[d["task_type"]].add(name)
        for task in (d.get("class_overrides") or {}).values():
            task_datasets[task].add(name)

    return {task: sorted(names) for task, names in task_datasets.items()
            if task in ("disease", "pest_damage", "weed_identification")}


def select_datasets_delta(
    vis_dir: str,
    cot_dir: str,
    model_id: str,
    top_n: int = 5,
    config_path: str = "src/config/datasets.yaml",
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Return (gains, regressions, ds_to_task, ds_to_crop, deltas, baseline_f1)
    based on cot_yaml judged F1 delta vs baseline vision-assessment results.
    """
    import yaml
    from sklearn.metrics import f1_score as sk_f1

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"] if not d.get("skip")}

    # Baseline judged F1 per dataset
    safe = model_id.replace("/", "_")
    vis_path = Path(vis_dir) / f"cls_{safe}.csv"
    if not vis_path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {vis_path}")
    vis_rows = _read_csv(vis_path)

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in vis_rows:
        by_ds[r["agml_dataset"]].append(r)

    baseline_f1: dict[str, float] = {}
    for ds, rows in by_ds.items():
        judged = [r for r in rows if r.get("judged_correct", "") != ""]
        if judged:
            refs  = [r["class_label"] for r in judged]
            preds = [r["class_label"] if int(r["judged_correct"]) else "" for r in judged]
            baseline_f1[ds] = sk_f1(refs, preds, average="macro", zero_division=0)

    # cot_yaml per-dataset F1 from metrics JSON
    metrics_path = Path(cot_dir) / "cot_vision_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"cot_vision_metrics.json not found: {metrics_path}")
    data = json.loads(metrics_path.read_text())
    cot_yaml_per_ds = (
        data.get("conditions", {}).get("cot_yaml", {}).get("per_dataset", {})
        or data.get("cot_yaml", {}).get("per_dataset", {})
    )

    deltas: dict[str, float] = {}
    for ds, m in cot_yaml_per_ds.items():
        if ds in baseline_f1:
            cot_f1 = m.get("judge_f1_macro") or m.get("f1_macro", 0.0)
            if cot_f1 is not None:
                deltas[ds] = float(cot_f1) - baseline_f1[ds]

    sorted_ds = sorted(deltas, key=lambda d: deltas[d])
    regressions = sorted_ds[:top_n]
    gains       = sorted_ds[-top_n:][::-1]

    all_selected = list(dict.fromkeys(gains + regressions))
    ds_to_task   = {ds: ds_info.get(ds, {}).get("task_type", "disease") for ds in all_selected}
    ds_to_crop   = {ds: ds_info.get(ds, {}).get("crop", "plant") for ds in all_selected}

    return gains, regressions, ds_to_task, ds_to_crop, deltas, baseline_f1


# ── Data loading ──────────────────────────────────────────────────────────────

def load_items(
    ds_name: str,
    task_type: str,
    crop: str | None,
    n_samples: int | None = None,
    class_overrides: dict[str, str] | None = None,
    seed: int = SEED,
    test_split: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Load items for ds_name filtered to classes that belong to task_type.

    When test_split is provided (dict mapping ds_name -> list of image filenames),
    only those images are included (mirroring cot_vision's test split behaviour).
    n_samples is applied before test_split filtering when test_split is absent.
    """
    from src.cot_vision import load_dataset_classes, DATASET_TASK_TYPES
    by_class = load_dataset_classes(ds_name, n_samples if test_split is None else None)

    # Filter to test split when provided
    if test_split and ds_name in test_split:
        allowed = set(test_split[ds_name])
        by_class = {
            cls: [p for p in paths if Path(p).name in allowed]
            for cls, paths in by_class.items()
        }

    primary_task = DATASET_TASK_TYPES.get(ds_name, (task_type, crop))[0]

    items: list[dict] = []
    for cls, paths in by_class.items():
        effective = (class_overrides or {}).get(cls, primary_task)
        if effective != task_type:
            continue
        for p in paths:
            items.append({
                "ds": ds_name, "task_type": task_type,
                "crop": crop, "cls": cls, "image_path": p,
            })
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


# ── vLLM helpers (imported from cot_vision) ──────────────────────────────────

def _import_cot_helpers():
    from src.cot_vision import (
        _sys_msg, _cls_question, _build_prompt, _clean, _load_image,
        _COT_PARAMS, _CLS_PARAMS, load_llm, destroy_llm,
    )
    from src.cot_vision import _COT_PROMPTS
    return (_sys_msg, _cls_question, _build_prompt, _clean, _load_image,
            _COT_PARAMS, _CLS_PARAMS, load_llm, destroy_llm, _COT_PROMPTS)


def _cot_key(task_type: str) -> str:
    return "weed_identification" if task_type == "weed_identification" else "disease_pest"


# ── Inference: cumulative ─────────────────────────────────────────────────────

def run_cumulative(
    items: list[dict],
    llm,
    tokenizer,
    prompt_batch_size: int = 128,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Predict after each step k using the growing conversation history.

    For step k the model has seen responses to steps 1..k then gets the
    classification question. The 'reasoning' column accumulates all Q&A pairs
    seen so far at each step.
    """
    (_sys_msg, _cls_question, _build_prompt, _clean, _load_image,
     _COT_PARAMS, _CLS_PARAMS, _, _, _COT_PROMPTS) = _import_cot_helpers()

    from vllm import SamplingParams
    _COT_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64,  repetition_penalty=1.1)

    by_cot: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_cot[_cot_key(item["task_type"])].append(item)

    all_rows: list[dict] = []

    for cot_key, group in by_cot.items():
        steps   = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        n_steps = len(steps)
        print(f"\n[cumulative] {cot_key}: {len(group)} images × {n_steps} steps")

        # Build initial history (step 0 question with image)
        histories: list[list[dict]] = [
            _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": steps[0]},
            ]}]
            for _ in group
        ]

        # Track per-image step responses for reasoning field
        step_responses_by_img: list[list[str]] = [[] for _ in group]

        for step_idx in range(n_steps):
            # Generate response for this step
            print(f"  Step {step_idx + 1}/{n_steps} — response generation")
            responses: list[str] = []
            for start in range(0, len(group), prompt_batch_size):
                batch  = group[start:start + prompt_batch_size]
                hist   = histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in zip(hist, batch)]
                responses.extend(_clean(o.outputs[0].text)
                                  for o in llm.generate(prompts, _COT_P))
                del prompts

            # Append response to each history and per-image tracker
            for i, resp in enumerate(responses):
                histories[i].append({"role": "assistant", "content": resp})
                step_responses_by_img[i].append(resp)

            # Classification after step_idx+1 steps
            print(f"  Step {step_idx + 1}/{n_steps} — classification")
            cls_histories = [
                h + [{"role": "user",
                      "content": _cls_question(item["task_type"], item["crop"])}]
                for h, item in zip(histories, group)
            ]
            preds: list[str] = []
            for start in range(0, len(group), prompt_batch_size):
                batch  = group[start:start + prompt_batch_size]
                cls_h  = cls_histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in zip(cls_h, batch)]
                preds.extend(_clean(o.outputs[0].text)
                              for o in llm.generate(prompts, _CLS_P))
                del prompts

            for i, (item, pred) in enumerate(zip(group, preds)):
                from src.cot_vision import resolve_class, contains_match
                gt = item["cls"]
                rc = resolve_class(pred, item["ds"])
                # Cumulative reasoning: all Q&A pairs up to this step
                reasoning = "\n\n".join(
                    f"Q: {steps[j]}\nA: {step_responses_by_img[i][j]}"
                    for j in range(step_idx + 1)
                )
                all_rows.append({
                    "agml_dataset":    item["ds"],
                    "class_label":     gt,
                    "task_type":       item["task_type"],
                    "crop":            item["crop"] or "",
                    "image_name":      str(item["image_path"]),
                    "mode":            "cumulative",
                    "step_idx":        step_idx + 1,
                    "prediction":      pred,
                    "reasoning":       reasoning,
                    "ground_truth":    gt,
                    "correct":         int(rc == gt) if rc else 0,
                    "contains_match":  int(contains_match(pred, gt)),
                    "judged_correct":  "",
                    "judge_reasoning": "",
                })

            # Prepare next step question (if not last)
            if step_idx < n_steps - 1:
                for i in range(len(group)):
                    histories[i].append({"role": "user",
                                         "content": steps[step_idx + 1]})

    if ckpt_path:
        _write_csv(all_rows, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path.name}")
    return all_rows


# ── Inference: independent ────────────────────────────────────────────────────

def run_independent(
    items: list[dict],
    llm,
    tokenizer,
    prompt_batch_size: int = 128,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Predict after each step k in isolation (no prior steps in history)."""
    (_sys_msg, _cls_question, _build_prompt, _clean, _load_image,
     _COT_PARAMS, _CLS_PARAMS, _, _, _COT_PROMPTS) = _import_cot_helpers()

    from vllm import SamplingParams
    _COT_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64,  repetition_penalty=1.1)

    by_cot: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_cot[_cot_key(item["task_type"])].append(item)

    all_rows: list[dict] = []

    for cot_key, group in by_cot.items():
        steps   = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        n_steps = len(steps)
        print(f"\n[independent] {cot_key}: {len(group)} images × {n_steps} steps")

        for step_idx, step_text in enumerate(steps):
            print(f"  Step {step_idx + 1}/{n_steps} — response generation")
            histories: list[list[dict]] = [
                _sys_msg() + [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": step_text},
                ]}]
                for _ in group
            ]

            responses: list[str] = []
            for start in range(0, len(group), prompt_batch_size):
                batch   = group[start:start + prompt_batch_size]
                hist    = histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in zip(hist, batch)]
                responses.extend(_clean(o.outputs[0].text)
                                  for o in llm.generate(prompts, _COT_P))
                del prompts

            print(f"  Step {step_idx + 1}/{n_steps} — classification")
            cls_histories = [
                h + [{"role": "assistant", "content": resp},
                     {"role": "user",
                      "content": _cls_question(item["task_type"], item["crop"])}]
                for h, resp, item in zip(histories, responses, group)
            ]
            preds: list[str] = []
            for start in range(0, len(group), prompt_batch_size):
                batch  = group[start:start + prompt_batch_size]
                cls_h  = cls_histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in zip(cls_h, batch)]
                preds.extend(_clean(o.outputs[0].text)
                              for o in llm.generate(prompts, _CLS_P))
                del prompts

            for item, pred, resp in zip(group, preds, responses):
                from src.cot_vision import resolve_class, contains_match
                gt = item["cls"]
                rc = resolve_class(pred, item["ds"])
                all_rows.append({
                    "agml_dataset":    item["ds"],
                    "class_label":     gt,
                    "task_type":       item["task_type"],
                    "crop":            item["crop"] or "",
                    "image_name":      str(item["image_path"]),
                    "mode":            "independent",
                    "step_idx":        step_idx + 1,
                    "prediction":      pred,
                    "reasoning":       f"Q: {step_text}\nA: {resp}",
                    "ground_truth":    gt,
                    "correct":         int(rc == gt) if rc else 0,
                    "contains_match":  int(contains_match(pred, gt)),
                    "judged_correct":  "",
                    "judge_reasoning": "",
                })

    if ckpt_path:
        _write_csv(all_rows, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path.name}")
    return all_rows


# ── Judging ───────────────────────────────────────────────────────────────────

def run_judge(
    rows: list[dict],
    llm,
    tokenizer,
    ckpt_path: Path | None = None,
) -> list[dict]:
    from src.cot_vision import _PROMPTS
    from src.common.metrics import apply_judge
    rows = apply_judge(
        rows=rows,
        llm=llm,
        tokenizer=tokenizer,
        system_prompt=_PROMPTS.get("judge_system_vision", ""),
        judge_template=_PROMPTS["judge_classification_template"],
        gt_key="ground_truth",
        pred_key="prediction",
    )
    if ckpt_path:
        _write_csv(rows, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path.name}")
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_step_f1(rows: list[dict]) -> dict[str, dict[str, dict[int, float]]]:
    """Return {mode: {dataset: {step_idx: judged_f1}}}."""
    from sklearn.metrics import f1_score

    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct", "") == "":
            continue
        key = (r["mode"], r["agml_dataset"], int(r["step_idx"]))
        by_key[key].append(r)

    result: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (mode, ds, step), ds_rows in by_key.items():
        judged = [r for r in ds_rows if r["judged_correct"] != ""]
        if not judged:
            continue
        refs  = [r["class_label"] for r in judged]
        preds = [r["class_label"] if int(r["judged_correct"]) else "" for r in judged]
        f1 = f1_score(refs, preds, average="macro", zero_division=0)
        result[mode][ds][step] = float(f1)

    return result


# ── Step charts ───────────────────────────────────────────────────────────────

_TASK_COLOR = {
    "disease":             "#2ca02c",
    "pest_damage":         "#d62728",
    "weed_identification": "#9467bd",
}


def _plot_step_chart(
    ax,
    step_f1_by_ds: dict[str, dict[int, float]],
    ds_to_task: dict[str, str],
    title: str,
    color_by: str = "task",
    palette: list[str] | None = None,
) -> None:
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    pal = palette or default_palette

    all_steps_sets = [set(v.keys()) for v in step_f1_by_ds.values() if v]
    if not all_steps_sets:
        return
    all_steps = sorted(set.union(*all_steps_sets))

    ds_list = sorted(step_f1_by_ds.keys())
    for i, ds in enumerate(ds_list):
        f1_by_step = step_f1_by_ds[ds]
        steps_sorted = sorted(f1_by_step)
        ys = [f1_by_step[s] for s in steps_sorted]
        color = (_TASK_COLOR.get(ds_to_task.get(ds, ""), pal[i % len(pal)])
                 if color_by == "task" else pal[i % len(pal)])
        label = ds.replace("_classification", "").replace("_", " ")
        ax.plot(steps_sorted, ys, marker="o", color=color,
                alpha=0.55, linewidth=1.2, label=label)

    mean_ys = []
    for s in all_steps:
        vals = [step_f1_by_ds[ds][s] for ds in ds_list if s in step_f1_by_ds[ds]]
        mean_ys.append(float(np.mean(vals)) if vals else np.nan)

    ax.plot(all_steps, mean_ys, marker="o", color="black",
            linewidth=2.2, label="mean", zorder=5)
    for s, y in zip(all_steps, mean_ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.2f}", (s, y), fontsize=7,
                        xytext=(0, 6), textcoords="offset points", ha="center")

    ax.set_xticks(all_steps)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Judged F1", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)


def plot_all(
    step_f1: dict[str, dict[str, dict[int, float]]],
    selected_datasets: dict[str, list[str]],
    ds_to_task: dict[str, str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["cumulative", "independent"]

    for mode in modes:
        fig, ax = plt.subplots(figsize=(8, 5))
        for task, task_color in _TASK_COLOR.items():
            ds_list = selected_datasets.get(task, [])
            all_steps_sets = [set(step_f1[mode].get(ds, {}).keys()) for ds in ds_list]
            all_steps_sets = [s for s in all_steps_sets if s]
            if not all_steps_sets:
                continue
            all_steps = sorted(set.union(*all_steps_sets))
            mean_ys = []
            for s in all_steps:
                vals = [step_f1[mode][ds][s]
                        for ds in ds_list if s in step_f1[mode].get(ds, {})]
                mean_ys.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(all_steps, mean_ys, marker="o", color=task_color,
                    linewidth=2, label=TASK_LABEL[task])
            for s, y in zip(all_steps, mean_ys):
                if not np.isnan(y):
                    ax.annotate(f"{y:.2f}", (s, y), fontsize=7,
                                xytext=(0, 6), textcoords="offset points", ha="center")

        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Mean Judged F1", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"CoT Step Analysis — {mode.capitalize()} (all tasks)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = output_dir / f"step_{mode}_overall.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    for mode in modes:
        for task, ds_list in selected_datasets.items():
            ds_f1 = {ds: step_f1[mode].get(ds, {}) for ds in ds_list}
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_step_chart(
                ax, ds_f1, ds_to_task,
                title=f"{mode.capitalize()} — {TASK_LABEL[task]}",
                color_by="dataset",
            )
            plt.tight_layout()
            path = output_dir / f"step_{mode}_{task}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")


# ── Response card helpers ─────────────────────────────────────────────────────

def _format_step_response(row: dict, method: str, max_chars: int = 280) -> str:
    """Format the response text for a card, showing reasoning for cot_yaml."""
    reasoning  = (row.get("reasoning") or "").strip()
    prediction = (row.get("prediction") or "").strip()
    if method == "cot_yaml" and reasoning:
        return reasoning[:max_chars]
    if method in ("desc_aug", "self_cot") and reasoning:
        prefix = "[Desc] " if method == "desc_aug" else "[CoT] "
        return (prefix + reasoning)[:max_chars]
    return prediction[:max_chars]


def _load_existing_method_rows(
    method: str,
    aug_dir: str | None,
    cot_dir: str | None,
    step_output_dir: Path,
    model_id: str,
    selected_datasets: set[str],
) -> list[dict]:
    """Load existing classification rows for a method, filtered to selected datasets."""
    safe = model_id.replace("/", "_")

    if method == "desc_aug" and aug_dir:
        p = Path(aug_dir) / f"aug_cls_all_{safe}.csv"
        if p.exists():
            return [r for r in _read_csv(p)
                    if r.get("condition") == "desc_aug"
                    and r.get("agml_dataset") in selected_datasets]
        print(f"  [warn] desc_aug CSV not found: {p}")
        return []

    if method == "self_cot" and cot_dir:
        # Try both cot_cls_*.csv and ckpt_judged.csv in cot_dir
        p = Path(cot_dir) / f"cot_cls_{safe}.csv"
        if p.exists():
            return [r for r in _read_csv(p)
                    if r.get("condition") == "self_cot"
                    and r.get("agml_dataset") in selected_datasets]
        p2 = Path(cot_dir) / "ckpt_judged.csv"
        if p2.exists():
            return [r for r in _read_csv(p2)
                    if r.get("condition") == "self_cot"
                    and r.get("agml_dataset") in selected_datasets]
        print(f"  [warn] self_cot CSV not found in {cot_dir}")
        return []

    if method == "cot_yaml":
        judged_path = step_output_dir / "ckpt_judged.csv"
        if not judged_path.exists():
            print(f"  [warn] cot_yaml judged checkpoint not found: {judged_path}")
            return []
        all_step = _read_csv(judged_path)
        # Keep only cumulative final-step rows per (ds, image_name, class_label)
        max_step: dict[tuple, int] = {}
        for r in all_step:
            if r.get("mode") != "cumulative":
                continue
            key = (r["agml_dataset"], r["image_name"], r["class_label"])
            max_step[key] = max(max_step.get(key, 0), int(r.get("step_idx", 0)))
        return [
            r for r in all_step
            if r.get("mode") == "cumulative"
            and r.get("agml_dataset") in selected_datasets
            and int(r.get("step_idx", 0)) == max_step.get(
                (r["agml_dataset"], r["image_name"], r["class_label"]), 0)
        ]

    return []


def plot_response_cards(
    gain_ds: list[str],
    regression_ds: list[str],
    deltas: dict[str, float],
    baseline_f1: dict[str, float],
    method: str,
    rows: list[dict],
    output_dir: Path,
    examples_per_ds: int = 2,
) -> None:
    """Produce PNG response cards + CSV for one method, both gain and regression directions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    METHOD_LABEL = {"desc_aug": "Desc Aug", "self_cot": "Self CoT", "cot_yaml": "CoT YAML"}
    DIR_LABEL    = {"gain": "Top Gains", "regression": "Top Regressions"}

    FIG_W    = 13.5
    MARGIN_X = 0.45
    HDR_H    = 0.42
    CARD_H   = 1.55
    DS_GAP   = 0.28

    for direction, ds_list in [("gain", gain_ds), ("regression", regression_ds)]:
        if not ds_list:
            continue

        by_ds: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get("agml_dataset") in ds_list:
                by_ds[r["agml_dataset"]].append(r)

        present = [d for d in ds_list if d in by_ds and by_ds[d]]
        if not present:
            print(f"  [warn] No rows found for {method} {direction}")
            continue

        n_ds    = len(present)
        total_h = n_ds * (HDR_H + examples_per_ds * CARD_H + DS_GAP) + 0.5

        fig = plt.figure(figsize=(FIG_W, max(total_h, 4)))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, total_h)
        ax.axis("off")

        title_str = f"{METHOD_LABEL.get(method, method)} — {DIR_LABEL[direction]}"
        ax.text(FIG_W / 2, total_h - 0.12, title_str,
                ha="center", va="top", fontsize=12, fontweight="bold")

        y        = total_h - 0.35
        csv_rows: list[dict] = []

        for ds in ds_list:
            ds_rows = by_ds.get(ds)
            if not ds_rows:
                continue

            delta  = deltas.get(ds, 0.0)
            base   = baseline_f1.get(ds, 0.0)
            cot_f1 = base + delta
            sign   = "▲" if delta >= 0 else "▼"
            ds_label = ds.replace("_classification", "").replace("_", " ")

            ax.text(MARGIN_X, y,
                    f"{ds_label}   {sign}{abs(delta):.3f}  (base={base:.3f} → {cot_f1:.3f})",
                    fontsize=9, fontweight="bold", va="top", color="#333333")
            y -= HDR_H

            rng_local = random.Random(42)
            correct   = [r for r in ds_rows if str(r.get("judged_correct", "")) == "1"]
            incorrect = [r for r in ds_rows if str(r.get("judged_correct", "")) == "0"]
            n_each    = max(1, examples_per_ds // 2)
            chosen: list[dict] = []
            chosen += rng_local.sample(correct,   min(n_each, len(correct)))
            chosen += rng_local.sample(incorrect, min(examples_per_ds - len(chosen), len(incorrect)))
            if len(chosen) < examples_per_ds:
                pool    = [r for r in ds_rows if r not in chosen]
                chosen += rng_local.sample(pool, min(examples_per_ds - len(chosen), len(pool)))

            for r in chosen[:examples_per_ds]:
                is_correct = str(r.get("judged_correct", "")) == "1"
                face_color = "#e8f5e9" if is_correct else "#ffebee"
                edge_color = "#4caf50" if is_correct else "#f44336"
                verdict    = "CORRECT" if is_correct else "INCORRECT"
                vc         = "#2e7d32" if is_correct else "#c62828"

                patch = mpatches.FancyBboxPatch(
                    (MARGIN_X, y - CARD_H + 0.06),
                    FIG_W - 2 * MARGIN_X, CARD_H - 0.12,
                    boxstyle="round,pad=0.04", linewidth=1.2,
                    edgecolor=edge_color, facecolor=face_color,
                )
                ax.add_patch(patch)

                label_x = MARGIN_X + 0.15
                label_y = y - 0.20

                gt_text   = r.get("ground_truth") or r.get("class_label", "?")
                pred_text = (r.get("prediction") or "")[:60]
                ax.text(label_x,       label_y, f"GT: {gt_text}",
                        fontsize=7.5, va="top", color="#555")
                ax.text(label_x + 3.5, label_y, f"Pred: {pred_text}",
                        fontsize=7.5, va="top", color="#222")
                ax.text(label_x + 9.5, label_y, verdict,
                        fontsize=7.5, va="top", color=vc, fontweight="bold")

                resp_text = _format_step_response(r, method)
                ax.text(label_x, label_y - 0.26, resp_text,
                        fontsize=6.5, va="top", color="#444", style="italic")

                y -= CARD_H
                csv_rows.append({**r, "delta": delta, "direction": direction})

            y -= DS_GAP

        cards_dir = output_dir / "delta_examples"
        cards_dir.mkdir(parents=True, exist_ok=True)

        png_path = cards_dir / f"{method}_{direction}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png_path}")

        if csv_rows:
            all_keys = list(csv_rows[0].keys())
            csv_path = cards_dir / f"{method}_{direction}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(csv_rows)
            print(f"  Saved: {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CoT step-level F1 analysis")
    parser.add_argument("--model",      required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir", default="outputs/cot_step_analysis/")
    parser.add_argument("--config",     default="src/config/datasets.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--from_ckpt",  action="store_true",
                        help="Skip inference if checkpoints exist")
    parser.add_argument("--judge_only", action="store_true",
                        help="Skip inference, load ckpts and run judge only")
    parser.add_argument("--plot_only",  action="store_true",
                        help="Skip inference+judge, load judged ckpts and plot only")
    # Delta-based dataset selection (replaces --n_datasets / --seed / --n_samples)
    parser.add_argument("--vis_dir",    default=None,
                        help="vision_assessment output dir (baseline CSVs)")
    parser.add_argument("--aug_dir",    default=None,
                        help="desc_augmented_vision output dir (for response cards)")
    parser.add_argument("--cot_dir",    default=None,
                        help="cot_vision output dir (for cot_vision_metrics.json + self_cot rows)")
    parser.add_argument("--top_n",      type=int, default=5,
                        help="Top-N gain + regression datasets to select (default 5)")
    parser.add_argument("--test_split", default=None,
                        help="Path to test_split.json {ds_name: [filenames]}")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cumulative  = output_dir / "ckpt_cumulative.csv"
    ckpt_independent = output_dir / "ckpt_independent.csv"
    ckpt_judged      = output_dir / "ckpt_judged.csv"

    # ── Load config ────────────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    # ── Test split ─────────────────────────────────────────────────────────────
    test_split: dict[str, list[str]] | None = None
    if args.test_split and Path(args.test_split).exists():
        test_split = json.loads(Path(args.test_split).read_text())
        print(f"Loaded test split: {len(test_split)} datasets")

    # ── Dataset selection ──────────────────────────────────────────────────────
    gains: list[str] = []
    regressions: list[str] = []
    baseline_f1_map: dict[str, float] = {}
    deltas: dict[str, float] = {}

    if args.vis_dir and args.cot_dir:
        print(f"\n=== Delta-based dataset selection (top {args.top_n}) ===")
        gains, regressions, ds_to_task, ds_to_crop, deltas, baseline_f1_map = (
            select_datasets_delta(
                vis_dir=args.vis_dir,
                cot_dir=args.cot_dir,
                model_id=args.model,
                top_n=args.top_n,
                config_path=args.config,
            )
        )
        print(f"  Gains ({len(gains)}):       {gains}")
        print(f"  Regressions ({len(regressions)}): {regressions}")

        all_selected = list(dict.fromkeys(gains + regressions))
        selected: dict[str, list[str]] = {}
        for ds in all_selected:
            t = ds_to_task[ds]
            selected.setdefault(t, []).append(ds)
    else:
        # Fallback: random selection (original behaviour, for dev/testing)
        print("\n=== Random dataset selection (--vis_dir/--cot_dir not provided) ===")
        import random as _random
        task_datasets = build_task_datasets(args.config)
        rng = _random.Random(SEED)
        selected = {task: rng.sample(names, min(3, len(names)))
                    for task, names in task_datasets.items()}
        all_selected = [ds for names in selected.values() for ds in names]
        ds_to_task   = {ds: t for t, names in selected.items() for ds in names}
        ds_to_crop   = {ds: ds_info.get(ds, {}).get("crop", "plant") for ds in all_selected}
        print(f"  Selected ({len(all_selected)}): {all_selected}")

    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task}: {names}")

    # ── Plot-only shortcut ─────────────────────────────────────────────────────
    if args.plot_only:
        rows = _read_csv(ckpt_judged)
        step_f1 = compute_step_f1(rows)
        print("\n=== Plotting step charts ===")
        plot_all(step_f1, selected, ds_to_task, output_dir)
        if gains or regressions:
            selected_set = set(all_selected)
            print("\n=== Plotting response cards ===")
            for method in ("desc_aug", "self_cot", "cot_yaml"):
                method_rows = _load_existing_method_rows(
                    method, args.aug_dir, args.cot_dir, output_dir,
                    args.model, selected_set,
                )
                if method_rows:
                    plot_response_cards(
                        gains, regressions, deltas, baseline_f1_map,
                        method, method_rows, output_dir,
                    )
        return

    # ── Phase 1 & 2: inference ─────────────────────────────────────────────────
    cum_rows: list[dict] = []
    ind_rows: list[dict] = []

    skip_inference = args.from_ckpt and ckpt_cumulative.exists() and ckpt_independent.exists()

    if skip_inference:
        print("\n=== Inference [SKIPPED — checkpoints found] ===")
        cum_rows = _read_csv(ckpt_cumulative)
        ind_rows = _read_csv(ckpt_independent)
    elif not args.judge_only:
        all_items: list[dict] = []
        for ds in all_selected:
            task    = ds_to_task[ds]
            crop    = ds_to_crop.get(ds, "plant")
            info    = ds_info.get(ds, {})
            overrides = info.get("class_overrides", {})
            items   = load_items(ds, task, crop, n_samples=None,
                                 class_overrides=overrides, seed=SEED,
                                 test_split=test_split)
            all_items.extend(items)
        print(f"Total items loaded: {len(all_items)}")

        from src.cot_vision import load_llm, destroy_llm
        print(f"\n=== Loading model: {args.model} ===")
        llm, tokenizer = load_llm(args.model)
        t0 = time.perf_counter()

        if args.from_ckpt and ckpt_cumulative.exists():
            print("\n=== Phase 1: cumulative [SKIPPED] ===")
            cum_rows = _read_csv(ckpt_cumulative)
        else:
            print(f"\n=== Phase 1: cumulative ({len(all_items)} images) ===")
            cum_rows = run_cumulative(all_items, llm, tokenizer,
                                      prompt_batch_size=args.batch_size,
                                      ckpt_path=ckpt_cumulative)

        if args.from_ckpt and ckpt_independent.exists():
            print("\n=== Phase 2: independent [SKIPPED] ===")
            ind_rows = _read_csv(ckpt_independent)
        else:
            print(f"\n=== Phase 2: independent ({len(all_items)} images) ===")
            ind_rows = run_independent(all_items, llm, tokenizer,
                                       prompt_batch_size=args.batch_size,
                                       ckpt_path=ckpt_independent)

        elapsed = time.perf_counter() - t0
        print(f"\nInference done in {elapsed:.1f}s")

        # Phase 3: judge
        if args.from_ckpt and ckpt_judged.exists():
            print("\n=== Phase 3: judge [SKIPPED] ===")
            all_rows = _read_csv(ckpt_judged)
        else:
            all_rows = cum_rows + ind_rows
            print(f"\n=== Phase 3: judge ({len(all_rows)} rows) ===")
            all_rows = run_judge(all_rows, llm, tokenizer, ckpt_path=ckpt_judged)

        destroy_llm(llm)
        del tokenizer, llm
        gc.collect()

    # ── Deferred judge (judge_only or after from_ckpt inference skip) ──────────
    all_rows: list[dict] = []
    need_judge = (args.judge_only or skip_inference) and not args.plot_only

    if need_judge:
        if ckpt_judged.exists():
            print("\n=== Phase 3: judge [SKIPPED — checkpoint found] ===")
            all_rows = _read_csv(ckpt_judged)
        else:
            from src.cot_vision import load_llm, destroy_llm
            print(f"\n=== Loading model for judge: {args.model} ===")
            llm, tokenizer = load_llm(args.model)
            all_rows = cum_rows + ind_rows
            print(f"\n=== Phase 3: judge ({len(all_rows)} rows) ===")
            all_rows = run_judge(all_rows, llm, tokenizer, ckpt_path=ckpt_judged)
            destroy_llm(llm)
            del tokenizer, llm
            gc.collect()

    # ── Phase 4: metrics + step charts ────────────────────────────────────────
    print("\n=== Computing step F1 ===")
    step_f1 = compute_step_f1(all_rows)
    for mode in ("cumulative", "independent"):
        for ds, by_step in step_f1.get(mode, {}).items():
            steps_str = ", ".join(f"s{s}={v:.3f}" for s, v in sorted(by_step.items()))
            print(f"  [{mode}] {ds}: {steps_str}")

    print("\n=== Plotting step charts ===")
    plot_all(step_f1, selected, ds_to_task, output_dir)

    # ── Phase 5: response cards ────────────────────────────────────────────────
    if gains or regressions:
        selected_set = set(all_selected)
        print("\n=== Plotting response cards ===")
        for method in ("desc_aug", "self_cot", "cot_yaml"):
            method_rows = _load_existing_method_rows(
                method, args.aug_dir, args.cot_dir, output_dir,
                args.model, selected_set,
            )
            if not method_rows:
                print(f"  [skip] No rows for {method}")
                continue
            print(f"  {method}: {len(method_rows)} rows")
            plot_response_cards(
                gains, regressions, deltas, baseline_f1_map,
                method, method_rows, output_dir,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
