"""
Analyse how judged F1 changes with each CoT YAML step.

Two modes per image:
  cumulative  — predict after steps 1, 1+2, 1+2+3, ... (growing context)
  independent — predict after each step alone (no prior context)

Produces 8 charts:
  step_cumulative_overall.png
  step_independent_overall.png
  step_cumulative_disease.png
  step_cumulative_pest_damage.png
  step_cumulative_weed_identification.png
  step_independent_disease.png
  step_independent_pest_damage.png
  step_independent_weed_identification.png

Usage:
  uv run python src/cot_step_analysis.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/cot_step_analysis/ \\
    --n_samples 50
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

N_DATASETS_PER_TASK = 3
SEED = 42

TASK_LABEL = {
    "disease":             "Disease",
    "pest_damage":         "Pest Damage",
    "weed_identification": "Weed ID",
}

CLS_CSV_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "mode", "step_idx",
    "prediction", "ground_truth", "correct", "contains_match",
    "judged_correct", "judge_reasoning",
]


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
        # Also include datasets that have class_overrides for other task types
        for task in (d.get("class_overrides") or {}).values():
            task_datasets[task].add(name)

    return {task: sorted(names) for task, names in task_datasets.items()
            if task in ("disease", "pest_damage", "weed_identification")}


def select_datasets(
    n_per_task: int = N_DATASETS_PER_TASK,
    seed: int = SEED,
    config_path: str = "src/config/datasets.yaml",
) -> dict[str, list[str]]:
    task_datasets = build_task_datasets(config_path)
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for task, names in task_datasets.items():
        k = min(n_per_task, len(names))
        selected[task] = rng.sample(names, k)
    return selected


# ── Data loading (mirrors cot_vision.load_dataset_classes) ────────────────────

def load_items(
    ds_name: str,
    task_type: str,
    crop: str | None,
    n_samples: int | None,
    class_overrides: dict[str, str] | None = None,
    seed: int = SEED,
) -> list[dict]:
    """Load items for ds_name, filtered to classes that belong to task_type.

    For datasets whose primary task_type differs (e.g. tomato_leaf_disease
    selected for pest_damage via class_overrides), only classes whose
    effective task_type matches the requested task_type are included.
    """
    from src.cot_vision import load_dataset_classes, DATASET_TASK_TYPES
    by_class = load_dataset_classes(ds_name, n_samples)

    primary_task = DATASET_TASK_TYPES.get(ds_name, (task_type, crop))[0]

    items: list[dict] = []
    for cls, paths in by_class.items():
        # Effective task for this class
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

    For step k, the model has seen responses to steps 1..k then gets
    the classification question.
    """
    (_sys_msg, _cls_question, _build_prompt, _clean, _load_image,
     _COT_PARAMS, _CLS_PARAMS, _, _, _COT_PROMPTS) = _import_cot_helpers()

    from vllm import SamplingParams
    _COT_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64,  repetition_penalty=1.1)

    # Group by cot_key so step counts are consistent
    by_cot: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_cot[_cot_key(item["task_type"])].append(item)

    all_rows: list[dict] = []

    for cot_key, group in by_cot.items():
        steps = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        n_steps = len(steps)
        print(f"\n[cumulative] {cot_key}: {len(group)} images × {n_steps} steps")

        # Build initial history (step 0 question with image)
        histories: list[list[dict]] = [
            _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": steps[0]},
            ]}]
            for _ in group
        ]

        for step_idx in range(n_steps):
            # Generate response for this step
            print(f"  Step {step_idx + 1}/{n_steps} — response generation")
            responses: list[str] = []
            for start in range(0, len(group), prompt_batch_size):
                batch = group[start:start + prompt_batch_size]
                hist  = histories[start:start + prompt_batch_size]
                prompts = [_build_prompt(h, tokenizer, _load_image(item))
                           for h, item in zip(hist, batch)]
                responses.extend(_clean(o.outputs[0].text)
                                  for o in llm.generate(prompts, _COT_P))
                del prompts

            # Append response to each history
            for i, resp in enumerate(responses):
                histories[i].append({"role": "assistant", "content": resp})

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

            for item, pred in zip(group, preds):
                from src.cot_vision import resolve_class, contains_match
                gt = item["cls"]
                rc = resolve_class(pred, item["ds"])
                all_rows.append({
                    "agml_dataset":  item["ds"],
                    "class_label":   gt,
                    "task_type":     item["task_type"],
                    "crop":          item["crop"] or "",
                    "image_name":    str(item["image_path"]),
                    "mode":          "cumulative",
                    "step_idx":      step_idx + 1,
                    "prediction":    pred,
                    "ground_truth":  gt,
                    "correct":       int(rc == gt) if rc else 0,
                    "contains_match": int(contains_match(pred, gt)),
                    "judged_correct": "",
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
        steps = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        n_steps = len(steps)
        print(f"\n[independent] {cot_key}: {len(group)} images × {n_steps} steps")

        for step_idx, step_text in enumerate(steps):
            print(f"  Step {step_idx + 1}/{n_steps} — response generation")
            # Fresh history: just this one step (with image)
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

            for item, pred in zip(group, preds):
                from src.cot_vision import resolve_class, contains_match
                gt = item["cls"]
                rc = resolve_class(pred, item["ds"])
                all_rows.append({
                    "agml_dataset":  item["ds"],
                    "class_label":   gt,
                    "task_type":     item["task_type"],
                    "crop":          item["crop"] or "",
                    "image_name":    str(item["image_path"]),
                    "mode":          "independent",
                    "step_idx":      step_idx + 1,
                    "prediction":    pred,
                    "ground_truth":  gt,
                    "correct":       int(rc == gt) if rc else 0,
                    "contains_match": int(contains_match(pred, gt)),
                    "judged_correct": "",
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

    # group by (mode, dataset, step_idx, class_label)
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


# ── Plotting ──────────────────────────────────────────────────────────────────

_MODE_COLOR = {"cumulative": "#1f77b4", "independent": "#ff7f0e"}
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
    color_by: str = "task",  # "task" or "dataset"
    palette: list[str] | None = None,
) -> None:
    """Draw lines (one per dataset) + a bold mean line onto ax."""
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

    # Mean across datasets at each step
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

    # ── 2 overall charts ──────────────────────────────────────────────────────
    for mode in modes:
        fig, ax = plt.subplots(figsize=(8, 5))
        # one line per task (mean across selected datasets for that task)
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

    # ── 6 per-task charts ─────────────────────────────────────────────────────
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLS_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CoT step-level F1 analysis")
    parser.add_argument("--model",       required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir",  default="outputs/cot_step_analysis/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--n_samples",   type=int, default=50,
                        help="Max images per class per dataset (default 50)")
    parser.add_argument("--n_datasets",  type=int, default=N_DATASETS_PER_TASK,
                        help="Datasets per task (default 3)")
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--from_ckpt",   action="store_true",
                        help="Skip inference if checkpoints exist")
    parser.add_argument("--judge_only",  action="store_true",
                        help="Skip inference, load ckpts and run judge only")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip inference+judge, load judged ckpts and plot only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cumulative = output_dir / "ckpt_cumulative.csv"
    ckpt_independent = output_dir / "ckpt_independent.csv"
    ckpt_judged = output_dir / "ckpt_judged.csv"

    # ── Dataset selection ──────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed,
                               config_path=args.config)
    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")

    ds_to_task = {ds: task for task, names in selected.items() for ds in names}

    skip_inference = args.plot_only or (
        args.from_ckpt and ckpt_cumulative.exists() and ckpt_independent.exists()
    )
    skip_judge = args.plot_only or (args.from_ckpt and ckpt_judged.exists())

    if args.plot_only:
        rows = _read_csv(ckpt_judged)
        step_f1 = compute_step_f1(rows)
        print("\n=== Plotting ===")
        plot_all(step_f1, selected, ds_to_task, output_dir)
        return

    # ── Phase 1 & 2: inference ─────────────────────────────────────────────────
    cum_rows: list[dict] = []
    ind_rows: list[dict] = []

    if args.plot_only or (args.from_ckpt and ckpt_cumulative.exists() and ckpt_independent.exists()):
        print("\n=== Inference [SKIPPED — checkpoints found] ===")
        cum_rows = _read_csv(ckpt_cumulative)
        ind_rows = _read_csv(ckpt_independent)
    else:
        # Load dataset items only when inference is actually needed
        all_items: list[dict] = []
        for task, ds_names in selected.items():
            for ds in ds_names:
                info = ds_info.get(ds, {})
                crop = info.get("crop", "plant")
                class_overrides = info.get("class_overrides", {})
                items = load_items(ds, task, crop, n_samples=args.n_samples,
                                   class_overrides=class_overrides, seed=args.seed)
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

        # ── Phase 3: judge ─────────────────────────────────────────────────────
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

    # ── Phase 3 (judge_only path) ──────────────────────────────────────────────
    if args.judge_only:
        from src.cot_vision import load_llm, destroy_llm
        llm, tokenizer = load_llm(args.model)
        all_rows = cum_rows + ind_rows
        print(f"\n=== Judge only ({len(all_rows)} rows) ===")
        all_rows = run_judge(all_rows, llm, tokenizer, ckpt_path=ckpt_judged)
        destroy_llm(llm)
        del tokenizer, llm
    else:
        if ckpt_judged.exists() and not (cum_rows and ind_rows):
            all_rows = _read_csv(ckpt_judged)

    # ── Phase 4: metrics + plots ───────────────────────────────────────────────
    print("\n=== Computing step F1 ===")
    step_f1 = compute_step_f1(all_rows)
    for mode in ("cumulative", "independent"):
        for ds, by_step in step_f1.get(mode, {}).items():
            steps_str = ", ".join(f"s{s}={v:.3f}" for s, v in sorted(by_step.items()))
            print(f"  [{mode}] {ds}: {steps_str}")

    print("\n=== Plotting ===")
    plot_all(step_f1, selected, ds_to_task, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
