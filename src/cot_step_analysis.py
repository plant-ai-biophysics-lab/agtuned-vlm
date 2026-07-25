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
import re
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
    "image_name", "mode", "step_idx", "rotation", "question_id",
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
    n_per_class: int | None = None,
) -> list[dict]:
    """Load items for ds_name filtered to classes that belong to task_type.

    When test_split is provided (dict mapping ds_name -> list of image filenames),
    only those images are included (mirroring cot_vision's test split behaviour).
    n_samples is applied before test_split filtering when test_split is absent.
    n_per_class caps the number of images sampled per class (random subsample,
    applied after test_split filtering) — used by --swap_order for fast iteration.
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
    rng = random.Random(seed)

    items: list[dict] = []
    for cls, paths in by_class.items():
        effective = (class_overrides or {}).get(cls, primary_task)
        if effective != task_type:
            continue
        paths = list(paths)
        if n_per_class is not None and len(paths) > n_per_class:
            paths = rng.sample(paths, n_per_class)
        for p in paths:
            items.append({
                "ds": ds_name, "task_type": task_type,
                "crop": crop, "cls": cls, "image_path": p,
            })
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
    step_order: list[int] | None = None,
    rotation_tag: int | None = None,
) -> list[dict]:
    """Predict after each step k using the growing conversation history.

    For step k the model has seen responses to steps 1..k then gets the
    classification question. The 'reasoning' column accumulates all Q&A pairs
    seen so far at each step.

    step_order, if given, is a permutation of question indices (0-based, into
    the track's default step list) — used to test whether results depend on
    which question is asked first. All items passed in a single call must
    belong to the same track (disease_pest or weed_identification) when
    step_order is given, since a permutation is track-specific.
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
        orig_steps = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
        steps   = [orig_steps[i] for i in step_order] if step_order else orig_steps
        n_steps = len(steps)
        tag = f" [rotation {rotation_tag}]" if rotation_tag is not None else ""
        print(f"\n[cumulative] {cot_key}: {len(group)} images × {n_steps} steps{tag}")

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

            orig_idx_at_pos = step_order[step_idx] if step_order else step_idx
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
                    "rotation":        str(rotation_tag) if rotation_tag is not None else "",
                    "question_id":     str(orig_idx_at_pos),
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


# ── Inference: swap-order (anchor-rate robustness check) ──────────────────────

def run_swap_order(
    items: list[dict],
    llm,
    tokenizer,
    prompt_batch_size: int = 128,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Run cumulative inference once per cyclic rotation of each track's step
    order, so every question gets a turn going first. Checkpoints after each
    rotation (accumulated rows) so a long run can be resumed/inspected.
    """
    (*_, _COT_PROMPTS) = _import_cot_helpers()

    by_track: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_track[_cot_key(item["task_type"])].append(item)

    all_rows: list[dict] = []
    for track, group in by_track.items():
        n_steps = len(_COT_PROMPTS["thinking_prompts"][track]["steps"])
        print(f"\n=== Swap-order track: {track}  "
              f"({len(group)} images × {n_steps} rotations) ===")

        for rot in range(n_steps):
            step_order = list(range(rot, n_steps)) + list(range(rot))
            print(f"\n--- Rotation {rot + 1}/{n_steps}: question {rot + 1} goes first ---")
            rows = run_cumulative(
                group, llm, tokenizer,
                prompt_batch_size=prompt_batch_size,
                step_order=step_order, rotation_tag=rot,
            )
            all_rows.extend(rows)
            if ckpt_path:
                _write_csv(all_rows, ckpt_path)
                print(f"  Checkpoint saved ({len(all_rows):,} rows so far): {ckpt_path.name}")

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


def compute_swap_anchor_rates(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Return {track: {rotation: {anchor_rate, recovery_rate, n_step1_correct,
    n_step1_wrong}}} from swap-order rows — i.e. does the anchor rate hold up
    regardless of which question was asked first?
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct", "") == "":
            continue
        key = (r.get("agml_dataset", ""), r.get("image_name", ""),
               r.get("class_label", ""), r.get("rotation", ""))
        groups[key].append(r)

    by_track_rot: dict[tuple, dict] = defaultdict(lambda: {
        "step1_correct": 0, "step1_correct_final_correct": 0,
        "step1_wrong": 0, "step1_wrong_final_correct": 0,
    })
    for grp in groups.values():
        grp.sort(key=lambda r: int(r["step_idx"]))
        if len(grp) < 2:
            continue
        track = _cot_key(grp[0]["task_type"])
        rot = grp[0].get("rotation", "")
        first_ok = grp[0]["judged_correct"] == "1"
        final_ok = grp[-1]["judged_correct"] == "1"
        d = by_track_rot[(track, rot)]
        if first_ok:
            d["step1_correct"] += 1
            d["step1_correct_final_correct"] += int(final_ok)
        else:
            d["step1_wrong"] += 1
            d["step1_wrong_final_correct"] += int(final_ok)

    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for (track, rot), d in by_track_rot.items():
        result[track][str(rot)] = {
            "anchor_rate": (d["step1_correct_final_correct"] / d["step1_correct"]
                            if d["step1_correct"] else float("nan")),
            "recovery_rate": (d["step1_wrong_final_correct"] / d["step1_wrong"]
                              if d["step1_wrong"] else float("nan")),
            "n_step1_correct": d["step1_correct"],
            "n_step1_wrong": d["step1_wrong"],
        }
    return {t: dict(sorted(v.items(), key=lambda kv: int(kv[0])))
            for t, v in result.items()}


def plot_swap_anchor_rates(result: dict[str, dict[str, dict]], output_dir: Path) -> None:
    """One bar chart per track: anchor/recovery rate by which question went first."""
    import matplotlib.pyplot as plt

    for track, by_rot in result.items():
        rots = sorted(by_rot, key=int)
        if not rots:
            continue
        labels   = [f"Q{int(r) + 1}\nfirst" for r in rots]
        anchor   = [by_rot[r]["anchor_rate"] * 100 for r in rots]
        recovery = [by_rot[r]["recovery_rate"] * 100 for r in rots]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(rots)))
        w = 0.35
        ax.bar([i - w / 2 for i in x], anchor, width=w, color="#1f77b4",
               label="Anchor rate (step-1 right -> final right)")
        ax.bar([i + w / 2 for i in x], recovery, width=w, color="#ff7f0e",
               label="Recovery rate (step-1 wrong -> final right)")
        for i, v in enumerate(anchor):
            if not np.isnan(v):
                ax.annotate(f"{v:.0f}", (i - w / 2, v), fontsize=7, ha="center",
                            xytext=(0, 4), textcoords="offset points")
        for i, v in enumerate(recovery):
            if not np.isnan(v):
                ax.annotate(f"{v:.0f}", (i + w / 2, v), fontsize=7, ha="center",
                            xytext=(0, 4), textcoords="offset points")

        mean_anchor = float(np.nanmean(anchor))
        ax.axhline(mean_anchor, color="#1f77b4", linestyle="--", linewidth=1.2,
                   alpha=0.6, label=f"mean anchor ({mean_anchor:.0f}%)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Rate (%)", fontsize=10)
        ax.set_xlabel("Question asked first (rotation)", fontsize=10)
        ax.set_title(f"{track.replace('_', ' ').title()} — anchor/recovery rate "
                     f"by which question leads", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()

        path = output_dir / f"swap_anchor_{track}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


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
    baseline_f1: dict[str, float] | None = None,
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
        # Per-dataset baseline as a dashed horizontal line in the same colour
        if baseline_f1 and ds in baseline_f1:
            ax.axhline(baseline_f1[ds], color=color, linestyle="--",
                       linewidth=0.9, alpha=0.45)

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

    # Mean baseline across all datasets in this chart
    if baseline_f1:
        base_vals = [baseline_f1[ds] for ds in ds_list if ds in baseline_f1]
        if base_vals:
            mean_base = float(np.mean(base_vals))
            ax.axhline(mean_base, color="red", linestyle="--", linewidth=1.6,
                       label=f"baseline mean ({mean_base:.2f})", zorder=4)

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
    baseline_f1: dict[str, float] | None = None,
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
            # Baseline mean for this task as dashed line in same colour
            if baseline_f1:
                base_vals = [baseline_f1[ds] for ds in ds_list if ds in baseline_f1]
                if base_vals:
                    mean_base = float(np.mean(base_vals))
                    ax.axhline(mean_base, color=task_color, linestyle="--",
                               linewidth=1.2, alpha=0.6)

        # Single red dashed line = grand mean baseline across all selected datasets
        if baseline_f1:
            all_ds = [ds for names in selected_datasets.values() for ds in names]
            grand_base_vals = [baseline_f1[ds] for ds in all_ds if ds in baseline_f1]
            if grand_base_vals:
                grand_mean = float(np.mean(grand_base_vals))
                ax.axhline(grand_mean, color="red", linestyle="--", linewidth=1.8,
                           label=f"baseline (mean {grand_mean:.2f})", zorder=4)

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
                baseline_f1=baseline_f1,
            )
            plt.tight_layout()
            path = output_dir / f"step_{mode}_{task}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")


# ── Per-dataset plots ─────────────────────────────────────────────────────────

def plot_per_dataset(
    step_f1: dict[str, dict[str, dict[int, float]]],
    all_datasets: list[str],
    ds_to_task: dict[str, str],
    output_dir: Path,
    baseline_f1: dict[str, float] | None = None,
    deltas: dict[str, float] | None = None,
) -> None:
    """One PNG per dataset: cumulative and independent on the same axes vs baseline."""
    import matplotlib.pyplot as plt

    per_ds_dir = output_dir / "per_dataset"
    per_ds_dir.mkdir(parents=True, exist_ok=True)

    MODE_COLOR  = {"cumulative": "#1f77b4", "independent": "#ff7f0e"}
    MODE_LABEL  = {"cumulative": "Cumulative", "independent": "Independent"}

    for ds in sorted(all_datasets):
        task  = ds_to_task.get(ds, "disease")
        base  = (baseline_f1 or {}).get(ds)
        delta = (deltas or {}).get(ds)
        label = ds.replace("_classification", "").replace("_", " ")

        # Determine max steps across modes
        all_steps: set[int] = set()
        for mode in ("cumulative", "independent"):
            all_steps.update(step_f1.get(mode, {}).get(ds, {}).keys())
        if not all_steps:
            continue
        xs = sorted(all_steps)

        fig, ax = plt.subplots(figsize=(7, 4))

        for mode in ("cumulative", "independent"):
            ys_by_step = step_f1.get(mode, {}).get(ds, {})
            if not ys_by_step:
                continue
            ys = [ys_by_step.get(s, float("nan")) for s in xs]
            ax.plot(xs, ys, marker="o", color=MODE_COLOR[mode],
                    linewidth=2, label=MODE_LABEL[mode])
            for x, y in zip(xs, ys):
                if not np.isnan(y):
                    ax.annotate(f"{y:.3f}", (x, y), fontsize=7,
                                xytext=(0, 7), textcoords="offset points", ha="center")

        if base is not None:
            ax.axhline(base, color="red", linestyle="--", linewidth=1.6,
                       label=f"baseline ({base:.3f})", zorder=4)

        direction = ""
        if delta is not None:
            sign = "▲" if delta >= 0 else "▼"
            direction = f"  [{sign}{abs(delta):.3f} vs baseline]"

        ax.set_xticks(xs)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Judged F1", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{label}{direction}\n({TASK_LABEL.get(task, task)})", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        path = per_ds_dir / f"{ds}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(all_datasets)} per-dataset plots → {per_ds_dir}")


# ── Step-example cards ────────────────────────────────────────────────────────

def _extract_step_qa(reasoning: str) -> tuple[str, str]:
    """Return (question, answer) for the last step in a cumulative reasoning string."""
    # Split on blank-line + Q: to isolate each step's block
    blocks = re.split(r"\n\nQ:\s*", reasoning.strip())
    last = blocks[-1].strip()
    # First block already has "Q: "; later blocks don't after the split
    if last.startswith("Q: "):
        last = last[3:]
    elif last.startswith("Q:"):
        last = last[2:]
    if "\nA: " in last:
        q, a = last.split("\nA: ", 1)
        return q.strip(), a.strip()
    if "\nA:" in last:
        q, a = last.split("\nA:", 1)
        return q.strip(), a.strip()
    return "", last.strip()


def plot_step_examples(
    rows: list[dict],
    all_datasets: list[str],
    output_dir: Path,
    examples_per_ds: int = 3,
    mode: str = "cumulative",
) -> None:
    """Per-dataset PNG showing intermediate CoT step responses for selected examples.

    Layout: each example is a column; each step is a row within that column.
    Green card = model was correct at that step; red = incorrect.
    """
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    step_dir = output_dir / "step_examples"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by (ds, image_name, class_label) for the chosen mode
    from collections import defaultdict
    by_ds: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("mode") != mode:
            continue
        key = (r.get("image_name", ""), r.get("class_label", ""))
        by_ds[r.get("agml_dataset", "")][key].append(r)

    # Layout constants (inches / figure units)
    COL_W    = 7.0    # width of each example column
    STEP_H   = 1.55   # height of each step card
    HDR_H    = 0.55   # height of per-example header
    PAD      = 0.18   # horizontal padding inside card
    MARGIN   = 0.10   # gap between columns
    TOP_H    = 0.45   # figure-level title

    for ds in sorted(all_datasets):
        img_groups = by_ds.get(ds)
        if not img_groups:
            continue

        # Sort each image's steps; find max step count
        sorted_groups: dict[tuple, list[dict]] = {
            k: sorted(v, key=lambda r: int(r.get("step_idx", 0)))
            for k, v in img_groups.items()
        }
        max_steps = max(len(v) for v in sorted_groups.values())

        # Select examples: prefer 1 correct + rest incorrect at final step
        correct_keys   = [k for k, v in sorted_groups.items()
                          if str(v[-1].get("judged_correct", "")) == "1"]
        incorrect_keys = [k for k, v in sorted_groups.items()
                          if str(v[-1].get("judged_correct", "")) == "0"]
        rng_local = random.Random(42)
        chosen_keys: list[tuple] = []
        n_correct = max(1, examples_per_ds // 3)
        chosen_keys += rng_local.sample(correct_keys,   min(n_correct, len(correct_keys)))
        remaining_n  = examples_per_ds - len(chosen_keys)
        chosen_keys += rng_local.sample(incorrect_keys, min(remaining_n, len(incorrect_keys)))
        # Fill up if needed
        if len(chosen_keys) < examples_per_ds:
            pool = [k for k in sorted_groups if k not in chosen_keys]
            chosen_keys += rng_local.sample(pool, min(examples_per_ds - len(chosen_keys), len(pool)))
        chosen_keys = chosen_keys[:examples_per_ds]

        n_ex    = len(chosen_keys)
        fig_w   = n_ex * COL_W + (n_ex - 1) * MARGIN
        fig_h   = TOP_H + HDR_H + max_steps * STEP_H + 0.2
        fig     = plt.figure(figsize=(fig_w, fig_h))
        ax      = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.axis("off")

        ds_label = ds.replace("_classification", "").replace("_", " ")
        ax.text(fig_w / 2, fig_h - 0.12,
                f"{ds_label}  —  Step Reasoning ({mode.capitalize()})",
                ha="center", va="top", fontsize=11, fontweight="bold")

        for col_i, key in enumerate(chosen_keys):
            step_rows = sorted_groups[key]
            final     = step_rows[-1]
            gt        = final.get("ground_truth") or final.get("class_label", "?")
            final_pred = (final.get("prediction") or "")[:70]
            is_correct = str(final.get("judged_correct", "")) == "1"
            verdict    = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            v_color    = "#2e7d32" if is_correct else "#c62828"
            img_name   = Path(key[0]).name

            col_x = col_i * (COL_W + MARGIN)

            # Column header
            hdr_y = fig_h - TOP_H - 0.05
            ax.text(col_x + PAD, hdr_y,
                    f"GT: {gt}",
                    fontsize=8, va="top", color="#333")
            ax.text(col_x + PAD, hdr_y - 0.18,
                    f"Final pred: {final_pred}",
                    fontsize=7.5, va="top", color="#333")
            ax.text(col_x + COL_W - PAD, hdr_y,
                    verdict,
                    fontsize=8.5, va="top", ha="right",
                    color=v_color, fontweight="bold")
            ax.text(col_x + PAD, hdr_y - 0.36,
                    img_name[:60],
                    fontsize=6, va="top", color="#888", style="italic")

            # Step cards
            card_top = fig_h - TOP_H - HDR_H
            for step_row in step_rows:
                step_idx   = int(step_row.get("step_idx", 0))
                step_pred  = (step_row.get("prediction") or "")[:70]
                step_ok    = str(step_row.get("judged_correct", "")) == "1"
                face_color = "#e8f5e9" if step_ok else "#ffebee"
                edge_color = "#4caf50" if step_ok else "#f44336"
                pred_color = "#2e7d32" if step_ok else "#c62828"

                # Extract just this step's Q and A
                q_text, a_text = _extract_step_qa(step_row.get("reasoning", ""))
                q_wrapped = textwrap.fill(q_text, width=58, max_lines=2, placeholder="…")
                a_wrapped = textwrap.fill(a_text, width=72, max_lines=4, placeholder="…")

                card_bot = card_top - STEP_H
                patch = mpatches.FancyBboxPatch(
                    (col_x + 0.05, card_bot + 0.06),
                    COL_W - 0.10, STEP_H - 0.12,
                    boxstyle="round,pad=0.04", linewidth=1.0,
                    edgecolor=edge_color, facecolor=face_color,
                )
                ax.add_patch(patch)

                # Step label + prediction (right-aligned)
                ax.text(col_x + PAD, card_top - 0.14,
                        f"Step {step_idx}",
                        fontsize=7.5, va="top", fontweight="bold", color="#444")
                ax.text(col_x + COL_W - PAD, card_top - 0.14,
                        f"→ {step_pred}",
                        fontsize=7, va="top", ha="right",
                        color=pred_color, fontweight="bold")

                # Q
                if q_text:
                    ax.text(col_x + PAD, card_top - 0.34,
                            f"Q: {q_wrapped}",
                            fontsize=6.5, va="top", color="#555", style="italic")
                    a_y = card_top - 0.34 - (q_wrapped.count("\n") + 1) * 0.155
                else:
                    a_y = card_top - 0.34

                # A
                ax.text(col_x + PAD, a_y,
                        f"A: {a_wrapped}",
                        fontsize=6.5, va="top", color="#222")

                card_top -= STEP_H

        path = step_dir / f"{ds}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    print(f"  → {step_dir}")


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


# ── Swap-order pipeline ────────────────────────────────────────────────────────

def run_swap_order_pipeline(
    all_selected: list[str],
    ds_to_task: dict[str, str],
    ds_to_crop: dict[str, str],
    ds_info: dict[str, dict],
    test_split: dict[str, list[str]] | None,
    n_per_class: int,
    model: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    """Run cumulative CoT inference with rotated step order on the given
    datasets (capped to n_per_class images/class), then check whether the
    anchor rate holds up regardless of which question is asked first.
    """
    swap_dir = output_dir / "swap_order"
    swap_dir.mkdir(parents=True, exist_ok=True)
    ckpt_raw    = swap_dir / "ckpt_swap_raw.csv"
    ckpt_judged = swap_dir / "ckpt_swap_judged.csv"

    all_items: list[dict] = []
    for ds in all_selected:
        task      = ds_to_task[ds]
        crop      = ds_to_crop.get(ds, "plant")
        info      = ds_info.get(ds, {})
        overrides = info.get("class_overrides", {})
        items = load_items(ds, task, crop, n_samples=None, class_overrides=overrides,
                           seed=SEED, test_split=test_split, n_per_class=n_per_class)
        all_items.extend(items)
    print(f"\n=== Swap-order: {len(all_items):,} images (<= {n_per_class}/class) "
          f"across {len(all_selected)} datasets ===")

    from src.cot_vision import load_llm, destroy_llm
    print(f"\n=== Loading model: {model} ===")
    llm, tokenizer = load_llm(model)

    rows = run_swap_order(all_items, llm, tokenizer,
                          prompt_batch_size=batch_size, ckpt_path=ckpt_raw)

    print(f"\n=== Judging {len(rows):,} rows ===")
    rows = run_judge(rows, llm, tokenizer, ckpt_path=ckpt_judged)

    destroy_llm(llm)
    del tokenizer, llm
    gc.collect()

    print("\n=== Computing anchor/recovery rate by rotation ===")
    result = compute_swap_anchor_rates(rows)
    (swap_dir / "swap_anchor_rates.json").write_text(json.dumps(result, indent=2))
    for track, by_rot in result.items():
        print(f"  {track}:")
        for rot, v in by_rot.items():
            print(f"    Q{int(rot) + 1} first: anchor={v['anchor_rate']:.3f} "
                  f"(n={v['n_step1_correct']})  recovery={v['recovery_rate']:.3f} "
                  f"(n={v['n_step1_wrong']})")

    print("\n=== Plotting swap-order anchor rate chart ===")
    plot_swap_anchor_rates(result, swap_dir)

    print(f"\nDone. Swap-order artifacts in {swap_dir}")


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
    parser.add_argument("--swap_order", action="store_true",
                        help="Robustness check: for each track, cyclically rotate the "
                             "CoT step order so every question gets a turn going first, "
                             "then compare anchor/recovery rate across rotations. Runs on "
                             "the same top-N gain/regression datasets (requires --vis_dir "
                             "and --cot_dir). Writes to <output_dir>/swap_order/.")
    parser.add_argument("--n_per_class", type=int, default=None,
                        help="Cap images sampled per class (random subsample). "
                             "Defaults to 50 when --swap_order is set.")
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

    # ── Swap-order shortcut ────────────────────────────────────────────────────
    if args.swap_order:
        if not (args.vis_dir and args.cot_dir):
            parser.error("--swap_order requires --vis_dir and --cot_dir "
                         "(delta-based dataset selection)")
        n_per_class = args.n_per_class if args.n_per_class is not None else 50
        run_swap_order_pipeline(
            all_selected=all_selected, ds_to_task=ds_to_task, ds_to_crop=ds_to_crop,
            ds_info=ds_info, test_split=test_split, n_per_class=n_per_class,
            model=args.model, batch_size=args.batch_size, output_dir=output_dir,
        )
        return

    # ── Plot-only shortcut ─────────────────────────────────────────────────────
    if args.plot_only:
        rows = _read_csv(ckpt_judged)
        step_f1 = compute_step_f1(rows)
        print("\n=== Plotting step charts ===")
        plot_all(step_f1, selected, ds_to_task, output_dir, baseline_f1=baseline_f1_map or None)
        print("\n=== Plotting per-dataset charts ===")
        plot_per_dataset(step_f1, all_selected, ds_to_task, output_dir,
                         baseline_f1=baseline_f1_map or None, deltas=deltas or None)
        print("\n=== Plotting step-example cards ===")
        plot_step_examples(rows, all_selected, output_dir)
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
    # Always prefer the on-disk checkpoint so phase 4 gets complete data even
    # when all_rows was populated inline (inline path doesn't set need_judge).
    if not all_rows and ckpt_judged.exists():
        all_rows = _read_csv(ckpt_judged)

    print("\n=== Computing step F1 ===")
    step_f1 = compute_step_f1(all_rows)
    for mode in ("cumulative", "independent"):
        for ds, by_step in step_f1.get(mode, {}).items():
            steps_str = ", ".join(f"s{s}={v:.3f}" for s, v in sorted(by_step.items()))
            print(f"  [{mode}] {ds}: {steps_str}")

    print("\n=== Plotting step charts ===")
    plot_all(step_f1, selected, ds_to_task, output_dir, baseline_f1=baseline_f1_map or None)

    print("\n=== Plotting per-dataset charts ===")
    plot_per_dataset(step_f1, all_selected, ds_to_task, output_dir,
                     baseline_f1=baseline_f1_map or None, deltas=deltas or None)

    print("\n=== Plotting step-example cards ===")
    plot_step_examples(all_rows, all_selected, output_dir)

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
