"""
Verifier Vision — Baseline Verifier Experiment (subsampled)

Two generation strategies × N ∈ {1, 5, 10} verified candidates:
  direct   — N diverse classification samples at T=0.8; verifier picks best
  self_cot — greedy description (T=0.0) then N classification samples (T=0.8); verifier picks best

Verifier: LLM scores each (image, candidate) 1–10 with task-specific rubric derived
from CoT prompt steps. Final prediction = argmax over verifier scores.

Outputs compared against subsampled Vision Baseline, Self-CoT, CoT YAML, Desc-Aug.
Final judged F1 via apply_judge — same pipeline as all other experiments.

Usage:
  uv run python src/verifier_vision.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_vision/ \\
    --vis_dir outputs/vision_assessment/ \\
    --cot_dir outputs/cot_vision/ \\
    --aug_dir outputs/desc_augmented_vision/ \\
    --n_samples 50 --n_datasets 3 --seed 42 --batch_size 256

  # Resume from checkpoints:
  uv run python src/verifier_vision.py ... --from_ckpt

  # Plot only (all checkpoints must exist):
  uv run python src/verifier_vision.py ... --plot_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

N_DATASETS_PER_TASK = 3
SEED = 42
N_MAX_DEFAULT = 10

CANDIDATE_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "image_path", "gen_mode", "n_idx", "prediction",
]
SCORED_FIELDS = CANDIDATE_FIELDS + ["verifier_score", "verifier_reasoning"]
JUDGED_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "gen_mode", "n_max",
    "final_prediction", "verifier_score", "ground_truth",
    "judged_correct", "judge_reasoning",
]

# Task-specific verifier rubrics derived from cot_prompts.yaml steps
_RUBRIC_DISEASE = (
    "1. Does it correctly identify the crop or plant species affected?\n"
    "2. Does it match the pattern of affected plant parts visible "
    "(leaves, stem, fruit, roots)?\n"
    "3. Does it align with the visible symptom characteristics — "
    "color, shape, and texture of damage?\n"
    "4. If pest-related, does it account for visible pest signs "
    "(webbing, frass, eggs, insect bodies)?\n"
    "5. Does it reflect whether the damage is systemic/uniform or isolated/localized?\n"
    "6. Is the stated condition (disease name, pest, or stress type) visually "
    "consistent with the image?"
)
_RUBRIC_WEED = (
    "1. Does it match the visible leaf arrangement, shape, margins, and venation pattern?\n"
    "2. Does it align with the plant's growth habit (erect, prostrate, climbing) "
    "and stem characteristics?\n"
    "3. If flowers or seed structures are visible, does the prediction account for them?\n"
    "4. Is the botanical identification (Scientific Name and/or Common Name) consistent "
    "with all visible features?"
)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None          # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]

# ── CSV helpers ────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Verifier score parsing ─────────────────────────────────────────────────────

def _parse_verifier_score(text: str) -> tuple[int, str]:
    """Extract score and reasoning from verifier JSON response."""
    # Strip thinking tags
    text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>",
                  "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            score = max(1, min(10, int(data.get("score", 1))))
            reasoning = str(data.get("reasoning", ""))[:400]
            return score, reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    score = max(1, min(10, int(m.group(1)))) if m else 1
    m2 = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
    reasoning = m2.group(1)[:400] if m2 else ""
    return score, reasoning


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _import_cot_helpers():
    from src.cot_vision import (
        _sys_msg, _cls_question, _desc_question,
        _build_prompt, _clean, load_llm, destroy_llm,
    )
    return _sys_msg, _cls_question, _desc_question, _build_prompt, _clean, load_llm, destroy_llm


def _raw_cls_question(task_type: str, crop: str | None) -> str:
    """Inner classification question without the assessment_classification wrapper."""
    import yaml
    _p = yaml.safe_load(
        (Path(__file__).parent / "config" / "prompts.yaml").read_text()
    )
    if task_type == "weed_identification":
        return _p["visual_classification_weed"]
    return _p["visual_classification_disease"].format(crop=crop or "plant")


def _verifier_text(task_type: str, crop: str | None, candidate: str) -> str:
    rubric = _RUBRIC_DISEASE if task_type != "weed_identification" else _RUBRIC_WEED
    question = _raw_cls_question(task_type, crop)
    return (
        "You are evaluating a classification prediction for an agricultural image.\n\n"
        f"Question asked: \"{question}\"\n"
        f"Prediction: \"{candidate[:300]}\"\n\n"
        "Evaluate the prediction against these diagnostic dimensions:\n"
        f"{rubric}\n\n"
        "Score this prediction from 1 to 10:\n"
        "1 = clearly wrong or irrelevant to the image\n"
        "5 = plausible but vague or imprecise\n"
        "10 = precisely correct, specific, and visually grounded\n\n"
        'Respond with JSON only:\n{ "score": <int 1-10>, "reasoning": "<one sentence>" }'
    )


# ── Dataset selection (same logic as cot_step_analysis) ───────────────────────

def _build_task_datasets(config_path: str) -> dict[str, dict[str, list[str]]]:
    """Return {task: {"primary": [...], "override": [...]}} for each task type."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    primary: dict[str, set[str]] = defaultdict(set)
    override: dict[str, set[str]] = defaultdict(set)
    for d in cfg["datasets"]:
        if d.get("skip"):
            continue
        name = d["agml_name"]
        primary[d["task_type"]].add(name)
        for task in (d.get("class_overrides") or {}).values():
            if task != d["task_type"]:
                override[task].add(name)
    tasks = ("disease", "pest_damage", "weed_identification")
    return {
        t: {
            "primary":  sorted(primary.get(t, [])),
            "override": sorted(override.get(t, [])),
        }
        for t in tasks
    }


def select_datasets(n_per_task: int = N_DATASETS_PER_TASK,
                    seed: int = SEED,
                    config_path: str = "src/config/datasets.yaml") -> dict[str, list[str]]:
    """Sample n_per_task datasets per task, prioritising primary task_type datasets
    before filling remaining slots with class_override datasets."""
    pools = _build_task_datasets(config_path)
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for task, buckets in pools.items():
        primary  = buckets["primary"]
        override = buckets["override"]
        # Fill from primary first, then override for remaining slots
        n_primary = min(n_per_task, len(primary))
        chosen = rng.sample(primary, n_primary)
        remaining = n_per_task - len(chosen)
        if remaining > 0 and override:
            chosen += rng.sample(override, min(remaining, len(override)))
        selected[task] = chosen
    return selected


def load_items(
    ds_name: str,
    task_type: str,
    crop: str | None,
    n_samples: int | None,
    class_overrides: dict[str, str] | None = None,
    seed: int = SEED,
    test_split: dict | None = None,
) -> list[dict]:
    """Load image items for ds_name filtered to classes belonging to task_type."""
    from src.cot_vision import load_dataset_classes, DATASET_TASK_TYPES
    by_class = load_dataset_classes(ds_name, n_samples, test_split)
    primary_task = DATASET_TASK_TYPES.get(ds_name, (task_type, crop))[0]
    items: list[dict] = []
    for cls, paths in by_class.items():
        effective = (class_overrides or {}).get(cls, primary_task)
        if effective != task_type:
            continue
        for p in paths:
            items.append({
                "ds": ds_name, "task_type": task_type,
                "crop": crop, "cls": cls,
                "image_path": str(p),
                "image_name": p.name,
            })
    random.Random(seed).shuffle(items)
    return items


# ── Phase 1: Generate candidates ──────────────────────────────────────────────

def run_generate(
    items: list[dict],
    llm,
    tokenizer,
    n_max: int = N_MAX_DEFAULT,
    batch_size: int = 256,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Generate n_max candidates per image for both direct and self_cot strategies."""
    (_sys_msg, _cls_question, _desc_question,
     _build_prompt, _clean, _, _) = _import_cot_helpers()

    _GEN_P = SamplingParams(temperature=0.8, max_tokens=64, repetition_penalty=1.1, n=n_max)
    _DESC_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)

    rows: list[dict] = []

    # ── Strategy A: direct — N=n_max samples per image ─────────────────────────
    print(f"\n[generate/direct] {len(items)} images × {n_max} candidates")
    for start in tqdm(range(0, len(items), batch_size), desc="direct", unit="batch"):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": _cls_question(item["task_type"], item["crop"])},
            ]}]
            prompts.append(_build_prompt(msgs, tokenizer,
                                         PILImage.open(item["image_path"]).convert("RGB")))
        outputs = llm.generate(prompts, _GEN_P)
        for item, out in zip(batch, outputs):
            for n_idx, completion in enumerate(out.outputs):
                rows.append({
                    "agml_dataset": item["ds"],
                    "class_label":  item["cls"],
                    "task_type":    item["task_type"],
                    "crop":         item["crop"] or "",
                    "image_name":   item["image_name"],
                    "image_path":   item["image_path"],
                    "gen_mode":     "direct",
                    "n_idx":        n_idx,
                    "prediction":   _clean(completion.text),
                })
        del prompts

    # ── Strategy B: self_cot — greedy description then N=n_max classification ──
    print(f"\n[generate/self_cot] {len(items)} images — describe (T=0) then ×{n_max} classify")

    # Stage 1: greedy description for each image
    descriptions: list[str] = []
    desc_histories: list[list[dict]] = []
    for start in tqdm(range(0, len(items), batch_size), desc="self_cot desc", unit="batch"):
        batch = items[start:start + batch_size]
        prompts, hists = [], []
        for item in batch:
            desc_q = _desc_question(item["task_type"], item["crop"])
            hist = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc_q},
            ]}]
            hists.append(hist)
            prompts.append(_build_prompt(hist, tokenizer,
                                          PILImage.open(item["image_path"]).convert("RGB")))
        outputs = llm.generate(prompts, _DESC_P)
        for hist, out in zip(hists, outputs):
            resp = _clean(out.outputs[0].text)
            descriptions.append(resp)
            desc_histories.append(hist + [{"role": "assistant", "content": resp}])
        del prompts

    # Stage 2: N=n_max diverse classification samples using the description as context
    for start in tqdm(range(0, len(items), batch_size), desc="self_cot cls", unit="batch"):
        batch = items[start:start + batch_size]
        batch_hists = desc_histories[start:start + batch_size]
        prompts = []
        for item, hist in zip(batch, batch_hists):
            cls_q = _cls_question(item["task_type"], item["crop"])
            prompts.append(_build_prompt(
                hist + [{"role": "user", "content": cls_q}],
                tokenizer,
                PILImage.open(item["image_path"]).convert("RGB"),
            ))
        outputs = llm.generate(prompts, _GEN_P)
        for item, out in zip(batch, outputs):
            for n_idx, completion in enumerate(out.outputs):
                rows.append({
                    "agml_dataset": item["ds"],
                    "class_label":  item["cls"],
                    "task_type":    item["task_type"],
                    "crop":         item["crop"] or "",
                    "image_name":   item["image_name"],
                    "image_path":   item["image_path"],
                    "gen_mode":     "self_cot",
                    "n_idx":        n_idx,
                    "prediction":   _clean(completion.text),
                })
        del prompts

    if ckpt_path:
        _write_csv(rows, ckpt_path, CANDIDATE_FIELDS)
    return rows


# ── Phase 2: Score candidates with verifier ───────────────────────────────────

def run_verify(
    candidates: list[dict],
    llm,
    tokenizer,
    batch_size: int = 256,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Score each (image, candidate) pair with the LLM verifier (1–10)."""
    (_sys_msg, _, _, _build_prompt, _, _, _) = _import_cot_helpers()
    _VER_P = SamplingParams(temperature=0.0, max_tokens=128)

    print(f"\n[verify] scoring {len(candidates)} (image, candidate) pairs")
    scored: list[dict] = []

    for start in tqdm(range(0, len(candidates), batch_size), desc="verify", unit="batch"):
        batch = candidates[start:start + batch_size]
        prompts = []
        for row in batch:
            text = _verifier_text(row["task_type"], row.get("crop") or None,
                                  row["prediction"])
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            prompts.append(_build_prompt(
                msgs, tokenizer,
                PILImage.open(row["image_path"]).convert("RGB"),
            ))
        outputs = llm.generate(prompts, _VER_P)
        for row, out in zip(batch, outputs):
            score, reasoning = _parse_verifier_score(out.outputs[0].text)
            scored.append({**row, "verifier_score": score, "verifier_reasoning": reasoning})
        del prompts

    if ckpt_path:
        _write_csv(scored, ckpt_path, SCORED_FIELDS)
    return scored


# ── Phase 3: Argmax selection + LLM judge ─────────────────────────────────────

def run_argmax_and_judge(
    scored: list[dict],
    llm,
    tokenizer,
    n_max: int = N_MAX_DEFAULT,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Pick best candidate per (image, gen_mode, n_slice) and judge vs ground truth."""
    from src.common.metrics import apply_judge
    import yaml
    _p = yaml.safe_load(
        (Path(__file__).parent / "config" / "prompts.yaml").read_text()
    )
    judge_system   = _p["judge_system_vision"]
    judge_template = _p["judge_classification_template"]

    # Group candidates by (dataset, image_name, gen_mode)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scored:
        key = (row["agml_dataset"], row["image_name"], row["gen_mode"])
        groups[key].append(row)

    # For each N slice, select the argmax candidate
    to_judge: list[dict] = []
    n_slices = [n for n in (1, 5, 10) if n <= n_max]
    for (ds, img, gen_mode), candidates in groups.items():
        candidates = sorted(candidates, key=lambda r: int(r.get("n_idx", 0)))
        gt = candidates[0]["class_label"]
        for n_slice in n_slices:
            subset = candidates[:n_slice]
            best = max(subset, key=lambda r: int(r.get("verifier_score", 0)))
            to_judge.append({
                "agml_dataset":     ds,
                "class_label":      gt,
                "task_type":        candidates[0]["task_type"],
                "crop":             candidates[0].get("crop", ""),
                "image_name":       img,
                "gen_mode":         gen_mode,
                "n_max":            n_slice,
                "final_prediction": best["prediction"],
                "verifier_score":   best.get("verifier_score", 0),
                "ground_truth":     gt,
                "judged_correct":   "",
                "judge_reasoning":  "",
            })

    print(f"\n[judge] {len(to_judge)} rows ({len(groups)} images × "
          f"{len(['direct','self_cot'])} modes × {len(n_slices)} N-slices)")
    judged = apply_judge(
        to_judge, llm, tokenizer,
        system_prompt=judge_system,
        judge_template=judge_template,
        gt_key="ground_truth",
        pred_key="final_prediction",
    )

    if ckpt_path:
        _write_csv(judged, ckpt_path, JUDGED_FIELDS)
    return judged


# ── Subsample existing CSVs to the same image set ────────────────────────────

def load_existing_rows(
    image_keys: set[tuple[str, str]],
    vis_dir: Path,
    cot_dir: Path,
    aug_dir: Path,
    model_safe: str,
) -> dict[str, list[dict]]:
    """Load and filter existing experiment CSVs to only the selected images.

    image_keys: set of (agml_dataset, image_name) tuples from the verifier run.
    """
    def _filter(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if (r.get("agml_dataset"), r.get("image_name")) in image_keys]

    result: dict[str, list[dict]] = {}

    baseline_csv = vis_dir / f"cls_{model_safe}.csv"
    if baseline_csv.exists():
        result["Vision Baseline"] = _filter(_read_csv(baseline_csv))
        print(f"  Vision Baseline: {len(result['Vision Baseline'])} rows")
    else:
        print(f"  [skip] baseline CSV not found: {baseline_csv}")

    cot_csv = cot_dir / f"cot_cls_{model_safe}.csv"
    if cot_csv.exists():
        cot_rows = _filter(_read_csv(cot_csv))
        result["Self-CoT"]  = [r for r in cot_rows if r.get("condition") == "self_cot"]
        result["CoT YAML"]  = [r for r in cot_rows if r.get("condition") == "cot_yaml"]
        print(f"  Self-CoT: {len(result['Self-CoT'])} rows  "
              f"CoT YAML: {len(result['CoT YAML'])} rows")
    else:
        print(f"  [skip] cot CSV not found: {cot_csv}")

    aug_csv = aug_dir / f"aug_cls_all_{model_safe}.csv"
    if aug_csv.exists():
        result["Desc-Aug"] = _filter(_read_csv(aug_csv))
        print(f"  Desc-Aug: {len(result['Desc-Aug'])} rows")
    else:
        print(f"  [skip] aug CSV not found: {aug_csv}")

    return result


# ── Build rows_by_condition for plotting ──────────────────────────────────────

def _condition_label(gen_mode: str, n_max: int) -> str:
    prefix = "Direct" if gen_mode == "direct" else "SelfCoT"
    return f"{prefix}-N{n_max}"


def build_rows_by_condition(
    judged: list[dict],
    existing: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Combine verifier rows (split by gen_mode × n_max) with existing condition rows."""
    result: dict[str, list[dict]] = {}

    # Verifier conditions
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in judged:
        label = _condition_label(row["gen_mode"], int(row["n_max"]))
        by_cond[label].append(row)
    result.update(by_cond)

    # Existing baselines (only add if non-empty)
    for label, rows in existing.items():
        if rows:
            result[label] = rows

    return result


# ── Score spread analysis ──────────────────────────────────────────────────────

def plot_score_spread(
    ckpt_scored: Path,
    ckpt_judged: Path,
    output_dir: Path,
) -> None:
    """Three charts per gen_mode showing verifier score variation vs correctness.

    One figure per N slice (N=1, N=5, N=10).  Each figure is a 2×3 grid:
      rows = gen_mode (direct, self_cot)
      cols = score distribution (violin) | score std (box) | winning margin (box)

    Scores used per image are the first N candidates (n_idx 0..N-1).
    Correctness is taken from the n_max=N row in ckpt_judged.
    """
    import matplotlib.pyplot as plt

    scored_all = _read_csv(ckpt_scored)
    judged     = _read_csv(ckpt_judged)

    # correctness_by_n[(dataset, image_name, gen_mode)][n] = judged_correct
    correctness_by_n: dict[tuple, dict[int, int]] = defaultdict(dict)
    for r in judged:
        n = int(r.get("n_max", 0))
        if n in (1, 5, 10):
            key = (r["agml_dataset"], r["image_name"], r["gen_mode"])
            correctness_by_n[key][n] = int(r["judged_correct"])

    # Group ALL scored rows by (dataset, image_name, gen_mode), sorted by n_idx
    raw_groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for r in scored_all:
        key = (r["agml_dataset"], r["image_name"], r["gen_mode"])
        raw_groups[key].append((int(r["n_idx"]), int(r["verifier_score"])))
    # Sort by n_idx so slicing 0..N-1 is deterministic
    groups: dict[tuple, list[int]] = {
        k: [s for _, s in sorted(v)] for k, v in raw_groups.items()
    }

    gen_modes  = ["direct", "self_cot"]
    gen_labels = {"direct": "Direct", "self_cot": "Self-CoT"}
    colors     = {"correct": "#2ca02c", "incorrect": "#d62728"}

    for n_slice in (1, 5, 10):
        all_scores:  dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}
        std_data:    dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}
        margin_data: dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}

        for key, all_s in groups.items():
            _, _, gen_mode = key
            if gen_mode not in all_scores:
                continue
            jc = correctness_by_n.get(key, {}).get(n_slice)
            if jc is None:
                continue
            scores = all_s[:n_slice]
            label  = "correct" if jc else "incorrect"
            all_scores[gen_mode][label].extend(scores)
            std_data[gen_mode][label].append(float(np.std(scores)))
            sorted_s = sorted(scores, reverse=True)
            margin_data[gen_mode][label].append(
                float(sorted_s[0] - sorted_s[1]) if len(sorted_s) > 1 else 0.0
            )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        for row, gm in enumerate(gen_modes):
            lbl = gen_labels[gm]

            # ── Chart 1: Score distribution (violin) ───────────────────────────
            ax = axes[row, 0]
            data_v = [all_scores[gm]["correct"], all_scores[gm]["incorrect"]]
            if all(len(d) > 1 for d in data_v):
                parts = ax.violinplot(data_v, positions=[0, 1], showmedians=True, showextrema=True)
                for pc, c in zip(parts["bodies"], [colors["correct"], colors["incorrect"]]):
                    pc.set_facecolor(c)
                    pc.set_alpha(0.6)
                for vkey in ("cmedians", "cmins", "cmaxes", "cbars"):
                    if vkey in parts:
                        parts[vkey].set_color("black")
                        parts[vkey].set_linewidth(1.2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Correct\n(n={len(std_data[gm]['correct'])})",
                                 f"Incorrect\n(n={len(std_data[gm]['incorrect'])})"])
            ax.set_ylabel("Verifier Score (1–10)")
            ax.set_ylim(0, 11)
            ax.set_title(f"{lbl} — Score Distribution")
            ax.yaxis.grid(True, alpha=0.3)

            # ── Chart 2: Score spread / std ────────────────────────────────────
            ax = axes[row, 1]
            if n_slice > 1:
                bp = ax.boxplot(
                    [std_data[gm]["correct"], std_data[gm]["incorrect"]],
                    labels=[f"Correct\n(n={len(std_data[gm]['correct'])})",
                            f"Incorrect\n(n={len(std_data[gm]['incorrect'])})"],
                    patch_artist=True, widths=0.5,
                )
                for patch, c in zip(bp["boxes"], [colors["correct"], colors["incorrect"]]):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.5)
            else:
                ax.text(0.5, 0.5, "N=1: std undefined", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(f"Score Std across N={n_slice} candidates")
            ax.set_title(f"{lbl} — Score Spread (Std)")
            ax.yaxis.grid(True, alpha=0.3)

            # ── Chart 3: Winning margin (best − 2nd best) ──────────────────────
            ax = axes[row, 2]
            if n_slice > 1:
                bp = ax.boxplot(
                    [margin_data[gm]["correct"], margin_data[gm]["incorrect"]],
                    labels=[f"Correct\n(n={len(margin_data[gm]['correct'])})",
                            f"Incorrect\n(n={len(margin_data[gm]['incorrect'])})"],
                    patch_artist=True, widths=0.5,
                )
                for patch, c in zip(bp["boxes"], [colors["correct"], colors["incorrect"]]):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.5)
            else:
                ax.text(0.5, 0.5, "N=1: margin undefined", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("Score gap (1st − 2nd candidate)")
            ax.set_title(f"{lbl} — Winning Margin")
            ax.yaxis.grid(True, alpha=0.3)

        fig.suptitle(f"Verifier Score Spread — Correct vs Incorrect (N={n_slice})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"verifier_score_spread_N{n_slice}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")
    print(f"  Saved: {out}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_per_dataset(
    rows_by_cond: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Grouped bar chart: one group per dataset, one bar per condition."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.common.method_comparison import CONDITION_STYLES, CONDITION_COLOR
    from src.common.distribution_plots import gather_judge_per_class_by_dataset

    judge_by_cls = gather_judge_per_class_by_dataset(rows_by_cond)

    # Collect per-dataset mean F1 per condition: {cond: {ds: mean_f1}}
    ds_f1: dict[str, dict[str, float]] = {}
    all_ds: set[str] = set()
    for cond, by_dt in judge_by_cls.items():
        ds_f1[cond] = {}
        for (ds, task), vals in by_dt.items():
            ds_f1[cond][ds] = float(np.mean(vals)) if vals else 0.0
            all_ds.add(ds)

    # Order datasets by task then name
    def _ds_sort_key(ds):
        for cond_rows in rows_by_cond.values():
            for r in cond_rows:
                if r.get("agml_dataset") == ds:
                    return (r.get("task_type", ""), ds)
        return ("", ds)

    datasets = sorted(all_ds, key=_ds_sort_key)
    cond_order = [lbl for lbl, _ in CONDITION_STYLES if lbl in ds_f1]
    n_conds = len(cond_order)
    n_ds = len(datasets)

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 1.4), 6))
    bar_w = 0.8 / max(n_conds, 1)
    offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_w

    for i, cond in enumerate(cond_order):
        vals = [ds_f1[cond].get(ds, np.nan) for ds in datasets]
        xs = np.arange(n_ds) + offsets[i]
        color = CONDITION_COLOR.get(cond, "#888888")
        bars = ax.bar(xs, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=cond)
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    # Dataset labels — short names
    short = [ds.replace("_classification", "").replace("_disease", "")
               .replace("_", " ") for ds in datasets]
    ax.set_xticks(np.arange(n_ds))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Judged F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Verifier (Subsample) — Mean Judged F1 per Dataset")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=CONDITION_COLOR.get(c, "#888"), label=c)
                        for c in cond_order],
              fontsize=7, ncol=2, loc="upper right")

    # Task separator lines
    task_seen: list[str] = []
    for idx, ds in enumerate(datasets):
        tt = _ds_sort_key(ds)[0]
        if not task_seen or task_seen[-1] != tt:
            if task_seen:
                ax.axvline(idx - 0.5, color="gray", lw=0.8, linestyle="--")
            task_seen.append(tt)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_all(
    judged: list[dict],
    existing: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    from src.common.method_comparison import (
        plot_overall_judge_comparison,
        plot_per_task_judge_comparison,
        plot_baseline_vs_condition_scatter,
    )

    rows_by_cond = build_rows_by_condition(judged, existing)
    present = [c for c in rows_by_cond if rows_by_cond[c]]
    print(f"\n[plot] conditions: {present}")

    plot_overall_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_overall.png",
        title="Verifier (Subsample) — Overall Mean Judged F1",
    )
    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_per_task.png",
        title="Verifier (Subsample) — Mean Judged F1 by Task",
    )
    plot_per_dataset(
        rows_by_cond,
        output_path=output_dir / "verifier_per_dataset.png",
    )

    # Scatter: baseline vs verifier conditions, separate panels per gen_mode × N
    direct_conds = {k: v for k, v in rows_by_cond.items()
                    if k.startswith("Direct") or k == "Vision Baseline"}
    selfcot_conds = {k: v for k, v in rows_by_cond.items()
                     if k.startswith("SelfCoT") or k == "Vision Baseline"}

    plot_baseline_vs_condition_scatter(
        direct_conds,
        output_path=output_dir / "verifier_scatter_direct.png",
        title="Verifier Direct — Per-Dataset vs Baseline",
        separate=True,
    )
    plot_baseline_vs_condition_scatter(
        selfcot_conds,
        output_path=output_dir / "verifier_scatter_selfcot.png",
        title="Verifier SelfCoT — Per-Dataset vs Baseline",
        separate=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier Vision — subsample experiment")
    parser.add_argument("--model",       required=True,  help="HuggingFace model ID")
    parser.add_argument("--output_dir",  default="outputs/verifier_vision/")
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/")
    parser.add_argument("--cot_dir",     default="outputs/cot_vision/")
    parser.add_argument("--aug_dir",     default="outputs/desc_augmented_vision/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json",
                        help="Path to test_split.json (same as used by baseline/cot/desc_aug)")
    parser.add_argument("--n_samples",   type=int, default=50,
                        help="Max images per class per dataset")
    parser.add_argument("--n_datasets",  type=int, default=N_DATASETS_PER_TASK,
                        help="Datasets per task")
    parser.add_argument("--n_max",       type=int, default=N_MAX_DEFAULT,
                        help="Max candidates to generate per image (sliced to 1/5/10)")
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--from_ckpt",   action="store_true",
                        help="Skip phases if their checkpoint CSV exists")
    parser.add_argument("--score_only",  action="store_true",
                        help="Skip generation; load ckpt_candidates and run verify+judge+plot")
    parser.add_argument("--judge_only",  action="store_true",
                        help="Skip gen+verify; load ckpt_scored and run judge+plot")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip all inference; load ckpt_judged and plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_candidates = output_dir / "ckpt_candidates.csv"
    ckpt_scored     = output_dir / "ckpt_scored.csv"
    ckpt_judged     = output_dir / "ckpt_judged.csv"

    # ── Dataset selection ──────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    test_split: dict | None = None
    ts_path = Path(args.test_split)
    if ts_path.exists():
        with open(ts_path) as f:
            test_split = json.load(f)
        print(f"Loaded test_split: {ts_path} ({len(test_split)} datasets)")
    else:
        print(f"[warn] test_split not found at {ts_path} — using all images")

    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed,
                               config_path=args.config)
    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")

    # ── Checkpoint skip flags ──────────────────────────────────────────────────
    skip_generate = (args.plot_only or args.score_only or args.judge_only or
                     (args.from_ckpt and ckpt_candidates.exists()))
    skip_score    = (args.plot_only or args.judge_only or
                     (args.from_ckpt and ckpt_scored.exists()))
    skip_judge    = (args.plot_only or
                     (args.from_ckpt and ckpt_judged.exists()))

    # ── Load model only if needed ──────────────────────────────────────────────
    need_model = not (skip_generate and skip_score and skip_judge)
    llm = tokenizer = None

    if need_model:
        from src.cot_vision import load_llm, destroy_llm
        print(f"\n=== Loading model: {args.model} ===")
        llm, tokenizer = load_llm(args.model)
        t0 = time.perf_counter()

    # ── Phase 1: Generate ──────────────────────────────────────────────────────
    if skip_generate:
        print(f"\n=== Phase 1: generate [SKIPPED] ===")
        candidates = _read_csv(ckpt_candidates)
    else:
        all_items: list[dict] = []
        for task, ds_names in selected.items():
            for ds in ds_names:
                info = ds_info.get(ds, {})
                crop = info.get("crop", "plant")
                class_overrides = info.get("class_overrides", {})
                items = load_items(ds, task, crop, n_samples=args.n_samples,
                                   class_overrides=class_overrides, seed=args.seed,
                                   test_split=test_split)
                all_items.extend(items)
                print(f"  {ds} ({task}): {len(items)} items")
        print(f"Total items: {len(all_items)}")

        print(f"\n=== Phase 1: generate ({len(all_items)} images × {args.n_max} candidates) ===")
        candidates = run_generate(all_items, llm, tokenizer,
                                  n_max=args.n_max, batch_size=args.batch_size,
                                  ckpt_path=ckpt_candidates)

    # ── Phase 2: Verify ────────────────────────────────────────────────────────
    if skip_score:
        print(f"\n=== Phase 2: verify [SKIPPED] ===")
        scored = _read_csv(ckpt_scored)
    else:
        print(f"\n=== Phase 2: verify ({len(candidates)} candidates) ===")
        scored = run_verify(candidates, llm, tokenizer,
                            batch_size=args.batch_size, ckpt_path=ckpt_scored)

    # ── Phase 3: Argmax + Judge ────────────────────────────────────────────────
    if skip_judge:
        print(f"\n=== Phase 3: judge [SKIPPED] ===")
        judged = _read_csv(ckpt_judged)
    else:
        print(f"\n=== Phase 3: argmax + judge ===")
        judged = run_argmax_and_judge(scored, llm, tokenizer,
                                      n_max=args.n_max, ckpt_path=ckpt_judged)

    # ── Destroy model ──────────────────────────────────────────────────────────
    if need_model:
        from src.cot_vision import destroy_llm
        elapsed = time.perf_counter() - t0
        print(f"\nInference done in {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer, llm
        gc.collect()

    # ── Phase 4: Plot ──────────────────────────────────────────────────────────
    model_safe = args.model.replace("/", "_")
    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    print(f"\n=== Phase 4: plot ({len(image_keys)} unique images) ===")

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        model_safe=model_safe,
    )

    plot_all(judged, existing, output_dir)

    if ckpt_scored.exists():
        plot_score_spread(ckpt_scored, ckpt_judged, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
