"""
Verifier LMV — Logprob-based Verification (arxiv 2607.05391)

Implements the three scaling axes from the LLM-as-a-Verifier paper:
  G = 20  (fixed): logprob expectation over A-T score tokens → continuous reward
  K (--n_evals)  : repeated evaluations averaged → lower variance
  C (--n_criteria): criteria decomposition averaged → richer rubric signal

Two selection modes (--selection_mode):
  pointwise  — each candidate scored independently; argmax = winner
  ppt        — Probabilistic Pivot Tournament (pairwise logprob comparisons)

Two generation strategies × N ∈ {1, 5, 10}:
  direct   — N classification samples at T=0.8
  self_cot — greedy description (T=0.0) then N classification samples at T=0.8

Outputs compared against subsampled Vision Baseline, Self-CoT, CoT YAML,
Desc-Aug, and the text-score verifier from verifier_vision.py.

Usage:
  uv run python src/verifier_lmv.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_lmv/ \\
    --vis_dir outputs/vision_assessment/ \\
    --cot_dir outputs/cot_vision/ \\
    --aug_dir outputs/desc_augmented_vision/ \\
    --verifier_dir outputs/verifier_vision/ \\
    --n_samples 50 --n_datasets 3 --seed 42 \\
    --n_evals 4 --n_criteria 3 --selection_mode pointwise

  # PPT mode:
  uv run python src/verifier_lmv.py ... --selection_mode ppt --ppt_pivots 3

  # Ablation — different K/C → use a distinct --output_dir:
  uv run python src/verifier_lmv.py ... --n_evals 1 --n_criteria 1 \\
    --output_dir outputs/verifier_lmv_K1C1/

  # Resume from checkpoints:
  uv run python src/verifier_lmv.py ... --from_ckpt

  # Plot only (all checkpoints must exist):
  uv run python src/verifier_lmv.py ... --plot_only
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
from PIL import Image as PILImage
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

N_DATASETS_PER_TASK = 3
SEED                = 42
N_MAX_DEFAULT       = 10

# Score tokens A=20 (best) … T=1 (worst), matching the paper's φ mapping
SCORE_TOKENS: list[str] = list("ABCDEFGHIJKLMNOPQRST")
SCORE_VALUES: dict[str, int] = {letter: 20 - i for i, letter in enumerate(SCORE_TOKENS)}
SCORE_TOKENS_SET = set(SCORE_TOKENS)
G = len(SCORE_TOKENS)  # = 20

# ── Criteria (C=1 and C=3) ─────────────────────────────────────────────────────

_CRITERIA_DISEASE_C1 = {
    "Visual_Match": (
        "Does the prediction correctly identify the crop/plant and condition "
        "(disease, pest, or stress) in a way that is specific and visually "
        "consistent with the symptom pattern, affected parts, and damage "
        "characteristics visible in the image?"
    ),
}
_CRITERIA_DISEASE_C3 = {
    "Crop_and_Parts": (
        "Does the prediction correctly identify the crop or plant species "
        "and the affected parts (leaves, stem, fruit, roots) visible in the image?"
    ),
    "Symptom_Characteristics": (
        "Does the prediction align with the visible symptom characteristics — "
        "color, texture, shape, and spatial pattern of damage or stress?"
    ),
    "Condition_Specificity": (
        "Is the stated condition (disease name, pest type, or stress factor) "
        "specific, not generic, and visually consistent with all features in the image?"
    ),
}

_CRITERIA_WEED_C1 = {
    "Visual_Match": (
        "Does the prediction correctly identify the weed species in a way that "
        "is botanically consistent with the visible leaf shape, arrangement, "
        "margins, growth habit, and any reproductive structures in the image?"
    ),
}
_CRITERIA_WEED_C3 = {
    "Morphological_Features": (
        "Does the prediction match visible leaf arrangement, shape, margins, "
        "venation pattern, and stem characteristics shown in the image?"
    ),
    "Growth_Habit": (
        "Does the prediction align with the plant's growth habit (erect, "
        "prostrate, climbing) and any visible reproductive structures "
        "(flowers, seeds, seed heads)?"
    ),
    "Botanical_Identification": (
        "Is the botanical identification (scientific name and/or common name) "
        "specific and consistent with all visible features in the image?"
    ),
}


def _criteria_for(task_type: str, n_criteria: int) -> dict[str, str]:
    """Return the appropriate criteria dict for a task and C value."""
    is_weed = task_type == "weed_identification"
    if n_criteria == 1:
        return _CRITERIA_WEED_C1 if is_weed else _CRITERIA_DISEASE_C1
    return _CRITERIA_WEED_C3 if is_weed else _CRITERIA_DISEASE_C3


try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None          # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]

# ── CSV helpers ────────────────────────────────────────────────────────────────

CANDIDATE_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "image_path", "gen_mode", "n_idx", "prediction",
]
SCORED_FIELDS  = CANDIDATE_FIELDS + ["ppt_reward"]
JUDGED_FIELDS  = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "gen_mode", "n_max",
    "final_prediction", "ppt_reward", "ground_truth",
    "judged_correct", "judge_reasoning",
]


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


# ── Logprob reward extraction ──────────────────────────────────────────────────

def _logprob_reward(logprobs_dict: dict) -> float:
    """Compute E[φ(v)] over score tokens from a vLLM logprobs dict.

    logprobs_dict: {token_id: Logprob} for one generated token position.
    Returns normalized reward in [0, 1].
    """
    total = 0.0
    mass  = 0.0
    for token_id, lp_obj in logprobs_dict.items():
        decoded = getattr(lp_obj, "decoded_token", "").strip().upper()
        if decoded in SCORE_TOKENS_SET:
            p = float(np.exp(lp_obj.logprob))
            total += p * SCORE_VALUES[decoded]
            mass  += p
    if mass < 1e-9:
        return 0.5  # no score token in top logprobs — fallback to neutral
    return (total / mass) / 20.0  # normalize [1,20] → [0,1]


def _find_score_tokens_in_output(output) -> list[dict]:
    """Return logprob dicts at positions where the model generated an A-T letter."""
    score_logprobs: list[dict] = []
    for lp_dict in (output.outputs[0].logprobs or []):
        if lp_dict is None:
            continue
        # Check if the generated token at this position is a score letter
        for token_id, lp_obj in lp_dict.items():
            if getattr(lp_obj, "decoded_token", "").strip().upper() in SCORE_TOKENS_SET:
                score_logprobs.append(lp_dict)
                break
        if len(score_logprobs) == 2:
            break
    return score_logprobs


def _extract_pairwise_rewards(output) -> tuple[float, float]:
    """Extract rewards for candidates A and B from a single pairwise generation."""
    positions = _find_score_tokens_in_output(output)
    r_a = _logprob_reward(positions[0]) if len(positions) > 0 else 0.5
    r_b = _logprob_reward(positions[1]) if len(positions) > 1 else 0.5
    return r_a, r_b


# ── Prompt builders ────────────────────────────────────────────────────────────

def _import_cot_helpers():
    from src.cot_vision import (
        _sys_msg, _cls_question, _desc_question,
        _build_prompt, _clean, load_llm, destroy_llm,
    )
    return _sys_msg, _cls_question, _desc_question, _build_prompt, _clean, load_llm, destroy_llm


def _raw_cls_question(task_type: str, crop: str | None) -> str:
    import yaml
    _p = yaml.safe_load(
        (Path(__file__).parent / "config" / "prompts.yaml").read_text()
    )
    if task_type == "weed_identification":
        return _p["visual_classification_weed"]
    return _p["visual_classification_disease"].format(crop=crop or "plant")


def _pointwise_prompt_text(task_type: str, crop: str | None,
                            prediction: str, criterion_desc: str) -> str:
    question = _raw_cls_question(task_type, crop)
    return (
        "You are evaluating an agricultural image classification prediction.\n\n"
        f'Question asked: "{question}"\n'
        f'Prediction: "{prediction[:300]}"\n\n'
        f"Evaluation criterion: {criterion_desc}\n\n"
        "Rate this prediction with a single letter grade where:\n"
        "  A = 20 (perfectly correct, specific, and visually grounded)\n"
        "  J = 11 (borderline — partially correct or vague)\n"
        "  T = 1  (completely wrong or irrelevant to the image)\n\n"
        "Output only the single letter grade (A through T):"
    )


def _pairwise_prompt_text(task_type: str, crop: str | None,
                           pred_a: str, pred_b: str, criterion_desc: str) -> str:
    question = _raw_cls_question(task_type, crop)
    return (
        "You are evaluating two agricultural image classification predictions.\n\n"
        f'Question asked: "{question}"\n'
        f'Candidate A: "{pred_a[:300]}"\n'
        f'Candidate B: "{pred_b[:300]}"\n\n'
        f"Evaluation criterion: {criterion_desc}\n\n"
        "Rate each prediction with a letter grade where:\n"
        "  A = 20 (perfectly correct, specific, and visually grounded)\n"
        "  J = 11 (borderline — partially correct or vague)\n"
        "  T = 1  (completely wrong or irrelevant to the image)\n\n"
        "Output exactly two letters separated by a space: first for Candidate A, "
        "then for Candidate B. Example: \"F K\" means A is good, B is mediocre.\n"
        "Output:"
    )


# ── Dataset selection (shared with verifier_vision) ───────────────────────────

def _build_task_datasets(config_path: str) -> dict[str, dict[str, list[str]]]:
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
        t: {"primary": sorted(primary.get(t, [])), "override": sorted(override.get(t, []))}
        for t in tasks
    }


def select_datasets(n_per_task: int = N_DATASETS_PER_TASK,
                    seed: int = SEED,
                    config_path: str = "src/config/datasets.yaml") -> dict[str, list[str]]:
    pools = _build_task_datasets(config_path)
    rng   = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for task, buckets in pools.items():
        primary  = buckets["primary"]
        override = buckets["override"]
        n_primary = min(n_per_task, len(primary))
        chosen    = rng.sample(primary, n_primary)
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
    from src.cot_vision import load_dataset_classes, DATASET_TASK_TYPES
    by_class    = load_dataset_classes(ds_name, n_samples, test_split)
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
                "image_path": str(p), "image_name": p.name,
            })
    random.Random(seed).shuffle(items)
    return items


# ── Phase 1: Generate candidates (same as verifier_vision) ────────────────────

def run_generate(
    items: list[dict],
    llm,
    tokenizer,
    n_max: int = N_MAX_DEFAULT,
    batch_size: int = 256,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Generate n_max candidates per image for direct and self_cot strategies."""
    (_sys_msg, _cls_question, _desc_question,
     _build_prompt, _clean, _, _) = _import_cot_helpers()

    _GEN_P  = SamplingParams(temperature=0.8, max_tokens=64, repetition_penalty=1.1, n=n_max)
    _DESC_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)

    rows: list[dict] = []

    # ── Strategy A: direct ────────────────────────────────────────────────────
    print(f"\n[generate/direct] {len(items)} images × {n_max} candidates")
    for start in tqdm(range(0, len(items), batch_size), desc="direct", unit="batch"):
        batch   = items[start:start + batch_size]
        prompts = [
            _build_prompt(
                _sys_msg() + [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": _cls_question(item["task_type"], item["crop"])},
                ]}],
                tokenizer,
                PILImage.open(item["image_path"]).convert("RGB"),
            )
            for item in batch
        ]
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

    # ── Strategy B: self_cot ──────────────────────────────────────────────────
    print(f"\n[generate/self_cot] {len(items)} images — describe (T=0) then ×{n_max} classify")
    descriptions:   list[str]        = []
    desc_histories: list[list[dict]] = []

    for start in tqdm(range(0, len(items), batch_size), desc="self_cot desc", unit="batch"):
        batch    = items[start:start + batch_size]
        prompts, hists = [], []
        for item in batch:
            desc_q = _desc_question(item["task_type"], item["crop"])
            hist   = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc_q},
            ]}]
            hists.append(hist)
            prompts.append(_build_prompt(
                hist, tokenizer,
                PILImage.open(item["image_path"]).convert("RGB"),
            ))
        outputs = llm.generate(prompts, _DESC_P)
        for hist, out in zip(hists, outputs):
            resp = _clean(out.outputs[0].text)
            descriptions.append(resp)
            desc_histories.append(hist + [{"role": "assistant", "content": resp}])
        del prompts

    for start in tqdm(range(0, len(items), batch_size), desc="self_cot cls", unit="batch"):
        batch       = items[start:start + batch_size]
        batch_hists = desc_histories[start:start + batch_size]
        prompts = [
            _build_prompt(
                hist + [{"role": "user", "content": _cls_question(item["task_type"], item["crop"])}],
                tokenizer,
                PILImage.open(item["image_path"]).convert("RGB"),
            )
            for item, hist in zip(batch, batch_hists)
        ]
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


# ── Phase 2a: Pointwise logprob scoring ───────────────────────────────────────

def run_verify_pointwise(
    candidates: list[dict],
    llm,
    tokenizer,
    n_evals: int = 4,
    n_criteria: int = 3,
    batch_size: int = 256,
    seed: int = SEED,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Score each candidate with G=20 logprob expected reward, averaged over K×C.

    Builds (candidate × criterion × eval_rep) scoring requests and runs them
    in a single batched pass for efficiency.
    """
    (_sys_msg, _, _, _build_prompt, _, _, _) = _import_cot_helpers()
    _SCORE_P = SamplingParams(temperature=0.0, max_tokens=5, logprobs=G)

    # Build flat request list: (row, criterion_desc, k_rep)
    requests: list[tuple[dict, str, int]] = [
        (row, crit_desc, k)
        for row in candidates
        for crit_desc in _criteria_for(row["task_type"], n_criteria).values()
        for k in range(n_evals)
    ]

    print(f"\n[verify/pointwise] {len(candidates)} candidates × "
          f"{n_criteria} criteria × {n_evals} evals = {len(requests)} scoring calls")

    # Accumulate rewards per candidate key
    reward_acc: dict[tuple, list[float]] = defaultdict(list)
    _key = lambda r: (r["agml_dataset"], r["image_name"], r["gen_mode"], r["n_idx"])

    for start in tqdm(range(0, len(requests), batch_size), desc="pointwise", unit="batch"):
        batch   = requests[start:start + batch_size]
        prompts = []
        for row, crit_desc, _k in batch:
            text = _pointwise_prompt_text(
                row["task_type"], row.get("crop") or None,
                row["prediction"], crit_desc,
            )
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            prompts.append(_build_prompt(
                msgs, tokenizer,
                PILImage.open(row["image_path"]).convert("RGB"),
            ))
        outputs = llm.generate(prompts, _SCORE_P)
        for (row, _cd, _k), out in zip(batch, outputs):
            lp_list = out.outputs[0].logprobs or []
            reward  = _logprob_reward(lp_list[0]) if lp_list else 0.5
            reward_acc[_key(row)].append(reward)
        del prompts

    # Average K×C rewards per candidate
    scored = []
    for row in candidates:
        k   = _key(row)
        avg = float(np.mean(reward_acc[k])) if reward_acc[k] else 0.5
        scored.append({**row, "ppt_reward": avg})

    if ckpt_path:
        _write_csv(scored, ckpt_path, SCORED_FIELDS)
    return scored


# ── Phase 2b: PPT pairwise logprob scoring ────────────────────────────────────

def _score_pairs_batch(
    pair_specs: list[tuple],           # [(image_key, i, j), ...]
    groups: dict[tuple, list[dict]],   # image_key → [candidate_rows]
    criteria_by_task: dict[tuple, dict[str, str]],  # image_key → criteria dict
    llm,
    tokenizer,
    n_evals: int,
    batch_size: int,
) -> list[tuple[float, float]]:
    """Score all (i, j) pairs across all images, returns (reward_i, reward_j) per spec."""
    (_sys_msg, _, _, _build_prompt, _, _, _) = _import_cot_helpers()
    _PAIR_P = SamplingParams(temperature=0.0, max_tokens=10, logprobs=G)

    # Expand to (pair_spec, criterion_desc, k_rep)
    requests: list[tuple[int, str, int]] = [
        (spec_idx, crit_desc, k)
        for spec_idx, (key, _i, _j) in enumerate(pair_specs)
        for crit_desc in criteria_by_task[key].values()
        for k in range(n_evals)
    ]

    rewards_a: dict[int, list[float]] = defaultdict(list)
    rewards_b: dict[int, list[float]] = defaultdict(list)

    for start in range(0, len(requests), batch_size):
        batch   = requests[start:start + batch_size]
        prompts = []
        for spec_idx, crit_desc, _k in batch:
            key, i, j = pair_specs[spec_idx]
            row_a, row_b = groups[key][i], groups[key][j]
            text = _pairwise_prompt_text(
                row_a["task_type"], row_a.get("crop") or None,
                row_a["prediction"], row_b["prediction"], crit_desc,
            )
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            prompts.append(_build_prompt(
                msgs, tokenizer,
                PILImage.open(row_a["image_path"]).convert("RGB"),
            ))
        outputs = llm.generate(prompts, _PAIR_P)
        for (spec_idx, _cd, _k), out in zip(batch, outputs):
            r_a, r_b = _extract_pairwise_rewards(out)
            rewards_a[spec_idx].append(r_a)
            rewards_b[spec_idx].append(r_b)
        del prompts

    return [
        (float(np.mean(rewards_a[i])), float(np.mean(rewards_b[i])))
        for i in range(len(pair_specs))
    ]


def run_verify_ppt(
    candidates: list[dict],
    llm,
    tokenizer,
    n_evals: int = 4,
    n_criteria: int = 3,
    k_pivots: int = 3,
    batch_size: int = 256,
    seed: int = SEED,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Probabilistic Pivot Tournament: ring pass + pivot rounds using pairwise logprob scoring.

    Algorithm (Figure 6 / Algorithm 1 of arxiv 2607.05391):
      1. Ring pass: random Hamiltonian cycle → N pairwise comparisons (bias-cancelling)
      2. Pivot selection: top-k by ring-pass mean preference w_i/c_i
      3. Pivot rounds: non-pivot×pivot + pivot×pivot comparisons
      4. Winner: argmax_i (w_i / c_i)
    """
    rng = random.Random(seed)

    # Group candidates by (dataset, image_name, gen_mode)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        key = (row["agml_dataset"], row["image_name"], row["gen_mode"])
        groups[key].append(row)

    # Sort each group by n_idx for deterministic indexing
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("n_idx", 0)))

    # Pre-compute criteria dict per image key (keyed by task_type)
    criteria_by_task: dict[tuple, dict[str, str]] = {
        key: _criteria_for(rows[0]["task_type"], n_criteria)
        for key, rows in groups.items()
    }

    # Build ring-pass pair specs for all images simultaneously
    ring_specs: list[tuple] = []      # (key, i, j)
    group_perms: dict[tuple, list[int]] = {}
    for key, rows in groups.items():
        n    = len(rows)
        perm = list(range(n))
        rng.shuffle(perm)
        group_perms[key] = perm
        for t in range(n):
            ring_specs.append((key, perm[t], perm[(t + 1) % n]))

    print(f"\n[verify/ppt] {len(groups)} images, "
          f"{sum(len(r) for r in groups.values())} candidates, "
          f"k_pivots={k_pivots}")
    print(f"  Ring pass: {len(ring_specs)} pairs")

    ring_rewards = _score_pairs_batch(
        ring_specs, groups, criteria_by_task, llm, tokenizer, n_evals, batch_size,
    )

    # Accumulate wins and counts from ring pass
    wins:   dict[tuple, dict[int, float]] = {key: defaultdict(float) for key in groups}
    counts: dict[tuple, dict[int, int]]   = {key: defaultdict(int)   for key in groups}
    for (key, i, j), (r_i, r_j) in zip(ring_specs, ring_rewards):
        wins[key][i] += r_i;  counts[key][i] += 1
        wins[key][j] += r_j;  counts[key][j] += 1

    # Select pivots per image
    pivot_specs: list[tuple] = []
    group_pivots: dict[tuple, list[int]] = {}
    for key, rows in groups.items():
        n         = len(rows)
        mean_w    = [wins[key][i] / counts[key][i] if counts[key][i] else 0.0 for i in range(n)]
        pivots    = sorted(range(n), key=lambda i: -mean_w[i])[:k_pivots]
        non_pivot = [i for i in range(n) if i not in pivots]
        group_pivots[key] = pivots
        for i in non_pivot:
            for p in pivots:
                pivot_specs.append((key, i, p))
        for a in range(len(pivots)):
            for b in range(a + 1, len(pivots)):
                pivot_specs.append((key, pivots[a], pivots[b]))

    print(f"  Pivot rounds: {len(pivot_specs)} pairs")

    if pivot_specs:
        pivot_rewards = _score_pairs_batch(
            pivot_specs, groups, criteria_by_task, llm, tokenizer, n_evals, batch_size,
        )
        for (key, i, j), (r_i, r_j) in zip(pivot_specs, pivot_rewards):
            wins[key][i] += r_i;  counts[key][i] += 1
            wins[key][j] += r_j;  counts[key][j] += 1

    # Build scored rows
    scored = []
    for key, rows in groups.items():
        n     = len(rows)
        final = [wins[key][i] / counts[key][i] if counts[key][i] else 0.0 for i in range(n)]
        for idx, row in enumerate(rows):
            scored.append({**row, "ppt_reward": final[idx]})

    if ckpt_path:
        _write_csv(scored, ckpt_path, SCORED_FIELDS)
    return scored


# ── Phase 3: Argmax + Judge ───────────────────────────────────────────────────

def run_argmax_and_judge(
    scored: list[dict],
    llm,
    tokenizer,
    n_max: int = N_MAX_DEFAULT,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Pick best candidate per (image, gen_mode, n_slice) by ppt_reward and judge."""
    from src.common.metrics import apply_judge
    import yaml
    _p = yaml.safe_load(
        (Path(__file__).parent / "config" / "prompts.yaml").read_text()
    )
    judge_system   = _p["judge_system_vision"]
    judge_template = _p["judge_classification_template"]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scored:
        key = (row["agml_dataset"], row["image_name"], row["gen_mode"])
        groups[key].append(row)

    to_judge: list[dict] = []
    n_slices = [n for n in (1, 5, 10) if n <= n_max]
    for (ds, img, gen_mode), cands in groups.items():
        cands = sorted(cands, key=lambda r: int(r.get("n_idx", 0)))
        gt    = cands[0]["class_label"]
        for n_slice in n_slices:
            subset = cands[:n_slice]
            best   = max(subset, key=lambda r: float(r.get("ppt_reward", 0)))
            to_judge.append({
                "agml_dataset":     ds,
                "class_label":      gt,
                "task_type":        cands[0]["task_type"],
                "crop":             cands[0].get("crop", ""),
                "image_name":       img,
                "gen_mode":         gen_mode,
                "n_max":            n_slice,
                "final_prediction": best["prediction"],
                "ppt_reward":       best.get("ppt_reward", 0),
                "ground_truth":     gt,
                "judged_correct":   "",
                "judge_reasoning":  "",
            })

    print(f"\n[judge] {len(to_judge)} rows")
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


# ── Load and filter existing CSVs ─────────────────────────────────────────────

def load_existing_rows(
    image_keys: set[tuple[str, str]],
    vis_dir: Path,
    cot_dir: Path,
    aug_dir: Path,
    verifier_dir: Path,
    model_safe: str,
) -> dict[str, list[dict]]:
    """Load and filter existing experiment CSVs to the selected image set."""
    def _filter(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if (r.get("agml_dataset"), r.get("image_name")) in image_keys]

    result: dict[str, list[dict]] = {}

    baseline_csv = vis_dir / f"cls_{model_safe}.csv"
    if baseline_csv.exists():
        result["Vision Baseline"] = _filter(_read_csv(baseline_csv))
        print(f"  Vision Baseline: {len(result['Vision Baseline'])} rows")
    else:
        print(f"  [skip] {baseline_csv.name}")

    cot_csv = cot_dir / f"cot_cls_{model_safe}.csv"
    if cot_csv.exists():
        cot_rows = _filter(_read_csv(cot_csv))
        result["Self-CoT"] = [r for r in cot_rows if r.get("condition") == "self_cot"]
        result["CoT YAML"] = [r for r in cot_rows if r.get("condition") == "cot_yaml"]
        print(f"  Self-CoT: {len(result['Self-CoT'])} rows  "
              f"CoT YAML: {len(result['CoT YAML'])} rows")
    else:
        print(f"  [skip] {cot_csv.name}")

    aug_csv = aug_dir / f"aug_cls_all_{model_safe}.csv"
    if aug_csv.exists():
        result["Desc-Aug"] = _filter(_read_csv(aug_csv))
        print(f"  Desc-Aug: {len(result['Desc-Aug'])} rows")
    else:
        print(f"  [skip] {aug_csv.name}")

    # Text-score verifier (verifier_vision.py) for direct comparison
    ver_csv = verifier_dir / "ckpt_judged.csv"
    if ver_csv.exists():
        ver_rows = _filter(_read_csv(ver_csv))
        for cond in ("Direct-N1", "Direct-N5", "Direct-N10",
                     "SelfCoT-N1", "SelfCoT-N5", "SelfCoT-N10"):
            gen_mode = "direct" if cond.startswith("Direct") else "self_cot"
            n_max    = int(cond.split("N")[1])
            result[cond] = [
                r for r in ver_rows
                if r.get("gen_mode") == gen_mode and str(r.get("n_max")) == str(n_max)
            ]
        total = sum(len(v) for k, v in result.items() if k.startswith("Direct") or k.startswith("SelfCoT"))
        print(f"  Text-verifier conditions: {total} rows total")
    else:
        print(f"  [skip] {ver_csv.name} (verifier_vision judged CSV)")

    return result


# ── Condition labels and row merging ──────────────────────────────────────────

def _condition_label(gen_mode: str, n_max: int, selection_mode: str) -> str:
    prefix = "LMV" if selection_mode == "pointwise" else "PPT"
    gm     = "Direct" if gen_mode == "direct" else "SelfCoT"
    return f"{prefix}-{gm}-N{n_max}"


def build_rows_by_condition(
    judged: list[dict],
    existing: dict[str, list[dict]],
    selection_mode: str,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in judged:
        label = _condition_label(row["gen_mode"], int(row["n_max"]), selection_mode)
        by_cond[label].append(row)
    result.update(by_cond)

    for label, rows in existing.items():
        if rows:
            result[label] = rows

    return result


# ── Reward spread analysis ────────────────────────────────────────────────────

def plot_reward_spread(
    ckpt_scored: Path,
    ckpt_judged: Path,
    output_dir: Path,
    selection_mode: str,
) -> None:
    """Continuous reward [0,1] spread per gen_mode vs correctness at each N slice."""
    import matplotlib.pyplot as plt

    scored_all = _read_csv(ckpt_scored)
    judged     = _read_csv(ckpt_judged)

    correctness_by_n: dict[tuple, dict[int, int]] = defaultdict(dict)
    for r in judged:
        n = int(r.get("n_max", 0))
        if n in (1, 5, 10):
            key = (r["agml_dataset"], r["image_name"], r["gen_mode"])
            try:
                correctness_by_n[key][n] = int(r["judged_correct"])
            except (ValueError, KeyError):
                pass

    raw_groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in scored_all:
        key = (r["agml_dataset"], r["image_name"], r["gen_mode"])
        raw_groups[key].append((int(r.get("n_idx", 0)), float(r.get("ppt_reward", 0.5))))
    groups: dict[tuple, list[float]] = {
        k: [s for _, s in sorted(v)] for k, v in raw_groups.items()
    }

    gen_modes  = ["direct", "self_cot"]
    gen_labels = {"direct": "Direct", "self_cot": "Self-CoT"}
    colors     = {"correct": "#2ca02c", "incorrect": "#d62728"}
    mode_title = "LMV Pointwise" if selection_mode == "pointwise" else "PPT Pairwise"

    for n_slice in (1, 5, 10):
        all_rewards: dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}
        std_data:    dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}
        margin_data: dict[str, dict[str, list[float]]] = {gm: {"correct": [], "incorrect": []} for gm in gen_modes}

        for key, all_s in groups.items():
            _, _, gen_mode = key
            if gen_mode not in all_rewards:
                continue
            jc = correctness_by_n.get(key, {}).get(n_slice)
            if jc is None:
                continue
            rewards = all_s[:n_slice]
            label   = "correct" if jc else "incorrect"
            all_rewards[gen_mode][label].extend(rewards)
            std_data[gen_mode][label].append(float(np.std(rewards)))
            sorted_r = sorted(rewards, reverse=True)
            margin_data[gen_mode][label].append(
                float(sorted_r[0] - sorted_r[1]) if len(sorted_r) > 1 else 0.0
            )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        for row, gm in enumerate(gen_modes):
            lbl = gen_labels[gm]

            ax = axes[row, 0]
            data_v = [all_rewards[gm]["correct"], all_rewards[gm]["incorrect"]]
            if all(len(d) > 1 for d in data_v):
                parts = ax.violinplot(data_v, positions=[0, 1], showmedians=True, showextrema=True)
                for pc, c in zip(parts["bodies"], [colors["correct"], colors["incorrect"]]):
                    pc.set_facecolor(c); pc.set_alpha(0.6)
                for vk in ("cmedians", "cmins", "cmaxes", "cbars"):
                    if vk in parts:
                        parts[vk].set_color("black"); parts[vk].set_linewidth(1.2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Correct\n(n={len(std_data[gm]['correct'])})",
                                 f"Incorrect\n(n={len(std_data[gm]['incorrect'])})"])
            ax.set_ylabel("Reward [0–1]")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{lbl} — Reward Distribution")
            ax.yaxis.grid(True, alpha=0.3)

            ax = axes[row, 1]
            if n_slice > 1:
                bp = ax.boxplot(
                    [std_data[gm]["correct"], std_data[gm]["incorrect"]],
                    labels=[f"Correct\n(n={len(std_data[gm]['correct'])})",
                            f"Incorrect\n(n={len(std_data[gm]['incorrect'])})"],
                    patch_artist=True, widths=0.5,
                )
                for patch, c in zip(bp["boxes"], [colors["correct"], colors["incorrect"]]):
                    patch.set_facecolor(c); patch.set_alpha(0.5)
            else:
                ax.text(0.5, 0.5, "N=1: std undefined",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(f"Reward Std (N={n_slice})")
            ax.set_title(f"{lbl} — Reward Spread (Std)")
            ax.yaxis.grid(True, alpha=0.3)

            ax = axes[row, 2]
            if n_slice > 1:
                bp = ax.boxplot(
                    [margin_data[gm]["correct"], margin_data[gm]["incorrect"]],
                    labels=[f"Correct\n(n={len(margin_data[gm]['correct'])})",
                            f"Incorrect\n(n={len(margin_data[gm]['incorrect'])})"],
                    patch_artist=True, widths=0.5,
                )
                for patch, c in zip(bp["boxes"], [colors["correct"], colors["incorrect"]]):
                    patch.set_facecolor(c); patch.set_alpha(0.5)
            else:
                ax.text(0.5, 0.5, "N=1: margin undefined",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("Reward gap (1st − 2nd)")
            ax.set_title(f"{lbl} — Winning Margin")
            ax.yaxis.grid(True, alpha=0.3)

        fig.suptitle(f"{mode_title} — Reward Spread vs Correctness (N={n_slice})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"lmv_reward_spread_N{n_slice}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── Per-dataset bar chart ─────────────────────────────────────────────────────

def plot_per_dataset(
    rows_by_cond: dict[str, list[dict]],
    output_path: Path,
    selection_mode: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.common.method_comparison import CONDITION_STYLES, CONDITION_COLOR
    from src.common.distribution_plots import gather_judge_per_class_by_dataset

    judge_by_cls = gather_judge_per_class_by_dataset(rows_by_cond)
    ds_f1: dict[str, dict[str, float]] = {}
    all_ds: set[str] = set()
    for cond, by_dt in judge_by_cls.items():
        ds_f1[cond] = {}
        for (ds, task), vals in by_dt.items():
            ds_f1[cond][ds] = float(np.mean(vals)) if vals else 0.0
            all_ds.add(ds)

    def _sort_key(ds: str) -> tuple[str, str]:
        for rows in rows_by_cond.values():
            for r in rows:
                if r.get("agml_dataset") == ds:
                    return (r.get("task_type", ""), ds)
        return ("", ds)

    datasets  = sorted(all_ds, key=_sort_key)
    cond_order = [lbl for lbl, _ in CONDITION_STYLES if lbl in ds_f1]
    n_conds    = len(cond_order)
    n_ds       = len(datasets)
    bar_w      = 0.8 / max(n_conds, 1)
    offsets    = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_w

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 1.5), 6))
    for i, cond in enumerate(cond_order):
        vals  = [ds_f1[cond].get(ds, np.nan) for ds in datasets]
        xs    = np.arange(n_ds) + offsets[i]
        color = CONDITION_COLOR.get(cond, "#888888")
        bars  = ax.bar(xs, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=cond)
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    short = [ds.replace("_classification", "").replace("_disease", "").replace("_", " ")
             for ds in datasets]
    ax.set_xticks(np.arange(n_ds))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Judged F1")
    ax.set_ylim(0, 1.05)
    mode_lbl = "LMV Pointwise" if selection_mode == "pointwise" else "PPT Pairwise"
    ax.set_title(f"Verifier {mode_lbl} — Mean Judged F1 per Dataset")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[mpatches.Patch(color=CONDITION_COLOR.get(c, "#888"), label=c) for c in cond_order],
        fontsize=7, ncol=2, loc="upper right",
    )

    task_seen: list[str] = []
    for idx, ds in enumerate(datasets):
        tt = _sort_key(ds)[0]
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
    selection_mode: str,
) -> None:
    from src.common.method_comparison import (
        plot_overall_judge_comparison,
        plot_per_task_judge_comparison,
        plot_baseline_vs_condition_scatter,
    )

    rows_by_cond = build_rows_by_condition(judged, existing, selection_mode)
    present      = [c for c in rows_by_cond if rows_by_cond[c]]
    print(f"\n[plot] conditions: {present}")

    prefix = "LMV" if selection_mode == "pointwise" else "PPT"

    plot_overall_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "lmv_overall.png",
        title=f"Verifier {prefix} — Overall Mean Judged F1",
    )
    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "lmv_per_task.png",
        title=f"Verifier {prefix} — Mean Judged F1 by Task",
    )
    plot_per_dataset(
        rows_by_cond,
        output_path=output_dir / "lmv_per_dataset.png",
        selection_mode=selection_mode,
    )

    direct_conds = {k: v for k, v in rows_by_cond.items()
                    if "Direct" in k or k == "Vision Baseline"}
    selfcot_conds = {k: v for k, v in rows_by_cond.items()
                     if "SelfCoT" in k or k == "Vision Baseline"}

    plot_baseline_vs_condition_scatter(
        direct_conds,
        output_path=output_dir / "lmv_scatter_direct.png",
        title=f"Verifier {prefix} Direct — Per-Dataset vs Baseline",
        separate=True,
    )
    plot_baseline_vs_condition_scatter(
        selfcot_conds,
        output_path=output_dir / "lmv_scatter_selfcot.png",
        title=f"Verifier {prefix} SelfCoT — Per-Dataset vs Baseline",
        separate=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier LMV — logprob-based verification")
    parser.add_argument("--model",          required=True)
    parser.add_argument("--output_dir",     default="outputs/verifier_lmv/")
    parser.add_argument("--vis_dir",        default="outputs/vision_assessment/")
    parser.add_argument("--cot_dir",        default="outputs/cot_vision/")
    parser.add_argument("--aug_dir",        default="outputs/desc_augmented_vision/")
    parser.add_argument("--verifier_dir",   default="outputs/verifier_vision/",
                        help="Dir with ckpt_judged.csv from verifier_vision.py (for comparison)")
    parser.add_argument("--config",         default="src/config/datasets.yaml")
    parser.add_argument("--test_split",     default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_samples",      type=int, default=50)
    parser.add_argument("--n_datasets",     type=int, default=N_DATASETS_PER_TASK)
    parser.add_argument("--n_max",          type=int, default=N_MAX_DEFAULT)
    parser.add_argument("--seed",           type=int, default=SEED)
    parser.add_argument("--batch_size",     type=int, default=256)
    # LMV-specific
    parser.add_argument("--n_evals",        type=int, default=4,
                        help="K: repeated evaluations per criterion (paper ablation: 1,4,8)")
    parser.add_argument("--n_criteria",     type=int, default=3, choices=[1, 3],
                        help="C: criteria decomposition (1=single, 3=decomposed)")
    parser.add_argument("--selection_mode", default="pointwise",
                        choices=["pointwise", "ppt"],
                        help="pointwise=independent scoring; ppt=Probabilistic Pivot Tournament")
    parser.add_argument("--ppt_pivots",     type=int, default=3,
                        help="k: number of pivots in PPT (only used when --selection_mode ppt)")
    # Control flags
    parser.add_argument("--from_ckpt",      action="store_true")
    parser.add_argument("--score_only",     action="store_true")
    parser.add_argument("--judge_only",     action="store_true")
    parser.add_argument("--plot_only",      action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_candidates = output_dir / "ckpt_candidates.csv"
    ckpt_scored     = output_dir / "ckpt_scored.csv"
    ckpt_judged     = output_dir / "ckpt_judged.csv"

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

    print(f"\n=== Config: K={args.n_evals}, C={args.n_criteria}, "
          f"mode={args.selection_mode}"
          + (f", pivots={args.ppt_pivots}" if args.selection_mode == "ppt" else "")
          + " ===")

    # ── Checkpoint skip flags ──────────────────────────────────────────────────
    skip_generate = (args.plot_only or args.score_only or args.judge_only or
                     (args.from_ckpt and ckpt_candidates.exists()))
    skip_score    = (args.plot_only or args.judge_only or
                     (args.from_ckpt and ckpt_scored.exists()))
    skip_judge    = (args.plot_only or
                     (args.from_ckpt and ckpt_judged.exists()))

    need_model = not (skip_generate and skip_score and skip_judge)
    llm = tokenizer = None

    if need_model:
        from src.cot_vision import load_llm, destroy_llm
        print(f"\n=== Loading model: {args.model} ===")
        llm, tokenizer = load_llm(args.model)

    # ── Phase 1: Generate ──────────────────────────────────────────────────────
    if skip_generate:
        print("\n=== Phase 1: generate [SKIPPED] ===")
        candidates = _read_csv(ckpt_candidates)
    else:
        all_items: list[dict] = []
        for task, ds_names in selected.items():
            for ds in ds_names:
                info            = ds_info.get(ds, {})
                crop            = info.get("crop", "plant")
                class_overrides = info.get("class_overrides", {})
                items           = load_items(ds, task, crop, n_samples=args.n_samples,
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
        print("\n=== Phase 2: verify [SKIPPED] ===")
        scored = _read_csv(ckpt_scored)
    else:
        print(f"\n=== Phase 2: verify ({len(candidates)} candidates, "
              f"mode={args.selection_mode}) ===")
        if args.selection_mode == "ppt":
            scored = run_verify_ppt(
                candidates, llm, tokenizer,
                n_evals=args.n_evals, n_criteria=args.n_criteria,
                k_pivots=args.ppt_pivots, batch_size=args.batch_size,
                seed=args.seed, ckpt_path=ckpt_scored,
            )
        else:
            scored = run_verify_pointwise(
                candidates, llm, tokenizer,
                n_evals=args.n_evals, n_criteria=args.n_criteria,
                batch_size=args.batch_size, seed=args.seed,
                ckpt_path=ckpt_scored,
            )

    # ── Phase 3: Argmax + Judge ────────────────────────────────────────────────
    if skip_judge:
        print("\n=== Phase 3: judge [SKIPPED] ===")
        judged = _read_csv(ckpt_judged)
    else:
        judged = run_argmax_and_judge(
            scored, llm, tokenizer,
            n_max=args.n_max, ckpt_path=ckpt_judged,
        )

    if need_model and llm is not None:
        from src.cot_vision import destroy_llm
        destroy_llm(llm)
        del tokenizer, llm
        gc.collect()

    # ── Phase 4: Plot ──────────────────────────────────────────────────────────
    model_safe  = args.model.replace("/", "_")
    image_keys  = {(r["agml_dataset"], r["image_name"]) for r in judged}
    print(f"\n=== Phase 4: plot ({len(image_keys)} unique images) ===")

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        verifier_dir=Path(args.verifier_dir),
        model_safe=model_safe,
    )

    plot_all(judged, existing, output_dir, args.selection_mode)

    if ckpt_scored.exists():
        plot_reward_spread(ckpt_scored, ckpt_judged, output_dir, args.selection_mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
