"""
Verifier LMV — LLM-as-a-Verifier (arxiv 2607.05391)

Uses Probabilistic Pivot Tournament (PPT) from the official llm-verifier package:
  - llm_verifier.pivot_tournament: ring_cycle, select_pivots, pivot_round_pairs, bradley_terry
  - llm_verifier.fine_grained_reward: extract_score
  - Inline vLLM for generation (Phase 1), PPT scoring, and judge (Phase 3)

Three scaling axes from the paper:
  G = 20  (fixed): score token granularity (A-T letters), baked into extract_score
  K (--n_evals): repeated evaluations per pairwise comparison (averaged)
  C (--n_criteria): criteria decomposition (1 generic or 3 task-specific)

Two generation strategies × N ∈ {1, 5, 10}:
  direct   — N diverse samples at T=0.8
  self_cot — greedy description (T=0.0) then N diverse samples

Condition labels: PPT-Direct-N{1,5,10}, PPT-SelfCoT-N{1,5,10}

Scoring mechanism — two-pass prefill, factored into src/common/
ppt_scoring.py (shared with src/verifier_cot.py's --baseline belief
pairwise duel, which uses this exact same mechanism): pass 1 generates
the comparative analysis only (temperature=1.0, max_tokens=150, no
logprobs needed — the tag content it writes here is discarded, not
scored). Pass 2 continues that exact prefix with "\n<score_A>" (then
separately "\n<score_B>") appended and the very next token constrained to
just the 20 scale letters (+ their leading-space tokenization variants)
via a grammar mask. The resulting top-20 logprobs are then a distribution
already renormalized over the scale itself, not competing against the
model's full vocabulary. See ppt_scoring.py's module docstring for why
this replaced an earlier single-free-generation-call version that
collapsed onto "A" ~96% of the time on a real feasibility test.

Usage:
  uv run python src/verifier_lmv.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_lmv/ \\
    --n_samples 50 --n_datasets 3 --n_evals 4 --n_criteria 3

  # Ablation — different K or C → distinct --output_dir:
  uv run python src/verifier_lmv.py ... --n_evals 1 --n_criteria 1 \\
    --output_dir outputs/verifier_lmv_K1C1/

  # Resume from checkpoints:
  uv run python src/verifier_lmv.py ... --from_ckpt

  # Plot only:
  uv run python src/verifier_lmv.py ... --plot_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import functools
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from PIL import Image as PILImage

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from src.common.ppt_scoring import (
    analysis_before_tags, build_score_prompt, extract_tag_score,
    pairwise_scoring_prompt, score_prefill_params,
)
from src.common.progress import ProgressBar

try:
    from llm_verifier.pivot_tournament import (
        ring_cycle, select_pivots, pivot_round_pairs, bradley_terry,
    )
    from llm_verifier.fine_grained_reward import extract_score
    _HAS_LMV = True
except ImportError:
    _HAS_LMV = False
    ring_cycle = select_pivots = pivot_round_pairs = bradley_terry = extract_score = None

# ── Constants ──────────────────────────────────────────────────────────────────

N_DATASETS_PER_TASK = 3
SEED = 42
N_MAX_DEFAULT = 10   # candidates generated per image (paper calls this N)
DEFAULT_PIVOTS = 2

# ── Criteria ────────────────────────────────────────────────────────────────────

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
_CRITERIA_SPECIES_C1 = {
    "Visual_Match": (
        "Does the prediction correctly identify the plant/weed species in a way that "
        "is botanically consistent with the visible leaf shape, arrangement, "
        "margins, growth habit, and any reproductive structures in the image?"
    ),
}
_CRITERIA_SPECIES_C3 = {
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
_CRITERIA_QUALITY_C1 = {
    "Visual_Match": (
        "Does the prediction correctly identify the quality, ripeness, or condition "
        "in a way that is specific and visually consistent with the size, color, and "
        "surface texture cues visible in the image?"
    ),
}
_CRITERIA_QUALITY_C3 = {
    "Size_and_Shape": (
        "Does the prediction match the visible size and shape relative to a "
        "typical mature specimen?"
    ),
    "Color_and_Ripeness": (
        "Does the prediction align with the observed color and its position "
        "between unripe and overripe/spoiled?"
    ),
    "Surface_Condition": (
        "Does the prediction account for the visible surface texture and firmness "
        "cues (bruising, wrinkling, gloss, softness), and is the stated quality/"
        "ripeness/condition visually consistent with the image?"
    ),
}

# task_type values as they actually appear in datasets.yaml today (DISEASE_TASK_TYPES
# / IDENTIFICATION_TASK_TYPES in src/common/dataset_config.py) — "weed_identification"
# was an earlier task-type name this file predates; every weed/species dataset is now
# tagged species_identification (see _build_task_datasets's own tasks tuple below,
# which had the same stale name and silently selected 0 datasets for it).
_CRITERIA_BY_TASK_TYPE = {
    "species_identification": (_CRITERIA_SPECIES_C1, _CRITERIA_SPECIES_C3),
    "quality_identification": (_CRITERIA_QUALITY_C1, _CRITERIA_QUALITY_C3),
}


def _criteria_for(task_type: str, n_criteria: int) -> dict[str, str]:
    c1, c3 = _CRITERIA_BY_TASK_TYPE.get(task_type, (_CRITERIA_DISEASE_C1, _CRITERIA_DISEASE_C3))
    return c1 if n_criteria == 1 else c3


# ── CSV helpers ────────────────────────────────────────────────────────────────

CANDIDATE_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop",
    "image_name", "image_path", "gen_mode", "n_idx", "prediction",
]
SCORED_FIELDS = CANDIDATE_FIELDS + ["ppt_reward"]
JUDGED_FIELDS = [
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
    if task_type == "species_identification":
        return _p["visual_classification_weed"]
    if task_type == "quality_identification":
        return _p["visual_classification_quality"].format(crop=crop or "plant")
    return _p["visual_classification_disease"].format(crop=crop or "plant")


# ── Dataset selection (same logic as verifier_vision.py) ─────────────────────

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
    tasks = ("disease", "pest_damage", "species_identification", "quality_identification")
    return {
        t: {"primary": sorted(primary.get(t, [])), "override": sorted(override.get(t, []))}
        for t in tasks
    }


def select_datasets(
    n_per_task: int = N_DATASETS_PER_TASK,
    seed: int = SEED,
    config_path: str = "src/config/datasets.yaml",
    subsample: set[str] | None = None,
) -> dict[str, list[str]]:
    """subsample (--subsample) restricts to exactly these agml_name(s) —
    every one of them that falls in a given task's pool, not a further
    n_per_task-capped random draw from within them — instead of the usual
    random n_per_task-per-task selection. Same convention as
    cot_yaml.py's --swap_datasets / verifier_cot.py's --subsample, for
    fast testing against a small, deliberate set (e.g. cot-compare's own
    top gain/loss cot_yaml_swap_datasets.txt)."""
    pools = _build_task_datasets(config_path)
    if subsample is not None:
        selected = {
            task: sorted((set(buckets["primary"]) | set(buckets["override"])) & subsample)
            for task, buckets in pools.items()
        }
        found = {d for names in selected.values() for d in names}
        missing = subsample - found
        if missing:
            print(f"  [warn] --subsample not found (skip/not in any known task pool?): {sorted(missing)}")
        return selected

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for task, buckets in pools.items():
        primary = buckets["primary"]
        override = buckets["override"]
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
    manifest: dict,
    images_root: Path,
    class_overrides: dict[str, str] | None = None,
    excluded_classes: list[str] | None = None,
    seed: int = SEED,
    test_split: dict | None = None,
) -> list[dict]:
    """Same [{"ds","task_type","crop","cls","image_path","image_name"}, ...]
    shape as before this was migrated — now sourced from the same
    HF-manifest pipeline every other stage uses (load_hf_samples /
    load_manifest, src/common/dataset_config.py) instead of the legacy
    agml pip-package loader (cot_vision.load_dataset_classes ->
    agml.data.AgMLDataLoader), which only covers ~20% of this project's
    current dataset corpus — most datasets were added later via
    scripts/agml_hf/download_and_sample.py's HuggingFace-based pipeline
    and were never registered with the old agml package at all, so the
    old loader raised "invalid public source" for the large majority of
    datasets select_datasets() could pick. class_overrides/excluded_classes
    are the same raw (manifest-label-keyed) dicts/lists DatasetConfig
    carries — see excluded_class_pairs()'s docstring for why raw form,
    not the cleaned display form, is correct here."""
    from src.common.dataset_config import load_hf_samples
    from src.common.metrics import sample_class_images

    samples = load_hf_samples(ds_name, manifest, images_root, excluded_classes=excluded_classes)
    by_class: dict[str, list[Path]] = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(Path(s["path"]))

    items: list[dict] = []
    split_cls = (test_split or {}).get(ds_name, {})
    for cls, paths in by_class.items():
        effective = (class_overrides or {}).get(cls, task_type)
        if effective != task_type:
            continue
        if split_cls and cls in split_cls:
            # Use test_split JSON order (same as vision_assessment) for deterministic sampling.
            allowed_ordered = split_cls[cls]
            path_by_name = {p.name: p for p in paths}
            paths = [path_by_name[n] for n in allowed_ordered if n in path_by_name]
        else:
            paths = sorted(paths, key=lambda p: p.name)
        path_by_name = {p.name: p for p in paths}
        for n in sample_class_images(list(path_by_name.keys()), n_samples, seed=seed):
            p = path_by_name[n]
            items.append({
                "ds": ds_name, "task_type": task_type,
                "crop": crop, "cls": cls,
                "image_path": str(p), "image_name": p.name,
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
    """Generate n_max candidates per image for direct and self_cot strategies."""
    (_sys_msg, _cls_question, _desc_question,
     _build_prompt, _clean, _, _) = _import_cot_helpers()

    _GEN_P  = SamplingParams(temperature=0.8, max_tokens=64, repetition_penalty=1.1, n=n_max)
    _DESC_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)

    rows: list[dict] = []

    print(f"\n[generate/direct] {len(items)} images × {n_max} candidates")
    for start in ProgressBar(range(0, len(items), batch_size), desc="direct"):
        batch = items[start:start + batch_size]
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

    print(f"\n[generate/self_cot] {len(items)} images — describe then ×{n_max} classify")
    descriptions:   list[str]        = []
    desc_histories: list[list[dict]] = []

    for start in ProgressBar(range(0, len(items), batch_size), desc="self_cot desc"):
        batch = items[start:start + batch_size]
        prompts, hists = [], []
        for item in batch:
            desc_q = _desc_question(item["task_type"], item["crop"])
            hist = _sys_msg() + [{"role": "user", "content": [
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

    for start in ProgressBar(range(0, len(items), batch_size), desc="self_cot cls"):
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


# ── Phase 2: PPT verification ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _pairwise_prompt_template() -> str:
    """src/config/verifier_prompts.yaml's pairwise_prompt — same shared,
    editable template src/verifier_cot.py's --baseline belief uses (see
    that YAML entry's header comment for why this moved out of a hardcoded
    Python string). Cached — this file has no other reason to load YAML,
    so avoid re-reading it on every batch."""
    path = Path(__file__).parent / "config" / "verifier_prompts.yaml"
    return yaml.safe_load(path.read_text())["pairwise_prompt"]


def _score_pairs_batch(
    pair_specs: list[tuple],              # [(key, a_idx, b_idx), ...]
    groups: dict[tuple, list[dict]],
    criteria_by_key: dict[tuple, dict],   # key → {name: desc}
    llm,
    tokenizer,
    n_evals: int,
    batch_size: int,
    desc: str = "scoring",
) -> list[tuple[float, float]]:
    """Score all (key, a_idx, b_idx) pairs via src/common/ppt_scoring.py's
    two-pass prefill scheme: pass 1 free-generates the comparative
    analysis, pass 2 forces the very next token under a grammar mask (once
    per tag) and reads its logprobs via extract_score.

    Each pair is scored n_evals × n_criteria times; rewards averaged per
    pair. Returns [(avg_ra, avg_rb)] in the same order as pair_specs.
    """
    (_sys_msg, _, _, _build_prompt, _, _, _) = _import_cot_helpers()
    _ANALYSIS_P = SamplingParams(temperature=1.0, max_tokens=150)
    _SCORE_P = score_prefill_params()

    # Expand: each spec × criteria × n_evals evaluations
    requests: list[tuple[int, str, str]] = [
        (spec_idx, crit_name, crit_desc)
        for spec_idx, (key, _a, _b) in enumerate(pair_specs)
        for crit_name, crit_desc in criteria_by_key[key].items()
        for _ in range(n_evals)
    ]

    rewards_a: dict[int, list[float]] = defaultdict(list)
    rewards_b: dict[int, list[float]] = defaultdict(list)

    for start in ProgressBar(range(0, len(requests), batch_size), desc=desc):
        batch = requests[start:start + batch_size]

        base_prompts = []
        for spec_idx, crit_name, crit_desc in batch:
            key, a_idx, b_idx = pair_specs[spec_idx]
            row_a = groups[key][a_idx]
            row_b = groups[key][b_idx]
            question = _raw_cls_question(row_a["task_type"], row_a.get("crop") or None)
            text = pairwise_scoring_prompt(
                rubric=f"{crit_name}: {crit_desc}", question=question,
                trajectory_a=row_a["prediction"], trajectory_b=row_b["prediction"],
                template=_pairwise_prompt_template(),
            )
            msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            base_prompts.append(_build_prompt(
                msgs, tokenizer,
                PILImage.open(row_a["image_path"]).convert("RGB"),
            ))

        # ── Pass 1: free-generate the comparative analysis ───────────────
        analysis_outputs = llm.generate(base_prompts, _ANALYSIS_P)
        analyses = [analysis_before_tags(out.outputs[0].text) for out in analysis_outputs]

        # ── Pass 2: force each tag's very next token, once per tag ───────
        for tag, rewards in (("<score_A>", rewards_a), ("<score_B>", rewards_b)):
            score_prompts = [build_score_prompt(bp, an, tag)
                             for bp, an in zip(base_prompts, analyses)]
            score_outputs = llm.generate(score_prompts, _SCORE_P)
            for (spec_idx, _, _), analysis, out in zip(batch, analyses, score_outputs):
                comp = out.outputs[0]
                reward = extract_tag_score(analysis, tag, comp.text, comp.logprobs or [])
                rewards[spec_idx].append(reward)
        del base_prompts

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
    k_pivots: int = DEFAULT_PIVOTS,
    batch_size: int = 256,
    seed: int = SEED,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Probabilistic Pivot Tournament using llm_verifier.pivot_tournament.

    Algorithm (Algorithm 1, arxiv 2607.05391):
      1. Ring pass: random Hamiltonian cycle (N directed pairs, bias-cancelling)
      2. Pivot selection: top-k by mean preference w_i / c_i
      3. Pivot rounds: non-pivot×pivot + pivot×pivot
      4. ppt_reward = w_i / c_i (Bradley-Terry mean preference)

    Uses ring_cycle, select_pivots, pivot_round_pairs, bradley_terry from
    llm_verifier.pivot_tournament, and extract_score from
    llm_verifier.fine_grained_reward for reward extraction from logprobs.
    """
    if not _HAS_LMV:
        raise RuntimeError(
            "llm-verifier is not installed. Run: uv add llm-verifier"
        )

    rng = random.Random(seed)

    # Group candidates by (ds, image, gen_mode), sort by n_idx
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        key = (row["agml_dataset"], row["image_name"], row["gen_mode"])
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("n_idx", 0)))

    # Precompute criteria per image group
    criteria_by_key: dict[tuple, dict[str, str]] = {
        key: _criteria_for(rows[0]["task_type"], n_criteria)
        for key, rows in groups.items()
    }

    # ── Step 1: Ring pass ────────────────────────────────────────────────────
    rings: dict[tuple, list[tuple[int, int]]] = {
        key: ring_cycle(len(rows), rng)
        for key, rows in groups.items()
    }
    ring_specs: list[tuple] = [
        (key, a, b)
        for key, ring in rings.items()
        for (a, b) in ring
    ]
    n_scoring = len(ring_specs) * len(next(iter(criteria_by_key.values()))) * n_evals
    print(f"\n[PPT] {len(groups)} images × {len(next(iter(groups.values())))} candidates")
    print(f"  Ring pass: {len(ring_specs)} pairs "
          f"({n_scoring} scoring calls with K={n_evals}, C={n_criteria})")

    ring_rewards = _score_pairs_batch(
        ring_specs, groups, criteria_by_key, llm, tokenizer, n_evals, batch_size,
        desc=f"ring ({len(ring_specs)} pairs)",
    )

    # Accumulate ring pass wins / counts (Bradley-Terry)
    wins:   dict[tuple, list[float]] = {key: [0.0] * len(rows) for key, rows in groups.items()}
    counts: dict[tuple, list[int]]   = {key: [0]   * len(rows) for key, rows in groups.items()}
    for (key, a_idx, b_idx), (ra, rb) in zip(ring_specs, ring_rewards):
        p = bradley_terry(ra, rb)
        wins[key][a_idx]   += p
        counts[key][a_idx] += 1
        wins[key][b_idx]   += 1.0 - p
        counts[key][b_idx] += 1

    # ── Steps 2+3: Pivot selection + pivot rounds ────────────────────────────
    pivot_specs: list[tuple] = []
    for key, rows in groups.items():
        n      = len(rows)
        pivots = select_pivots(wins[key], counts[key], k_pivots)
        for (a, b) in pivot_round_pairs(n, pivots):
            pivot_specs.append((key, a, b))

    n_piv_scoring = len(pivot_specs) * len(next(iter(criteria_by_key.values()))) * n_evals
    print(f"  Pivot rounds: {len(pivot_specs)} pairs ({n_piv_scoring} scoring calls)")

    if pivot_specs:
        pivot_rewards = _score_pairs_batch(
            pivot_specs, groups, criteria_by_key, llm, tokenizer, n_evals, batch_size,
            desc=f"pivot ({len(pivot_specs)} pairs)",
        )
        for (key, a_idx, b_idx), (ra, rb) in zip(pivot_specs, pivot_rewards):
            p = bradley_terry(ra, rb)
            wins[key][a_idx]   += p
            counts[key][a_idx] += 1
            wins[key][b_idx]   += 1.0 - p
            counts[key][b_idx] += 1

    # Build scored rows: ppt_reward = mean Bradley-Terry preference
    scored: list[dict] = []
    for key, rows in groups.items():
        for i, row in enumerate(rows):
            reward = wins[key][i] / counts[key][i] if counts[key][i] else 0.5
            scored.append({**row, "ppt_reward": reward})

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
    """Pick best candidate per (image, gen_mode, n_slice) by ppt_reward, then judge."""
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


# ── Load existing CSVs for comparison ─────────────────────────────────────────

def load_existing_rows(
    image_keys: set[tuple[str, str]],
    vis_dir: Path,
    cot_dir: Path,
    aug_dir: Path,
    verifier_dir: Path,
    model_safe: str,
) -> dict[str, list[dict]]:
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

    # Text-score verifier (verifier_vision.py) for head-to-head comparison
    ver_csv = verifier_dir / "ckpt_judged.csv"
    if ver_csv.exists():
        ver_rows = _filter(_read_csv(ver_csv))
        for cond in ("Direct-N1", "Direct-N5", "Direct-N10",
                     "SelfCoT-N1", "SelfCoT-N5", "SelfCoT-N10"):
            gen_mode = "direct" if cond.startswith("Direct") else "self_cot"
            n_val    = int(cond.split("N")[1])
            result[cond] = [
                r for r in ver_rows
                if r.get("gen_mode") == gen_mode and str(r.get("n_max")) == str(n_val)
            ]
        total = sum(len(v) for k, v in result.items()
                    if k.startswith("Direct") or k.startswith("SelfCoT"))
        print(f"  Text-verifier conditions: {total} rows total")
    else:
        print(f"  [skip] {ver_csv.name} (verifier_vision judged CSV)")

    return result


# ── Condition labels ───────────────────────────────────────────────────────────

def _condition_label(gen_mode: str, n_max: int) -> str:
    gm = "Direct" if gen_mode == "direct" else "SelfCoT"
    return f"PPT-{gm}-N{n_max}"


def build_rows_by_condition(
    judged: list[dict],
    existing: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in judged:
        label = _condition_label(row["gen_mode"], int(row["n_max"]))
        by_cond[label].append(row)
    result: dict[str, list[dict]] = dict(by_cond)
    for label, rows in existing.items():
        if rows:
            result[label] = rows
    return result


# ── Reward spread analysis ────────────────────────────────────────────────────

def plot_reward_spread(
    ckpt_scored: Path,
    ckpt_judged: Path,
    output_dir: Path,
) -> None:
    """PPT reward [0,1] spread per gen_mode vs correctness at N=1,5,10."""
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

    for n_slice in (1, 5, 10):
        all_rewards: dict[str, dict[str, list[float]]] = {
            gm: {"correct": [], "incorrect": []} for gm in gen_modes
        }
        std_data:    dict[str, dict[str, list[float]]] = {
            gm: {"correct": [], "incorrect": []} for gm in gen_modes
        }
        margin_data: dict[str, dict[str, list[float]]] = {
            gm: {"correct": [], "incorrect": []} for gm in gen_modes
        }

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
            ax.set_ylabel("PPT Reward [0–1]")
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

        fig.suptitle(f"PPT Reward Spread vs Correctness (N={n_slice})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"ppt_reward_spread_N{n_slice}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ── Per-dataset bar chart ─────────────────────────────────────────────────────

def plot_per_dataset(
    rows_by_cond: dict[str, list[dict]],
    output_path: Path,
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

    datasets   = sorted(all_ds, key=_sort_key)
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
    ax.set_title("Verifier PPT — Mean Judged F1 per Dataset")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[mpatches.Patch(color=CONDITION_COLOR.get(c, "#888"), label=c)
                 for c in cond_order],
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
        output_path=output_dir / "ppt_overall.png",
        title="Verifier PPT — Overall Mean Judged F1",
    )
    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "ppt_per_task.png",
        title="Verifier PPT — Mean Judged F1 by Task",
    )
    plot_per_dataset(rows_by_cond, output_path=output_dir / "ppt_per_dataset.png")

    direct_conds = {k: v for k, v in rows_by_cond.items()
                    if "Direct" in k or k == "Vision Baseline"}
    selfcot_conds = {k: v for k, v in rows_by_cond.items()
                     if "SelfCoT" in k or k == "Vision Baseline"}

    plot_baseline_vs_condition_scatter(
        direct_conds,
        output_path=output_dir / "ppt_scatter_direct.png",
        title="Verifier PPT Direct — Per-Dataset vs Baseline",
        separate=True,
    )
    plot_baseline_vs_condition_scatter(
        selfcot_conds,
        output_path=output_dir / "ppt_scatter_selfcot.png",
        title="Verifier PPT SelfCoT — Per-Dataset vs Baseline",
        separate=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier LMV — PPT logprob verification")
    parser.add_argument("--model",        required=True)
    parser.add_argument("--output_dir",   default="outputs/verifier_lmv/")
    parser.add_argument("--vis_dir",      default="outputs/vision_assessment/")
    parser.add_argument("--cot_dir",      default="outputs/cot_vision/")
    parser.add_argument("--aug_dir",      default="outputs/desc_augmented_vision/")
    parser.add_argument("--verifier_dir", default="outputs/verifier_vision/",
                        help="Dir with ckpt_judged.csv from verifier_vision.py")
    parser.add_argument("--config",       default="src/config/datasets.yaml")
    parser.add_argument("--manifest",     default="dataset/selected_images.json",
                        help="scripts/agml_hf/download_and_sample.py's manifest — same one "
                             "every other stage (vision_baseline.py, cot_yaml.py, etc.) reads via "
                             "load_hf_samples().")
    parser.add_argument("--images_root",  default="dataset/images")
    parser.add_argument("--test_split",   default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_samples",    type=int, default=50)
    parser.add_argument("--n_datasets",   type=int, default=N_DATASETS_PER_TASK)
    parser.add_argument("--subsample",    nargs="+", default=None,
                        help="Restrict to these dataset(s) only (agml_name, space-separated) "
                             "instead of the usual random n_datasets-per-task draw — for fast "
                             "testing. Same convention as verifier_cot.py's --subsample / "
                             "cot_yaml.py's --swap_datasets; pass the SAME list to cot-compare's "
                             "--filter_datasets so the comparisons are scoped consistently.")
    parser.add_argument("--n_max",        type=int, default=N_MAX_DEFAULT,
                        help="N: candidates per image for PPT to select from (sliced to 1/5/10 at judge time)")
    parser.add_argument("--seed",         type=int, default=SEED)
    parser.add_argument("--batch_size",   type=int, default=256)
    # LMV paper axes (ablation: use distinct --output_dir per config)
    parser.add_argument("--n_evals",      type=int, default=4,
                        help="K: repeated evaluations per comparison (paper ablation: 1,4,8)")
    parser.add_argument("--n_criteria",   type=int, default=3, choices=[1, 3],
                        help="C: criteria count (1=single generic, 3=task-decomposed)")
    parser.add_argument("--ppt_pivots",   type=int, default=DEFAULT_PIVOTS,
                        help="k: PPT pivot count")
    # Control flags
    parser.add_argument("--from_ckpt",   action="store_true",
                        help="Skip phases whose checkpoint CSV already exists")
    parser.add_argument("--score_only",  action="store_true")
    parser.add_argument("--judge_only",  action="store_true")
    parser.add_argument("--plot_only",   action="store_true")
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

    from src.common.dataset_config import load_manifest
    hf_manifest = load_manifest(Path(args.manifest))
    images_root = Path(args.images_root)

    test_split: dict | None = None
    ts_path = Path(args.test_split)
    if ts_path.exists():
        with open(ts_path) as f:
            test_split = json.load(f)
        print(f"Loaded test_split: {ts_path} ({len(test_split)} datasets)")
    else:
        print(f"[warn] test_split not found at {ts_path} — using all images")

    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed,
                               config_path=args.config,
                               subsample=set(args.subsample) if args.subsample else None)
    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")

    print(f"\n=== Config: N={args.n_max}, K={args.n_evals}, C={args.n_criteria}, "
          f"pivots={args.ppt_pivots} (G=20 fixed by scoring scale) ===")

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
                excluded_classes = info.get("excluded_classes", [])
                items           = load_items(ds, task, crop, n_samples=args.n_samples,
                                             manifest=hf_manifest, images_root=images_root,
                                             class_overrides=class_overrides,
                                             excluded_classes=excluded_classes, seed=args.seed,
                                             test_split=test_split)
                all_items.extend(items)
                print(f"  {ds} ({task}): {len(items)} items")
        print(f"Total items: {len(all_items)}")

        print(f"\n=== Phase 1: generate ({len(all_items)} images × {args.n_max} candidates) ===")
        candidates = run_generate(all_items, llm, tokenizer,
                                  n_max=args.n_max, batch_size=args.batch_size,
                                  ckpt_path=ckpt_candidates)

    # ── Phase 2: PPT verify ────────────────────────────────────────────────────
    if skip_score:
        print("\n=== Phase 2: PPT verify [SKIPPED] ===")
        scored = _read_csv(ckpt_scored)
    else:
        print(f"\n=== Phase 2: PPT verify ({len(candidates)} candidates) ===")
        scored = run_verify_ppt(
            candidates, llm, tokenizer,
            n_evals=args.n_evals, n_criteria=args.n_criteria,
            k_pivots=args.ppt_pivots, batch_size=args.batch_size,
            seed=args.seed, ckpt_path=ckpt_scored,
        )

    # ── Phase 3: Argmax + Judge ────────────────────────────────────────────────
    if skip_judge:
        print("\n=== Phase 3: judge [SKIPPED] ===")
        judged = _read_csv(ckpt_judged)
    else:
        print("\n=== Phase 3: argmax + judge ===")
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
    model_safe = args.model.replace("/", "_")
    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    print(f"\n=== Phase 4: plot ({len(image_keys)} unique images) ===")

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        verifier_dir=Path(args.verifier_dir),
        model_safe=model_safe,
    )

    plot_all(judged, existing, output_dir)

    if ckpt_scored.exists():
        plot_reward_spread(ckpt_scored, ckpt_judged, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
