"""
Stage 4 — Verifier-CoT: single-round K-candidate + PPT-tournament verifier
(our contribution), per docs/stage4_verifier.md.

Per image: generate K candidates from a structured, criteria-based
template (vprompts["structured_templates"], filled with the task's
rubric), then run the FULL Probabilistic Pivot Tournament from the paper
(arXiv:2607.05391 Section 3.2 — ring pass, pivot selection, pivot rounds,
via llm_verifier.pivot_tournament, UNMODIFIED, the same library
src/verifier_lmv.py uses) exactly once over that one pool of K
candidates. Every scheduled pair is scored via the PPT paper's own
dual-trajectory logprob mechanism (src/common/ppt_scoring.py — the exact
same scoring the full-tournament baseline uses), aggregated into
Bradley-Terry win-mass to rank the pool. The reported prediction is
simply argmax(win_rate) over that single tournament pass — no incumbent
carried across rounds, no persistent per-answer clusters, no stopping
rule; there is only one round.

This is a deliberate simplification (2026-08-23) from an earlier
multi-round design that carried a persistent per-trajectory net score
(wins - losses) across rounds with a leader-margin gate and a
consecutive-defends lock. On a full 9,104-image run, that multi-round
machinery did not reliably improve on stopping after round 1 once
judge-evaluation noise was controlled for (round 1 judge_f1_macro 0.3556
vs. corrected final 0.3547; genuine mechanism-caused swaps net -8 images
out of 9,104) — a legitimate negative finding, not a bug. Meanwhile
round 1 alone (structured candidate generation + PPT tournament
selection) substantially beats the naive single-shot Lower Bound
baseline (~0.36 vs ~0.18-0.20, roughly 2x) — that comparison is the real
contribution this module now exists to produce. The retired multi-round
code lives in git history; see docs/legacy/verifier_score_belief_
update.md for the historical design writeup.

Checkpoint (one row per image, CKPT_FIELDS-shaped + a small set of
verifier-specific extras — see VERIFIER_EXTRA_FIELDS):
  <output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics.json

Usage:
    uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it

Collate after all models have run:
    uv run python main.py verifier-cot --collate
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict

from PIL import Image as PILImage

from src.common.chat_prompt import render_chat_template
from src.common.dataset_config import (
    DISEASE_TASK_TYPES, active_datasets, crop_display, load_hf_samples,
    load_manifest, parse_class_label,
)
from src.common.io import append_jsonl, load_prompts
from src.common.metrics import (
    contains_match, extract_answer_tag, resolve_class, rouge_scores_per_sample, sample_class_images,
)
from src.common.ppt_scoring import score_prefill_params
from src.common.ppt_orchestration import (
    PairScoreResult, run_ppt_tournament, score_pairs_batch as _shared_score_pairs_batch,
)
from src.common.progress import ProgressBar

try:
    from llm_verifier.pivot_tournament import DEFAULT_PIVOTS
except ImportError:
    DEFAULT_PIVOTS = 2
from src.common.vision_eval import (
    CKPT_FIELDS, append_ckpt_row, archive_ckpt_if_rerun, compute_model_metrics,
    judge_gt_hint, load_ckpt, run_judge_phase, safe_model_name, write_collated,
)
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)

# Confidence-language markers (Paper 2 Stage 2, docs/stage2_verifier_corrections.md)
# — the same phenomenon flagged in the CoT-verifier findings memory ("confidence
# language is anti-correlated with correctness"): logged per pairwise-analysis
# so verifier-score-analysis can test whether it carries over into the PPT
# verifier's own comparative analysis text, not assumed.
_CONFIDENCE_MARKERS_RE = re.compile(
    r"\b(clearly|obviously|definitely|certainly|without doubt|undoubtedly|"
    r"characteristic of|classic(?:ally)?|hallmark|unmistakabl\w*|textbook)\b",
    re.IGNORECASE,
)

_VERIFIER_PROMPTS_PATH = Path(__file__).parent / "config" / "verifier_prompts.yaml"

# Extra columns inserted right after image_name — same convention as
# every other CKPT_FIELDS-based stage's own extras.
VERIFIER_EXTRA_FIELDS = [
    "pool_size", "n_comparisons", "top1_win_rate", "verifier_feedback",
    "prompt_tokens", "completion_tokens", "elapsed_s", "avg_time_per_sample_s",
]
VERIFIER_CKPT_FIELDS = (
    CKPT_FIELDS[: CKPT_FIELDS.index("image_name") + 1]
    + VERIFIER_EXTRA_FIELDS
    + CKPT_FIELDS[CKPT_FIELDS.index("image_name") + 1 :]
)


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Prompt building ─────────────────────────────────────────────────────────

def _sys_msg(prompts: dict) -> list[dict]:
    s = prompts.get("assessment_system", "")
    return [{"role": "system", "content": s}] if s else []


def _cls_question(task_type: str, crop_val, prompts: dict) -> str:
    crop = crop_display(crop_val)
    if task_type in DISEASE_TASK_TYPES:
        inner = prompts["visual_classification_disease"].format(crop=crop)
    elif task_type == "quality_identification":
        inner = prompts["visual_classification_quality"].format(crop=crop)
    else:
        inner = prompts["visual_classification_weed"]
    return prompts["assessment_classification"].format(question=inner)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    m = _REPEAT_RE.search(text)
    return text[:m.start()].strip() if m else text


def run_batch_n(
    llm, tokenizer, images: list, histories: list[list[dict]], params,
    call_stats: dict | None = None, enable_thinking: bool = False,
) -> list[list[dict]]:
    """n completions per prompt. Returns one list of {"text", "prompt_tokens",
    "completion_tokens"} dicts per prompt. `call_stats` (a plain dict shared
    across a whole run — see run_model), if given, is incremented in place:
    n_calls += 1 (this is the single choke point every verifier/generation
    call in this module passes through, so counting invocations here is
    the true, unambiguous "how many separate vLLM calls did this run make"
    — batch size doesn't inflate it), n_prompts += len(prompts).

    enable_thinking: this is the ONLY generation call in this module that
    takes it — the verifier's pairwise-duel analysis (_score_pairs_batch)
    always renders with thinking off, since it's not what --enable_thinking
    is meant to test and turning thinking on for every verifier call too
    would multiply cost across k_candidates x pairwise-comparisons for no
    benefit to the actual question (does the MODEL BEING TESTED reason
    better)."""
    prompts = [{"prompt": render_chat_template(tokenizer, h, enable_thinking=enable_thinking),
                "multi_modal_data": {"image": img}}
               for img, h in zip(images, histories)]
    outputs = llm.generate(prompts, sampling_params=params)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(prompts)
    results = []
    for o in outputs:
        p_tok = len(getattr(o, "prompt_token_ids", None) or [])
        results.append([
            {"text": _clean(c.text), "prompt_tokens": p_tok,
             "completion_tokens": len(getattr(c, "token_ids", None) or [])}
            for c in o.outputs
        ])
    return results


def _log_verifier_call(log_path: Path | None, **fields) -> None:
    """Full prompt/response transparency for every pairwise score call —
    same convention as run_judge_phase's judge_log.jsonl (src/common/
    vision_eval.py). No-op if log_path is None."""
    if log_path is None:
        return
    append_jsonl(log_path, fields)


# ── Dataset grouping (same shape as cot_yaml.py's _build_groups) ───────────

def _load_reference_images(path: Path) -> dict[tuple[str, str], list[str]]:
    """Read an existing single-prediction-per-row checkpoint (e.g.
    vision_baseline.py's ckpt_<model>.csv) and group its image_name column
    by (agml_dataset, class_label), preserving each name's first-seen row
    order — that row order IS that script's own sample_class_images draw,
    so replaying it in order (then capping to --n_samples) selects a true
    PREFIX of that run's own sample, not a fresh independent draw. Used by
    --match_images_from so a smaller verifier-cot run (e.g. 25/class) is
    guaranteed to overlap 100% with an existing larger run (e.g. 50/class)
    of vision_baseline/desc_aug/self_cot/cot_yaml — sample_class_images'
    own np.random.Generator.choice(N, size=n) is NOT nested across
    different n with the same seed (confirmed empirically: only ~64%
    overlap between independent 25- and 50-image draws), so two scripts
    independently sampling different n from the same seed do NOT produce
    a paper-consistent, apples-to-apples image set on their own."""
    import csv as _csv
    out: dict[tuple[str, str], list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            names = out.setdefault(key, [])
            name = r.get("image_name", "")
            if name and name not in names:
                names.append(name)
    return out


def _build_groups(
    args: argparse.Namespace, manifest: dict, dataset_filter: set[str] | None = None,
    reference_images: dict[tuple[str, str], list[str]] | None = None,
):
    """dataset_filter (--subsample) restricts to just these agml_name(s) —
    same convention as cot_yaml.py's --swap_datasets, meant for fast
    testing against a small, deliberate set instead of the whole config.

    reference_images (--match_images_from, see _load_reference_images):
    when given, per-class image selection replays THAT run's own sampled
    image names (capped to --n_samples) instead of drawing independently
    via sample_class_images — falls back to an independent draw with a
    warning for any (dataset, class) the reference has no rows for."""
    groups = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip or not ds.task_type:
            continue
        if dataset_filter is not None and ds.agml_name not in dataset_filter:
            continue
        try:
            samples = load_hf_samples(
                ds.agml_name, manifest, Path(args.images_root), excluded_classes=ds.excluded_classes,
            )
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest")
            continue
        by_label: dict[str, list[dict]] = {}
        for s in samples:
            by_label.setdefault(s["label"], []).append(s)
        for raw_label, items in by_label.items():
            path_by_name = {Path(it["path"]).name: it["path"] for it in items}
            crop_val, clean_label = parse_class_label(raw_label, ds.crop)
            if reference_images is not None:
                ref_names = [n for n in reference_images.get((ds.agml_name, clean_label), []) if n in path_by_name]
                if ref_names:
                    names = ref_names[: args.n_samples]
                else:
                    print(f"  [warn] --match_images_from: no overlapping images for "
                          f"{ds.agml_name}/{clean_label} — falling back to an independent sample")
                    names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            else:
                names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            task_type = ds.effective_task_type(raw_label)
            aliases = ds.effective_search_terms(raw_label)[1:]
            gt_hint = judge_gt_hint(clean_label, aliases)
            groups.append((ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name))
    return groups


def _sample_random_classes_per_task(groups: list, n_per_task: int, seed: int) -> list:
    """--random_classes_per_task's implementation: group `groups` (one per
    (dataset, class_label), from _build_groups) by task_type and randomly
    keep `n_per_task` of them per type — an alternative to --subsample's
    explicit dataset-name list for fast testing, when you'd rather sample
    a random spread of classes across the whole config than hand-pick
    which datasets to test. Seeded by --seed for reproducibility. A
    task_type with fewer than n_per_task groups keeps all of them."""
    rng = random.Random(seed)
    by_task: dict[str, list] = defaultdict(list)
    for g in groups:
        by_task[g[3]].append(g)  # g[3] = task_type
    selected = []
    for task_type in sorted(by_task):
        task_groups = by_task[task_type]
        rng.shuffle(task_groups)
        selected.extend(task_groups[:n_per_task])
    return selected


# ── Single-round tournament ──────────────────────────────────────────────────
# The tournament mechanics (two-pass pairwise scoring + ring-pass/pivot-round
# schedule/aggregation) live in src/common/ppt_orchestration.py — shared,
# byte-for-byte identical logic with detection_verifier.py, extracted
# 2026-08-26 so both callers (and src/ppt_verify/tournament.py) run through
# exactly one implementation. This module only wraps that shared core to add
# its own logging (verifier_log.jsonl's "tournament_pair" call_type/field
# names, unchanged from before the extraction).

def _score_pairs_batch(
    llm, tokenizer, images: list, active: list[int], pools: dict[int, list[str]],
    pair_by_image: dict[int, tuple[int, int]], prompts: dict, vprompts: dict, cls_q: str, rubric: str,
    rng: random.Random, analysis_params, score_params, call_stats: dict | None,
    agml_dataset: str, class_label: str, image_names: list[str],
    log_path: Path | None,
) -> dict[int, tuple[float, float, str, list[int]]]:
    """Thin logging wrapper around ppt_orchestration.score_pairs_batch —
    same call shape/return shape (r_a, r_b, analysis, [prompt_tokens,
    completion_tokens]) this module's own callers already expect, so
    run_group below needed no changes beyond its import."""
    results, owners = _shared_score_pairs_batch(
        llm, tokenizer, images, active, pools, pair_by_image, prompts, vprompts, cls_q, rubric,
        rng, analysis_params, score_params, call_stats,
    )
    result: dict[int, tuple[float, float, str, list[int]]] = {}
    for i in active:
        r, o = results[i], owners[i]
        result[i] = (r.r_a, r.r_b, r.analysis, [r.prompt_tokens, r.completion_tokens])
        # Paper 2 Stage 2 granularity (docs/stage2_verifier_corrections.md):
        # raw letter-probability max/sum breakdown (tests the Step 2
        # max-not-sum hypothesis), trajectory word counts (length-bias
        # hypothesis), and a confidence-language marker count on the
        # free-text analysis (carries the CoT-verifier finding — confident
        # phrasing anti-correlated with correctness — forward to this
        # verifier). All additive/read-only: never fed back into scoring.
        n_confidence_markers = len(_CONFIDENCE_MARKERS_RE.findall(r.analysis))
        _log_verifier_call(
            log_path, agml_dataset=agml_dataset, class_label=class_label,
            image_name=image_names[i], pool_pair=[o.a, o.b],
            call_type="tournament_pair", pool_text_a=pools[i][o.a], pool_text_b=pools[i][o.b],
            slot_swapped=o.swapped, user_prompt=o.prompt_text, analysis=r.analysis,
            score_a=r.r_a, score_b=r.r_b,
            prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
            n_confidence_markers=n_confidence_markers,
            trajectory_a_words=r.words_a, trajectory_b_words=r.words_b,
            letter_probs_a_max=(r.diag_a or {}).get("probs_max"),
            letter_probs_a_sum=(r.diag_a or {}).get("probs_sum"),
            letter_probs_b_max=(r.diag_b or {}).get("probs_max"),
            letter_probs_b_sum=(r.diag_b or {}).get("probs_sum"),
        )
    return result


def run_group(
    llm, tokenizer, images: list, cls_q: str, gen_template: str, rubric: str, prompts: dict, vprompts: dict,
    k_candidates: int, k_pivots: int, gen_temperature: float, rng: random.Random,
    agml_dataset: str, class_label: str, image_names: list[str],
    log_path: Path | None = None, call_stats: dict | None = None,
    seed: int | None = None, enable_thinking: bool = False, gen_max_tokens: int = 180,
) -> list[dict]:
    """Run one image's worth of the whole single-round pipeline for every
    image in one (dataset, class) group, batched across the group: generate
    K candidates from the structured template, then run the existing,
    unmodified PPT tournament (arXiv:2607.05391 Section 3.2, via
    llm_verifier.pivot_tournament's ring_cycle/select_pivots/
    pivot_round_pairs/bradley_terry — the exact same library
    src/verifier_lmv.py uses) exactly once over that pool of K, scoring
    every scheduled pair via src/common/ppt_scoring.py's two-pass prefill
    mechanism. Returns one result dict per image: {"prediction",
    "reasoning", "pool_size", "n_comparisons", "top1_win_rate",
    "verifier_feedback", "prompt_tokens", "completion_tokens", "elapsed_s",
    "avg_time_per_sample_s"}.

    seed: fixes the RNG for every SamplingParams this function creates
    (gen_params' k-candidate sampling, analysis_params' pairwise-duel
    analysis) so two runs over identical images/config are reproducible.
    None (vLLM's own default, unseeded) if not given.

    enable_thinking/gen_max_tokens: --enable_thinking (default off, matches
    every other generation script in this repo — see chat_prompt.py's
    module docstring) turns on the model-being-tested's own extended
    thinking for the k_candidates generation step ONLY — not the
    verifier's own pairwise calls, see run_batch_n's docstring for why.
    gen_max_tokens defaults to 180 (fits a direct fill-in-the-blank answer
    with no thinking); raise it via config when enabling thinking."""
    t_start = time.perf_counter()
    n = len(images)
    active = list(range(n))
    gen_q = gen_template.format(question=cls_q)
    hist = [_sys_msg(prompts) + [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": gen_q},
    ]}] for _ in range(n)]

    gen_params = SamplingParams(
        temperature=gen_temperature, max_tokens=gen_max_tokens,
        repetition_penalty=1.1, n=k_candidates, seed=seed,
    )
    gen_out = run_batch_n(llm, tokenizer, images, hist, gen_params, call_stats, enable_thinking=enable_thinking)
    pools: dict[int, list[str]] = {i: [c["text"] for c in gen_out[i]] for i in active}
    answer_lists: list[list[str]] = []
    for cands in gen_out:
        answers = [extract_answer_tag(c["text"])[0] for c in cands]
        answer_lists.append(answers)

    prompt_tokens = {i: gen_out[i][0]["prompt_tokens"] for i in active}
    completion_tokens = {i: sum(c["completion_tokens"] for c in gen_out[i]) for i in active}

    for i in active:
        for cand_idx, c in enumerate(gen_out[i]):
            _log_verifier_call(
                log_path, agml_dataset=agml_dataset, class_label=class_label,
                image_name=image_names[i], call_type="candidate",
                candidate_index=cand_idx, raw_text=c["text"],
                extracted_answer=answer_lists[i][cand_idx],
                prompt_tokens=c["prompt_tokens"], completion_tokens=c["completion_tokens"],
            )

    pool_size = k_candidates
    analysis_params = SamplingParams(temperature=1.0, max_tokens=150, seed=seed)
    score_params = score_prefill_params()

    def _score_batch_fn(pair_by_image: dict[int, tuple[int, int]]):
        result = _score_pairs_batch(
            llm, tokenizer, images, active, pools, pair_by_image, prompts, vprompts, cls_q, rubric, rng,
            analysis_params, score_params, call_stats,
            agml_dataset, class_label, image_names, log_path,
        )
        return {
            i: PairScoreResult(r_a=r_a, r_b=r_b, analysis=analysis,
                                prompt_tokens=tok[0], completion_tokens=tok[1])
            for i, (r_a, r_b, analysis, tok) in result.items()
        }

    # ── The tournament itself (ring pass -> pivot selection -> pivot
    # rounds -> Bradley-Terry aggregation -> argmax) — shared with
    # detection_verifier.py and src/ppt_verify, see
    # src/common/ppt_orchestration.py's module docstring.
    tournament = run_ppt_tournament(pool_size, active, k_pivots, rng, _score_batch_fn)

    elapsed = time.perf_counter() - t_start
    avg_time_per_sample_s = elapsed / n if n else 0.0

    results = []
    for i in active:
        tr = tournament[i]
        prediction = pools[i][tr.top1]
        final_prompt_tokens = prompt_tokens[i] + tr.prompt_tokens_delta
        final_completion_tokens = completion_tokens[i] + tr.completion_tokens_delta
        results.append({
            "prediction": prediction,
            "reasoning": (f"pool={pool_size}, comparisons={tr.n_comparisons}, "
                          f"top1_win_rate={tr.win_rate[tr.top1]:.3f}"),
            "pool_size": pool_size,
            "n_comparisons": tr.n_comparisons,
            "top1_win_rate": round(tr.win_rate[tr.top1], 4),
            "verifier_feedback": tr.first_analysis.get(tr.top1, ""),
            "prompt_tokens": final_prompt_tokens,
            "completion_tokens": final_completion_tokens,
            "elapsed_s": round(elapsed, 3),
            "avg_time_per_sample_s": round(avg_time_per_sample_s, 3),
        })
    return results


def _result_row(result: dict, ds, clean_label: str, crop_val, task_type: str, name: str, gt_hint: str) -> dict:
    resolved = resolve_class(result["prediction"], ds.clean_classes)
    correct = resolved is not None and resolved.lower() == clean_label.lower()
    c_match = contains_match(resolved or result["prediction"], clean_label)
    scores = rouge_scores_per_sample(result["prediction"], clean_label, include_rouge=False)
    return {
        "agml_dataset": ds.agml_name, "class_label": clean_label,
        "task_type": task_type, "crop": crop_display(crop_val),
        "image_name": name,
        "pool_size": result["pool_size"], "n_comparisons": result["n_comparisons"],
        "top1_win_rate": result["top1_win_rate"], "verifier_feedback": result["verifier_feedback"],
        "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
        "elapsed_s": result["elapsed_s"], "avg_time_per_sample_s": result["avg_time_per_sample_s"],
        "prediction": result["prediction"], "resolved_class": resolved or "",
        "ground_truth": clean_label, "correct": int(correct), "contains_match": int(c_match),
        "resolvable": int(resolved is not None), "reasoning": result["reasoning"],
        "judge_gt_hint": gt_hint, **scores,
    }


# ── Main run ─────────────────────────────────────────────────────────────────

def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    vprompts = _load_yaml(Path(args.verifier_prompts))
    manifest = load_manifest(Path(args.manifest))

    # Flat <output_dir>/<model>/ layout so this stays compatible with the
    # rest of the repo's load_all_rows()-style loaders.
    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    # verifier_log.jsonl is this script's own raw per-call log (every
    # tournament_pair/candidate prompt+response) — must be archived
    # alongside the checkpoint on --rerun too, or old and new run entries
    # silently concatenate in one ever-growing file with no way to tell
    # them apart.
    archive_ckpt_if_rerun(ckpt_path, args.rerun, extra_logs=("judge_log.jsonl", "verifier_log.jsonl"))
    rows, seen = load_ckpt(ckpt_path, fields=VERIFIER_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} row(s) already scored in {ckpt_path.name}")

    dataset_filter = set(args.subsample) if args.subsample else None
    match_images_from = args.match_images_from
    if match_images_from is None:
        # Auto-detected default: vision_baseline.py's own checkpoint for
        # THIS model — every apples-to-apples comparison in verifier-
        # summary is against vision_baseline (Lower Bound) anyway, so
        # defaulting to its image set means a bare verifier-cot invocation
        # (no --match_images_from typed) still gets a guaranteed-overlap
        # sample rather than sample_class_images' own independent draw.
        # Falls back to that independent draw (with a note) if no
        # vision_baseline run exists yet for this model.
        _candidate = Path("outputs_final/vision_baseline") / safe_model_name(model_id) / f"ckpt_{safe_model_name(model_id)}.csv"
        if _candidate.exists():
            match_images_from = str(_candidate)
            print(f"  [match_images_from] auto-detected {match_images_from} (pass --match_images_from "
                  f"to override, or an empty/nonexistent path to force an independent sample)")
        else:
            print(f"  [match_images_from] no vision_baseline checkpoint found at {_candidate} — "
                  f"sampling independently via sample_class_images instead")

    reference_images = None
    if match_images_from and Path(match_images_from).exists():
        reference_images = _load_reference_images(Path(match_images_from))
        print(f"  [match_images_from] {sum(len(v) for v in reference_images.values())} image name(s) "
              f"loaded across {len(reference_images)} (dataset, class) group(s) from {match_images_from}")
    all_groups = list(_build_groups(args, manifest, dataset_filter=dataset_filter, reference_images=reference_images))
    if args.random_classes_per_task:
        all_groups = _sample_random_classes_per_task(all_groups, args.random_classes_per_task, args.seed)
        by_task_counts = Counter(g[3] for g in all_groups)
        print(f"  [random_classes_per_task={args.random_classes_per_task}, seed={args.seed}] "
              f"{len(all_groups)} (dataset, class) group(s) selected: {dict(by_task_counts)}")

    if args.k_candidates > 1 and args.temperature <= 0:
        print(f"  [warn] k_candidates={args.k_candidates} with temperature={args.temperature}: "
              f"greedy decoding returns identical samples — raise --temperature above 0 or the "
              f"extra candidates are wasted compute")

    groups = []
    for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in all_groups:
        pending = [n for n in names if (ds.agml_name, clean_label, n) not in seen]
        if not pending:
            continue
        groups.append((ds, clean_label, crop_val, task_type, gt_hint, pending, path_by_name))
    if dataset_filter is not None:
        missing = dataset_filter - {ds.agml_name for ds, *_ in groups}
        if missing:
            print(f"  [warn] --subsample not found (skip/not in manifest/already complete?): {sorted(missing)}")

    n_rows_before = len(rows)
    call_stats: dict = {"n_calls": 0, "n_prompts": 0}
    log_path = model_dir / "verifier_log.jsonl"
    elapsed = 0.0
    # One rng, seeded once, threaded through every group in a fixed
    # processing order — reproducible positional-bias randomization
    # (the pairwise duel's A/B slot) across a full run.
    rng = random.Random(args.seed)

    if not groups:
        print("  Nothing to do — all samples already checkpointed.")
    else:
        n_pending = sum(len(g[5]) for g in groups)
        print(f"  {len(groups)} (dataset, class) group(s), {n_pending} pending sample(s)")

        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            caller_tag="verifier_cot",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Classes")
        for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in groups:
            pbar.desc = f"Classes ({ds.agml_name} / {clean_label})"
            images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]
            cls_q = _cls_question(task_type, crop_val, prompts)
            rubric_key = vprompts["task_type_to_rubric"].get(task_type, "disease_pest")
            rubric = vprompts["rubrics"][rubric_key]
            gen_template = vprompts["structured_templates"][rubric_key]

            results = run_group(
                llm, tokenizer, images, cls_q, gen_template, rubric, prompts, vprompts,
                k_candidates=args.k_candidates, k_pivots=args.k_pivots,
                gen_temperature=args.temperature, rng=rng,
                agml_dataset=ds.agml_name, class_label=clean_label, image_names=names,
                log_path=log_path, call_stats=call_stats,
                seed=args.seed, enable_thinking=args.enable_thinking, gen_max_tokens=args.gen_max_tokens,
            )

            for name, result in zip(names, results):
                row = _result_row(result, ds, clean_label, crop_val, task_type, name, gt_hint)
                append_ckpt_row(ckpt_path, row, fields=VERIFIER_CKPT_FIELDS)
                rows.append(row)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer

    rows = run_judge_phase(rows, ckpt_path, args, prompts, fields=VERIFIER_CKPT_FIELDS)

    metrics = compute_model_metrics(rows)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Overall: f1_macro={metrics['overall'].get('f1_macro')} "
          f"judge_accuracy={metrics['overall'].get('judge_accuracy')} (n={metrics['overall'].get('n_total')})")

    # ── Compute summary — "this_run" is just the new work this invocation
    #    did (call_stats/elapsed only ever count calls actually made here,
    #    never re-derived from resumed/skipped rows); "cumulative_all_rows"
    #    sums prompt/completion tokens across every row in the checkpoint,
    #    old + new, for the full lifetime picture of a resumed run.
    compute_summary = {
        "this_run": {
            "n_llm_calls": call_stats.get("n_calls", 0),
            "n_prompts_sent": call_stats.get("n_prompts", 0),
            "wall_clock_s": round(elapsed, 1),
            "n_new_rows": len(rows) - n_rows_before,
        },
        "cumulative_all_rows": {
            "n_rows": len(rows),
            "n_images": len({(r["agml_dataset"], r["image_name"]) for r in rows}),
            "total_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
            "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        },
    }
    (model_dir / "compute_summary.json").write_text(json.dumps(compute_summary, indent=2))
    print(f"  Compute: {compute_summary['this_run']['n_llm_calls']} LLM call(s) this run "
          f"({compute_summary['this_run']['n_prompts_sent']} prompts), "
          f"{compute_summary['cumulative_all_rows']['total_prompt_tokens']}+"
          f"{compute_summary['cumulative_all_rows']['total_completion_tokens']} prompt+completion "
          f"tokens across all rows — full verifier call log: {log_path.name}")


def main(args: argparse.Namespace) -> None:
    if args.collate:
        collated: dict[str, dict] = {}
        base = Path(args.output_dir)
        if base.exists():
            for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                ckpt_path = model_dir / f"ckpt_{model_dir.name}.csv"
                if not ckpt_path.exists():
                    continue
                rows, _ = load_ckpt(ckpt_path, fields=VERIFIER_CKPT_FIELDS)
                if rows:
                    collated[model_dir.name] = compute_model_metrics(rows)
        if not collated:
            print(f"No per-model output folders with checkpoints found under {args.output_dir}")
            return
        write_collated(Path(args.output_dir), collated)
        return

    if not args.model:
        print("ERROR: --model is required (or pass --collate to combine finished runs).")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--verifier_config", default="src/config/verifier_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--verifier_config", default=_pre_args.verifier_config,
        help="YAML file of defaults (k_candidates, k_pivots, etc.) — edit it directly for "
             "routine changes instead of passing flags every time. Any flag also passed on the "
             "command line overrides this file.",
    )
    parser.add_argument("--model", default=None, help="Single model to run. Not needed with --collate.")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--verifier_prompts", default=str(_VERIFIER_PROMPTS_PATH))
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--output_dir", default="outputs_final/verifier_cot/")
    parser.add_argument("--n_samples", type=int, default=10, help="Images/class to evaluate.")
    parser.add_argument("--match_images_from", default=None,
                         help="Path to an existing single-prediction-per-row checkpoint (e.g. "
                              "outputs_final/vision_baseline/<model>/ckpt_<model>.csv) — per-class "
                              "image selection replays THAT run's own sampled image names (capped "
                              "to --n_samples) instead of an independent sample_class_images draw, "
                              "so a smaller run here is guaranteed 100%% image overlap with a larger "
                              "existing run for paper-consistent comparisons. sample_class_images' "
                              "own seeded draw is NOT nested across different --n_samples values "
                              "(confirmed: only ~64%% overlap between independent 25- and 50-image "
                              "draws from the same seed), so this is the only way to guarantee "
                              "overlap without re-running the larger baseline.")
    parser.add_argument("--subsample", nargs="+", default=None,
                         help="Restrict to these dataset(s) only (agml_name, space-separated) "
                              "instead of the whole active config — for fast testing. Same "
                              "convention as cot_yaml.py's --swap_datasets; e.g. feed it "
                              "cot-compare's own cot_yaml_swap_datasets.txt (top gain/loss "
                              "datasets) via $(cat outputs_final/cot_compare/<model>/"
                              "cot_yaml_swap_datasets.txt). Pass the SAME list to cot-compare's "
                              "--filter_datasets so the lower/upper-bound and self-cot/cot-yaml "
                              "comparisons are scoped to the same datasets, not the full config.")
    parser.add_argument("--random_classes_per_task", type=int, default=None,
                         help="Instead of (or combined with, if --subsample also given — it "
                              "pre-filters, this samples from what's left) hand-picking datasets, "
                              "randomly keep this many (dataset, class) groups PER task_type "
                              "(disease/pest_damage/species_identification/quality_identification) "
                              "for fast testing across a randomized spread instead of chosen "
                              "datasets. Seeded by --seed for reproducibility. Pass the same value "
                              "+ --seed to reproduce the exact same selection, or read the printed "
                              "[random_classes_per_task=...] line / the resulting ckpt's own "
                              "agml_dataset column for cot-compare's --filter_datasets.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seeds the k-candidate generation and the pairwise duel's A/B slot "
                              "randomization / analysis sampling, for reproducibility.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--k_candidates", type=int, default=3,
                         help="Candidates generated per image (K) via temperature sampling — the "
                              "single tournament pool size.")
    parser.add_argument("--k_pivots", type=int, default=DEFAULT_PIVOTS,
                         help="k — pivot-set size for the full PPT tournament (arXiv:2607.05391 "
                              "Section 3.2, via llm_verifier.pivot_tournament) that ranks the pool.")
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Sampling temperature for the K candidates. Needs to be > 0 "
                              "for --k_candidates > 1 to actually produce differing candidates — "
                              "greedy decoding (temperature=0) returns identical copies.")
    parser.add_argument("--enable_thinking", action="store_true",
                         help="Turn on the model-being-tested's own extended thinking "
                              "(<think>...</think>) for the k_candidates generation step only — every "
                              "other generation script in this repo disables it by default (see "
                              "src/common/chat_prompt.py's module docstring: it compounds badly across "
                              "multi-turn pipelines). Off by default to match that convention. Does NOT "
                              "affect the verifier's own pairwise-duel calls — see run_batch_n's "
                              "docstring. Raise --gen_max_tokens when using this, or thinking traces "
                              "will routinely get truncated before the model reaches its actual answer.")
    parser.add_argument("--gen_max_tokens", type=int, default=180,
                         help="max_tokens for the k_candidates generation step. 180 fits a direct "
                              "fill-in-the-blank answer with no thinking; raise substantially (e.g. "
                              "600-1000) when --enable_thinking is set.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it",
                         help="Fixed judge model used for every baseline/model being evaluated.")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--collate", action="store_true",
                         help="Skip inference — combine every <output_dir>/<model>/ folder into "
                              "collated_summary.csv / collated_metrics.json.")

    if _pre_args.verifier_config and Path(_pre_args.verifier_config).exists():
        _cfg = _load_yaml(Path(_pre_args.verifier_config)) or {}
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests
        if _unknown:
            print(f"[warn] {_pre_args.verifier_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
