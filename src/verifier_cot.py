"""
Stage 4 — Verifier-CoT: round-based belief tracking (our contribution),
per docs/verifier_score_belief_update.md.

One evolving thread per image (see the doc's Section 13 + flow chart for
the full derivation). The FULL Probabilistic Pivot Tournament from the
paper (arXiv:2607.05391 Section 3.2 — ring pass, pivot selection, pivot
rounds, via llm_verifier.pivot_tournament, UNMODIFIED, the same library
src/verifier_lmv.py uses) runs fresh each round over this round's
POOL = [the incumbent from the previous round, if any] + the K new
candidates — the incumbent participates AS A POOL MEMBER, not as a
privileged always-compared anchor. Every scheduled pair is scored via the
PPT paper's own dual-trajectory logprob mechanism (src/common/
ppt_scoring.py — the exact same scoring the full-tournament baseline
uses), aggregated into Bradley-Terry win-mass to rank the pool. Our
contribution is ENTIRELY the stopping rule layered on top of this
unmodified tournament — nothing about PPT itself is changed. Exactly ONE
persistent-belief update happens per round (verifier scores are
ephemeral/per-comparison; the belief is persistent): the round's top-1
(from the tournament's own win-rate ranking — defined for every pool
member, since ring-pass alone already compares each of them at least
once) becomes the winner against the incumbent, or, if the incumbent IS
top-1 (survived), against the strongest actual challenger instead.
pairwise_update() only ever needs the winner/loser IDENTITY, never a raw
score, so this needs no specific pair to have been directly scored under
PPT's O(Nk) budget. That single win/loss folds into the running belief
(mu, sigma^2) via Bradley-Terry moment matching (Sec. 5). Round 1 has no
incumbent, so it uses (top-1, runner-up) the same way — no special-cased
seeding step. Identity for persistence is the ANSWER, not the pool slot:
every candidate's free generation also carries an open-ended
<answer>...</answer> tag (extract_answer_tag — no closed-set vocabulary),
and when a NEW candidate outranks the incumbent's own slot, a lightweight
text-only judge (_judge_same_answer — just the two extracted answers, no
image) decides whether it's a genuine answer change (belief resets) or a
reworded restatement of the same conclusion (belief carries forward onto
the new text). `incumbent_survived` tracks whether the ANSWER (not the
literal text) won again this round. Top-1 only for now (a single
incumbent carried forward, not a pool of k) — see
_process_round_belief's docstring.

A single round (--max_round 1) IS the argmax-selection baseline: with no
incumbent and nothing to persist, the round's own tournament win-rate
argmax is the entire result — no separate scoring mechanism or code path
needed (an earlier, separate --baseline argmax mode — independent 1-20
scoring per candidate, no pairwise comparison — was tried and removed;
--max_round 1 through this same pipeline supersedes it).

Stopping: checked from round 3, hard cap at round 10 (config: min_round/
max_round) — Phi((mu-mu_1)/sqrt(sigma^2+sigma_1^2+2*beta^2)) >= 0.85 stops
early ("high confidence"); otherwise runs to the cap and reports whichever
confidence bin (high/medium/low) it lands in.

Checkpoint (one row per (image, round), STEP_CKPT_FIELDS-shaped like
cot_yaml.py's --per_step/--swap_datasets so cot_yaml_step_report.py-style
per-round F1 works the same way via compute_step_metrics — see the doc's
"What gets stored" section):
  <output_dir>/<safe_model_name>/ckpt_belief_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics_belief.json  (final-round rows only)
  <output_dir>/<safe_model_name>/belief_by_round_belief.png

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

from src.common.belief_update import (
    Belief, confidence_bin, evidence_scaled_beta, pairwise_update, should_stop, stop_probability,
)
from src.common.chat_prompt import render_chat_template
from src.common.dataset_config import (
    DISEASE_TASK_TYPES, active_datasets, crop_display, load_hf_samples,
    load_manifest, parse_class_label,
)
from src.common.io import append_jsonl, load_prompts
from src.common.metrics import contains_match, resolve_class, rouge_scores_per_sample, sample_class_images
from src.common.ppt_scoring import (
    analysis_before_tags, build_score_prompt, extract_tag_score,
    pairwise_scoring_prompt, score_prefill_params,
)
from src.common.progress import ProgressBar

try:
    from llm_verifier.pivot_tournament import (
        DEFAULT_PIVOTS, bradley_terry, pivot_round_pairs, ring_cycle, select_pivots,
    )
except ImportError:
    DEFAULT_PIVOTS = 2
    bradley_terry = pivot_round_pairs = ring_cycle = select_pivots = None  # type: ignore[assignment]
from src.common.vision_eval import (
    CKPT_FIELDS, append_ckpt_row, archive_ckpt_if_rerun, compute_model_metrics,
    judge_gt_hint, load_ckpt, run_judge_phase, safe_model_name, write_collated,
)
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
    _SAME_ANSWER_PARAMS = SamplingParams(temperature=0.0, max_tokens=40)
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    _SAME_ANSWER_PARAMS = None

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)

_VERIFIER_PROMPTS_PATH = Path(__file__).parent / "config" / "verifier_prompts.yaml"

# Extra columns inserted right after image_name — same convention as
# cot_yaml.py's STEP_CKPT_FIELDS (round-level rows, one image contributes
# many rows here instead of CKPT_FIELDS's one row per image).
VERIFIER_EXTRA_FIELDS = [
    "round", "mu", "sigma2", "stop_confidence", "confidence_bin",
    "pool_size", "n_comparisons", "top1_win_rate", "incumbent_survived",
    "verifier_answer", "verifier_feedback", "pairwise_winner", "is_final_round",
    "prompt_tokens", "completion_tokens", "elapsed_s",
]
VERIFIER_STEP_FIELDS = (
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


def extract_answer_tag(text: str) -> tuple[str, bool]:
    """Pull the free-form <answer>...</answer> content out of a round's
    candidate text — open-ended, no closed-set vocabulary (real-world
    answers aren't a fixed label set; see docs/verifier_score_belief_
    update.md's class-identity note). Falls back to the full cleaned text
    if the tag is missing, so every candidate stays comparable rather than
    empty; `found` records which case happened, so the tag's own hit rate
    is directly queryable from verifier_log_<baseline>.jsonl's "candidate"
    records for quality-checking the mechanism over time."""
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip(), True
    return text.strip(), False


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
    takes it — the verifier's pairwise-duel analysis and same-answer check
    (_score_pairs_batch/_judge_same_answer) always render with thinking
    off, since they're not what --enable_thinking is meant to test and
    turning thinking on for every verifier call too would multiply cost
    across k_candidates x max_round x pairwise-comparisons for no benefit
    to the actual question (does the MODEL BEING TESTED reason better)."""
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
    """Full prompt/response transparency for every score/pairwise call —
    same convention as run_judge_phase's judge_log.jsonl (src/common/
    vision_eval.py), just for this module's own verifier calls, which
    previously had no such record at all (only the parsed result made it
    into the checkpoint CSV, the raw prompt/response was discarded).
    No-op if log_path is None."""
    if log_path is None:
        return
    append_jsonl(log_path, fields)


# ── Verifier response parsing ───────────────────────────────────────────────

def _parse_same(text: str) -> bool:
    """Parse the same_answer_prompt's {"same": true/false} verdict.
    Defaults to False (treat as a genuine answer change) on any parse
    failure — the conservative choice, since defaulting True would let a
    malformed judge response silently carry a persistent belief forward
    onto what might be an unrelated answer, while defaulting False only
    costs an unnecessary belief reset."""
    text = _THINK_RE.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            return bool(data.get("same", False))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m = re.search(r'"same"\s*:\s*(true|false)', text, re.IGNORECASE)
    return m.group(1).lower() == "true" if m else False


# ── Dataset grouping (same shape as cot_yaml.py's _build_groups) ───────────

def _build_groups(args: argparse.Namespace, manifest: dict, dataset_filter: set[str] | None = None):
    """dataset_filter (--subsample) restricts to just these agml_name(s) —
    same convention as cot_yaml.py's --swap_datasets, meant for fast
    testing against a small, deliberate set (e.g. cot-compare's own
    top gain/loss cot_yaml_swap_datasets.txt) instead of the whole config."""
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
            names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            crop_val, clean_label = parse_class_label(raw_label, ds.crop)
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


def _rounds_seen_complete(rows: list[dict]) -> set:
    return {(r["agml_dataset"], r["class_label"], r["image_name"])
            for r in rows if str(r.get("is_final_round")) == "1"}


# ── Round loop ───────────────────────────────────────────────────────────────

def _gen_histories_for_round(
    base_hist: list[list[dict]], state: list[dict], active: list[int], round_idx: int, gen_q: str,
) -> list[list[dict]]:
    """History each active image's K candidates get generated from this
    round — round 1 asks gen_q (the structured template) cold; round
    k>=2 replays EVERY prior round's winning trajectory in order — not
    just the immediately preceding one — as alternating assistant/user
    turns on top of base_hist, each followed by a plain reask of gen_q
    (no feedback text — see the reask assignment below for why that was
    dropped). state[i]["history"] holds the (winner, feedback) list,
    appended once per round in run_group_rounds right after
    _process_round_belief finalizes that round's C_best/feedback (see the
    docstring there) — feedback is still carried in that list and still
    logged to the CSV, just not injected into the prompt here anymore."""
    hists = []
    for i in active:
        if round_idx == 1:
            hists.append(base_hist[i])
            continue
        hist = list(base_hist[i])
        for winner_text, feedback_text in state[i]["history"]:
            # Feedback text dropped entirely (was: neutral "for context,
            # here is the verifier's analysis..." framing — see git history
            # "no improved in feedback"). That neutral framing fixed the
            # worst of the drift (regression dropped from 26/160 to 9/160
            # at 6 rounds) but a real 10-round run still showed net
            # regression (23 vs. 11 recovered) — the hypothesis being
            # tested now is that even NEUTRAL analysis text, replayed
            # cumulatively every round, still reads as "something to react
            # to" and nudges the model to keep commenting/adjusting. This
            # version reasks with NOTHING but the winning trajectories
            # themselves in history plus the plain original question — no
            # feedback content at all — to isolate whether it's the
            # feedback TEXT itself (not just its old "improve" framing)
            # driving residual drift. feedback_text is still unpacked from
            # state[i]["history"] (still logged to verifier_feedback in the
            # CSV — useful for analysis) but no longer makes it into the
            # prompt.
            reask = gen_q
            hist.append({"role": "assistant", "content": winner_text})
            hist.append({"role": "user", "content": reask})
        hists.append(hist)
    return hists


def _score_pairs_batch(
    llm, tokenizer, images: list, active: list[int], pools: dict[int, list[str]],
    pair_by_image: dict[int, tuple[int, int]], prompts: dict, vprompts: dict, cls_q: str, rubric: str,
    rng: random.Random, analysis_params, score_params, call_stats: dict | None,
    agml_dataset: str, class_label: str, image_names: list[str],
    round_idx: int, log_path: Path | None,
) -> dict[int, tuple[float, float, str, list[int]]]:
    """Score one pool-index pair PER IMAGE via src/common/ppt_scoring.py's
    two-pass prefill mechanism, batched across every active image in one
    set of vLLM calls — pairs may differ across images (ring-pass cycles
    and pivot selections are randomized/ranked per image), only the COUNT
    of pairs per round is uniform (guaranteed by ring_cycle/
    pivot_round_pairs depending only on pool size and k, not on which
    specific indices end up as pivots). Positional slot (A/B) randomized
    per (image, pair) — a coin flip via `rng` — as an extra safeguard on
    top of the paper's own ring-pass cancellation. Returns {i: (r_a, r_b,
    analysis, [prompt_tokens, completion_tokens])} — scores already
    unswapped back to (pool[a], pool[b]) order."""
    base_prompts, owners = [], []
    for i in active:
        a, b = pair_by_image[i]
        text_a, text_b = pools[i][a], pools[i][b]
        swapped = rng.random() < 0.5
        traj_a, traj_b = (text_b, text_a) if swapped else (text_a, text_b)
        pw_text = pairwise_scoring_prompt(rubric=rubric, question=cls_q, trajectory_a=traj_a, trajectory_b=traj_b,
                                           template=vprompts["pairwise_prompt"])
        msgs = _sys_msg(prompts) + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": pw_text},
        ]}]
        base_prompts.append({
            "prompt": render_chat_template(tokenizer, msgs),
            "multi_modal_data": {"image": images[i]},
        })
        owners.append((i, a, b, swapped, pw_text))

    analysis_outputs = llm.generate(base_prompts, sampling_params=analysis_params)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(base_prompts)
    analyses = [analysis_before_tags(out.outputs[0].text) for out in analysis_outputs]
    pass_tokens = [
        [len(getattr(out, "prompt_token_ids", None) or []),
         len(getattr(out.outputs[0], "token_ids", None) or [])]
        for out in analysis_outputs
    ]

    scores_a: list[float | None] = [None] * len(owners)
    scores_b: list[float | None] = [None] * len(owners)
    for tag, scores in (("<score_A>", scores_a), ("<score_B>", scores_b)):
        score_prompts = [build_score_prompt(bp, an, tag) for bp, an in zip(base_prompts, analyses)]
        score_outputs = llm.generate(score_prompts, sampling_params=score_params)
        if call_stats is not None:
            call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
            call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(score_prompts)
        for idx, (analysis, out) in enumerate(zip(analyses, score_outputs)):
            comp = out.outputs[0]
            scores[idx] = extract_tag_score(analysis, tag, comp.text, comp.logprobs or [])
            pass_tokens[idx][0] += len(getattr(out, "prompt_token_ids", None) or [])
            pass_tokens[idx][1] += len(getattr(comp, "token_ids", None) or [])

    result: dict[int, tuple[float, float, str, list[int]]] = {}
    for idx, (i, a, b, swapped, pw_text) in enumerate(owners):
        r_swapped_a, r_swapped_b = scores_a[idx], scores_b[idx]
        r_a, r_b = (r_swapped_b, r_swapped_a) if swapped else (r_swapped_a, r_swapped_b)
        result[i] = (r_a, r_b, analyses[idx], pass_tokens[idx])
        _log_verifier_call(
            log_path, agml_dataset=agml_dataset, class_label=class_label,
            image_name=image_names[i], round=round_idx, pool_pair=[a, b],
            call_type="tournament_pair", pool_text_a=pools[i][a], pool_text_b=pools[i][b],
            slot_swapped=swapped, user_prompt=pw_text, analysis=analyses[idx],
            score_a=r_a, score_b=r_b,
            prompt_tokens=pass_tokens[idx][0], completion_tokens=pass_tokens[idx][1],
        )
    return result


def _judge_same_answer(
    llm, tokenizer, cls_q: str, vprompts: dict, pairs: dict[int, tuple[str, str]],
    call_stats: dict | None, agml_dataset: str, class_label: str, image_names: list[str],
    round_idx: int, log_path: Path | None,
) -> dict[int, tuple[bool, int, int]]:
    """Text-only equivalence check between two extracted <answer> tags —
    deliberately no image and no full trajectory (the tournament has
    already judged trajectory quality; this call only asks "is this the
    same underlying answer as before"). `pairs` = {image_idx: (answer_a,
    answer_b)} for images where round_idx>=2 and this round's tournament
    top-1 is a NEW candidate, not the incumbent's own pool slot — the only
    case this check is needed (the incumbent's own slot winning again, or
    round 1 with no incumbent yet, are both trivial and never call this).
    Returns {image_idx: (same, prompt_tokens, completion_tokens)}."""
    if not pairs:
        return {}
    items = list(pairs.items())
    batch_prompts, prompt_texts = [], []
    for _, (answer_a, answer_b) in items:
        text = vprompts["same_answer_prompt"].format(
            question=cls_q,
            answer_a=(answer_a or "(no answer extracted)")[:200],
            answer_b=(answer_b or "(no answer extracted)")[:200],
        )
        prompt_texts.append(text)
        batch_prompts.append({"prompt": render_chat_template(tokenizer, [{"role": "user", "content": text}])})

    outputs = llm.generate(batch_prompts, sampling_params=_SAME_ANSWER_PARAMS)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(batch_prompts)

    result: dict[int, tuple[bool, int, int]] = {}
    for (i, (answer_a, answer_b)), text, out in zip(items, prompt_texts, outputs):
        comp = out.outputs[0]
        same = _parse_same(comp.text)
        p_tok = len(getattr(out, "prompt_token_ids", None) or [])
        c_tok = len(getattr(comp, "token_ids", None) or [])
        result[i] = (same, p_tok, c_tok)
        _log_verifier_call(
            log_path, agml_dataset=agml_dataset, class_label=class_label,
            image_name=image_names[i], round=round_idx, call_type="same_answer",
            answer_a=answer_a, answer_b=answer_b, user_prompt=text,
            raw_response=comp.text, parsed_same=same,
            prompt_tokens=p_tok, completion_tokens=c_tok,
        )
    return result


def _process_round_belief(
    llm, tokenizer, images: list, active: list[int], candidate_lists: list[list[str]],
    answer_lists: list[list[str]], state: list[dict], prompts: dict, vprompts: dict, cls_q: str, rubric: str,
    k_candidates: int, round_idx: int, beta: float, k_pivots: int, rng: random.Random,
    agml_dataset: str, class_label: str, image_names: list[str],
    log_path: Path | None, call_stats: dict | None,
    floor_threshold: float = 0.5, margin_ref: float = 0.3, max_beta: float = 4.0,
    seed: int | None = None,
) -> None:
    """The full Probabilistic Pivot Tournament (arXiv:2607.05391 Section
    3.2), via llm_verifier.pivot_tournament's own ring_cycle/select_pivots/
    pivot_round_pairs/bradley_terry — the exact same library
    src/verifier_lmv.py uses, UNMODIFIED — over this round's pool:
    [incumbent from the previous round, if any] + K new candidates, the
    incumbent a full pool member rather than a privileged
    always-compared anchor. Every pair scored via
    src/common/ppt_scoring.py's two-pass prefill mechanism (the same
    dual-trajectory logprob-extracted scoring the full-tournament baseline
    uses), aggregated into Bradley-Terry win-mass to rank the pool.

    Our contribution is ENTIRELY the stopping rule layered on top of this
    unmodified tournament, not a change to PPT itself (Section 10/11 of
    the design doc). So the only thing added here is: after the
    tournament ranks the pool (top1 = argmax win_rate — defined for every
    pool member, since ring-pass alone already compares each of them at
    least once), decide the round's winner/loser for the PERSISTENT
    belief update. pairwise_update() never consumes a raw score, only the
    winner/loser IDENTITY — so this doesn't require the specific
    (top1, incumbent) pair to have been directly scored under PPT's
    budget-efficient schedule; top1's identity from the tournament's own
    aggregate output is sufficient.

    Persistence is keyed off the ANSWER, not the pool slot. Every
    candidate's free-generated text also carries an open-ended
    <answer>...</answer> tag (extract_answer_tag — no closed-set
    vocabulary, real answers aren't a fixed label set); when this round's
    tournament top-1 is the incumbent's own pool slot, "same answer" is
    trivial. When top-1 is a NEW candidate, a lightweight text-only judge
    (_judge_same_answer — just the two extracted answer strings, no image,
    no full trajectory) decides whether it's a genuine answer change or
    just a reworded restatement of the same conclusion — only in the
    former case does the persistent belief reset to a fresh prior; in the
    latter it carries forward onto the new (better-worded) winning text.
    This avoids the belief getting wiped out by cosmetic rephrasing that
    doesn't actually change the answer.

    Mutates state[i] in place: C_best, answer, belief, feedback,
    pairwise_winner, pool_size, n_comparisons, top1_win_rate,
    incumbent_survived, transcript, round_prompt_tokens,
    round_completion_tokens.

    seed: fixes analysis_params' (temperature=1.0) sampling RNG so the
    pairwise duel's free-text analysis is reproducible run-to-run for
    identical inputs — without it, two runs with identical images/config
    can genuinely diverge in which candidate wins a round's tournament
    purely from unseeded sampling noise, on top of gen_params' own
    unseeded noise (see run_group_rounds). None (vLLM's own default,
    unseeded) if not given.

    floor_threshold/margin_ref/max_beta — evidence_scaled_beta's params
    (see its docstring): the belief update's effective beta gets scaled
    per-call by how decisive AND how absolutely good the winner/loser's
    raw verifier scores were this round, instead of a flat beta for every
    win regardless of score quality.
    """
    pools: dict[int, list[str]] = {}
    pool_answers: dict[int, list[str]] = {}
    for local_j, i in enumerate(active):
        incumbent = [state[i]["C_best"]] if round_idx >= 2 else []
        incumbent_answer = [state[i]["answer"]] if round_idx >= 2 else []
        pools[i] = incumbent + candidate_lists[local_j]
        pool_answers[i] = incumbent_answer + answer_lists[local_j]
    pool_size = k_candidates + (1 if round_idx >= 2 else 0)

    w = {i: [0.0] * pool_size for i in active}
    c = {i: [0] * pool_size for i in active}
    # Raw verifier scores (r_a/r_b, before the bradley_terry transform) per
    # pool index — used only for evidence_scaled_beta's margin/floor-gate
    # calculation at finalize time. w/c (soft win-rate) still drives
    # ranking/top1 selection, unchanged; this is a separate signal for how
    # ABSOLUTELY good the winner's own score was, which win-rate alone
    # can't express (win-rate is relative to whatever this round's pool
    # happened to contain, even if the whole pool is weak).
    score_sum = {i: [0.0] * pool_size for i in active}
    score_n = {i: [0] * pool_size for i in active}
    n_comparisons = {i: 0 for i in active}
    # First analysis text that ever mentioned each pool index — used as
    # this round's feedback for whichever index ends up the winner. Every
    # pool member is guaranteed an entry once the ring pass finishes,
    # since ring_cycle touches every index at least once.
    first_analysis: dict[int, dict[int, str]] = {i: {} for i in active}

    analysis_params = SamplingParams(temperature=1.0, max_tokens=150, seed=seed)
    score_params = score_prefill_params()

    def _accumulate(pair_by_image: dict[int, tuple[int, int]]) -> None:
        result = _score_pairs_batch(
            llm, tokenizer, images, active, pools, pair_by_image, prompts, vprompts, cls_q, rubric, rng,
            analysis_params, score_params, call_stats,
            agml_dataset, class_label, image_names, round_idx, log_path,
        )
        for i in active:
            r_a, r_b, analysis, tok = result[i]
            a, b = pair_by_image[i]
            p = bradley_terry(r_a, r_b)
            w[i][a] += p
            c[i][a] += 1
            w[i][b] += 1.0 - p
            c[i][b] += 1
            score_sum[i][a] += r_a
            score_n[i][a] += 1
            score_sum[i][b] += r_b
            score_n[i][b] += 1
            n_comparisons[i] += 1
            first_analysis[i].setdefault(a, analysis)
            first_analysis[i].setdefault(b, analysis)
            state[i]["round_prompt_tokens"] += tok[0]
            state[i]["round_completion_tokens"] += tok[1]

    if pool_size > 1:
        # ── Step 1: ring pass ────────────────────────────────────────────
        rings = {i: ring_cycle(pool_size, rng) for i in active}
        for t in range(pool_size):  # ring_cycle always returns pool_size pairs
            _accumulate({i: rings[i][t] for i in active})

        # ── Step 2: pivot selection (per image, from ring-pass win rate) ──
        pivots = {i: select_pivots(w[i], c[i], k_pivots) for i in active}

        # ── Step 3: pivot rounds ───────────────────────────────────────────
        pivot_pairs = {i: pivot_round_pairs(pool_size, pivots[i]) for i in active}
        n_pivot_pairs = len(next(iter(pivot_pairs.values())))  # uniform across images, see docstring
        for t in range(n_pivot_pairs):
            _accumulate({i: pivot_pairs[i][t] for i in active})

    # ── Pass 1: rank each pool from the tournament's own aggregate win-rate.
    ranked: dict[int, tuple[list[float], int]] = {}
    for i in active:
        if pool_size == 1:
            win_rate, top1 = [0.5], 0
        else:
            win_rate = [w[i][k] / c[i][k] if c[i][k] else 0.0 for k in range(pool_size)]
            top1 = max(range(pool_size), key=lambda k: (win_rate[k], -k))
        ranked[i] = (win_rate, top1)

    # ── Pass 2: same-answer check, only where a NEW candidate outranked
    # the incumbent's own pool slot (round>=2, top1 != 0) — decides whether
    # the persistent belief carries forward (reworded, same conclusion) or
    # resets (genuine answer change). Trivial/skipped everywhere else.
    need_judge = {
        i: (pool_answers[i][0], pool_answers[i][ranked[i][1]])
        for i in active if round_idx >= 2 and ranked[i][1] != 0
    }
    same_result = _judge_same_answer(
        llm, tokenizer, cls_q, vprompts, need_judge, call_stats,
        agml_dataset, class_label, image_names, round_idx, log_path,
    )
    for i, (_same, p_tok, c_tok) in same_result.items():
        state[i]["round_prompt_tokens"] += p_tok
        state[i]["round_completion_tokens"] += c_tok

    # ── Pass 3: finalize this round's winner/loser and persistent update.
    for i in active:
        s = state[i]
        pool_texts = pools[i]
        win_rate, top1 = ranked[i]

        def _runner_up(exclude: int, win_rate=win_rate) -> int:
            return max((k for k in range(pool_size) if k != exclude), key=lambda k: (win_rate[k], -k))

        if round_idx == 1:  # no incumbent yet
            winner_idx = top1
            loser_idx = _runner_up(top1) if pool_size > 1 else top1
            status = "seed"
            class_match = False
        elif top1 == 0:  # incumbent's own pool slot won again — trivially the same answer
            winner_idx = 0
            loser_idx = _runner_up(0) if pool_size > 1 else 0
            status = "defended"
            class_match = True
        else:  # a NEW candidate outranked the incumbent — same_result decides which
            class_match = same_result.get(i, (False, 0, 0))[0]
            if class_match:
                winner_idx = top1
                loser_idx = _runner_up(top1) if pool_size > 1 else top1
                status = "defended"
            else:
                winner_idx, loser_idx, status = top1, 0, "replaced"

        persistent_is_winner = round_idx >= 2 and class_match
        persistent_is_loser = round_idx >= 2 and status == "replaced"
        winner_belief = s["belief"] if persistent_is_winner else Belief(0.0, 1.0)
        loser_belief = s["belief"] if persistent_is_loser else Belief(0.0, 1.0)

        # Evidence-scaled beta: how decisive AND how absolutely good this
        # round's winner/loser raw scores were, not just a flat beta for
        # every win. See evidence_scaled_beta's docstring. Falls back to
        # the flat beta when there's no score data for one side (pool_size
        # == 1 — no comparisons ever happened this round).
        if score_n[i][winner_idx] and score_n[i][loser_idx]:
            winner_score = score_sum[i][winner_idx] / score_n[i][winner_idx]
            loser_score = score_sum[i][loser_idx] / score_n[i][loser_idx]
            eff_beta = evidence_scaled_beta(beta, winner_score, loser_score, floor_threshold, margin_ref, max_beta)
        else:
            eff_beta = beta
        new_winner, _new_loser = pairwise_update(winner=winner_belief, loser=loser_belief, beta=eff_beta)

        s["belief"] = new_winner
        s["C_best"] = pool_texts[winner_idx]
        s["answer"] = pool_answers[i][winner_idx]
        # No truncation here — the analysis is already hard-capped at
        # generation time (analysis_params.max_tokens, currently 280), so
        # a second, tighter [:N] cut on top of that only loses information
        # without providing any real additional bound. This used to be
        # [:300], which — now that pairwise_prompt requires a full
        # per-criterion walkthrough before its 1-2 sentence summary
        # verdict — would have cut off well before the walkthrough even
        # finished, discarding exactly the specific, actionable detail
        # (e.g. "never mentions the yellow discoloration") that makes
        # this feedback useful for the next round's re-ask.
        s["feedback"] = first_analysis[i].get(winner_idx, "")
        s["pool_size"] = pool_size
        s["n_comparisons"] = n_comparisons[i]
        s["top1_win_rate"] = round(win_rate[top1], 4)
        s["incumbent_survived"] = int(round_idx >= 2 and class_match)
        s["pairwise_winner"] = status
        s["transcript"].append(
            f"Round {round_idx} ({status}, pool={pool_size}, comparisons={n_comparisons[i]}, "
            f"top1_win_rate={win_rate[top1]:.3f}): winner={pool_texts[winner_idx]} "
            f"[answer={s['answer']!r}]"
        )


def run_group_rounds(
    llm, tokenizer, images: list, cls_q: str, gen_template: str, rubric: str, prompts: dict, vprompts: dict,
    k_candidates: int, min_round: int, max_round: int,
    beta: float, high: float, medium: float, gen_temperature: float, k_pivots: int, rng: random.Random,
    agml_dataset: str, class_label: str, image_names: list[str],
    log_path: Path | None = None, call_stats: dict | None = None,
    floor_threshold: float = 0.5, margin_ref: float = 0.3, max_beta: float = 4.0,
    seed: int | None = None, enable_thinking: bool = False, gen_max_tokens: int = 180,
) -> list[list[dict]]:
    """Run the full round loop for every image in one (dataset, class) group
    in parallel, each image stopping independently once its own stopping
    rule fires (unlike cot_yaml.py's fixed step count, round count varies
    per image here — that's the point of an adaptive stopping rule).
    Returns per_image_rounds: one list per image, one entry per round
    actually run for that image.

    log_path: if given, every verifier call (score or pairwise) is logged
    in full (prompt + raw response + parsed result) to this JSONL file —
    same convention as run_judge_phase's judge_log.jsonl.
    call_stats: if given, a shared dict incremented in place with n_calls/
    n_prompts across every run_batch_n call this function makes — see
    run_batch_n's docstring. Both are run_model()'s to own and pass
    through; this function never creates them itself, so a caller that
    doesn't want logging/counting can simply omit them.

    gen_template: the candidate-generation prompt for this group's task
    type — vprompts["structured_templates"][rubric_key], resolved by the
    caller (run_model already looks up rubric_key to select `rubric`, so
    it resolves gen_template the same way). Filled with {question}=cls_q
    and sampled k_candidates times at gen_temperature every round — same
    generation call shape as before this existed, just a different prompt
    (fill-in-the-blank fields mirroring the rubric's own criteria, ending
    in the same <answer></answer> convention extract_answer_tag already
    depends on) in place of a bare "answer directly" instruction. See
    structured_templates' comment in verifier_prompts.yaml for why (two
    earlier alternatives — rephrasing the question, and asking for several
    different hypotheses in one response — were tried and reverted).

    seed: fixes the RNG for every SamplingParams this function and
    _process_round_belief create (gen_params' k-candidate sampling,
    analysis_params' pairwise-duel analysis) so two runs over identical
    images/config are reproducible rather than genuinely re-drawing
    different candidates/tournament outcomes each time — this was the
    root cause of a real 1-round-vs-6-round comparison looking misleading
    (round 1 alone swung between two runs from unseeded generation noise
    plus judge-side batching nondeterminism, not any change in the
    belief-tracking mechanism itself). run_model passes args.seed (the
    same seed already used for rng's pivot-selection/incumbent-slot
    randomization) — pass None to opt back into vLLM's own unseeded
    default.

    enable_thinking/gen_max_tokens: --enable_thinking (default off, matches
    every other generation script in this repo — see chat_prompt.py's
    module docstring) turns on the model-being-tested's own extended
    thinking for the k_candidates generation step ONLY (run_batch_n's
    call below) — not the verifier's own pairwise/same-answer calls, see
    run_batch_n's docstring for why. gen_max_tokens defaults to 180 (fits
    a direct fill-in-the-blank answer with no thinking); raise it via
    config when enabling thinking or a <think>...</think> block will
    routinely get truncated before the model ever reaches its actual
    filled-in template / <answer> tag."""
    n = len(images)
    gen_q = gen_template.format(question=cls_q)
    base_hist = [_sys_msg(prompts) + [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": gen_q},
    ]}] for _ in range(n)]

    state = [{"C_best": "", "answer": "", "belief": None, "anchor": None,
              "feedback": "", "transcript": [], "pairwise_winner": "",
              "pool_size": 0, "n_comparisons": 0, "top1_win_rate": 0.0, "incumbent_survived": 0,
              "round_prompt_tokens": 0, "round_completion_tokens": 0,
              # (winning_trajectory, feedback) appended once per round —
              # replayed IN FULL by _gen_histories_for_round so round k's
              # generation sees every prior round's winner + feedback, not
              # just the immediately preceding one. See run_group_rounds'
              # docstring.
              "history": []}
             for _ in range(n)]
    per_image_rounds: list[list[dict]] = [[] for _ in range(n)]
    active = list(range(n))
    round_idx = 0

    while active and round_idx < max_round:
        round_idx += 1
        t_round_start = time.perf_counter()
        for i in active:
            state[i]["round_prompt_tokens"] = 0
            state[i]["round_completion_tokens"] = 0

        gen_hists = _gen_histories_for_round(base_hist, state, active, round_idx, gen_q)
        gen_params = SamplingParams(
            temperature=gen_temperature, max_tokens=gen_max_tokens,
            repetition_penalty=1.1, n=k_candidates, seed=seed,
        )
        gen_out = run_batch_n(llm, tokenizer, [images[i] for i in active], gen_hists, gen_params, call_stats,
                               enable_thinking=enable_thinking)
        candidate_lists = [[c["text"] for c in cands] for cands in gen_out]
        answer_lists, answer_found_lists = [], []
        for cands in gen_out:
            answers, founds = [], []
            for c in cands:
                a, f = extract_answer_tag(c["text"])
                answers.append(a)
                founds.append(f)
            answer_lists.append(answers)
            answer_found_lists.append(founds)

        for local_j, i in enumerate(active):
            state[i]["round_prompt_tokens"] += gen_out[local_j][0]["prompt_tokens"]
            state[i]["round_completion_tokens"] += sum(c["completion_tokens"] for c in gen_out[local_j])
            # Full per-candidate transparency — every trajectory generated
            # this round, whether or not it ends up scored/selected, plus
            # its extracted <answer> and whether the tag was actually
            # found (lets the tag mechanism's own hit rate be checked from
            # the log later, not just trusted).
            for cand_idx, c in enumerate(gen_out[local_j]):
                _log_verifier_call(
                    log_path, agml_dataset=agml_dataset, class_label=class_label,
                    image_name=image_names[i], round=round_idx, call_type="candidate",
                    candidate_index=cand_idx, raw_text=c["text"],
                    extracted_answer=answer_lists[local_j][cand_idx],
                    answer_tag_found=answer_found_lists[local_j][cand_idx],
                    prompt_tokens=c["prompt_tokens"], completion_tokens=c["completion_tokens"],
                )
            # Round-level diversity summary — one record per image per round,
            # separate from the per-candidate ones above, so "how much do
            # this round's k_candidates actually vary" is a single readable
            # field instead of something you have to reconstruct by
            # re-grouping every "candidate" row yourself. Distinct count
            # normalizes the extracted answer (lowercased/stripped) the
            # same way the earlier rephrase/multi_hypothesis diagnostics
            # did, so this is directly comparable to those measurements
            # (e.g. the 35.6%-of-image-rounds-collapsed finding that
            # originally motivated looking at this at all).
            round_answers = answer_lists[local_j]
            _log_verifier_call(
                log_path, agml_dataset=agml_dataset, class_label=class_label,
                image_name=image_names[i], round=round_idx, call_type="round_diversity",
                n_candidates=len(round_answers),
                n_distinct_answers=len({a.strip().lower() for a in round_answers if a}),
                answers=round_answers,
                # Prompt tokens are shared across all k_candidates of this
                # round (same gen_hists[i] input), so this IS the size of
                # that round's replayed cumulative history — watch this
                # grow round-over-round to see the accumulated-history
                # feature (run_group_rounds' docstring) actually costing
                # tokens, since the 300-char feedback truncation was
                # removed and now nothing caps how far it can grow.
                round_prompt_tokens=state[i]["round_prompt_tokens"],
                round_completion_tokens=state[i]["round_completion_tokens"],
            )

        _process_round_belief(
            llm, tokenizer, images, active, candidate_lists, answer_lists, state, prompts, vprompts,
            cls_q, rubric, k_candidates, round_idx, beta, k_pivots, rng,
            agml_dataset, class_label, image_names, log_path, call_stats,
            floor_threshold, margin_ref, max_beta, seed,
        )

        round_elapsed = time.perf_counter() - t_round_start

        # ── Stopping check + row build ───────────────────────────────────────
        still_active = []
        for i in active:
            s = state[i]
            if round_idx == 1:
                s["anchor"] = s["belief"]
                p = 0.5  # anchor compared to itself — not meaningful, see doc Sec. 13
            else:
                p = stop_probability(s["belief"], s["anchor"], beta)
            mu, sigma2 = s["belief"].mu, s["belief"].sigma2
            pool_size, n_comparisons = s["pool_size"], s["n_comparisons"]
            top1_win_rate, incumbent_survived = s["top1_win_rate"], s["incumbent_survived"]
            pairwise_winner = s["pairwise_winner"]
            verifier_answer = s["answer"]
            # Append THIS round's (winner, feedback) for the NEXT round's
            # _gen_histories_for_round to replay — after this point s["C_best"]/
            # s["feedback"] are this round's final values (set inside
            # _process_round_belief above).
            s["history"].append((s["C_best"], s["feedback"]))

            stop_now = should_stop(round_idx, p, min_round, max_round, high)
            per_image_rounds[i].append({
                "round": round_idx,
                "prediction": s["C_best"],
                "reasoning": "\n\n".join(s["transcript"]),
                "pool_size": pool_size, "n_comparisons": n_comparisons,
                "top1_win_rate": top1_win_rate, "incumbent_survived": incumbent_survived,
                "pairwise_winner": pairwise_winner, "verifier_answer": verifier_answer,
                "verifier_feedback": s["feedback"],
                "mu": mu, "sigma2": sigma2,
                "stop_confidence": round(p, 4),
                "confidence_bin": confidence_bin(p, high, medium),
                "is_final_round": int(stop_now),
                "prompt_tokens": s["round_prompt_tokens"],
                "completion_tokens": s["round_completion_tokens"],
                # Wall-clock for this whole round's batch (generation +
                # verifier calls), shared across every image active this
                # round — not exclusive per-image, since everything here is
                # batched. Still informative for "does cost grow with round
                # depth" even though it double-counts across images.
                "elapsed_s": round(round_elapsed, 3),
            })
            if not stop_now:
                still_active.append(i)
        active = still_active

    return per_image_rounds


def _round_row(entry: dict, ds, clean_label: str, crop_val, task_type: str, name: str, gt_hint: str) -> dict:
    resolved = resolve_class(entry["prediction"], ds.clean_classes)
    correct = resolved is not None and resolved.lower() == clean_label.lower()
    c_match = contains_match(resolved or entry["prediction"], clean_label)
    scores = rouge_scores_per_sample(entry["prediction"], clean_label, include_rouge=False)
    return {
        "agml_dataset": ds.agml_name, "class_label": clean_label,
        "task_type": task_type, "crop": crop_display(crop_val),
        "image_name": name,
        "round": entry["round"], "mu": entry["mu"], "sigma2": entry["sigma2"],
        "stop_confidence": entry["stop_confidence"], "confidence_bin": entry["confidence_bin"],
        "pool_size": entry["pool_size"], "n_comparisons": entry["n_comparisons"],
        "top1_win_rate": entry["top1_win_rate"], "incumbent_survived": entry["incumbent_survived"],
        "verifier_answer": entry["verifier_answer"], "verifier_feedback": entry["verifier_feedback"],
        "pairwise_winner": entry["pairwise_winner"], "is_final_round": entry["is_final_round"],
        "prompt_tokens": entry["prompt_tokens"], "completion_tokens": entry["completion_tokens"],
        "elapsed_s": entry["elapsed_s"],
        "prediction": entry["prediction"], "resolved_class": resolved or "",
        "ground_truth": clean_label, "correct": int(correct), "contains_match": int(c_match),
        "resolvable": int(resolved is not None), "reasoning": entry["reasoning"],
        "judge_gt_hint": gt_hint, **scores,
    }


# ── Plot: belief per round, with bootstrap CI ───────────────────────────────

def plot_belief_by_round(model_dir: Path, rows: list[dict]) -> None:
    """Mean (+ bootstrap 95% CI) of the belief mean (mu) per round, one line
    per task_type plus overall — the belief-tracking analogue of
    cot_yaml_step_report.py's step_f1_overall.png (doc Section 14)."""
    import numpy as np
    import matplotlib.pyplot as plt

    y_key = "mu"
    by_group: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        val = r.get(y_key, "")
        if val == "":
            continue
        rnd = int(r["round"])
        by_group["overall"][rnd].append(float(val))
        by_group[r.get("task_type", "")][rnd].append(float(val))

    if not by_group.get("overall"):
        print("  [plot] no belief data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    for label in sorted(by_group, key=lambda k: (k != "overall", k)):
        per_round = by_group[label]
        rounds = sorted(per_round)
        means, lo, hi = [], [], []
        for rnd in rounds:
            vals = np.array(per_round[rnd])
            means.append(float(vals.mean()))
            if len(vals) > 1:
                boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
                l, h = np.percentile(boot, [2.5, 97.5])
            else:
                l = h = float(vals[0])
            lo.append(l)
            hi.append(h)
        lw = 2.5 if label == "overall" else 1.2
        ax.plot(rounds, means, marker="o", label=label, linewidth=lw)
        ax.fill_between(rounds, lo, hi, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel("Belief mean (mu)")
    ax.set_title("Verifier-CoT (belief) — belief per round, 95% CI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = model_dir / "belief_by_round_belief.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main run ─────────────────────────────────────────────────────────────────

def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    vprompts = _load_yaml(Path(args.verifier_prompts))
    manifest = load_manifest(Path(args.manifest))

    # Flat <output_dir>/<model>/ layout so this stays compatible with the
    # rest of the repo's load_all_rows()-style loaders.
    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{args.baseline}_{safe_model_name(model_id)}.csv"
    # verifier_log_<baseline>.jsonl is this script's own raw per-call log
    # (every tournament_pair/same_answer prompt+response) — must be archived
    # alongside the checkpoint on --rerun too, or old and new run entries
    # silently concatenate in one ever-growing file with no way to tell
    # them apart (this bit a real prompt-fix verification: a diagnostic run
    # counted old-prompt AND new-prompt calls together and looked like the
    # fix had only partially worked, when it had fully worked).
    archive_ckpt_if_rerun(ckpt_path, args.rerun,
                           extra_logs=("judge_log.jsonl", f"verifier_log_{args.baseline}.jsonl"))
    rows, _ = load_ckpt(ckpt_path, fields=VERIFIER_STEP_FIELDS)
    finished = _rounds_seen_complete(rows)
    if rows:
        print(f"  [resume] {len(rows)} round row(s) already scored in {ckpt_path.name} "
              f"({len(finished)} image(s) complete)")

    dataset_filter = set(args.subsample) if args.subsample else None
    all_groups = list(_build_groups(args, manifest, dataset_filter=dataset_filter))
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
        pending = [n for n in names if (ds.agml_name, clean_label, n) not in finished]
        if not pending:
            continue
        groups.append((ds, clean_label, crop_val, task_type, gt_hint, pending, path_by_name))
    if dataset_filter is not None:
        missing = dataset_filter - {ds.agml_name for ds, *_ in groups}
        if missing:
            print(f"  [warn] --subsample not found (skip/not in manifest/already complete?): {sorted(missing)}")

    n_rows_before = len(rows)
    call_stats: dict = {"n_calls": 0, "n_prompts": 0}
    log_path = model_dir / f"verifier_log_{args.baseline}.jsonl"
    elapsed = 0.0
    # One rng, seeded once, threaded through every group in a fixed
    # processing order — reproducible positional-bias randomization
    # (--baseline belief's incumbent A/B slot per duel) across a full run.
    rng = random.Random(args.seed)

    if not groups:
        print("  Nothing to do — all samples already checkpointed.")
    else:
        n_pending = sum(len(g[5]) for g in groups)
        print(f"  {len(groups)} (dataset, class) group(s), {n_pending} pending sample(s) "
              f"— --baseline {args.baseline}")

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

            per_image_rounds = run_group_rounds(
                llm, tokenizer, images, cls_q, gen_template, rubric, prompts, vprompts,
                k_candidates=args.k_candidates,
                min_round=args.min_round, max_round=args.max_round,
                beta=args.beta, high=args.high_conf, medium=args.med_conf,
                gen_temperature=args.temperature, k_pivots=args.k_pivots, rng=rng,
                agml_dataset=ds.agml_name, class_label=clean_label, image_names=names,
                log_path=log_path, call_stats=call_stats,
                floor_threshold=args.floor_threshold, margin_ref=args.margin_ref, max_beta=args.max_beta,
                seed=args.seed, enable_thinking=args.enable_thinking, gen_max_tokens=args.gen_max_tokens,
            )

            for name, entries in zip(names, per_image_rounds):
                for entry in entries:
                    row = _round_row(entry, ds, clean_label, crop_val, task_type, name, gt_hint)
                    append_ckpt_row(ckpt_path, row, fields=VERIFIER_STEP_FIELDS)
                    rows.append(row)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer

    rows = run_judge_phase(rows, ckpt_path, args, prompts, fields=VERIFIER_STEP_FIELDS)

    final_rows = [r for r in rows if str(r.get("is_final_round")) == "1"]
    metrics = compute_model_metrics(final_rows)
    (model_dir / f"metrics_{args.baseline}.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Overall (final rounds only): f1_macro={metrics['overall'].get('f1_macro')} "
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
    (model_dir / f"compute_summary_{args.baseline}.json").write_text(json.dumps(compute_summary, indent=2))
    print(f"  Compute: {compute_summary['this_run']['n_llm_calls']} LLM call(s) this run "
          f"({compute_summary['this_run']['n_prompts_sent']} prompts), "
          f"{compute_summary['cumulative_all_rows']['total_prompt_tokens']}+"
          f"{compute_summary['cumulative_all_rows']['total_completion_tokens']} prompt+completion "
          f"tokens across all rows — full verifier call log: {log_path.name}")

    plot_belief_by_round(model_dir, rows)


def main(args: argparse.Namespace) -> None:
    if args.collate:
        collated: dict[str, dict] = {}
        base = Path(args.output_dir)
        if base.exists():
            for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                for ckpt_path in sorted(model_dir.glob("ckpt_*.csv")):
                    baseline = ckpt_path.stem.removeprefix("ckpt_").removesuffix(f"_{model_dir.name}")
                    rows, _ = load_ckpt(ckpt_path, fields=VERIFIER_STEP_FIELDS)
                    final_rows = [r for r in rows if str(r.get("is_final_round")) == "1"]
                    if final_rows:
                        collated[f"{model_dir.name}/{baseline}"] = compute_model_metrics(final_rows)
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
        help="YAML file of defaults (k_candidates, beta, min_round/max_round, etc.) — edit it "
             "directly for routine changes instead of passing flags every time. Any flag also "
             "passed on the command line overrides this file.",
    )
    parser.add_argument("--model", default=None, help="Single model to run. Not needed with --collate.")
    parser.add_argument("--baseline", choices=["belief"], default="belief",
                         help="Pairwise Bradley-Terry belief tracking (docs/verifier_score_"
                              "belief_update.md). Kept as a flag (rather than removed outright) "
                              "so output paths (ckpt_belief_*.csv, verifier_log_belief.jsonl, "
                              "etc.) stay stable for cot_compare.py/verifier_summary.py, which "
                              "parameterize on this value. --max_round 1 is now the "
                              "argmax-equivalent baseline — see module docstring.")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--verifier_prompts", default=str(_VERIFIER_PROMPTS_PATH))
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--output_dir", default="outputs_final/verifier_cot/")
    parser.add_argument("--n_samples", type=int, default=10, help="Images/class to evaluate.")
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
                         help="Seeds the incumbent-A/B-slot randomization in --baseline belief's "
                              "pairwise scoring (positional-bias mitigation).")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint aside, timestamped, and regenerate "
                              "every row from scratch. Old files are archived, never deleted.")
    parser.add_argument("--k_candidates", type=int, default=3,
                         help="Candidates generated per round (K) via temperature sampling.")
    parser.add_argument("--k_pivots", type=int, default=DEFAULT_PIVOTS,
                         help="k — pivot-set size for the full PPT tournament (arXiv:2607.05391 "
                              "Section 3.2, via llm_verifier.pivot_tournament) that ranks each "
                              "round's pool — --baseline belief only.")
    parser.add_argument("--min_round", type=int, default=3)
    parser.add_argument("--max_round", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Sampling temperature for the K per-round candidates. Needs to be > 0 "
                              "for --k_candidates > 1 to actually produce differing candidates — "
                              "greedy decoding (temperature=0) returns identical copies.")
    parser.add_argument("--enable_thinking", action="store_true",
                         help="Turn on the model-being-tested's own extended thinking "
                              "(<think>...</think>) for the k_candidates generation step only — every "
                              "other generation script in this repo disables it by default (see "
                              "src/common/chat_prompt.py's module docstring: it compounds badly across "
                              "multi-turn pipelines). Off by default to match that convention; this "
                              "flag exists to A/B test whether reasoning before committing to a "
                              "diagnosis reduces the generic-category hedging found in judge-leniency "
                              "analysis (e.g. answering 'leaf spot disease' instead of naming the "
                              "specific disease). Does NOT affect the verifier's own pairwise-duel or "
                              "same-answer calls — see run_batch_n's docstring. Raise --gen_max_tokens "
                              "when using this, or thinking traces will routinely get truncated before "
                              "the model reaches its actual answer.")
    parser.add_argument("--gen_max_tokens", type=int, default=180,
                         help="max_tokens for the k_candidates generation step. 180 fits a direct "
                              "fill-in-the-blank answer with no thinking; raise substantially (e.g. "
                              "600-1000) when --enable_thinking is set.")
    parser.add_argument("--beta", type=float, default=0.5,
                         help="Bradley-Terry per-comparison noise (doc Sec. 5) — --baseline belief only. "
                              "This is the FLOOR value used for a high-confidence win; the actual "
                              "per-update beta is scaled UP toward --max_beta as evidence weakens "
                              "(evidence_scaled_beta), using each round's actual winner/loser scores — "
                              "not applied as a flat constant. See src/common/belief_update.py's "
                              "evidence_scaled_beta docstring for why it blends toward a much larger "
                              "max_beta rather than dividing beta down (the belief-update math is "
                              "nearly insensitive to beta below ~1.0).")
    parser.add_argument("--floor_threshold", type=float, default=0.5,
                         help="--baseline belief only. Belief-update discount gate: if the winning "
                              "candidate's own raw verifier score (averaged across its comparisons "
                              "this round) is below this threshold, the update is weakened toward "
                              "--max_beta regardless of how decisively it won — 'best of several bad "
                              "candidates' shouldn't build belief just because the margin was large. "
                              "See evidence_scaled_beta's docstring for the full rationale.")
    parser.add_argument("--margin_ref", type=float, default=0.3,
                         help="--baseline belief only. Winner-vs-loser score gap (out of 1.0) that "
                              "counts as a 'decisive enough' win — margins at or above this get full "
                              "confidence from the margin side of evidence_scaled_beta; smaller margins "
                              "scale down proportionally. 0 disables margin-sensitivity (only the floor "
                              "gate still applies).")
    parser.add_argument("--max_beta", type=float, default=4.0,
                         help="--baseline belief only. Belief-update beta used for the WEAKEST possible "
                              "evidence (winner_score at/below 0, or margin at/below 0) — a value in "
                              "the range where pairwise_update's moment-matching formula actually "
                              "produces a visibly smaller belief shift (unlike --beta's default 0.5, "
                              "which sits in a nearly-flat/insensitive region). See "
                              "evidence_scaled_beta's docstring.")
    parser.add_argument("--high_conf", type=float, default=0.85, help="Stop-early / 'high' bin threshold.")
    parser.add_argument("--med_conf", type=float, default=0.6, help="'medium' vs 'low' bin threshold.")
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
                         help="Skip inference — combine every <output_dir>/<model>/<baseline>/ folder "
                              "into collated_summary.csv / collated_metrics.json.")

    if _pre_args.verifier_config and Path(_pre_args.verifier_config).exists():
        _cfg = _load_yaml(Path(_pre_args.verifier_config)) or {}
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests
        if _unknown:
            print(f"[warn] {_pre_args.verifier_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
