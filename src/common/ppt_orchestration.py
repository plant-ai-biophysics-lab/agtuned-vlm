"""Shared K-candidate + single-round PPT-tournament orchestration —
extracted (2026-08-26) from `verifier_cot.py::run_group`/`_score_pairs_batch`
and `detection_verifier.py::run_group`/`_score_pairs_batch`, which had
independently carried byte-for-byte identical ring-pass/pivot-selection/
pivot-round/Bradley-Terry-aggregation logic (confirmed via diff before this
extraction — the only differences were docstrings, local variable names, and
each caller's own log field naming). Kept as ONE implementation from here on
so "does the wrapper/skill get the same results as the paper pipeline" is a
structural fact (one function, one call path), not an ongoing claim to keep
two copies in sync.

This module is intentionally domain-agnostic: it knows nothing about images,
AgML datasets, or whether the trajectories being compared are text answers
or box lists — it only knows how to run the PPT tournament's three stages
(arXiv:2607.05391 Section 3.2, via llm_verifier.pivot_tournament, UNMODIFIED)
over an abstract pool of `pool_size` candidates per "active" item, given a
caller-supplied `score_batch_fn` that scores one scheduled pair per active
item and returns each pair's Bradley-Terry-ready (R_a, R_b) rewards. Reused
by:
  - src/verifier_cot.py (classification, text trajectories)
  - src/detection_verifier.py (object detection, box-list trajectories)
  - src/ppt_verify/tournament.py (generic CLI/backend-agnostic entry point)

`score_pairs_batch` (the two-pass prefill scoring call itself, batched
across active items) is ALSO extracted here — same "one implementation"
rationale — but deliberately does NOT do any logging; callers wrap it in
their own thin function that calls this core and then logs however/whatever
fields they need (verifier_cot.py and detection_verifier.py log slightly
different field names/content today; preserving that is a log-schema
concern, not part of the actual scoring math, so it stays caller-owned)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.common.chat_prompt import render_chat_template
from src.common.ppt_scoring import (
    analysis_before_tags, build_score_prompt, extract_tag_score, letter_probability_diagnostics,
    pairwise_scoring_prompt,
)

try:
    from llm_verifier.pivot_tournament import bradley_terry, pivot_round_pairs, ring_cycle, select_pivots
except ImportError:
    bradley_terry = pivot_round_pairs = ring_cycle = select_pivots = None  # type: ignore[assignment]


@dataclass
class PairScoreResult:
    """One scored, already-unswapped pairwise comparison for one active
    item: (r_a, r_b) are Bradley-Terry-ready [0,1] rewards for pool
    indices a and b respectively (see the pair's own `a`/`b` in
    PairOwner), `analysis` is the pass-1 free-generated comparison text,
    `prompt_tokens`/`completion_tokens` are this ONE comparison's own
    token usage (both passes combined) — callers accumulate these into
    their own running totals."""
    r_a: float
    r_b: float
    analysis: str
    prompt_tokens: int
    completion_tokens: int
    # Diagnostic-only (Paper 2 Stage 2, docs/legacy/stage2_verifier_corrections_paused.md)
    # — letter_probability_diagnostics' raw max/sum breakdown for each
    # side's forced score-token position, and each side's trajectory word
    # count. None for any caller that doesn't need this (no behavior
    # change; nothing here feeds the actual r_a/r_b score).
    diag_a: dict | None = None
    diag_b: dict | None = None
    words_a: int | None = None
    words_b: int | None = None


@dataclass
class PairOwner:
    """Everything about one scheduled pair a caller needs to log it —
    mirrors the `owners` tuple `_score_pairs_batch` built internally in
    both original implementations."""
    index: int          # active-item index (the "i" in pair_by_image)
    a: int               # pool index of trajectory A (pre-swap)
    b: int               # pool index of trajectory B (pre-swap)
    swapped: bool         # True if A/B were flipped before scoring (rng coin flip)
    prompt_text: str      # the pairwise scoring prompt actually sent


def score_pairs_batch(
    llm, tokenizer, images: list, active: list[int], pools: dict[int, list],
    pair_by_image: dict[int, tuple[int, int]], prompts: dict, vprompts: dict,
    question: str, rubric: str, rng: random.Random, analysis_params, score_params,
    call_stats: dict | None,
    trajectory_text_fn: Callable[[int, int], str] | None = None,
) -> tuple[dict[int, PairScoreResult], dict[int, PairOwner]]:
    """Score one pool-index pair PER active item via ppt_scoring.py's
    two-pass prefill mechanism, batched across every active item in one
    set of vLLM calls — pairs may differ per item (ring-pass cycles and
    pivot selections are randomized/ranked per item), only the COUNT of
    pairs per pass is uniform (guaranteed by ring_cycle/pivot_round_pairs
    depending only on pool size and k, not on which specific indices end
    up as pivots). Positional slot (A/B) randomized per (item, pair) — a
    coin flip via `rng` — as an extra safeguard on top of the paper's own
    ring-pass cancellation.

    `trajectory_text_fn(item_index, pool_index) -> str`: how to turn a
    pool entry into the text shown to the pairwise verifier prompt.
    Defaults to `pools[item_index][pool_index]` used as-is (classification's
    convention, where pool entries already ARE the trajectory text);
    detection's candidates are also plain text (structured criteria answer
    + box list), so the default covers both current callers — the hook
    exists for a future trajectory kind that needs its own text rendering
    (e.g. a non-text-native representation) without forking this function
    again.

    Returns (results, owners) — both keyed by active-item index. Does NOT
    log anything; callers combine `results`/`owners` with whatever log
    fields they want and call their own logging function per pair (see
    module docstring)."""
    if trajectory_text_fn is None:
        trajectory_text_fn = lambda i, k: pools[i][k]  # noqa: E731

    base_prompts, owners_list, words_list = [], [], []
    for i in active:
        a, b = pair_by_image[i]
        text_a, text_b = trajectory_text_fn(i, a), trajectory_text_fn(i, b)
        words_list.append((len(text_a.split()), len(text_b.split())))
        swapped = rng.random() < 0.5
        traj_a, traj_b = (text_b, text_a) if swapped else (text_a, text_b)
        pw_text = pairwise_scoring_prompt(rubric=rubric, question=question, trajectory_a=traj_a,
                                           trajectory_b=traj_b, template=vprompts["pairwise_prompt"])
        s = prompts.get("assessment_system", "")
        sys_msg = [{"role": "system", "content": s}] if s else []
        msgs = sys_msg + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": pw_text},
        ]}]
        base_prompts.append({
            "prompt": render_chat_template(tokenizer, msgs),
            "multi_modal_data": {"image": images[i]},
        })
        owners_list.append((i, a, b, swapped, pw_text))

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

    scores_a: list[float | None] = [None] * len(owners_list)
    scores_b: list[float | None] = [None] * len(owners_list)
    diags_a: list[dict | None] = [None] * len(owners_list)
    diags_b: list[dict | None] = [None] * len(owners_list)
    for tag, scores, diags in (("<score_A>", scores_a, diags_a), ("<score_B>", scores_b, diags_b)):
        score_prompts = [build_score_prompt(bp, an, tag) for bp, an in zip(base_prompts, analyses)]
        score_outputs = llm.generate(score_prompts, sampling_params=score_params)
        if call_stats is not None:
            call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
            call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(score_prompts)
        for idx, (analysis, out) in enumerate(zip(analyses, score_outputs)):
            comp = out.outputs[0]
            scores[idx] = extract_tag_score(analysis, tag, comp.text, comp.logprobs or [])
            diags[idx] = letter_probability_diagnostics(comp.logprobs or [])
            pass_tokens[idx][0] += len(getattr(out, "prompt_token_ids", None) or [])
            pass_tokens[idx][1] += len(getattr(comp, "token_ids", None) or [])

    results: dict[int, PairScoreResult] = {}
    owners: dict[int, PairOwner] = {}
    for idx, (i, a, b, swapped, pw_text) in enumerate(owners_list):
        sa, sb = scores_a[idx], scores_b[idx]
        r_a, r_b = (sb, sa) if swapped else (sa, sb)
        da, db = diags_a[idx], diags_b[idx]
        diag_a, diag_b = (db, da) if swapped else (da, db)
        wa, wb = words_list[idx]
        results[i] = PairScoreResult(
            r_a=r_a, r_b=r_b, analysis=analyses[idx],
            prompt_tokens=pass_tokens[idx][0], completion_tokens=pass_tokens[idx][1],
            diag_a=diag_a, diag_b=diag_b, words_a=wa, words_b=wb,
        )
        owners[i] = PairOwner(index=i, a=a, b=b, swapped=swapped, prompt_text=pw_text)
    return results, owners


@dataclass
class TournamentResult:
    """Per active-item outcome of one full single-round PPT tournament
    pass. `win_rate` is the full pool's aggregate Bradley-Terry win-rate
    vector (index-aligned with the pool); `top1` is argmax(win_rate)
    (ties broken toward the lowest pool index, matching both original
    implementations' `key=lambda k: (win_rate[k], -k)`)."""
    win_rate: list[float]
    top1: int
    n_comparisons: int
    first_analysis: dict[int, str]      # pool_index -> first analysis text that mentioned it
    prompt_tokens_delta: int             # tokens spent scoring THIS item across all comparisons
    completion_tokens_delta: int


def run_ppt_tournament(
    pool_size: int, active: list[int], k_pivots: int, rng: random.Random,
    score_batch_fn: Callable[[dict[int, tuple[int, int]]], dict[int, PairScoreResult]],
) -> dict[int, TournamentResult]:
    """The tournament itself: ring pass -> pivot selection -> pivot rounds
    -> Bradley-Terry win-rate aggregation -> argmax, exactly as originally
    inlined in both `verifier_cot.py::run_group` and
    `detection_verifier.py::run_group` (see module docstring — this is a
    verbatim extraction of that shared control flow, not a
    reimplementation). `pool_size == 1` short-circuits to a trivial
    win_rate=[0.5], top1=0 result with zero comparisons, matching both
    originals (nothing to compare a single candidate against).

    `score_batch_fn(pair_by_image) -> {item_index: PairScoreResult}` is
    the caller's own wrapper around `score_pairs_batch` above (so the
    caller controls its own pools/images/vprompts/logging — this function
    only needs to know pool_size/active/k_pivots/rng and how to score a
    round of scheduled pairs)."""
    if pool_size == 1:
        return {
            i: TournamentResult(win_rate=[0.5], top1=0, n_comparisons=0, first_analysis={},
                                 prompt_tokens_delta=0, completion_tokens_delta=0)
            for i in active
        }

    w = {i: [0.0] * pool_size for i in active}
    c = {i: [0] * pool_size for i in active}
    n_comparisons = {i: 0 for i in active}
    first_analysis: dict[int, dict[int, str]] = {i: {} for i in active}
    prompt_tokens_delta = {i: 0 for i in active}
    completion_tokens_delta = {i: 0 for i in active}

    def _accumulate(pair_by_image: dict[int, tuple[int, int]]) -> None:
        result = score_batch_fn(pair_by_image)
        for i in active:
            r = result[i]
            a, b = pair_by_image[i]
            p = bradley_terry(r.r_a, r.r_b)
            w[i][a] += p
            c[i][a] += 1
            w[i][b] += 1.0 - p
            c[i][b] += 1
            n_comparisons[i] += 1
            first_analysis[i].setdefault(a, r.analysis)
            first_analysis[i].setdefault(b, r.analysis)
            prompt_tokens_delta[i] += r.prompt_tokens
            completion_tokens_delta[i] += r.completion_tokens

    # Step 1: ring pass
    rings = {i: ring_cycle(pool_size, rng) for i in active}
    for t in range(pool_size):  # ring_cycle always returns pool_size pairs
        _accumulate({i: rings[i][t] for i in active})

    # Step 2: pivot selection (per item, from ring-pass win rate)
    pivots = {i: select_pivots(w[i], c[i], k_pivots) for i in active}

    # Step 3: pivot rounds
    pivot_pairs = {i: pivot_round_pairs(pool_size, pivots[i]) for i in active}
    n_pivot_pairs = len(next(iter(pivot_pairs.values())))  # uniform across items, see original docstring
    for t in range(n_pivot_pairs):
        _accumulate({i: pivot_pairs[i][t] for i in active})

    out: dict[int, TournamentResult] = {}
    for i in active:
        win_rate = [w[i][k] / c[i][k] if c[i][k] else 0.0 for k in range(pool_size)]
        top1 = max(range(pool_size), key=lambda k: (win_rate[k], -k))
        out[i] = TournamentResult(
            win_rate=win_rate, top1=top1, n_comparisons=n_comparisons[i],
            first_analysis=first_analysis[i],
            prompt_tokens_delta=prompt_tokens_delta[i], completion_tokens_delta=completion_tokens_delta[i],
        )
    return out
