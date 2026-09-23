"""Shared PPT-style dual-trajectory scoring (arXiv:2607.05391 Section 3.2)
— two-pass teacher-forced/prefill extraction. Corrected from an earlier
single free-generation version that collapsed onto the scale's first
letter ("A") ~96% of the time on a real feasibility test (see
docs/verifier_score_belief_update.md Section 15's note, and
src/legacy/verifier_ppt/run_verifier_ppt.py's module docstring, archived
prototype, for the original test that found this) — the model's unconstrained top-20
logprobs at a freely-reached tag position usually aren't dominated by the
20 target scale letters at all.

Used by:
  - src/verifier_lmv.py — the full PPT tournament baseline (ring-pass +
    pivot scheduling, static win-mass aggregation, no adaptive stopping).
  - src/verifier_cot.py — the belief-tracking pairwise duel (--baseline
    belief). Same scoring mechanism; the two conditions differ in what
    happens AFTER a pairwise comparison is scored (docs/
    verifier_score_belief_update.md Sections 10/11), not in how it's
    scored — this module is deliberately shared, not duplicated a second
    time, since a subtle bug here (see the fix above) would otherwise need
    fixing twice.

Mechanism: pass 1 free-generates the comparative analysis only
(temperature=1.0, no logprobs needed — any tag content it writes here is
discarded, not trusted). Pass 2 (once per tag) continues that EXACT prompt
text with the analysis plus a prefilled tag appended, and forces exactly
one more token under a grammar mask constraining it to the 20 scale
letters (+ their leading-space tokenization variants) via
SamplingParams(structured_outputs=StructuredOutputsParams(choice=...)).
The resulting top-20 logprobs are then already a distribution over the
scale itself, not competing against the model's full vocabulary — read via
llm_verifier.fine_grained_reward.extract_score (the vendored package's own
expectation-over-logprobs formula, paper Eq. 3.1). This is the
offline-vLLM equivalent of an OpenAI-compatible server's
continue_final_message=True (which is what the vendored package's own
call_verifier/_score_tags_by_prefill uses when talking to a server — not
directly usable here, since this repo generates via vLLM's offline batch
LLM.generate() API, not a served endpoint).
"""
from __future__ import annotations

import math

try:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    StructuredOutputsParams = None  # type: ignore[assignment,misc]

try:
    from llm_verifier.fine_grained_reward import extract_score
    HAS_LMV = True
except ImportError:
    HAS_LMV = False
    extract_score = None  # type: ignore[assignment]

# The 20-letter scale plus a leading-space variant of each letter — mirrors
# llm_verifier's own _score_tags_by_prefill comment: "Some models (e.g.
# Qwen VL) put nearly all their mass on the letter WITH a leading space
# after the prefilled '>'; allow both spellings so the grammar mask keeps
# the real distribution." Constrains pass 2's single generated token to
# exactly these 40 candidates.
SCORE_LETTERS = [chr(65 + i) for i in range(20)] + [f" {chr(65 + i)}" for i in range(20)]


def _truncate_keep_answer(trajectory: str, limit: int = 500) -> str:
    """Truncate to `limit` chars, but never cut off a trailing
    <answer>...</answer> tag — a plain trajectory[:limit] silently dropped
    the final answer for any trajectory whose reasoning ran long, leaving
    the verifier scoring a trajectory it could never actually see the
    conclusion of (found empirically: ~2% of pairs in a real run). If the
    tag starts past `limit`, keep the head up to `limit` MINUS the tag's
    own length, then append the tag itself, so the total stays roughly
    bounded while the answer is always present."""
    if len(trajectory) <= limit:
        return trajectory
    tag_start = trajectory.find("<answer>")
    if tag_start == -1 or tag_start < limit:
        return trajectory[:limit]
    answer_tag = trajectory[tag_start:]
    head_budget = max(limit - len(answer_tag), 0)
    return trajectory[:head_budget].rstrip() + "\n...\n" + answer_tag


# Fallback only for callers that don't pass a `template` (kept in sync with
# src/config/verifier_prompts.yaml's pairwise_prompt — that YAML file is
# the actual source of truth callers should load and pass in; see its
# header comment for the full rationale behind this exact wording).
_DEFAULT_PAIRWISE_TEMPLATE = (
    "You are an expert agricultural scientist reviewer. You will see a "
    "task description and two trajectories.\n\n"
    "Evaluation Criteria: {rubric}\n\n"
    "Task: {question}\n\n"
    "Trajectory A: {trajectory_a}\n"
    "Trajectory B: {trajectory_b}\n\n"
    "Write 2-4 sentences comparing what each trajectory actually "
    "diagnoses and how well it matches the visual evidence and the "
    "evaluation criteria above. Name the specific difference between "
    "A and B — do not just restate that both look reasonable, and do "
    "not treat them as tied unless one is genuinely indistinguishable "
    "from the other in correctness. If either trajectory's stated "
    "answer has a visually similar but distinct condition it could "
    "easily be confused with (e.g. Black Sigatoka vs. Yellow "
    "Sigatoka, or two look-alike species), explicitly check whether "
    "the trajectory's cited evidence actually distinguishes between "
    "them — not just whether it names a plausible general category. "
    "Immediately after this analysis you will be asked to rate each "
    "trajectory on a 20-point letter scale from A to T (A = clearly and "
    "completely correct, T = clearly and completely incorrect, with the "
    "letters in between spanning that range), so make sure your analysis "
    "actually justifies whatever gap (or lack of gap) you're about to "
    "score.\n\n"
    "Analysis:"
)


def pairwise_scoring_prompt(
    rubric: str, question: str, trajectory_a: str, trajectory_b: str,
    template: str | None = None,
) -> str:
    """The dual-trajectory scoring prompt (PPT paper, arXiv:2607.05391
    Section 3.2 — adapted; see src/config/verifier_prompts.yaml's
    pairwise_prompt header comment for exactly what changed and why).
    `template` should be src/config/verifier_prompts.yaml's
    "pairwise_prompt" entry, loaded once by the caller and passed straight
    through — that YAML file is the editable, citable record of this
    prompt; `_DEFAULT_PAIRWISE_TEMPLATE` here is only a fallback for a
    caller that hasn't been wired up to load it yet. Deliberately does
    NOT include the <score_A>/<score_B> tags — those are supplied only by
    build_score_prompt's pass-2 forced continuation, never sampled
    freely (see the YAML comment for why that matters)."""
    tmpl = template if template is not None else _DEFAULT_PAIRWISE_TEMPLATE
    return tmpl.format(
        rubric=rubric, question=question,
        trajectory_a=_truncate_keep_answer(trajectory_a),
        trajectory_b=_truncate_keep_answer(trajectory_b),
    )


def build_score_prompt(base_prompt: dict, analysis: str, tag: str) -> dict:
    """Continue `base_prompt`'s exact text with the analysis plus a
    prefilled score tag."""
    return {
        "prompt": base_prompt["prompt"] + analysis + f"\n{tag}",
        "multi_modal_data": base_prompt["multi_modal_data"],
    }


def analysis_before_tags(text: str) -> str:
    """Truncate a free-generated response at the first score tag it
    attempted to write, if any — only the analysis prefix is kept; the
    sampled tag content is discarded and re-generated under a grammar mask
    in pass 2, since an unconstrained free-generation tag isn't reliable
    on its own."""
    positions = [i for i in (text.find("<score_A>"), text.find("<score_B>")) if i != -1]
    idx = min(positions) if positions else len(text)
    return text[:idx].rstrip()


def score_prefill_params(temperature: float = 1.0) -> "SamplingParams":
    return SamplingParams(
        temperature=temperature, max_tokens=1, logprobs=20,
        structured_outputs=StructuredOutputsParams(choice=SCORE_LETTERS),
    )


def extract_tag_score(analysis: str, tag: str, letter_raw: str, logprobs_list) -> float:
    """Reconstruct the (text, tokens, position_logprobs) shape
    extract_score expects, matching llm_verifier's own
    _score_tags_by_prefill return convention exactly (so the shared
    library function needs no changes), and extract the Eq. 3.1
    expectation-over-logprobs score for this one tag. `logprobs_list` is a
    vLLM CompletionOutput's `.logprobs` for the single forced token
    (a list with exactly one {token_id: Logprob} dict, from
    score_prefill_params's max_tokens=1)."""
    lp_dict = logprobs_list[0] if logprobs_list else {}
    alts = [(lp.decoded_token, lp.logprob) for lp in lp_dict.values()] if lp_dict else [(letter_raw, 0.0)]
    closing = "</" + tag[1:]
    synth_text = f"{analysis}\n{tag}{letter_raw}{closing}"
    synth_tokens = [f"\n{tag}", letter_raw, closing]
    synth_position_logprobs = [[(f"\n{tag}", 0.0)], alts, [(closing, 0.0)]]
    return extract_score(synth_text, synth_tokens, synth_position_logprobs, tag)


def letter_probability_diagnostics(logprobs_list) -> dict:
    """Diagnostic-only breakdown of the raw per-letter probability mass at
    a forced score-token position (Paper 2 Stage 2 — see
    docs/legacy/stage2_verifier_corrections_paused.md). Read-only: never
    used to score anything, so it can't change tournament behavior —
    extract_tag_score above remains the sole authority on the actual score.

    llm_verifier.fine_grained_reward.extract_score (called via
    extract_tag_score) folds a letter's plain and leading-space spellings
    ("C" / " C") together by taking the MAX of their two probabilities,
    not the sum — see extract_tag_score's docstring /
    docs/legacy/stage2_verifier_corrections_paused.md's Step 2 for why
    that's a plausible contributor to the letter-scale's low variance
    (Paper 1 Appendix C, Figure 9). This returns both aggregations side by side so
    verifier-score-analysis can measure how much probability mass the max
    convention is actually discarding on real data, rather than assuming
    it: `probs_max` mirrors what extract_score itself uses, `probs_sum`
    is the candidate alternative."""
    lp_dict = logprobs_list[0] if logprobs_list else {}
    letters = {chr(65 + i) for i in range(20)}
    probs_max: dict[str, float] = {}
    probs_sum: dict[str, float] = {}
    for lp in lp_dict.values():
        tok = lp.decoded_token.strip()
        if tok not in letters:
            continue
        p = math.exp(lp.logprob)
        probs_max[tok] = max(probs_max.get(tok, 0.0), p)
        probs_sum[tok] = probs_sum.get(tok, 0.0) + p
    return {"probs_max": probs_max, "probs_sum": probs_sum}
