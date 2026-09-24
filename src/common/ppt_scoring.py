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
extract_tag_score below, this module's own expectation-over-logprobs
formula (paper Eq. 3.1). This is the offline-vLLM equivalent of an
OpenAI-compatible server's continue_final_message=True (which is what
llm_verifier's own call_verifier/_score_tags_by_prefill uses when talking
to a server — not directly usable here, since this repo generates via
vLLM's offline batch LLM.generate() API, not a served endpoint).

extract_tag_score used to delegate to the vendored
llm_verifier.fine_grained_reward.extract_score, which folds a letter's
plain/leading-space spellings ("C" / " C") together via MAX. Paper 2 Stage
2 (docs/legacy/stage2_verifier_corrections_paused.md) flagged this as a
candidate cause of the letter-scale's low variance (Paper 1 Appendix C,
Fig. 9) and built letter_probability_diagnostics below to measure it, but
never got real data confirming which direction a fix should go (does the
mass loss disproportionately hit the top-scoring letter, or is it
roughly uniform across letters — see the diagnostics function's own
docstring). extract_tag_score now sums the space-variants instead of
maxing (reimplemented locally rather than patching the vendored
extract_score) — a real, live change to the scoring path, not just
diagnostic. This is NOT confirmed to be strictly better (see the
docstring below) — treat this as the experiment, not a settled fix.
"""
from __future__ import annotations

import math

try:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    StructuredOutputsParams = None  # type: ignore[assignment,misc]

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
    "them — not just whether it names a plausible general category.\n\n"
    "Immediately after this analysis you will be asked to rate EACH "
    "trajectory independently on a 20-point letter scale from A to T, "
    "anchored to how many of the numbered Evaluation Criteria above "
    "that trajectory actually satisfies WITH EVIDENCE CITED FROM ITS "
    "OWN TEXT — not a vague overall impression of correctness:\n"
    "  A-C — satisfies essentially ALL of the criteria, each with "
    "specific supporting evidence. No real gap.\n"
    "  D-H — satisfies MOST of the criteria (clear majority), but at "
    "least one is unaddressed, unclear, or only weakly supported.\n"
    "  I-M — satisfies SOME of the criteria — genuine, cited gaps "
    "outnumber the supported points, even if the overall diagnosis "
    "still sounds plausible.\n"
    "  N-T — satisfies at most ONE criterion, or the cited evidence "
    "actively contradicts the stated diagnosis.\n"
    "Apply this same four-band anchor separately to Trajectory A and "
    "Trajectory B — each trajectory's score reflects how well THAT "
    "trajectory satisfies the criteria, not which one you'd pick as "
    "the winner.\n\n"
    "Before scoring, state in your analysis which band (A-C / D-H / "
    "I-M / N-T) each trajectory falls into and why. If your analysis "
    "finds a real difference in how many criteria the two trajectories "
    "satisfy, THEIR SCORES MUST LAND IN DIFFERENT BANDS — do not "
    "describe a criteria-satisfaction gap between A and B and then "
    "score them within a few letters of each other; let the bands "
    "above drive real separation whenever a genuine difference "
    "exists, not just a tie-break in the same band.\n\n"
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


_LETTER_VALUES = {chr(65 + i): float(20 - i) for i in range(20)}  # A=20 ... T=1, phi() in Eq. 3.1


def extract_tag_score(analysis: str, tag: str, letter_raw: str, logprobs_list) -> float:
    """Eq. 3.1 expectation-over-logprobs score for one forced letter-token
    position, normalized to [0, 1] — same formula the vendored
    llm_verifier.fine_grained_reward.extract_score uses, reimplemented
    locally so we can fold a letter's plain/leading-space spellings
    ("C" / " C") by SUMMING their probabilities instead of the vendored
    function's MAX (see this module's docstring for why, and the caveat
    that this isn't confirmed to be strictly better).

    `analysis`/`tag` are unused for the score itself now (kept in the
    signature for caller compatibility — ppt_orchestration.py and
    verifier_lmv.py both call this positionally). `letter_raw` is the
    single fallback used only if `logprobs_list` is somehow empty (not
    expected in practice — score_prefill_params always requests
    logprobs=20, so this is a defensive fallback, not the normal path,
    mirroring the vendored function's own text-fallback behavior).
    `logprobs_list` is a vLLM CompletionOutput's `.logprobs` for the single
    forced token (a list with exactly one {token_id: Logprob} dict, from
    score_prefill_params's max_tokens=1)."""
    lp_dict = logprobs_list[0] if logprobs_list else {}
    probs_sum: dict[str, float] = {}
    for lp in lp_dict.values():
        tok = lp.decoded_token.strip()
        if tok not in _LETTER_VALUES:
            continue
        probs_sum[tok] = probs_sum.get(tok, 0.0) + math.exp(lp.logprob)

    if not probs_sum:
        raw_val = _LETTER_VALUES.get(letter_raw.strip())
        return (raw_val - 1) / 19 if raw_val is not None else 0.5

    total_p = sum(probs_sum.values())
    expected = sum(_LETTER_VALUES[t] * p for t, p in probs_sum.items()) / total_p
    return (expected - 1) / 19


def letter_probability_diagnostics(logprobs_list) -> dict:
    """Diagnostic-only breakdown of the raw per-letter probability mass at
    a forced score-token position (Paper 2 Stage 2 — see
    docs/legacy/stage2_verifier_corrections_paused.md). Read-only: never
    used to score anything itself, so it can't change tournament behavior.

    `extract_tag_score` above now SUMS a letter's plain/leading-space
    spellings ("C" / " C") — `probs_sum` here mirrors that live scoring
    path; `probs_max` is the vendored llm_verifier behavior this replaced.
    Kept computing both (not just the live one) because the sum-vs-max
    question was never fully settled: whether summing actually helps
    depends on whether the top-scoring letter specifically loses more mass
    to the max operation than lower letters do, not just the average gap
    size — verifier-score-analysis's max_vs_sum_mass_gap only reports the
    average, so a real before/after judged-accuracy comparison (not just
    this mass-gap number) is what actually tells you whether summing
    helped, hurt, or was a wash on a given run."""
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
