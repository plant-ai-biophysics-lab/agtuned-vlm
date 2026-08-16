"""Shared PPT-style dual-trajectory scoring (arXiv:2607.05391 Section 3.2)
— two-pass teacher-forced/prefill extraction. Corrected from an earlier
single free-generation version that collapsed onto the scale's first
letter ("A") ~96% of the time on a real feasibility test (see
docs/verifier_score_belief_update.md Section 15's note, and
experiments/verifier_ppt/run_verifier_ppt.py's module docstring for the
original test that found this) — the model's unconstrained top-20
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


def pairwise_scoring_prompt(rubric: str, question: str, trajectory_a: str, trajectory_b: str) -> str:
    """The PPT paper's own consolidated dual-trajectory template (Section
    3.2) — one prompt carrying both trajectories and both score tags
    together, not two separate calls."""
    return (
        "You are an expert agricultural scientist reviewer. You will see a "
        "task description and two trajectories.\n\n"
        f"Evaluation Criteria: {rubric}\n\n"
        f"Task: {question}\n\n"
        f"Trajectory A: {trajectory_a[:500]}\n"
        f"Trajectory B: {trajectory_b[:500]}\n\n"
        "Carefully analyze each trajectory, then provide your final scores:\n"
        "<score_A>LETTER_A_TO_T</score_A>\n"
        "<score_B>LETTER_A_TO_T</score_B>\n\n"
        "Rating Rules: rate correctness on a 1-20 scale based on the evaluation "
        "criteria (1 = incorrect, 10 = borderline, 20 = correct). The scale is "
        "encoded as letters A (=20, best) through T (=1, worst)."
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
