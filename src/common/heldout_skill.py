"""
Paper 2, Stage 3 (docs/stage3_context_memory.md) — the ONE place that
builds the skill-memory system-prompt wrapper, shared by every caller
that injects it (heldout_zeroshot.py's own Step 4 ablation,
vision_baseline.py's/verifier_cot.py's Step 5 real injection). Factored
out specifically so the wrapper's exact wording can never drift between
what the ablation validated and what production actually uses — the
wording itself was empirically tuned twice against real regressions
(see build_skill_system_prompt's own docstring), so a caller rolling its
own slightly-different wrapper text would be injecting something never
actually tested.

Real ablation history (held-out eval split, n=199, RedHatAI/gemma-4-31B-it-FP8-dynamic):
  v1 (no instruction at all): delta +0.0101 — not a real signal. The
    model borrowed the skill file's own CATEGORY-level phrasing
    ("Chewing Insects:", "Sucking Pests:") instead of the specific
    diagnosis name (e.g. "Fall armyworm" -> "Chewing pests"), which the
    hybrid judge template explicitly penalizes as not matching a
    specific reference class — genuine fixes almost exactly canceled
    out by this regression.
  v2 (specificity instruction added): delta +0.0503 (18 wrong->correct,
    8 correct->wrong). Two failure modes persisted: (a) bare category
    words still leaked through sometimes despite being told not to, and
    (b) a NEW regression appeared — over-diagnosing genuinely healthy
    plants (e.g. "Healthy" -> "Nitrogen Deficiency"), the opposite bias
    from the one this whole stage started from.
  v3 (forbidden words named explicitly + healthy counter-instruction):
    delta +0.0503 again (same net, but 12 wrong->correct / 2
    correct->wrong this time — fewer of BOTH), zero bare-category
    answers, healthy over-diagnosis eliminated. Remaining 2 regressions
    were specific-vs-specific confusions (e.g. "Mosaic virus" ->
    "Watermelon Leaf Curl Virus"), not instruction-following failures.
  v4 (skill_memory.md's own "Common Diseases..." reference section
    removed — that section repeated some of the same generic category
    labels, e.g. "Chewing Pests:", "Sucking Pests:", as standalone
    entries): delta +0.0603 (14 wrong->correct, 2 correct->wrong) — a
    further real improvement. THIS is the current config.
  wrapper-only (the two firm rules below, injected with NO skill_memory
    content at all — "baseline + wrapper instructions"): a Step 4
    ablation added specifically to isolate how much of v2/v3/v4's gain
    comes from the two instructions themselves (specificity,
    don't-over-diagnose-healthy) versus the skill file's actual lesson
    content. See build_wrapper_only_system_prompt below and
    docs/stage3_context_memory.md's Step 4 section for the result.
"""
from __future__ import annotations

from pathlib import Path


def build_wrapper_only_system_prompt(base_system_prompt: str) -> str:
    """The SAME two firm rules build_skill_system_prompt injects, with NO
    skill_memory content appended and no "Reference material... below"
    framing (which would dangle, pointing at lessons that don't exist
    here) — "baseline + wrapper instructions," the ablation condition
    that isolates whether the two instructions alone (independent of any
    actual lesson content) account for some of v2/v3/v4's gain. Always
    applied (unlike build_skill_system_prompt, this one has no falsy-path
    passthrough — a caller wiring this in has already decided to use
    it)."""
    return (
        f"{base_system_prompt}\n\nTwo firm rules for your diagnosis answer:\n"
        f"1. Always give the MOST SPECIFIC diagnosis name you can (e.g. \"Fall armyworm\", "
        f"\"Diamondback moth\", \"Chili Leaf Curl Virus\") — NEVER answer with a bare category "
        f"word like \"Pest\", \"Disease\", \"Deficiency\", \"Chewing pests\", \"Sucking pests\", "
        f"or \"Viral infection\" on its own.\n"
        f"2. Do not over-diagnose a plant that actually looks healthy just because you're "
        f"actively looking for problems — \"Healthy\" is a valid, common answer, and reading a "
        f"problem into a healthy-looking plant is just as wrong as missing a real one."
    )


def build_skill_system_prompt(base_system_prompt: str, skill_memory_path: str) -> str:
    """base_system_prompt + skill_memory_path's own file content, wrapped
    in the two rules the real ablation above found necessary. Returns
    base_system_prompt UNCHANGED if skill_memory_path is falsy, so a
    caller can always call this unconditionally
    (build_skill_system_prompt(system_prompt, args.skill_memory_path))
    rather than branching on the flag itself at every call site."""
    if not skill_memory_path:
        return base_system_prompt
    skill_memory = Path(skill_memory_path).read_text(encoding="utf-8").strip()
    return (
        f"{base_system_prompt}\n\nReference material — lessons learned from reviewing past crop "
        f"diagnosis attempts on similar images. Use it to inform your answer, but still base "
        f"your diagnosis on what's actually visible in THIS image, under two firm rules:\n"
        f"1. Always give the MOST SPECIFIC diagnosis name you can (e.g. \"Fall armyworm\", "
        f"\"Diamondback moth\", \"Chili Leaf Curl Virus\") — NEVER answer with a bare category "
        f"word like \"Pest\", \"Disease\", \"Deficiency\", \"Chewing pests\", \"Sucking pests\", "
        f"or \"Viral infection\" on its own, and never answer with one of this reference "
        f"material's own section headers, even if that's the language it uses to organize the "
        f"lessons below.\n"
        f"2. Do not over-diagnose a plant that actually looks healthy just because you're "
        f"actively looking for problems — \"Healthy\" is a valid, common answer, and reading a "
        f"problem into a healthy-looking plant is just as wrong as missing a real one.\n\n"
        f"{skill_memory}"
    )


def build_detection_skill_system_prompt(base_system_prompt: str, skill_memory_path: str) -> str:
    """Detection analog of build_skill_system_prompt, for
    heldout_detection_zeroshot.py's own Step 4 ablation
    (docs/stage4_detection_context_memory.md). Returns base_system_prompt
    UNCHANGED if skill_memory_path is falsy, same unconditional-call
    convention as the classification version.

    Deliberately NO "firm rules" wrapper the way the classification
    version has — those two rules (specificity, don't-over-diagnose-
    healthy) were empirically tuned against REAL measured regressions in
    4 rounds of classification-specific ablation (v1-v4, see this
    module's own docstring); nothing equivalent has been validated for
    detection's box-prediction task yet, so inventing box-prediction
    "firm rules" here would be injecting untested wording, exactly the
    failure mode this module exists to prevent. Plain reference-material
    framing only — if a real detection ablation run finds its own
    regression pattern worth a firm rule, add it here then, grounded in
    that evidence, not pre-emptively."""
    if not skill_memory_path:
        return base_system_prompt
    skill_memory = Path(skill_memory_path).read_text(encoding="utf-8").strip()
    return (
        f"{base_system_prompt}\n\nReference material — lessons learned from reviewing past object "
        f"detection box predictions on similar images. Use it to inform your predictions, but "
        f"still base them on what's actually visible in THIS image.\n\n{skill_memory}"
    )
