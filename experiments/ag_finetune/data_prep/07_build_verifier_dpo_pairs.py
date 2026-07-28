"""
Stage 3b — Build DPO pairs for the per-statement verifier from real CoT failures.

Unlike 05_build_verifier_data.py (which synthesizes hard negatives by asking a
model to hallucinate wrong answers), this stage sources its "rejected" side
from REAL model output: actual confidently-wrong statements the vision model
produced during CoT evaluation, pulled from
outputs/cot_step_analysis/swap_order/ckpt_swap_judged.csv (already generated
by src/cot_step_analysis.py --swap_order).

This deliberately implements two findings from
outputs/verifier_analysis/verifier_findings.pdf (see that report for full
evidence/charts):

  Finding 1 (presupposition_confabulation): CoT questions that presuppose a
  fact ("which parts show symptoms") make the model invent supporting
  "evidence" on genuinely healthy tissue. Report's own worked example:
  arabica_coffee_leaf_disease_classification, GT=Healthy, step 2 confabulates
  "chlorotic lesions" and diagnoses Coffee Leaf Rust. The single largest, most
  fixable failure mode per the report.

  Finding 3 (confident_misdiagnosis): assertive language ("clearly",
  "characteristic of", "definitively") is ANTI-correlated with correctness —
  the model reaches for confident phrasing exactly when it's wrong. On
  specific (non-catch-all) ground-truth labels, "commits" stance answers are
  still wrong 70.8% of the time. A rejected example built from a random wrong
  answer is a weaker signal than one built from a confidently-wrong one.

For both pools, the "rejected" verdict is a fixed, deterministic template
representing the naive failure mode itself (a verifier that rubber-stamps
confident-sounding statements as SUPPORTED without checking evidence) - this
is not fabricated by an LLM, it's the literal behavior the report measures.
The "chosen" verdict is also deterministic, built from the row's own known
ground truth - standard supervised-training-label construction, not a leak,
since the model being trained never sees the ground truth at inference.

Output format matches training/stage2_dpo.py's expected schema exactly
(chosen_prompt / chosen_response / rejected_response), but is NOT written to
stage2_chi_dpo_pairs.jsonl (that name is reserved for the distinct, currently-
empty Chi-Metrics DPO source - see 02_filter_chi_metrics.py). Pass this file's
path explicitly via --data when training:

  uv run python experiments/ag_finetune/training/stage2_dpo.py \
    --model outputs/checkpoints/stage1_sft \
    --data  experiments/ag_finetune/outputs/stage7_verifier_dpo_pairs.jsonl \
    --output_dir outputs/checkpoints/stage7_verifier_dpo

Outputs:
  outputs/stage7_verifier_dpo_pairs.jsonl
  outputs/stage7_verifier_dpo_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/07_build_verifier_dpo_pairs.py \
    --swap_path outputs/cot_step_analysis/swap_order/ckpt_swap_judged.csv \
    --rotation 0 \
    --limit_per_pool 3000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from _progress import Progress, fmt_duration, stage_banner

csv.field_size_limit(sys.maxsize)

OUT_DIR = Path(__file__).parent.parent / "outputs"
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_SWAP_PATH = REPO_ROOT / "outputs" / "cot_step_analysis" / "swap_order" / "ckpt_swap_judged.csv"

# ── Reused from src/verifier_analysis.py's marker/stratum logic ────────────
# (duplicated, not imported, to keep this script self-contained per repo
# convention — data_prep scripts don't import from src/)

_NONSPECIFIC_RE = re.compile(
    r"healthy|normal|fresh|no[_\s-]?disease|nodisease|uninfected|good", re.IGNORECASE
)
_NONSPECIFIC_EXACT = {
    "negative", "grass", "broadleaf", "soil", "background",
    "other", "unknown", "none", "nan",
}

COMMITMENT_MARKERS = [
    "characteristic of", "diagnostic of", "clearly", "definitively",
    "pathognomonic", "classic symptom", "hallmark", "unmistakab",
]
HEDGE_MARKERS = [
    "most likely", "appears to be", "suggests", "may be", "might be",
    "could be", "likely a", "probably", "possibly", "consistent with",
]
ABSTENTION_MARKERS = [
    "impossible to", "not possible to", "cannot provide", "cannot be",
    "i cannot", "unable to", "cannot definitively", "cannot make",
    "no definitive", "cannot determine", "impossible to determine",
]

_PRESUPPOSING_QUESTION_RE = re.compile(
    r"which\s+(specific\s+)?(parts?|areas?|regions?)\b.*\b(showing|show|exhibit)",
    re.IGNORECASE,
)


def is_nonspecific_class(gt: str) -> bool:
    g = (gt or "").strip().lower()
    return g in _NONSPECIFIC_EXACT or bool(_NONSPECIFIC_RE.search(g))


def stance(text: str) -> str:
    tl = text.lower()
    if any(p in tl for p in ABSTENTION_MARKERS):
        return "abstains"
    if any(p in tl for p in HEDGE_MARKERS):
        return "hedges"
    return "commits"


def is_presupposing_question(q: str) -> bool:
    return bool(_PRESUPPOSING_QUESTION_RE.search(q or ""))


def split_all_steps(reasoning: str) -> list[tuple[str, str]]:
    """Parse a cumulative reasoning blob into ordered (question, answer)
    pairs — same logic as verifier_analysis_transcripts.py's split_step_qa."""
    blocks = re.split(r"\n\nQ:\s*", reasoning.strip())
    pairs = []
    for i, block in enumerate(blocks):
        text = block.strip()
        if i == 0:
            if text.startswith("Q: "):
                text = text[3:]
            elif text.startswith("Q:"):
                text = text[2:]
        if "\nA: " in text:
            q, a = text.split("\nA: ", 1)
        elif "\nA:" in text:
            q, a = text.split("\nA:", 1)
        else:
            q, a = text, ""
        pairs.append((q.strip(), a.strip()))
    return pairs


# ── Verifier prompt template (from src/verifier_report.py's VERIFIER_PROMPT,
# duplicated here as a build function rather than imported, since
# verifier_report.py pulls in heavy PDF-rendering deps at module level and
# data_prep scripts stay dependency-light) ──────────────────────────────────

def build_verifier_prompt(verified_history: str, step_question: str, statement: str) -> str:
    return f"""\
You are a specialist agricultural verifier - a small model fine-tuned on
agricultural disease/pest/weed data - checking ONE statement from a chain of
reasoning about an image. Verify only the statement given, not the whole
chain. The step that produced it may be from a fixed diagnostic question
(guided, e.g. cot_yaml) or the model's own self-generated train of thought
(e.g. self_cot) - treat both the same way.

Image: <image>
Verified history (statements already checked and confirmed SUPPORTED only - \
does NOT include anything flagged below): {verified_history or "(none yet - this is the first step)"}
This step's question or reasoning prompt (if any): {step_question}
Statement to verify: {statement}

Apply these checks in order. Do not skip to a verdict.

1. PRESUPPOSITION CHECK - decide this before anything else.
   Does the step's question, or the statement itself, assume something not
   yet established (e.g. "which parts show symptoms" presupposes symptoms
   exist)? If nothing in the verified history confirms the premise, the
   correct move is to say the premise is unconfirmed, not to invent content
   that satisfies it.

2. EVIDENCE CHECK - is this specific statement visible in the image?
   Name the specific pixels/region that support it. If you cannot point to a
   region, the statement is unsupported regardless of how confident it sounds.

3. CONSISTENCY CHECK.
   Does this statement contradict anything in the verified history? The
   chain cannot hold two incompatible claims - flag the conflict rather than
   silently letting the newer statement overwrite the earlier one.

4. CONFIDENCE-LANGUAGE WARNING.
   Ignore words like "clearly", "characteristic of", "definitively" when
   judging - in this model they correlate with WRONG statements, not right
   ones. Judge only the visible evidence.

5. IMAGE-QUALITY CHECK.
   If resolution, focus, or crop make this specific statement unverifiable,
   say so. INSUFFICIENT_EVIDENCE is a correct outcome, not a failure.

Return JSON:
{{"verdict": "SUPPORTED" | "UNSUPPORTED" | "PRESUPPOSES_UNCONFIRMED" | "CONTRADICTS_PRIOR" | "INSUFFICIENT_EVIDENCE",
 "corrected_statement": "<neutral restatement if UNSUPPORTED/PRESUPPOSES, or null>",
 "visual_evidence": "<region and what is visible, or null>",
 "halt_chain": true | false,
 "confidence": "high" | "medium" | "low",
 "reasoning": "<1-2 sentences>"}}"""


REJECTED_VERDICT = {
    "verdict": "SUPPORTED",
    "corrected_statement": None,
    "visual_evidence": "consistent with the description given in the statement",
    "halt_chain": False,
    "confidence": "high",
    "reasoning": ("The statement is phrased with clear, definitive language and appears "
                  "consistent with the described visual features."),
}


def build_chosen_verdict_presupposition(ground_truth: str) -> dict:
    return {
        "verdict": "PRESUPPOSES_UNCONFIRMED",
        "corrected_statement": ("No symptoms have been independently confirmed in the "
                                 "verified history; the question presupposes their presence. "
                                 "State that plainly rather than describing symptoms that "
                                 "satisfy the premise."),
        "visual_evidence": None,
        "halt_chain": True,
        "confidence": "high",
        "reasoning": (f"Ground truth for this image is '{ground_truth}' - the premise that "
                       "symptoms are present is unconfirmed and should not be accepted "
                       "just because the question assumes it."),
    }


def build_chosen_verdict_misdiagnosis(ground_truth: str) -> dict:
    return {
        "verdict": "UNSUPPORTED",
        "corrected_statement": (f"This specific claim is not supported by the visible "
                                 f"evidence; the confirmed diagnosis for this specimen is "
                                 f"'{ground_truth}', not what is stated."),
        "visual_evidence": None,
        "halt_chain": True,
        "confidence": "high",
        "reasoning": ("The statement uses confident, definitive language, but confident "
                      "phrasing in this model correlates with incorrect claims, not correct "
                      "ones - evidence should be checked independently of tone."),
    }


def build_pair(row: dict, failure_mode: str) -> dict:
    reasoning = row.get("reasoning", "") or ""
    steps = split_all_steps(reasoning)
    step_question, statement = steps[-1] if steps else ("", row.get("prediction", ""))
    verified_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in steps[:-1])

    chosen_prompt = build_verifier_prompt(verified_history, step_question, statement)
    gt = row.get("ground_truth", "")
    chosen_verdict = (build_chosen_verdict_presupposition(gt) if failure_mode == "presupposition_confabulation"
                      else build_chosen_verdict_misdiagnosis(gt))

    return {
        "stage": "verifier_dpo",
        "source": "cot_step_analysis_swap_order",
        "failure_mode": failure_mode,
        "agml_dataset": row.get("agml_dataset", ""),
        "task_type": row.get("task_type", ""),
        "image_name": row.get("image_name", ""),
        "class_label": row.get("class_label", ""),
        "ground_truth": gt,
        "step_idx": row.get("step_idx", ""),
        "step_question": step_question,
        "statement": statement,
        "chosen_prompt": chosen_prompt,
        "chosen_response": json.dumps(chosen_verdict),
        "rejected_response": json.dumps(REJECTED_VERDICT),
    }


def load_candidates(swap_path: Path, rotation: str) -> tuple[list[dict], list[dict]]:
    presupposition_pool: list[dict] = []
    misdiagnosis_pool: list[dict] = []

    with open(swap_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "cumulative" or row.get("rotation") != rotation:
                continue
            jc = row.get("judged_correct", "")
            if jc == "" or int(jc) != 0:
                continue  # only wrong answers are candidates for rejected-side sourcing

            pred = row.get("prediction", "") or ""
            gt = row.get("ground_truth", "")
            nonspecific = is_nonspecific_class(gt)

            reasoning = row.get("reasoning", "") or ""
            steps = split_all_steps(reasoning)
            step_question = steps[-1][0] if steps else ""

            if nonspecific and is_presupposing_question(step_question):
                presupposition_pool.append(row)
            elif not nonspecific and stance(pred) == "commits":
                misdiagnosis_pool.append(row)

    return presupposition_pool, misdiagnosis_pool


def main() -> None:
    stage_banner(7, "Build verifier DPO pairs from real CoT failures")
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_path", default=str(DEFAULT_SWAP_PATH))
    parser.add_argument("--rotation", default="0",
                        help="Which swap-order rotation to source from (0 = natural, unrotated order)")
    parser.add_argument("--limit_per_pool", type=int, default=3000,
                        help="Cap on examples per failure_mode pool (balances the two pools)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(OUT_DIR / "stage7_verifier_dpo_pairs.jsonl"))
    args = parser.parse_args()

    swap_path = Path(args.swap_path)
    if not swap_path.exists():
        print(f"ERROR: Run cot_step_analysis.py --swap_order first. Missing: {swap_path}")
        return

    print(f"Scanning {swap_path.name} (rotation={args.rotation})...")
    presupposition_pool, misdiagnosis_pool = load_candidates(swap_path, args.rotation)
    print(f"  presupposition_confabulation candidates: {len(presupposition_pool):,}")
    print(f"  confident_misdiagnosis candidates:        {len(misdiagnosis_pool):,}")

    rng = random.Random(args.seed)
    if len(presupposition_pool) > args.limit_per_pool:
        presupposition_pool = rng.sample(presupposition_pool, args.limit_per_pool)
    if len(misdiagnosis_pool) > args.limit_per_pool:
        misdiagnosis_pool = rng.sample(misdiagnosis_pool, args.limit_per_pool)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = {"presupposition_confabulation": 0, "confident_misdiagnosis": 0}

    with open(out_path, "w") as f:
        prog = Progress(total=len(presupposition_pool) + len(misdiagnosis_pool),
                        label="DPO pairs", every=500)
        for row in presupposition_pool:
            entry = build_pair(row, "presupposition_confabulation")
            f.write(json.dumps(entry) + "\n")
            n_written["presupposition_confabulation"] += 1
            prog.update(1)
        for row in misdiagnosis_pool:
            entry = build_pair(row, "confident_misdiagnosis")
            f.write(json.dumps(entry) + "\n")
            n_written["confident_misdiagnosis"] += 1
            prog.update(1)

    stats = {
        "total": sum(n_written.values()),
        "by_failure_mode": n_written,
        "rotation": args.rotation,
        "source": str(swap_path),
    }
    (OUT_DIR / "stage7_verifier_dpo_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. {stats['total']:,} DPO pairs → {out_path}")
    print(json.dumps(n_written, indent=2))


if __name__ == "__main__":
    main()
