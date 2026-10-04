"""
Paper 2, Stage 3 — top3 verifier_rubric correct-and-reverify. Classification
analog of detection_box_refine.py's mechanism, applied to a
--verifier_type top3 --top3_mode verifier_rubric checkpoint (already
graded once — this script never re-does round 1's generation/evidence-
fill/scoring, it only acts on what's already there).

For each row, look at the winning candidate's own criterion_scores
(already in the input --ckpt): if EVERY criterion scored >= 3, the row
passes through unchanged (no extra model/verifier calls at all, and its
existing judged_correct is kept — nothing about it changed, so there's
no reason to re-judge it). If ANY criterion scored <= 2, one corrective
round runs:

  1. Correction call (shown to the MODEL being tested, not the
     verifier) — the SAME evidence block the verifier used in round 1
     (parsed back out of the input checkpoint's own verifier_feedback
     column, never re-derived), exactly which criteria failed (with
     their own rubric question text), and the other round-1 candidates
     that lost. The model can keep its original answer, switch to a
     losing candidate, or give something new — one corrective
     <answer></answer> back.
  2. Re-scoring call — the corrected answer graded against that SAME
     evidence, via the identical verifier_score_against_evidence_prompt
     round 1 used.

This whole cycle repeats up to --max_rounds (default 2 — i.e. round 1 is
the ALREADY-DONE input checkpoint's own grading, so the default allows
exactly ONE corrective round before a still-failing row is kept as-is
and flagged "exhausted"). A row whose correction passes before
--max_rounds is reached stops early ("passed").

round-1-only columns (candidate_predictions, candidate_verifier_scores,
answer_in_pool, verifier_picked_correct) are passed through UNCHANGED —
these describe the original 3-candidate pool, which correction doesn't
regenerate, so verifier-cot-qc-plot-top3's pool_summary chart continues
to reflect round 1 only, exactly as before, with no script changes
needed on that side.

Only rows that actually get corrected are re-judged (run_judge_phase,
same shared mechanism every other stage uses) — a passing row's
judged_correct is left untouched, since nothing about it changed.

Usage:
    uv run python main.py verifier-cot-top3-refine \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --ckpt outputs_final/verifier_cot_top3_verifier_rubric_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv \\
        --output_dir outputs_final/verifier_cot_top3_verifier_rubric_refine_smoketest/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import re

from PIL import Image as PILImage

from src.common.chat_prompt import render_chat_template
from src.common.dataset_config import datasets_by_name
from src.common.io import load_prompts
from src.common.metrics import (
    contains_match, extract_answer_tag, extract_channel_thought, resolve_class, rouge_scores_per_sample,
)
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, run_judge_phase, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.progress import ProgressBar
from src.verifier_cot import (
    VERIFIER_CKPT_FIELDS, _CRITERION_TAG_RE, _clean, _cls_question, _count_rubric_criteria, _parse_criterion_scores,
    _normalize_criterion_scores, _sys_msg,
)
from src.verifier_cot_qc_plot import load_yaml, resolve_image_paths

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

# Same schema as a plain top3 checkpoint (VERIFIER_CKPT_FIELDS) plus ONE
# extra column — refinement_info — holding a per-row JSON summary of
# what, if anything, changed. Every column every existing QC/comparison
# tool reads (prediction, resolved_class, verifier_score, criterion_scores,
# judged_correct, candidate_predictions, answer_in_pool, ...) keeps its
# EXACT original meaning, so verifier-cot-qc-plot-top3/
# verifier-score-correlation-plot work against this output unmodified.
REFINE_CKPT_FIELDS = VERIFIER_CKPT_FIELDS + ["refinement_info"]

_EVIDENCE_RE = re.compile(r"^\[evidence\]\n(.*?)\n\n\[scoring\]\n", re.DOTALL)


def extract_round1_evidence(verifier_feedback: str) -> str:
    """Pulls the evidence block back out of run_group_top3_verifier_rubric's
    own stored verifier_feedback format ("[evidence]\\n...\\n\\n[scoring]\\n...")
    — reusing round 1's EXACT evidence text rather than re-deriving it with
    a fresh call, so correction is graded against the identical grounding
    the original scoring was."""
    m = _EVIDENCE_RE.search(verifier_feedback or "")
    return m.group(1).strip() if m else ""


def _failing_criteria_text(criterion_scores: dict[int, int], rubric_lines: dict[int, str]) -> list[int]:
    return sorted(n for n, score in criterion_scores.items() if score <= 2)


def run_model(model_id: str, args: argparse.Namespace) -> None:
    vprompts = load_yaml(Path(args.verifier_prompts))
    refine_template = vprompts["top3_refine_prompt"]
    scoring_template = vprompts["verifier_score_against_evidence_prompt"]
    rubrics_raw = vprompts.get("rubrics", {})
    task_type_to_rubric = vprompts.get("task_type_to_rubric", {})
    system_prompt = vprompts.get("assessment_system", "")
    prompts = load_prompts(Path(args.prompts))
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    def _rubric_lines(task_type: str) -> dict[int, str]:
        rubric_key = task_type_to_rubric.get(task_type, "disease_pest")
        raw = rubrics_raw.get(rubric_key, "")
        lines: dict[int, str] = {}
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or "." not in line:
                continue
            n_str, _, text = line.partition(".")
            if n_str.strip().isdigit():
                lines[int(n_str.strip())] = text.strip()
        return lines

    ckpt_path = Path(args.ckpt)
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    if not input_rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    input_rows = [r for r in input_rows if r.get("criterion_scores")]
    if not input_rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows with criterion_scores — is this a graded "
                          f"--top3_mode verifier_rubric checkpoint?")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    out_ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}_top3.csv"
    archive_ckpt_if_rerun(out_ckpt_path, args.rerun)
    out_rows, seen = load_ckpt(out_ckpt_path, fields=REFINE_CKPT_FIELDS)
    if out_rows:
        print(f"  [resume] {len(out_rows)} rows already refined in {out_ckpt_path.name}")

    pending = [r for r in input_rows if (r["agml_dataset"], r["class_label"], r["image_name"]) not in seen]
    if not pending:
        print("  Nothing to do — all rows already checkpointed.")
        return
    print(f"  {len(pending)} pending row(s), max_rounds={args.max_rounds}")

    image_paths = resolve_image_paths(pending, Path(args.images_root))
    n_missing_image = sum(
        1 for r in pending if (r["agml_dataset"], r["image_name"]) not in image_paths
    )
    if n_missing_image:
        print(f"  [warn] {n_missing_image} row(s) have no resolvable local image — kept unrefined "
              f"(pass-through) since a correction call needs the image.")

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="verifier_cot_top3_refine",
    )

    # State per row: current answer/criterion_scores/verifier_score (round
    # N so far), finalized once passed or exhausted.
    state: dict[int, dict] = {}
    for i, r in enumerate(pending):
        criterion_scores = {int(k): int(v) for k, v in json.loads(r["criterion_scores"]).items()}
        state[i] = {
            "answer": extract_answer_tag(r["prediction"])[0],
            "criterion_scores": criterion_scores,
            "verifier_score": float(r["verifier_score"]) if r.get("verifier_score") not in (None, "") else 0.0,
            "evidence": extract_round1_evidence(r.get("verifier_feedback", "")),
            "round1_answer": extract_answer_tag(r["prediction"])[0],
            "round1_verifier_score": r.get("verifier_score", ""),
            "round1_criterion_scores": criterion_scores,
            "refined": False,
            "status": "passed",
        }

    def _rubric_text(task_type: str) -> str:
        return "\n".join(f"{n}. {t}" for n, t in sorted(_rubric_lines(task_type).items()))

    for round_i in range(2, args.max_rounds + 1):
        needs_refine = [
            i for i, r in enumerate(pending)
            if any(s <= 2 for s in state[i]["criterion_scores"].values())
            and (pending[i]["agml_dataset"], pending[i]["image_name"]) in image_paths
        ]
        if not needs_refine:
            continue

        # Chunked (same --chunk_size convention run_judge_phase's own
        # --judge_chunk_size uses) so a run over hundreds/thousands of
        # rows gives real progress ticks — one llm.generate() pair per
        # chunk, not one single giant batch for the whole round (which
        # would otherwise sit at "0/1" for the entire run, then jump
        # straight to 100%, with zero intermediate visibility).
        chunks = [needs_refine[i:i + args.chunk_size] for i in range(0, len(needs_refine), args.chunk_size)]
        pbar = ProgressBar(total=len(chunks), desc=f"Refine round {round_i}")
        for chunk in chunks:
            images = [PILImage.open(image_paths[(pending[i]["agml_dataset"], pending[i]["image_name"])]).convert("RGB")
                      for i in chunk]

            correction_texts = []
            for i in chunk:
                r, s = pending[i], state[i]
                rubric_lines = _rubric_lines(r["task_type"])
                failing = _failing_criteria_text(s["criterion_scores"], rubric_lines)
                failing_text = "\n".join(f"  {n}. {rubric_lines.get(n, '')}" for n in failing) or "  (none)"
                try:
                    other_answers = [a for a in json.loads(r.get("candidate_predictions") or "[]")
                                      if a.strip().lower() != s["answer"].strip().lower()]
                except json.JSONDecodeError:
                    other_answers = []
                other_text = "\n".join(f"  - {a}" for a in other_answers) or "  (none)"
                question = _cls_question(r["task_type"], r.get("crop", ""), prompts)
                correction_texts.append(refine_template.format(
                    question=question, evidence=s["evidence"], current_answer=s["answer"],
                    failing_criteria=failing_text, other_candidates=other_text,
                ))

            correction_hist = [_sys_msg(prompts) + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": t},
            ]}] for t in correction_texts]
            correction_prompts = [
                {"prompt": render_chat_template(tokenizer, h, enable_thinking=args.enable_thinking,
                                                 chat_template=chat_template),
                 "multi_modal_data": {"image": img}}
                for h, img in zip(correction_hist, images)
            ]
            gen_params = SamplingParams(
                temperature=0.0, max_tokens=args.gen_max_tokens, seed=args.seed,
                skip_special_tokens=not args.enable_thinking,
            )
            correction_outs = llm.generate(correction_prompts, sampling_params=gen_params)

            new_answers = []
            for out in correction_outs:
                raw = out.outputs[0].text
                _, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
                answer, _ = extract_answer_tag(_clean(cleaned))
                new_answers.append(answer)

            # Re-score every corrected answer against its own round-1
            # evidence — identical mechanism/template round 1's own
            # scoring step used.
            scoring_texts = [
                scoring_template.format(
                    rubric=_rubric_text(pending[i]["task_type"]),
                    question=_cls_question(pending[i]["task_type"], pending[i].get("crop", ""), prompts),
                    evidence=state[i]["evidence"], answer=answer,
                )
                for i, answer in zip(chunk, new_answers)
            ]
            scoring_hist = [_sys_msg(prompts) + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": t},
            ]}] for t in scoring_texts]
            scoring_prompts = [
                {"prompt": render_chat_template(tokenizer, h, enable_thinking=args.enable_thinking,
                                                 chat_template=chat_template),
                 "multi_modal_data": {"image": img}}
                for h, img in zip(scoring_hist, images)
            ]
            scoring_params = SamplingParams(
                temperature=0.0, max_tokens=args.grading_max_tokens, seed=args.seed,
                skip_special_tokens=not args.enable_thinking,
            )
            scoring_outs = llm.generate(scoring_prompts, sampling_params=scoring_params)

            for i, answer, out in zip(chunk, new_answers, scoring_outs):
                raw = out.outputs[0].text
                _, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
                text = cleaned.strip()
                n_criteria = _count_rubric_criteria(_rubric_text(pending[i]["task_type"]))
                criterion_scores = _parse_criterion_scores(text, n_criteria)
                verifier_score = _normalize_criterion_scores(criterion_scores, n_criteria)
                s = state[i]
                s["refined"] = s["refined"] or (answer.strip().lower() != s["answer"].strip().lower())
                s["answer"] = answer
                s["criterion_scores"] = criterion_scores
                s["verifier_score"] = round(verifier_score, 4)
                s["final_round"] = round_i
                if all(v >= 3 for v in criterion_scores.values()):
                    s["status"] = "passed"
                else:
                    s["status"] = "exhausted" if round_i == args.max_rounds else "pending"

            pbar.update()
        pbar.finish()
    destroy_llm(llm)
    del tokenizer

    # Build final rows — passing-through rows keep their input judge
    # verdict AND resolved_class/correct/contains_match/rouge scores
    # unchanged (nothing about them changed). Refined rows get all of
    # those recomputed against the NEW answer (extract_answer_tag then
    # resolve_class — same convention verifier_cot.py's own _result_row
    # uses), and judged_correct cleared so run_judge_phase re-grades them.
    dataset_classes = {name: cfg.clean_classes for name, cfg in datasets_by_name(Path(args.config)).items()}
    judge_rows = []
    for i, r in enumerate(pending):
        s = state[i]
        row = dict(r)  # start from the input row — copies round-1-only columns (candidate_predictions,
                       # candidate_verifier_scores, answer_in_pool, verifier_picked_correct) UNCHANGED.
        row["prediction"] = f"<answer>{s['answer']}</answer>"
        row["extracted_answer"] = s["answer"]
        row["verifier_score"] = s["verifier_score"]
        row["criterion_scores"] = json.dumps(s["criterion_scores"])
        if s["refined"]:
            classes = dataset_classes.get(r["agml_dataset"], [])
            resolved = resolve_class(s["answer"], classes)
            ground_truth = r.get("ground_truth", r.get("class_label", ""))
            correct = resolved is not None and resolved.lower() == ground_truth.lower()
            c_match = contains_match(resolved or s["answer"], ground_truth)
            row["resolved_class"] = resolved or ""
            row["correct"] = int(correct)
            row["contains_match"] = int(c_match)
            row["resolvable"] = int(resolved is not None)
            row.update(rouge_scores_per_sample(row["prediction"], ground_truth, include_rouge=False))
            row["judged_correct"] = ""
            row["judge_reasoning"] = ""
        row["refinement_info"] = json.dumps({
            "refined": s["refined"], "status": s["status"],
            "round1_answer": s["round1_answer"], "round1_verifier_score": s["round1_verifier_score"],
            "round1_criterion_scores": s["round1_criterion_scores"],
            "final_answer": s["answer"], "final_verifier_score": s["verifier_score"],
            "final_criterion_scores": s["criterion_scores"],
        })
        judge_rows.append(row)

    judge_rows = run_judge_phase(judge_rows, out_ckpt_path, args, prompts, fields=REFINE_CKPT_FIELDS)

    n_refined = 0
    for i, row in enumerate(judge_rows):
        append_ckpt_row(out_ckpt_path, row, fields=REFINE_CKPT_FIELDS)
        out_rows.append(row)
        if state[i]["refined"]:
            n_refined += 1

    print(f"\n  {n_refined}/{len(pending)} row(s) corrected this run.")
    if out_rows:
        judged = [r for r in out_rows if r.get("judged_correct") not in (None, "")]
        if judged:
            acc = sum(int(r["judged_correct"]) for r in judged) / len(judged)
            print(f"  judge_accuracy={acc:.4f} (n={len(judged)}/{len(out_rows)} judged)")
        print(f"  Checkpoint: {out_ckpt_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--ckpt", required=True,
                         help="A graded --verifier_type top3 --top3_mode verifier_rubric checkpoint CSV "
                              "(ckpt_<model>_top3.csv) — never modified in place.")
    parser.add_argument("--images_root", default="dataset/images",
                         help="Images resolved by globbing <images_root>/<dataset>/*/<image_name> "
                              "against what's already cached on disk — never downloads/re-caches.")
    parser.add_argument("--verifier_prompts", default="src/config/verifier_prompts.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/verifier_cot_top3_refine/")
    parser.add_argument("--max_rounds", type=int, default=2,
                         help="Round 1 is the ALREADY-GRADED input checkpoint (no calls made for it) — "
                              "the default of 2 allows exactly ONE corrective round before a "
                              "still-failing row is kept as-is and flagged \"exhausted\".")
    parser.add_argument("--chunk_size", type=int, default=200,
                         help="Rows needing correction are processed in chunks of this size (one "
                              "correction + one re-scoring llm.generate() pair per chunk), same "
                              "convention as --judge_chunk_size below — gives real progress-bar "
                              "ticks on a large run instead of one single giant batch per round.")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Applies to BOTH the correction call and the re-scoring call, same as "
                              "--top3_mode verifier_rubric's own generation/evidence-fill/scoring "
                              "calls. Pass --no-enable_thinking to opt out.")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=1024,
                         help="max_tokens for the correction call — short answer, but real headroom "
                              "for a thinking trace first.")
    parser.add_argument("--grading_max_tokens", type=int, default=1024,
                         help="max_tokens for the re-scoring call — same reasoning as --gen_max_tokens.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    # run_judge_phase's required contract (same block judge_checkpoint.py uses).
    parser.add_argument("--config", default="src/config/datasets.yaml",
                         help="Used to fill the hybrid judge template's {class_list} placeholder.")
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--judge_enable_thinking", action="store_true")
    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--judge_chat_template_path", default=None)

    main(parser.parse_args())
