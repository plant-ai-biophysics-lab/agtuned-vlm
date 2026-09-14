"""Ablation stage, "judge-recheck" step — checks whether the LLM judge's
judged_correct verdict on verifier_cot.py rows is being driven by the
model's actual stated diagnosis, or by the rubric-filled description text
that surrounds it.

verifier_cot.py judges the FULL winning trajectory (result["prediction"] =
the raw candidate text, every rubric fill-in-the-blank field included, not
just the <answer> tag) — see run_judge_phase's pred_key="prediction" in
src/verifier_cot.py. Real data confirms this matters: pulling judge_log.jsonl
for rows where the direct resolved-class match failed but the judge still
said judged_correct=1 shows the judge crediting descriptive symptom text
("dark spots and necrotic areas... hallmark symptoms of X") even when the
model's own stated Diagnosis line is a generic, wrong, or non-committal
label (one single generic answer, "Leaf spot disease", was credited correct
against 126 distinct actual disease labels in a spot-check).

This module re-judges the SAME rows using only the extracted <answer> tag
content (extract_answer_tag(prediction), from src.verifier_cot — the exact
same extraction the tournament itself uses internally) instead of the full
trajectory, so the size of the leniency effect can be measured directly:
does judged F1 drop once the judge can no longer see the rubric's
descriptive fields, and if so, by how much relative to the lower/upper
bounds.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.common.metrics import apply_judge, extract_answer_tag, judged_summary
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm


def load_ckpt_rows(ckpt_path: Path) -> list[dict]:
    with open(ckpt_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_answer_only_field(rows: list[dict]) -> None:
    """In place: row["answer_only"] = the <answer> tag content of
    row["prediction"] (falls back to the full cleaned text if no tag was
    found, same as extract_answer_tag's own contract) — what the judge
    SHOULD see if it's meant to grade the stated diagnosis, not the
    surrounding rubric narrative."""
    for r in rows:
        answer, _found = extract_answer_tag(r.get("prediction", ""))
        r["answer_only"] = answer


def rejudge_condition(
    rows: list[dict],
    judge_llm,
    judge_tok,
    prompts: dict,
    confidence_threshold: int,
    chunk_size: int,
    log_path: Path | None,
    pred_key: str = "answer_only",
    template_key: str = "judge_classification_template",
    progress_desc: str = "Re-judging",
    temperature: float | None = None,
    seed: int | None = None,
    extra_keys: dict[str, str] | None = None,
    enable_thinking: bool = False,
    judge_max_tokens: int | None = None,
) -> list[dict]:
    """Re-judges `rows` against `pred_key` using `prompts[template_key]`
    instead of the original "prediction"/judge_classification_template pair.
    Preserves the ORIGINAL full-trajectory verdict under
    judged_correct_full_trajectory/judge_reasoning_full_trajectory before
    calling apply_judge, since apply_judge's return dict overwrites
    judged_correct/judge_reasoning with the new result.

    Two re-judge conditions currently use this:
      - answer_only: pred_key="answer_only" (needs add_answer_only_field
        first), template_key="judge_classification_template" (unchanged) —
        strips the rubric's observational fields out entirely.
      - strict_context: pred_key="prediction" (the full trajectory,
        unchanged), template_key="judge_classification_strict_template" —
        keeps the full context but explicitly instructs the judge to grade
        only the model's own stated final diagnosis, not the narrative
        around it, and to require taxonomic/common-name specificity rather
        than crediting a generic symptom-shape label.

    temperature/seed: forwarded to apply_judge (both default None, the
    pipeline's normal greedy behavior). src/judge_leniency.py's
    --temperatures sweep passes these so leniency_gap can be computed AT
    each swept temperature (both the full-response and resolved_class
    passes need a fresh, same-temperature call — reusing a checkpoint's
    already-computed judged_correct only gives the T=0 case) instead of
    only at the judge's fixed default.

    extra_keys: forwarded to apply_judge — {template_placeholder: row_field}
    for template variants needing a third placeholder beyond {gt}/{pred},
    e.g. judge_classification_class_list_template's {class_list}
    (src/common/judge_template_sensitivity.py, Phase 3).

    enable_thinking/judge_max_tokens: forwarded to apply_judge (both default
    False/None, unchanged behavior) — lets a caller test whether the judge
    model exposes a controllable reasoning mode, and raise the token budget
    to accommodate a thinking trace if so.
    """
    from src.common.progress import ProgressBar

    for r in rows:
        r["judged_correct_full_trajectory"] = r.get("judged_correct", "")
        r["judge_reasoning_full_trajectory"] = r.get("judge_reasoning", "")

    out: list[dict] = []
    pbar = ProgressBar(total=len(rows), desc=progress_desc)
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        judged_chunk = apply_judge(
            rows=chunk, llm=judge_llm, tokenizer=judge_tok,
            system_prompt=prompts.get("judge_system_vision", ""),
            judge_template=prompts[template_key],
            gt_key="judge_gt_hint", pred_key=pred_key,
            confidence_threshold=confidence_threshold,
            log_path=log_path,
            temperature=temperature, seed=seed, extra_keys=extra_keys,
            enable_thinking=enable_thinking, max_tokens=judge_max_tokens,
        )
        out.extend(judged_chunk)
        pbar.update(len(chunk))
    pbar.finish()
    return out


# Backward-compatible alias — the original name, kept so any existing
# callers/scripts referencing it directly don't break.
rejudge_answer_only = rejudge_condition


def rows_with_judged_correct(rows: list[dict], key: str) -> list[dict]:
    """Copies of `rows` with judged_correct set from `rows[i][key]` — lets
    the SAME underlying rows (resolved_class/ground_truth/class_label/
    task_type unchanged) be scored under either the original full-trajectory
    verdict or the new answer-only verdict, for a like-for-like
    method_comparison chart."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["judged_correct"] = r.get(key, "")
        out.append(r2)
    return out


def summary_table(rows_by_condition: dict[str, list[dict]]) -> dict:
    """{condition: {"overall": judged_summary(...), "<task_type>": judged_summary(...)}}"""
    from collections import defaultdict
    out = {}
    for label, rows in rows_by_condition.items():
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_task[r.get("task_type", "")].append(r)
        out[label] = {"overall": judged_summary(rows)}
        for tt, trows in by_task.items():
            out[label][tt] = judged_summary(trows)
    return out
