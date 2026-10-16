"""Judge leniency baseline — Phase 1 of the judge-improvements evaluation plan
("LLM as a judge improvements" note). Loads the judge model and, at EACH of
a swept set of temperatures, freshly re-judges every taggable row TWICE —
once on the full model response (pred_key="prediction"), once on just the
raw text between its own <answer></answer> tags — then compares the two
verdicts. Same judge, same template, same temperature on both sides at a
given sweep point; only the input text differs. See
src/common/leniency_analysis.py's module docstring for the full reasoning
(and why this replaced an earlier version that fed the judge resolve_class's
OUTPUT instead — that conflated judge leniency with resolve_class's own,
much larger, string-matching gap).

"resolved_class" as a row-field/pred_key name is kept below for continuity
with the rest of this module (rejudge_condition's pred_key, the output CSV
column, the leniency_analysis.py plot/JSON key names) — but it now holds
extract_answer_tag's raw output, NOT the checkpoint's own resolve_class-
matched value. The checkpoint's real resolved_class (and the deterministic
`correct` it feeds) is left completely untouched on disk; this script only
overwrites the field on its own in-memory working copy of each row.

Both sides are re-judged fresh at every temperature (no reuse of the
checkpoint's already-stored judged_correct, even for temperatures close to
the pipeline's normal greedy default) so leniency_gap is computed under the
exact same sampling conditions on both sides at every sweep point — this
also makes each temperature's leniency_gap directly comparable to Phase 2's
(src/judge_consistency.py) flip_rate at that same temperature, for the
leniency-vs-instability correlation plot.

Needs a GPU/judge model — NOT a pure-CPU step. Input is an already-produced
classification checkpoint CSV (vision_baseline.py / self_cot.py /
cot_yaml.py / desc_augmented_vision_v2.py's ckpt_*.csv, or any CSV sharing
CKPT_FIELDS' judged_correct/prediction/task_type columns) that has already
been through the judge phase at least once (judged_correct populated —
used only to know which rows are worth including, not reused as a value).

Writes, per checkpoint:
  leniency_gap_T<temp>.png         — one per temperature: full-response vs.
                                      <answer>-tag-only judge accuracy, per
                                      task_type (taggable rows only)
  leniency_gap_by_temperature.png  — leniency_gap per task_type, grouped by
                                      temperature (signed y-axis)
  disagreement_matrix_T<temp>.png  — one per temperature: the SAME rows'
                                      full-response/<answer>-tag verdicts
                                      broken into a 2x2 (both correct/both
                                      wrong/full-correct-tag-wrong/
                                      full-wrong-tag-correct) — raw counts
                                      behind leniency_gap's single net rate
  missing_answer_tag.txt           — how many/which fraction of rows had no
                                      <answer> tag at all (excluded from
                                      every temperature's comparison)
  rechecked_T<temp>.csv            — per temperature, per-row: prediction,
                                      resolved_class (here: the extracted
                                      <answer> tag text), judged_correct_full_trajectory
                                      (full-response verdict at this T),
                                      judged_correct (<answer>-tag verdict
                                      at this T)
  leniency_gap.json                — leniency_gap_by_temperature,
                                      disagreement_by_temperature +
                                      missing_answer_tag counts, machine-
                                      readable — this is what
                                      --leniency_gap_json in
                                      src/judge_consistency.py reads

Usage (--judge_model/--judge_enable_thinking/--judge_max_tokens/
--judge_chat_template_path and --temperatures 0.7 are all defaults now —
only pass them to override):
    uv run python main.py judge-leniency \\
        --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \\
        --output_dir ablation/judge_leniency/vision_baseline_thinking/ \\
        --label "Vision Baseline (thinking) — Gemma-4-31B-it-FP8"
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

from src.common.ablation_judge_recheck import rejudge_condition
from src.common.dataset_config import datasets_by_name
from src.common.io import load_prompts
from src.common.judge_template_sensitivity import add_class_list_field
from src.common.leniency_analysis import (
    compute_disagreement_matrix, compute_leniency_gap, missing_answer_tag_report,
    plot_disagreement_matrix, plot_leniency_gap, plot_leniency_gap_by_temperature,
    taggable_rows_with_extracted_answer, write_missing_answer_tag_report,
)
from src.common.metrics import DEFAULT_JUDGE_TEMPLATE_KEY
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm


_SHARD_RE = re.compile(r"_shard\d+of\d+\.csv$")


def _is_excluded_ckpt(name: str) -> bool:
    """True for a checkpoint file that must NEVER be concatenated alongside
    others matched by the same ckpt_*.csv glob:
      - "*.bak.csv" — vision_eval.py's --rerun archive of a prior checkpoint.
      - "*_shard<i>of<n>.csv" — vision_baseline.py's --num_shards output.
        Once --merge_shards has combined these into the canonical
        ckpt_<model>.csv, the shard files are deliberately left in place
        (never deleted — see merge_shards' own docstring), so a directory
        can contain BOTH the canonical file and its own already-merged
        shards. A plain glob then reads every row TWICE — this is exactly
        what caused the judge-template-sensitivity double-judging found on
        a real run (rechecked_T0.7.csv: every key appearing exactly twice,
        the leniency run's own judge cost silently doubled) — root-caused
        to this exact gap, not a judging-phase bug."""
    return ".bak." in name or bool(_SHARD_RE.search(name))


def load_ckpt_rows(ckpt_arg: str) -> list[dict]:
    """--ckpt accepts a single CSV path, a glob pattern, or a directory
    (globs ckpt_*.csv inside it) — concatenates every match, since a
    resumed/collated run can be split across multiple ckpt_*.csv files.

    Excludes anything _is_excluded_ckpt flags: "*.bak.csv" (an archived
    --rerun predecessor) and "*_shard<i>of<n>.csv" (already-merged shard
    files sitting alongside their own canonical merge) — either one would
    otherwise double-count every row it covers."""
    p = Path(ckpt_arg)
    if p.is_dir():
        paths = sorted(f for f in p.glob("ckpt_*.csv") if not _is_excluded_ckpt(f.name))
    else:
        matches = sorted(m for m in glob.glob(ckpt_arg) if not _is_excluded_ckpt(Path(m).name))
        paths = [Path(m) for m in matches] if matches else [p]
    if not paths:
        raise FileNotFoundError(f"No checkpoint CSV found for --ckpt {ckpt_arg!r}")

    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main(args: argparse.Namespace, judge_llm=None, judge_tok=None) -> None:
    """judge_llm/judge_tok (both default None): pass an already-loaded judge
    model to reuse it instead of loading/destroying a fresh one — used by
    src/judge_template_sensitivity.py's orchestrator, which calls this once
    per template variant and would otherwise reload the (slow) judge model
    from scratch every time. When None (the normal single-run CLI case),
    behavior is unchanged: load here, destroy here."""
    owns_judge_llm = judge_llm is None
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("prediction", "task_type") if c not in rows[0]]
    if missing:
        raise ValueError(
            f"--ckpt is missing expected column(s) {missing} — leniency_analysis needs "
            "CKPT_FIELDS-shaped rows (vision_baseline.py / self_cot.py / cot_yaml.py / "
            "desc_augmented_vision_v2.py's checkpoint CSV shape)."
        )
    # No requirement that judged_correct already be populated — every row's
    # verdict is re-judged FRESH on both sides at every swept temperature
    # (see rejudge_condition), so a checkpoint produced with --skip_judge
    # (judged_correct blank everywhere) works fine here; the checkpoint's
    # own stored judged_correct value, if any, is never read.
    if getattr(args, "limit", None):
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.ckpt).stem
    judge_chat_template = (
        Path(args.judge_chat_template_path).read_text(encoding="utf-8")
        if getattr(args, "judge_chat_template_path", None) else None
    )

    # missing_answer_tag is computed over EVERY row (before filtering) so it
    # reports what fraction of the whole dataset gets excluded from the
    # leniency comparison below, not just a count within an already-filtered set.
    failures = missing_answer_tag_report(rows)
    write_missing_answer_tag_report(failures, output_dir / "missing_answer_tag.txt", model_label=label)

    taggable_rows = taggable_rows_with_extracted_answer(rows)
    if not taggable_rows:
        raise ValueError(
            "No rows have a parseable <answer> tag — nothing to re-judge. "
            f"See {output_dir / 'missing_answer_tag.txt'} for the breakdown."
        )
    print(f"  {len(taggable_rows)}/{len(rows)} row(s) have a parseable <answer> tag "
          f"({failures['overall']['pct_missing']:.2f}% excluded as missing) — re-judging those.")

    # Built unconditionally (not gated on --class_list) since DEFAULT_JUDGE_TEMPLATE_KEY
    # (hybrid, the new --template_key default) has a {class_list} placeholder —
    # a template without that placeholder just ignores the unused format()
    # kwarg, so this is harmless regardless of which --template_key is active.
    # --class_list is kept as a flag for _variant_args/documentation purposes
    # but is no longer load-bearing here.
    dataset_classes = {name: cfg.clean_classes for name, cfg in datasets_by_name(Path(args.config)).items()}
    add_class_list_field(taggable_rows, dataset_classes)
    extra_keys = {"class_list": "class_list_text"}

    prompts = load_prompts(Path(args.prompts))
    if owns_judge_llm:
        judge_llm, judge_tok = _load_judge_llm(
            args.judge_model, quantization=args.judge_quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
        )
    fields = ["agml_dataset", "class_label", "image_name", "task_type", "ground_truth",
              "resolved_class", "correct", "prediction",
              "judged_correct_full_trajectory", "judge_reasoning_full_trajectory",
              "judged_correct", "judge_reasoning",
              "judge_raw_response_full_trajectory", "judge_raw_response"]

    results_by_temp: dict[float, dict] = {}
    disagreement_by_temp: dict[float, dict] = {}
    try:
        for temperature in args.temperatures:
            full_rechecked = rejudge_condition(
                taggable_rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                log_path=output_dir / f"full_response_T{temperature}_judge_log.jsonl",
                pred_key="prediction", template_key=args.template_key,
                progress_desc=f"Re-judging {label} (full response, T={temperature})",
                temperature=temperature, seed=0, extra_keys=extra_keys,
                enable_thinking=getattr(args, "judge_enable_thinking", False),
                judge_max_tokens=getattr(args, "judge_max_tokens", None),
                judge_chat_template=judge_chat_template,
            )
            resolved_rechecked = rejudge_condition(
                taggable_rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                log_path=output_dir / f"answer_tag_T{temperature}_judge_log.jsonl",
                pred_key="resolved_class", template_key=args.template_key,
                progress_desc=f"Re-judging {label} (<answer> tag only, T={temperature})",
                temperature=temperature, seed=0, extra_keys=extra_keys,
                enable_thinking=getattr(args, "judge_enable_thinking", False),
                judge_max_tokens=getattr(args, "judge_max_tokens", None),
                judge_chat_template=judge_chat_template,
            )

            # Same row order/length (both are taggable_rows re-judged
            # independently) — zip by index to pair each row's two fresh
            # verdicts into the shape compute_leniency_gap expects.
            combined = []
            for base, full_r, resolved_r in zip(taggable_rows, full_rechecked, resolved_rechecked):
                combined.append({
                    **base,
                    "judged_correct_full_trajectory": full_r["judged_correct"],
                    "judge_reasoning_full_trajectory": full_r["judge_reasoning"],
                    "judge_raw_response_full_trajectory": full_r.get("judge_raw_response", ""),
                    "judged_correct": resolved_r["judged_correct"],
                    "judge_reasoning": resolved_r["judge_reasoning"],
                    "judge_raw_response": resolved_r.get("judge_raw_response", ""),
                })

            results = compute_leniency_gap(combined)
            results_by_temp[temperature] = results
            plot_leniency_gap(results, output_dir / f"leniency_gap_T{temperature}.png")

            disagreement = compute_disagreement_matrix(combined)
            disagreement_by_temp[temperature] = disagreement
            plot_disagreement_matrix(disagreement, output_dir / f"disagreement_matrix_T{temperature}.png")

            with open(output_dir / f"rechecked_T{temperature}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in combined:
                    w.writerow(r)

            d = results["overall"]
            dis = disagreement["overall"]
            print(f"  T={temperature}: leniency_gap={d['leniency_gap']:+.4f} "
                  f"(full_response={d['full_response_rate']:.4f}, answer_tag={d['resolved_class_rate']:.4f}) — "
                  f"disagreements: full-correct/tag-wrong={dis['full_correct_resolved_wrong']['n']} "
                  f"({dis['full_correct_resolved_wrong']['rate']:.4f}), "
                  f"full-wrong/tag-correct={dis['full_wrong_resolved_correct']['n']} "
                  f"({dis['full_wrong_resolved_correct']['rate']:.4f})")
    finally:
        if owns_judge_llm:
            _destroy_judge_llm(judge_llm)
            del judge_tok

    plot_leniency_gap_by_temperature(results_by_temp, output_dir / "leniency_gap_by_temperature.png")

    (output_dir / "leniency_gap.json").write_text(json.dumps(
        {"label": label, "n_rows_total": len(rows), "n_rows_rejudged": len(taggable_rows),
         "temperatures": args.temperatures,
         "leniency_gap_by_temperature": {str(t): d for t, d in results_by_temp.items()},
         "disagreement_by_temperature": {str(t): d for t, d in disagreement_by_temp.items()},
         "missing_answer_tag": failures},
        indent=2,
    ))

    print(f"Wrote leniency_gap_T<temp>.png, leniency_gap_by_temperature.png, "
          f"disagreement_matrix_T<temp>.png, missing_answer_tag.txt, "
          f"rechecked_T<temp>.csv, leniency_gap.json to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_leniency/")
    parser.add_argument("--label", default=None, help="Chart title / report header label (defaults to the ckpt filename).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample to this many rows first (each temperature re-judges both "
                              "sides over every taggable row in the subsample — start small).")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7],
                         help="Swept judge temperatures — matches src/judge_consistency.py's default so "
                              "leniency_gap(T) and flip_rate(T) line up for the correlation plot. Both the "
                              "full-response and <answer>-tag-only sides are freshly re-judged at each value "
                              "(no reuse of the checkpoint's stored judged_correct).")
    parser.add_argument("--template_key", default=DEFAULT_JUDGE_TEMPLATE_KEY,
                         help="prompts.yaml key used on BOTH the full-response and <answer>-tag-only sides "
                              "at every temperature — override to re-run this whole sweep under a "
                              "different judge template (e.g. Phase 3's specificity/class_list variants, "
                              "see src/judge_template_sensitivity.py).")
    parser.add_argument("--class_list", action="store_true",
                         help="Fill a {class_list} placeholder in --template_key from each row's own "
                              "dataset (src/common/judge_template_sensitivity.py's class_list variant) — "
                              "only meaningful if --template_key's prompt actually has that placeholder.")
    parser.add_argument("--config", default="src/config/datasets.yaml",
                         help="datasets.yaml, for each row's class list when --class_list is set.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--judge_enable_thinking", action="store_true",
                         help="Attempt tokenizer.apply_chat_template(..., enable_thinking=True) for the "
                              "judge's own prompt — off by default, exact original prompt rendering. "
                              "A no-op for a Qwen judge model's stock template if it doesn't accept the "
                              "kwarg (falls back to the plain call); a no-op for a Gemma-4 judge model "
                              "UNLESS --judge_chat_template_path is also given — Gemma-4's own bundled "
                              "template has no enable_thinking support at all (confirmed empirically).")
    parser.add_argument("--judge_max_tokens", type=int, default=None,
                         help="Override the judge's default max_tokens=1024 for THIS run — a thinking "
                              "trace (--judge_enable_thinking) can consume the normal budget before "
                              "reaching the final JSON verdict, truncating the response.")
    parser.add_argument("--judge_chat_template_path", default=None,
                         help="Override the judge tokenizer's own bundled chat template with a Jinja "
                              "file at this path — needed for --judge_enable_thinking to do anything on "
                              "a Gemma-4 judge model. Pass src/config/tool_chat_template_gemma4.jinja "
                              "alongside --judge_enable_thinking for a Gemma-4 judge. Not needed for a "
                              "Qwen judge model.")
    args = parser.parse_args()
    main(args)
