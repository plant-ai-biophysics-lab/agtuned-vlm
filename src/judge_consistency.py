"""Judge self-consistency — Phase 2 of the judge-improvements evaluation plan
("LLM as a judge improvements" note). Loads the judge model and repeats the
judge call N times per row at each of a swept set of temperatures, for BOTH
conditions Phase 1 compares — the full model response (pred_key="prediction")
and just the raw text between the model's own <answer></answer> tags
(pred_key="resolved_class", same "resolved_class"-holds-the-extracted-answer
convention as src/judge_leniency.py) — then measures how often each
condition's verdict flips across repeats. Both conditions run over the SAME
row set (rows with a parseable <answer> tag), matching Phase 1's design, so
flip_rate_full and flip_rate_answer_only are directly comparable and both
line up with Phase 1's leniency_gap at the same temperature.

Needs a GPU/judge model — NOT a pure-CPU step, and (unlike Phase 1's
leniency check) forces real sampling: at the pipeline's normal
temperature=0.0, repeat judge calls are bit-for-bit identical and there is
nothing to measure a flip rate over.

Running BOTH conditions DOUBLES this script's judge-call budget relative to
before (now 2 x n_temperatures x n_repeats x n_taggable_rows) — start with
--limit and a short --temperatures list.

Writes, per checkpoint:
  flip_rate_by_temperature_full.png         — flip_rate per task_type,
                                               grouped by temperature, full
                                               response condition
  flip_rate_by_temperature_answer_only.png  — same, <answer>-tag-only condition
  leniency_vs_flip_rate_full.png            — only if --leniency_gap_json is
                                               given: scatter of leniency_gap
                                               vs. flip_rate_full
  leniency_vs_flip_rate_answer_only.png     — same, vs. flip_rate_answer_only
  flip_rate.json                            — machine-readable, both
                                               conditions, per temperature/task_type

Usage:
    uv run python main.py judge-consistency \\
        --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \\
        --output_dir ablation/judge_consistency/vision_baseline_gemma4_31b_fp8/ \\
        --label "Vision Baseline — Gemma-4-31B-it-FP8" \\
        --temperatures 0.1 0.5 1.0 --n_repeats 5 \\
        --leniency_gap_json ablation/judge_leniency/vision_baseline_gemma4_31b_fp8/leniency_gap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.dataset_config import datasets_by_name
from src.common.io import load_prompts
from src.common.judge_consistency import (
    compute_flip_rate, plot_flip_rate_by_temperature, plot_leniency_vs_flip_rate, repeat_judge_calls,
)
from src.common.judge_template_sensitivity import add_class_list_field
from src.common.leniency_analysis import taggable_rows_with_extracted_answer
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm
from src.judge_leniency import load_ckpt_rows

CONDITIONS = [
    # (condition_key, pred_key, file_suffix, label)
    ("full", "prediction", "full", "full response"),
    ("answer_only", "resolved_class", "answer_only", "<answer> tag only"),
]


def main(args: argparse.Namespace, judge_llm=None, judge_tok=None) -> None:
    """judge_llm/judge_tok (both default None): pass an already-loaded judge
    model to reuse it instead of loading/destroying a fresh one — used by
    src/judge_template_sensitivity.py's orchestrator. When None (the normal
    single-run CLI case), behavior is unchanged: load here, destroy here."""
    owns_judge_llm = judge_llm is None
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("prediction", "task_type") if c not in rows[0]]
    if missing:
        raise ValueError(f"--ckpt is missing expected column(s) {missing}.")
    # No requirement that judged_correct already be populated — every
    # repeat call re-judges from scratch (see repeat_judge_calls), so a
    # checkpoint produced with --skip_judge works fine here too.
    if args.limit:
        rows = rows[: args.limit]

    # Both conditions run over the SAME row set (rows with a parseable
    # <answer> tag) — matches Phase 1's design, so flip_rate_full and
    # flip_rate_answer_only are computed over identical n, and both line up
    # with leniency_gap's own row set at the same temperature.
    taggable_rows = taggable_rows_with_extracted_answer(rows)
    if not taggable_rows:
        raise ValueError("No rows have a parseable <answer> tag — nothing to re-judge.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.ckpt).stem

    n_calls = len(taggable_rows) * len(args.temperatures) * args.n_repeats * len(CONDITIONS)
    print(f"  {len(taggable_rows)}/{len(rows)} row(s) have a parseable <answer> tag, "
          f"{len(args.temperatures)} temperature(s) x {args.n_repeats} repeat(s) x {len(CONDITIONS)} condition(s) "
          f"= {n_calls} judge call(s) total.")

    extra_keys = None
    if args.class_list:
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
    flip_rate_by_temp_by_condition: dict[str, dict[float, dict]] = {}
    try:
        for condition_key, pred_key, file_suffix, condition_label in CONDITIONS:
            log_dir = (output_dir / file_suffix) if args.save_logs else None
            rows_by_temp = repeat_judge_calls(
                taggable_rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                temperatures=args.temperatures, n_repeats=args.n_repeats,
                log_dir=log_dir, pred_key=pred_key,
                template_key=args.template_key, extra_keys=extra_keys,
                enable_thinking=getattr(args, "judge_enable_thinking", False),
                judge_max_tokens=getattr(args, "judge_max_tokens", None),
            )
            flip_rate_by_temp = {t: compute_flip_rate(rows_by_temp[t]) for t in args.temperatures}
            flip_rate_by_temp_by_condition[condition_key] = flip_rate_by_temp
            plot_flip_rate_by_temperature(flip_rate_by_temp, output_dir / f"flip_rate_by_temperature_{file_suffix}.png")
            for t in args.temperatures:
                d = flip_rate_by_temp[t]["overall"]
                print(f"  [{condition_label}] T={t}: overall flip_rate={d['flip_rate']:.4f}, "
                      f"mean_judged_correct={d['mean_judged_correct']:.4f}")
    finally:
        if owns_judge_llm:
            _destroy_judge_llm(judge_llm)
            del judge_tok

    (output_dir / "flip_rate.json").write_text(json.dumps(
        {"label": label, "n_rows": len(taggable_rows), "n_repeats": args.n_repeats,
         "flip_rate_by_temperature_full": {str(t): d for t, d in flip_rate_by_temp_by_condition["full"].items()},
         "flip_rate_by_temperature_answer_only": {
             str(t): d for t, d in flip_rate_by_temp_by_condition["answer_only"].items()
         }},
        indent=2,
    ))
    print(f"Wrote flip_rate_by_temperature_full.png, flip_rate_by_temperature_answer_only.png, "
          f"flip_rate.json to {output_dir}")

    if args.leniency_gap_json:
        leniency_data = json.loads(Path(args.leniency_gap_json).read_text())
        leniency_by_temp = {float(t): d for t, d in leniency_data["leniency_gap_by_temperature"].items()}
        for condition_key, _, file_suffix, condition_label in CONDITIONS:
            flip_rate_by_temp = flip_rate_by_temp_by_condition[condition_key]
            corr_stats = plot_leniency_vs_flip_rate(
                leniency_by_temp, flip_rate_by_temp,
                output_dir / f"leniency_vs_flip_rate_{file_suffix}.png",
            )
            (output_dir / f"leniency_vs_flip_rate_{file_suffix}_stats.json").write_text(
                json.dumps(corr_stats if corr_stats is not None else {"note": "not enough points/variance to compute"}, indent=2)
            )
            print(f"Wrote leniency_vs_flip_rate_{file_suffix}.png, "
                  f"leniency_vs_flip_rate_{file_suffix}_stats.json to {output_dir}")
            if corr_stats:
                for temp_label, d in corr_stats.items():
                    print(f"  [vs. {condition_label}] {temp_label}: Pearson r={d['pearson_r']:.2f}, "
                          f"p={d['p_value']:.3f}, n={d['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_consistency/")
    parser.add_argument("--label", default=None, help="Report/log label (defaults to the ckpt filename).")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--n_repeats", type=int, default=5, help="Repeat judge calls per row, per temperature, per condition.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample to this many rows first (repeat calls multiply cost by "
                              "n_temperatures * n_repeats * 2 conditions — start small).")
    parser.add_argument("--leniency_gap_json", default=None,
                         help="Phase 1's leniency_gap.json for the SAME checkpoint — if given, also "
                              "writes leniency_vs_flip_rate_full.png and leniency_vs_flip_rate_answer_only.png.")
    parser.add_argument("--save_logs", action="store_true",
                         help="Write one judge_log.jsonl per (condition, temperature, repeat) — off by "
                              "default, this is 2 * n_temperatures * n_repeats files.")
    parser.add_argument("--template_key", default="judge_classification_template",
                         help="prompts.yaml key used for every repeat call, both conditions — override to "
                              "re-run this whole sweep under a different judge template (Phase 3's "
                              "specificity/class_list variants, see src/judge_template_sensitivity.py).")
    parser.add_argument("--class_list", action="store_true",
                         help="Fill a {class_list} placeholder in --template_key from each row's own "
                              "dataset — only meaningful if --template_key's prompt has that placeholder.")
    parser.add_argument("--config", default="src/config/datasets.yaml",
                         help="datasets.yaml, for each row's class list when --class_list is set.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--judge_enable_thinking", action="store_true",
                         help="Attempt tokenizer.apply_chat_template(..., enable_thinking=True) for the "
                              "judge's own prompt — off by default, exact original prompt rendering. "
                              "A no-op if the judge tokenizer's chat template doesn't accept this kwarg. "
                              "Unverified for Gemma-4 — may use a different channel-based reasoning "
                              "format than Qwen3's enable_thinking.")
    parser.add_argument("--judge_max_tokens", type=int, default=None,
                         help="Override the judge's default max_tokens=1024 for THIS run — a thinking "
                              "trace (--judge_enable_thinking) can consume the normal budget before "
                              "reaching the final JSON verdict, truncating the response.")
    args = parser.parse_args()
    main(args)
