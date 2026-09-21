"""Judge template sensitivity sweep — Phase 3 (+ Phase 4's hybrid variant) of
the judge-improvements evaluation plan ("LLM as a judge improvements" note).
For each of four judge templates (see src/common/judge_template_sensitivity.py's
TEMPLATE_VARIANTS / module docstring for what each one tests), re-runs
BOTH Phase 1 (src/judge_leniency.py — leniency_gap across a temperature
sweep) and Phase 2 (src/judge_consistency.py — flip_rate across that same
sweep, plus the leniency-vs-flip-rate correlation), under that variant's
template. Each variant gets its own output subfolder with the full Phase
1+2 figure set — this is Phase 3 done properly (compare templates on the
SAME axes Phase 1/2 already established), not just a single-temperature
summary against the deterministic baseline.

Also still runs the plan's two Phase-3-specific checks, which don't need
the temperature sweep: score_against_deterministic (each variant's greedy-
temperature judged_correct vs. the checkpoint's own resolve_class match —
the plan's literal "scored against Phase 1's deterministic baseline") and
the length/verbosity bias report.

The judge model is loaded ONCE and shared across every variant/phase call
(judge_leniency.main() and judge_consistency.main() both accept an
already-loaded judge_llm/judge_tok) — six separate slow model loads would
otherwise be paid for 4 variants x 2 phases.

Needs a GPU/judge model. This is the most expensive script in the judge-
improvements evaluation plan by far: 4 variants x (leniency's 2 sides x
n_temperatures judge passes + consistency's n_temperatures x n_repeats
judge passes) x n_resolvable_rows. Start with --limit and a short
--temperatures list.

All rate-type figures (leniency_gap, flip_rate, judged_correct_rate, the
length-bias curves) carry bootstrap 95% CI error bars — see
src/common/bootstrap.py — computed by resampling rows, not by re-running
the judge across multiple seeds (cheap given the row counts here, at zero
extra judge-call cost).

Writes, under --output_dir:
  <variant_key>/leniency/                 — Phase 1's full output for this
                                             variant (leniency_gap_T<temp>.png,
                                             leniency_gap_by_temperature.png,
                                             disagreement_matrix_T<temp>.png —
                                             the full-response/resolved-class
                                             2x2 breakdown, raw counts behind
                                             leniency_gap's single net rate —
                                             leniency_gap.json, etc.)
  <variant_key>/consistency/              — Phase 2's full output for this
                                             variant, BOTH conditions
                                             (flip_rate_by_temperature_full.png,
                                             flip_rate_by_temperature_answer_only.png,
                                             leniency_vs_flip_rate_full.png,
                                             leniency_vs_flip_rate_answer_only.png,
                                             flip_rate.json, etc.)
  <variant_key>/leniency_vs_flip_rate_{full,answer_only}.png — copied up
                                             from consistency/ for a quick
                                             look without descending into
                                             that subfolder
  <variant_key>/disagreement_matrix_T<temp>.png — copied up from leniency/,
                                             same reason, one per swept
                                             temperature
  template_sensitivity.png                — single-temperature (greedy) summary
                                             across all 4 variants vs. the
                                             deterministic baseline
  length_bias.png                         — length/verbosity bias, current template
  template_sensitivity_table.tex          — LaTeX booktabs table: leniency_gap,
                                             flip_rate, and both disagreement-
                                             direction counts/rates (overall,
                                             with 95% CI) per variant x temperature
  judge_accuracy_f1.png                   — judge accuracy + judge-scored
                                             macro-F1 per variant (bootstrap
                                             95% CI), reusing
                                             judged_summary/bootstrap_judge_f1_ci
  leniency_and_false_positive_summary.png — two-panel: leniency_gap (overall,
                                             all rows) + false-positive credit
                                             rate (deterministic-wrong rows
                                             only) per variant, at
                                             --summary_temperature (default
                                             0.7 — must be included in
                                             --temperatures for a variant to
                                             appear here)
  leniency_and_false_positive_summary_stats.json — the above, machine-readable
  oracle_validation_sample.csv            — hybrid vs. current, sampled for
                                             manual human review (mostly
                                             disagreement rows, some agreement
                                             for base-rate context) — blank
                                             human_verdict/human_notes columns
                                             to fill in; needs both variants
                                             present in --output_dir
  template_sensitivity.json               — template_sensitivity.png +
                                             length_bias.png + judge_accuracy_f1's
                                             data, machine-readable

Usage (--judge_model/--judge_enable_thinking/--judge_max_tokens/
--judge_chat_template_path and --temperatures 0.7 are all defaults now —
only pass them to override):
    uv run python main.py judge-template-sensitivity \\
        --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \\
        --output_dir outputs_final/judge_template_sensitivity/vision_baseline_thinking/ \\
        --template hybrid
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import src.judge_consistency as judge_consistency
import src.judge_leniency as judge_leniency
from src.common.ablation_judge_recheck import rejudge_condition
from src.common.dataset_config import datasets_by_name
from src.common.io import load_prompts
from src.common.judge_template_sensitivity import (
    TEMPLATE_VARIANTS, add_class_list_field, compute_judge_accuracy_f1, false_positive_credit_rate,
    length_bias_report, plot_judge_accuracy_f1, plot_leniency_and_false_positive_summary,
    plot_length_bias, plot_template_sensitivity, score_against_deterministic,
    write_oracle_validation_csv, write_variant_temperature_latex_table,
)
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm
from src.judge_leniency import load_ckpt_rows

# Persisted so a LATER invocation covering a DIFFERENT --template can still
# read this variant's greedy-summary verdicts back off disk and fold them
# into the cross-variant summary (score_against_deterministic, length_bias,
# template_sensitivity.png/table.tex) without re-judging anything.
GREEDY_SUMMARY_FIELDS = [
    "agml_dataset", "class_label", "image_name", "task_type", "ground_truth",
    "prediction", "resolved_class", "extracted_answer", "correct",
    "judged_correct", "judge_reasoning", "judge_raw_response",
]


def _variant_args(base: argparse.Namespace, output_dir: Path, label: str, template_key: str, needs_class_list: bool,
                   **overrides) -> argparse.Namespace:
    """Build the argparse.Namespace judge_leniency.main()/judge_consistency.main()
    expect, from this orchestrator's own args plus per-variant overrides —
    both scripts' CLIs and this one intentionally share flag names
    (--judge_model, --temperatures, etc.) so this is mostly a pass-through.

    limit=base.limit is included here (not just applied once to this
    orchestrator's own `rows`) because judge_leniency.main()/
    judge_consistency.main() each independently re-read --ckpt from disk —
    they never see this orchestrator's already-subsampled `rows` list, so
    --limit has to be forwarded explicitly or every per-variant Phase 1/2
    call would silently run over the FULL checkpoint regardless of --limit.
    """
    return argparse.Namespace(
        ckpt=base.ckpt, output_dir=str(output_dir), label=label,
        temperatures=base.temperatures, template_key=template_key, class_list=needs_class_list,
        config=base.config, prompts=base.prompts, limit=base.limit,
        judge_model=base.judge_model, judge_quantize=base.judge_quantize,
        judge_confidence_threshold=base.judge_confidence_threshold, judge_chunk_size=base.judge_chunk_size,
        gpu_memory_utilization=base.gpu_memory_utilization, max_model_len=base.max_model_len,
        enforce_eager=base.enforce_eager,
        judge_enable_thinking=base.judge_enable_thinking, judge_max_tokens=base.judge_max_tokens,
        judge_chat_template_path=base.judge_chat_template_path,
        **overrides,
    )


def main(args: argparse.Namespace) -> None:
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("correct", "prediction", "agml_dataset", "task_type") if c not in rows[0]]
    if missing:
        raise ValueError(f"--ckpt is missing expected column(s) {missing}.")
    # No requirement that judged_correct already be populated — every
    # variant (including "current") gets a fresh greedy-temperature judge
    # call below, so a checkpoint produced with --skip_judge works fine.
    if args.limit:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.ckpt).stem
    judge_chat_template = (
        Path(args.judge_chat_template_path).read_text(encoding="utf-8")
        if getattr(args, "judge_chat_template_path", None) else None
    )

    selected_variants = [v for v in TEMPLATE_VARIANTS if args.template in ("all", v[0])]
    if not selected_variants:
        raise ValueError(f"--template {args.template!r} doesn't match any TEMPLATE_VARIANTS key.")
    print(f"Running {len(selected_variants)}/{len(TEMPLATE_VARIANTS)} variant(s) this invocation: "
          f"{[v[0] for v in selected_variants]}")

    dataset_classes = {name: cfg.clean_classes for name, cfg in datasets_by_name(Path(args.config)).items()}
    prompts = load_prompts(Path(args.prompts))

    judge_llm, judge_tok = _load_judge_llm(
        args.judge_model, quantization=args.judge_quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
    )
    rows_by_variant: dict[str, list[dict]] = {}
    variant_data: dict[str, dict] = {}
    try:
        for variant_key, template_key, needs_class_list, variant_label in selected_variants:
            variant_dir = output_dir / variant_key
            print(f"\n=== {variant_label} ({variant_key}) ===")

            # ── Phase 3's own cheap greedy-temperature summary (vs. the
            # deterministic `correct` column) — a fresh judge call for
            # EVERY variant, including "current" (whose template_key is
            # exactly judge_classification_template, i.e. what the
            # checkpoint's own judge phase would have used) — never reuses
            # a possibly-blank stored judged_correct, so this works
            # whether or not Phase 0 ran with --skip_judge.
            if needs_class_list:
                add_class_list_field(rows, dataset_classes)
            extra_keys = {"class_list": "class_list_text"} if needs_class_list else None
            rechecked = rejudge_condition(
                rows, judge_llm, judge_tok, prompts,
                confidence_threshold=args.judge_confidence_threshold,
                chunk_size=args.judge_chunk_size,
                log_path=variant_dir / "greedy_summary_judge_log.jsonl",
                pred_key="prediction", template_key=template_key,
                progress_desc=f"[greedy summary] Re-judging {label} ({variant_label})",
                extra_keys=extra_keys,
                enable_thinking=args.judge_enable_thinking, judge_max_tokens=args.judge_max_tokens,
                judge_chat_template=judge_chat_template,
            )
            rows_by_variant[variant_key] = rechecked
            print(f"  [greedy summary] {len(rechecked)} row(s) re-judged.")

            variant_dir.mkdir(parents=True, exist_ok=True)
            with open(variant_dir / "greedy_summary_rechecked.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=GREEDY_SUMMARY_FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in rechecked:
                    w.writerow(r)

            # ── Phase 1 (full temperature sweep) for this variant ──
            leniency_dir = variant_dir / "leniency"
            leniency_args = _variant_args(
                args, leniency_dir, f"{label} — {variant_label}", template_key, needs_class_list,
            )
            judge_leniency.main(leniency_args, judge_llm=judge_llm, judge_tok=judge_tok)

            # ── Phase 2 (full temperature sweep) for this variant, wired
            # to Phase 1's just-written leniency_gap.json for the
            # correlation plot — skippable via --skip_consistency, since
            # flip-rate is a repeated-sampling metric (n_repeats judge
            # calls per row per condition per temperature) and by far the
            # most expensive part of a variant's run; nothing in the
            # cross-variant summary strictly needs it EXCEPT the LaTeX
            # table's "Flip rate" column (that table just won't cover a
            # variant missing flip_rate.json, same graceful-partial-
            # coverage behavior as a variant that hasn't run at all yet). ──
            consistency_dir = variant_dir / "consistency"
            if args.skip_consistency:
                print(f"  [Phase 2] skipped (--skip_consistency) — leniency_and_false_positive_summary.png "
                      f"and judge_accuracy_f1.png don't need it; the LaTeX table's Flip rate column will "
                      f"be missing for this variant.")
            else:
                consistency_args = _variant_args(
                    args, consistency_dir, f"{label} — {variant_label}", template_key, needs_class_list,
                    n_repeats=args.n_repeats,
                    leniency_gap_json=str(leniency_dir / "leniency_gap.json"),
                    save_logs=args.save_logs,
                )
                judge_consistency.main(consistency_args, judge_llm=judge_llm, judge_tok=judge_tok)

            # ── Collect this variant's Phase 1/2 results (read back from
            # the JSON each just wrote) for the cross-variant LaTeX table,
            # and copy the correlation plot up to the variant's own root
            # folder for visibility (it's naturally produced inside
            # consistency/ since that's where judge_consistency.main()
            # writes it, but "within this folder" for a quick look
            # shouldn't require descending into consistency/ first) ──
            leniency_json_path = leniency_dir / "leniency_gap.json"
            flip_json_path = consistency_dir / "flip_rate.json"
            if leniency_json_path.exists() and flip_json_path.exists():
                leniency_data = json.loads(leniency_json_path.read_text())
                flip_data = json.loads(flip_json_path.read_text())
                variant_data[variant_key] = {
                    "leniency_by_temp": {float(t): d for t, d in leniency_data["leniency_gap_by_temperature"].items()},
                    # The table's "Flip rate" column reports the full-response
                    # condition — flip_rate_by_temperature_answer_only is still
                    # available in flip_rate.json and its own PNGs, just not
                    # folded into this table (would need a 5th column).
                    "flip_by_temp": {
                        float(t): d for t, d in flip_data["flip_rate_by_temperature_full"].items()
                    },
                    "disagreement_by_temp": {
                        float(t): d for t, d in leniency_data.get("disagreement_by_temperature", {}).items()
                    },
                }
            # Promote both correlation plots (one per condition) up to the
            # variant's own root folder for visibility.
            for corr_suffix in ("full", "answer_only"):
                corr_src = consistency_dir / f"leniency_vs_flip_rate_{corr_suffix}.png"
                if corr_src.exists():
                    shutil.copy(corr_src, variant_dir / corr_src.name)
            # Same promotion for the disagreement-matrix figures — one per
            # swept temperature, naturally written inside leniency/ by
            # judge_leniency.main(), copied up so this variant's own
            # full-response/resolved-class disagreement breakdown is
            # visible without descending into leniency/ first.
            for dis_src in sorted(leniency_dir.glob("disagreement_matrix_T*.png")):
                shutil.copy(dis_src, variant_dir / dis_src.name)
    finally:
        _destroy_judge_llm(judge_llm)
        del judge_tok

    # ── Cross-variant summary covers every variant with COMPLETED output on
    # disk, not just the ones --template selected THIS invocation — so
    # running "current", then later "specificity", then later "class_list"
    # (each its own job/invocation, same --output_dir) progressively fills
    # in template_sensitivity.png / .tex / length_bias.png without ever
    # re-judging a variant that already finished. ──
    for variant_key, template_key, needs_class_list, variant_label in TEMPLATE_VARIANTS:
        variant_dir = output_dir / variant_key
        if variant_key not in rows_by_variant:
            greedy_csv = variant_dir / "greedy_summary_rechecked.csv"
            if greedy_csv.exists():
                with open(greedy_csv, encoding="utf-8") as f:
                    rows_by_variant[variant_key] = list(csv.DictReader(f))
                print(f"  [{variant_label}] loaded {len(rows_by_variant[variant_key])} row(s) "
                      f"from a prior invocation's greedy_summary_rechecked.csv.")
        if variant_key not in variant_data:
            leniency_json_path = variant_dir / "leniency" / "leniency_gap.json"
            flip_json_path = variant_dir / "consistency" / "flip_rate.json"
            if leniency_json_path.exists() and flip_json_path.exists():
                leniency_data = json.loads(leniency_json_path.read_text())
                flip_data = json.loads(flip_json_path.read_text())
                variant_data[variant_key] = {
                    "leniency_by_temp": {float(t): d for t, d in leniency_data["leniency_gap_by_temperature"].items()},
                    "flip_by_temp": {float(t): d for t, d in flip_data["flip_rate_by_temperature_full"].items()},
                    "disagreement_by_temp": {
                        float(t): d for t, d in leniency_data.get("disagreement_by_temperature", {}).items()
                    },
                }

    template_results = score_against_deterministic(rows_by_variant)
    plot_template_sensitivity(template_results, output_dir / "template_sensitivity.png")
    for variant_key, variant_results in template_results.items():
        d = variant_results["overall"]
        print(f"  {variant_key}: judged_correct_rate={d['judged_correct_rate']:.4f}, "
              f"deterministic_correct_rate={d['deterministic_correct_rate']:.4f}, gap={d['gap']:+.4f}")

    length_report = None
    if "current" not in rows_by_variant:
        print("  [length_bias] skipped — \"current\" variant hasn't been run yet in this --output_dir.")
    else:
        length_report = length_bias_report(rows_by_variant["current"], n_buckets=args.n_length_buckets)
        plot_length_bias(length_report, output_dir / "length_bias.png")

    # ── Phase 4 output B1: judge-scored F1 + accuracy across every variant
    # with data on disk so far (same rows_by_variant used above). ──
    accuracy_f1_results = compute_judge_accuracy_f1(rows_by_variant) if rows_by_variant else {}
    if accuracy_f1_results:
        plot_judge_accuracy_f1(accuracy_f1_results, output_dir / "judge_accuracy_f1.png")
        for variant_key, d in accuracy_f1_results.items():
            print(f"  {variant_key}: judge_accuracy={d['accuracy']:.4f}, judge_f1_macro={d['f1_macro']:.4f}")

    # ── Phase 4 output B2: leniency_gap / false-positive-rate two-panel
    # summary at --summary_temperature — reads each variant's OWN
    # leniency/leniency_gap.json + rechecked_T<summary_temperature>.csv
    # (Phase 1's per-temperature output), skipping (not erroring) any
    # variant whose --temperatures sweep didn't include summary_temperature. ──
    leniency_summary_by_variant: dict[str, dict] = {}
    fp_summary_by_variant: dict[str, dict] = {}
    for variant_key, template_key, needs_class_list, variant_label in TEMPLATE_VARIANTS:
        variant_dir = output_dir / variant_key
        leniency_json_path = variant_dir / "leniency" / "leniency_gap.json"
        rechecked_csv_path = variant_dir / "leniency" / f"rechecked_T{args.summary_temperature}.csv"
        if not leniency_json_path.exists() or not rechecked_csv_path.exists():
            continue
        leniency_data = json.loads(leniency_json_path.read_text())
        temp_key = str(args.summary_temperature)
        if temp_key not in leniency_data["leniency_gap_by_temperature"]:
            continue
        leniency_summary_by_variant[variant_key] = leniency_data["leniency_gap_by_temperature"][temp_key]["overall"]
        with open(rechecked_csv_path, encoding="utf-8") as f:
            rechecked_rows = list(csv.DictReader(f))
        fp_summary_by_variant[variant_key] = false_positive_credit_rate(rechecked_rows)

    if leniency_summary_by_variant and fp_summary_by_variant:
        plot_leniency_and_false_positive_summary(
            leniency_summary_by_variant, fp_summary_by_variant,
            output_dir / "leniency_and_false_positive_summary.png",
        )
        (output_dir / "leniency_and_false_positive_summary_stats.json").write_text(json.dumps(
            {"summary_temperature": args.summary_temperature,
             "leniency_by_variant": leniency_summary_by_variant,
             "false_positive_by_variant": fp_summary_by_variant},
            indent=2,
        ))
        print(f"  Wrote leniency_and_false_positive_summary.png covering "
              f"{sorted(leniency_summary_by_variant)} at T={args.summary_temperature} "
              f"(re-run with --summary_temperature or add it to --temperatures for any variant missing)")
    else:
        print(f"  [leniency_and_false_positive_summary] skipped — no variant has both leniency_gap.json "
              f"and rechecked_T{args.summary_temperature}.csv (need --summary_temperature "
              f"{args.summary_temperature} included in that run's --temperatures).")

    # ── Phase 4 output B3: oracle validation CSV (hybrid vs. current). ──
    oracle_stats = write_oracle_validation_csv(rows_by_variant, output_dir / "oracle_validation_sample.csv")
    if oracle_stats is None:
        print("  [oracle_validation_sample.csv] skipped — needs both \"hybrid\" and \"current\" variants "
              "present in this --output_dir.")
    else:
        print(f"  Wrote oracle_validation_sample.csv: {oracle_stats['n_total']} row(s) "
              f"({oracle_stats['n_disagreement']} disagreement, {oracle_stats['n_agreement']} agreement)")

    (output_dir / "template_sensitivity.json").write_text(json.dumps(
        {"label": label, "n_rows": len(rows), "template_sensitivity": template_results,
         "length_bias": length_report, "judge_accuracy_f1": accuracy_f1_results,
         "summary_temperature": args.summary_temperature,
         "leniency_and_false_positive_summary": {
             "leniency_by_variant": leniency_summary_by_variant,
             "false_positive_by_variant": fp_summary_by_variant,
         } if leniency_summary_by_variant else None},
        indent=2,
    ))

    if variant_data:
        write_variant_temperature_latex_table(variant_data, output_dir / "template_sensitivity_table.tex")
        print(f"Wrote template_sensitivity_table.tex to {output_dir} "
              f"(covering {sorted(variant_data)} — re-run with --template for any variant still missing)")

    print(f"\nWrote template_sensitivity.png, template_sensitivity.json"
          f"{', length_bias.png' if length_report is not None else ''}"
          f"{', judge_accuracy_f1.png' if accuracy_f1_results else ''}"
          f"{', leniency_and_false_positive_summary.png' if leniency_summary_by_variant else ''}"
          f"{', oracle_validation_sample.csv' if oracle_stats else ''} to {output_dir}")
    print(f"Per-variant Phase 1/2 output under {output_dir}/<variant>/leniency/ and /<variant>/consistency/ "
          f"(leniency_vs_flip_rate_{{full,answer_only}}.png also copied to {output_dir}/<variant>/ directly)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_template_sensitivity/")
    parser.add_argument("--label", default=None, help="Report/log label (defaults to the ckpt filename).")
    parser.add_argument("--template", choices=["all", "current", "specificity", "class_list", "hybrid"], default="all",
                         help="Run just ONE template variant this invocation instead of all 4 — lets each "
                              "variant be its own job/run. The cross-variant summary "
                              "(template_sensitivity.png/.tex, length_bias.png) reads back EVERY variant "
                              "that has completed output under --output_dir so far (not just the one just "
                              "run), so running \"current\", then later \"specificity\", then later "
                              "\"class_list\", then later \"hybrid\" (Phase 4 — class_list's context + "
                              "specificity's guardrail + credits necessary category membership without "
                              "requiring exact wording) against the SAME --output_dir progressively "
                              "completes the summary without re-judging anything already done.")
    parser.add_argument("--config", default="src/config/datasets.yaml", help="datasets.yaml, for each row's class list.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7],
                         help="Swept judge temperatures, passed through to both Phase 1 and Phase 2 "
                              "for every variant.")
    parser.add_argument("--summary_temperature", type=float, default=0.7,
                         help="Which swept temperature the Phase 4 leniency_and_false_positive_summary.png "
                              "reads from each variant's rechecked_T<temp>.csv/leniency_gap.json — MUST be "
                              "one of --temperatures' values (e.g. pass --temperatures 0.1 0.5 0.7 1.0 to "
                              "use the 0.7 default), or that variant is skipped with a printed note rather "
                              "than silently substituting a different temperature.")
    parser.add_argument("--n_repeats", type=int, default=5,
                         help="Phase 2's repeat judge calls per row, per temperature — ignored if "
                              "--skip_consistency is set.")
    parser.add_argument("--skip_consistency", action="store_true",
                         help="Skip Phase 2 (flip-rate/consistency) entirely for every variant this "
                              "invocation — by far the most expensive part of a variant's run (n_repeats "
                              "judge calls per row per condition per temperature, vs. Phase 1's single "
                              "call). Nothing else needs it: judge_accuracy_f1.png, "
                              "leniency_and_false_positive_summary.png, template_sensitivity.png, and "
                              "oracle_validation_sample.csv all read only Phase 1 / greedy-summary output. "
                              "The one thing that degrades: template_sensitivity_table.tex's Flip rate "
                              "column, and the leniency_vs_flip_rate_*.png correlation plots, won't exist "
                              "for a variant run this way.")
    parser.add_argument("--save_logs", action="store_true",
                         help="Phase 2: write one judge_log.jsonl per (variant, temperature, repeat) — "
                              "off by default, this multiplies fast across 4 variants. Ignored if "
                              "--skip_consistency is set.")
    parser.add_argument("--n_length_buckets", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample to this many rows first, applied ONCE before any variant runs "
                              "(each variant re-runs the full Phase 1+2 sweep over the same subsample) — "
                              "start small, this is the most expensive script in the plan.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--judge_enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Attempt tokenizer.apply_chat_template(..., enable_thinking=True) for the "
                              "judge's own prompt on every variant/phase call — ON by default (this repo's "
                              "judge model is Gemma-4, and --judge_chat_template_path defaults to the "
                              "template that makes this work for it). Pass --no-judge_enable_thinking to "
                              "fall back to the exact original prompt rendering.")
    parser.add_argument("--judge_max_tokens", type=int, default=2048,
                         help="Override the judge's default max_tokens=1024 for every call this run — a "
                              "thinking trace (--judge_enable_thinking, on by default) can consume the "
                              "normal budget before reaching the final JSON verdict, truncating the response.")
    parser.add_argument("--judge_chat_template_path", default="src/config/tool_chat_template_gemma4.jinja",
                         help="Override the judge tokenizer's own bundled chat template with a Jinja "
                              "file at this path, for every variant/phase call — needed for "
                              "--judge_enable_thinking to do anything on a Gemma-4 judge model (the "
                              "default judge model here). Pass an empty string "
                              "(--judge_chat_template_path '') to disable for a non-Gemma-4 judge model.")
    args = parser.parse_args()
    if not args.judge_chat_template_path:
        args.judge_chat_template_path = None
    main(args)
