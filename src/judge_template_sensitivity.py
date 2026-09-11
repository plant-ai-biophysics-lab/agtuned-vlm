"""Judge template sensitivity sweep — Phase 3 of the judge-improvements
evaluation plan ("LLM as a judge improvements" note). For each of three
judge templates (see src/common/judge_template_sensitivity.py's
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
otherwise be paid for 3 variants x 2 phases.

Needs a GPU/judge model. This is the most expensive script in the judge-
improvements evaluation plan by far: 3 variants x (leniency's 2 sides x
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
                                             across all 3 variants vs. the
                                             deterministic baseline
  length_bias.png                         — length/verbosity bias, current template
  template_sensitivity_table.tex          — LaTeX booktabs table: leniency_gap,
                                             flip_rate, and both disagreement-
                                             direction counts/rates (overall,
                                             with 95% CI) per variant x temperature
  template_sensitivity.json               — template_sensitivity.png +
                                             length_bias.png's data, machine-readable

Usage:
    uv run python main.py judge-template-sensitivity \\
        --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \\
        --output_dir ablation/judge_template_sensitivity/vision_baseline_e4b/ \\
        --label "Vision Baseline — Gemma-4-E4B-it" \\
        --temperatures 0.1 0.5 1.0 --n_repeats 5 \\
        --judge_model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import src.judge_consistency as judge_consistency
import src.judge_leniency as judge_leniency
from src.common.ablation_judge_recheck import rejudge_condition
from src.common.dataset_config import datasets_by_name
from src.common.io import load_prompts
from src.common.judge_template_sensitivity import (
    TEMPLATE_VARIANTS, add_class_list_field, length_bias_report,
    plot_length_bias, plot_template_sensitivity, score_against_deterministic,
    write_variant_temperature_latex_table,
)
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm
from src.judge_leniency import load_ckpt_rows


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
        **overrides,
    )


def main(args: argparse.Namespace) -> None:
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("judged_correct", "correct", "prediction", "agml_dataset", "task_type") if c not in rows[0]]
    if missing:
        raise ValueError(f"--ckpt is missing expected column(s) {missing}.")
    rows = [r for r in rows if str(r.get("judged_correct", "")).strip() != ""]
    if not rows:
        raise ValueError("--ckpt has no judged rows (judged_correct is blank everywhere) — run the judge phase first.")
    if args.limit:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.ckpt).stem

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
        for variant_key, template_key, needs_class_list, variant_label in TEMPLATE_VARIANTS:
            variant_dir = output_dir / variant_key
            print(f"\n=== {variant_label} ({variant_key}) ===")

            # ── Phase 3's own cheap greedy-temperature summary (vs. the
            # deterministic `correct` column) — "current" reuses the
            # checkpoint's already-stored judged_correct, no fresh call.
            if variant_key == "current":
                rows_by_variant[variant_key] = rows
                print(f"  [greedy summary] reusing checkpoint's judged_correct (no fresh call).")
            else:
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
                )
                rows_by_variant[variant_key] = rechecked
                print(f"  [greedy summary] {len(rechecked)} row(s) re-judged.")

            # ── Phase 1 (full temperature sweep) for this variant ──
            leniency_dir = variant_dir / "leniency"
            leniency_args = _variant_args(
                args, leniency_dir, f"{label} — {variant_label}", template_key, needs_class_list,
            )
            judge_leniency.main(leniency_args, judge_llm=judge_llm, judge_tok=judge_tok)

            # ── Phase 2 (full temperature sweep) for this variant, wired
            # to Phase 1's just-written leniency_gap.json for the
            # correlation plot ──
            consistency_dir = variant_dir / "consistency"
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

    template_results = score_against_deterministic(rows_by_variant)
    plot_template_sensitivity(template_results, output_dir / "template_sensitivity.png")
    for variant_key, variant_results in template_results.items():
        d = variant_results["overall"]
        print(f"  {variant_key}: judged_correct_rate={d['judged_correct_rate']:.4f}, "
              f"deterministic_correct_rate={d['deterministic_correct_rate']:.4f}, gap={d['gap']:+.4f}")

    length_report = length_bias_report(rows_by_variant["current"], n_buckets=args.n_length_buckets)
    plot_length_bias(length_report, output_dir / "length_bias.png")

    (output_dir / "template_sensitivity.json").write_text(json.dumps(
        {"label": label, "n_rows": len(rows), "template_sensitivity": template_results,
         "length_bias": length_report},
        indent=2,
    ))

    if variant_data:
        write_variant_temperature_latex_table(variant_data, output_dir / "template_sensitivity_table.tex")
        print(f"Wrote template_sensitivity_table.tex to {output_dir}")

    print(f"\nWrote template_sensitivity.png, length_bias.png, template_sensitivity.json to {output_dir}")
    print(f"Per-variant Phase 1/2 output under {output_dir}/<variant>/leniency/ and /<variant>/consistency/ "
          f"(leniency_vs_flip_rate_{{full,answer_only}}.png also copied to {output_dir}/<variant>/ directly)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_template_sensitivity/")
    parser.add_argument("--label", default=None, help="Report/log label (defaults to the ckpt filename).")
    parser.add_argument("--config", default="src/config/datasets.yaml", help="datasets.yaml, for each row's class list.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                         help="Swept judge temperatures, passed through to both Phase 1 and Phase 2 "
                              "for every variant.")
    parser.add_argument("--n_repeats", type=int, default=5, help="Phase 2's repeat judge calls per row, per temperature.")
    parser.add_argument("--save_logs", action="store_true",
                         help="Phase 2: write one judge_log.jsonl per (variant, temperature, repeat) — "
                              "off by default, this multiplies fast across 3 variants.")
    parser.add_argument("--n_length_buckets", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample to this many rows first, applied ONCE before any variant runs "
                              "(each variant re-runs the full Phase 1+2 sweep over the same subsample) — "
                              "start small, this is the most expensive script in the plan.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()
    main(args)
