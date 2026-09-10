"""Judge self-consistency — Phase 2 of the judge-improvements evaluation plan
("LLM as a judge improvements" note). Loads the judge model and repeats the
current judge_classification_template call (full model response, same as
the checkpoint's original judge pass) N times per row at each of a swept set
of temperatures, then measures how often the verdict flips across repeats.

Needs a GPU/judge model — NOT a pure-CPU step, and (unlike Phase 1's
leniency check) forces real sampling: at the pipeline's normal
temperature=0.0, repeat judge calls are bit-for-bit identical and there is
nothing to measure a flip rate over.

Writes, per checkpoint:
  flip_rate_by_temperature.png   — flip_rate per task_type, grouped by
                                    temperature
  leniency_vs_flip_rate.png      — only if --leniency_gap_json is given
                                    (Phase 1's leniency_gap.json for the SAME
                                    checkpoint): scatter of leniency_gap vs.
                                    flip_rate across task_types x temperatures
  flip_rate.json                 — machine-readable, per temperature/task_type

Usage:
    uv run python main.py judge-consistency \\
        --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \\
        --output_dir ablation/judge_consistency/vision_baseline_e4b/ \\
        --label "Vision Baseline — Gemma-4-E4B-it" \\
        --temperatures 0.1 0.5 1.0 --n_repeats 5 \\
        --leniency_gap_json ablation/judge_leniency/vision_baseline_e4b/leniency_gap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.io import load_prompts
from src.common.judge_consistency import (
    compute_flip_rate, plot_flip_rate_by_temperature, plot_leniency_vs_flip_rate, repeat_judge_calls,
)
from src.common.vision_eval import _destroy_judge_llm, _load_judge_llm
from src.judge_leniency import load_ckpt_rows


def main(args: argparse.Namespace) -> None:
    rows = load_ckpt_rows(args.ckpt)
    missing = [c for c in ("judged_correct", "prediction", "task_type") if c not in rows[0]]
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

    print(f"  {len(rows)} row(s), {len(args.temperatures)} temperature(s) x {args.n_repeats} repeat(s) "
          f"= {len(rows) * len(args.temperatures) * args.n_repeats} judge call(s) total.")

    prompts = load_prompts(Path(args.prompts))
    judge_llm, judge_tok = _load_judge_llm(
        args.judge_model, quantization=args.judge_quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
    )
    try:
        rows_by_temp = repeat_judge_calls(
            rows, judge_llm, judge_tok, prompts,
            confidence_threshold=args.judge_confidence_threshold,
            chunk_size=args.judge_chunk_size,
            temperatures=args.temperatures, n_repeats=args.n_repeats,
            log_dir=output_dir if args.save_logs else None,
        )
    finally:
        _destroy_judge_llm(judge_llm)
        del judge_tok

    flip_rate_by_temp = {t: compute_flip_rate(rows_by_temp[t]) for t in args.temperatures}
    plot_flip_rate_by_temperature(flip_rate_by_temp, output_dir / "flip_rate_by_temperature.png")

    (output_dir / "flip_rate.json").write_text(json.dumps(
        {"label": label, "n_rows": len(rows), "n_repeats": args.n_repeats,
         "flip_rate_by_temperature": {str(t): d for t, d in flip_rate_by_temp.items()}},
        indent=2,
    ))

    for t in args.temperatures:
        d = flip_rate_by_temp[t]["overall"]
        print(f"  T={t}: overall flip_rate={d['flip_rate']:.4f}, mean_judged_correct={d['mean_judged_correct']:.4f}")
    print(f"Wrote flip_rate_by_temperature.png, flip_rate.json to {output_dir}")

    if args.leniency_gap_json:
        leniency_data = json.loads(Path(args.leniency_gap_json).read_text())
        leniency_by_temp = {float(t): d for t, d in leniency_data["leniency_gap_by_temperature"].items()}
        corr_stats = plot_leniency_vs_flip_rate(
            leniency_by_temp, flip_rate_by_temp,
            output_dir / "leniency_vs_flip_rate.png",
        )
        (output_dir / "leniency_vs_flip_rate_stats.json").write_text(
            json.dumps(corr_stats if corr_stats is not None else {"note": "not enough points/variance to compute"}, indent=2)
        )
        print(f"Wrote leniency_vs_flip_rate.png, leniency_vs_flip_rate_stats.json to {output_dir}")
        if corr_stats:
            for temp_label, d in corr_stats.items():
                print(f"  {temp_label}: Pearson r={d['pearson_r']:.2f}, p={d['p_value']:.3f}, n={d['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="Checkpoint CSV path, glob pattern, or a model output directory "
                              "(globs ckpt_*.csv inside it).")
    parser.add_argument("--output_dir", default="ablation/judge_consistency/")
    parser.add_argument("--label", default=None, help="Report/log label (defaults to the ckpt filename).")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--n_repeats", type=int, default=5, help="Repeat judge calls per row, per temperature.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample to this many rows first (repeat calls multiply cost by "
                              "n_temperatures * n_repeats — start small).")
    parser.add_argument("--leniency_gap_json", default=None,
                         help="Phase 1's leniency_gap.json for the SAME checkpoint — if given, also "
                              "writes leniency_vs_flip_rate.png.")
    parser.add_argument("--save_logs", action="store_true",
                         help="Write one judge_log.jsonl per (temperature, repeat) — off by default, "
                              "this is n_temperatures * n_repeats files.")
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
