"""
Paper 2 Stage 0 — simple judge: grade an existing baseline checkpoint's
predictions with the hybrid judge template
(src/common/metrics.py's DEFAULT_JUDGE_TEMPLATE_KEY,
prompts.yaml's judge_classification_hybrid_template) — ONE judging pass,
no leniency/consistency/template-sensitivity comparison. Reuses
run_judge_phase directly, the exact same judge mechanism vision-baseline's
own inline judge phase uses.

Copies --ckpt into --output_dir first — the ORIGINAL checkpoint is never
mutated. This is the point: to try a different hybrid-template wording
(hand-edit prompts.yaml's judge_classification_hybrid_template), just
point --output_dir at a fresh folder and re-run with --rejudge — each
variant's results land in their own directory, nothing overwrites a
previous variant's.

Usage:
    uv run python main.py judge-checkpoint \\
        --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --output_dir outputs_final/judge_variants/hybrid_v2/

Re-judging the SAME checkpoint under a different template wording
(--output_dir points somewhere new, --rejudge forces every row to be
re-graded instead of just ones missing a verdict):
    uv run python main.py judge-checkpoint \\
        --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --output_dir outputs_final/judge_variants/hybrid_v3/ \\
        --rejudge
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import shutil

from src.common.io import load_prompts
from src.common.vision_eval import load_ckpt, run_judge_phase
from src.vision_baseline import VISION_BASELINE_CKPT_FIELDS


def main(args: argparse.Namespace) -> None:
    src_ckpt = Path(args.ckpt)
    if not src_ckpt.exists():
        raise SystemExit(f"ERROR: {src_ckpt} does not exist.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = out_dir / src_ckpt.name

    if not dest_ckpt.exists():
        shutil.copy2(src_ckpt, dest_ckpt)
        print(f"  [judge-checkpoint] copied {src_ckpt} -> {dest_ckpt}")
    else:
        print(f"  [judge-checkpoint] {dest_ckpt} already exists — judging/resuming in place "
              f"(pass --rejudge to force every row to be re-graded, or point --output_dir at a "
              f"fresh folder to start clean from --ckpt again)")

    rows, _ = load_ckpt(dest_ckpt, fields=VISION_BASELINE_CKPT_FIELDS)
    if not rows:
        raise SystemExit(f"ERROR: {dest_ckpt} has no rows.")

    prompts = load_prompts(Path(args.prompts))
    # run_judge_phase's shared arg contract requires a skip_judge
    # attribute — hardcoded off rather than exposed as a real CLI flag,
    # since passing it would defeat the entire point of this script.
    args.skip_judge = False
    rows = run_judge_phase(rows, dest_ckpt, args, prompts, fields=VISION_BASELINE_CKPT_FIELDS)

    judged = [r for r in rows if r.get("judged_correct") not in (None, "")]
    if judged:
        acc = sum(int(r["judged_correct"]) for r in judged) / len(judged)
        print(f"\n  judge_accuracy={acc:.4f} (n={len(judged)}/{len(rows)} judged)")
        print(f"  Checkpoint: {dest_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A vision-baseline (VISION_BASELINE_CKPT_FIELDS-shaped) checkpoint CSV "
                              "to judge. Never modified in place — copied into --output_dir first.")
    parser.add_argument("--output_dir", required=True,
                         help="Where the copied checkpoint + judge_log.jsonl are written. Point this "
                              "at a fresh folder per hybrid-template wording you're trying, so "
                              "different variants' results never overwrite each other.")
    parser.add_argument("--config", default="src/config/datasets.yaml",
                         help="Used to fill the hybrid template's {class_list} placeholder — same "
                              "convention as vision-baseline's own judge phase.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--rejudge", action="store_true",
                         help="Clear judged_correct/judge_reasoning on every row and re-judge from "
                              "scratch — use this after hand-editing prompts.yaml's hybrid template "
                              "to re-grade everything under the new wording. Without it, only rows "
                              "still missing a verdict get judged (plain resume).")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--judge_enable_thinking", action="store_true",
                         help="Needs --judge_chat_template_path too for a Gemma-4 judge — its stock "
                              "template has no thinking support at all.")
    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--judge_chat_template_path", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")

    main(parser.parse_args())
