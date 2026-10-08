"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 2 — reveal ground
truth on EVERY heldout-zeroshot row (not just misses) and ask the model
to explain the gap for a wrong answer, or confirm/articulate the
supporting evidence for a right one. Reads Step 1's own checkpoint
(--ckpt) — all of it, both judged_correct == 0 and == 1 — and for each
row shows the model its OWN prediction (the full narrative Step 1 now
stores, not just the bare answer — see heldout_zeroshot.py's own
prediction-field fix), the REAL ground truth label, and the dataset's
expert-written `details` observations.

Two prompt templates (src/config/heldout_prompts.yaml), chosen per row
by judged_correct:
  - heldout_explain_miss_prompt ("what didn't work") — explain what
    evidence should have led to the correct answer, and where the
    model's own stated reasoning went wrong.
  - heldout_explain_correct_prompt ("what worked") — confirm and
    articulate the evidence that correctly supports the answer. Explicitly
    instructed to call out a right-answer-for-the-wrong-reasons case
    (a lucky guess) rather than crediting unsound reasoning just because
    the final label happened to match.

Deliberately NOT reusing --verifier_type verifier_rubric's own
evidence-gathering mechanism for this — Stage 4 found that call can
pre-commit to a wrong diagnosis during "neutral" observation; this step
needs the real label (and the real expert's own observations) revealed
up front to produce a trustworthy explanation, not another guess.

System prompt (heldout_lesson_system) bakes in the GENERAL error trends
found in Step 1's own real results (heavy "Healthy" over-prediction,
weak pest-damage recognition, weak nutrient-deficiency recognition) as
reminders, not per-image facts — see src/config/heldout_prompts.yaml's
own header comment for the real numbers behind each one.

Usage:
    uv run python main.py heldout-explain-traces \\
        --ckpt outputs_final/heldout_zeroshot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv

from PIL import Image as PILImage

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_thinking
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.progress import ProgressBar
from src.heldout_zeroshot import HELDOUT_CKPT_FIELDS

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

HELDOUT_EXPLAIN_CKPT_FIELDS = HELDOUT_CKPT_FIELDS + ["trace_type", "explanation"]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt_text(r: dict, miss_template: str, correct_template: str) -> tuple[str, str]:
    """Returns (trace_type, prompt_text) — judged_correct picks the
    template; `prediction` is Step 1's full narrative (not the bare
    extracted answer) in both cases, so the explanation can address the
    model's own stated reasoning, not just its final label."""
    if r.get("judged_correct") == "1":
        text = correct_template.format(
            crop=r["crop"], model_reasoning=r["prediction"], correct_prediction=r["extracted_answer"],
            ground_truth=r["class_label"], expert_details=r["details"],
        )
        return "worked", text
    text = miss_template.format(
        crop=r["crop"], model_reasoning=r["prediction"], wrong_prediction=r["extracted_answer"],
        ground_truth=r["class_label"], expert_details=r["details"],
    )
    return "didnt_work", text


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prefix = "heldout_quality_" if getattr(args, "domain", "disease_pest") == "quality" else "heldout_"
    hprompts = load_yaml(Path(args.heldout_prompts))
    system_prompt = hprompts[f"{prefix}lesson_system"]
    miss_template = hprompts[f"{prefix}explain_miss_prompt"]
    correct_template = hprompts[f"{prefix}explain_correct_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    with open(args.ckpt, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    judged = [r for r in input_rows if r.get("judged_correct") in ("0", "1")]
    if not judged:
        raise SystemExit(f"ERROR: {args.ckpt} has no judged rows — nothing to explain.")
    n_worked = sum(1 for r in judged if r["judged_correct"] == "1")
    print(f"  {len(judged)} judged row(s) out of {len(input_rows)} in {args.ckpt} "
          f"({n_worked} worked, {len(judged) - n_worked} didn't)")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=HELDOUT_EXPLAIN_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} rows already explained in {ckpt_path.name}")

    pending = [r for r in judged if (r["agml_dataset"], r["class_label"], r["image_name"]) not in seen]
    if not pending:
        print("  Nothing to do — all rows already checkpointed.")
        return
    print(f"  {len(pending)} pending row(s)")

    images_root = Path(args.images_root)
    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        kv_cache_dtype=args.kv_cache_dtype,
        caller_tag="heldout_explain_traces",
    )

    chunk_size = max(args.chunk_size, 1)
    pbar = ProgressBar(total=len(pending), desc="Explaining")
    for i in range(0, len(pending), chunk_size):
        chunk = pending[i : i + chunk_size]
        images = [PILImage.open(images_root / r["image_name"]).convert("RGB") for r in chunk]
        trace_types_and_texts = [build_prompt_text(r, miss_template, correct_template) for r in chunk]
        histories = [
            [{"role": "system", "content": system_prompt}]
            + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
            for _, t in trace_types_and_texts
        ]
        prompts_batch = [
            {"prompt": render_chat_template(tokenizer, h, enable_thinking=args.enable_thinking,
                                             chat_template=chat_template),
             "multi_modal_data": {"image": img}}
            for h, img in zip(histories, images)
        ]
        params = SamplingParams(
            temperature=0.0, max_tokens=args.gen_max_tokens, seed=args.seed,
            skip_special_tokens=not args.enable_thinking,
        )
        outputs = llm.generate(prompts_batch, sampling_params=params)

        for r, (trace_type, _), out in zip(chunk, trace_types_and_texts, outputs):
            raw = out.outputs[0].text
            # Unconditional — see extract_thinking's own docstring; strips
            # either Gemma-4's channel format or Qwen's <think> tags.
            _, cleaned = extract_thinking(raw)
            explanation = cleaned.strip()
            row = dict(r)
            row["trace_type"] = trace_type
            row["explanation"] = explanation
            append_ckpt_row(ckpt_path, row, fields=HELDOUT_EXPLAIN_CKPT_FIELDS)
            rows.append(row)
        pbar.update(len(chunk))
    pbar.finish()
    destroy_llm(llm)
    del tokenizer

    n_worked_out = sum(1 for r in rows if r["trace_type"] == "worked")
    print(f"\n  {len(rows)} total trace(s) ({n_worked_out} worked, {len(rows) - n_worked_out} didn't).")
    print(f"  Checkpoint: {ckpt_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--domain", choices=["disease_pest", "quality"], default="disease_pest",
                         help="Selects which heldout_prompts.yaml template set to use "
                              "(heldout_* vs heldout_quality_*).")
    parser.add_argument("--ckpt", required=True,
                         help="heldout-zeroshot's own output checkpoint CSV — ALL judged rows "
                              "(judged_correct in {0,1}) are processed, not just misses.")
    parser.add_argument("--images_root", default="dataset/images_heldout")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_explain_traces/")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Same mechanism/default as every other generation stage — on by "
                              "default. Pass --no-enable_thinking to opt out.")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096,
                         help="Short explanation output, but real headroom for a thinking trace first "
                              "— bumped from an original 2048 after a real run found ~5% of rows "
                              "(40/827) truncating mid-thinking-trace, which leaves the closing "
                              "<channel|> marker missing so extract_channel_thought's regex can't "
                              "match and falls back to the RAW unstripped <|channel>thought...text "
                              "leaking straight into the explanation column (confirmed: those rows' "
                              "explanation length was 15-20x the median).")
    parser.add_argument("--chunk_size", type=int, default=64,
                         help="Rows per llm.generate() batch (checkpointed after each chunk).")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
