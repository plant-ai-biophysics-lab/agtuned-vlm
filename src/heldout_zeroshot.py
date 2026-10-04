"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 1 — zero-shot
classification on the held-out Digital Green dataset's lesson-building
slice. Plain single-shot prediction (no rubric, no verifier — same shape
as vision-baseline's own lower bound), judged against the real expert
`diagnosis` label via the pipeline-wide hybrid judge template.

Reads dataset/heldout_selected_images.json (download-heldout-digitalgreen's
own output) — restricted to --split (default "lessons"; the "eval" split
must never be touched until Step 4's ablation). Judge's own {class_list}
placeholder (checks whether the prediction is SPECIFIC enough to
distinguish the reference diagnosis from every other one, not just
broadly consistent with it) is filled from this dataset's OWN unique
diagnosis values — this isn't an AgML dataset (no `datasets.yaml`
--config, no per-row agml_dataset to key a class list off), so
run_judge_phase's new `extra_keys` override is used directly instead of
its usual args.config-derived AgML path.

Usage:
    uv run python main.py heldout-zeroshot \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from PIL import Image as PILImage

from src.common.chat_prompt import render_chat_template
from src.common.io import load_prompts
from src.common.metrics import extract_answer_tag, extract_channel_thought
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, run_judge_phase, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.common.progress import ProgressBar

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

# "agml_dataset"/"class_label"/"image_name" are NOT a stylistic choice —
# load_ckpt()'s resume logic hardcodes exactly these 3 column names to
# build its seen-keys set (`(r["agml_dataset"], r["class_label"],
# r["image_name"])`), so a schema missing any of them would KeyError on
# resume. Reused here with the closest-matching real meaning rather than
# dummy placeholders: agml_dataset is a constant tag for this one held-out
# source, class_label is the real ground truth (diagnosis, joined) —
# same "the thing being judged against" role it plays in every AgML
# checkpoint — and image_name is the actual cached filename.
_HELDOUT_TAG = "DigiGreen_crop_disease_images"
HELDOUT_CKPT_FIELDS = [
    "agml_dataset", "class_label", "crop", "image_name", "details", "country", "state", "split",
    "prediction", "raw_response", "extracted_answer", "reasoning",
    "judge_gt_hint", "class_list_text", "judged_correct", "judge_reasoning",
]


def load_heldout_manifest(path: Path) -> dict:
    return json.loads(path.read_text())["rows"]


def build_class_list_text(rows: dict) -> str:
    """Every unique `diagnosis` value across the WHOLE held-out dataset
    (not just --split) — "the full list of possible classes for this
    dataset" the hybrid judge template asks for, same convention
    class_list_extra_keys builds per-AgML-dataset from
    DatasetConfig.clean_classes. Computed from ALL rows regardless of
    split so the class list itself never differs between a "lessons" run
    and a future "eval" run — only the images judged against it differ."""
    classes = sorted({d for r in rows.values() for d in r["diagnosis"]})
    return "\n".join(f"- {c}" for c in classes)


def _cls_question(crop: str) -> str:
    return (f"What disease, pest, or nutrient deficiency (if any) is affecting this {crop} plant? "
            f"If the plant appears healthy, say so.")


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    gen_template = prompts["assessment_classification_tagged"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    all_rows = load_heldout_manifest(Path(args.manifest))
    class_list_text = build_class_list_text(all_rows)
    selected = {k: r for k, r in all_rows.items() if r["split"] == args.split}
    if not selected:
        raise SystemExit(f"ERROR: no rows with split={args.split!r} in {args.manifest}.")
    print(f"  {len(selected)} row(s) in split={args.split!r} ({len(all_rows)} total in manifest)")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=HELDOUT_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} rows already scored in {ckpt_path.name}")

    pending = {k: r for k, r in selected.items() if (_HELDOUT_TAG, "; ".join(r["diagnosis"]), f"{k}.jpg") not in seen}
    if not pending:
        print("  Nothing to do — all rows already checkpointed.")
    else:
        images_root = Path(args.images_root)
        keys = list(pending.keys())
        images = [PILImage.open(images_root / f"{k}.jpg").convert("RGB") for k in keys]
        questions = [gen_template.format(question=_cls_question(pending[k]["crop"])) for k in keys]
        histories = [
            ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            for q in questions
        ]

        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_zeroshot",
        )

        chunk_size = max(args.chunk_size, 1)
        pbar = ProgressBar(total=len(keys), desc="Generating")
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i : i + chunk_size]
            chunk_images = images[i : i + chunk_size]
            chunk_hist = histories[i : i + chunk_size]
            prompts_batch = [
                {"prompt": render_chat_template(tokenizer, h, enable_thinking=args.enable_thinking,
                                                 chat_template=chat_template),
                 "multi_modal_data": {"image": img}}
                for h, img in zip(chunk_hist, chunk_images)
            ]
            params = SamplingParams(
                temperature=0.0, max_tokens=args.gen_max_tokens, repetition_penalty=1.1,
                skip_special_tokens=not args.enable_thinking,
            )
            outputs = llm.generate(prompts_batch, sampling_params=params)

            for k, out in zip(chunk_keys, outputs):
                raw = out.outputs[0].text
                thinking, cleaned = extract_channel_thought(raw) if args.enable_thinking else ("", raw)
                answer, _ = extract_answer_tag(cleaned)
                r = pending[k]
                diagnosis_text = "; ".join(r["diagnosis"])
                row = {
                    "agml_dataset": _HELDOUT_TAG, "class_label": diagnosis_text,
                    "crop": r["crop"], "image_name": f"{k}.jpg",
                    "details": r["details"], "country": r["country"], "state": r["state"],
                    "split": r["split"],
                    "prediction": answer, "raw_response": raw, "extracted_answer": answer,
                    "reasoning": thinking,
                    "judge_gt_hint": diagnosis_text, "class_list_text": class_list_text,
                    "judged_correct": "", "judge_reasoning": "",
                }
                append_ckpt_row(ckpt_path, row, fields=HELDOUT_CKPT_FIELDS)
                rows.append(row)
            pbar.update(len(chunk_keys))
        pbar.finish()
        destroy_llm(llm)
        del tokenizer

    rows = run_judge_phase(rows, ckpt_path, args, prompts, fields=HELDOUT_CKPT_FIELDS,
                            extra_keys={"class_list": "class_list_text"})

    judged = [r for r in rows if r.get("judged_correct") not in (None, "")]
    if judged:
        acc = sum(int(r["judged_correct"]) for r in judged) / len(judged)
        print(f"\n  judge_accuracy={acc:.4f} (n={len(judged)}/{len(rows)} judged)")
        print(f"  Checkpoint: {ckpt_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--manifest", default="dataset/heldout_selected_images.json")
    parser.add_argument("--images_root", default="dataset/images_heldout")
    parser.add_argument("--split", choices=["lessons", "eval"], default="lessons",
                         help="Restrict to this split — \"eval\" must never be used before Step 4's "
                              "ablation.")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--output_dir", default="outputs_final/heldout_zeroshot/")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Same mechanism/default as vision-baseline's own --enable_thinking — "
                              "on by default. Pass --no-enable_thinking to opt out.")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=64,
                         help="Rows per llm.generate() batch (checkpointed after each chunk).")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    # run_judge_phase's required contract (same block judge_checkpoint.py uses) — no --config, since
    # this dataset's {class_list} is supplied directly via extra_keys above, not an AgML datasets.yaml.
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
