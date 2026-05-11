"""
Preprocess AgriCoT reasoning into clean visual descriptions using Qwen3.6-35B-A3B.

For each MCQ record in VQRA.json:
  - Rewrites the reasoning as a visual-only description (no answer, no leakage)
  - Rewrites the question to remove "this image" phrasing (textual branch)
  - Preserves the original question for the visual branch

Output JSON schema per record:
  item_id, dimension_id, task, options, answer, image_path,
  question_visual    ← original question (used with image)
  question_textual   ← reworded, no "this image" (used with description only)
  visual_description ← LLM-generated visual-only description, no answer leakage
  reasoning_original ← raw reasoning for reference

Usage:
    python preprocess_reasoning.py --output agricot_preprocessed.json
    python preprocess_reasoning.py --dry_run --n 5
    python preprocess_reasoning.py --output agricot_preprocessed.json --resume
"""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DIMENSION_NAMES: dict[int, str] = {
    1: "Produce Counting",
    2: "Pest Localization",
    3: "Crop Species ID",
    4: "Disease Assessment",
    5: "Field Management",
}

DESCRIPTION_SYSTEM = (
    "You write one-sentence visual captions for agricultural images.\n\n"
    "Rules — follow all strictly:\n"
    "- Exactly one sentence. No more.\n"
    "- Describe only the high-level visual subject: what type of plant/crop/object "
    "is shown and its most prominent visible feature (color, condition, part).\n"
    "- Do NOT mention the correct answer or any word that names it.\n"
    "- Do NOT mention any option letter (A, B, C, D).\n"
    "- Do NOT include any analysis, conclusion, or reasoning.\n"
    "- Style: 'A rice leaf showing brown circular lesions.' / "
    "'A corn cob with scattered discolored kernels.' / "
    "'A wheat stem with orange powdery pustules.'\n"
    "Return only the one-sentence caption, nothing else."
)

DESCRIPTION_USER_TEMPLATE = (
    "Correct answer (do NOT mention this in your caption): {answer} = {answer_text}\n\n"
    "Reasoning (for context only):\n{reasoning}"
)

QUESTION_REWORD_SYSTEM = (
    "Rewrite the question so it does not reference 'this image', 'the image', "
    "'the picture', 'the photo', or any phrase implying the reader can see an image. "
    "Instead, phrase it as if the reader only has a written description.\n"
    "Keep the question otherwise identical — same topic, same meaning.\n"
    "Return only the rewritten question, nothing else."
)


def _load_valid_records(zip_file: str | None) -> list[dict]:
    candidates = (zip_file,) if zip_file else ("AgriCoT_2.zip", "AgriCoT.zip")
    for filename in candidates:
        print(f"Downloading {filename} ...")
        try:
            zip_path = hf_hub_download(
                repo_id="wenyb/AgriCoT", filename=filename, repo_type="dataset"
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read("AgriCoT_2/VQRA.json"))
            image_names = {
                Path(m).name
                for m in zf.namelist()
                if m.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            }

        valid = [
            r for r in data
            if r.get("options")
            and r.get("reasoning")
            and r.get("question")
            and Path(r.get("image_path", "")).name in image_names
        ]
        print(f"  {len(valid)} valid MCQ records")
        return valid

    raise RuntimeError("Could not load AgriCoT.")


def _needs_reword(question: str) -> bool:
    return bool(re.search(
        r"\b(image|picture|photo|shown|depicted)\b", question, re.IGNORECASE
    ))


def _batch_generate(llm, tokenizer, prompts: list[str], sampling_params) -> list[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def _build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AgriCoT reasoning into visual descriptions via Qwen"
    )
    parser.add_argument("--output",      type=str, default="agricot_preprocessed.json")
    parser.add_argument("--zip",         type=str, default=None)
    parser.add_argument("--model",       type=str, default="Qwen/Qwen3.6-35B-A3B",
        help="vLLM-compatible HF model ID")
    parser.add_argument("--gpu_mem",     type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--n",           type=int, default=None,
        help="Limit records processed (for testing)")
    parser.add_argument("--dry_run",     action="store_true",
        help="Print first N raw records without calling the model")
    parser.add_argument("--resume",      action="store_true",
        help="Skip records already in --output")
    args = parser.parse_args()

    records = _load_valid_records(args.zip)
    if args.n:
        records = records[: args.n]

    if args.dry_run:
        print("\n── DRY RUN — first 3 records ──")
        for r in records[:3]:
            opts = "\n".join(f"  {k}: {v}" for k, v in sorted(r["options"].items()))
            print(f"\nitem_id={r.get('item_id')}  dim={r.get('dimension_id')}")
            print(f"  Q: {r['question']}")
            print(f"  Options:\n{opts}")
            print(f"  Reasoning (first 400 chars):\n{r['reasoning'][:400]!r}")
        return

    # Load existing output for resume
    done_ids: set = set()
    results: list[dict] = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        done_ids = {r["item_id"] for r in results}
        print(f"  Resuming: {len(done_ids)} already processed")

    pending = [r for r in records if r.get("item_id") not in done_ids]
    if not pending:
        print("Nothing to do.")
        return

    # Load model
    from vllm import LLM, SamplingParams

    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_num_seqs=args.batch_size,
    )
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.92,
        max_tokens=80,
        stop=["\n\n\n"],
    )

    print(f"\nProcessing {len(pending)} records in batches of {args.batch_size} ...")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]

        # Build description prompts
        desc_prompts = [
            _build_prompt(
                tokenizer,
                DESCRIPTION_SYSTEM,
                DESCRIPTION_USER_TEMPLATE.format(
                    answer=r["answer"],
                    answer_text=r["options"].get(r["answer"], ""),
                    reasoning=r["reasoning"],
                ),
            )
            for r in batch
        ]

        # Build question-reword prompts (only for questions referencing the image)
        reword_prompts = [
            _build_prompt(tokenizer, QUESTION_REWORD_SYSTEM, r["question"])
            if _needs_reword(r["question"])
            else None
            for r in batch
        ]

        # Run description generation
        desc_outputs = _batch_generate(llm, tokenizer, desc_prompts, sampling_params)

        # Run reword generation (only non-None prompts)
        reword_indices = [i for i, p in enumerate(reword_prompts) if p is not None]
        if reword_indices:
            reword_out = _batch_generate(
                llm, tokenizer,
                [reword_prompts[i] for i in reword_indices],
                sampling_params,
            )
            reword_map = dict(zip(reword_indices, reword_out))
        else:
            reword_map = {}

        for i, r in enumerate(batch):
            q_textual = reword_map[i] if i in reword_map else r["question"]
            task = DIMENSION_NAMES.get(r.get("dimension_id"), "unknown")
            results.append({
                "item_id":            r.get("item_id", start + i),
                "dimension_id":       r.get("dimension_id"),
                "task":               task,
                "options":            r["options"],
                "answer":             r["answer"],
                "image_path":         r.get("image_path", ""),
                "question_visual":    r["question"],
                "question_textual":   q_textual,
                "visual_description": desc_outputs[i],
                "reasoning_original": r["reasoning"],
            })

        done = min(start + args.batch_size, len(pending))
        print(f"  [{done}/{len(pending)}] processed")

        # Save incrementally every batch
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} records → {args.output}")

    # Print samples for inspection
    print("\n── Sample outputs (first 3) ──")
    for r in results[:3]:
        print(f"\nitem {r['item_id']}  [{r['task']}]")
        print(f"  Q visual   : {r['question_visual']}")
        print(f"  Q textual  : {r['question_textual']}")
        print(f"  Description: {r['visual_description']}")
        print(f"  Answer     : {r['answer']}")
        print(f"  Original reasoning (first 200): {r['reasoning_original'][:200]!r}")


if __name__ == "__main__":
    main()
