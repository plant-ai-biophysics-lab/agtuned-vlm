"""
fix_jsonl_compression.py — one-off migration script
=====================================================
Adds compressed_description (Pass 3) to existing agml_descriptions.jsonl
and agml_classification.jsonl that were generated before Pass 3 was added.

For each description record:
  - Runs compression on messages[1].content (the old synthesized answer)
  - Sets messages[1].content = compressed (1 sentence)
  - Adds metadata.synthesized_description = old answer
  - Adds metadata.compressed_description = compressed

For each classification record:
  - Looks up the compressed description by (agml_dataset, class_label)
  - Adds metadata.compressed_description

Writes fixed files alongside originals as:
  agml_descriptions.fixed.jsonl
  agml_classification.fixed.jsonl

Usage:
    uv run python fix_jsonl_compression.py \
      --descriptions_jsonl outputs/agml_descriptions.jsonl \
      --classification_jsonl outputs/agml_classification.jsonl \
      --model Qwen/Qwen3.6-35B-A3B
"""

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

COMPRESSION_PARAMS = SamplingParams(
    temperature=0.0, max_tokens=96, repetition_penalty=1.05
)

_THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r'(\b\w+\b)(\s+\1){4,}', re.IGNORECASE)


def _clean_response(text: str) -> str:
    text = _THINK_RE.sub('', text).strip()
    think_start = text.lower().find('<think>')
    if think_start != -1:
        text = text[:think_start].strip()
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text


def _apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "/no_think"}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_compression_prompt(synthesized: str) -> str:
    return (
        f"Compress the following visual description into exactly one concise sentence. "
        f"Preserve the most distinctive visual features. "
        f"Do not name or repeat any disease, pest, species, or class label.\n\n"
        f"Description: {synthesized}"
    )


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {path.name}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records to {path.name}")


def main(args: argparse.Namespace) -> None:
    desc_path = Path(args.descriptions_jsonl)
    cls_path  = Path(args.classification_jsonl)

    print("Loading records...")
    desc_records = load_jsonl(desc_path)
    cls_records  = load_jsonl(cls_path)

    # Skip records that already have compressed_description
    to_compress = [
        r for r in desc_records
        if "compressed_description" not in r.get("metadata", {})
    ]
    already_done = len(desc_records) - len(to_compress)
    if already_done:
        print(f"  {already_done} records already have compressed_description — skipping")
    print(f"  {len(to_compress)} records to compress")

    if not to_compress:
        print("Nothing to do.")
        return

    print(f"\nLoading model: {args.model}")
    llm       = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts from the current answer (synthesized description)
    print("\nRunning compression pass...")
    prompts = [
        _apply_chat_template(tokenizer, build_compression_prompt(r["messages"][1]["content"]))
        for r in to_compress
    ]
    outputs = llm.generate(prompts, sampling_params=COMPRESSION_PARAMS)
    compressed_texts = [_clean_response(o.outputs[0].text) for o in outputs]

    # Build lookup: (agml_dataset, class_label) → compressed
    compressed_lookup: dict[tuple, str] = {}

    # Apply fixes to description records
    compress_iter = iter(compressed_texts)
    for r in desc_records:
        if "compressed_description" in r.get("metadata", {}):
            key = (r["metadata"]["agml_dataset"], r["metadata"]["class_label"])
            compressed_lookup[key] = r["metadata"]["compressed_description"]
            continue
        synthesized = r["messages"][1]["content"]
        compressed  = next(compress_iter)
        key = (r["metadata"]["agml_dataset"], r["metadata"]["class_label"])
        compressed_lookup[key] = compressed

        r["messages"][1]["content"]               = compressed
        r["metadata"]["synthesized_description"]  = synthesized
        r["metadata"]["compressed_description"]   = compressed
        print(f"  [{r['metadata']['agml_dataset']}] {r['metadata']['class_label']}")
        print(f"    was: {synthesized[:100]}...")
        print(f"    now: {compressed}")

    # Apply fixes to classification records
    for r in cls_records:
        if "compressed_description" in r.get("metadata", {}):
            continue
        key = (r["metadata"]["agml_dataset"], r["metadata"]["class_label"])
        compressed = compressed_lookup.get(key)
        if compressed:
            r["metadata"]["compressed_description"] = compressed
        else:
            print(f"  [warn] no compressed found for {key}")

    # Write fixed files
    fixed_desc_path = desc_path.with_suffix(".fixed.jsonl")
    fixed_cls_path  = cls_path.with_suffix(".fixed.jsonl")
    print(f"\nWriting fixed files...")
    write_jsonl(fixed_desc_path, desc_records)
    write_jsonl(fixed_cls_path,  cls_records)

    print(f"\nDone. Review the .fixed.jsonl files, then replace originals:")
    print(f"  mv {fixed_desc_path} {desc_path}")
    print(f"  mv {fixed_cls_path} {cls_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions_jsonl", default="outputs/agml_descriptions.jsonl",
        help="Path to agml_descriptions.jsonl",
    )
    parser.add_argument(
        "--classification_jsonl", default="outputs/agml_classification.jsonl",
        help="Path to agml_classification.jsonl",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.6-35B-A3B",
        help="Same model used for generation",
    )
    args = parser.parse_args()
    main(args)
