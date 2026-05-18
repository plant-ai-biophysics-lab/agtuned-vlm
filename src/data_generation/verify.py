"""
Stage 1d: Verify and rewrite descriptions using reference sources.

Takes the flat CSV produced by export_csv.py and the reference_db.json
produced by build_reference_db.py. For each class label that has at least
one reference source, uses a vLLM model to rewrite the compressed description
so it is grounded in the reference excerpt while remaining purely visual.

Entries with no matching source are passed through unchanged (status: no_source).

Output: descriptions_corrected.csv — consumed by app.py for review.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv
import json
import re
from collections import defaultdict

from vllm import LLM, SamplingParams

from src.common.io import load_prompts

REWRITE_PARAMS = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.05)
EXCERPT_LIMIT  = 600

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


def _make_key(agml_dataset: str, class_label: str, crop: str) -> str:
    def _slug(s):
        return s.strip().lower().replace(" ", "_")
    return f"{_slug(agml_dataset)}__{_slug(class_label)}__{_slug(crop)}"


def _build_rewrite_prompt(prompts: dict, reference_text: str, compressed_description: str) -> str:
    return prompts["verify_rewrite"].format(
        reference_text=reference_text,
        compressed_description=compressed_description,
    )


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.descriptions_csv)
    db_path  = Path(args.reference_db)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))

    db_entries = json.loads(db_path.read_text(encoding="utf-8")).get("entries", {})

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    llm       = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()

    to_rewrite: list[int] = []
    prompts_list: list[str] = []

    for i, row in enumerate(rows):
        key   = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        entry = db_entries.get(key)
        if not entry or entry.get("status") != "found":
            continue
        excerpt = " ".join(
            s.get("excerpt", "") for s in entry["sources"] if s.get("excerpt")
        )[:EXCERPT_LIMIT]
        prompt = _build_rewrite_prompt(prompts, excerpt, row["compressed_description"])
        prompts_list.append(_apply_chat_template(tokenizer, prompt))
        to_rewrite.append(i)

    rewritten: dict[int, str] = {}
    if prompts_list:
        outputs = llm.generate(prompts_list, sampling_params=REWRITE_PARAMS)
        for idx, out in zip(to_rewrite, outputs):
            rewritten[idx] = _clean_response(out.outputs[0].text)

    FIELDNAMES = [
        "agml_dataset", "crop", "class_label", "task_type",
        "compressed_description", "corrected_description",
        "status", "source_orgs", "source_urls", "excerpt_used",
    ]

    out_rows = []
    for i, row in enumerate(rows):
        key   = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        entry = db_entries.get(key)

        if i in rewritten:
            sources = entry["sources"]
            excerpt = " ".join(s.get("excerpt", "") for s in sources if s.get("excerpt"))[:EXCERPT_LIMIT]
            out_rows.append({
                "agml_dataset":           row["agml_dataset"],
                "crop":                   row["crop"],
                "class_label":            row["class_label"],
                "task_type":              row["task_type"],
                "compressed_description": row["compressed_description"],
                "corrected_description":  rewritten[i],
                "status":                 "corrected",
                "source_orgs":            ",".join(s["organization"] for s in sources),
                "source_urls":            ",".join(s["url"] for s in sources),
                "excerpt_used":           excerpt,
            })
        else:
            out_rows.append({
                "agml_dataset":           row["agml_dataset"],
                "crop":                   row["crop"],
                "class_label":            row["class_label"],
                "task_type":              row["task_type"],
                "compressed_description": row["compressed_description"],
                "corrected_description":  row["compressed_description"],
                "status":                 "no_source" if (not entry or entry.get("status") != "found") else "error",
                "source_orgs":            "",
                "source_urls":            "",
                "excerpt_used":           "",
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    total     = len(out_rows)
    corrected = sum(1 for r in out_rows if r["status"] == "corrected")
    no_source = sum(1 for r in out_rows if r["status"] == "no_source")

    print(f"\n=== Verification Summary ===")
    print(f"  Total     : {total}")
    print(f"  Corrected : {corrected} ({corrected / total * 100:.1f}%)")
    print(f"  No source : {no_source} ({no_source / total * 100:.1f}%)")

    by_task: dict = defaultdict(lambda: {"corrected": 0, "total": 0})
    for r in out_rows:
        tt = r.get("task_type", "unknown")
        by_task[tt]["total"] += 1
        if r["status"] == "corrected":
            by_task[tt]["corrected"] += 1
    print(f"\n  By task type:")
    for tt, counts in sorted(by_task.items()):
        pct = counts["corrected"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {tt:<35} {counts['corrected']:>4}/{counts['total']:<4} ({pct:.1f}%)")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify and rewrite descriptions using reference sources."
    )
    parser.add_argument("--model",             default="Qwen/Qwen3.6-27B",
                        help="vLLM model for rewriting (default: Qwen/Qwen3.6-27B)")
    parser.add_argument("--descriptions_csv",  default="outputs/data_generation/descriptions_export.csv")
    parser.add_argument("--reference_db",      default="outputs/data_generation/reference_db.json")
    parser.add_argument("--output_csv",        default="outputs/data_generation/descriptions_corrected.csv")
    parser.add_argument("--prompts",           default="src/config/prompts.yaml",
                        help="Path to prompts.yaml")
    main(parser.parse_args())
