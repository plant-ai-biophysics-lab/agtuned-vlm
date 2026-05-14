"""Rewrite compressed descriptions using reference sources and a local vLLM instance."""

import argparse
import csv
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

REWRITE_PARAMS = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.05)
EXCERPT_LIMIT = 600

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
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


def _build_prompt(reference_text: str, compressed_description: str) -> str:
    return (
        "You are a plant pathology expert writing visual descriptions for a machine learning image dataset.\n\n"
        "Your task: rewrite the description below as a single sentence based on the reference text.\n\n"
        "Rules:\n"
        "- Focus ONLY on observable visual symptoms: colors, textures, shapes, patterns, lesions, spots, discoloration, size, location on plant\n"
        "- Do NOT mention the disease name, pest name, pathogen name, or any direct class identifier\n"
        "- Do NOT mention the crop, plant species, or genus name\n"
        "- Be specific enough to distinguish this condition from other diseases visually\n"
        "- Output only the single sentence, nothing else\n\n"
        f"Reference text:\n{reference_text}\n\n"
        f"Current description:\n{compressed_description}\n\n"
        "Improved visual description:"
    )


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.descriptions_csv)
    db_path = Path(args.reference_db)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = json.loads(db_path.read_text(encoding="utf-8"))
    db_entries = db.get("entries", {})

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()

    to_rewrite_indices = []
    prompts = []

    for i, row in enumerate(rows):
        key = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        entry = db_entries.get(key)
        if not entry or entry.get("status") != "found":
            continue
        combined = " ".join(
            s.get("excerpt", "") for s in entry["sources"] if s.get("excerpt")
        )[:EXCERPT_LIMIT]
        prompts.append(_apply_chat_template(tokenizer, _build_prompt(combined, row["compressed_description"])))
        to_rewrite_indices.append(i)

    rewritten = {}
    if prompts:
        outputs = llm.generate(prompts, sampling_params=REWRITE_PARAMS)
        for idx, out in zip(to_rewrite_indices, outputs):
            rewritten[idx] = _clean_response(out.outputs[0].text)

    out_rows = []
    for i, row in enumerate(rows):
        key = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        entry = db_entries.get(key)

        if i in rewritten:
            sources = entry["sources"]
            combined_excerpt = " ".join(
                s.get("excerpt", "") for s in sources if s.get("excerpt")
            )[:EXCERPT_LIMIT]
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
                "excerpt_used":           combined_excerpt,
            })
        elif not entry or entry.get("status") != "found":
            out_rows.append({
                "agml_dataset":           row["agml_dataset"],
                "crop":                   row["crop"],
                "class_label":            row["class_label"],
                "task_type":              row["task_type"],
                "compressed_description": row["compressed_description"],
                "corrected_description":  row["compressed_description"],
                "status":                 "no_source",
                "source_orgs":            "",
                "source_urls":            "",
                "excerpt_used":           "",
            })
        else:
            out_rows.append({
                "agml_dataset":           row["agml_dataset"],
                "crop":                   row["crop"],
                "class_label":            row["class_label"],
                "task_type":              row["task_type"],
                "compressed_description": row["compressed_description"],
                "corrected_description":  row["compressed_description"],
                "status":                 "error",
                "source_orgs":            "",
                "source_urls":            "",
                "excerpt_used":           "",
            })

    fieldnames = [
        "agml_dataset", "crop", "class_label", "task_type",
        "compressed_description", "corrected_description",
        "status", "source_orgs", "source_urls", "excerpt_used",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    total     = len(out_rows)
    corrected = sum(1 for r in out_rows if r["status"] == "corrected")
    no_source = sum(1 for r in out_rows if r["status"] == "no_source")
    error     = sum(1 for r in out_rows if r["status"] == "error")

    print(f"\n=== Verification Summary ===")
    print(f"  Total entries : {total}")
    print(f"  Corrected     : {corrected} ({corrected/total*100:.1f}%)")
    print(f"  No source     : {no_source} ({no_source/total*100:.1f}%)")
    print(f"  Error         : {error} ({error/total*100:.1f}%)")

    # Per task type
    from collections import defaultdict
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

    # Per dataset
    by_ds: dict = defaultdict(lambda: {"corrected": 0, "total": 0})
    for r in out_rows:
        ds = r.get("agml_dataset", "unknown")
        by_ds[ds]["total"] += 1
        if r["status"] == "corrected":
            by_ds[ds]["corrected"] += 1
    print(f"\n  By dataset:")
    for ds, counts in sorted(by_ds.items()):
        pct = counts["corrected"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {ds:<45} {counts['corrected']:>4}/{counts['total']:<4} ({pct:.1f}%)")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and rewrite descriptions using reference sources.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--descriptions_csv", default="outputs/descriptions_export.csv")
    parser.add_argument("--reference_db", default="outputs/reference_db.json")
    parser.add_argument("--output_csv", default="outputs/descriptions_corrected.csv")
    main(parser.parse_args())
