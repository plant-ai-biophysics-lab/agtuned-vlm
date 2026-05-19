"""
Stage 1c: Verify and rewrite descriptions using reference sources.

Two-pass pipeline:
  Pass 1 — Rewrite: for each class label with a reference source, rewrites the
            compressed description using the reference excerpt as grounding.
  Pass 2 — Sanitize: scans every description (rewritten or original) for words
            that directly reveal the class label or crop, and rewrites any that
            contain them so the visual content is preserved but the answer is not.

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

REWRITE_PARAMS  = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.05)
SANITIZE_PARAMS = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.05)
EXCERPT_PER_SOURCE = 1000
MAX_SOURCES        = 3

# Short words that are too generic to flag (articles, prepositions, etc.)
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "as", "by", "is", "are", "was", "were", "be", "it",
    "its", "this", "that", "from", "no", "not",
}

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


def _label_words(class_label: str, crop: str) -> set[str]:
    """Return lowercase significant words from the class label and crop."""
    raw = (class_label + " " + (crop or "")).lower()
    tokens = re.findall(r"[a-z]+", raw)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _contains_label(description: str, label_words: set[str]) -> bool:
    desc_words = set(re.findall(r"[a-z]+", description.lower()))
    return bool(desc_words & label_words)


def _build_excerpt(sources: list[dict]) -> str:
    parts = []
    for s in sources[:MAX_SOURCES]:
        text = s.get("excerpt", "").strip()
        if not text:
            continue
        org = s.get("organization", "Source")
        parts.append(f"[{org}]\n{text[:EXCERPT_PER_SOURCE]}")
    return "\n\n---\n\n".join(parts)


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.descriptions_csv)
    db_path  = Path(args.reference_db)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts    = load_prompts(Path(args.prompts))
    db_entries = json.loads(db_path.read_text(encoding="utf-8")).get("entries", {})

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    llm       = LLM(model=args.model, dtype="bfloat16", max_model_len=4096,
                    gpu_memory_utilization=0.9, max_num_seqs=128)
    tokenizer = llm.get_tokenizer()

    # ── Pass 1: rewrite descriptions that have reference sources ──────────────

    to_rewrite: list[int] = []
    rewrite_prompts: list[str] = []

    for i, row in enumerate(rows):
        key   = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        entry = db_entries.get(key)
        if not entry or entry.get("status") != "found":
            continue
        excerpt = _build_excerpt(entry["sources"])
        prompt = prompts["verify_rewrite"].format(
            reference_text=excerpt,
            compressed_description=row["compressed_description"],
        )
        rewrite_prompts.append(_apply_chat_template(tokenizer, prompt))
        to_rewrite.append(i)

    rewritten: dict[int, str] = {}
    if rewrite_prompts:
        print(f"Pass 1 — Rewriting {len(rewrite_prompts)} descriptions using reference sources...")
        outputs = llm.generate(rewrite_prompts, sampling_params=REWRITE_PARAMS)
        for idx, out in zip(to_rewrite, outputs):
            rewritten[idx] = _clean_response(out.outputs[0].text)

    # Build out_rows with pass-1 results before sanitization
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
            excerpt = _build_excerpt(sources)
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

    # ── Pass 2: sanitize any description that still contains the label ────────

    to_sanitize: list[int] = []
    sanitize_prompts: list[str] = []

    for i, row in enumerate(out_rows):
        words = _label_words(row["class_label"], row["crop"])
        if words and _contains_label(row["corrected_description"], words):
            prompt = prompts["verify_sanitize"].format(
                class_label=row["class_label"],
                crop=row["crop"] or "plant",
                description=row["corrected_description"],
            )
            sanitize_prompts.append(_apply_chat_template(tokenizer, prompt))
            to_sanitize.append(i)

    sanitized: dict[int, str] = {}
    if sanitize_prompts:
        print(f"Pass 2 — Sanitizing {len(sanitize_prompts)} descriptions that contain label words...")
        outputs = llm.generate(sanitize_prompts, sampling_params=SANITIZE_PARAMS)
        for idx, out in zip(to_sanitize, outputs):
            sanitized[idx] = _clean_response(out.outputs[0].text)

    for idx, text in sanitized.items():
        prev_status = out_rows[idx]["status"]
        out_rows[idx]["corrected_description"] = text
        out_rows[idx]["status"] = "sanitized" if prev_status == "no_source" else f"{prev_status}+sanitized"

    # ── Save ──────────────────────────────────────────────────────────────────

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    # ── Summary ───────────────────────────────────────────────────────────────

    total      = len(out_rows)
    corrected  = sum(1 for r in out_rows if "corrected" in r["status"])
    san_only   = sum(1 for r in out_rows if r["status"] == "sanitized")
    no_source  = sum(1 for r in out_rows if r["status"] == "no_source")

    print(f"\n=== Verification Summary ===")
    print(f"  Total              : {total}")
    print(f"  Rewritten (pass 1) : {len(rewritten)} ({len(rewritten) / total * 100:.1f}%)")
    print(f"  Sanitized (pass 2) : {len(sanitized)} ({len(sanitized) / total * 100:.1f}%)")
    print(f"  No source          : {no_source} ({no_source / total * 100:.1f}%)")

    by_task: dict = defaultdict(lambda: {"corrected": 0, "sanitized": 0, "total": 0})
    for r in out_rows:
        tt = r.get("task_type", "unknown")
        by_task[tt]["total"] += 1
        if "corrected" in r["status"]:
            by_task[tt]["corrected"] += 1
        if "sanitized" in r["status"]:
            by_task[tt]["sanitized"] += 1
    print(f"\n  By task type:")
    for tt, counts in sorted(by_task.items()):
        pct_c = counts["corrected"] / counts["total"] * 100 if counts["total"] else 0
        pct_s = counts["sanitized"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {tt:<35} rewritten={counts['corrected']:>4}/{counts['total']:<4} ({pct_c:.1f}%)  "
              f"sanitized={counts['sanitized']:>4} ({pct_s:.1f}%)")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify, rewrite, and sanitize descriptions using reference sources."
    )
    parser.add_argument("--model",             default="Qwen/Qwen3.6-35B-A3B",
                        help="vLLM model (default: Qwen/Qwen3.6-35B-A3B)")
    parser.add_argument("--descriptions_csv",  default="outputs/data_generation/descriptions_export.csv")
    parser.add_argument("--reference_db",      default="outputs/data_generation/reference_db.json")
    parser.add_argument("--output_csv",        default="outputs/data_generation/descriptions_corrected.csv")
    parser.add_argument("--prompts",           default="src/config/prompts.yaml",
                        help="Path to prompts.yaml")
    main(parser.parse_args())
