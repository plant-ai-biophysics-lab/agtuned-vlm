"""
Stage 1a — Filter and deduplicate AI71ai/agrillm-train-146k.

Outputs:
  outputs/stage1_agrillm_filtered.jsonl   — cleaned rows
  outputs/stage1_agrillm_filtered_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/01_filter_agrillm.py
"""

import json
import re
import hashlib
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ANSWER_TOKENS = 20
MAX_ANSWER_TOKENS = 1024

# Keywords that strongly suggest non-agricultural content
NON_AG_SIGNALS = [
    "garment", "textile", "manufacturing plant", "fashion", "apparel",
    "cryptocurrency", "blockchain", "stock market", "investment portfolio",
    "software engineering", "web development", "machine learning model",
]

AG_SIGNALS = [
    "crop", "soil", "fertilizer", "pest", "disease", "harvest", "irrigation",
    "farm", "livestock", "seed", "plant", "agricultural", "agronomy",
    "yield", "nutrient", "weed", "insect", "fungal", "bacterial", "drought",
    "rainfall", "cultivar", "variety", "farmer", "field", "pasture",
]


def rough_token_count(text: str) -> int:
    return len(text.split())


def is_agricultural(question: str, answer: str) -> bool:
    combined = (question + " " + answer).lower()
    for sig in NON_AG_SIGNALS:
        if sig in combined:
            return False
    return any(sig in combined for sig in AG_SIGNALS)


def extract_qa(row: dict) -> tuple[str, str, str] | None:
    """Return (system_prompt, question, answer) or None if malformed."""
    system = row.get("system_prompt", "")
    turns = row.get("turns", [])
    if not turns:
        return None
    first = turns[0] if isinstance(turns, list) else turns
    if isinstance(first, dict):
        q = first.get("user", "") or ""
        a = first.get("assistant", "") or ""
        # assistant can be a list of content parts in some rows
        if isinstance(a, list):
            a = " ".join(str(x.get("text", x) if isinstance(x, dict) else x) for x in a)
        if isinstance(q, list):
            q = " ".join(str(x.get("text", x) if isinstance(x, dict) else x) for x in q)
    else:
        return None
    if not str(q).strip() or not str(a).strip():
        return None
    # Strip prompt scaffolding from question
    q, a = str(q), str(a)
    q_clean = re.sub(r"###\s*QUESTION:\s*", "", q)
    q_clean = re.sub(r"(Develop|Craft|Generate|Formulate|Compose|Create)[^#\n]*\n+", "", q_clean)
    q_clean = re.sub(r"###\s*Answer:\s*$", "", q_clean, flags=re.MULTILINE).strip()
    # Strip markdown answer header
    a_clean = re.sub(r"^###\s*Answer:?\s*", "", a, flags=re.MULTILINE).strip()
    return system, q_clean, a_clean


def fingerprint(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _iter_raw_rows():
    """Download raw JSONL shards and parse manually to avoid PyArrow schema conflicts."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    shard_paths = fs.glob("datasets/AI71ai/agrillm-train-146k/*.jsonl")
    if not shard_paths:
        shard_paths = fs.glob("datasets/AI71ai/agrillm-train-146k/*/*.jsonl")
    print(f"  Found {len(shard_paths)} shard(s)")
    for path in sorted(shard_paths):
        with fs.open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    print("Loading agrillm-train-146k (raw JSONL to handle schema conflicts)...")

    seen: set[str] = set()
    kept, skipped_malformed, skipped_length, skipped_domain, skipped_dup = 0, 0, 0, 0, 0

    out_path = OUT_DIR / "stage1_agrillm_filtered.jsonl"
    with open(out_path, "w") as f:
        for row in _iter_raw_rows():
            parsed = extract_qa(row)
            if parsed is None:
                skipped_malformed += 1
                continue
            system, question, answer = parsed

            n_tok = rough_token_count(answer)
            if n_tok < MIN_ANSWER_TOKENS or n_tok > MAX_ANSWER_TOKENS:
                skipped_length += 1
                continue

            if not is_agricultural(question, answer):
                skipped_domain += 1
                continue

            fp = fingerprint(question)
            if fp in seen:
                skipped_dup += 1
                continue
            seen.add(fp)

            out = {
                "stage": "sft_reasoning",
                "source": "agrillm-train-146k",
                "system": system,
                "question": question,
                "answer": answer,
                "raw": {"system_prompt": row.get("system_prompt", ""), "turns": row.get("turns", [])},
            }
            f.write(json.dumps(out) + "\n")
            kept += 1

    stats = {
        "kept": kept,
        "skipped_malformed": skipped_malformed,
        "skipped_length": skipped_length,
        "skipped_domain": skipped_domain,
        "skipped_duplicate": skipped_dup,
        "total_seen": kept + skipped_malformed + skipped_length + skipped_domain + skipped_dup,
    }
    (OUT_DIR / "stage1_agrillm_filtered_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. Kept {kept:,} rows → {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
