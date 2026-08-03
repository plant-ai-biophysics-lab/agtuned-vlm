"""
Consolidate SFT data — merges the flat, open-ended Q&A already produced by the
old 7-stage pipeline into one plain {question, answer} dataset. No CoT/<think>
wrapper (that was stage4's job, now unused), no verifier JSON schema, no
confidence scoring. Pure reshape/merge of data that already exists on disk —
no model calls, no GPU.

Inputs (all optional — skips silently if a file doesn't exist):
  outputs/stage1_agrillm_filtered.jsonl  — general ag Q&A (already flat)
  outputs/stage2_chi_sft.jsonl           — CGIAR advisory Q&A (score=3 filtered)
  outputs/stage6_gardian_qa.jsonl        — Q&A generated from real ag literature
                                            (GARDIAN/CIRAD/EMBRAPA/USDA-NAL) —
                                            this is the actual knowledge-grounding
                                            piece, the rest is general Q&A.

Cleaning (no row is dropped for this — only the `question` text is rewritten):
  A chunk of source questions still carry leftover prompt-scaffolding from the
  original synthetic-generation templates — leading section-header lines like
  "### TASK", "### INSTRUCTIONS", "### CONTEXT", "### Reference", trailing
  "### RESPONSE" markers, and boilerplate lead-in sentences ("Please create a
  thoughtful, accurate, and appropriate response to the following question.").
  These are stripped so `question` reads as a normal open-ended question, not
  an instruction-tuning template. Deliberately NOT applied to `answer` — its
  own "### Heading" lines are real content structure (e.g. "### Core Crop
  List"), not scaffolding, and stripping those would damage genuine answers.
  Topic filtering (e.g. dropping rows unrelated to disease/pest/weed) is
  intentionally NOT done here — see conversation: ~35-40% keyword-relevant is
  considered large enough to keep everything for now.

Output:
  outputs/sft_data.jsonl   — [{"question", "answer", "source"}, ...]
  outputs/sft_data_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/build_sft_data.py
"""

import json
import re
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"

SOURCES = [
    ("stage1_agrillm_filtered.jsonl", "agrillm"),
    ("stage2_chi_sft.jsonl", "chi_metrics"),
    ("stage6_gardian_qa.jsonl", "gardian_qa"),
]

# Known scaffolding section-header labels found in real `question` text —
# a small fixed vocabulary, not a generic "any markdown header" strip, since
# a generic strip would also eat genuinely-structured questions.
_SCAFFOLD_HEADER_RE = re.compile(
    r"^#{1,4}\s*(TASK INSTRUCTIONS|TASK|INSTRUCTIONS|GUIDELINES|CONTEXT|"
    r"REFERENCE|QUESTION|RESPONSE|ANSWER)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BOILERPLATE_PREFIXES = (
    "Please create a thoughtful, accurate, and appropriate response to the following question.",
    "Formulate a precise, insightful, and relevant response to the question presented.",
)


def _clean_question(text: str) -> str:
    for prefix in _BOILERPLATE_PREFIXES:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):]
    text = _SCAFFOLD_HEADER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines left by removed headers
    return text.strip()


def main() -> None:
    rows = []
    seen_questions = set()
    by_source = {}
    duplicates = 0

    for filename, tag in SOURCES:
        path = OUT_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename} not found")
            continue
        kept = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                q, a = row.get("question", "").strip(), row.get("answer", "").strip()
                if not q or not a:
                    continue
                q = _clean_question(q)
                if not q:
                    continue
                key = q.lower()
                if key in seen_questions:
                    duplicates += 1
                    continue
                seen_questions.add(key)
                rows.append({"question": q, "answer": a, "source": tag})
                kept += 1
        by_source[tag] = kept
        print(f"  {filename}: kept {kept}")

    out_path = OUT_DIR / "sft_data.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats = {"total": len(rows), "by_source": by_source, "duplicates_dropped": duplicates}
    (OUT_DIR / "sft_data_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nSaved {len(rows)} rows -> {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
