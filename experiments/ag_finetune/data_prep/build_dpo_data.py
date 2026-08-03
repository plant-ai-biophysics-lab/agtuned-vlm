"""
Consolidate DPO data — reshapes stage5's (question, correct, wrong_pred)
triples into plain {prompt, chosen, rejected} pairs. No verifier JSON schema,
no confidence field, no correction/messages wrapper — those were for the
verifier-specific training path (see stage7_verifier_dpo_pairs.jsonl instead,
left untouched, if that's revisited later). Pure reshape of data that already
exists on disk — no model calls, no GPU.

Input:
  outputs/stage5_verifier_data.jsonl   — "question"/"correct"/"wrong_pred"
                                          fields already present per row (the
                                          verifier-correction wrapper around
                                          them is what's being dropped here)

Cleaning (no pair is dropped for this — only the `prompt` text is rewritten):
  Same leftover scaffolding as build_sft_data.py's source data (stage5 was
  itself built from stage1/stage4's questions) — "### TASK", "### CONTEXT",
  boilerplate lead-in sentences, etc. Stripped from `prompt` only, never from
  `chosen`/`rejected` (their own "### Heading" lines are real answer
  structure). Topic filtering intentionally NOT done here — see
  build_sft_data.py's docstring for why.

Output:
  outputs/dpo_pairs.jsonl   — [{"prompt", "chosen", "rejected"}, ...]
  outputs/dpo_pairs_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/build_dpo_data.py
"""

import json
import re
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"

# Duplicated from build_sft_data.py rather than imported, per this repo's
# convention of keeping experiment scripts self-contained.
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    path = OUT_DIR / "stage5_verifier_data.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 05_build_verifier_data.py first.")
        return

    rows = []
    skipped_missing, skipped_identical = 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("question", "").strip()
            chosen = row.get("correct", "").strip()
            rejected = row.get("wrong_pred", "").strip()
            if not prompt or not chosen or not rejected:
                skipped_missing += 1
                continue
            prompt = _clean_question(prompt)
            if not prompt:
                skipped_missing += 1
                continue
            if chosen.lower() == rejected.lower():
                skipped_identical += 1
                continue
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    out_path = OUT_DIR / "dpo_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats = {
        "total": len(rows),
        "skipped_missing_field": skipped_missing,
        "skipped_identical_chosen_rejected": skipped_identical,
    }
    (OUT_DIR / "dpo_pairs_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Saved {len(rows)} pairs -> {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
