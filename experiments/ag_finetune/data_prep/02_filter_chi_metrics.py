"""
Stage 1c — Filter CGIAR/Chi-Metrics-2024 into high-quality SFT examples.

Note: The dataset scores range 1–3 only (not 1–5 as initially assumed).
Score=1 rows are almost entirely "Not Clear" prompts, leaving no clean
same-topic pairs for DPO. Instead we save score=3 rows as advisory SFT
data and note this in the plan update.

Outputs:
  outputs/stage2_chi_sft.jsonl         — score=3 advisory Q&A rows
  outputs/stage2_chi_dpo_pairs.jsonl   — empty placeholder (see note above)
  outputs/stage2_chi_dpo_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/02_filter_chi_metrics.py
"""

import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_RESPONSE_LEN = 30


def main() -> None:
    print("Loading Chi-Metrics-2024...")
    ds = load_dataset("CGIAR/Chi-Metrics-2024", split="train", streaming=True)

    sft_rows = []
    skipped_unclear, skipped_short, skipped_low_score, total = 0, 0, 0, 0

    for row in ds:
        total += 1
        score = int(row.get("overall_score") or 0)
        if row.get("prompt_clarity") == "Not Clear":
            skipped_unclear += 1
            continue
        msg  = row.get("translated_message") or row.get("original_message") or ""
        resp = row.get("message_response") or ""
        if len(msg.strip()) < 3 or len(resp.strip()) < MIN_RESPONSE_LEN:
            skipped_short += 1
            continue
        if score < 3:
            skipped_low_score += 1
            continue
        # Drop RAG-artifact answers that admit no knowledge
        if "does not provide" in resp.lower() or "not mentioned in the" in resp.lower():
            skipped_low_score += 1
            continue
        sft_rows.append({
            "stage":    "sft_advisory",
            "source":   "chi-metrics-2024",
            "question": msg.strip(),
            "answer":   resp.strip(),
            "topic":    row.get("topic") or "",
            "entity":   row.get("entity") or "",
            "score":    score,
        })

    sft_path = OUT_DIR / "stage2_chi_sft.jsonl"
    with open(sft_path, "w") as f:
        for r in sft_rows:
            f.write(json.dumps(r) + "\n")

    # Empty DPO placeholder (not feasible with this dataset's score distribution)
    (OUT_DIR / "stage2_chi_dpo_pairs.jsonl").write_text("")

    stats = {
        "total_rows": total,
        "skipped_unclear_prompt": skipped_unclear,
        "skipped_short": skipped_short,
        "skipped_low_score": skipped_low_score,
        "sft_rows_kept": len(sft_rows),
        "note": "Score range is 1-3 only; DPO not feasible — use SFT on score=3 rows instead",
    }
    (OUT_DIR / "stage2_chi_dpo_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. {len(sft_rows):,} SFT rows → {sft_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
