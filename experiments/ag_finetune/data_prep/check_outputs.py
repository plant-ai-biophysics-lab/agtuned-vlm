"""
One-off QC script — prints row counts and known-risk checks for every
data_prep output file, so results can be pasted back for review without
needing direct filesystem access to wherever this actually ran.

Usage:
  uv run python experiments/ag_finetune/data_prep/check_outputs.py
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"


def count_lines(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def think_leak_count(path: Path, field: str) -> int:
    if not path.exists():
        return -1
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "<think" in str(row.get(field, "")).lower():
            n += 1
    return n


def print_stats_json(path: Path) -> None:
    if path.exists():
        print(f"  stats: {json.dumps(json.loads(path.read_text()), indent=2)}")
    else:
        print(f"  [no stats file at {path.name}]")


print("=" * 70)
print("Stage 1 — stage1_agrillm_filtered.jsonl")
print("=" * 70)
p = OUT_DIR / "stage1_agrillm_filtered.jsonl"
print(f"  rows: {count_lines(p):,}")
print_stats_json(OUT_DIR / "stage1_agrillm_filtered_stats.json")

print("\n" + "=" * 70)
print("Stage 2 — stage2_chi_sft.jsonl / stage2_chi_dpo_pairs.jsonl")
print("=" * 70)
print(f"  SFT rows: {count_lines(OUT_DIR / 'stage2_chi_sft.jsonl'):,}")
print(f"  DPO rows: {count_lines(OUT_DIR / 'stage2_chi_dpo_pairs.jsonl'):,} (expected 0 — intentional)")
print_stats_json(OUT_DIR / "stage2_chi_dpo_stats.json")

print("\n" + "=" * 70)
print("Stage 3 — stage3_gardian_passages.jsonl")
print("=" * 70)
p = OUT_DIR / "stage3_gardian_passages.jsonl"
print(f"  rows: {count_lines(p):,}")
print_stats_json(OUT_DIR / "stage3_gardian_stats.json")

print("\n" + "=" * 70)
print("Stage 4 — stage4_reasoning_chains.jsonl")
print("=" * 70)
p = OUT_DIR / "stage4_reasoning_chains.jsonl"
n = count_lines(p)
leaks = think_leak_count(p, "chain")
print(f"  rows: {n:,}")
print(f"  <think> leaks in 'chain' field: {leaks} (expect 0 with patched script)")
print_stats_json(OUT_DIR / "stage4_reasoning_stats.json")

print("\n" + "=" * 70)
print("Stage 5 — stage5_verifier_data.jsonl")
print("=" * 70)
p = OUT_DIR / "stage5_verifier_data.jsonl"
n = count_lines(p)
leaks = think_leak_count(p, "correction")
print(f"  rows: {n:,} (target was up to --n_hardneg, e.g. 5000)")
print(f"  <think> leaks in 'correction' field: {leaks} (expect 0 with patched script)")
print_stats_json(OUT_DIR / "stage5_verifier_stats.json")

print("\n" + "=" * 70)
print("Stage 6 — stage6_gardian_qa.jsonl")
print("=" * 70)
p = OUT_DIR / "stage6_gardian_qa.jsonl"
n = count_lines(p)
leaks = think_leak_count(p, "answer")
print(f"  rows: {n:,} (target was up to --n_samples x --n_qa, e.g. 5000x2=10000)")
print(f"  <think> leaks in 'answer' field: {leaks} (expect 0 with patched script)")
print_stats_json(OUT_DIR / "stage6_gardian_qa_stats.json")

print("\n" + "=" * 70)
print("Stage 7 — stage7_verifier_dpo_pairs.jsonl")
print("=" * 70)
p = OUT_DIR / "stage7_verifier_dpo_pairs.jsonl"
n = count_lines(p)
print(f"  rows: {n:,}")
if p.exists():
    from collections import Counter
    modes = Counter()
    for line in p.read_text().splitlines():
        if line.strip():
            modes[json.loads(line).get("failure_mode", "?")] += 1
    print(f"  by failure_mode: {dict(modes)}")
print_stats_json(OUT_DIR / "stage7_verifier_dpo_stats.json")

print("\nDone.")
