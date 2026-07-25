"""
Validate converted dataset files before training.

Checks each stage output for schema correctness, length constraints,
empty fields, and prints N sample rows per file.

Usage:
  uv run python experiments/ag_finetune/data_prep/test_dataset.py
  uv run python experiments/ag_finetune/data_prep/test_dataset.py --n_samples 10
  uv run python experiments/ag_finetune/data_prep/test_dataset.py --file outputs/stage1_agrillm_filtered.jsonl
"""

import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"

# Schema: required keys and their types per stage
SCHEMAS: dict[str, dict] = {
    "sft_reasoning": {
        "required": ["stage", "source", "question", "answer"],
        "min_len":  {"question": 4, "answer": 20},
        "max_len":  {"answer": 1024},
    },
    "sft_advisory": {
        "required": ["stage", "source", "question", "answer"],
        "min_len":  {"question": 4, "answer": 20},
        "max_len":  {"answer": 1024},
    },
    "rag_qa": {
        "required": ["stage", "source", "passage", "chunk_id"],
        "min_len":  {"passage": 100},
    },
    "dpo": {
        "required": ["stage", "source", "chosen_prompt", "chosen_response", "rejected_response"],
        "min_len":  {"chosen_response": 20, "rejected_response": 20},
    },
    "verifier": {
        "required": ["stage", "source", "messages", "correct", "wrong_pred"],
        "min_len":  {"correct": 2, "wrong_pred": 2},
    },
}


def token_count(text: str) -> int:
    return len(text.split())


def validate_row(row: dict, idx: int) -> list[str]:
    errors = []
    stage = row.get("stage", "unknown")
    schema = SCHEMAS.get(stage)
    if schema is None:
        errors.append(f"[{idx}] Unknown stage: {stage!r}")
        return errors

    for key in schema.get("required", []):
        if key not in row or not row[key]:
            errors.append(f"[{idx}] Missing/empty required field: {key!r}")

    for field, min_tok in schema.get("min_len", {}).items():
        val = row.get(field, "")
        if isinstance(val, str) and token_count(val) < min_tok:
            errors.append(f"[{idx}] {field!r} too short ({token_count(val)} < {min_tok} tokens)")

    for field, max_tok in schema.get("max_len", {}).items():
        val = row.get(field, "")
        if isinstance(val, str) and token_count(val) > max_tok:
            errors.append(f"[{idx}] {field!r} too long ({token_count(val)} > {max_tok} tokens)")

    return errors


def check_file(path: Path, n_samples: int) -> None:
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        return

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"  EMPTY: {path.name}")
        return

    all_errors = []
    for i, row in enumerate(rows):
        all_errors.extend(validate_row(row, i))

    stage = rows[0].get("stage", "?")
    n_err = len(all_errors)
    print(f"\n{'='*60}")
    print(f"  File:    {path.name}")
    print(f"  Stage:   {stage}")
    print(f"  Rows:    {len(rows):,}")
    print(f"  Errors:  {n_err} ({'✓ OK' if n_err == 0 else '✗ ISSUES FOUND'})")

    if all_errors:
        for e in all_errors[:10]:
            print(f"    {e}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors) - 10} more")

    print(f"\n  --- {n_samples} sample rows ---")
    step = max(1, len(rows) // n_samples)
    for i in range(0, min(len(rows), n_samples * step), step):
        row = rows[i]
        print(f"\n  [Row {i}]")
        for k, v in row.items():
            if k in ("raw", "messages"):
                continue
            display = str(v)
            if len(display) > 120:
                display = display[:117] + "..."
            print(f"    {k}: {display}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      default=None, help="Check a single file")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    if args.file:
        check_file(Path(args.file), args.n_samples)
        return

    # Check all stage outputs
    files = sorted(OUT_DIR.glob("stage*.jsonl"))
    if not files:
        print(f"No stage*.jsonl files found in {OUT_DIR}")
        print("Run the data_prep scripts first.")
        return

    for path in files:
        check_file(path, args.n_samples)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
