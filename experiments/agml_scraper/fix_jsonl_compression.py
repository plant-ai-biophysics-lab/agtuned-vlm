"""
fix_jsonl_compression.py — one-off migration script
=====================================================
Moves metadata.compressed_description into messages[1].content for
agml_descriptions.jsonl, and saves the old answer as
metadata.synthesized_description.

No model required — pure JSON reshuffling.

Writes fixed files alongside originals:
  agml_descriptions.fixed.jsonl
  agml_classification.fixed.jsonl  (unchanged, already has both fields)

Usage:
    python fix_jsonl_compression.py \
      --descriptions_jsonl outputs/agml_descriptions.jsonl \
      --classification_jsonl outputs/agml_classification.jsonl
"""

import argparse
import json
from pathlib import Path


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

    desc_records = load_jsonl(desc_path)
    cls_records  = load_jsonl(cls_path)

    fixed = 0
    skipped = 0
    for r in desc_records:
        meta       = r.get("metadata", {})
        compressed = meta.get("compressed_description")

        if not compressed:
            print(f"  [warn] no compressed_description in metadata for "
                  f"{meta.get('agml_dataset')} / {meta.get('class_label')} — skipping")
            skipped += 1
            continue

        if meta.get("synthesized_description"):
            # already fixed
            skipped += 1
            continue

        synthesized                      = r["messages"][1]["content"]
        r["messages"][1]["content"]      = compressed
        meta["synthesized_description"]  = synthesized
        fixed += 1

    print(f"\nDescriptions: {fixed} fixed, {skipped} skipped")

    fixed_desc_path = desc_path.with_suffix(".fixed.jsonl")
    fixed_cls_path  = cls_path.with_suffix(".fixed.jsonl")

    write_jsonl(fixed_desc_path, desc_records)
    # classification JSONL already has both fields — copy as-is
    write_jsonl(fixed_cls_path, cls_records)

    print(f"\nReview then replace originals with:")
    print(f"  mv {fixed_desc_path} {desc_path}")
    print(f"  mv {fixed_cls_path} {cls_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions_jsonl", default="outputs/agml_descriptions.jsonl",
    )
    parser.add_argument(
        "--classification_jsonl", default="outputs/agml_classification.jsonl",
    )
    args = parser.parse_args()
    main(args)
