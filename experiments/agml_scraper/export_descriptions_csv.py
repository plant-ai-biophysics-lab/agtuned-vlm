"""Export agml_descriptions.jsonl to a flat CSV for downstream verification."""

import argparse
import csv
import json
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    src = Path(args.descriptions_jsonl)
    dst = Path(args.output_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = record.get("metadata", {})
            rows.append({
                "agml_dataset":           meta.get("agml_dataset", ""),
                "crop":                   meta.get("crop", ""),
                "class_label":            meta.get("class_label", ""),
                "task_type":              meta.get("task_type", ""),
                "compressed_description": record["messages"][1]["content"],
            })

    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["agml_dataset", "crop", "class_label", "task_type", "compressed_description"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} records → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export agml_descriptions.jsonl to CSV.")
    parser.add_argument("--descriptions_jsonl", default="outputs/agml_descriptions.jsonl")
    parser.add_argument("--output_csv", default="outputs/descriptions_export.csv")
    main(parser.parse_args())
