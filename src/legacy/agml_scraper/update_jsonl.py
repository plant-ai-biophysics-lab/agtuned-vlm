"""Apply corrected descriptions from CSV back into agml_descriptions.jsonl and agml_classification.jsonl."""

import argparse
import csv
import json
import re
from pathlib import Path


def _load_corrections(csv_path: Path) -> dict[tuple[str, str], dict]:
    corrections = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "corrected":
                key = (row["agml_dataset"], row["class_label"])
                corrections[key] = row
    return corrections


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _update_descriptions(records: list[dict], corrections: dict) -> tuple[list[dict], int, int]:
    updated = skipped = 0
    out = []
    for record in records:
        meta = record.get("metadata", {})
        key = (meta.get("agml_dataset", ""), meta.get("class_label", ""))
        if key in corrections:
            corr = corrections[key]
            desc = corr["corrected_description"]
            source_urls = [u for u in corr["source_urls"].split(",") if u]
            record["messages"][1]["content"] = desc
            record["metadata"]["compressed_description"] = desc
            record["metadata"]["verified"] = True
            record["metadata"]["verification_sources"] = source_urls
            updated += 1
        else:
            skipped += 1
        out.append(record)
    return out, updated, skipped


def _update_classification(records: list[dict], corrections: dict) -> tuple[list[dict], int, int]:
    # Matches "Description: <anything>" at the end of messages[0]["content"]
    desc_re = re.compile(r'(Description: )(.+)$', re.DOTALL)
    updated = skipped = 0
    out = []
    for record in records:
        meta = record.get("metadata", {})
        key = (meta.get("agml_dataset", ""), meta.get("class_label", ""))
        if key in corrections:
            corr = corrections[key]
            desc = corr["corrected_description"]
            source_urls = [u for u in corr["source_urls"].split(",") if u]
            original_content = record["messages"][0]["content"]
            record["messages"][0]["content"] = desc_re.sub(
                lambda m: m.group(1) + desc, original_content
            )
            record["metadata"]["verified"] = True
            record["metadata"]["verification_sources"] = source_urls
            updated += 1
        else:
            skipped += 1
        out.append(record)
    return out, updated, skipped


def main(args: argparse.Namespace) -> None:
    corrections = _load_corrections(Path(args.corrected_csv))
    print(f"Corrections loaded: {len(corrections)}")

    desc_records = _load_jsonl(Path(args.descriptions_jsonl))
    cls_records = _load_jsonl(Path(args.classification_jsonl))

    desc_out, desc_updated, desc_skipped = _update_descriptions(desc_records, corrections)
    cls_out, cls_updated, cls_skipped = _update_classification(cls_records, corrections)

    out_desc = Path(args.output_descriptions_jsonl)
    out_cls = Path(args.output_classification_jsonl)
    out_desc.parent.mkdir(parents=True, exist_ok=True)
    out_cls.parent.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out_desc, desc_out)
    _write_jsonl(out_cls, cls_out)

    print(f"Descriptions  — updated: {desc_updated}  skipped: {desc_skipped}  → {out_desc}")
    print(f"Classification — updated: {cls_updated}  skipped: {cls_skipped}  → {out_cls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply corrected descriptions back to JSONL files.")
    parser.add_argument("--corrected_csv", default="outputs/descriptions_corrected.csv")
    parser.add_argument("--descriptions_jsonl", default="outputs/agml_descriptions.jsonl")
    parser.add_argument("--classification_jsonl", default="outputs/agml_classification.jsonl")
    parser.add_argument("--output_descriptions_jsonl", default="outputs/agml_descriptions_verified.jsonl")
    parser.add_argument("--output_classification_jsonl", default="outputs/agml_classification_verified.jsonl")
    main(parser.parse_args())
