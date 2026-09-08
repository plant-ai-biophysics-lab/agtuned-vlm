"""
Patch healthy classes in an already-generated descriptions_corrected.csv.

Sets corrected_description to "This leaf/plant is healthy." and status to
"healthy_override" for any row whose class_label contains "healthy" (case-insensitive).

Usage:
    python main.py patch-healthy   (via main.py dispatcher)
    python src/legacy/data_generation/patch_healthy.py --csv outputs/data_generation/descriptions_corrected.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv

HEALTHY_DESCRIPTION = "This leaf/plant is healthy."
FIELDNAMES = [
    "agml_dataset", "crop", "class_label", "task_type",
    "compressed_description", "corrected_description",
    "status", "source_orgs", "source_urls", "excerpt_used",
]


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    patched = 0
    for row in rows:
        if "healthy" in row["class_label"].lower():
            if row.get("corrected_description") != HEALTHY_DESCRIPTION:
                row["corrected_description"] = HEALTHY_DESCRIPTION
                row["status"] = "healthy_override"
                row["source_orgs"] = ""
                row["source_urls"] = ""
                row["excerpt_used"] = ""
                patched += 1

    if patched == 0:
        print("No healthy classes found that needed patching.")
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched {patched} rows → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch healthy classes in descriptions_corrected.csv")
    parser.add_argument(
        "--csv", default="outputs/data_generation/descriptions_corrected.csv",
        help="Path to descriptions_corrected.csv",
    )
    main(parser.parse_args())
