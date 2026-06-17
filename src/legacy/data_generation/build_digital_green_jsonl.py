"""
Build description + classification JSONL files from a Digital Green annotations CSV.

Direct adapter — no inference required. Converts existing `diagnosis` + `details`
columns into the same JSONL format produced by Stage 1 (generate + verify), so all
downstream pipeline stages (language, vision, error-overlap, etc.) work unchanged.

Also writes `digital_green_datasets.yaml` alongside the JSONL so the assess scripts
can resolve predictions against the discovered class list.

Usage:
    python main.py build-dg-jsonl \\
        --annotations  data/digital_green/annotations.csv \\
        --images_dir   data/digital_green/images/ \\
        --output_dir   outputs/data_generation/ \\
        --prompts      src/config/prompts.yaml
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.io import load_prompts


# ── Text cleaning (mirrors src/finetune/train.py) ─────────────────────────────

def _clean_details(text: str) -> str:
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[•\-\*]\s+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _clean_diagnosis(text: str) -> str:
    parts = [p.strip() for p in text.split(';')]
    return ' and '.join(p for p in parts if p)


def _image_filename(url: str) -> str:
    return Path(url).name


# ── Image downloading ──────────────────────────────────────────────────────────

def download_images(rows: list[dict], images_dir: Path) -> dict[str, Path]:
    """Download missing images. Returns {url: local_path}."""
    try:
        import requests
    except ImportError:
        print("  [warn] requests not installed — skipping image download")
        return {r["image_url"]: images_dir / _image_filename(r["image_url"]) for r in rows if r.get("image_url")}

    images_dir.mkdir(parents=True, exist_ok=True)
    url_to_path: dict[str, Path] = {}
    to_fetch = []
    for r in rows:
        if not r.get("image_url"):
            continue
        dst = images_dir / _image_filename(r["image_url"])
        url_to_path[r["image_url"]] = dst
        if not dst.exists():
            to_fetch.append((r["image_url"], dst))

    if to_fetch:
        print(f"  Downloading {len(to_fetch)} images to {images_dir} ...")
        for url, dst in to_fetch:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dst.write_bytes(resp.content)
            except Exception as e:
                print(f"    [warn] Failed {url}: {e}")
    else:
        print(f"  All {len(url_to_path)} images already cached in {images_dir}")

    return url_to_path


# ── Record builders ────────────────────────────────────────────────────────────

def _build_desc_record(
    crop: str,
    diagnosis: str,
    details: str,
    image_names: list[str],
    prompts: dict,
) -> dict:
    text_q = prompts["text_description_disease"].format(cls=diagnosis, crop=crop)
    vis_q  = prompts["visual_description_disease"].format(crop=crop)
    vis_cls_q = prompts["visual_classification_disease"].format(crop=crop)
    return {
        "messages": [
            {"role": "user",      "content": text_q},
            {"role": "assistant", "content": details},
        ],
        "metadata": {
            "record_type":      "description",
            "agml_dataset":     "digital_green",
            "crop":             crop,
            "class_label":      diagnosis,
            "task_type":        "disease",
            "visual_question":      vis_q,
            "visual_cls_question":  vis_cls_q,
            "image_names":      image_names,
            "n_images_used":    len(image_names),
            "generator_model":  "digital_green_csv",
        },
    }


def _build_cls_record(
    crop: str,
    diagnosis: str,
    details: str,
    prompts: dict,
) -> dict:
    text_q = prompts["text_classification_disease"].format(crop=crop, description=details)
    return {
        "messages": [
            {"role": "user",      "content": text_q},
            {"role": "assistant", "content": diagnosis},
        ],
        "metadata": {
            "record_type":  "classification",
            "agml_dataset": "digital_green",
            "crop":         crop,
            "class_label":  diagnosis,
            "task_type":    "disease",
            "generator_model": "digital_green_csv",
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))

    print(f"Reading annotations: {args.annotations}")
    with open(args.annotations, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows loaded")

    # Download / verify images
    images_dir = Path(args.images_dir)
    url_to_path = download_images(rows, images_dir)

    # Group by (crop, diagnosis) — each group = one JSONL record
    groups: dict[tuple, list[dict]] = defaultdict(list)
    skipped = 0
    for row in rows:
        crop      = row.get("crop", "").strip()
        diagnosis = _clean_diagnosis(row.get("diagnosis", ""))
        details   = _clean_details(row.get("details", ""))
        url       = row.get("image_url", "")
        if not crop or not diagnosis or not details:
            skipped += 1
            continue
        groups[(crop, diagnosis)].append({
            "details":  details,
            "filename": _image_filename(url) if url else "",
        })
    print(f"  {len(groups)} (crop, diagnosis) groups  |  {skipped} rows skipped (missing fields)")

    desc_path = output_dir / "dg_descriptions.jsonl"
    cls_path  = output_dir / "dg_classification.jsonl"
    n_desc = n_cls = 0

    with open(desc_path, "w", encoding="utf-8") as fd, \
         open(cls_path,  "w", encoding="utf-8") as fc:

        for (crop, diagnosis), group_rows in sorted(groups.items()):
            # Canonical description: most common details text in the group
            details = Counter(r["details"] for r in group_rows).most_common(1)[0][0]
            image_names = [r["filename"] for r in group_rows if r["filename"]]

            fd.write(json.dumps(_build_desc_record(crop, diagnosis, details, image_names, prompts)) + "\n")
            fc.write(json.dumps(_build_cls_record(crop, diagnosis, details, prompts)) + "\n")
            n_desc += 1
            n_cls  += 1

    print(f"\nWrote {n_desc} description records → {desc_path}")
    print(f"Wrote {n_cls}  classification records → {cls_path}")

    # Unique diagnoses for the dataset config
    unique_diagnoses = sorted({diagnosis for _, diagnosis in groups.keys()})

    # Write digital_green_datasets.yaml
    dg_yaml_path = output_dir / "digital_green_datasets.yaml"
    dg_config = {
        "datasets": [{
            "agml_name":  "digital_green",
            "crop":       "multiple",
            "task_type":  "disease",
            "n_images":   len(rows),
            "classes":    unique_diagnoses,
            "location":   ["asia", "india"],
            "skip":       False,
            "notes":      "Digital Green advisory dataset; adapted from diagnosis+details CSV columns",
        }]
    }
    with open(dg_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dg_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Wrote dataset config  → {dg_yaml_path}")
    print(f"  {len(unique_diagnoses)} unique diagnoses discovered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Digital Green JSONL files from annotations CSV."
    )
    parser.add_argument("--annotations",  default="data/digital_green/annotations.csv")
    parser.add_argument("--images_dir",   default="data/digital_green/images/")
    parser.add_argument("--output_dir",   default="outputs/data_generation/")
    parser.add_argument("--prompts",      default="src/config/prompts.yaml")
    main(parser.parse_args())
