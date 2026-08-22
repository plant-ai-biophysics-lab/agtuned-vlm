"""
Stage 5 (prep) — no-GPU, no-network report: summarize
catalog_detection_datasets.py's output (src/config/hf_detection_catalog.json)
into a compact table — class-name resolution status, bbox scale/format,
image counts — easy to eyeball locally or paste back for review, instead of
scrolling raw JSON.

Usage:
  uv run python main.py report-detection-catalog \\
    --catalog src/config/hf_detection_catalog.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/config/hf_detection_catalog.json")
    ap.add_argument("--csv_out", default=None, help="Optional: also write the table to a CSV file.")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text())
    rows = []
    for name, meta in sorted(catalog.items()):
        bc = meta.get("bbox_convention") or {}
        class_names = meta.get("class_names")
        cfg = meta.get("configs", {}).get(meta.get("preferred_config", "default"), {})
        n_images = sum(cfg.get("splits", {}).values()) if cfg.get("splits") else 0
        rows.append({
            "dataset": name,
            "resolved": "yes" if class_names else "NO",
            "n_classes": len(class_names) if class_names else 0,
            "bbox_scale": bc.get("scale", "unknown"),
            "bbox_format": bc.get("format", "unknown"),
            "xyxy_frac": bc.get("xyxy_consistent_frac", ""),
            "n_probed": bc.get("n_probed", 0),
            "n_images": n_images,
            "error": bc.get("error", ""),
        })

    n_resolved = sum(1 for r in rows if r["resolved"] == "yes")
    n_pixel = sum(1 for r in rows if r["bbox_scale"] == "pixel")
    n_normalized = sum(1 for r in rows if r["bbox_scale"] == "normalized")
    n_unknown_scale = sum(1 for r in rows if r["bbox_scale"] == "unknown")

    col_widths = {
        "dataset": max(len(r["dataset"]) for r in rows) if rows else 7,
        "resolved": 8, "n_classes": 9, "bbox_scale": 10, "bbox_format": 11,
        "xyxy_frac": 9, "n_probed": 8, "n_images": 8,
    }
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(str(r[k]).ljust(col_widths[k]) for k in col_widths)
        if r["error"]:
            line += f"  ERROR: {r['error']}"
        print(line)

    print(f"\n{len(rows)} dataset(s): {n_resolved} class_names resolved, {len(rows) - n_resolved} unresolved")
    print(f"bbox_scale: {n_pixel} pixel, {n_normalized} normalized, {n_unknown_scale} unknown")

    if args.csv_out:
        import csv
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
