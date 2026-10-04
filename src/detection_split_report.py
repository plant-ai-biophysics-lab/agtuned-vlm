"""
Paper 2 Stage 5 — Split-behavior report: for each image in a
detection-box-refine checkpoint, does a SPLIT correction actually grow
the box count, or does it mostly resolve to 1-in-1-out no-ops (the model
disagreeing a split was needed)? No GPU needed — reads straight from the
checkpoint's own per-box `history` (detection_box_refine.py's per-round
snapshot list, now also carrying `split_group_size` on the round where a
split was decided — how many boxes THAT specific split call produced).

Sibling boxes from the SAME split share an identical COPIED history
entry up through the split round (see detection_box_refine.py's history-
inheritance) — naively scanning every box's own history would count the
same split event once per sibling it produced. This report deduplicates
on (round, pre-split bbox, pre-split category_name) so each real split
event is counted exactly once, regardless of how many (or how few)
boxes it ended up producing.

Usage:
    uv run python main.py detection-split-report \\
        --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
    uv run python main.py detection-split-report \\
        --ckpt ... --image_name 84.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection_box_verifier_summary import load_rows  # noqa: E402


def report_split_behavior(rows: list[dict]) -> list[dict]:
    """One dict per image: {"agml_dataset", "image_name", "n_pred_final"
    (straight off pred_boxes_json's own length, pre-filter), "n_split_events"
    (deduplicated), "n_real_splits" (group_size > 1), "n_noop_splits"
    (group_size <= 1 — model disagreed a split was needed),
    "split_group_sizes" (list, one per event), "boxes_added_by_splits"
    (sum of max(group_size - 1, 0) across events — net boxes splitting
    actually contributed to this image's final count).

    NOTE: only covers boxes that SURVIVED to the final checkpoint — a
    split that returned [] (dropped, nothing real there) isn't visible
    here at all, since dropped boxes never make it into pred_boxes_json.
    This under-counts total split CALLS relative to
    detection_box_refine.py's own `stats["split_calls"]` (which does
    count every call, drops included) — this report is specifically
    about what happened to boxes that made it into the final output."""
    out = []
    for r in rows:
        boxes = json.loads(r.get("pred_boxes_json") or "[]")
        seen_events: set[tuple] = set()
        group_sizes = []
        for b in boxes:
            for h in b.get("history") or []:
                if h.get("action") != "split":
                    continue
                key = (h["round"], tuple(h["bbox"]), h["category_name"])
                if key in seen_events:
                    continue
                seen_events.add(key)
                group_sizes.append(h.get("split_group_size", 1))
        out.append({
            "agml_dataset": r["agml_dataset"], "image_name": r["image_name"],
            "n_pred_final": len(boxes),
            "n_split_events": len(group_sizes),
            "n_real_splits": sum(1 for g in group_sizes if g > 1),
            "n_noop_splits": sum(1 for g in group_sizes if g <= 1),
            "split_group_sizes": group_sizes,
            "boxes_added_by_splits": sum(max(g - 1, 0) for g in group_sizes),
        })
    return out


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    if args.image_name:
        rows = [r for r in rows if r["image_name"] == args.image_name]
        if not rows:
            raise SystemExit(f"ERROR: no row with image_name={args.image_name!r} in {ckpt_path}.")

    report = report_split_behavior(rows)

    print(f"  {'dataset':40s} {'image':15s} {'n_pred':>7s} {'splits':>7s} "
          f"{'real':>5s} {'noop':>5s} {'added':>6s}  group_sizes")
    for row in report:
        if row["n_split_events"] == 0 and not args.show_all:
            continue
        print(f"  {row['agml_dataset']:40s} {row['image_name']:15s} {row['n_pred_final']:>7d} "
              f"{row['n_split_events']:>7d} {row['n_real_splits']:>5d} {row['n_noop_splits']:>5d} "
              f"{row['boxes_added_by_splits']:>6d}  {row['split_group_sizes']}")

    n_with_splits = sum(1 for r in report if r["n_split_events"] > 0)
    total_events = sum(r["n_split_events"] for r in report)
    total_real = sum(r["n_real_splits"] for r in report)
    total_added = sum(r["boxes_added_by_splits"] for r in report)
    print(f"\n  {len(report)} image(s), {n_with_splits} with at least one split event")
    print(f"  {total_events} total split event(s): {total_real} real (grew the box count), "
          f"{total_events - total_real} no-op (model disagreed a split was needed)")
    print(f"  {total_added} net box(es) added to final checkpoints by splitting overall")
    if not args.show_all:
        print("  (pass --show_all to also list images with zero split events)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="A detection-box-refine checkpoint CSV.")
    parser.add_argument("--image_name", default=None, help="Restrict to one image (e.g. 84.jpg).")
    parser.add_argument("--show_all", action="store_true",
                         help="Also list images with zero split events (default: only images where "
                              "at least one split happened).")
    main(parser.parse_args())
