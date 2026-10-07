"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 2 QC —
read heldout_detection_explain_traces.py's own checkpoint and render a
handful of samples (image + predicted/ground-truth boxes + the
explanation the model actually wrote) for visual sanity-checking, same
"read it yourself before trusting it" purpose Step -1's
detection_skill_representation_ablation.py / detection_qc_plot.py
already serve for the stages before this one.

No model calls here — this is a pure report generator over Step 2's
already-produced CSV, so it costs nothing to re-run after every real
Step 2 pass.

Round-robin sampled across ALL error_type values actually present in
the checkpoint (false_positive/missed/loose_match/wrong_class_label/
worked — see heldout_detection_explain_traces.py's own docstring for
what each means), not hardcoded to a fixed 3- or 5-way list — so this
still works unchanged if that taxonomy ever grows. Without this,
--n_samples would skew almost entirely toward whichever error_type is
most numerous (false_positive, from detection-multiclass's own
deliberate over-generation) and a QC read would barely see the other
categories at all.

Usage:
    uv run python main.py heldout-detection-explain-traces-qc \\
        --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --n_samples 20
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — pred_bbox_json/gt_bbox_json/explanation can exceed the 131072-byte default.
import json
import random
import textwrap
from collections import defaultdict

from PIL import Image as PILImage

from src.common.bbox_viz import plot_predictions


def sample_round_robin_by_error_type(rows: list[dict], n_samples: int, seed: int) -> list[dict]:
    """Up to n_samples rows, round-robin across every DISTINCT error_type
    value present — generic over however many categories exist (not
    hardcoded to 3 or 5), so no single type dominates the sample purely
    by base rate. Order of types is first-seen in `rows` (deterministic
    given a fixed checkpoint, not alphabetical/random) so re-running with
    the same --seed on the same file reproduces the identical sample."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    types: list[str] = []
    for r in rows:
        t = r.get("error_type") or "unknown"
        if t not in by_type:
            types.append(t)
        by_type[t].append(r)
    for v in by_type.values():
        rng.shuffle(v)

    sampled: list[dict] = []
    idx = {t: 0 for t in types}
    while len(sampled) < n_samples and any(idx[t] < len(by_type[t]) for t in types):
        for t in types:
            if idx[t] < len(by_type[t]):
                sampled.append(by_type[t][idx[t]])
                idx[t] += 1
                if len(sampled) >= n_samples:
                    break
    return sampled


def render_qc_report(sampled: list[dict], images_root: Path, out_dir: Path) -> int:
    import matplotlib.pyplot as plt

    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    n_rendered = 0

    for i, r in enumerate(sampled):
        img_path = images_root / r["agml_dataset"] / r["image_name"]
        if not img_path.exists():
            print(f"  [skip] {r['agml_dataset']}/{r['image_name']} — image not found at {img_path}")
            continue
        image = PILImage.open(img_path).convert("RGB")

        pred_box = json.loads(r["pred_bbox_json"]) if r.get("pred_bbox_json") else None
        gt_box = json.loads(r["gt_bbox_json"]) if r.get("gt_bbox_json") else None
        iou = float(r.get("iou") or 0.0)

        annotated_path = annotated_dir / f"sample{i:03d}_{r['error_type']}.png"
        plot_predictions(
            image, [pred_box] if pred_box else [], [gt_box] if gt_box else None,
            out_path=annotated_path, title=f"{r['error_type']} — {r['category']} (IoU={iou:.2f})",
        )
        annotated_image = PILImage.open(annotated_path).convert("RGB")

        fig = plt.figure(figsize=(11, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[5, 3.5], hspace=0.08)

        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(annotated_image)
        ax_img.axis("off")
        ax_img.set_title(f"{r['agml_dataset']}/{r['image_name']}", fontsize=9)

        ax_text = fig.add_subplot(gs[1])
        ax_text.axis("off")
        pred_text = f"{[round(v, 1) for v in pred_box['bbox']]} ({pred_box.get('category_name')})" \
            if pred_box else "(none predicted)"
        gt_text = f"{[round(v, 1) for v in gt_box['bbox']]} ({gt_box.get('category_name')})" \
            if gt_box else "(no real instance here)"
        question_wrapped = "\n".join(textwrap.wrap(r.get("question", ""), width=150))
        explanation_wrapped = "\n".join(textwrap.wrap(r.get("explanation", ""), width=150))
        ax_text.text(
            0.0, 1.0,
            f"error_type: {r['error_type']}    category: {r['category']}    IoU: {iou:.2f}\n"
            f"Predicted box: {pred_text}\n"
            f"Ground truth box: {gt_text}\n\n"
            f"QUESTION:\n{question_wrapped}\n\n"
            f"EXPLANATION (model's own words):\n{explanation_wrapped}",
            ha="left", va="top", fontsize=9, family="monospace", wrap=True,
        )

        out_path = out_dir / f"qc_sample{i:03d}_{r['error_type']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_rendered += 1

    return n_rendered


def main(args: argparse.Namespace) -> None:
    with open(args.ckpt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: {args.ckpt} has no rows.")

    sampled = sample_round_robin_by_error_type(rows, args.n_samples, args.seed)
    counts = defaultdict(int)
    for r in sampled:
        counts[r["error_type"]] += 1
    print(f"  {len(sampled)} sample(s) from {len(rows)} total: {dict(counts)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rendered = render_qc_report(sampled, Path(args.images_root), out_dir)
    print(f"\n  {n_rendered} QC report image(s) written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="heldout-detection-explain-traces' own output checkpoint CSV.")
    parser.add_argument("--images_root", default="dataset/images_detection_heldout",
                         help="Same root Step 2 itself read from (<images_root>/<agml_dataset>/<image_name>).")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="outputs_final/heldout_detection_explain_traces_qc/")
    main(parser.parse_args())
