"""
Paper 2, Stage 4 (docs/stage4_detection_context_memory.md), Step 4 —
ablation: does the detection skill-memory .md file (heldout-detection-
summarize's own output) actually improve zero-shot precision/recall/F1
on the held-out EVAL split, or is it just memorization of the lesson-
building slice? No GPU, no model calls — pure post-hoc comparison of two
already-scored heldout-detection-zeroshot checkpoints, both run against
--split eval. Detection analog of heldout_ablation_summary.py — same
shape, `precision`/`recall`/`f1` instead of `judge_accuracy` (detection
has no judge-based correctness metric — IoU-matched P/R/F1 is the
ground truth here, already deterministic, no judge-leniency confound to
worry about the way classification has).

  --baseline_ckpt  heldout-detection-zeroshot --split eval            (no skill file)
  --skill_ckpt     heldout-detection-zeroshot --split eval --skill_memory_path ...

Both checkpoints come from the SAME eval split (same manifest, same
dataset-membership split), so they cover the identical set of images —
no filtering/intersection needed before comparing.

Macro-averaged (mean of each row's own precision_at_iou50/
recall_at_iou50/f1_at_iou50), same convention detection_final_metrics.py's
own _aggregate() helper already uses elsewhere in this pipeline — not a
pooled/micro-average over summed n_pred/n_gt/n_matched, which is also
reported alongside for reference but never used to compute the metric
delta itself.

Usage:
    uv run python main.py heldout-detection-ablation-summary \\
        --baseline_ckpt outputs_final/heldout_detection_zeroshot_eval_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --skill_ckpt outputs_final/heldout_detection_zeroshot_eval_with_skill/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — pred_boxes_json/gt_boxes_json can exceed the 131072-byte default on a full run.
import json

_CONDITION_COLORS = {"Without skill file": "dodgerblue", "With skill file": "mediumorchid"}
_METRICS = ("precision", "recall", "f1")


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for metric, col in (("precision", "precision_at_iou50"), ("recall", "recall_at_iou50"), ("f1", "f1_at_iou50")):
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        out[metric] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["n_pred"] = sum(int(r.get("n_pred") or 0) for r in rows)
    out["n_gt"] = sum(int(r.get("n_gt") or 0) for r in rows)
    out["n_matched"] = sum(int(r.get("n_matched") or 0) for r in rows)
    return out


def plot_ablation(baseline: dict, skill: dict, out_path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = ["Without skill file", "With skill file"]
    colors = [_CONDITION_COLORS[n] for n in names]
    x = np.arange(len(_METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, cond) in enumerate(zip(names, (baseline, skill))):
        vals = [cond[m] for m in _METRICS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in _METRICS])
    ax.set_ylabel("Score (macro-averaged)")
    ax.set_ylim(0, 1.05)
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Detection skill-memory ablation on held-out eval split{subtitle}\n(n={baseline['n']})",
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    baseline_rows = load_rows(Path(args.baseline_ckpt))
    skill_rows = load_rows(Path(args.skill_ckpt))
    if not baseline_rows:
        raise SystemExit(f"ERROR: {args.baseline_ckpt} has no rows.")
    if not skill_rows:
        raise SystemExit(f"ERROR: {args.skill_ckpt} has no rows.")

    baseline_images = {r["image_name"] for r in baseline_rows}
    skill_images = {r["image_name"] for r in skill_rows}
    if baseline_images != skill_images:
        only_baseline = baseline_images - skill_images
        only_skill = skill_images - baseline_images
        print(f"  [warn] checkpoints don't cover the identical image set — "
              f"{len(only_baseline)} only in --baseline_ckpt, {len(only_skill)} only in --skill_ckpt "
              f"(did both come from the SAME --split eval run of heldout-detection-zeroshot?)")

    baseline = compute_metrics(baseline_rows)
    skill = compute_metrics(skill_rows)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.skill_ckpt).parent / "ablation"
    out_path = out_dir / "ablation.png"
    plot_ablation(baseline, skill, out_path, title=args.title)

    deltas = {m: round(skill[m] - baseline[m], 4) for m in _METRICS}
    result = {"without_skill_file": baseline, "with_skill_file": skill, "delta": deltas}
    json_path = out_dir / "ablation.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"  Without skill file: " + ", ".join(f"{m}={baseline[m]:.4f}" for m in _METRICS) + f" (n={baseline['n']})")
    print(f"  With skill file:    " + ", ".join(f"{m}={skill[m]:.4f}" for m in _METRICS) + f" (n={skill['n']})")
    print(f"  Delta: " + ", ".join(f"{m}={deltas[m]:+.4f}" for m in _METRICS))
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {json_path}")
    if deltas["f1"] <= 0:
        print("\n  [finding] skill file did NOT improve held-out F1 — "
              "do not inject into detection-multiclass based on this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_ckpt", required=True,
                         help="heldout-detection-zeroshot --split eval checkpoint, run WITHOUT "
                              "--skill_memory_path.")
    parser.add_argument("--skill_ckpt", required=True,
                         help="heldout-detection-zeroshot --split eval checkpoint, run WITH "
                              "--skill_memory_path.")
    parser.add_argument("--out_dir", default=None, help="Default: <skill_ckpt's own dir>/ablation/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
