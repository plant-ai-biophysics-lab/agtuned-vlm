"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 4 — ablation: does
the skill-memory .md file (heldout-summarize's own output) actually
improve zero-shot `judge_accuracy` on the held-out EVAL split, or is it
just memorization of the lesson-building slice? No GPU, no model calls —
pure post-hoc comparison of two already-judged heldout-zeroshot
checkpoints, both run against --split eval:

  --baseline_ckpt  heldout-zeroshot --split eval            (no skill file)
  --skill_ckpt     heldout-zeroshot --split eval --skill_memory_path ...

Both checkpoints come from the SAME eval split (same manifest, same
seeded stratified split), so they cover the identical set of images —
no filtering/intersection needed before comparing, unlike
verifier-cot-comparison-summary's baseline-dataset-scope filtering.

Usage:
    uv run python main.py heldout-ablation-summary \\
        --baseline_ckpt outputs_final/heldout_zeroshot_eval_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --skill_ckpt outputs_final/heldout_zeroshot_eval_with_skill/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json

_CONDITION_COLORS = {"Without skill file": "dodgerblue", "With skill file": "mediumorchid"}


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_judge_accuracy(rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("judged_correct") not in (None, "")]
    acc = sum(int(r["judged_correct"]) for r in judged) / len(judged) if judged else 0.0
    return {"n": len(rows), "n_judged": len(judged), "judge_accuracy": round(acc, 4)}


def plot_ablation(baseline: dict, skill: dict, out_path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Without skill file", "With skill file"]
    vals = [baseline["judge_accuracy"], skill["judge_accuracy"]]
    colors = [_CONDITION_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(names, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("judge_accuracy")
    ax.set_ylim(0, 1.05)
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Skill-memory ablation on held-out eval split{subtitle}\n(n={baseline['n']})", fontsize=11)
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
              f"(did both come from the SAME --split eval run of heldout-zeroshot?)")

    baseline = compute_judge_accuracy(baseline_rows)
    skill = compute_judge_accuracy(skill_rows)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.skill_ckpt).parent / "ablation"
    out_path = out_dir / "ablation.png"
    plot_ablation(baseline, skill, out_path, title=args.title)

    result = {"without_skill_file": baseline, "with_skill_file": skill,
              "delta": round(skill["judge_accuracy"] - baseline["judge_accuracy"], 4)}
    json_path = out_dir / "ablation.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"  Without skill file: judge_accuracy={baseline['judge_accuracy']:.4f} (n={baseline['n']})")
    print(f"  With skill file:    judge_accuracy={skill['judge_accuracy']:.4f} (n={skill['n']})")
    print(f"  Delta: {result['delta']:+.4f}")
    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {json_path}")
    if result["delta"] <= 0:
        print("\n  [finding] skill file did NOT improve held-out judge_accuracy — "
              "do not inject into vision-baseline/verifier-cot based on this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_ckpt", required=True,
                         help="heldout-zeroshot --split eval checkpoint, run WITHOUT --skill_memory_path.")
    parser.add_argument("--skill_ckpt", required=True,
                         help="heldout-zeroshot --split eval checkpoint, run WITH --skill_memory_path.")
    parser.add_argument("--out_dir", default=None, help="Default: <skill_ckpt's own dir>/ablation/")
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
