"""
Compare Verifier-CoT performance across rounds — normal (early-stopping) vs
forced (--force_max_rounds). No model/GPU needed.

Produces two charts, using whatever data is actually available:

  accuracy_by_stopping_round.png  (Option A — works with any existing run,
    including data from before this script existed)
    x = the round at which each image's chain ended (n_rounds)
    y = judged accuracy of images that ended at that round
    Normal-run only shows a real curve here, since forcing makes every image
    end at max_rounds by definition — the forced run is drawn as a single
    reference line at its overall accuracy, not a curve, since there's no
    round-to-round variation to plot for it under this framing.

  accuracy_by_round.png  (Option B — needs round_prediction logged in
    transcript_json, only present in runs from AFTER run_verifier_cot.py
    started generating a classification snapshot every round, not just at
    the end. Older CSVs simply won't have this chart drawn — the round-1
    infrastructure (per-round classification) was added to give both normal
    and forced runs a real per-round accuracy curve, not just normal.)
    x = round number
    y = accuracy of that round's classification snapshot (resolve_class vs
        ground truth — a cheap proxy, not a full LLM-judge pass, since
        judging every round for every image would be substantially more
        expensive than judging only the final answer)

Usage:
  uv run python experiments/verifier_cot/compare_rounds.py \\
    --normal_judged_csv outputs/verifier_cot/ckpt_judged.csv \\
    --forced_judged_csv outputs/verifier_cot_forced_5/ckpt_judged.csv \\
    --output_dir outputs/verifier_cot_comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import datasets_by_name  # noqa: E402
from src.common.metrics import resolve_class  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def accuracy_by_stopping_round(rows: list[dict]) -> dict[int, dict]:
    by_round: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        jc = r.get("judged_correct", "")
        if jc == "":
            continue
        by_round[int(r["n_rounds"])].append(int(jc))
    return {
        rnd: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for rnd, vals in sorted(by_round.items())
    }


def accuracy_by_round_snapshot(rows: list[dict], config_path: str) -> dict[int, dict]:
    """Requires round_prediction in transcript_json (Option B logging).
    Returns {} if no row has it — caller should treat that as 'not available yet'."""
    cfg_by_name = datasets_by_name(Path(config_path))
    by_round: dict[int, list[int]] = defaultdict(list)
    any_found = False
    for r in rows:
        try:
            transcript = json.loads(r.get("transcript_json", "[]"))
        except json.JSONDecodeError:
            continue
        cfg = cfg_by_name.get(r["agml_dataset"])
        clean_classes = cfg.clean_classes if cfg else None
        gt = r.get("class_label", "")
        for entry in transcript:
            pred = entry.get("round_prediction")
            if pred is None:
                continue
            any_found = True
            resolved = resolve_class(pred, clean_classes) if clean_classes else pred
            correct = int((resolved or "").strip().lower() == gt.strip().lower())
            by_round[int(entry["round"])].append(correct)
    if not any_found:
        return {}
    return {
        rnd: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for rnd, vals in sorted(by_round.items())
    }


def make_charts(normal_rows: list[dict] | None, forced_rows: list[dict] | None,
                config_path: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Option A: accuracy by stopping round ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    if normal_rows:
        acc = accuracy_by_stopping_round(normal_rows)
        rounds = sorted(acc)
        ax.plot(rounds, [acc[r]["accuracy"] * 100 for r in rounds], marker="o",
               color="#4C72B0", linewidth=2, label="Normal (early-stopping)")
        for r in rounds:
            ax.annotate(f"n={acc[r]['n']}", (r, acc[r]["accuracy"] * 100), fontsize=7,
                       ha="center", xytext=(0, 8), textcoords="offset points")
    if forced_rows:
        acc = accuracy_by_stopping_round(forced_rows)
        overall = sum(v["n"] * v["accuracy"] for v in acc.values()) / sum(v["n"] for v in acc.values())
        ax.axhline(overall * 100, color="#C44E52", linestyle="--", linewidth=2,
                  label=f"Forced (all @ max round) — overall {overall*100:.1f}%")
    ax.set_xlabel("Round chain ended at (n_rounds)")
    ax.set_ylabel("Judged accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by stopping round — Normal vs Forced")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "accuracy_by_stopping_round.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Option B: real per-round accuracy curve (needs round_prediction) ───
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for rows, label, color in [(normal_rows, "Normal (early-stopping)", "#4C72B0"),
                               (forced_rows, "Forced (all rounds)", "#C44E52")]:
        if not rows:
            continue
        acc = accuracy_by_round_snapshot(rows, config_path)
        if not acc:
            print(f"  [skip] {label}: no round_prediction found in transcript_json — "
                  f"this CSV predates per-round classification logging, rerun to get it.")
            continue
        any_data = True
        rounds = sorted(acc)
        ax.plot(rounds, [acc[r]["accuracy"] * 100 for r in rounds], marker="o",
               color=color, linewidth=2, label=label)
        for r in rounds:
            ax.annotate(f"n={acc[r]['n']}", (r, acc[r]["accuracy"] * 100), fontsize=7,
                       ha="center", xytext=(0, 8), textcoords="offset points")
    if any_data:
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy of that round's classification snapshot (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Per-round accuracy — Normal vs Forced")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "accuracy_by_round.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        print("  [skip] accuracy_by_round.png — no round_prediction data in either CSV yet.")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Verifier-CoT accuracy across rounds: normal vs forced")
    parser.add_argument("--normal_judged_csv", default=None)
    parser.add_argument("--forced_judged_csv", default=None)
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--output_dir", default="outputs/verifier_cot_comparison/")
    args = parser.parse_args()

    if not args.normal_judged_csv and not args.forced_judged_csv:
        print("ERROR: pass at least one of --normal_judged_csv / --forced_judged_csv")
        return

    normal_rows = _read_csv(Path(args.normal_judged_csv)) if args.normal_judged_csv else None
    forced_rows = _read_csv(Path(args.forced_judged_csv)) if args.forced_judged_csv else None
    if normal_rows:
        print(f"Loaded {len(normal_rows)} normal-run rows")
    if forced_rows:
        print(f"Loaded {len(forced_rows)} forced-run rows")

    make_charts(normal_rows, forced_rows, args.config, Path(args.output_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
