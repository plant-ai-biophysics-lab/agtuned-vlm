"""
Confidence analysis for Verifier-CoT — post-hoc analysis of an already-judged
CSV from run_verifier_cot.py. No model/GPU needed.

Answers three questions:
  1. How does verifier confidence vary across rounds (does it start low and
     climb as the VLM's statement improves, or is it flat)?
  2. What does the final confidence distribution look like (is the verifier
     actually discriminating, or clustering at one end regardless of context)?
  3. Does higher confidence actually predict a more accurate final answer —
     both using the FINAL round's confidence, and using ROUND 1's confidence
     alone (a cheaper, earlier signal that wouldn't require running the full
     loop if it turns out to be just as predictive)?

Outputs:
  confidence_by_round.png
  final_confidence_distribution.png
  accuracy_vs_confidence.png
  confidence_analysis_summary.json

Usage:
  uv run python experiments/verifier_cot/analyze_confidence.py \\
    --judged_csv outputs/verifier_cot/ckpt_judged.csv \\
    --output_dir outputs/verifier_cot/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_round_confidences(rows: list[dict]) -> dict[int, list[int]]:
    """{round_number: [confidence, confidence, ...]} across all images."""
    by_round: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        try:
            transcript = json.loads(r.get("transcript_json", "[]"))
        except json.JSONDecodeError:
            continue
        for entry in transcript:
            by_round[int(entry["round"])].append(int(entry["verifier_confidence"]))
    return dict(by_round)


def accuracy_by_confidence(rows: list[dict], confidence_key: str) -> dict[int, dict]:
    """{confidence_value: {"n": int, "accuracy": float}} — does this
    confidence value actually predict judged_correct?"""
    by_conf: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        jc = r.get("judged_correct", "")
        if jc == "":
            continue
        conf = int(r[confidence_key])
        by_conf[conf].append(int(jc))
    return {
        conf: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for conf, vals in sorted(by_conf.items())
    }


def round1_confidence(rows: list[dict]) -> list[tuple[dict, int]]:
    """(row, round_1_confidence) pairs, for rows that have a round 1 entry."""
    out = []
    for r in rows:
        try:
            transcript = json.loads(r.get("transcript_json", "[]"))
        except json.JSONDecodeError:
            continue
        if transcript:
            first = min(transcript, key=lambda e: int(e["round"]))
            out.append((r, int(first["verifier_confidence"])))
    return out


def make_charts(rows: list[dict], output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ── 1. Confidence by round ──────────────────────────────────────────────
    by_round = collect_round_confidences(rows)
    rounds = sorted(by_round)
    summary["confidence_by_round"] = {
        str(rnd): {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": min(vals),
            "max": max(vals),
        }
        for rnd, vals in by_round.items()
    }

    fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.3), 5))
    data = [by_round[r] for r in rounds]
    bp = ax.boxplot(data, tick_labels=[f"Round {r}\n(n={len(by_round[r])})" for r in rounds],
                    patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)
    ax.set_ylabel("Verifier confidence (1-20)")
    ax.set_ylim(0, 21)
    ax.set_title("Verifier confidence by round")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "confidence_by_round.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 2. Final confidence distribution ────────────────────────────────────
    final_conf = [int(r["final_verifier_confidence"]) for r in rows if r.get("final_verifier_confidence", "") != ""]
    summary["final_confidence_distribution"] = {
        "n": len(final_conf),
        "mean": float(np.mean(final_conf)) if final_conf else None,
        "std": float(np.std(final_conf)) if final_conf else None,
        "histogram": {str(v): final_conf.count(v) for v in sorted(set(final_conf))},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.5, 21.5, 1)
    ax.hist(final_conf, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Final verifier confidence (1-20)")
    ax.set_ylabel("Number of images")
    ax.set_xticks(range(1, 21))
    ax.set_title(f"Final confidence distribution (n={len(final_conf)})")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "final_confidence_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 3. Accuracy vs confidence — final AND round-1 ───────────────────────
    acc_final = accuracy_by_confidence(rows, "final_verifier_confidence")
    summary["accuracy_vs_final_confidence"] = {
        str(c): v for c, v in acc_final.items()
    }

    r1_pairs = round1_confidence(rows)
    by_r1_conf: dict[int, list[int]] = defaultdict(list)
    for row, conf in r1_pairs:
        jc = row.get("judged_correct", "")
        if jc != "":
            by_r1_conf[conf].append(int(jc))
    acc_round1 = {
        conf: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for conf, vals in sorted(by_r1_conf.items())
    }
    summary["accuracy_vs_round1_confidence"] = {str(c): v for c, v in acc_round1.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, acc, title in [(ax1, acc_final, "Final confidence"), (ax2, acc_round1, "Round-1 confidence")]:
        confs = sorted(acc)
        accs = [acc[c]["accuracy"] * 100 for c in confs]
        ns = [acc[c]["n"] for c in confs]
        bars = ax.bar([str(c) for c in confs], accs, color="#55A868", alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for bar, n in zip(bars, ns):
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                       ha="center", fontsize=8)
        ax.set_xlabel(f"{title} score")
        ax.set_ylabel("Judged accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Accuracy vs {title.lower()}")
        ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "accuracy_vs_confidence.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # Overall calibration check: does confidence correlate with correctness at all?
    all_conf = [int(r["final_verifier_confidence"]) for r in rows if r.get("judged_correct", "") != ""]
    all_correct = [int(r["judged_correct"]) for r in rows if r.get("judged_correct", "") != ""]
    if len(all_conf) > 2 and np.std(all_conf) > 0:
        corr = float(np.corrcoef(all_conf, all_correct)[0, 1])
    else:
        corr = None
    summary["correlation_final_confidence_vs_correct"] = corr
    print(f"\n  Correlation(final_confidence, judged_correct) = {corr}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Verifier-CoT confidence scores")
    parser.add_argument("--judged_csv", required=True,
                        help="Path to ckpt_judged.csv from a run_verifier_cot.py run")
    parser.add_argument("--output_dir", default="outputs/verifier_cot/")
    args = parser.parse_args()

    judged_path = Path(args.judged_csv)
    if not judged_path.exists():
        print(f"ERROR: judged CSV not found: {judged_path}")
        return
    rows = _read_csv(judged_path)
    print(f"Loaded {len(rows)} judged rows from {judged_path}")

    output_dir = Path(args.output_dir)
    summary = make_charts(rows, output_dir)

    summary_path = output_dir / "confidence_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
