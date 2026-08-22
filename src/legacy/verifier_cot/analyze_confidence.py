"""
Score analysis for Verifier-CoT — post-hoc analysis of an already-judged
CSV from run_verifier_cot.py. No model/GPU needed.

Answers four questions:
  1. How does the verifier's fine-grained score vary across rounds (does it
     start low and climb as the VLM's statement improves, or is it flat)?
  2. What does the final score distribution look like (is the verifier
     actually discriminating, or clustering at one end regardless of context —
     same question this session's verifier_ppt experiments asked, now on the
     multi-round loop)?
  3. Does a higher score actually predict a more accurate final answer — both
     using the FINAL round's score, and using ROUND 1's score alone (a
     cheaper, earlier signal that wouldn't require running the full loop if
     it turns out to be just as predictive)?
  4. How many rounds does it actually take to reach a final answer — is the
     loop mostly ending early (score cleared --score_threshold) or mostly
     grinding to --max_rounds (forced stop)? Directly diagnostic of whether
     the threshold is calibrated reasonably for this verifier.

Outputs:
  score_by_round.png
  final_score_distribution.png
  accuracy_vs_score.png
  n_rounds_distribution.png
  confidence_analysis_summary.json (filename kept for continuity with older
    runs' file layout; content is about the fine-grained score now)

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


def _as_float(v) -> float | None:
    """Handles both native floats/None (fresh in-memory rows) and strings/""
    (rows reloaded from CSV, where every cell is a string and empty cells are
    "" rather than None) — same gotcha as run_verifier_ppt.py's helper of the
    same name."""
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def collect_round_scores(rows: list[dict]) -> dict[int, list[float]]:
    """{round_number: [score, score, ...]} across all images, skipping any
    round entry where extraction failed (verifier_fine_grained_score is
    None/missing) rather than crashing or silently coercing it to 0."""
    by_round: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        try:
            transcript = json.loads(r.get("transcript_json", "[]"))
        except json.JSONDecodeError:
            continue
        for entry in transcript:
            score = _as_float(entry.get("verifier_fine_grained_score"))
            if score is not None:
                by_round[int(entry["round"])].append(score)
    return dict(by_round)


def accuracy_by_score_bucket(rows: list[dict], score_key: str) -> dict[float, dict]:
    """{score_bucket: {"n": int, "accuracy": float}} — does this score
    bucket (rounded to 1 decimal, matching run_verifier_ppt.py's convention)
    actually predict judged_correct?"""
    by_bucket: dict[float, list[int]] = defaultdict(list)
    for r in rows:
        jc = r.get("judged_correct", "")
        score = _as_float(r.get(score_key))
        if jc == "" or score is None:
            continue
        bucket = round(score, 1)
        by_bucket[bucket].append(int(jc))
    return {
        b: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for b, vals in sorted(by_bucket.items())
    }


def round1_score(rows: list[dict]) -> list[tuple[dict, float]]:
    """(row, round_1_fine_grained_score) pairs, for rows with a round 1 entry
    and a successfully-extracted score."""
    out = []
    for r in rows:
        try:
            transcript = json.loads(r.get("transcript_json", "[]"))
        except json.JSONDecodeError:
            continue
        if not transcript:
            continue
        first = min(transcript, key=lambda e: int(e["round"]))
        score = _as_float(first.get("verifier_fine_grained_score"))
        if score is not None:
            out.append((r, score))
    return out


def collect_n_rounds(rows: list[dict]) -> list[tuple[int, str]]:
    """(n_rounds, final_verifier_verdict) pairs for every row with a valid
    n_rounds — verdict is "SUPPORTED" (ended early, score cleared threshold)
    or "UNSUPPORTED_MAXROUNDS" (forced stop at max_rounds), so the two can be
    told apart in the distribution rather than just plotting raw counts."""
    out = []
    for r in rows:
        n = r.get("n_rounds", "")
        if n is None or n == "":
            continue
        out.append((int(n), r.get("final_verifier_verdict", "")))
    return out


def make_charts(rows: list[dict], output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ── 1. Score by round ────────────────────────────────────────────────────
    by_round = collect_round_scores(rows)
    rounds = sorted(by_round)
    summary["score_by_round"] = {
        str(rnd): {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": min(vals),
            "max": max(vals),
        }
        for rnd, vals in by_round.items()
    }

    if rounds:
        fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.3), 5))
        data = [by_round[r] for r in rounds]
        bp = ax.boxplot(data, tick_labels=[f"Round {r}\n(n={len(by_round[r])})" for r in rounds],
                        patch_artist=True, widths=0.5)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        ax.set_ylabel("Verifier fine-grained score [0,1]")
        ax.set_ylim(0, 1.05)
        ax.set_title("Verifier fine-grained score by round")
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "score_by_round.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print("  [skip] score_by_round.png — no round entries with a valid score")

    # ── 2. Final score distribution ──────────────────────────────────────────
    final_scores = [x for x in (_as_float(r.get("final_verifier_fine_grained_score")) for r in rows)
                    if x is not None]
    summary["final_score_distribution"] = {
        "n": len(final_scores),
        "mean": float(np.mean(final_scores)) if final_scores else None,
        "std": float(np.std(final_scores)) if final_scores else None,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [i / 20 for i in range(21)]
    ax.hist(final_scores, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Final verifier fine-grained score [0,1]")
    ax.set_ylabel("Number of images")
    ax.set_xlim(0, 1)
    ax.set_title(f"Final score distribution (n={len(final_scores)})")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "final_score_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 3. Accuracy vs score — final AND round-1 ─────────────────────────────
    acc_final = accuracy_by_score_bucket(rows, "final_verifier_fine_grained_score")
    summary["accuracy_vs_final_score"] = {f"{b:.1f}": v for b, v in acc_final.items()}

    r1_pairs = round1_score(rows)
    by_r1_bucket: dict[float, list[int]] = defaultdict(list)
    for row, score in r1_pairs:
        jc = row.get("judged_correct", "")
        if jc != "":
            by_r1_bucket[round(score, 1)].append(int(jc))
    acc_round1 = {
        b: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for b, vals in sorted(by_r1_bucket.items())
    }
    summary["accuracy_vs_round1_score"] = {f"{b:.1f}": v for b, v in acc_round1.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, acc, title in [(ax1, acc_final, "Final score"), (ax2, acc_round1, "Round-1 score")]:
        buckets = sorted(acc)
        accs = [acc[b]["accuracy"] * 100 for b in buckets]
        ns = [acc[b]["n"] for b in buckets]
        bars = ax.bar([f"{b:.1f}" for b in buckets], accs, color="#55A868", alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for bar, n in zip(bars, ns):
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                       ha="center", fontsize=8)
        ax.set_xlabel(f"{title} bucket")
        ax.set_ylabel("Judged accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Accuracy vs {title.lower()}")
        ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "accuracy_vs_score.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 4. Rounds-to-final-answer distribution ───────────────────────────────
    round_verdict_pairs = collect_n_rounds(rows)
    n_rounds_vals = [n for n, _ in round_verdict_pairs]
    summary["n_rounds_distribution"] = {
        "n": len(n_rounds_vals),
        "mean": float(np.mean(n_rounds_vals)) if n_rounds_vals else None,
        "median": float(np.median(n_rounds_vals)) if n_rounds_vals else None,
        "histogram": {str(v): n_rounds_vals.count(v) for v in sorted(set(n_rounds_vals))},
    }

    if n_rounds_vals:
        round_range = list(range(1, max(n_rounds_vals) + 1))
        early_counts = [sum(1 for n, v in round_verdict_pairs if n == r and v == "SUPPORTED")
                        for r in round_range]
        forced_counts = [sum(1 for n, v in round_verdict_pairs if n == r and v == "UNSUPPORTED_MAXROUNDS")
                         for r in round_range]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(round_range, early_counts, color="#55A868", alpha=0.85, edgecolor="black",
              linewidth=0.5, label="Ended early (score >= threshold)")
        ax.bar(round_range, forced_counts, bottom=early_counts, color="#C44E52", alpha=0.85,
              edgecolor="black", linewidth=0.5, label="Hit max_rounds")
        for r, e, f in zip(round_range, early_counts, forced_counts):
            total = e + f
            if total:
                ax.annotate(str(total), (r, total), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Rounds taken to reach final answer")
        ax.set_ylabel("Number of images")
        ax.set_xticks(round_range)
        ax.set_title(f"Rounds-to-final-answer distribution (n={len(n_rounds_vals)}, "
                    f"mean={np.mean(n_rounds_vals):.2f})")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "n_rounds_distribution.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print("  [skip] n_rounds_distribution.png — no rows with a valid n_rounds")

    # ── 4b. Accuracy vs rounds-to-final-answer ───────────────────────────────
    # Every row in ckpt_judged.csv already reached a final answer (early-stop
    # or forced at max_rounds), so this is accuracy over ALL completed
    # conversations grouped by how many rounds they took — not a subset.
    # Built directly from `rows` in one pass (not zipped against
    # round_verdict_pairs, which collect_n_rounds() already filtered down —
    # zipping two independently-filtered sequences would silently misalign
    # judged_correct with the wrong row wherever either filter dropped one).
    acc_pairs = []
    for r in rows:
        n = r.get("n_rounds", "")
        if n is None or n == "" or r.get("judged_correct", "") == "":
            continue
        acc_pairs.append((int(n), int(r["judged_correct"])))
    if acc_pairs:
        round_range = list(range(1, max(n for n, _ in acc_pairs) + 1))
        accs, ns = [], []
        for r in round_range:
            correct = [c for n, c in acc_pairs if n == r]
            accs.append(float(np.mean(correct)) if correct else None)
            ns.append(len(correct))
        summary["accuracy_by_n_rounds"] = {
            str(r): {"accuracy": a, "n": n} for r, a, n in zip(round_range, accs, ns)
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_r = [r for r, a in zip(round_range, accs) if a is not None]
        plot_a = [a for a in accs if a is not None]
        plot_n = [n for n, a in zip(ns, accs) if a is not None]
        ax.bar(plot_r, plot_a, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
        for r, a, n in zip(plot_r, plot_a, plot_n):
            ax.annotate(f"{a:.2f}\n(n={n})", (r, a), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Rounds taken to reach final answer")
        ax.set_ylabel("Judged accuracy")
        ax.set_xticks(round_range)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Accuracy by rounds-to-final-answer (n={len(acc_pairs)})")
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "accuracy_by_n_rounds.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print("  [skip] accuracy_by_n_rounds.png — no rows with both n_rounds and judged_correct")

    # Overall calibration check: does the score correlate with correctness at all?
    paired = [(x, int(r["judged_correct"])) for r, x in
             ((r, _as_float(r.get("final_verifier_fine_grained_score"))) for r in rows)
             if x is not None and r.get("judged_correct", "") != ""]
    if len(paired) > 2 and np.std([a for a, _ in paired]) > 0:
        corr = float(np.corrcoef([a for a, _ in paired], [b for _, b in paired])[0, 1])
    else:
        corr = None
    summary["correlation_final_score_vs_correct"] = corr
    print(f"\n  Correlation(final_fine_grained_score, judged_correct) = {corr}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Verifier-CoT fine-grained scores")
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
