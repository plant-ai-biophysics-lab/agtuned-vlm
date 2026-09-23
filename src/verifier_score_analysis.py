"""
Stage 4-diagnostic — investigate the verifier's own score distribution,
independent of any downstream judge change, per the paper's Appendix C
findings (arXiv preprint "Vision-language models know more about
agriculture..."): the fine-grained-reward letter scale (A-T, via
llm_verifier.fine_grained_reward.extract_score, src/common/ppt_scoring.py)
(1) collapses onto the top 3 letters (80-92% of mass, Figure 9) instead of
spanning the scale, and (2) its retained-set accuracy DECLINES as the
win-rate threshold rises instead of rising (Figure 8) — i.e. the score is
both under-varied and, in the sense that matters for filtering, inverted.

This script re-derives both charts directly from a real verifier_cot run's
own output — no model calls, no GPU, nothing beyond files already on disk
— so the failure mode can be inspected BEFORE picking a fix (a rescoring
prompt change, a different scale, temperature, etc. — see docs/
stage1_judge_improvements.md's judge-improvements plan, which this
script's findings are meant to inform, not preempt).

Two independent signals, both worth checking (they can disagree — see
"why two charts" below):

  1. Image-level: ckpt_<model>.csv's own `top1_win_rate` (the tournament's
     final Bradley-Terry win-rate for the selected candidate) vs. `correct`
     — exactly what the paper's Figure 8 plots. Needs only the checkpoint
     CSV, always available.
  2. Pair-level (needs verifier_log.jsonl, richer diagnosis): every raw
     `score_a`/`score_b` fine-grained-reward value from every scored
     "tournament_pair" entry, matched back to whether THAT specific
     candidate's own extracted answer was locally correct (via the
     "candidate" log entries) — this is the verifier's raw per-comparison
     signal before Bradley-Terry aggregation smooths it into a win_rate,
     so it's the more direct place to see whether the scoring call itself
     is discriminating or just noise.

Why two charts, not one: if (1) is flat/inverted but (2) is not, the
scoring call itself is fine and the aggregation (tournament structure, K,
pivots) is where the signal gets lost. If both are flat/inverted, the
scoring call itself (prompt, scale, or how it's read off logits) is the
place to fix — which is exactly the diagnostic question the user needs
answered before deciding where to spend the next round of changes.

Outputs (under <verifier_dir>/score_analysis/):
  image_level_threshold.png   — Fig 8 analog: top1_win_rate histogram +
                                 accuracy-of-retained-set vs. threshold.
  pair_level_threshold.png    — same shape, but on raw per-comparison
                                 scores (needs verifier_log.jsonl).
  letter_distribution.png     — Fig 9 analog: histogram of the nearest
                                 A-T letter for every scored score_a/
                                 score_b value.
  score_analysis_summary.json — point-biserial correlations, mean score by
                                 correct/incorrect, % mass in top-3
                                 letters, per-task-type breakdown.

Usage:
    uv run python main.py verifier-score-analysis \\
        --model google/gemma-4-26B-A4B-it --output_dir outputs_final/verifier_cot/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict

from src.common.io import load_jsonl
from src.common.metrics import contains_match
from src.common.vision_eval import safe_model_name, load_ckpt
from src.verifier_cot import VERIFIER_CKPT_FIELDS

# A=20 ... T=1 (llm_verifier.fine_grained_reward.SCALE["valid_tokens"]).
# extract_score normalizes (expected_value - 1) / (20 - 1) to [0, 1], so
# this is the exact inverse used to recover the nearest letter for display.
_LETTER_MIN, _LETTER_MAX = 1, 20


def _score_to_letter(score: float) -> str:
    value = score * (_LETTER_MAX - _LETTER_MIN) + _LETTER_MIN
    idx = round(_LETTER_MAX - value)  # A(=20) -> 0, T(=1) -> 19
    idx = max(0, min(19, idx))
    return chr(65 + idx)


def _point_biserial(scores: list[float], binary: list[int]) -> float | None:
    """Correlation between a continuous score and a 0/1 outcome. Returns
    None if either side has zero variance (undefined) or too few points."""
    if len(scores) < 2:
        return None
    try:
        from scipy.stats import pointbiserialr
        r, _p = pointbiserialr(binary, scores)
        return None if r != r else float(r)  # NaN check without numpy import
    except ImportError:
        n = len(scores)
        mean_s = sum(scores) / n
        mean_b = sum(binary) / n
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(scores, binary)) / n
        var_s = sum((s - mean_s) ** 2 for s in scores) / n
        var_b = sum((b - mean_b) ** 2 for b in binary) / n
        if var_s == 0 or var_b == 0:
            return None
        return cov / (var_s ** 0.5 * var_b ** 0.5)


def _threshold_curve(scores: list[float], correct: list[int], n_thresholds: int = 25) -> tuple[list[float], list[float | None], list[int]]:
    """Accuracy of the RETAINED set (score >= t) as t sweeps the observed
    range — the exact quantity Figure 8's bottom panel plots. Returns
    (thresholds, accuracy_or_None, n_retained); accuracy is None where
    nothing clears the threshold."""
    if not scores:
        return [], [], []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [lo], [sum(correct) / len(correct)], [len(correct)]
    thresholds = [lo + (hi - lo) * i / (n_thresholds - 1) for i in range(n_thresholds)]
    accs, ns = [], []
    for t in thresholds:
        kept = [c for s, c in zip(scores, correct) if s >= t]
        ns.append(len(kept))
        accs.append(sum(kept) / len(kept) if kept else None)
    return thresholds, accs, ns


def _plot_threshold_chart(
    scores: list[float], correct: list[int], out_path: Path, title: str, xlabel: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds, accs, ns = _threshold_curve(scores, correct)
    fig, (ax_hist, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_hist.hist(scores, bins=30, color="#4C72B0", edgecolor="white")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{title} — score distribution")

    valid = [(t, a) for t, a in zip(thresholds, accs) if a is not None]
    if valid:
        tx, ty = zip(*valid)
        ax_acc.plot(tx, ty, color="#DD8452", marker="o", markersize=3)
    overall_acc = sum(correct) / len(correct) if correct else 0.0
    ax_acc.axhline(overall_acc, color="gray", linestyle="--", linewidth=1,
                    label=f"unfiltered accuracy ({overall_acc:.2f})")
    ax_acc.set_ylabel("Accuracy of retained set")
    ax_acc.set_xlabel(xlabel)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.legend(loc="best", fontsize=8)
    ax_acc.set_title("Retained-set accuracy vs. threshold (rising = useful confidence signal)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _granularity_diagnostics(
    log_entries: list[dict], candidate_correct: dict[tuple[str, str, str, int], int],
) -> dict | None:
    """Paper 2 Stage 2 (docs/legacy/stage2_verifier_corrections_paused.md) — tests the
    three concrete hypotheses raised for why the letter-scale score is
    under-varied/inverted, using the extra fields verifier_cot.py now logs
    per tournament_pair (letter_probs_*_max/sum, trajectory_*_words,
    n_confidence_markers). Returns None if a run predates this logging
    (old verifier_log.jsonl files silently have none of these keys — this
    is additive, not a schema requirement)."""
    mass_gaps: list[float] = []
    length_score_pairs: list[tuple[int, float]] = []
    confidence_counts: list[int] = []
    winner_correct: list[int] = []

    for entry in log_entries:
        if entry.get("call_type") != "tournament_pair":
            continue
        ds, cls, img = entry["agml_dataset"], entry["class_label"], entry["image_name"]
        a_idx, b_idx = entry["pool_pair"]

        for side, idx, words_key in (("a", a_idx, "trajectory_a_words"), ("b", b_idx, "trajectory_b_words")):
            probs_max = entry.get(f"letter_probs_{side}_max")
            probs_sum = entry.get(f"letter_probs_{side}_sum")
            if probs_max is not None and probs_sum is not None:
                mass_gaps.append(sum(probs_sum.values()) - sum(probs_max.values()))
            words = entry.get(words_key)
            score = entry.get(f"score_{side}")
            if words is not None and score is not None:
                length_score_pairs.append((words, float(score)))

        n_markers = entry.get("n_confidence_markers")
        if n_markers is None:
            continue
        key_a = (ds, cls, img, a_idx)
        key_b = (ds, cls, img, b_idx)
        if key_a not in candidate_correct or key_b not in candidate_correct:
            continue
        winner_key = key_a if entry["score_a"] >= entry["score_b"] else key_b
        confidence_counts.append(n_markers)
        winner_correct.append(candidate_correct[winner_key])

    if not mass_gaps and not length_score_pairs and not confidence_counts:
        return None

    out: dict = {}
    if mass_gaps:
        out["max_vs_sum_mass_gap"] = {
            "n": len(mass_gaps), "mean": sum(mass_gaps) / len(mass_gaps),
            "note": "mean probability mass DISCARDED per score by extract_score's max-of-spellings "
                    "convention vs. summing them (Step 2) — near 0 means this isn't a meaningful "
                    "contributor to the low variance; consistently large means it plausibly is.",
        }
    if length_score_pairs:
        words, scores = zip(*length_score_pairs)
        out["length_vs_score"] = {
            "n": len(length_score_pairs),
            "pearson_r": _point_biserial(list(words), list(scores)),
            "note": "correlation between a trajectory's word count and its own letter-scale score — "
                    "positive means the verifier rewards longer responses regardless of correctness.",
        }
    if confidence_counts:
        out["confidence_language_vs_winner_correct"] = {
            "n": len(confidence_counts),
            "point_biserial_r": _point_biserial([float(c) for c in confidence_counts], winner_correct),
            "mean_markers_when_winner_correct": (
                sum(c for c, w in zip(confidence_counts, winner_correct) if w) / max(sum(winner_correct), 1)),
            "mean_markers_when_winner_incorrect": (
                sum(c for c, w in zip(confidence_counts, winner_correct) if not w)
                / max(len(winner_correct) - sum(winner_correct), 1)),
            "note": "negative r / higher mean when incorrect replicates the CoT-verifier finding "
                    "(confident-sounding language anti-correlated with correctness) in THIS verifier's "
                    "own comparative-analysis text.",
        }
    return out


def _plot_letter_distribution(scores: list[float], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    letters = [chr(65 + i) for i in range(20)]
    counts = {L: 0 for L in letters}
    for s in scores:
        counts[_score_to_letter(s)] += 1
    total = sum(counts.values()) or 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(letters, [counts[L] for L in letters], color="#55A868")
    ax.set_ylabel("Count")
    ax.set_xlabel("Nearest letter (A = best, T = worst)")
    top3 = sum(counts[L] for L in letters[:3])
    ax.set_title(f"Letter-scale distribution (n={total}) — {top3 / total:.0%} in top 3 (A-C)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze(verifier_dir: Path) -> dict:
    ckpt_path = verifier_dir / f"ckpt_{verifier_dir.name}.csv"
    rows, _ = load_ckpt(ckpt_path, fields=VERIFIER_CKPT_FIELDS)
    if not rows:
        raise SystemExit(f"No checkpoint rows found at {ckpt_path}")

    out_dir = verifier_dir / "score_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"verifier_dir": str(verifier_dir), "n_rows": len(rows)}

    # ── 1. Image-level: top1_win_rate vs. correctness (Figure 8 analog, no log needed).
    # Two signals, both worth reporting: `judged_correct` is what the paper's own
    # Figure 8 actually plots (LLM-judge verdict on the free-text prediction) and
    # is the PRIMARY chart here — the deterministic `correct` column (exact/
    # substring resolve_class match) is kept as a secondary reference chart since
    # it's known to undercount free-text correctness (that gap is the whole reason
    # the judge exists, per Stage 1) and the two can genuinely disagree.
    def _win_rate_correctness(key: str, truthy_filter) -> tuple[list[float], list[int]]:
        wr, co = [], []
        for r in rows:
            if r.get("top1_win_rate") in (None, "") or not truthy_filter(r.get(key)):
                continue
            wr.append(float(r["top1_win_rate"]))
            co.append(int(r[key]))
        return wr, co

    win_rates, corrects = _win_rate_correctness("judged_correct", lambda v: v not in (None, ""))
    n_unjudged = sum(1 for r in rows if r.get("judged_correct") in (None, "") and r.get("top1_win_rate") not in (None, ""))
    if win_rates:
        _plot_threshold_chart(
            win_rates, corrects, out_dir / "image_level_threshold.png",
            title="Image-level (top1_win_rate vs. JUDGED correctness)", xlabel="top1_win_rate threshold",
        )
        r_image = _point_biserial(win_rates, corrects)
        summary["image_level"] = {
            "n": len(win_rates),
            "n_unjudged_excluded": n_unjudged,
            "point_biserial_r": r_image,
            "mean_win_rate_correct": (sum(w for w, c in zip(win_rates, corrects) if c) / max(sum(corrects), 1)),
            "mean_win_rate_incorrect": (sum(w for w, c in zip(win_rates, corrects) if not c)
                                         / max(len(corrects) - sum(corrects), 1)),
        }
    else:
        print(f"  [note] no rows have judged_correct populated (--skip_judge run?) — "
              f"falling back to deterministic `correct` as the primary image-level signal.")
        win_rates, corrects = _win_rate_correctness("correct", lambda v: v not in (None, ""))
        _plot_threshold_chart(
            win_rates, corrects, out_dir / "image_level_threshold.png",
            title="Image-level (top1_win_rate vs. deterministic correctness — no judge results available)",
            xlabel="top1_win_rate threshold",
        )
        summary["image_level"] = {
            "n": len(win_rates), "n_unjudged_excluded": n_unjudged,
            "point_biserial_r": _point_biserial(win_rates, corrects),
            "mean_win_rate_correct": (sum(w for w, c in zip(win_rates, corrects) if c) / max(sum(corrects), 1)),
            "mean_win_rate_incorrect": (sum(w for w, c in zip(win_rates, corrects) if not c)
                                         / max(len(corrects) - sum(corrects), 1)),
        }

    # Secondary reference chart — deterministic correctness, always produced when
    # available, so the two signals can be compared directly instead of guessed at.
    det_win_rates, det_corrects = _win_rate_correctness("correct", lambda v: v not in (None, ""))
    if det_win_rates:
        _plot_threshold_chart(
            det_win_rates, det_corrects, out_dir / "image_level_threshold_deterministic.png",
            title="Image-level (top1_win_rate vs. deterministic correctness — reference only)",
            xlabel="top1_win_rate threshold",
        )
        summary["image_level_deterministic"] = {
            "n": len(det_win_rates),
            "point_biserial_r": _point_biserial(det_win_rates, det_corrects),
            "mean_win_rate_correct": (sum(w for w, c in zip(det_win_rates, det_corrects) if c)
                                       / max(sum(det_corrects), 1)),
            "mean_win_rate_incorrect": (sum(w for w, c in zip(det_win_rates, det_corrects) if not c)
                                         / max(len(det_corrects) - sum(det_corrects), 1)),
        }

    # ── 2. Pair-level: raw score_a/score_b vs. local candidate correctness (needs log) ──
    log_path = verifier_dir / "verifier_log.jsonl"
    if not log_path.exists():
        print(f"  [note] {log_path} not found — skipping pair-level chart/letter distribution "
              f"(image-level chart above still reflects the aggregated win_rate signal).")
        return summary

    task_type_by_group: dict[tuple[str, str], str] = {}
    for r in rows:
        task_type_by_group[(r["agml_dataset"], r["class_label"])] = r.get("task_type", "")

    log_entries = load_jsonl(log_path)
    candidate_correct: dict[tuple[str, str, str, int], int] = {}
    for entry in log_entries:
        if entry.get("call_type") != "candidate":
            continue
        key = (entry["agml_dataset"], entry["class_label"], entry["image_name"], entry["candidate_index"])
        gt = entry["class_label"]
        candidate_correct[key] = int(contains_match(entry.get("extracted_answer", ""), gt))

    pair_scores: list[float] = []
    pair_correct: list[int] = []
    pair_task_type: list[str] = []
    n_pairs_matched = 0
    n_pairs_unmatched = 0
    for entry in log_entries:
        if entry.get("call_type") != "tournament_pair":
            continue
        ds, cls, img = entry["agml_dataset"], entry["class_label"], entry["image_name"]
        a_idx, b_idx = entry["pool_pair"]
        tt = task_type_by_group.get((ds, cls), "")
        for idx, score_key in ((a_idx, "score_a"), (b_idx, "score_b")):
            key = (ds, cls, img, idx)
            if key not in candidate_correct:
                n_pairs_unmatched += 1
                continue
            pair_scores.append(float(entry[score_key]))
            pair_correct.append(candidate_correct[key])
            pair_task_type.append(tt)
            n_pairs_matched += 1

    if not pair_scores:
        print(f"  [note] {log_path} has no tournament_pair/candidate entries to join — "
              f"skipping pair-level chart/letter distribution.")
        return summary

    _plot_threshold_chart(
        pair_scores, pair_correct, out_dir / "pair_level_threshold.png",
        title="Pair-level (raw score_a/score_b)", xlabel="fine-grained reward score threshold",
    )
    _plot_letter_distribution(pair_scores, out_dir / "letter_distribution.png")

    r_pair = _point_biserial(pair_scores, pair_correct)
    letters = [chr(65 + i) for i in range(20)]
    letter_counts = {L: 0 for L in letters}
    for s in pair_scores:
        letter_counts[_score_to_letter(s)] += 1
    top3_frac = sum(letter_counts[L] for L in letters[:3]) / len(pair_scores)

    by_task: dict[str, dict] = {}
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_correct: dict[str, list[int]] = defaultdict(list)
    for s, c, tt in zip(pair_scores, pair_correct, pair_task_type):
        grouped_scores[tt].append(s)
        grouped_correct[tt].append(c)
    for tt in sorted(grouped_scores):
        by_task[tt] = {
            "n": len(grouped_scores[tt]),
            "point_biserial_r": _point_biserial(grouped_scores[tt], grouped_correct[tt]),
            "mean_score_correct": (sum(s for s, c in zip(grouped_scores[tt], grouped_correct[tt]) if c)
                                    / max(sum(grouped_correct[tt]), 1)),
            "mean_score_incorrect": (sum(s for s, c in zip(grouped_scores[tt], grouped_correct[tt]) if not c)
                                      / max(len(grouped_correct[tt]) - sum(grouped_correct[tt]), 1)),
        }

    summary["pair_level"] = {
        "n_matched": n_pairs_matched,
        "n_unmatched": n_pairs_unmatched,
        "point_biserial_r": r_pair,
        "mean_score_correct": (sum(s for s, c in zip(pair_scores, pair_correct) if c)
                                / max(sum(pair_correct), 1)),
        "mean_score_incorrect": (sum(s for s, c in zip(pair_scores, pair_correct) if not c)
                                  / max(len(pair_correct) - sum(pair_correct), 1)),
        "top3_letter_fraction": top3_frac,
        "letter_counts": letter_counts,
        "by_task_type": by_task,
    }
    granularity = _granularity_diagnostics(log_entries, candidate_correct)
    if granularity:
        summary["pair_level"]["granularity"] = granularity
    return summary


def main(args: argparse.Namespace) -> None:
    verifier_dir = Path(args.verifier_dir) if args.verifier_dir else (
        Path(args.output_dir) / safe_model_name(args.model)
    )
    if not verifier_dir.exists():
        raise SystemExit(f"{verifier_dir} does not exist — pass --model/--output_dir or --verifier_dir")

    summary = analyze(verifier_dir)
    out_path = verifier_dir / "score_analysis" / "score_analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    il = summary.get("image_level", {})
    print(f"  Image-level (top1_win_rate vs. JUDGED correct, n={il.get('n')}, "
          f"{il.get('n_unjudged_excluded', 0)} unjudged row(s) excluded): "
          f"point-biserial r={il.get('point_biserial_r')}, "
          f"mean win_rate correct={il.get('mean_win_rate_correct'):.3f} "
          f"vs. incorrect={il.get('mean_win_rate_incorrect'):.3f}")
    ild = summary.get("image_level_deterministic")
    if ild:
        print(f"  Image-level (top1_win_rate vs. deterministic correct, reference only, n={ild.get('n')}): "
              f"point-biserial r={ild.get('point_biserial_r')}, "
              f"mean win_rate correct={ild.get('mean_win_rate_correct'):.3f} "
              f"vs. incorrect={ild.get('mean_win_rate_incorrect'):.3f}")
    pl = summary.get("pair_level")
    if pl:
        print(f"  Pair-level (raw score vs. local correctness, n={pl['n_matched']}, "
              f"{pl['n_unmatched']} unmatched): point-biserial r={pl['point_biserial_r']}, "
              f"mean score correct={pl['mean_score_correct']:.3f} vs. "
              f"incorrect={pl['mean_score_incorrect']:.3f}, "
              f"{pl['top3_letter_fraction']:.0%} of scores land in top-3 letters (A-C)")
        gr = pl.get("granularity")
        if gr:
            mg = gr.get("max_vs_sum_mass_gap")
            if mg:
                print(f"    [granularity] max-vs-sum mass gap: mean={mg['mean']:.4f} over {mg['n']} scores")
            lv = gr.get("length_vs_score")
            if lv:
                print(f"    [granularity] length vs. score correlation: r={lv['pearson_r']} over {lv['n']} scores")
            cl = gr.get("confidence_language_vs_winner_correct")
            if cl:
                print(f"    [granularity] confidence-language vs. winner-correct: r={cl['point_biserial_r']}, "
                      f"mean markers correct={cl['mean_markers_when_winner_correct']:.2f} vs. "
                      f"incorrect={cl['mean_markers_when_winner_incorrect']:.2f}")
    print(f"  Wrote charts + {out_path.name} to {out_path.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Model whose verifier-cot run to analyze.")
    parser.add_argument("--output_dir", default="outputs_final/verifier_cot/",
                         help="Base dir verifier_cot.py wrote to (used with --model).")
    parser.add_argument("--verifier_dir", default=None,
                         help="Exact <output_dir>/<safe_model_name>/ folder to analyze — overrides "
                              "--model/--output_dir when given.")
    main(parser.parse_args())
