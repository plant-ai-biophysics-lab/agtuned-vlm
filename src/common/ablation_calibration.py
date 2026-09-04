"""Ablation stage, "calibration" step (Q4) — separates two things the
paper's single aggregate correlation number conflates:

  1. Ranking accuracy: does the verifier pick a correct candidate WHEN ONE
     EXISTS in the pool? (selection quality, within one image)
  2. Calibration: is a given top1_win_rate value comparable in an absolute
     sense ACROSS images? (is the scalar score usable as a confidence gate)

Also runs an explicit score-direction sanity check (is "higher score" really
"more correct", per the vendored llm_verifier scale) before trusting either
of the above, and reproduces the appendix's lost threshold-filter /
selective-risk figures as committed code instead of an ad hoc script.

Consumes verifier_cot.py / detection_verifier.py checkpoint CSVs (needs
`top1_win_rate` + `correct`/`judged_correct`) plus the per-row arm table
produced by src.common.ablation_selection.score_run (needs `verifier` and
`oracle_in_pool` arms to compute ranking accuracy).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _as_float(v) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def load_win_rate_rows(ckpt_csv: Path) -> list[dict]:
    """[{agml_dataset, class_label, image_name, top1_win_rate, correct,
    judged_correct}, ...] — judged_correct falls back to `correct` (direct
    resolved-class match) when the judge phase hasn't been run for this
    checkpoint, so calibration/selective-risk can still be computed."""
    out = []
    with open(ckpt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wr = _as_float(row.get("top1_win_rate"))
            if wr is None:
                continue
            jc = row.get("judged_correct", "")
            correct = int(jc) if jc not in ("", None) else int(row.get("correct", 0) or 0)
            out.append({
                "agml_dataset": row["agml_dataset"], "class_label": row["class_label"],
                "image_name": row["image_name"], "top1_win_rate": wr, "correct": correct,
            })
    return out


def score_direction_check() -> dict:
    """Confirms the letter-score / win-rate direction against the vendored
    library's own scale definition, rather than leaving it as an inferred
    claim. A = best (highest raw value, normalizes to 1.0), T = worst
    (normalizes to 0.0) — so "higher top1_win_rate / higher normalized
    letter score" should mean "more likely correct" throughout this
    codebase. If this ever comes back False, ppt_scoring.py's mapping (or
    its use in verifier_cot.py) has a sign bug and every downstream
    calibration number here is meaningless until fixed."""
    from llm_verifier.fine_grained_reward import SCALE

    valid_tokens = SCALE["valid_tokens"]  # {"A": 20.0, ..., "T": 1.0, "a": 20.0, ...}
    a_raw, t_raw = valid_tokens["A"], valid_tokens["T"]
    min_val, max_val = min(valid_tokens.values()), max(valid_tokens.values())
    a_norm = (a_raw - min_val) / (max_val - min_val)
    t_norm = (t_raw - min_val) / (max_val - min_val)

    return {
        "scale_description_head": SCALE.get("scale_description", "")[:80],
        "a_raw": a_raw, "t_raw": t_raw,
        "a_normalized": a_norm, "t_normalized": t_norm,
        "direction_ok": bool(a_norm > t_norm),
        "interpretation": (
            "A normalizes higher than T, confirming higher top1_win_rate / "
            "letter score means MORE correct, not less — no sign flip in "
            "the scoring pipeline."
        ),
    }


def ranking_accuracy(arm_rows: list[dict]) -> dict:
    """P(verifier's top1 pick is correct | at least one correct candidate
    exists in the pool) — the within-image selection-quality number,
    computed from src.common.ablation_selection.score_run's per-row arm
    table (needs the "verifier" and "oracle_in_pool" arms for the SAME
    row key). Distinct from raw judged accuracy: a pool with no correct
    candidate at all can never be "selected correctly" regardless of
    verifier quality, so those rows are excluded from the denominator."""
    by_key: dict[tuple, dict] = defaultdict(dict)
    for r in arm_rows:
        key = (r["model"], r["condition"], r["agml_dataset"], r["class_label"], r["image_name"])
        by_key[key][r["arm"]] = r["correct"]

    has_correct_in_pool = [v for v in by_key.values() if "oracle_in_pool" in v and v["oracle_in_pool"]]
    n_with_correct = len(has_correct_in_pool)
    n_selected_correct = sum(1 for v in has_correct_in_pool if v.get("verifier", 0))

    return {
        "n_pools_with_a_correct_candidate": n_with_correct,
        "n_pools_total": len(by_key),
        "ranking_accuracy": (n_selected_correct / n_with_correct) if n_with_correct else None,
        "oracle_in_pool_rate": (n_with_correct / len(by_key)) if by_key else None,
    }


def calibration_curve(rows: list[dict], n_buckets: int = 10) -> dict[str, dict]:
    """{bucket_label: {n, accuracy, mean_win_rate}} — top1_win_rate binned
    into n_buckets equal-width buckets over [0, 1]. A well-calibrated score
    should show accuracy climbing monotonically with the bucket."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        idx = min(int(r["top1_win_rate"] * n_buckets), n_buckets - 1)
        buckets[idx].append(r)
    out = {}
    for idx in sorted(buckets):
        grp = buckets[idx]
        lo, hi = idx / n_buckets, (idx + 1) / n_buckets
        out[f"{lo:.1f}-{hi:.1f}"] = {
            "n": len(grp),
            "accuracy": sum(r["correct"] for r in grp) / len(grp),
            "mean_win_rate": sum(r["top1_win_rate"] for r in grp) / len(grp),
        }
    return out


def selective_risk_curve(rows: list[dict], n_thresholds: int = 20) -> list[dict]:
    """[{threshold, coverage, accuracy_at_or_above_threshold}, ...] — sweeps
    top1_win_rate thresholds from 0 to 1, keeping only rows at or above each
    threshold. A genuine confidence signal should show accuracy rising as
    coverage falls; a flat or falling line is exactly the failure this
    ablation step exists to quantify precisely (rather than a single
    correlation number)."""
    n = len(rows)
    out = []
    for i in range(n_thresholds + 1):
        t = i / n_thresholds
        kept = [r for r in rows if r["top1_win_rate"] >= t]
        out.append({
            "threshold": t,
            "coverage": len(kept) / n if n else 0.0,
            "n_kept": len(kept),
            "accuracy": (sum(r["correct"] for r in kept) / len(kept)) if kept else None,
        })
    return out


def plot_calibration(win_rate_rows: list[dict], output_dir: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    cal = calibration_curve(win_rate_rows)
    risk = selective_risk_curve(win_rate_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    buckets = list(cal)
    accs = [cal[b]["accuracy"] * 100 for b in buckets]
    ns = [cal[b]["n"] for b in buckets]
    bars = ax.bar(buckets, accs, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, ns):
        ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                    ha="center", fontsize=7, rotation=90 if len(buckets) > 8 else 0)
    ax.set_xlabel("top1_win_rate bucket")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Calibration curve — {label}")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "calibration_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    thresholds = [r["threshold"] for r in risk]
    coverage = [r["coverage"] * 100 for r in risk]
    accuracy = [r["accuracy"] * 100 if r["accuracy"] is not None else None for r in risk]
    ax1.plot(thresholds, coverage, marker="o", color="#55A868", label="Coverage (%)")
    ax1.set_xlabel("top1_win_rate threshold (keep >= threshold)")
    ax1.set_ylabel("Coverage (%)", color="#55A868")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, accuracy, marker="s", color="#C44E52", label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)", color="#C44E52")
    ax1.set_title(f"Selective risk / coverage — {label}")
    ax1.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "selective_risk_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")
