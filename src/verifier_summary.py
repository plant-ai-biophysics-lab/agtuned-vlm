"""
Stage 4-report — one-glance verifier performance reference (no GPU, no model).

Answers "how did this verifier run actually do vs. the bounds and the CoT
methods?" without touching outputs_final/cot_compare/ — that directory holds
the full-corpus, paper-facing charts, and running `cot-compare
--filter_datasets` to answer this question would overwrite them with a
subsample-scoped version (a real mistake made once during development).
Everything this writes lands in the verifier run's OWN --verifier_dir.

The comparison is strictly apples-to-apples: every condition is filtered to
the EXACT (agml_dataset, class_label, image_name) triples the verifier run
actually scored, not merely the same datasets. That matters because
verifier-cot defaults to --n_samples 10 while vision_baseline/desc-aug were
run at 50/class — comparing dataset-level would silently put 160 verifier
images against 800 baseline images.

Conditions plotted (any missing one is skipped with a note rather than
failing — e.g. cot_yaml currently exists only for gemma-4-26B-A4B-it, so an
E4B-it run legitimately has no Manual-CoT bar):

  Lower Bound      vision_baseline        — single-shot, no help
  Self CoT         self_cot               — self-generated description, then classify
  Manual CoT       cot_yaml               — fixed human-authored step chain
  Verifier-Belief  <verifier_dir>         — this run
  Upper Bound      desc_augmented_vision  — oracle class description supplied in-prompt

judge_f1_macro is the metric (not strict f1_macro): free-text verifier
answers frequently fail resolve_class's exact/substring match against the
ground-truth vocabulary even when correct — the LLM judge is the only
condition-comparable scorer here. 95% CIs come from the same bootstrap
machinery cot_compare uses, so error bars mean the same thing as in the
paper-facing charts.

Usage:
    python main.py verifier-summary \\
        --verifier_dir outputs_final/verifier_cot_fasttest/ \\
        --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.vision_eval import load_all_rows  # noqa: E402
from src.cot_compare import compute_condition_stats, load_verifier_cot_rows  # noqa: E402

# (display label, output_dir, bar color). Order = left-to-right plot order,
# deliberately ascending in "how much help the condition gets" so the chart
# reads as a ladder from unaided to oracle-assisted.
_CONDITIONS = [
    ("Lower Bound",     "outputs_final/vision_baseline/",        "#1f77b4"),
    ("Self CoT",        "outputs_final/self_cot/",               "#2ca02c"),
    ("Manual CoT",      "outputs_final/cot_yaml/",               "#9467bd"),
    ("Verifier-Belief", None,                                    "#e06b00"),  # None => --verifier_dir
    ("Upper Bound",     "outputs_final/desc_augmented_vision/",  "#d62728"),
]


def verifier_image_keys(verifier_dir: Path, model_safe: str, baseline: str) -> set[tuple[str, str, str]]:
    """The exact (dataset, class, image) triples this verifier run scored —
    read from its final-round rows, which are one-per-image."""
    ckpt = verifier_dir / model_safe / f"ckpt_{baseline}_{model_safe}.csv"
    if not ckpt.exists():
        raise SystemExit(f"No verifier checkpoint at {ckpt}")
    keys = set()
    with open(ckpt, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_final_round")) == "1":
                keys.add((r["agml_dataset"], r["class_label"], r["image_name"]))
    return keys


def collect(verifier_dir: Path, model_safe: str, baseline: str, n_boot: int, seed: int):
    """[(label, color, stats_or_None, n_rows)] for every condition, filtered
    to the verifier run's own image set."""
    keys = verifier_image_keys(verifier_dir, model_safe, baseline)
    print(f"  Verifier run scored {len(keys)} image(s) — filtering every condition to exactly those")

    def _filt(rows):
        return [r for r in rows
                if (r.get("agml_dataset"), r.get("class_label"), r.get("image_name")) in keys]

    out = []
    for label, dirname, color in _CONDITIONS:
        if dirname is None:
            rows_map = load_verifier_cot_rows(verifier_dir, baseline)
        else:
            d = Path(dirname)
            rows_map = load_all_rows(d) if d.exists() else {}
        rows = _filt(rows_map.get(model_safe, []))
        if not rows:
            print(f"  [skip] {label}: no rows for {model_safe} on these images")
            out.append((label, color, None, 0))
            continue
        stats = compute_condition_stats(rows, n_boot, seed).get("overall", {})
        # A condition whose overlapping rows were never JUDGED reports
        # judge_f1_macro=0.0 — which is missing data, not a score of zero.
        # Plotting that as a real bar would be actively misleading (E4B-it's
        # self_cot rows are exactly this case: present but unjudged), so
        # treat it the same as "no rows at all".
        if not stats.get("judge_n_classes"):
            print(f"  [skip] {label}: {len(rows)} row(s) overlap but none are judged "
                  f"(judge_f1_macro would be a meaningless 0.0)")
            out.append((label, color, None, len(rows)))
            continue
        out.append((label, color, stats, len(rows)))
        print(f"  {label:16s} judge_f1_macro={stats.get('judge_f1_macro'):.4f}  n={len(rows)}")
    return out, len(keys)


def _boot_ci(vals, rng, n_boot: int = 2000):
    """(mean, ci_low, ci_high) — same percentile-bootstrap convention as
    plot_belief_by_round in verifier_cot.py, duplicated here (not imported)
    since this module is deliberately GPU/model-free and this is a ~10
    line helper, not worth a shared-module round-trip for."""
    import numpy as np
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 2:
        v = float(arr[0]) if len(arr) else 0.0
        return v, v, v
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def plot_round_diagnostics(verifier_dir: Path, model_safe: str, baseline: str, out_dir: Path) -> None:
    """Four per-round diagnostic charts read directly from the verifier's
    own round-shaped checkpoint (every round's own row, not just
    is_final_round==1 — unlike collect()/plot() above, which only care
    about the final answer). No-ops with a note if the checkpoint is
    single-round (max_round=1: a belief/confidence trend needs >1 round to
    say anything).

    Built to answer three questions that came up tuning min_round/
    max_round/high_conf on a real run: (1) is stop_confidence actually
    calibrated to correctness, or just to "this survived a lot of duels"
    (diag_confidence_calibration.png) — on a real run this came back FLAT
    (~35% judged-correct in every confidence bin from 0.4-0.9), meaning
    stop_confidence is not a correctness signal, only a compute-saving
    lever; (2) where does stop_confidence actually sit relative to
    high_conf round by round, so you can pick a threshold instead of
    guessing (diag_round_vs_stop_confidence.png); (3) how much of the
    round budget is spent AFTER the answer already stopped changing
    (diag_answer_settle_round.png) — on that same run all 40/40 images'
    final answer was locked in by round 3, yet confidence was still only
    ~0.76 by round 10, meaning most of the round budget bought confidence
    in an already-decided answer, not new correctness."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt = verifier_dir / model_safe / f"ckpt_{baseline}_{model_safe}.csv"
    if not ckpt.exists():
        return
    with open(ckpt, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    max_round = max(int(r["round"]) for r in rows)
    if max_round <= 1:
        print("  [round-diagnostics] skip — single-round checkpoint (max_round=1), nothing to trend")
        return

    by_img: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        key = (r["agml_dataset"], r["class_label"], r["image_name"])
        by_img.setdefault(key, {})[int(r["round"])] = r
    n_images = len(by_img)
    rng = np.random.default_rng(42)

    # ── diag_confidence_calibration.png: stop_confidence bin -> judged accuracy
    bins = np.arange(0.0, 1.01, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    bin_acc, bin_n, bin_lo, bin_hi = [], [], [], []
    for i in range(len(bins) - 1):
        lo_b, hi_b = bins[i], bins[i + 1]
        sub = [r for r in rows if lo_b <= float(r["stop_confidence"]) < hi_b
               or (hi_b >= 1.0 and float(r["stop_confidence"]) == 1.0)]
        if not sub:
            bin_acc.append(np.nan); bin_n.append(0); bin_lo.append(np.nan); bin_hi.append(np.nan)
            continue
        m, lo, hi = _boot_ci([1.0 if r["judged_correct"] == "1" else 0.0 for r in sub], rng)
        bin_acc.append(m); bin_n.append(len(sub)); bin_lo.append(m - lo); bin_hi.append(hi - m)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(bin_labels))
    valid = [i for i in x if bin_n[i] > 0]
    ax.bar([x[i] for i in valid], [bin_acc[i] for i in valid], color="#e06b00",
           edgecolor="black", linewidth=0.6,
           yerr=[[bin_lo[i] for i in valid], [bin_hi[i] for i in valid]], capsize=3)
    for i in valid:
        ax.annotate(f"n={bin_n[i]}", (x[i], 0.02), ha="center", fontsize=8,
                     color="white" if bin_acc[i] > 0.08 else "black")
    ax.plot([-0.5, len(bins) - 1.5], [0, 1], color="gray", linestyle=":", linewidth=1,
            alpha=0.5, label="perfect calibration (identity)")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("stop_confidence bin"); ax.set_ylabel("judged_correct rate")
    ax.set_title(f"Calibration: stop_confidence vs. judged accuracy — {model_safe}\n"
                 f"(all rounds pooled, n={len(rows)} rows, {n_images} images x {max_round} rounds)")
    ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_confidence_calibration.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'diag_confidence_calibration.png'}")

    # ── diag_round_vs_stop_confidence.png
    by_round_conf: dict[int, list[float]] = {}
    for r in rows:
        by_round_conf.setdefault(int(r["round"]), []).append(float(r["stop_confidence"]))
    rr = sorted(by_round_conf)
    means, lo_e, hi_e = [], [], []
    for k in rr:
        m, lo, hi = _boot_ci(by_round_conf[k], rng)
        means.append(m); lo_e.append(m - lo); hi_e.append(hi - m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rr, means, marker="o", color="#e06b00", linewidth=2)
    ax.fill_between(rr, np.array(means) - np.array(lo_e), np.array(means) + np.array(hi_e),
                     alpha=0.15, color="#e06b00")
    ax.set_xlabel("Round"); ax.set_ylabel("stop_confidence")
    ax.set_title(f"Round vs. stop_confidence, 95% CI — {model_safe}")
    ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_round_vs_stop_confidence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'diag_round_vs_stop_confidence.png'}")

    # ── diag_round_vs_accuracy.png: judged_correct by round, overall + per
    # task_type. contains_match deliberately dropped (per explicit
    # request) — agricultural disease naming varies enough internationally
    # that strict string match undercounts correct answers systematically;
    # judged_correct is the metric actually being optimized here.
    by_round_rows: dict[int, list[dict]] = {}
    for r in rows:
        by_round_rows.setdefault(int(r["round"]), []).append(r)
    task_types = sorted({r.get("task_type", "") for r in rows} - {""})
    _task_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    groups = [("overall", None, "#1f77b4", 2.5)] + [
        (tt, tt, _task_colors[k % len(_task_colors)], 1.3) for k, tt in enumerate(task_types)
    ]
    for label, tt_filter, color, lw in groups:
        means, lo_e, hi_e = [], [], []
        for k in rr:
            sub = by_round_rows[k] if tt_filter is None else [r for r in by_round_rows[k] if r.get("task_type") == tt_filter]
            if not sub:
                means.append(np.nan); lo_e.append(0.0); hi_e.append(0.0)
                continue
            m, lo, hi = _boot_ci([1.0 if r["judged_correct"] == "1" else 0.0 for r in sub], rng)
            means.append(m); lo_e.append(m - lo); hi_e.append(hi - m)
        line, = ax.plot(rr, means, marker="o", label=label, color=color, linewidth=lw)
        ax.fill_between(rr, np.array(means) - np.array(lo_e), np.array(means) + np.array(hi_e),
                         alpha=0.12, color=line.get_color())
    ax.set_xlabel("Round"); ax.set_ylabel("judged_correct rate")
    ax.set_title(f"Round vs. judged accuracy, 95% CI — {model_safe}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_round_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'diag_round_vs_accuracy.png'}")

    # ── diag_round_vs_timing.png: avg_time_per_sample_s by round — cost
    # side of the round-count tradeoff, next to the accuracy chart above.
    if all("avg_time_per_sample_s" in r for r in rows[:1]):
        fig, ax = plt.subplots(figsize=(8, 5))
        means, lo_e, hi_e = [], [], []
        for k in rr:
            vals = [float(r["avg_time_per_sample_s"]) for r in by_round_rows[k] if r.get("avg_time_per_sample_s")]
            if not vals:
                means.append(np.nan); lo_e.append(0.0); hi_e.append(0.0)
                continue
            m, lo, hi = _boot_ci(vals, rng)
            means.append(m); lo_e.append(m - lo); hi_e.append(hi - m)
        ax.plot(rr, means, marker="o", color="#d62728", linewidth=2)
        ax.fill_between(rr, np.array(means) - np.array(lo_e), np.array(means) + np.array(hi_e),
                         alpha=0.15, color="#d62728")
        ax.set_xlabel("Round"); ax.set_ylabel("avg time per sample (s)")
        ax.set_title(f"Round vs. per-sample wall-clock time, 95% CI — {model_safe}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "diag_round_vs_timing.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'diag_round_vs_timing.png'}")

    # ── diag_answer_settle_round.png: last round with a real (non-"defended") change
    last_change = []
    for key, r in by_img.items():
        lc = 1
        for i in range(2, max_round + 1):
            if i in r and r[i]["pairwise_winner"] != "defended":
                lc = i
        last_change.append(lc)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(last_change, bins=np.arange(0.5, max_round + 1.5, 1), color="#9467bd",
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Last round with a real answer change (stable afterward)")
    ax.set_ylabel("# images")
    ax.set_title(f"When do answers settle? — {model_safe}\n(n={n_images} images, max_round={max_round})")
    ax.set_xticks(range(1, max_round + 1))
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_answer_settle_round.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'diag_answer_settle_round.png'}")


def plot(results, n_images: int, model_safe: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [(lbl, color, st, n) for lbl, color, st, n in results if st]
    if not present:
        print("  [skip] nothing to plot")
        return

    labels = [lbl for lbl, _, _, _ in present]
    vals = [st.get("judge_f1_macro") or 0.0 for _, _, st, _ in present]
    colors = [c for _, c, _, _ in present]
    lo_err, hi_err = [], []
    for _, _, st, _ in present:
        m = st.get("judge_f1_macro") or 0.0
        lo_err.append(max(0.0, m - (st.get("judge_ci_low") or m)))
        hi_err.append(max(0.0, (st.get("judge_ci_high") or m) - m))

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.6), 5.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=[lo_err, hi_err], capsize=4,
                  error_kw={"elinewidth": 1.0, "alpha": 0.75})
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")

    # Bound reference lines make "did the verifier clear the floor / approach
    # the ceiling" readable without eyeballing bar heights.
    by_label = {lbl: v for lbl, v in zip(labels, vals)}
    if "Lower Bound" in by_label:
        ax.axhline(by_label["Lower Bound"], color="#1f77b4", linestyle=":", linewidth=1.4, alpha=0.8)
    if "Upper Bound" in by_label:
        ax.axhline(by_label["Upper Bound"], color="#d62728", linestyle=":", linewidth=1.4, alpha=0.8)

    ax.set_ylabel("Judged F1 Macro", fontsize=11)
    ax.set_ylim(0, max(v + h for v, h in zip(vals, hi_err)) * 1.18)
    ax.set_title(f"Verifier vs. bounds and CoT methods — {model_safe}\n"
                 f"same {n_images} images, 95% CI", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    verifier_dir = Path(args.verifier_dir)
    model_safe = args.model.replace("/", "_")
    results, n_images = collect(verifier_dir, model_safe, args.baseline, args.n_boot, args.seed)

    out_dir = verifier_dir / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(results, n_images, model_safe, out_dir / "verifier_summary.png")
    plot_round_diagnostics(verifier_dir, model_safe, args.baseline, out_dir)

    csv_path = out_dir / "verifier_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "judge_f1_macro", "judge_ci_low", "judge_ci_high",
                    "n_rows", "n_images", "status"])
        for label, _c, st, n in results:
            if st:
                w.writerow([label, st.get("judge_f1_macro"), st.get("judge_ci_low"),
                            st.get("judge_ci_high"), n, n_images, "ok"])
            else:
                # Keep the real overlap count: n>0 here means the rows exist
                # but weren't judged (very different from having no data).
                w.writerow([label, "", "", "", n, n_images,
                            "unjudged" if n else "no_rows"])
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verifier_dir", default="outputs_final/verifier_cot_fasttest/",
                   help="verifier_cot.py's --output_dir for the run being summarized.")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--baseline", choices=["belief"], default="belief")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CIs.")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
