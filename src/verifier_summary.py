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


def reference_image_keys(reference_ckpt: Path, n_samples: int) -> set[tuple[str, str, str]]:
    """The SAME deterministic 'first n_samples image_names per (dataset,
    class), in that checkpoint's own row order' selection verifier_cot.py's
    --match_images_from uses (_load_reference_images there) — duplicated
    here rather than imported, since this module is deliberately GPU/
    model-free and verifier_cot.py pulls in vllm at import time.

    Exists so a comparison report can be built BEFORE a verifier-cot run
    exists at all: since --match_images_from makes verifier-cot's own
    image selection a deterministic function of an existing baseline
    checkpoint (e.g. vision_baseline's), that same image set is already
    fully determined right now — no need to wait for the verifier run to
    finish just to know which images it WILL use."""
    by_group: dict[tuple[str, str], list[str]] = {}
    with open(reference_ckpt, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            names = by_group.setdefault(key, [])
            name = r.get("image_name", "")
            if name and name not in names:
                names.append(name)
    return {(ds, cl, name) for (ds, cl), names in by_group.items() for name in names[:n_samples]}


def collect(keys: set[tuple[str, str, str]], model_safe: str, n_boot: int, seed: int,
            conditions: list[tuple[str, str | None, str]] = _CONDITIONS,
            verifier_dir: Path | None = None, baseline: str = "belief"):
    """[(label, color, stats_or_None, n_rows)] for every condition in
    `conditions`, filtered to `keys` (agml_dataset, class_label,
    image_name triples). verifier_dir/baseline only needed if `conditions`
    includes the Verifier-Belief entry (dirname=None)."""
    print(f"  Filtering every condition to exactly {len(keys)} image(s)")

    def _filt(rows):
        return [r for r in rows
                if (r.get("agml_dataset"), r.get("class_label"), r.get("image_name")) in keys]

    out = []
    for label, dirname, color in conditions:
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
        # Full dict — {"overall": {...}, "<task_type>": {...}, ...} — kept
        # in full (not just ["overall"]) so plot_by_task can break the same
        # bar chart out per task_type without a second bootstrap pass.
        stats = compute_condition_stats(rows, n_boot, seed)
        overall = stats.get("overall", {})
        # A condition whose overlapping rows were never JUDGED reports
        # judge_f1_macro=0.0 — which is missing data, not a score of zero.
        # Plotting that as a real bar would be actively misleading (E4B-it's
        # self_cot rows are exactly this case: present but unjudged), so
        # treat it the same as "no rows at all".
        if not overall.get("judge_n_classes"):
            print(f"  [skip] {label}: {len(rows)} row(s) overlap but none are judged "
                  f"(judge_f1_macro would be a meaningless 0.0)")
            out.append((label, color, None, len(rows)))
            continue
        out.append((label, color, stats, len(rows)))
        print(f"  {label:16s} judge_f1_macro={overall.get('judge_f1_macro'):.4f}  n={len(rows)}")
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
        sub = [r for r in rows if (lo_b <= float(r["stop_confidence"]) < hi_b
               or (hi_b >= 1.0 and float(r["stop_confidence"]) == 1.0))
               and r.get("judged_correct") in ("0", "1")]  # exclude never-judged rows (e.g.
               # --judge_scope first_and_final leaves most rounds blank) — blank isn't a 0
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
            # Exclude never-judged rows (blank judged_correct) — with
            # --judge_scope first_and_final, every round except 1 and
            # each image's own final round is intentionally left
            # unjudged, and treating blank as "wrong" would fake a
            # 0-in-the-middle dip that isn't a real accuracy trend.
            sub = [r for r in sub if r.get("judged_correct") in ("0", "1")]
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

    # ── diag_round1_vs_last.png: simple 2-bar version — Round 1 (every
    # image's cold-start answer) vs. Last Round (is_final_round==1 — each
    # image's own actual final answer, whether it stopped early or ran the
    # full max_round budget; NOT literally round==max_round, so an
    # early-stopped image's real answer still counts here). Overall +
    # per-task grouped bars, same "exclude never-judged rows" guard.
    stage_defs2 = [
        ("Round 1", lambda r: str(r.get("round")) == "1"),
        ("Last Round", lambda r: str(r.get("is_final_round")) == "1"),
    ]

    def _stage_rate(sub_rows, pred):
        sub = [r for r in sub_rows if pred(r) and r.get("judged_correct") in ("0", "1")]
        if not sub:
            return None
        return _boot_ci([1.0 if r["judged_correct"] == "1" else 0.0 for r in sub], rng)

    scopes2 = ["overall"] + task_types
    fig, ax = plt.subplots(figsize=(max(7, len(scopes2) * 1.4), 5.5))
    x2 = np.arange(len(scopes2))
    width2 = 0.35
    stage_colors = ["#7f7f7f", "#e06b00"]
    any_plotted2 = False
    for j, (stage_label, pred) in enumerate(stage_defs2):
        vals, lo_e, hi_e, xs = [], [], [], []
        for i, scope in enumerate(scopes2):
            sub_rows = rows if scope == "overall" else [r for r in rows if r.get("task_type") == scope]
            res = _stage_rate(sub_rows, pred)
            if res is None:
                continue
            m, lo, hi = res
            xs.append(i); vals.append(m); lo_e.append(m - lo); hi_e.append(hi - m)
        if not xs:
            continue
        any_plotted2 = True
        offset = (j - 0.5) * width2
        ax.bar(np.array(xs) + offset, vals, width2, label=stage_label, color=stage_colors[j],
               edgecolor="black", linewidth=0.5, yerr=[lo_e, hi_e], capsize=3,
               error_kw={"elinewidth": 0.8, "alpha": 0.7})
    if any_plotted2:
        ax.set_xticks(x2); ax.set_xticklabels(scopes2, rotation=12, ha="right")
        ax.set_ylabel("judged_correct rate", fontsize=11)
        ax.set_title(f"Round 1 vs. Last Round — {model_safe}\nmax_round={max_round}, 95% CI", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "diag_round1_vs_last.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'diag_round1_vs_last.png'}")
    else:
        plt.close(fig)
        print("  [skip] diag_round1_vs_last: no judged rows")

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

    # ── diag_stop_round_distribution.png: which round did each image's
    # loop actually terminate at (is_final_round==1) — directly answers
    # "how many locked in within the lock_after_n_defends window" (a
    # sharp spike at that round) vs. how many kept going because a real
    # leader change reset their consecutive-defend streak.
    stop_rounds = [int(r) for i, rr in by_img.items() for r in rr if str(rr[r].get("is_final_round")) == "1"]
    if stop_rounds:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(stop_rounds, bins=np.arange(0.5, max_round + 1.5, 1), color="#2ca02c",
                edgecolor="black", linewidth=0.6)
        for r in range(1, max_round + 1):
            n_r = stop_rounds.count(r)
            if n_r:
                ax.annotate(str(n_r), (r, n_r), ha="center", va="bottom", fontsize=9, fontweight="bold",
                            xytext=(0, 2), textcoords="offset points")
        ax.set_xlabel("Round the image's loop actually stopped at")
        ax.set_ylabel("# images")
        ax.set_title(f"When do images lock in / stop? — {model_safe}\n(n={len(stop_rounds)} images, max_round={max_round})")
        ax.set_xticks(range(1, max_round + 1))
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_dir / "diag_stop_round_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'diag_stop_round_distribution.png'}")

    # ── diag_flip_cause.png: for images whose judged_correct changed
    # between round 1 and final (regressed OR recovered), was that because
    # a genuine leader swap happened (pairwise_winner=="replaced" at some
    # round — a challenger actually beat leader_margin) or because the
    # leader never changed at all (every round "seed"/"defended", so any
    # judged_correct flip is the JUDGE disagreeing with itself on the same
    # or a reworded-but-same-cluster answer, not the mechanism doing
    # anything)? Answers "did a margin-beaten swap cause this flip, or is
    # this just judge noise on an unchanged answer."
    flip_cause = {"regressed": {"swap": 0, "no_swap": 0}, "recovered": {"swap": 0, "no_swap": 0}}
    unchanged_text_count = 0
    total_pairs = 0
    for key, rr in by_img.items():
        if 1 not in rr:
            continue
        final = next((rr[i] for i in rr if str(rr[i].get("is_final_round")) == "1"), None)
        if final is None or final.get("judged_correct") not in ("0", "1") \
                or rr[1].get("judged_correct") not in ("0", "1"):
            continue
        total_pairs += 1
        fr = max(rr)
        if rr[1]["prediction"] == final["prediction"]:
            unchanged_text_count += 1
        r1_correct = rr[1]["judged_correct"] == "1"
        rf_correct = final["judged_correct"] == "1"
        if r1_correct == rf_correct:
            continue
        cat = "regressed" if (r1_correct and not rf_correct) else "recovered"
        had_swap = any(rr[i].get("pairwise_winner") == "replaced" for i in range(2, fr + 1) if i in rr)
        flip_cause[cat]["swap" if had_swap else "no_swap"] += 1

    print(f"  [answer stability] {unchanged_text_count}/{total_pairs} images ({unchanged_text_count/total_pairs:.1%}) "
          f"have BYTE-IDENTICAL prediction text at round 1 and final" if total_pairs else "")

    if any(flip_cause[c][k] for c in flip_cause for k in flip_cause[c]):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        cats3 = ["regressed", "recovered"]
        x3 = np.arange(len(cats3))
        width3 = 0.35
        swap_vals = [flip_cause[c]["swap"] for c in cats3]
        no_swap_vals = [flip_cause[c]["no_swap"] for c in cats3]
        b1 = ax.bar(x3 - width3 / 2, swap_vals, width3, label="margin-beaten leader swap",
                    color="#1f77b4", edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x3 + width3 / 2, no_swap_vals, width3, label="no swap (judge noise on same leader)",
                    color="#7f7f7f", edgecolor="black", linewidth=0.5)
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                if h:
                    ax.annotate(str(int(h)), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                                fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x3); ax.set_xticklabels(cats3)
        ax.set_ylabel("# images")
        ax.set_title(f"What caused each flip? — {model_safe}\n"
                     f"({unchanged_text_count}/{total_pairs} images never changed prediction text at all)",
                     fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / "diag_flip_cause.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'diag_flip_cause.png'}")
    else:
        print("  [skip] diag_flip_cause: no regressed/recovered images to break down")

    # ── diag_regression_recovery.png/csv: round-1 vs. final-round
    # judged_correct, bucketed into stayed_correct / stayed_wrong /
    # regressed (right->wrong) / recovered (wrong->right) — the same
    # by-hand check used repeatedly this session to tell whether extra
    # rounds are net helping or net hurting, now a standing chart. Uses
    # is_final_round==1 (not literally round==max_round) so this stays
    # correct once early-stopping means different images finish at
    # different rounds.
    _CATS = ["stayed_correct", "stayed_wrong", "regressed", "recovered"]
    _CAT_COLOR = {"stayed_correct": "#2ca02c", "stayed_wrong": "#7f7f7f",
                  "regressed": "#d62728", "recovered": "#1f77b4"}

    def _bucket(group_rows: dict[tuple, dict[int, dict]]) -> dict[str, int]:
        counts = dict.fromkeys(_CATS, 0)
        for rr in group_rows.values():
            if 1 not in rr:
                continue
            final = next((rr[i] for i in rr if str(rr[i].get("is_final_round")) == "1"), None)
            if final is None:
                continue
            r1_correct = rr[1]["judged_correct"] == "1"
            rf_correct = final["judged_correct"] == "1"
            if r1_correct and rf_correct:
                counts["stayed_correct"] += 1
            elif not r1_correct and not rf_correct:
                counts["stayed_wrong"] += 1
            elif r1_correct and not rf_correct:
                counts["regressed"] += 1
            else:
                counts["recovered"] += 1
        return counts

    overall_counts = _bucket(by_img)
    by_task_counts = {}
    for tt in task_types:
        by_task_counts[tt] = _bucket({k: v for k, v in by_img.items() if v[1].get("task_type") == tt})

    csv_path = out_dir / "diag_regression_recovery.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", *_CATS, "n"])
        w.writerow(["overall", *[overall_counts[c] for c in _CATS], sum(overall_counts.values())])
        for tt in task_types:
            w.writerow([tt, *[by_task_counts[tt][c] for c in _CATS], sum(by_task_counts[tt].values())])
    print(f"  Saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(max(7, (1 + len(task_types)) * 1.8), 5.5))
    scopes = ["overall"] + task_types
    x = np.arange(len(scopes))
    width = 0.2
    for j, cat in enumerate(_CATS):
        vals = [overall_counts[cat]] + [by_task_counts[tt][cat] for tt in task_types]
        ax.bar(x + (j - 1.5) * width, vals, width, label=cat, color=_CAT_COLOR[cat],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(scopes, rotation=12, ha="right")
    ax.set_ylabel("# images")
    ax.set_title(f"Round 1 -> final round: stayed/regressed/recovered — {model_safe}\n(n={n_images} images)")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "diag_regression_recovery.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'diag_regression_recovery.png'}")

    # ── diag_belief_vs_outcome.png: does final-round mu/std actually track
    # correctness and the regressed/recovered/stayed outcome, or is belief
    # just climbing regardless (the flat-calibration finding from earlier
    # manual checks this session — stop_confidence came back uncorrelated
    # with judged accuracy)? Per image: final mu, final std (sqrt(sigma2)),
    # and the SAME 4-way outcome category diag_regression_recovery.png
    # buckets by (stayed_correct/stayed_wrong/regressed/recovered).
    per_image = []
    for key, rr in by_img.items():
        if 1 not in rr:
            continue
        final = next((rr[i] for i in rr if str(rr[i].get("is_final_round")) == "1"), None)
        if final is None or final.get("judged_correct") not in ("0", "1") \
                or rr[1].get("judged_correct") not in ("0", "1"):
            continue
        r1_correct = rr[1]["judged_correct"] == "1"
        rf_correct = final["judged_correct"] == "1"
        if r1_correct and rf_correct:
            cat = "stayed_correct"
        elif not r1_correct and not rf_correct:
            cat = "stayed_wrong"
        elif r1_correct and not rf_correct:
            cat = "regressed"
        else:
            cat = "recovered"
        per_image.append({
            "cat": cat, "mu": float(final["mu"]), "std": float(final["sigma2"]) ** 0.5,
            "correct": rf_correct,
        })

    if per_image:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

        # Left: mean final mu/std by outcome category — does belief
        # actually look different for regressed vs. recovered images, or
        # is it blind to the difference?
        for ax, metric, ylabel in [(axes[0], "mu", "mean final mu"), (axes[1], "std", "mean final std")]:
            vals, lo_e, hi_e, labels_c = [], [], [], []
            for cat in _CATS:
                sub = [p[metric] for p in per_image if p["cat"] == cat]
                if not sub:
                    continue
                m, lo, hi = _boot_ci(sub, rng)
                vals.append(m); lo_e.append(m - lo); hi_e.append(hi - m); labels_c.append(cat)
            colors_c = [_CAT_COLOR[c] for c in labels_c]
            bars = ax.bar(labels_c, vals, color=colors_c, edgecolor="black", linewidth=0.6,
                          yerr=[lo_e, hi_e], capsize=4, error_kw={"elinewidth": 1.0, "alpha": 0.75})
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=8, fontweight="bold", xytext=(0, 3), textcoords="offset points")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticklabels(labels_c, rotation=15, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Final belief (mu/std) vs. round-1->final outcome — {model_safe}\n"
                     f"(n={len(per_image)} judged images) — do NOT expect these bars to differ if "
                     f"belief isn't actually tracking correctness", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / "diag_belief_vs_outcome.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'diag_belief_vs_outcome.png'}")

        # Right-hand-side companion: mu vs. correct/wrong directly (not
        # just the 4-way outcome split) — the same calibration question
        # diag_confidence_calibration.png asks of stop_confidence, asked
        # of mu instead, since mu is the raw signal and stop_confidence is
        # derived from it against a frozen round-1 anchor.
        correct_mu = [p["mu"] for p in per_image if p["correct"]]
        wrong_mu = [p["mu"] for p in per_image if not p["correct"]]
        if correct_mu and wrong_mu:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            for label, vals, color in [("correct", correct_mu, "#2ca02c"), ("wrong", wrong_mu, "#d62728")]:
                m, lo, hi = _boot_ci(vals, rng)
                ax.bar(label, m, color=color, edgecolor="black", linewidth=0.6,
                       yerr=[[m - lo], [hi - m]], capsize=4)
                ax.annotate(f"{m:.3f}\n(n={len(vals)})", (label, m), ha="center", va="bottom",
                            fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
            ax.set_ylabel("mean final mu", fontsize=11)
            ax.set_title(f"Final mu: judged-correct vs. judged-wrong — {model_safe}", fontsize=11)
            ax.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            fig.savefig(out_dir / "diag_mu_correct_vs_wrong.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: {out_dir / 'diag_mu_correct_vs_wrong.png'}")
    else:
        print("  [skip] diag_belief_vs_outcome: no judged round-1+final pairs available")


def plot(results, n_images: int, model_safe: str, out_path: Path,
         title: str = "Verifier vs. bounds and CoT methods") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [(lbl, color, st, n) for lbl, color, st, n in results if st]
    if not present:
        print("  [skip] nothing to plot")
        return

    labels = [lbl for lbl, _, _, _ in present]
    vals = [st.get("overall", {}).get("judge_f1_macro") or 0.0 for _, _, st, _ in present]
    colors = [c for _, c, _, _ in present]
    lo_err, hi_err = [], []
    for _, _, st, _ in present:
        o = st.get("overall", {})
        m = o.get("judge_f1_macro") or 0.0
        lo_err.append(max(0.0, m - (o.get("judge_ci_low") or m)))
        hi_err.append(max(0.0, (o.get("judge_ci_high") or m) - m))

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
    ax.set_title(f"{title} — {model_safe}\n"
                 f"same {n_images} images, 95% CI", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_task(results, n_images: int, model_safe: str, out_path: Path,
                  title: str = "Verifier vs. bounds and CoT methods") -> None:
    """Grouped bar chart — same conditions as plot(), but one bar per
    task_type per condition instead of one overall bar. Reuses the full
    per-condition stats dict collect() already returns (task_type keys
    computed by compute_condition_stats alongside "overall"), so no
    second bootstrap pass over the rows is needed."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [(lbl, color, st, n) for lbl, color, st, n in results if st]
    if not present:
        print("  [skip] nothing to plot")
        return

    task_types = sorted({k for _, _, st, _ in present for k in st if k != "overall"})
    if not task_types:
        print("  [skip] no task_type breakdown available")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 1.8), 5.5))
    n_cond = len(present)
    width = 0.8 / n_cond
    x = np.arange(len(task_types))
    for j, (label, color, st, _n) in enumerate(present):
        vals, lo_err, hi_err = [], [], []
        for tt in task_types:
            sub = st.get(tt, {})
            # No rows (or none judged) for this condition/task combo —
            # leave the bar out entirely (NaN height) rather than drawing
            # a misleading 0.0 bar, same reasoning as collect()'s overall
            # "unjudged is missing data, not a score of zero" guard.
            if not sub.get("judge_n_classes"):
                vals.append(np.nan); lo_err.append(0.0); hi_err.append(0.0)
                continue
            m = sub.get("judge_f1_macro") or 0.0
            vals.append(m)
            lo_err.append(max(0.0, m - (sub.get("judge_ci_low") or m)))
            hi_err.append(max(0.0, (sub.get("judge_ci_high") or m) - m))
        offset = (j - (n_cond - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.7})

    ax.set_xticks(x); ax.set_xticklabels(task_types, rotation=12, ha="right")
    ax.set_ylabel("Judged F1 Macro", fontsize=11)
    ax.set_title(f"{title} (by task) — {model_safe}\nsame {n_images} images, 95% CI", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    model_safe = args.model.replace("/", "_")

    if args.match_images_from:
        # Baseline-only mode: no verifier-cot run needed at all — the
        # image set is already fully determined by an existing checkpoint
        # (see reference_image_keys' docstring). Verifier-Belief is
        # dropped from the condition list since there's nothing to plot.
        keys = reference_image_keys(Path(args.match_images_from), args.n_samples)
        conditions = [c for c in _CONDITIONS if c[0] != "Verifier-Belief"]
        results, n_images = collect(keys, model_safe, args.n_boot, args.seed, conditions=conditions)
        out_dir = Path(args.out_dir) / model_safe
        out_dir.mkdir(parents=True, exist_ok=True)
        # n_samples in the filename — otherwise a --n_samples 10 run
        # silently overwrites a --n_samples 25 run's chart/csv at the
        # same path (a real mistake made once during development).
        plot(results, n_images, model_safe, out_dir / f"baseline_summary_n{args.n_samples}.png",
             title="Bounds and CoT methods (pre-verifier)")
        plot_by_task(results, n_images, model_safe, out_dir / f"baseline_summary_n{args.n_samples}_by_task.png",
                     title="Bounds and CoT methods (pre-verifier)")
        csv_path = out_dir / f"baseline_summary_n{args.n_samples}.csv"
    else:
        verifier_dir = Path(args.verifier_dir)
        keys = verifier_image_keys(verifier_dir, model_safe, args.baseline)
        results, n_images = collect(keys, model_safe, args.n_boot, args.seed,
                                     verifier_dir=verifier_dir, baseline=args.baseline)
        out_dir = verifier_dir / model_safe
        out_dir.mkdir(parents=True, exist_ok=True)
        plot(results, n_images, model_safe, out_dir / "verifier_summary.png")
        plot_by_task(results, n_images, model_safe, out_dir / "verifier_summary_by_task.png")
        plot_round_diagnostics(verifier_dir, model_safe, args.baseline, out_dir)
        csv_path = out_dir / "verifier_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "judge_f1_macro", "judge_ci_low", "judge_ci_high",
                    "n_rows", "n_images", "status"])
        for label, _c, st, n in results:
            if st:
                o = st.get("overall", {})
                w.writerow([label, o.get("judge_f1_macro"), o.get("judge_ci_low"),
                            o.get("judge_ci_high"), n, n_images, "ok"])
            else:
                # Keep the real overlap count: n>0 here means the rows exist
                # but weren't judged (very different from having no data).
                w.writerow([label, "", "", "", n, n_images,
                            "unjudged" if n else "no_rows"])
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verifier_dir", default="outputs_final/verifier_cot_fasttest/",
                   help="verifier_cot.py's --output_dir for the run being summarized. Ignored if "
                        "--match_images_from is given (baseline-only mode, no verifier run needed).")
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--baseline", choices=["belief"], default="belief")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CIs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match_images_from", default=None,
                   help="Baseline-only mode (no verifier-cot run required): path to an existing "
                        "single-prediction-per-row checkpoint (e.g. vision_baseline's own "
                        "ckpt_<model>.csv) to derive the image set from — same deterministic "
                        "'first --n_samples image_names per (dataset, class)' selection "
                        "verifier_cot.py's own --match_images_from uses, so this shows exactly "
                        "what a not-yet-run verifier-cot invocation with the same "
                        "--match_images_from/--n_samples WILL compare against. Skips the "
                        "Verifier-Belief bar and the per-round diagnostic charts entirely.")
    p.add_argument("--n_samples", type=int, default=25,
                   help="--match_images_from only: images/class to take from the reference "
                        "checkpoint's own row order.")
    p.add_argument("--out_dir", default="outputs_final/baseline_summary/",
                   help="--match_images_from only: where to write baseline_summary.png/.csv "
                        "(nested under <out_dir>/<model_safe>/, same convention as "
                        "--verifier_dir/<model_safe>/verifier_summary.png).")
    main(p.parse_args())
