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
    p.add_argument("--baseline", choices=["belief", "argmax"], default="belief")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CIs.")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
