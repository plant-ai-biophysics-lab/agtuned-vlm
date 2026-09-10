"""Judge template sensitivity sweep — Phase 3 of the judge-improvements
evaluation plan (see the "LLM as a judge improvements" note).

Re-judges Phase 0's SAME predictions (the full model response, pred_key=
"prediction" — no new generation needed) under three template variants:

  current       judge_classification_template — the pipeline's existing
                template, unchanged. Reuses the checkpoint's already-stored
                judged_correct rather than a fresh call (identical
                conditions: same template, same pred_key, same greedy
                temperature=0.0 — a fresh call would just reproduce it).
  specificity   judge_classification_specificity_template — requires the
                prediction to name the reference class or a recognized
                synonym; a generic symptom-shape label doesn't count as a
                match just because it's broadly consistent. Rewritten for
                plain classification (see prompts.yaml's comment — the
                existing judge_classification_strict_template assumes a
                rubric narrative to isolate a diagnosis from, which a plain
                1-2 sentence answer doesn't have).
  class_list    judge_classification_class_list_template — same base
                instruction, plus the full list of possible classes for
                that row's dataset, testing whether seeing the other
                possible classes alone makes the judge more specificity-
                demanding without being explicitly told to be.

Each variant is scored against Phase 1's deterministic baseline — NOT the
resolved_class-judge comparison from judge_leniency.py, but the checkpoint's
own `correct` column (resolve_class(prediction, classes) == ground_truth,
computed at generation time, zero-cost). This is a fixed yardstick shared
across all three variants, so which template narrows the judge-vs-
deterministic gap most is directly comparable.

Also covers the plan's length/verbosity bias check: buckets judged_correct
rate by len(prediction), split by whether the row is deterministically
correct or not (`correct` column) — does judged_correct rate rise with
response length even when the answer is deterministically wrong (the
leniency-by-verbosity signature)?

(Class-list order sensitivity was considered and dropped — shuffling which
classes appear where in a list the judge is already told is exhaustive
("here is the full list of possible classes") isn't expected to move the
verdict, and isn't worth the extra judge-call budget to confirm.)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.leniency_analysis import TASK_TYPE_LABELS, TASK_TYPE_ORDER

TEMPLATE_VARIANTS = [
    # (variant_key, template_key, needs_class_list, label)
    ("current", "judge_classification_template", False, "Current template"),
    ("specificity", "judge_classification_specificity_template", False, "Specificity-strict template"),
    ("class_list", "judge_classification_class_list_template", True, "+ full class list"),
]

_VARIANT_COLORS = {"current": "#7f7f7f", "specificity": "#d62728", "class_list": "#1f77b4"}


def add_class_list_field(rows: list[dict], dataset_classes: dict[str, list[str]]) -> None:
    """In place: row["class_list_text"] = a newline-bulleted rendering of
    dataset_classes[row["agml_dataset"]], joined for the {class_list}
    placeholder (judge_classification_class_list_template, filled via
    apply_judge's extra_keys) — datasets.yaml's own class order, unmodified."""
    for r in rows:
        classes = dataset_classes.get(r.get("agml_dataset", ""), [])
        r["class_list_text"] = "\n".join(f"- {c}" for c in classes)


def score_against_deterministic(
    rows_by_variant: dict[str, list[dict]], n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """{variant_key: {task_type: {n, judged_correct_rate,
    deterministic_correct_rate, gap, judged_correct_rate_ci,
    deterministic_correct_rate_ci, gap_ci}}, ...} + "overall" per variant.

    deterministic_correct_rate reads `correct` directly off each row (the
    checkpoint's own resolve_class(prediction, classes) == ground_truth,
    NOT recomputed) — the fixed yardstick every variant is compared
    against, per the plan's "scored against Phase 1's deterministic
    baseline." gap_ci is a PAIRED bootstrap (see src.common.bootstrap) —
    judged_correct and correct are two fields on the SAME row, not two
    independent samples.
    """
    from src.common.bootstrap import bootstrap_diff_ci, bootstrap_mean_ci

    out: dict[str, dict] = {}
    for variant_key, rows in rows_by_variant.items():
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_task[r.get("task_type", "")].append(r)

        def _summarize(trows: list[dict]) -> dict:
            n = len(trows)
            if n == 0:
                return {"n": 0, "judged_correct_rate": float("nan"),
                        "deterministic_correct_rate": float("nan"), "gap": float("nan"),
                        "judged_correct_rate_ci": (float("nan"), float("nan")),
                        "deterministic_correct_rate_ci": (float("nan"), float("nan")),
                        "gap_ci": (float("nan"), float("nan"))}
            judged_vals = [int(r.get("judged_correct", "") in ("1", 1, True)) for r in trows]
            det_vals = [int(r.get("correct", "") in ("1", 1, True)) for r in trows]
            judged = float(np.mean(judged_vals))
            det = float(np.mean(det_vals))
            return {
                "n": n, "judged_correct_rate": round(judged, 4),
                "deterministic_correct_rate": round(det, 4), "gap": round(judged - det, 4),
                "judged_correct_rate_ci": bootstrap_mean_ci(judged_vals, n_boot, ci, seed),
                "deterministic_correct_rate_ci": bootstrap_mean_ci(det_vals, n_boot, ci, seed),
                "gap_ci": bootstrap_diff_ci(judged_vals, det_vals, n_boot, ci, seed),
            }

        variant_results = {t: _summarize(trows) for t, trows in by_task.items()}
        variant_results["overall"] = _summarize(rows)
        out[variant_key] = variant_results
    return out


def plot_template_sensitivity(results_by_variant: dict[str, dict], output_path: Path) -> None:
    """Grouped bar chart: judged_correct_rate per task_type (+ overall), one
    group of bars per template variant, with the deterministic
    correct_rate drawn as a horizontal reference line per group (it's the
    same fixed number across variants — only judged_correct_rate should
    move) — the gap between each bar and that line IS this variant's
    leniency relative to the shared deterministic baseline. Error bars are
    bootstrap 95% CIs (judged_correct_rate_ci) when present."""
    import matplotlib.pyplot as plt

    from src.common.bootstrap import yerr_pairs

    variant_keys = [v[0] for v in TEMPLATE_VARIANTS if v[0] in results_by_variant]
    any_result = results_by_variant[variant_keys[0]]
    task_types = [t for t in TASK_TYPE_ORDER if t in any_result]
    task_types += [t for t in any_result if t not in TASK_TYPE_ORDER and t != "overall"]
    task_types.append("overall")
    labels = [TASK_TYPE_LABELS.get(t, t) if t != "overall" else "Overall" for t in task_types]

    x = np.arange(len(task_types))
    n_variants = len(variant_keys)
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 2.0), 5.5))
    for i, variant_key in enumerate(variant_keys):
        variant_label = next(lbl for k, _, _, lbl in TEMPLATE_VARIANTS if k == variant_key)
        results = results_by_variant[variant_key]
        vals = [results.get(t, {}).get("judged_correct_rate", float("nan")) for t in task_types]
        yerr = yerr_pairs(results, task_types, "judged_correct_rate", "judged_correct_rate_ci")
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=variant_label, color=_VARIANT_COLORS.get(variant_key, "#333"),
               yerr=yerr, capsize=2, ecolor="#333", error_kw={"linewidth": 0.8})

    # Deterministic reference line — same across variants (drawn once per
    # task_type group from the first variant's own `correct`-derived rate).
    det_vals = [any_result.get(t, {}).get("deterministic_correct_rate", float("nan")) for t in task_types]
    for xi, dv in zip(x, det_vals):
        if not np.isnan(dv):
            ax.plot([xi - 0.4, xi + 0.4], [dv, dv], color="black", linewidth=1.5, linestyle="-", zorder=4)
    ax.plot([], [], color="black", linewidth=1.5, linestyle="-", label="Deterministic match (resolve_class)")

    ax.set_xticks(x)
    ns = [any_result.get(t, {}).get("n", 0) for t in task_types]
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("judged_correct rate")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def length_bias_report(
    rows: list[dict], pred_key: str = "prediction", n_buckets: int = 4,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """Buckets rows by len(row[pred_key]) into n_buckets equal-count
    (quantile) buckets — computed globally across all rows so bucket edges
    are comparable across the two deterministic-correctness strata — then
    reports judged_correct_rate per bucket, split by whether the row is
    deterministically correct (`correct` column) or not.

    Returns {"correct_0": {"buckets": [{"range": [lo, hi], "n":..,
    "judged_correct_rate":.., "judged_correct_rate_ci": (lo, hi)}, ...]},
    "correct_1": {...}}. The signature to look for: correct_0's
    judged_correct_rate rising with bucket length (longer wrong answers
    getting undeserved credit) is the leniency-by-verbosity pattern;
    correct_1 should stay roughly flat near 1.0 regardless of length as a
    sanity check. judged_correct_rate_ci is a bootstrap 95% CI over that
    bucket's own rows (see src.common.bootstrap) — None for empty buckets.
    """
    from src.common.bootstrap import bootstrap_mean_ci

    lengths = np.array([len(str(r.get(pred_key, ""))) for r in rows])
    edges = np.quantile(lengths, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1  # right-inclusive top edge so the max-length row falls in the last bucket

    out: dict[str, dict] = {}
    for stratum_key, stratum_val in (("correct_0", 0), ("correct_1", 1)):
        stratum_rows = [r for r in rows if int(r.get("correct", "") in ("1", 1, True)) == stratum_val]
        stratum_lengths = np.array([len(str(r.get(pred_key, ""))) for r in stratum_rows])
        buckets = []
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            mask = (stratum_lengths >= lo) & (stratum_lengths < hi)
            bucket_rows = [r for r, m in zip(stratum_rows, mask) if m]
            n = len(bucket_rows)
            vals = [int(r.get("judged_correct", "") in ("1", 1, True)) for r in bucket_rows]
            rate = float(np.mean(vals)) if n else float("nan")
            buckets.append({
                "range": [round(float(lo), 1), round(float(hi), 1)], "n": n,
                "judged_correct_rate": round(rate, 4) if n else None,
                "judged_correct_rate_ci": bootstrap_mean_ci(vals, n_boot, ci, seed) if n else None,
            })
        out[stratum_key] = {"buckets": buckets, "n_total": len(stratum_rows)}
    return out


def plot_length_bias(report: dict, output_path: Path) -> None:
    """Line chart: judged_correct_rate vs. length bucket, one line for
    deterministically-wrong rows (correct_0) and one for deterministically-
    correct rows (correct_1). Error bars are bootstrap 95% CIs
    (judged_correct_rate_ci) when present."""
    import matplotlib.pyplot as plt

    from src.common.bootstrap import ci_to_yerr

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"correct_0": "#d62728", "correct_1": "#1f77b4"}
    labels = {"correct_0": "Deterministically WRONG (correct=0)", "correct_1": "Deterministically CORRECT (correct=1)"}
    for stratum_key in ("correct_0", "correct_1"):
        buckets = report.get(stratum_key, {}).get("buckets", [])
        xs = list(range(len(buckets)))
        ys = [b["judged_correct_rate"] for b in buckets]
        yerr = None
        if buckets and all(b.get("judged_correct_rate_ci") is not None for b in buckets):
            lowers, uppers = zip(*(ci_to_yerr(b["judged_correct_rate"], b["judged_correct_rate_ci"]) for b in buckets))
            yerr = [lowers, uppers]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color=colors[stratum_key], label=labels[stratum_key],
                    capsize=3, ecolor=colors[stratum_key], elinewidth=1)

    # Bucket edges are shared across both strata (computed globally in
    # length_bias_report), but each stratum has its own row count per
    # bucket — show both n's in the tick label rather than only the
    # last-plotted stratum's (a real bug in an earlier version of this
    # function: xticklabels was overwritten each loop iteration, so only
    # correct_1's counts ever made it onto the axis).
    wrong_buckets = report.get("correct_0", {}).get("buckets", [])
    correct_buckets = report.get("correct_1", {}).get("buckets", [])
    if wrong_buckets:
        xticklabels = [
            f"{b['range'][0]:.0f}-{b['range'][1]:.0f}\n(wrong n={b['n']}, correct n={c['n']})"
            for b, c in zip(wrong_buckets, correct_buckets)
        ]
        ax.set_xticks(range(len(wrong_buckets)))
        ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_xlabel("Prediction length (characters), quantile buckets")
    ax.set_ylabel("judged_correct rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_ci(value, ci) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    lo, hi = ci if ci is not None else (float("nan"), float("nan"))
    if isinstance(lo, float) and np.isnan(lo):
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def write_variant_temperature_latex_table(
    variant_data: dict[str, dict], output_path: Path,
    caption: str = "Leniency gap and self-consistency flip rate (overall, pooled across "
                   "task\\_types) per judge template variant and temperature. Bracketed "
                   "values are bootstrap 95\\% CIs.",
    label: str = "tab:judge_template_sensitivity",
) -> None:
    """variant_data: {variant_key: {"leniency_by_temp": {temperature:
    compute_leniency_gap(...) result}, "flip_by_temp": {temperature:
    compute_flip_rate(...) result}}} — as assembled by
    src/judge_template_sensitivity.py's orchestrator from each variant's
    own leniency_gap.json/flip_rate.json. Writes a LaTeX booktabs table,
    one row per (variant, temperature), reporting the "overall" (pooled
    across task_types) leniency_gap and flip_rate — the per-task_type
    breakdown stays in the PNGs; this table is the compact appendix-ready
    cross-variant summary.
    """
    variant_labels = {k: lbl for k, _, _, lbl in TEMPLATE_VARIANTS}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Template variant & Temperature & Leniency gap [95\% CI] & Flip rate [95\% CI] \\",
        r"\midrule",
    ]

    variant_keys = [k for k, _, _, _ in TEMPLATE_VARIANTS if k in variant_data]
    for vi, variant_key in enumerate(variant_keys):
        data = variant_data[variant_key]
        leniency_by_temp = data.get("leniency_by_temp", {})
        flip_by_temp = data.get("flip_by_temp", {})
        temperatures = sorted(set(leniency_by_temp) | set(flip_by_temp))
        n = len(temperatures)
        variant_label = variant_labels.get(variant_key, variant_key).replace("_", r"\_")
        for ti, temperature in enumerate(temperatures):
            variant_cell = rf"\multirow{{{n}}}{{*}}{{{variant_label}}}" if ti == 0 else ""
            leniency = leniency_by_temp.get(temperature, {}).get("overall", {})
            flip = flip_by_temp.get(temperature, {}).get("overall", {})
            leniency_cell = _fmt_ci(leniency.get("leniency_gap"), leniency.get("leniency_gap_ci"))
            flip_cell = _fmt_ci(flip.get("flip_rate"), flip.get("flip_rate_ci"))
            lines.append(f"{variant_cell} & T={temperature} & {leniency_cell} & {flip_cell} \\\\")
        if vi < len(variant_keys) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
