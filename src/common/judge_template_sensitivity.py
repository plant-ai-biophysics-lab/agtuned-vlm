"""Judge template sensitivity sweep — Phase 3 of the judge-improvements
evaluation plan (see the "LLM as a judge improvements" note).

Re-judges Phase 0's SAME predictions (the full model response, pred_key=
"prediction" — no new generation needed) under four template variants:

  current       judge_classification_template — the pipeline's existing
                template, unchanged. Always gets its own fresh greedy-
                temperature judge call, same as the other two variants —
                NOT a reuse of the checkpoint's own stored judged_correct
                (an earlier version of this orchestrator did that, since
                the conditions are identical and a fresh call reproduces
                the same verdict; but it breaks on a --skip_judge
                checkpoint, where judged_correct is blank everywhere).
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
  hybrid        judge_classification_hybrid_template — Phase 4. Combines
                class_list's context + specificity's generic-label guardrail,
                plus a distinction neither of the other two draws: the
                prediction doesn't need exact wording, but crediting it
                requires a NECESSARY category-membership relationship (e.g.
                a specific defect is always a case of a broad catch-all
                class like "Bad") — never a plausible-cause guess for a
                generically-named reference class (e.g. one specific
                pathogen offered as an explanation for a symptom-only label,
                when many different pathogens could equally cause that same
                symptom). Targets class_list's own remaining rigidity
                (concentrated in quality_identification's coarse ground
                truth) without reopening the leaf-crinckle-style leniency
                bug specificity/class_list both already guard against.

Each variant is scored against Phase 1's deterministic baseline — NOT the
resolved_class-judge comparison from judge_leniency.py, but the checkpoint's
own `correct` column (resolve_class(prediction, classes) == ground_truth,
computed at generation time, zero-cost). This is a fixed yardstick shared
across all four variants, so which template narrows the judge-vs-
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

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.leniency_analysis import TASK_TYPE_LABELS, TASK_TYPE_ORDER

TEMPLATE_VARIANTS = [
    # (variant_key, template_key, needs_class_list, label)
    ("current", "judge_classification_template", False, "Current template"),
    ("specificity", "judge_classification_specificity_template", False, "Specificity-strict template"),
    ("class_list", "judge_classification_class_list_template", True, "+ full class list"),
    # Phase 4 — combines class_list's context + specificity's generic-label
    # guardrail, plus a new is-a-membership-vs-plausible-cause distinction
    # (see prompts.yaml's judge_classification_hybrid_template docstring)
    # targeting quality_identification's remaining class_list rigidity
    # (e.g. "moldy" not literally matching catch-all "Bad") without
    # reopening the leaf-crinckle-style leniency bug specificity/class_list
    # both guard against.
    ("hybrid", "judge_classification_hybrid_template", True, "Specificity + class list (hybrid)"),
]

_VARIANT_COLORS = {"current": "#7f7f7f", "specificity": "#d62728", "class_list": "#1f77b4", "hybrid": "#2ca02c"}

# Re-exported from src.common.dataset_config (moved there so every judge-calling
# script, not just this ablation orchestrator, can build a {class_list} value —
# needed now that src.common.metrics.DEFAULT_JUDGE_TEMPLATE_KEY has a
# {class_list} placeholder) — kept importable from here for existing callers
# (src/judge_leniency.py, src/judge_consistency.py).
from src.common.dataset_config import add_class_list_field  # noqa: E402


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


def compute_judge_accuracy_f1(
    rows_by_variant: dict[str, list[dict]], n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """{variant_key: {n, n_classes, accuracy, accuracy_ci, f1_macro, f1_macro_ci}}.

    Both metrics are judge-scored (judged_correct=1 counts as a match,
    falling back to resolved_class otherwise) — reuses existing functions,
    no new metric logic: accuracy is judged_correct's own mean + a
    bootstrap_mean_ci (src.common.bootstrap) resampled over rows; f1_macro
    is src.common.vision_eval.bootstrap_judge_f1_ci, resampled over CLASSES
    per that function's own established convention (same one
    judged_summary()'s point-estimate judge_f1_macro and every other
    judge-F1 number in this repo already use)."""
    from src.common.bootstrap import bootstrap_mean_ci
    from src.common.vision_eval import bootstrap_judge_f1_ci

    out: dict[str, dict] = {}
    for variant_key, rows in rows_by_variant.items():
        judged = [r for r in rows if r.get("judged_correct", "") != ""]
        if not judged:
            out[variant_key] = {"n": 0, "n_classes": 0, "accuracy": float("nan"),
                                 "accuracy_ci": (float("nan"), float("nan")),
                                 "f1_macro": float("nan"), "f1_macro_ci": (float("nan"), float("nan"))}
            continue
        jc = [int(r["judged_correct"]) for r in judged]
        f1_result = bootstrap_judge_f1_ci(rows, n_boot=n_boot, seed=seed)
        out[variant_key] = {
            "n": len(judged),
            "n_classes": f1_result["judge_n_classes"],
            "accuracy": round(float(np.mean(jc)), 4),
            "accuracy_ci": bootstrap_mean_ci(jc, n_boot, ci, seed),
            "f1_macro": f1_result["judge_f1_macro"],
            "f1_macro_ci": (f1_result["judge_ci_low"], f1_result["judge_ci_high"]),
        }
    return out


def plot_judge_accuracy_f1(results_by_variant: dict, output_path: Path) -> None:
    """Grouped bar chart: judge accuracy + judge-scored macro-F1, one group
    per variant, from compute_judge_accuracy_f1's output. Error bars are
    bootstrap 95% CIs."""
    import matplotlib.pyplot as plt

    from src.common.bootstrap import ci_to_yerr

    variant_keys = [v[0] for v in TEMPLATE_VARIANTS if v[0] in results_by_variant]
    labels = [next(lbl for k, _, _, lbl in TEMPLATE_VARIANTS if k == vk) for vk in variant_keys]

    x = np.arange(len(variant_keys))
    width = 0.35

    acc_vals = [results_by_variant[vk]["accuracy"] for vk in variant_keys]
    f1_vals = [results_by_variant[vk]["f1_macro"] for vk in variant_keys]
    acc_yerr = np.array([ci_to_yerr(v, results_by_variant[vk]["accuracy_ci"])
                          for vk, v in zip(variant_keys, acc_vals)]).T
    f1_yerr = np.array([ci_to_yerr(v, results_by_variant[vk]["f1_macro_ci"])
                         for vk, v in zip(variant_keys, f1_vals)]).T

    fig, ax = plt.subplots(figsize=(max(7, len(variant_keys) * 2.2), 5.5))
    ax.bar(x - width / 2, acc_vals, width, yerr=acc_yerr, capsize=3, label="Judge accuracy",
           color="#2a78d6", ecolor="#333", error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, f1_vals, width, yerr=f1_yerr, capsize=3, label="Judge-scored macro-F1",
           color="#eb6834", ecolor="#333", error_kw={"linewidth": 0.8})

    for xi, v in zip(x - width / 2, acc_vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, f1_vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xticks(x)
    ns = [results_by_variant[vk]["n"] for vk in variant_keys]
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
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


def _fmt_disagreement_cell(cell: dict | None) -> str:
    if not cell or cell.get("n") is None:
        return "--"
    lo, hi = cell.get("rate_ci", (float("nan"), float("nan")))
    if isinstance(lo, float) and np.isnan(lo):
        return f"{cell['n']} ({cell['rate']:.3f})"
    return f"{cell['n']} ({cell['rate']:.3f} [{lo:.3f}, {hi:.3f}])"


def write_variant_temperature_latex_table(
    variant_data: dict[str, dict], output_path: Path,
    caption: str = "Leniency gap, self-consistency flip rate, and full-response/resolved-class "
                   "disagreement counts (overall, pooled across task\\_types) per judge template "
                   "variant and temperature. Bracketed values are bootstrap 95\\% CIs. "
                   "``Full$\\uparrow$/resolved$\\downarrow$'' is the leniency-relevant disagreement "
                   "(full response credited, resolved class name would not have been); "
                   "``full$\\downarrow$/resolved$\\uparrow$'' is the reverse.",
    label: str = "tab:judge_template_sensitivity",
) -> None:
    """variant_data: {variant_key: {"leniency_by_temp": {temperature:
    compute_leniency_gap(...) result}, "flip_by_temp": {temperature:
    compute_flip_rate(...) result}, "disagreement_by_temp": {temperature:
    compute_disagreement_matrix(...) result}}} — as assembled by
    src/judge_template_sensitivity.py's orchestrator from each variant's
    own leniency_gap.json (which now also carries disagreement_by_temperature)
    and flip_rate.json. Writes a LaTeX booktabs table, one row per (variant,
    temperature), reporting the "overall" (pooled across task_types)
    leniency_gap, flip_rate, and the two cross-diagonal disagreement cells
    (n and rate) — the per-task_type breakdown and the full 2x2 (including
    both_correct/both_wrong) stay in the PNGs; this table is the compact
    appendix-ready cross-variant summary. disagreement_by_temp is optional
    (older leniency_gap.json files won't have it) — those columns render
    "--" rather than raising.
    """
    variant_labels = {k: lbl for k, _, _, lbl in TEMPLATE_VARIANTS}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Template variant & Temperature & Leniency gap [95\% CI] & Flip rate [95\% CI] "
        r"& Full$\uparrow$/resolved$\downarrow$ (n, rate) & Full$\downarrow$/resolved$\uparrow$ (n, rate) \\",
        r"\midrule",
    ]

    variant_keys = [k for k, _, _, _ in TEMPLATE_VARIANTS if k in variant_data]
    for vi, variant_key in enumerate(variant_keys):
        data = variant_data[variant_key]
        leniency_by_temp = data.get("leniency_by_temp", {})
        flip_by_temp = data.get("flip_by_temp", {})
        disagreement_by_temp = data.get("disagreement_by_temp", {})
        temperatures = sorted(set(leniency_by_temp) | set(flip_by_temp))
        n = len(temperatures)
        variant_label = variant_labels.get(variant_key, variant_key).replace("_", r"\_")
        for ti, temperature in enumerate(temperatures):
            variant_cell = rf"\multirow{{{n}}}{{*}}{{{variant_label}}}" if ti == 0 else ""
            leniency = leniency_by_temp.get(temperature, {}).get("overall", {})
            flip = flip_by_temp.get(temperature, {}).get("overall", {})
            disagreement = disagreement_by_temp.get(temperature, {}).get("overall", {})
            leniency_cell = _fmt_ci(leniency.get("leniency_gap"), leniency.get("leniency_gap_ci"))
            flip_cell = _fmt_ci(flip.get("flip_rate"), flip.get("flip_rate_ci"))
            leniency_dis_cell = _fmt_disagreement_cell(disagreement.get("full_correct_resolved_wrong"))
            strict_dis_cell = _fmt_disagreement_cell(disagreement.get("full_wrong_resolved_correct"))
            lines.append(f"{variant_cell} & T={temperature} & {leniency_cell} & {flip_cell} "
                         f"& {leniency_dis_cell} & {strict_dis_cell} \\\\")
        if vi < len(variant_keys) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def false_positive_credit_rate(
    rechecked_rows: list[dict], n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """{"full": {n, rate, rate_ci}, "answer_only": {...}, "gap", "gap_ci"}.

    rechecked_rows is a <variant>/leniency/rechecked_T<temp>.csv-shaped row
    list (has `correct` — the checkpoint's own deterministic resolve_class
    match — plus `judged_correct_full_trajectory` and `judged_correct`, the
    full-response and <answer>-tag-only judge verdicts at that temperature).
    Restricts to rows where `correct` says WRONG, then measures how often
    the judge credits them anyway — the false-positive rate specifically,
    isolated from the true-positive agreement cases that otherwise dilute
    leniency_gap's own all-rows average. gap/gap_ci is a PAIRED bootstrap
    (full vs. answer_only on the SAME rows, src.common.bootstrap.bootstrap_diff_ci)."""
    from src.common.bootstrap import bootstrap_diff_ci, bootstrap_mean_ci

    wrong_rows = [r for r in rechecked_rows if r.get("correct", "") not in ("1", 1, True)]
    n = len(wrong_rows)
    if n == 0:
        empty = {"n": 0, "rate": float("nan"), "rate_ci": (float("nan"), float("nan"))}
        return {"full": empty, "answer_only": empty, "gap": float("nan"), "gap_ci": (float("nan"), float("nan"))}

    full_vals = [int(r.get("judged_correct_full_trajectory", "") in ("1", 1, True)) for r in wrong_rows]
    answer_vals = [int(r.get("judged_correct", "") in ("1", 1, True)) for r in wrong_rows]
    full_rate, answer_rate = float(np.mean(full_vals)), float(np.mean(answer_vals))
    return {
        "full": {"n": n, "rate": round(full_rate, 4), "rate_ci": bootstrap_mean_ci(full_vals, n_boot, ci, seed)},
        "answer_only": {"n": n, "rate": round(answer_rate, 4),
                         "rate_ci": bootstrap_mean_ci(answer_vals, n_boot, ci, seed)},
        "gap": round(full_rate - answer_rate, 4),
        "gap_ci": bootstrap_diff_ci(full_vals, answer_vals, n_boot, ci, seed),
    }


def plot_leniency_and_false_positive_summary(
    leniency_by_variant: dict[str, dict], fp_by_variant: dict[str, dict], output_path: Path,
) -> None:
    """Two-panel line chart, one line per condition (full response /
    answer-only / gap) x panel, one point per variant (in TEMPLATE_VARIANTS
    order) — left panel is leniency_gap (overall judged_correct rate over
    ALL rows, from <variant>/leniency/leniency_gap.json), right panel is
    false_positive_credit_rate (judge credit rate restricted to
    deterministically-WRONG rows only). Same validated palette (blue/
    orange/aqua — see the dataviz skill's palette validator, checked for
    this exact triple) and above/below label-offset convention as every
    other per-variant comparison in this module."""
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

    variant_keys = [v[0] for v in TEMPLATE_VARIANTS
                     if v[0] in leniency_by_variant and v[0] in fp_by_variant]
    # Short variant KEYS (not the longer descriptive labels) on the x-axis —
    # with up to 4 variants on screen, the full labels ("Specificity + class
    # list (hybrid)") crowd/overlap each other; the keys stay legible.
    labels = variant_keys
    x = np.arange(len(variant_keys))

    lg_full = [leniency_by_variant[vk]["full_response_rate"] * 100 for vk in variant_keys]
    lg_answer = [leniency_by_variant[vk]["resolved_class_rate"] * 100 for vk in variant_keys]
    lg_gap = [leniency_by_variant[vk]["leniency_gap"] * 100 for vk in variant_keys]

    fp_full = [fp_by_variant[vk]["full"]["rate"] * 100 for vk in variant_keys]
    fp_answer = [fp_by_variant[vk]["answer_only"]["rate"] * 100 for vk in variant_keys]
    fp_gap = [fp_by_variant[vk]["gap"] * 100 for vk in variant_keys]

    fig, axes = plt.subplots(1, 2, figsize=(max(13, len(variant_keys) * 3.2), 5.5))
    panels = [
        (axes[0], lg_full, lg_answer, lg_gap, "leniency_gap (overall accuracy, all rows)",
         "% of all rows judged correct"),
        (axes[1], fp_full, fp_answer, fp_gap, "false-positive credit rate (deterministic-wrong rows only)",
         "% of deterministically-wrong rows credited"),
    ]
    for ax, full, answer, gap, title, ylabel in panels:
        ax.plot(x, full, marker="o", markersize=8, linewidth=2, color=BLUE, label="Full response")
        ax.plot(x, answer, marker="o", markersize=8, linewidth=2, color=ORANGE, label="Answer-only")
        ax.plot(x, gap, marker="o", markersize=8, linewidth=2, color=AQUA, linestyle="--",
                label="Gap (full − answer-only)")
        # Full's label sits above its marker, answer-only's sits below —
        # keeps the two from colliding where the lines converge.
        for xi, yi in zip(x, full):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=BLUE, fontweight="bold")
        for xi, yi in zip(x, answer):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=9, color=ORANGE, fontweight="bold")
        for xi, yi in zip(x, gap):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=AQUA, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(full + [1]) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=True, ncol=3)
    fig.suptitle("Leniency shrinks as judge template gets stricter — two views of the same trend",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_oracle_validation_csv(
    rows_by_variant: dict[str, list[dict]], output_path: Path,
    sample_size: int = 200, disagreement_frac: float = 0.7, seed: int = 42,
) -> dict:
    """Samples rows from BOTH rows_by_variant["hybrid"] and
    rows_by_variant["current"] (joined on agml_dataset/class_label/
    image_name) for a human to manually adjudicate as an oracle — up to
    `sample_size` rows, `disagreement_frac` of them from rows where the two
    variants' judged_correct DISAGREE (the informative cases for validating
    which template is actually right), the remainder a plain random sample
    regardless of agreement (base-rate context), stratified by task_type
    where the pool allows it. Returns {"n_disagreement", "n_agreement",
    "n_total"} for the caller to report; writes nothing and returns None if
    either variant is missing (caller should skip with a printed note, same
    pattern as plot_length_bias's "current" gate).

    Output columns: agml_dataset, class_label, image_name, task_type,
    ground_truth, prediction, extracted_answer, resolved_class, correct
    (deterministic), judged_correct_current, judge_reasoning_current,
    judged_correct_hybrid, judge_reasoning_hybrid, agreement, plus two
    BLANK columns for the reviewer: human_verdict, human_notes."""
    import random

    if "hybrid" not in rows_by_variant or "current" not in rows_by_variant:
        return None

    def _key(r):
        return (r.get("agml_dataset", ""), r.get("class_label", ""), r.get("image_name", ""))

    current_by_key = {_key(r): r for r in rows_by_variant["current"]}
    joined = []
    for hyb_row in rows_by_variant["hybrid"]:
        cur_row = current_by_key.get(_key(hyb_row))
        if cur_row is None:
            continue
        hyb_correct = hyb_row.get("judged_correct", "") in ("1", 1, True)
        cur_correct = cur_row.get("judged_correct", "") in ("1", 1, True)
        joined.append({
            "agml_dataset": hyb_row.get("agml_dataset", ""),
            "class_label": hyb_row.get("class_label", ""),
            "image_name": hyb_row.get("image_name", ""),
            "task_type": hyb_row.get("task_type", ""),
            "ground_truth": hyb_row.get("ground_truth", ""),
            "prediction": hyb_row.get("prediction", ""),
            "extracted_answer": hyb_row.get("extracted_answer", ""),
            "resolved_class": hyb_row.get("resolved_class", ""),
            "correct": hyb_row.get("correct", ""),
            "judged_correct_current": cur_row.get("judged_correct", ""),
            "judge_reasoning_current": cur_row.get("judge_reasoning", ""),
            "judged_correct_hybrid": hyb_row.get("judged_correct", ""),
            "judge_reasoning_hybrid": hyb_row.get("judge_reasoning", ""),
            "agreement": hyb_correct == cur_correct,
            "human_verdict": "",
            "human_notes": "",
        })

    rng = random.Random(seed)
    disagreement_pool = [r for r in joined if not r["agreement"]]
    agreement_pool = [r for r in joined if r["agreement"]]
    rng.shuffle(disagreement_pool)
    rng.shuffle(agreement_pool)

    n_disagreement = min(len(disagreement_pool), round(sample_size * disagreement_frac))
    n_agreement = min(len(agreement_pool), sample_size - n_disagreement)
    sample = disagreement_pool[:n_disagreement] + agreement_pool[:n_agreement]
    rng.shuffle(sample)

    fields = ["agml_dataset", "class_label", "image_name", "task_type", "ground_truth",
              "prediction", "extracted_answer", "resolved_class", "correct",
              "judged_correct_current", "judge_reasoning_current",
              "judged_correct_hybrid", "judge_reasoning_hybrid", "agreement",
              "human_verdict", "human_notes"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow(r)

    return {"n_disagreement": n_disagreement, "n_agreement": n_agreement, "n_total": len(sample)}
