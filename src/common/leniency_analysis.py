"""Judge leniency baseline (Phase 1 of the judge-improvements evaluation plan,
see the "LLM as a judge improvements" note).

leniency_gap = P(judge is correct on the FULL model response)
             - P(judge is correct on JUST the resolved class string)

Both sides are graded by the SAME judge model and template
(judge_classification_template) — the only thing that changes is how much
text the judge sees: the model's full free-text response (prediction), vs.
just the class name resolve_class already extracted from it (resolved_class,
already computed at generation time by vision_baseline.py / self_cot.py /
cot_yaml.py / desc_augmented_vision_v2.py). This isolates genuine judge
leniency (does more surrounding narrative make the judge more generous) from
resolve_class's own string-matching limits — comparing the judge against
resolve_class directly (an earlier version of this module did that) instead
conflated the two, since resolve_class's exact/single-substring matching
fails on most naturally-phrased answers regardless of correctness.

Only resolvable rows (resolved_class non-empty) can be included — there's no
"resolved class" to re-judge otherwise. See resolve_failure_report /
write_resolve_failure_report for how many rows that excludes, which is
necessary context for reading the leniency_gap number (a small gap over a
small, cherry-picked resolvable subset means less than the same gap over
most of the dataset).

Re-judging the resolved_class field needs real judge-LLM calls — this is
NOT a pure-CPU step (see src/judge_leniency.py, which loads the judge model
and calls src.common.ablation_judge_recheck.rejudge_condition(pred_key=
"resolved_class") before handing rejudged rows to compute_leniency_gap
here).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

# Canonical task_type display order + color — extend here if a new task_type
# is added to datasets.yaml. Kept separate from method_comparison.py's
# CONDITION_STYLES since this module compares two judge PASSES within one
# condition/model, not across conditions.
TASK_TYPE_ORDER = [
    "disease", "pest_damage", "species_identification",
    "quality_identification", "weed_identification",
]
TASK_TYPE_LABELS = {
    "disease": "Disease", "pest_damage": "Pest/Damage",
    "species_identification": "Species ID", "quality_identification": "Quality ID",
    "weed_identification": "Weed ID",
}

_FULL_RESPONSE_COLOR = "#d62728"    # judge on full response — the leniency risk
_RESOLVED_CLASS_COLOR = "#1f77b4"   # judge on resolved_class only — the stricter reference




def _vals(rows: list[dict], key: str) -> list[int]:
    return [int(r.get(key, "") in ("1", 1, True)) for r in rows]


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean(_vals(rows, key)))


def compute_leniency_gap(
    rows: list[dict],
    full_key: str = "judged_correct_full_trajectory",
    resolved_key: str = "judged_correct",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """`rows` must already be the output of
    src.common.ablation_judge_recheck.rejudge_condition(pred_key="resolved_class"),
    restricted beforehand to rows with a non-empty resolved_class (rejudge_condition
    stores the pre-rejudge verdict under `full_key` and overwrites `judged_correct`
    with the resolved_class-only verdict, i.e. `resolved_key`).

    Returns {task_type: {n, full_response_rate, resolved_class_rate,
    leniency_gap, full_response_rate_ci, resolved_class_rate_ci,
    leniency_gap_ci}} plus an "overall" key pooling every task_type
    together. The *_ci fields are (lo, hi) bootstrap 95% CIs (see
    src.common.bootstrap) — leniency_gap_ci is a PAIRED bootstrap (same
    resampled row indices on both sides), not two independent CIs
    subtracted, since full_key/resolved_key are two verdicts on the SAME
    row.
    """
    from src.common.bootstrap import bootstrap_diff_ci, bootstrap_mean_ci

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    def _summarize(trows: list[dict]) -> dict:
        full_vals = _vals(trows, full_key)
        resolved_vals = _vals(trows, resolved_key)
        full = float(np.mean(full_vals)) if full_vals else float("nan")
        resolved = float(np.mean(resolved_vals)) if resolved_vals else float("nan")
        return {
            "n": len(trows),
            "full_response_rate": round(full, 4),
            "resolved_class_rate": round(resolved, 4),
            "leniency_gap": round(full - resolved, 4),
            "full_response_rate_ci": bootstrap_mean_ci(full_vals, n_boot, ci, seed),
            "resolved_class_rate_ci": bootstrap_mean_ci(resolved_vals, n_boot, ci, seed),
            "leniency_gap_ci": bootstrap_diff_ci(full_vals, resolved_vals, n_boot, ci, seed),
        }

    out = {task_type: _summarize(trows) for task_type, trows in by_task.items()}
    out["overall"] = _summarize(rows)
    return out


def resolve_failure_report(rows: list[dict]) -> dict:
    """{task_type: {n_total, n_unresolved, pct_unresolved}} + "overall".

    n_unresolved = rows where resolve_class found no single matching class
    at generation time (row["resolvable"] == "0" or row["resolved_class"]
    is empty) — these can't be included in compute_leniency_gap at all,
    since there's no resolved class string to re-judge. Reported separately
    so the leniency_gap number can be read against how much of the dataset
    it actually covers.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    def _count(trows: list[dict]) -> tuple[int, int]:
        n_total = len(trows)
        n_unresolved = sum(1 for r in trows if str(r.get("resolvable", "0")) != "1" or not r.get("resolved_class"))
        return n_total, n_unresolved

    out: dict[str, dict] = {}
    for task_type, trows in by_task.items():
        n_total, n_unresolved = _count(trows)
        out[task_type] = {
            "n_total": n_total, "n_unresolved": n_unresolved,
            "pct_unresolved": round(100 * n_unresolved / n_total, 2) if n_total else 0.0,
        }
    n_total, n_unresolved = _count(rows)
    out["overall"] = {
        "n_total": n_total, "n_unresolved": n_unresolved,
        "pct_unresolved": round(100 * n_unresolved / n_total, 2) if n_total else 0.0,
    }
    return out


def write_resolve_failure_report(report: dict, output_path: Path, model_label: str = "") -> None:
    lines = []
    if model_label:
        lines.append(f"resolve_class failure report — {model_label}")
        lines.append("=" * (len(lines[0])))
        lines.append("")
    lines.append("How many predictions resolve_class could NOT match to exactly one")
    lines.append("class name (row['resolvable'] == 0 / resolved_class empty) — these are")
    lines.append("excluded from the leniency_gap comparison entirely, since there's no")
    lines.append("resolved class string to re-judge against the full response.")
    lines.append("")
    header = f"{'task_type':<26}{'n_total':>10}{'n_unresolved':>14}{'pct_unresolved':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    ordered = [t for t in TASK_TYPE_ORDER if t in report] + \
              [t for t in report if t not in TASK_TYPE_ORDER and t != "overall"]
    for task_type in ordered:
        d = report[task_type]
        lines.append(f"{task_type:<26}{d['n_total']:>10}{d['n_unresolved']:>14}{d['pct_unresolved']:>15.2f}%")
    lines.append("-" * len(header))
    d = report["overall"]
    lines.append(f"{'overall':<26}{d['n_total']:>10}{d['n_unresolved']:>14}{d['pct_unresolved']:>15.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def plot_leniency_gap(results: dict, output_path: Path) -> None:
    """Grouped bar chart: judge-on-full-response rate vs. judge-on-resolved-
    class rate, one group per task_type (+ overall) — the gap between the
    two bars in each group IS leniency_gap. Error bars are bootstrap 95%
    CIs (compute_leniency_gap's *_rate_ci fields) when present."""
    import matplotlib.pyplot as plt

    from src.common.bootstrap import yerr_pairs

    task_types = [t for t in TASK_TYPE_ORDER if t in results]
    task_types += [t for t in results if t not in TASK_TYPE_ORDER and t != "overall"]
    labels = [TASK_TYPE_LABELS.get(t, t) for t in task_types] + ["Overall"]
    task_types = task_types + ["overall"]

    full = [results[t]["full_response_rate"] for t in task_types]
    resolved = [results[t]["resolved_class_rate"] for t in task_types]
    gaps = [results[t]["leniency_gap"] for t in task_types]
    ns = [results[t]["n"] for t in task_types]
    full_yerr = yerr_pairs(results, task_types, "full_response_rate", "full_response_rate_ci")
    resolved_yerr = yerr_pairs(results, task_types, "resolved_class_rate", "resolved_class_rate_ci")

    x = np.arange(len(task_types))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * 1.6), 5.5))
    b1 = ax.bar(x - width / 2, resolved, width, label="Judge on resolved class only",
                color=_RESOLVED_CLASS_COLOR, yerr=resolved_yerr, capsize=3, ecolor="#333")
    b2 = ax.bar(x + width / 2, full, width, label="Judge on full model response",
                color=_FULL_RESPONSE_COLOR, yerr=full_yerr, capsize=3, ecolor="#333")

    for xi, gap, fv in zip(x, gaps, full):
        if not np.isnan(gap):
            ax.annotate(f"+{gap:.2f}" if gap >= 0 else f"{gap:.2f}",
                        xy=(xi + width / 2, fv), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color=_FULL_RESPONSE_COLOR if gap > 0.02 else "#444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("Judge accuracy")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leniency_gap_by_temperature(results_by_temp: dict[float, dict], output_path: Path) -> None:
    """Grouped bar chart: leniency_gap per task_type (+ overall), one group
    of bars per temperature — the per-temperature counterpart to
    plot_leniency_gap (which shows one temperature's full_response vs.
    resolved_class rates directly; this shows how the GAP itself moves
    across temperatures). Unlike plot_flip_rate_by_temperature's [0, 1]
    axis, leniency_gap is signed (a task_type can show the judge being
    STRICTER on the full response than on the resolved class alone, a
    negative gap — seen for real on pest_damage in the pre-tagged-prompt
    baseline), so the y-axis is not clipped to non-negative. Error bars are
    bootstrap 95% CIs (leniency_gap_ci) when present."""
    import matplotlib.pyplot as plt

    from src.common.bootstrap import yerr_pairs

    temperatures = sorted(results_by_temp)
    any_result = results_by_temp[temperatures[0]]
    task_types = [t for t in TASK_TYPE_ORDER if t in any_result]
    task_types += [t for t in any_result if t not in TASK_TYPE_ORDER and t != "overall"]
    task_types.append("overall")
    labels = [TASK_TYPE_LABELS.get(t, t) if t != "overall" else "Overall" for t in task_types]

    x = np.arange(len(task_types))
    n_temps = len(temperatures)
    width = 0.8 / n_temps
    cmap = plt.get_cmap("Reds")

    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * 1.6), 5.5))
    for i, temperature in enumerate(temperatures):
        results = results_by_temp[temperature]
        vals = [results.get(t, {}).get("leniency_gap", float("nan")) for t in task_types]
        yerr = yerr_pairs(results, task_types, "leniency_gap", "leniency_gap_ci")
        offset = (i - (n_temps - 1) / 2) * width
        color = cmap(0.35 + 0.55 * (i / max(1, n_temps - 1)))
        ax.bar(x + offset, vals, width, label=f"T={temperature}", color=color,
               yerr=yerr, capsize=2, ecolor="#333", error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ns = [any_result.get(t, {}).get("n", 0) for t in task_types]
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("leniency_gap (full response - resolved class)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="best", fontsize=9, title="Judge temperature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
