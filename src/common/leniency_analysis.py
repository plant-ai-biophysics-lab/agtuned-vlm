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


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([int(r.get(key, "") in ("1", 1, True)) for r in rows]))


def compute_leniency_gap(
    rows: list[dict],
    full_key: str = "judged_correct_full_trajectory",
    resolved_key: str = "judged_correct",
) -> dict:
    """`rows` must already be the output of
    src.common.ablation_judge_recheck.rejudge_condition(pred_key="resolved_class"),
    restricted beforehand to rows with a non-empty resolved_class (rejudge_condition
    stores the pre-rejudge verdict under `full_key` and overwrites `judged_correct`
    with the resolved_class-only verdict, i.e. `resolved_key`).

    Returns {task_type: {n, full_response_rate, resolved_class_rate, leniency_gap}}
    plus an "overall" key pooling every task_type together.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    out: dict[str, dict] = {}
    for task_type, trows in by_task.items():
        full = _rate(trows, full_key)
        resolved = _rate(trows, resolved_key)
        out[task_type] = {
            "n": len(trows),
            "full_response_rate": round(full, 4),
            "resolved_class_rate": round(resolved, 4),
            "leniency_gap": round(full - resolved, 4),
        }
    full_all = _rate(rows, full_key)
    resolved_all = _rate(rows, resolved_key)
    out["overall"] = {
        "n": len(rows),
        "full_response_rate": round(full_all, 4),
        "resolved_class_rate": round(resolved_all, 4),
        "leniency_gap": round(full_all - resolved_all, 4),
    }
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
    two bars in each group IS leniency_gap."""
    import matplotlib.pyplot as plt

    task_types = [t for t in TASK_TYPE_ORDER if t in results]
    task_types += [t for t in results if t not in TASK_TYPE_ORDER and t != "overall"]
    labels = [TASK_TYPE_LABELS.get(t, t) for t in task_types] + ["Overall"]
    task_types = task_types + ["overall"]

    full = [results[t]["full_response_rate"] for t in task_types]
    resolved = [results[t]["resolved_class_rate"] for t in task_types]
    gaps = [results[t]["leniency_gap"] for t in task_types]
    ns = [results[t]["n"] for t in task_types]

    x = np.arange(len(task_types))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * 1.6), 5.5))
    b1 = ax.bar(x - width / 2, resolved, width, label="Judge on resolved class only",
                color=_RESOLVED_CLASS_COLOR)
    b2 = ax.bar(x + width / 2, full, width, label="Judge on full model response",
                color=_FULL_RESPONSE_COLOR)

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
