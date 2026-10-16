"""Judge leniency baseline (Phase 1 of the judge-improvements evaluation plan,
see the "LLM as a judge improvements" note).

leniency_gap = P(judge is correct on the FULL model response)
             - P(judge is correct on JUST the extracted <answer> tag content)

Both sides are graded by the SAME judge model and template
(judge_classification_template) — the only thing that changes is how much
text the judge sees: the model's full free-text response (prediction), vs.
just the raw text between its own <answer></answer> tags (extract_answer_tag,
src/common/metrics.py). This isolates genuine judge leniency (does more
surrounding narrative make the judge more generous) from a DIFFERENT,
much larger effect: resolve_class's own string-matching limits.

An earlier version of this module fed the judge resolve_class's OUTPUT
(the checkpoint's resolved_class column — already canonicalized to a
literal ground-truth class name, or empty if no exact/unique-substring
match was found) instead of the raw tag content. That was wrong two ways:
(1) when resolve_class succeeds, resolved_class IS already one of the
literal class names, so judging "does this match the ground truth" is
close to redundant with the already-deterministic `correct` column; (2)
resolve_class's exact/single-substring matching fails on most naturally-
phrased answers regardless of correctness (observed ~83% unresolved on a
real pre-tagged-prompt checkpoint), discarding the vast majority of rows
before the judge ever saw them, and conflating "the judge is lenient" with
"resolve_class couldn't parse the response" — two different problems.
Feeding the raw <answer> tag content instead fixes both: it's the
model's own genuinely-reduced statement (not a re-canonicalized version of
the ground truth), and every row with a parseable <answer> tag qualifies,
not just the much smaller set resolve_class happens to match.

Only rows with a parseable <answer> tag can be included — there's nothing
reduced to re-judge otherwise (extract_answer_tag falls back to the full
cleaned text when no tag is found, which would make this comparison
trivial). See missing_answer_tag_report / write_missing_answer_tag_report
for how many rows that excludes, which is necessary context for reading
the leniency_gap number (with the tagged classification prompt, this
should be a small fraction — unlike the old resolve_class-based
exclusion).

Re-judging the extracted answer needs real judge-LLM calls — this is NOT a
pure-CPU step (see src/judge_leniency.py, which loads the judge model and
calls src.common.ablation_judge_recheck.rejudge_condition(pred_key=
"resolved_class") — the row field name is unchanged, but judge_leniency.py
now populates it with extract_answer_tag's output rather than the
checkpoint's own resolve_class-matched value — before handing rejudged
rows to compute_leniency_gap here).
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


def compute_disagreement_matrix(
    rows: list[dict],
    full_key: str = "judged_correct_full_trajectory",
    resolved_key: str = "judged_correct",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """2x2 breakdown of the SAME rows compute_leniency_gap summarizes into
    one net rate — the raw disagreement counts, in both directions, rather
    than just their difference. leniency_gap only reports
    mean(full) - mean(resolved); a row-level breakdown is needed to answer
    "how many/which rows does this actually happen on," which is what the
    aggregate rate can't show on its own.

    Returns {task_type: {n, both_correct, both_wrong,
    full_correct_resolved_wrong, full_wrong_resolved_correct}} + "overall".
    Each of the four cells is {"n": count, "rate": fraction of n,
    "rate_ci": bootstrap 95% CI on that fraction}.

    full_correct_resolved_wrong is the LENIENCY-relevant disagreement — the
    judge credits the full response but would not have credited the bare
    resolved class name; this is the direction "leniency_gap > 0" is made
    of. full_wrong_resolved_correct is the reverse (the judge is actually
    STRICTER on the full response than on the class name alone) — real on
    pest_damage in the pre-tagged-prompt baseline, where leniency_gap came
    out negative; reporting only the leniency-direction cell would hide
    that this can go either way.
    """
    from src.common.bootstrap import bootstrap_mean_ci

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    def _cell(trows: list[dict], full_val: int, resolved_val: int) -> dict:
        flags = [
            int(int(r.get(full_key, "") in ("1", 1, True)) == full_val
                and int(r.get(resolved_key, "") in ("1", 1, True)) == resolved_val)
            for r in trows
        ]
        n = len(trows)
        rate = float(np.mean(flags)) if n else float("nan")
        return {
            "n": int(sum(flags)),
            "rate": round(rate, 4),
            "rate_ci": bootstrap_mean_ci(flags, n_boot, ci, seed),
        }

    def _summarize(trows: list[dict]) -> dict:
        return {
            "n": len(trows),
            "both_correct": _cell(trows, 1, 1),
            "both_wrong": _cell(trows, 0, 0),
            "full_correct_resolved_wrong": _cell(trows, 1, 0),
            "full_wrong_resolved_correct": _cell(trows, 0, 1),
        }

    out = {task_type: _summarize(trows) for task_type, trows in by_task.items()}
    out["overall"] = _summarize(rows)
    return out


def disagreement_examples(
    rows: list[dict], cell: str = "full_correct_resolved_wrong",
    full_key: str = "judged_correct_full_trajectory", resolved_key: str = "judged_correct",
    n: int = 20,
) -> list[dict]:
    """Up to `n` raw example rows from one disagreement cell (default: the
    leniency-relevant one) — for a spot-check appendix table or manual
    review, not aggregated. cell is one of "both_correct", "both_wrong",
    "full_correct_resolved_wrong", "full_wrong_resolved_correct"."""
    targets = {
        "both_correct": (1, 1), "both_wrong": (0, 0),
        "full_correct_resolved_wrong": (1, 0), "full_wrong_resolved_correct": (0, 1),
    }
    full_val, resolved_val = targets[cell]
    matches = [
        r for r in rows
        if int(r.get(full_key, "") in ("1", 1, True)) == full_val
        and int(r.get(resolved_key, "") in ("1", 1, True)) == resolved_val
    ]
    return matches[:n]


def plot_disagreement_matrix(results: dict, output_path: Path) -> None:
    """100%-stacked bar chart per task_type (+ overall): the four
    compute_disagreement_matrix cells as fractions of that task_type's
    rows, so the disagreement slices' relative size is visible directly
    (not just their raw rate against the same [0,1] axis every other plot
    in this module uses)."""
    import matplotlib.pyplot as plt

    task_types = [t for t in TASK_TYPE_ORDER if t in results]
    task_types += [t for t in results if t not in TASK_TYPE_ORDER and t != "overall"]
    task_types.append("overall")
    labels = [TASK_TYPE_LABELS.get(t, t) if t != "overall" else "Overall" for t in task_types]
    ns = [results[t]["n"] for t in task_types]

    cells = [
        ("both_correct", "Both correct", "#2ca02c"),
        ("both_wrong", "Both wrong", "#7f7f7f"),
        ("full_correct_resolved_wrong", "Full correct, <answer>-tag wrong (leniency)", _FULL_RESPONSE_COLOR),
        ("full_wrong_resolved_correct", "Full wrong, <answer>-tag correct (stricter)", _RESOLVED_CLASS_COLOR),
    ]

    x = np.arange(len(task_types))
    fig, ax = plt.subplots(figsize=(max(7, len(task_types) * 1.6), 5.5))
    bottoms = np.zeros(len(task_types))
    for cell_key, cell_label, color in cells:
        vals = np.array([results[t][cell_key]["rate"] for t in task_types])
        ax.bar(x, vals, bottom=bottoms, width=0.6, label=cell_label, color=color)
        for xi, v, b in zip(x, vals, bottoms):
            if v > 0.03:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("Fraction of rows")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def taggable_rows_with_extracted_answer(rows: list[dict], pred_key: str = "prediction") -> list[dict]:
    """Returns a NEW list containing only rows where row[pred_key] has a
    parseable <answer>...</answer> tag (extract_answer_tag's found=True) —
    each entry a shallow copy with row["resolved_class"] set to the raw tag
    content. Shared by Phase 1 (src/judge_leniency.py) and Phase 2
    (src/judge_consistency.py) so both filter/populate identically; a row
    with no tag falls back to the full cleaned text, which would make the
    "reduced information" comparison trivial (both sides nearly identical),
    so such rows are excluded rather than included with a degraded value.

    "resolved_class" here means the raw <answer> tag content, NOT
    resolve_class's class-list-matched output — see this module's top
    docstring for why."""
    from src.common.metrics import extract_answer_tag

    out = []
    for r in rows:
        answer, found = extract_answer_tag(r.get(pred_key, ""))
        if found:
            r2 = dict(r)
            r2["resolved_class"] = answer
            out.append(r2)
    return out


def missing_answer_tag_report(rows: list[dict], pred_key: str = "prediction") -> dict:
    """{task_type: {n_total, n_missing, pct_missing}} + "overall" — same
    shape as resolve_failure_report, but the exclusion criterion is
    different: how many rows have NO <answer>...</answer> tag at all in
    row[pred_key] (extract_answer_tag's `found` flag is False), rather than
    how many rows resolve_class failed to match against the class list.

    This is the criterion src/judge_leniency.py actually uses to decide
    which rows are eligible for the leniency comparison — see that
    module's docstring for why the judge is fed the RAW <answer> tag
    content (not resolve_class's class-list-matched output): matching
    against resolve_class conflates judge leniency with resolve_class's own
    (much larger) vocabulary-matching gap, and discards the vast majority
    of rows before the judge ever sees them.
    """
    from src.common.metrics import extract_answer_tag

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    def _count(trows: list[dict]) -> tuple[int, int]:
        n_total = len(trows)
        n_missing = sum(1 for r in trows if not extract_answer_tag(r.get(pred_key, ""))[1])
        return n_total, n_missing

    out: dict[str, dict] = {}
    for task_type, trows in by_task.items():
        n_total, n_missing = _count(trows)
        out[task_type] = {
            "n_total": n_total, "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n_total, 2) if n_total else 0.0,
        }
    n_total, n_missing = _count(rows)
    out["overall"] = {
        "n_total": n_total, "n_missing": n_missing,
        "pct_missing": round(100 * n_missing / n_total, 2) if n_total else 0.0,
    }
    return out


def write_missing_answer_tag_report(report: dict, output_path: Path, model_label: str = "") -> None:
    lines = []
    if model_label:
        lines.append(f"missing <answer> tag report — {model_label}")
        lines.append("=" * (len(lines[0])))
        lines.append("")
    lines.append("How many predictions had NO <answer>...</answer> tag at all — these are")
    lines.append("excluded from the leniency_gap comparison entirely, since there's no")
    lines.append("extracted answer text to re-judge against the full response.")
    lines.append("")
    header = f"{'task_type':<26}{'n_total':>10}{'n_missing':>12}{'pct_missing':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    ordered = [t for t in TASK_TYPE_ORDER if t in report] + \
              [t for t in report if t not in TASK_TYPE_ORDER and t != "overall"]
    for task_type in ordered:
        d = report[task_type]
        lines.append(f"{task_type:<26}{d['n_total']:>10}{d['n_missing']:>12}{d['pct_missing']:>13.2f}%")
    lines.append("-" * len(header))
    d = report["overall"]
    lines.append(f"{'overall':<26}{d['n_total']:>10}{d['n_missing']:>12}{d['pct_missing']:>13.2f}%")

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
    b1 = ax.bar(x - width / 2, resolved, width, label="Judge on <answer> tag only",
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
