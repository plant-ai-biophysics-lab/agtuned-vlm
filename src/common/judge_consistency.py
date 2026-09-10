"""Judge self-consistency (Phase 2 of the judge-improvements evaluation plan,
see the "LLM as a judge improvements" note).

Repeats the SAME judge call (judge_classification_template, pred_key=
"prediction" — the full model response, same as the checkpoint's original
judge pass) N times per row, at each of a swept set of temperatures, and
measures how often the verdict flips across those repeats. At temperature=0.0
(the pipeline's normal, fixed setting — see src.common.metrics._JUDGE_PARAMS)
vLLM decoding is greedy, so repeat calls are bit-for-bit identical and there
is nothing to measure; apply_judge's temperature/seed override (added
alongside this module) exists specifically so this sweep can force real
sampling variance.

Consistency is a DIFFERENT axis from Phase 1's leniency_gap — a judge can be
perfectly stable (flip_rate 0) and still systematically lenient (leniency_gap
large), or unstable and roughly unbiased on average. Kept as separate
metrics/figures; plot_leniency_vs_flip_rate below is the explicit place the
two get related to each other, once both are computed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.leniency_analysis import TASK_TYPE_LABELS, TASK_TYPE_ORDER

_FLIP_COLOR = "#9467bd"


def repeat_judge_calls(
    rows: list[dict],
    judge_llm,
    judge_tok,
    prompts: dict,
    confidence_threshold: int,
    chunk_size: int,
    temperatures: list[float],
    n_repeats: int,
    log_dir: Path | None = None,
    pred_key: str = "prediction",
    template_key: str = "judge_classification_template",
    base_seed: int = 0,
) -> dict[float, list[dict]]:
    """{temperature: rows}, each row a shallow copy of the input row plus
    "judged_correct_reps": [verdict_0, ..., verdict_{n_repeats-1}] (each an
    int 0/1, in repeat order) — the row's own agml_dataset/class_label/
    image_name/task_type carry through unchanged for grouping downstream.

    A different --seed per repeat (base_seed + repeat_idx) is passed to
    apply_judge so vLLM's sampler actually diverges call to call at a given
    temperature, rather than (at some vLLM/backend versions) reusing an
    internal default seed across separate .generate() calls.
    """
    from src.common.metrics import apply_judge
    from src.common.progress import ProgressBar

    out: dict[float, list[dict]] = {}
    for temperature in temperatures:
        reps_by_row_idx: list[list[int]] = [[] for _ in rows]
        pbar = ProgressBar(total=len(rows) * n_repeats, desc=f"Consistency @ T={temperature}")
        for rep_idx in range(n_repeats):
            log_path = None
            if log_dir is not None:
                log_path = log_dir / f"consistency_T{temperature}_rep{rep_idx}_judge_log.jsonl"
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i : i + chunk_size]
                judged_chunk = apply_judge(
                    rows=chunk, llm=judge_llm, tokenizer=judge_tok,
                    system_prompt=prompts.get("judge_system_vision", ""),
                    judge_template=prompts[template_key],
                    gt_key="judge_gt_hint", pred_key=pred_key,
                    confidence_threshold=confidence_threshold,
                    log_path=log_path,
                    temperature=temperature, seed=base_seed + rep_idx,
                )
                for j, r in enumerate(judged_chunk):
                    reps_by_row_idx[i + j].append(int(r["judged_correct"] or 0))
                pbar.update(len(chunk))
        pbar.finish()

        rows_with_reps = []
        for r, reps in zip(rows, reps_by_row_idx):
            r2 = dict(r)
            r2["judged_correct_reps"] = reps
            rows_with_reps.append(r2)
        out[temperature] = rows_with_reps
    return out


def compute_flip_rate(rows: list[dict], reps_key: str = "judged_correct_reps") -> dict:
    """{task_type: {n, flip_rate, mean_judged_correct}} + "overall".

    A row "flips" if its N repeat verdicts are not all identical
    (len(set(reps)) > 1) — the judge disagreed with itself at least once
    across the repeat calls, holding the row/prompt/temperature fixed.
    mean_judged_correct (mean over ALL reps of ALL rows, not just the
    per-row mode) is reported alongside flip_rate so a high flip_rate can be
    read against whether it's centered near the leniency_gap numbers or is
    just noise around a stable-on-average verdict.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    def _summarize(trows: list[dict]) -> dict:
        n = len(trows)
        if n == 0:
            return {"n": 0, "flip_rate": float("nan"), "mean_judged_correct": float("nan")}
        flips = sum(1 for r in trows if len(set(r[reps_key])) > 1)
        all_reps = [v for r in trows for v in r[reps_key]]
        return {
            "n": n,
            "flip_rate": round(flips / n, 4),
            "mean_judged_correct": round(float(np.mean(all_reps)), 4) if all_reps else float("nan"),
        }

    out = {task_type: _summarize(trows) for task_type, trows in by_task.items()}
    out["overall"] = _summarize(rows)
    return out


def plot_flip_rate_by_temperature(flip_rate_by_temp: dict[float, dict], output_path: Path) -> None:
    """Grouped bar chart: flip_rate per task_type (+ overall), one group of
    bars per temperature — shows whether instability grows with sampling
    temperature, and whether that growth is uniform across task_types."""
    import matplotlib.pyplot as plt

    temperatures = sorted(flip_rate_by_temp)
    any_result = flip_rate_by_temp[temperatures[0]]
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
        results = flip_rate_by_temp[temperature]
        vals = [results.get(t, {}).get("flip_rate", float("nan")) for t in task_types]
        offset = (i - (n_temps - 1) / 2) * width
        color = cmap(0.35 + 0.55 * (i / max(1, n_temps - 1)))
        ax.bar(x + offset, vals, width, label=f"T={temperature}", color=color)

    ax.set_xticks(x)
    ns = [any_result.get(t, {}).get("n", 0) for t in task_types]
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("Flip rate (fraction of rows with disagreement across repeats)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9, title="Judge temperature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_TASK_TYPE_MARKERS = {
    "disease": "o", "pest_damage": "s", "species_identification": "^",
    "quality_identification": "D", "weed_identification": "v",
}


def plot_leniency_vs_flip_rate(
    leniency_by_temp: dict[float, dict], flip_rate_by_temp: dict[float, dict], output_path: Path,
) -> dict | None:
    """Scatter: x = leniency_gap(T), y = flip_rate(T), one point per
    (task_type, temperature) — BOTH axes vary with temperature (leniency_gap
    must come from src/judge_leniency.py's --temperatures sweep, which
    re-judges both sides fresh at each T, not a single fixed-temperature
    value reused across every point).

    The hypothesis under test is per-temperature, not per-task_type: WITHIN
    a given temperature, does leniency_gap predict flip_rate across
    task_types? So each temperature gets its own color AND its own fitted
    trend line (not one pooled trend line across every point, and not a
    line connecting the same task_type across temperatures — task_type is
    encoded as marker SHAPE instead, via _TASK_TYPE_MARKERS, so which
    ag task falls where on a given temperature's line is visible without
    conflating "same task" with "same temperature" as the thing being
    correlated).

    Caveat worth keeping in mind reading this: there are only as many
    task_types as datasets.yaml defines (4-5) — each temperature's own
    trend line/correlation is fit on that few points, nowhere near enough
    for a reliable correlation coefficient. Treat this as a diagnostic/
    exploratory plot for THIS analysis, not a result to report a p-value
    on; the per-temperature Pearson r/p/n (in-plot text annotation, no
    title) is descriptive, not inferential. Returns
    {"T=<temp>": {"pearson_r", "p_value", "n"}, ...} for each temperature
    with enough points/variance to compute it — the caller can persist this
    alongside the figure since it isn't otherwise saved anywhere but the
    plot itself.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    temperatures = sorted(set(leniency_by_temp) & set(flip_rate_by_temp))
    missing = (set(leniency_by_temp) ^ set(flip_rate_by_temp))
    if missing:
        print(f"  [leniency_vs_flip_rate] temperature(s) {sorted(missing)} present in only one of "
              f"leniency_by_temp/flip_rate_by_temp — skipped (need both at the same T to plot a point).")
    cmap = plt.get_cmap("viridis")

    stats: dict[str, dict] = {}
    stats_lines = []
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    for i, temperature in enumerate(temperatures):
        leniency_results = leniency_by_temp[temperature]
        flip_results = flip_rate_by_temp[temperature]
        task_types = [t for t in leniency_results if t != "overall" and t in flip_results]
        tx = [leniency_results[t]["leniency_gap"] for t in task_types]
        ty = [flip_results[t]["flip_rate"] for t in task_types]
        color = cmap(i / max(1, len(temperatures) - 1))

        for t, xv, yv in zip(task_types, tx, ty):
            marker = _TASK_TYPE_MARKERS.get(t, "x")
            ax.scatter([xv], [yv], s=90, color=color, marker=marker, zorder=3,
                       edgecolors="black", linewidths=0.5)

        # Per-temperature trend line — the actual hypothesis being tested
        # (within one temperature, does leniency predict instability across
        # task_types), not a single line pooling every temperature together.
        if len(tx) >= 2 and np.std(tx) > 0:
            m, b = np.polyfit(tx, ty, 1)
            xline = np.linspace(min(tx), max(tx), 20)
            ax.plot(xline, m * xline + b, linestyle="--", color=color, linewidth=1.5, zorder=1,
                    label=f"T={temperature}")
        else:
            ax.plot([], [], linestyle="--", color=color, linewidth=1.5, label=f"T={temperature}")

        if len(tx) >= 3 and np.std(tx) > 0 and np.std(ty) > 0:
            r, p = pearsonr(tx, ty)
            stats[f"T={temperature}"] = {"pearson_r": round(float(r), 4), "p_value": round(float(p), 4), "n": len(tx)}
            stats_lines.append(f"T={temperature}: r={r:.2f} (p={p:.3f}, n={len(tx)})")

    if stats_lines:
        ax.text(0.03, 0.97, "\n".join(stats_lines) + "\ndescriptive only, not a formal test",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999", alpha=0.9))

    ax.margins(x=0.15, y=0.12)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("leniency_gap (full response judge - resolved class judge)")
    ax.set_ylabel("flip_rate (self-consistency)")

    temp_legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8,
                             title="Judge temperature (trend line color)", ncol=3)
    ax.add_artist(temp_legend)
    task_handles = [
        plt.Line2D([0], [0], marker=_TASK_TYPE_MARKERS.get(t, "x"), color="white",
                   markerfacecolor="#555", markeredgecolor="black", markersize=8, linestyle="none")
        for t in TASK_TYPE_ORDER if t in _TASK_TYPE_MARKERS
    ]
    task_labels = [TASK_TYPE_LABELS.get(t, t) for t in TASK_TYPE_ORDER if t in _TASK_TYPE_MARKERS]
    ax.legend(task_handles, task_labels, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              fontsize=8, title="Ag task (marker shape)", ncol=5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats or None
