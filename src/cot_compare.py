"""
Stage 3 — CoT / description method comparison (no GPU, no model needed).

For every model that has a vision-baseline (lower bound) run, compares each
of desc-aug-v2, self-cot, and cot-yaml against it, per dataset, using the
exact same per-dataset metrics vision_baseline.py / self_cot.py /
cot_yaml.py already write to their own metrics.json (via
collate_models()/compute_model_metrics() in src/common/vision_eval.py —
never a separate scoring pass). Per method, reports:

  win_rate   — fraction of datasets where the method beats the baseline
  mean_gain  — mean F1 delta among datasets where it won
  mean_loss  — mean F1 delta (negative) among datasets where it lost
  top N gains / top N losses — the individual datasets driving the above

The delta metric is judge_f1_macro where the judge phase has run, falling
back to f1_macro per dataset otherwise (same fallback cot_step_analysis.py's
select_datasets_delta used).

Also writes, per model, a plain space-separated dataset-name file combining
that model's cot-yaml top-N gains and top-N losses — paste straight into
cot_yaml.py's --swap_datasets to re-run those specific datasets with
per-step + swap-order robustness testing:
    uv run python main.py cot-yaml --model <model> \\
        --swap_datasets $(cat outputs_final/cot_compare/<model>/cot_yaml_swap_datasets.txt)

Outputs (into --output_dir/<model>/):
  method_comparison.csv/.json   — win_rate, mean_gain, mean_loss per method
  top_movers.csv                 — every method's top-N gain/loss datasets,
                                    PER TASK TYPE (not pooled across the model)
  method_comparison.png          — win rate / mean gain & loss bar charts
  cot_yaml_swap_datasets.txt     — cot-yaml's top-N gain + top-N loss dataset
                                    names, space-separated

Usage (run once every model has finished vision-baseline + whichever of
desc-aug-v2/self-cot/cot-yaml you want compared):
    uv run python main.py cot-compare
    uv run python main.py cot-compare --top_n 5
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from src.common.plot_style import family_shaded_colors, ordered_backbones
from src.common.vision_eval import (
    collate_models,
    compute_condition_stats,
    compute_delta_stats,
    compute_model_metrics,
    load_all_rows,
    load_ckpt,
    per_class_judge_f1_delta,
)

METHOD_DIRS_DEFAULT = {
    # desc_augmented_vision_v2.py's actual --output_dir (per
    # vision_baseline_config.yaml's aug_output_dir) has no "_v2" suffix —
    # only the script's filename does. Pointing here at "..._v2/" silently
    # found zero rows and dropped desc-aug from every comparison without
    # any error (collate_models() just returns {} for a dir with no
    # per-model subfolders).
    "desc-aug":  "outputs_final/desc_augmented_vision/",
    "self-cot":  "outputs_final/self_cot/",
    "cot-yaml":  "outputs_final/cot_yaml/",
}
METHOD_COLOR = {"desc-aug": "#ff7f0e", "self-cot": "#2ca02c", "cot-yaml": "#9467bd"}
# Display names for plot_summary's x-axis — the raw METHOD_DIRS_DEFAULT keys
# above (hyphenated, internal) aren't the paper-facing names used everywhere
# else in this script (_ALL_CONDITIONS's "Self CoT"/"Manual CoT" labels).
METHOD_DISPLAY = {"desc-aug": "Desc-Aug", "self-cot": "Self CoT", "cot-yaml": "Manual CoT"}

# All four conditions on one chart — same (dataset dir key, display label)
# shape as vision_baseline_collate.py's _CONDITIONS, extended with the two
# CoT methods. "Self CoT" / "Manual CoT" are the paper-facing names for
# self_cot.py ("CoT Self Description") and cot_yaml.py ("CoT User Defined")
# requested for this chart specifically.
_ALL_CONDITIONS = [
    ("baseline", "Lower Bound"),
    ("desc_aug", "Upper Bound"),
    ("self_cot", "Self CoT"),
    ("cot_yaml", "Manual CoT"),
]
# One hatch per condition (not per vision-only-ness like linear probe, and
# not just "solid vs one hatch" like the 2-condition stage2_f1_by_task.png —
# four conditions need four distinguishable fills). None of these reuse "//"
# (linear probe's vision-only marker) or clash with each other.
_CONDITION_HATCH = {"baseline": None, "desc_aug": "--", "self_cot": "..", "cot_yaml": "xx"}

# cot_all_conditions_by_task.png (Gemma-4 only, delta view): bars are
# Self CoT / Manual CoT judge-F1 delta vs the lower bound; the lower and
# upper bounds themselves become horizontal dotted reference lines instead
# of bars (0 by construction for the lower bound, the desc-aug delta —
# the ceiling — for the upper bound).
_DELTA_CONDITIONS = [("self_cot", "Self CoT"), ("cot_yaml", "Manual CoT")]
_BOUND_LINE_COLOR = {"desc_aug": "#d62728", "baseline": "#1f77b4"}  # red = upper bound, blue = lower bound
# Self CoT / Manual CoT fill colors for the three delta-distribution charts
# (bar/violin/boxplot) — yellow/orange, requested specifically so the two
# conditions read apart from each other without relying on hatch alone.
# Only ever looked up for these two condition keys (see the `.get(...) or
# model_colors[model]` fallback at each call site) — the 3-condition
# verifier charts (_VERIFIER_CONDITIONS) fall through to their existing
# per-model/hatch coloring untouched.
_DELTA_CONDITION_COLORS = {"self_cot": "#FDD835", "cot_yaml": "#E06B00"}
_DELTA_FILL_ALPHA = 0.55

# Stage 4 — round-based verifier belief tracking (src/verifier_cot.py) and
# its two baselines: verifier_cot.py's own --baseline argmax ablation, and
# the full PPT tournament from src/verifier_lmv.py (PPT-Direct-N10 — the
# N=10 condition since that's the closest equal-budget comparison to
# verifier_cot's typical total candidate count across a full 3-10 round
# run). Per docs/verifier_score_belief_update.md Sec. 16. Same lower/
# upper-bound reference-line convention as _DELTA_CONDITIONS above; plotted
# via the same two functions, just with a 3-condition set instead of 2 —
# see plot_gemma_cot_deltas/plot_gemma_cot_f1_by_task's
# `conditions`/`hatch_map`/`output_name` params.
_VERIFIER_CONDITIONS = [
    ("verifier_belief", "Verifier-Belief"),
    ("verifier_argmax", "Verifier-Argmax"),
    ("ppt_direct_n10",  "PPT-Direct-N10"),
]
_VERIFIER_HATCH = {"verifier_belief": "\\\\", "verifier_argmax": "oo", "ppt_direct_n10": "**"}


def collate_all_conditions(
    baseline_dir: Path, aug_dir: Path, self_cot_dir: Path, cot_yaml_dir: Path, n_boot: int, seed: int,
) -> dict[str, dict[str, dict]]:
    """{model: {condition_key: {scope: stats}}} across all four conditions,
    via load_all_rows()/compute_condition_stats() — the same bootstrap-CI
    machinery vision_baseline_collate.py's stage2_f1_by_task.png uses (NOT
    collate_models()'s point-estimate-only metrics, which is what the rest
    of this script uses for the win-rate/gain/loss comparison) — so this
    chart's error bars are directly comparable to the other two."""
    dirs = {"baseline": baseline_dir, "desc_aug": aug_dir, "self_cot": self_cot_dir, "cot_yaml": cot_yaml_dir}
    rows_by_cond = {cond: load_all_rows(Path(d)) for cond, d in dirs.items()}
    models = sorted(set().union(*(set(r) for r in rows_by_cond.values())))

    result: dict[str, dict[str, dict]] = {}
    for model in models:
        result[model] = {}
        for cond, rows_map in rows_by_cond.items():
            if model in rows_map:
                result[model][cond] = compute_condition_stats(rows_map[model], n_boot, seed)
    return result


def collate_gemma_cot_deltas(
    baseline_dir: Path, aug_dir: Path, self_cot_dir: Path, cot_yaml_dir: Path, n_boot: int, seed: int,
) -> dict[str, dict[str, dict]]:
    """Gemma-4 models only: {model: {condition_key: {scope: delta_stats}}}
    where condition_key is one of "desc_aug" / "self_cot" / "cot_yaml" and
    delta_stats is (condition - baseline) via compute_delta_stats's paired
    per-class bootstrap (see bootstrap_judge_f1_delta_ci) — NOT two
    independently-bootstrapped condition_stats CIs subtracted. desc_aug's
    delta is the ceiling plotted as the upper-bound reference line; the
    lower bound is baseline itself, whose delta from itself is trivially 0."""
    dirs = {"desc_aug": aug_dir, "self_cot": self_cot_dir, "cot_yaml": cot_yaml_dir}
    baseline_rows_map = load_all_rows(Path(baseline_dir))
    rows_by_cond = {cond: load_all_rows(Path(d)) for cond, d in dirs.items()}
    # Only the 26B-A4B-it run — the only model this CoT sweep was actually
    # run against (E4B-it's self_cot/cot_yaml data present under
    # outputs_final/ is not from that validated run).
    models = sorted(m for m in baseline_rows_map if "26b" in m.lower() and "gemma" in m.lower())

    result: dict[str, dict[str, dict]] = {}
    for model in models:
        base_rows = baseline_rows_map[model]
        result[model] = {}
        for cond, rows_map in rows_by_cond.items():
            if model in rows_map:
                result[model][cond] = compute_delta_stats(base_rows, rows_map[model], n_boot, seed)
    return result


def collate_gemma_cot_delta_distributions(
    baseline_dir: Path, aug_dir: Path, self_cot_dir: Path, cot_yaml_dir: Path,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Gemma-4 models only: {model: {condition_key: {scope: per-class delta
    array}}} — the same raw per-class (condition - baseline) judge-F1
    deltas collate_gemma_cot_deltas bootstrap-means into a point + CI
    (per_class_judge_f1_delta), kept here as the full array instead —
    feeds plot_gemma_cot_delta_violin, the distribution counterpart to
    plot_gemma_cot_deltas's bars."""
    dirs = {"desc_aug": aug_dir, "self_cot": self_cot_dir, "cot_yaml": cot_yaml_dir}
    baseline_rows_map = load_all_rows(Path(baseline_dir))
    rows_by_cond = {cond: load_all_rows(Path(d)) for cond, d in dirs.items()}
    models = sorted(m for m in baseline_rows_map if "26b" in m.lower() and "gemma" in m.lower())

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model in models:
        base_rows = baseline_rows_map[model]
        base_by_task: dict[str, list[dict]] = defaultdict(list)
        for r in base_rows:
            base_by_task[r["task_type"]].append(r)
        result[model] = {}
        for cond, rows_map in rows_by_cond.items():
            if model not in rows_map:
                continue
            cond_rows = rows_map[model]
            cond_by_task: dict[str, list[dict]] = defaultdict(list)
            for r in cond_rows:
                cond_by_task[r["task_type"]].append(r)
            by_scope: dict[str, np.ndarray] = {}
            _, by_scope["overall"] = per_class_judge_f1_delta(base_rows, cond_rows)
            for tt in set(base_by_task) | set(cond_by_task):
                _, by_scope[tt] = per_class_judge_f1_delta(base_by_task.get(tt, []), cond_by_task.get(tt, []))
            result[model][cond] = by_scope
    return result


def load_verifier_cot_rows(output_dir: Path, baseline: str) -> dict[str, list[dict]]:
    """{model: [row, ...]} for src/verifier_cot.py's --baseline
    {belief,argmax} output — flat <output_dir>/<model>/ckpt_<baseline>_
    <model>.csv layout, filtered to is_final_round rows only (one row per
    image, matching every other condition's shape here) since
    verifier_cot.py's checkpoint has one row per (image, round). Needs
    VERIFIER_STEP_FIELDS, not plain CKPT_FIELDS, to align columns
    correctly — see verifier_cot.py's module docstring for the schema."""
    from src.verifier_cot import VERIFIER_STEP_FIELDS
    if not output_dir.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_path = model_dir / f"ckpt_{baseline}_{model_dir.name}.csv"
        rows, _ = load_ckpt(ckpt_path, fields=VERIFIER_STEP_FIELDS)
        final_rows = [r for r in rows if str(r.get("is_final_round")) == "1"]
        if final_rows:
            result[model_dir.name] = final_rows
    return result


def load_verifier_lmv_rows(ckpt_judged_path: Path, gen_mode: str, n_max: int) -> list[dict]:
    """One (gen_mode, n_max) condition slice from src/verifier_lmv.py's
    ckpt_judged.csv — the full PPT-tournament baseline (arXiv:2607.05391),
    scored via the corrected two-pass prefill mechanism (see that module's
    docstring). Unlike every other condition here, verifier_lmv.py's
    --output_dir is flat (one model's worth of output per run, no <model>/
    subfolder) — point --verifier_lmv_ckpt at whichever model's run you
    want compared; its rows get attached to that model in
    collate_verifier_conditions/collate_verifier_deltas below. gen_mode:
    "direct" | "self_cot". No resolved_class column in this schema, so the
    strict f1_macro reads as 0 — only judge_f1_macro is meaningful, which
    is what this whole pipeline already prefers everywhere else."""
    from src.verifier_lmv import JUDGED_FIELDS
    if not ckpt_judged_path.exists():
        return []
    rows, _ = load_ckpt(ckpt_judged_path, fields=JUDGED_FIELDS)
    return [r for r in rows if r.get("gen_mode") == gen_mode and str(r.get("n_max")) == str(n_max)]


def collate_verifier_conditions(
    verifier_dir: Path, verifier_lmv_ckpt: Path, n_boot: int, seed: int,
) -> dict[str, dict[str, dict]]:
    """{model: {condition_key: {scope: stats}}} for the Stage 4 verifier
    conditions (belief-tracking, its argmax ablation, and the full-PPT
    baseline) — same compute_condition_stats machinery as
    collate_all_conditions, feeding plot_gemma_cot_f1_by_task via
    _VERIFIER_CONDITIONS/_VERIFIER_HATCH (docs/verifier_score_belief_update.md
    Sec. 16). PPT-Direct-N10 rows get attached to every model verifier_cot's
    own data identifies (in practice just gemma-4-26B-A4B-it, per this
    project's established single-model scoping for this whole line of
    work) rather than auto-discovered, since verifier_lmv.py's flat output
    doesn't self-identify a model."""
    belief_rows = load_verifier_cot_rows(verifier_dir, "belief")
    argmax_rows = load_verifier_cot_rows(verifier_dir, "argmax")
    ppt_rows = load_verifier_lmv_rows(verifier_lmv_ckpt, "direct", 10)

    result: dict[str, dict[str, dict]] = {}
    for model in sorted(set(belief_rows) | set(argmax_rows)):
        by_cond: dict[str, dict] = {}
        if model in belief_rows:
            by_cond["verifier_belief"] = compute_condition_stats(belief_rows[model], n_boot, seed)
        if model in argmax_rows:
            by_cond["verifier_argmax"] = compute_condition_stats(argmax_rows[model], n_boot, seed)
        if ppt_rows:
            by_cond["ppt_direct_n10"] = compute_condition_stats(ppt_rows, n_boot, seed)
        if by_cond:  # skip entirely rather than leaving an empty {model: {}} entry —
            result[model] = by_cond  # a truthy-but-empty dict would make `if verifier_condition_stats:`
    return result  # (main()'s no-op guard) fire even when neither script has ever been run.


def collate_verifier_deltas(
    baseline_dir: Path, verifier_dir: Path, verifier_lmv_ckpt: Path, n_boot: int, seed: int,
) -> dict[str, dict[str, dict]]:
    """Gemma-4-26B-A4B-it only, (condition - baseline) judge-F1 delta for
    the Stage 4 verifier conditions — same paired-bootstrap machinery as
    collate_gemma_cot_deltas, feeding plot_gemma_cot_deltas via
    _VERIFIER_CONDITIONS/_VERIFIER_HATCH."""
    baseline_rows_map = load_all_rows(baseline_dir)
    belief_rows = load_verifier_cot_rows(verifier_dir, "belief")
    argmax_rows = load_verifier_cot_rows(verifier_dir, "argmax")
    ppt_rows = load_verifier_lmv_rows(verifier_lmv_ckpt, "direct", 10)

    models = sorted(m for m in baseline_rows_map if "26b" in m.lower() and "gemma" in m.lower())
    result: dict[str, dict[str, dict]] = {}
    for model in models:
        base_rows = baseline_rows_map[model]
        by_cond: dict[str, dict] = {}
        if model in belief_rows:
            by_cond["verifier_belief"] = compute_delta_stats(base_rows, belief_rows[model], n_boot, seed)
        if model in argmax_rows:
            by_cond["verifier_argmax"] = compute_delta_stats(base_rows, argmax_rows[model], n_boot, seed)
        if ppt_rows:
            by_cond["ppt_direct_n10"] = compute_delta_stats(base_rows, ppt_rows, n_boot, seed)
        if by_cond:  # same empty-dict guard as collate_verifier_conditions above
            result[model] = by_cond
    return result


def write_all_conditions_summary(output_dir: Path, condition_stats: dict[str, dict[str, dict]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cot_all_conditions_summary.json").write_text(json.dumps(condition_stats, indent=2))

    fieldnames = ["model", "condition", "scope", "f1_macro", "ci_low", "ci_high", "n_classes",
                  "judge_f1_macro", "judge_ci_low", "judge_ci_high", "judge_n_classes", "n"]
    with open(output_dir / "cot_all_conditions_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, by_cond in condition_stats.items():
            for cond, by_scope in by_cond.items():
                for scope, stats in by_scope.items():
                    writer.writerow({"model": model, "condition": cond, "scope": scope, **stats})
    print(f"  Saved: {output_dir / 'cot_all_conditions_summary.csv'}")
    print(f"  Saved: {output_dir / 'cot_all_conditions_summary.json'}")


def plot_gemma_cot_deltas(
    output_dir: Path, delta_stats: dict[str, dict[str, dict]],
    conditions: list[tuple[str, str]] | None = None,
    hatch_map: dict[str, str | None] | None = None,
    condition_colors: dict[str, str] | None = None,
    output_name: str = "cot_all_conditions_by_task.png",
) -> None:
    """Gemma-4 models only. Bars: judge-F1 delta vs the lower bound for
    every condition in `conditions` (defaults to _DELTA_CONDITIONS — Self
    CoT / Manual CoT), per task type + overall. The lower and upper bounds
    themselves are drawn as horizontal dotted reference lines spanning each
    model's bar cluster — blue at 0 (lower bound: no improvement over
    baseline) and red at the desc-aug delta (upper bound / ceiling) —
    rather than as their own bars, since on a delta axis the lower bound is
    trivially 0 and the upper bound is just another delta value to compare
    the bars against. `conditions`/`hatch_map`/`output_name` let this same
    function drive a second chart for a different condition set (e.g. the
    Stage 4 verifier conditions, _VERIFIER_CONDITIONS/_VERIFIER_HATCH) —
    see docs/verifier_score_belief_update.md Sec. 16. `condition_colors`
    (defaults to _DELTA_CONDITION_COLORS — yellow Self CoT / orange Manual
    CoT) colors each condition directly instead of shading by model; a
    condition key missing from it (e.g. the 3-way verifier set, which has
    no entry here) falls back to the model's own shaded color, unaffected."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    conditions = conditions or _DELTA_CONDITIONS
    hatch_map = hatch_map or _CONDITION_HATCH
    condition_colors = _DELTA_CONDITION_COLORS if condition_colors is None else condition_colors

    models = [m for m in ordered_backbones(list(delta_stats.keys())) if "26b" in m.lower() and "gemma" in m.lower()]
    if not models:
        print("  No Gemma-4-26B-A4B-it data to plot.")
        return

    scopes_present = {
        scope for by_cond in delta_stats.values() for by_scope in by_cond.values() for scope in by_scope
    }
    task_types = sorted(s for s in scopes_present if s != "overall")
    scopes = task_types + ["overall"]

    model_colors = family_shaded_colors(models)
    n_cond = len(conditions)
    n_bars = len(models) * n_cond
    width = 0.92 / n_bars
    x = np.arange(len(scopes))

    fig, ax = plt.subplots(figsize=(max(9, len(scopes) * 2.4), 5.5))
    bar_i = 0
    max_top, min_bottom = 0.0, 0.0
    any_missing = False
    legend_handles, legend_labels = [], []
    for model in models:
        model_start = bar_i
        for cond_key, cond_label in conditions:
            by_scope = delta_stats[model].get(cond_key, {})
            if not by_scope:
                # No data at all for this (model, condition) — e.g. cot_yaml
                # was only run for one model. No legend placeholder needed
                # (the row-major grid layout that required one is gone now
                # that the legend sits inside the chart instead of below it).
                bar_i += 1
                continue
            means, lo_err, hi_err = [], [], []
            for scope in scopes:
                stats = by_scope.get(scope)
                if stats and stats.get("delta_n_classes", 0) > 0:
                    m, lo, hi = stats["delta_f1_macro"], stats["delta_ci_low"], stats["delta_ci_high"]
                else:
                    m = lo = hi = 0.0
                    any_missing = True
                means.append(m)
                lo_err.append(max(0.0, m - lo))
                hi_err.append(max(0.0, hi - m))
                max_top = max(max_top, hi)
                min_bottom = min(min_bottom, lo)

            offset = (bar_i - (n_bars - 1) / 2) * width
            label = f"{model.split('/')[-1]} ({cond_label})"
            hatch = hatch_map[cond_key]
            color = condition_colors.get(cond_key) or model_colors[model]
            bars = ax.bar(
                x + offset, means, width, label=label,
                color=color, hatch=hatch, alpha=_DELTA_FILL_ALPHA,
                edgecolor="white" if hatch else "black", linewidth=0.5,
                yerr=[lo_err, hi_err], capsize=2,
                error_kw={"elinewidth": 0.8, "alpha": 0.7}, zorder=2,
            )
            legend_handles.append(bars)
            legend_labels.append(label)
            bar_i += 1
        model_end = bar_i

        # Horizontal dotted reference line (upper bound / ceiling only —
        # the lower bound is dropped: on a delta-vs-lower-bound axis it's
        # trivially 0 everywhere, already legible from the y-axis itself)
        # spanning this model's full bar cluster, one per scope.
        lo_offset = (model_start - (n_bars - 1) / 2) * width - width / 2
        hi_offset = (model_end - 1 - (n_bars - 1) / 2) * width + width / 2
        aug_by_scope = delta_stats[model].get("desc_aug", {})
        for i, scope in enumerate(scopes):
            xmin, xmax = x[i] + lo_offset, x[i] + hi_offset
            aug_stats = aug_by_scope.get(scope)
            if aug_stats and aug_stats.get("delta_n_classes", 0) > 0:
                ceiling = aug_stats["delta_f1_macro"]
                ax.hlines(ceiling, xmin, xmax, colors=_BOUND_LINE_COLOR["desc_aug"],
                          linestyles=":", linewidth=1.8, zorder=3)
                max_top = max(max_top, ceiling)
                min_bottom = min(min_bottom, ceiling)

    if any_missing:
        print("  [warn] some (model, condition, scope) combinations had no paired judged classes — "
              "plotted their delta as 0")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                       rotation=15, ha="right")
    pad = max((max_top - min_bottom) * 0.15, 0.05)
    ax.set_ylim(min_bottom - pad, max_top + pad)
    ax.set_ylabel("Δ Judged F1 Macro (vs Lower Bound)")

    bound_handles = [
        Line2D([0], [0], color=_BOUND_LINE_COLOR["desc_aug"], linestyle=":", linewidth=1.8, label="Upper Bound (mean)"),
    ]
    full_handles = legend_handles + bound_handles
    full_labels = legend_labels + ["Upper Bound (mean)"]
    ax.legend(full_handles, full_labels, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = output_dir / output_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _half_violin(ax, x_center: float, values: np.ndarray, side: int, color: str, half_width: float) -> bool:
    """Draw one half of a split violin at x_center via a Gaussian KDE —
    side=-1 mirrors it to the left, side=+1 to the right — plus a short
    horizontal tick at the mean (matching the reference style: a plain
    filled half-teardrop with a mean marker, no whiskers/extrema). Returns
    False (draws nothing) if there are too few points for a KDE to mean
    anything, same "< 2" guard bootstrap_judge_f1_delta_ci itself uses."""
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.std(values) == 0:
        return False
    kde = gaussian_kde(values)
    y_min, y_max = float(values.min()), float(values.max())
    pad = (y_max - y_min) * 0.1 or abs(y_min) * 0.1 or 0.01
    ys = np.linspace(y_min - pad, y_max + pad, 200)
    density = kde(ys)
    density = density / density.max() * half_width
    ax.fill_betweenx(ys, x_center, x_center + side * density,
                      facecolor=color, edgecolor="black", linewidth=0.8, alpha=_DELTA_FILL_ALPHA, zorder=2)
    mean_y = float(values.mean())
    mean_half_width = float(np.interp(mean_y, ys, density))
    ax.plot([x_center, x_center + side * mean_half_width], [mean_y, mean_y],
            color="black", linewidth=1.8, zorder=3)
    return True


def plot_gemma_cot_delta_violin(
    output_dir: Path, distributions: dict[str, dict[str, dict[str, np.ndarray]]],
    conditions: list[tuple[str, str]] | None = None,
    condition_colors: dict[str, str] | None = None,
    output_name: str = "cot_all_conditions_by_task_violin.png",
) -> None:
    """Distribution counterpart to plot_gemma_cot_deltas — the same
    per-class judge-F1 deltas the bars collapse to a bootstrap mean + CI
    (collate_gemma_cot_delta_distributions), shown here as a SPLIT violin
    per scope instead: left half = conditions[0]'s distribution, right
    half = conditions[1]'s, sharing one x position (not two side-by-side
    violins) — exactly two conditions only, since a split violin has
    exactly two halves; pass a 2-entry `conditions` or don't call this.
    The upper bound (desc_aug's ceiling, already present in `distributions`
    even though it's not one of the two split halves) is kept as a
    horizontal dashed reference line per scope, same red as every other
    chart's Upper Bound."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    conditions = conditions or _DELTA_CONDITIONS
    if len(conditions) != 2:
        raise ValueError(f"split-violin plotting needs exactly 2 conditions, got {len(conditions)}")
    (cond0_key, cond0_label), (cond1_key, cond1_label) = conditions
    condition_colors = condition_colors or _DELTA_CONDITION_COLORS

    models = [m for m in ordered_backbones(list(distributions.keys())) if "26b" in m.lower() and "gemma" in m.lower()]
    if not models:
        print("  No Gemma-4-26B-A4B-it data to plot.")
        return

    scopes_present = {
        scope for by_cond in distributions.values() for by_scope in by_cond.values() for scope in by_scope
    }
    task_types = sorted(s for s in scopes_present if s != "overall")
    scopes = task_types + ["overall"]
    x = np.arange(len(scopes))

    group_width = 0.7
    model_width = group_width / len(models)
    half_width = model_width * 0.42

    fig, ax = plt.subplots(figsize=(max(9, len(scopes) * 2.2), 5.5))
    for model_i, model in enumerate(models):
        x_centers = x - group_width / 2 + model_width * (model_i + 0.5)
        by_cond = distributions[model]
        for i, scope in enumerate(scopes):
            xc = x_centers[i]
            left = by_cond.get(cond0_key, {}).get(scope, np.array([]))
            right = by_cond.get(cond1_key, {}).get(scope, np.array([]))
            _half_violin(ax, xc, left, -1, condition_colors[cond0_key], half_width)
            _half_violin(ax, xc, right, +1, condition_colors[cond1_key], half_width)

            ceiling_vals = by_cond.get("desc_aug", {}).get(scope, np.array([]))
            if ceiling_vals.size > 0:
                ceiling = float(ceiling_vals.mean())
                ax.hlines(ceiling, xc - model_width / 2, xc + model_width / 2,
                          colors=_BOUND_LINE_COLOR["desc_aug"], linestyles="--", linewidth=1.5, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                       rotation=15, ha="right")
    ax.set_ylabel("Δ Judged F1 (per class, vs Lower Bound)")

    legend_handles = [
        Patch(facecolor=condition_colors[cond0_key], edgecolor="black", alpha=_DELTA_FILL_ALPHA, label=cond0_label),
        Patch(facecolor=condition_colors[cond1_key], edgecolor="black", alpha=_DELTA_FILL_ALPHA, label=cond1_label),
        Line2D([0], [0], color=_BOUND_LINE_COLOR["desc_aug"], linestyle="--", linewidth=1.5, label="Upper Bound (mean)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = output_dir / output_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gemma_cot_delta_boxplot(
    output_dir: Path, distributions: dict[str, dict[str, dict[str, np.ndarray]]],
    conditions: list[tuple[str, str]] | None = None,
    condition_colors: dict[str, str] | None = None,
    ylim: tuple[float, float] | None = (-0.5, 0.5),
    output_name: str = "cot_all_conditions_by_task_boxplot.png",
) -> None:
    """Third distribution view of the same per-class judge-F1 deltas
    (collate_gemma_cot_delta_distributions) — a classic box-and-whisker per
    (model, condition) per scope, side-by-side within each scope like the
    bars (plot_gemma_cot_deltas), not split like the violin
    (plot_gemma_cot_delta_violin). Median/IQR/whiskers/outlier points
    instead of a KDE shape or a bootstrap mean+CI — purely a different way
    to look at the same numbers, not a different metric. Solid fills, no
    hatch (unlike the bar/violin charts) — color alone is enough to tell
    the two conditions apart here since there's no split/overlapping shape
    to disambiguate. `ylim` fixes the y-axis range so it's comparable across
    repeated regenerations instead of autoscaling to whatever the current
    data's extremes happen to be; pass None to autoscale."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    conditions = conditions or _DELTA_CONDITIONS
    condition_colors = _DELTA_CONDITION_COLORS if condition_colors is None else condition_colors

    models = [m for m in ordered_backbones(list(distributions.keys())) if "26b" in m.lower() and "gemma" in m.lower()]
    if not models:
        print("  No Gemma-4-26B-A4B-it data to plot.")
        return

    scopes_present = {
        scope for by_cond in distributions.values() for by_scope in by_cond.values() for scope in by_scope
    }
    task_types = sorted(s for s in scopes_present if s != "overall")
    scopes = task_types + ["overall"]
    x = np.arange(len(scopes))

    model_colors = family_shaded_colors(models)
    n_cond = len(conditions)
    n_bars = len(models) * n_cond
    width = 0.92 / n_bars

    fig, ax = plt.subplots(figsize=(max(9, len(scopes) * 2.2), 5.5))
    legend_handles, legend_labels = [], []
    bar_i = 0
    for model in models:
        model_start = bar_i
        for cond_key, cond_label in conditions:
            by_scope = distributions[model].get(cond_key, {})
            offset = (bar_i - (n_bars - 1) / 2) * width
            positions, data = [], []
            for i, scope in enumerate(scopes):
                arr = by_scope.get(scope, np.array([]))
                if arr.size >= 1:
                    positions.append(x[i] + offset)
                    data.append(arr)
            bar_i += 1
            if not data:
                continue

            color = condition_colors.get(cond_key) or model_colors[model]
            ax.boxplot(
                data, positions=positions, widths=width * 0.85, patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.3},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
                boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 0.6,
                          "alpha": _DELTA_FILL_ALPHA},
            )
            legend_handles.append(Patch(facecolor=color, edgecolor="black", alpha=_DELTA_FILL_ALPHA))
            legend_labels.append(f"{model.split('/')[-1]} ({cond_label})")
        model_end = bar_i

        # Upper Bound reference line spanning this model's bar cluster, one
        # per scope — same convention as plot_gemma_cot_deltas.
        lo_offset = (model_start - (n_bars - 1) / 2) * width - width / 2
        hi_offset = (model_end - 1 - (n_bars - 1) / 2) * width + width / 2
        aug_by_scope = distributions[model].get("desc_aug", {})
        for i, scope in enumerate(scopes):
            arr = aug_by_scope.get(scope, np.array([]))
            if arr.size > 0:
                ceiling = float(arr.mean())
                ax.hlines(ceiling, x[i] + lo_offset, x[i] + hi_offset,
                          colors=_BOUND_LINE_COLOR["desc_aug"], linestyles="--", linewidth=1.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                       rotation=15, ha="right")
    ax.set_ylabel("Δ Judged F1 (per class, vs Lower Bound)")
    if ylim is not None:
        ax.set_ylim(*ylim)

    bound_handles = [
        Line2D([0], [0], color=_BOUND_LINE_COLOR["desc_aug"], linestyle="--", linewidth=1.5, label="Upper Bound (mean)"),
    ]
    ax.legend(legend_handles + bound_handles, legend_labels + ["Upper Bound (mean)"],
              loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = output_dir / output_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gemma_cot_f1_by_task(
    output_dir: Path, condition_stats: dict[str, dict[str, dict]],
    conditions: list[tuple[str, str]] | None = None,
    hatch_map: dict[str, str | None] | None = None,
    output_name: str = "cot_gemma_f1_by_task.png",
) -> None:
    """Gemma-4-26B-A4B-it only — the non-delta companion to
    plot_gemma_cot_deltas: every condition in `conditions` (defaults to
    _DELTA_CONDITIONS — Self CoT / Manual CoT) plotted as raw Judged F1
    Macro bars, with Lower Bound / Upper Bound still drawn as horizontal
    dotted reference lines (same red/blue convention) rather than their own
    bars, so both figures read as a matched pair. `conditions`/`hatch_map`/
    `output_name` mirror plot_gemma_cot_deltas's — see its docstring."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    conditions = conditions or _DELTA_CONDITIONS
    hatch_map = hatch_map or _CONDITION_HATCH

    models = [m for m in ordered_backbones(list(condition_stats.keys())) if "26b" in m.lower() and "gemma" in m.lower()]
    if not models:
        print("  No Gemma-4-26B-A4B-it data to plot.")
        return
    model = models[0]
    by_cond = condition_stats[model]

    scopes_present = {scope for by_scope in by_cond.values() for scope in by_scope}
    task_types = sorted(s for s in scopes_present if s != "overall")
    scopes = task_types + ["overall"]

    def _val(stats: dict | None) -> tuple[float, float, float, bool]:
        # Same judge-first-with-strict-fallback logic as plot_f1_by_task.
        if stats and stats.get("judge_n_classes", 0) > 0:
            return stats["judge_f1_macro"], stats["judge_ci_low"], stats["judge_ci_high"], False
        elif stats:
            return stats["f1_macro"], stats["ci_low"], stats["ci_high"], True
        return 0.0, 0.0, 0.0, False

    model_color = family_shaded_colors(models)[model]
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = np.arange(len(scopes))

    fig, ax = plt.subplots(figsize=(max(9, len(scopes) * 2.4), 5.5))
    bar_i = 0
    max_top = 0.0
    any_unjudged = False
    legend_handles, legend_labels = [], []
    for cond_key, cond_label in conditions:
        by_scope = by_cond.get(cond_key, {})
        if not by_scope:
            bar_i += 1
            continue
        means, lo_err, hi_err = [], [], []
        for scope in scopes:
            m, lo, hi, unjudged = _val(by_scope.get(scope))
            any_unjudged = any_unjudged or unjudged
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))
            max_top = max(max_top, hi)

        offset = (bar_i - (n_cond - 1) / 2) * width
        hatch = hatch_map[cond_key]
        bars = ax.bar(
            x + offset, means, width, label=cond_label,
            color=model_color, hatch=hatch,
            edgecolor="white" if hatch else "black", linewidth=0.5,
            yerr=[lo_err, hi_err], capsize=2,
            error_kw={"elinewidth": 0.8, "alpha": 0.7}, zorder=2,
        )
        legend_handles.append(bars)
        legend_labels.append(cond_label)
        bar_i += 1

    # Horizontal dotted reference lines spanning the full bar cluster width,
    # one pair per scope: blue at the lower-bound (baseline) F1, red at the
    # upper-bound (desc-aug) F1.
    lo_offset = -width * n_cond / 2
    hi_offset = width * n_cond / 2
    baseline_by_scope = by_cond.get("baseline", {})
    aug_by_scope = by_cond.get("desc_aug", {})
    for i, scope in enumerate(scopes):
        xmin, xmax = x[i] + lo_offset, x[i] + hi_offset
        base_m, _, base_hi, base_unjudged = _val(baseline_by_scope.get(scope))
        any_unjudged = any_unjudged or base_unjudged
        ax.hlines(base_m, xmin, xmax, colors=_BOUND_LINE_COLOR["baseline"],
                  linestyles=":", linewidth=1.8, zorder=3)
        max_top = max(max_top, base_hi)
        aug_m, _, aug_hi, aug_unjudged = _val(aug_by_scope.get(scope))
        any_unjudged = any_unjudged or aug_unjudged
        ax.hlines(aug_m, xmin, xmax, colors=_BOUND_LINE_COLOR["desc_aug"],
                  linestyles=":", linewidth=1.8, zorder=3)
        max_top = max(max_top, aug_hi)

    if any_unjudged:
        print("  [warn] some (condition, scope) combinations had no judged rows — "
              "plotted their strict f1_macro instead (unjudged data, not a judge_f1_macro of 0)")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                       rotation=15, ha="right")
    ax.set_ylim(0, max(max_top * 1.15, 0.05))
    ax.set_ylabel("Judged F1 Macro")

    bound_handles = [
        Line2D([0], [0], color=_BOUND_LINE_COLOR["desc_aug"], linestyle=":", linewidth=1.8, label="Upper Bound (mean)"),
        Line2D([0], [0], color=_BOUND_LINE_COLOR["baseline"], linestyle=":", linewidth=1.8, label="Lower Bound"),
    ]
    ax.legend(legend_handles + bound_handles, legend_labels + ["Upper Bound (mean)", "Lower Bound"],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(legend_handles) + 2, fontsize=8)
    plt.tight_layout()
    path = output_dir / output_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Datasets whose ground-truth label is a specific cultivar/variety code
# rather than a species/common name (e.g. "BD70", "Akca Pear", "Amrapali")
# — identified by name since AgML doesn't tag this distinction itself, and
# this is exactly the split plot_species_identification_breakdown uses to
# show why species_identification's aggregate F1 is so much lower than the
# other task types (see the analysis behind this: cultivar ID from a photo
# alone is a much harder, more specialized task than species ID).
_VARIETY_KEYWORDS = ("variety", "varitey", "cultivar")


def collate_species_breakdown(baseline_dir: Path) -> dict | None:
    """Per-dataset accuracy + class count within species_identification for
    gemma-4-26B-A4B-it's lower bound (vision_baseline.py) — the raw material
    for plot_species_identification_breakdown's diagnostic chart. Returns
    None if that model has no baseline data."""
    rows_map = load_all_rows(Path(baseline_dir))
    model = next((m for m in rows_map if "26b" in m.lower() and "gemma" in m.lower()), None)
    if model is None:
        return None
    rows = [r for r in rows_map[model] if r.get("task_type") == "species_identification"]
    if not rows:
        return None

    metrics = compute_model_metrics(rows)
    by_dataset_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dataset_rows[r["agml_dataset"]].append(r)

    by_dataset = {}
    for ds, ds_rows in by_dataset_rows.items():
        m = dict(metrics["by_dataset"][ds])
        m["n_classes"] = len({r["ground_truth"] for r in ds_rows})
        m["is_variety"] = any(kw in ds.lower() for kw in _VARIETY_KEYWORDS)
        by_dataset[ds] = m
    return {"model": model, "by_dataset": by_dataset}


def plot_species_identification_breakdown(output_dir: Path, breakdown: dict) -> None:
    """Per-dataset judged accuracy within species_identification, sorted
    ascending, colored by whether the dataset's ground truth is a
    species/common name (blue) or a fine-grained cultivar/variety code
    (red) — visualizes why this task type's aggregate F1 is so much lower
    than disease/pest/quality: it's dominated by cultivar-level datasets
    (e.g. "BD70", "Akca Pear") that require domain expertise no
    general-purpose VLM was trained on, not by a model or pipeline defect."""
    import matplotlib.pyplot as plt

    by_dataset = breakdown["by_dataset"]

    def _acc(m: dict) -> float:
        return m.get("judge_accuracy", m.get("accuracy", 0.0))

    items = sorted(by_dataset.items(), key=lambda kv: _acc(kv[1]))
    names = [k for k, _ in items]
    accs = [_acc(v) for _, v in items]
    n_classes = [v["n_classes"] for _, v in items]
    is_variety = [v["is_variety"] for _, v in items]

    variety_color, species_color = "#d62728", "#1f77b4"
    colors = [variety_color if v else species_color for v in is_variety]

    fig, ax = plt.subplots(figsize=(9, max(6, len(names) * 0.32)))
    y = np.arange(len(names))
    ax.barh(y, accs, color=colors, edgecolor="black", linewidth=0.5, zorder=2)
    for yi, acc, n in zip(y, accs, n_classes):
        ax.text(acc + 0.01, yi, f"n={n}", va="center", fontsize=7, color="dimgray")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Judged Accuracy")
    ax.axvline(0, color="black", linewidth=0.8)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=variety_color, edgecolor="black", label="Cultivar / variety ID (e.g. \"BD70\", \"Akca Pear\")"),
            Patch(facecolor=species_color, edgecolor="black", label="Species / common-name ID (e.g. \"Aloe vera\")"),
        ],
        loc="lower right", fontsize=8,
    )
    plt.tight_layout()
    path = output_dir / "cot_gemma_species_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _f1(m: dict) -> float | None:
    v = m.get("judge_f1_macro")
    if v is None:
        v = m.get("f1_macro")
    return float(v) if v is not None else None


def dataset_task_map(rows: list[dict]) -> dict[str, str]:
    """agml_dataset -> task_type, from raw checkpoint rows. compute_deltas
    needs this separately because compute_model_metrics()'s by_dataset
    summaries only carry classification/rouge/judge metrics — task_type
    itself is never in there, which is why it silently came out blank in
    every top_movers.csv row until this was added."""
    m: dict[str, str] = {}
    for r in rows:
        m.setdefault(r["agml_dataset"], r.get("task_type", ""))
    return m


def compute_deltas(
    baseline_by_ds: dict[str, dict], method_by_ds: dict[str, dict], ds_task: dict[str, str],
) -> dict[str, dict]:
    """{agml_dataset: {baseline_f1, method_f1, delta, task_type}} for every
    dataset scored by both — datasets only one side ran aren't comparable."""
    out = {}
    for ds, base_m in baseline_by_ds.items():
        if ds not in method_by_ds:
            continue
        base_f1, meth_f1 = _f1(base_m), _f1(method_by_ds[ds])
        if base_f1 is None or meth_f1 is None:
            continue
        out[ds] = {
            "baseline_f1": base_f1, "method_f1": meth_f1, "delta": meth_f1 - base_f1,
            "task_type": ds_task.get(ds, ""),
        }
    return out


def summarize_method(deltas: dict[str, dict], top_n: int) -> dict:
    """top_gains_by_task / top_losses_by_task rank datasets WITHIN each
    task_type separately (top-N gain + top-N loss per task), not pooled
    across the whole model — a global top-N was letting task types with
    small deltas everywhere (e.g. species_identification, whose CoT effect
    is close to 0 across the board — see cot_gemma_f1_by_task.png) get
    crowded out entirely by task types with a few huge movers, so
    --swap_datasets (built from this) never got any species coverage."""
    if not deltas:
        return {"n_datasets": 0, "win_rate": None, "mean_gain": None, "mean_loss": None,
                "top_gains_by_task": {}, "top_losses_by_task": {}}
    wins = {ds: d for ds, d in deltas.items() if d["delta"] > 0}
    losses = {ds: d for ds, d in deltas.items() if d["delta"] < 0}

    by_task: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for ds, d in deltas.items():
        by_task[d["task_type"]].append((ds, d))

    top_gains_by_task: dict[str, list[dict]] = {}
    top_losses_by_task: dict[str, list[dict]] = {}
    for task, items in by_task.items():
        ranked = sorted(items, key=lambda kv: kv[1]["delta"], reverse=True)
        top_gains_by_task[task] = [{"agml_dataset": ds, **d} for ds, d in ranked[:top_n]]
        top_losses_by_task[task] = [{"agml_dataset": ds, **d} for ds, d in ranked[::-1][:top_n]]

    return {
        "n_datasets": len(deltas),
        "win_rate": round(len(wins) / len(deltas), 4),
        "mean_gain": round(sum(d["delta"] for d in wins.values()) / len(wins), 4) if wins else None,
        "mean_loss": round(sum(d["delta"] for d in losses.values()) / len(losses), 4) if losses else None,
        "top_gains_by_task": top_gains_by_task,
        "top_losses_by_task": top_losses_by_task,
    }


def write_summary(model_dir: Path, model: str, summaries: dict[str, dict]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "method_comparison.json").write_text(json.dumps(summaries, indent=2))

    with open(model_dir / "method_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "n_datasets", "win_rate", "mean_gain", "mean_loss"])
        for method, s in summaries.items():
            w.writerow([model, method, s["n_datasets"], s["win_rate"], s["mean_gain"], s["mean_loss"]])
    print(f"  Saved: {model_dir / 'method_comparison.csv'}")

    with open(model_dir / "top_movers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "direction", "task_type", "rank", "agml_dataset",
                     "baseline_f1", "method_f1", "delta"])
        for method, s in summaries.items():
            for direction, by_task in (("gain", s["top_gains_by_task"]), ("loss", s["top_losses_by_task"])):
                for task, entries in sorted(by_task.items()):
                    for rank, e in enumerate(entries, start=1):
                        w.writerow([model, method, direction, task, rank, e["agml_dataset"],
                                     e["baseline_f1"], e["method_f1"], e["delta"]])
    print(f"  Saved: {model_dir / 'top_movers.csv'}")


def plot_summary(model_dir: Path, model: str, summaries: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    methods = [m for m in summaries if summaries[m]["n_datasets"] > 0]
    if not methods:
        return
    colors = [METHOD_COLOR.get(m, "#7f7f7f") for m in methods]
    labels = [METHOD_DISPLAY.get(m, m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    win_rates = [summaries[m]["win_rate"] * 100 for m in methods]
    bars = ax1.bar(labels, win_rates, color=colors)
    for b, v in zip(bars, win_rates):
        ax1.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Win rate vs. vision-baseline (%)", fontsize=9)
    ax1.set_title("Win rate", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.25)

    gains = [summaries[m]["mean_gain"] or 0.0 for m in methods]
    losses = [summaries[m]["mean_loss"] or 0.0 for m in methods]
    x = range(len(methods))
    ax2.bar(x, gains, color=colors, label="mean gain")
    ax2.bar(x, losses, color=colors, alpha=0.5, hatch="//", label="mean loss")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Judged F1 delta vs. vision-baseline", fontsize=9)
    ax2.set_title("Mean gain / mean loss", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"CoT / description method comparison — {model}", fontsize=11)
    plt.tight_layout()
    path = model_dir / "method_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_swap_datasets_file(model_dir: Path, cot_yaml_summary: dict) -> None:
    if cot_yaml_summary["n_datasets"] == 0:
        return
    names = []
    for by_task in (cot_yaml_summary["top_gains_by_task"], cot_yaml_summary["top_losses_by_task"]):
        for task in sorted(by_task):
            names.extend(e["agml_dataset"] for e in by_task[task])
    names = list(dict.fromkeys(names))  # de-dupe (possible if top_n > n_datasets/2 in a task), keep order
    path = model_dir / "cot_yaml_swap_datasets.txt"
    path.write_text(" ".join(names) + "\n")
    print(f"  Saved: {path}  ({len(names)} dataset(s), top-N gain + top-N loss PER TASK TYPE — "
          f"feed straight into `cot-yaml --swap_datasets $(cat {path})`)")


def main(args: argparse.Namespace) -> None:
    baseline_all = collate_models(Path(args.baseline_dir))
    if not baseline_all:
        print(f"No per-model runs found under --baseline_dir {args.baseline_dir} "
              f"(vision-baseline is the reference every method is compared against).")
        return

    method_dirs = {"desc-aug": args.aug_dir, "self-cot": args.self_cot_dir, "cot-yaml": args.cot_yaml_dir}
    method_all = {method: collate_models(Path(d)) for method, d in method_dirs.items()}
    baseline_rows_all = load_all_rows(Path(args.baseline_dir))

    output_dir = Path(args.output_dir)
    for model, baseline_metrics in baseline_all.items():
        ds_task = dataset_task_map(baseline_rows_all.get(model, []))
        summaries = {}
        for method, all_models in method_all.items():
            if model not in all_models:
                continue
            deltas = compute_deltas(baseline_metrics["by_dataset"], all_models[model]["by_dataset"], ds_task)
            summaries[method] = summarize_method(deltas, args.top_n)

        if not summaries:
            print(f"[{model}] no desc-aug/self-cot/cot-yaml runs found to compare against baseline — skipping")
            continue

        print(f"\n[{model}]")
        for method, s in summaries.items():
            if s["n_datasets"] == 0:
                print(f"  {method}: no overlapping datasets with baseline")
                continue
            print(f"  {method}: win_rate={s['win_rate']:.2f}  mean_gain={s['mean_gain']}  "
                  f"mean_loss={s['mean_loss']}  (n={s['n_datasets']})")

        model_dir = output_dir / model
        write_summary(model_dir, model, summaries)
        plot_summary(model_dir, model, summaries)
        if "cot-yaml" in summaries:
            write_swap_datasets_file(model_dir, summaries["cot-yaml"])

    condition_stats = collate_all_conditions(
        Path(args.baseline_dir), Path(args.aug_dir), Path(args.self_cot_dir), Path(args.cot_yaml_dir),
        args.n_boot, args.seed,
    )
    if condition_stats:
        write_all_conditions_summary(output_dir, condition_stats)
        plot_gemma_cot_f1_by_task(output_dir, condition_stats)

    gemma_deltas = collate_gemma_cot_deltas(
        Path(args.baseline_dir), Path(args.aug_dir), Path(args.self_cot_dir), Path(args.cot_yaml_dir),
        args.n_boot, args.seed,
    )
    if gemma_deltas:
        plot_gemma_cot_deltas(output_dir, gemma_deltas)

    gemma_delta_distributions = collate_gemma_cot_delta_distributions(
        Path(args.baseline_dir), Path(args.aug_dir), Path(args.self_cot_dir), Path(args.cot_yaml_dir),
    )
    if gemma_delta_distributions:
        plot_gemma_cot_delta_violin(output_dir, gemma_delta_distributions)
        plot_gemma_cot_delta_boxplot(output_dir, gemma_delta_distributions)

    species_breakdown = collate_species_breakdown(Path(args.baseline_dir))
    if species_breakdown:
        plot_species_identification_breakdown(output_dir, species_breakdown)

    # ── Stage 4 — verifier belief tracking + baselines (Sec. 16 of
    #    docs/verifier_score_belief_update.md) — no-ops until either script
    #    has actually been run, same pattern as the `if condition_stats:` /
    #    `if gemma_deltas:` guards above.
    verifier_condition_stats = collate_verifier_conditions(
        Path(args.verifier_dir), Path(args.verifier_lmv_ckpt), args.n_boot, args.seed,
    )
    if verifier_condition_stats:
        plot_gemma_cot_f1_by_task(
            output_dir, verifier_condition_stats,
            conditions=_VERIFIER_CONDITIONS, hatch_map=_VERIFIER_HATCH,
            output_name="verifier_gemma_f1_by_task.png",
        )

    verifier_deltas = collate_verifier_deltas(
        Path(args.baseline_dir), Path(args.verifier_dir), Path(args.verifier_lmv_ckpt), args.n_boot, args.seed,
    )
    if verifier_deltas:
        plot_gemma_cot_deltas(
            output_dir, verifier_deltas,
            conditions=_VERIFIER_CONDITIONS, hatch_map=_VERIFIER_HATCH,
            output_name="verifier_all_conditions_by_task.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_dir", default="outputs_final/vision_baseline/",
                         help="vision_baseline.py's --output_dir (the lower bound every method is compared to).")
    parser.add_argument("--aug_dir", default=METHOD_DIRS_DEFAULT["desc-aug"])
    parser.add_argument("--self_cot_dir", default=METHOD_DIRS_DEFAULT["self-cot"])
    parser.add_argument("--cot_yaml_dir", default=METHOD_DIRS_DEFAULT["cot-yaml"])
    parser.add_argument("--verifier_dir", default="outputs_final/verifier_cot/",
                         help="src/verifier_cot.py's --output_dir (both --baseline belief/argmax).")
    parser.add_argument("--verifier_lmv_ckpt", default="outputs_final/verifier_lmv/ckpt_judged.csv",
                         help="src/verifier_lmv.py's ckpt_judged.csv (its --output_dir is flat, one "
                              "model's worth of output per run — point this at whichever model's "
                              "run you want compared; only the PPT-Direct-N10 slice is used here).")
    parser.add_argument("--output_dir", default="outputs_final/cot_compare/")
    parser.add_argument("--top_n", type=int, default=5, help="Top-N gains / top-N losses to report per method.")
    parser.add_argument("--n_boot", type=int, default=2000,
                         help="Bootstrap resamples for cot_all_conditions_by_task.png's 95%% CI.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
