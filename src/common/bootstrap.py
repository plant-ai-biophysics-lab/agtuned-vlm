"""Bootstrap confidence intervals — shared by the judge-improvements
evaluation plan's Phase 1-3 (leniency_analysis.py, judge_consistency.py,
judge_template_sensitivity.py).

Resamples ROWS (not repeat judge calls, not multiple seeded re-runs) with
replacement to get a 95% CI on a rate or a paired difference of rates — the
row counts in this pipeline (hundreds to tens of thousands per task_type)
make bootstrapping cheap and give a much tighter CI than re-running the
judge across a handful of seeds would, at zero extra judge-call cost.
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """95% (or `ci`) CI on the mean of `values`, via resampling rows with
    replacement n_boot times. Returns (nan, nan) if values is empty."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return (float(lo), float(hi))


def bootstrap_diff_ci(
    values_a: list[float], values_b: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """95% CI on mean(values_a) - mean(values_b), PAIRED — values_a[i] and
    values_b[i] must be the same row i (e.g. leniency_gap's full-response
    and resolved-class verdicts for the same row), so each bootstrap
    resample draws the same row index for both arrays rather than
    resampling them independently (which would overstate the CI width by
    ignoring the pairing)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = len(a)
    if n == 0 or n != len(b):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = a[idx].mean() - b[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_diffs, [alpha, 1 - alpha])
    return (float(lo), float(hi))


def ci_to_yerr(value: float, ci: tuple[float, float]) -> tuple[float, float]:
    """(lo, hi) -> (lower_err, upper_err) relative to `value`, for
    matplotlib's yerr=[[lower_errs], [upper_errs]] convention. Clips at 0 —
    bootstrap quantile noise can occasionally put `value` a hair outside
    its own CI (e.g. value computed from the true sample, ci from resamples
    of it — should bracket it, but floating point/quantile-interpolation
    edge cases can violate this by a negligible amount)."""
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return (0.0, 0.0)
    return (max(0.0, value - lo), max(0.0, hi - value))


def yerr_pairs(results: dict, task_types: list[str], value_key: str, ci_key: str) -> np.ndarray | None:
    """Builds matplotlib's yerr=[[lower,...],[upper,...]] array from a
    per-task_type results dict (as produced by leniency_analysis.py's
    compute_leniency_gap, judge_consistency.py's compute_flip_rate, or
    judge_template_sensitivity.py's score_against_deterministic — all share
    this {task_type: {value_key, ci_key: (lo, hi), ...}} shape), or returns
    None (no error bars drawn) if `ci_key` is missing anywhere — keeps every
    plotting function backward compatible with results computed before CI
    support was added, rather than crashing on a KeyError."""
    if not all(ci_key in results.get(t, {}) for t in task_types):
        return None
    lowers, uppers = [], []
    for t in task_types:
        value = results[t][value_key]
        if isinstance(value, float) and np.isnan(value):
            lo, hi = 0.0, 0.0
        else:
            lo, hi = ci_to_yerr(value, results[t][ci_key])
        lowers.append(lo)
        uppers.append(hi)
    return np.array([lowers, uppers])
