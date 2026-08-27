"""
Stage 4-report — collated K/pivot scaling summary across every verifier-cot
config that has been run so far (no GPU, no model).

Unlike verifier-summary.py (one run, one model, apples-to-apples vs. the
bounds/CoT conditions), this script looks across EVERY (model, K, pivots)
config that has ever been recorded into outputs_final/verifier_k_sweep/
(written automatically by `verifier-summary --k_pivots ...`) and builds
four comparison charts from whatever's there right now:

  verifier_deltas_by_task.png       — Delta judged F1 vs. Lower Bound, per
                                       task type, one bar cluster per
                                       (model, K, pivots) config.
  verifier_absolute_by_task.png     — Same layout/palette, but the
                                       Verifier's own raw judge_f1_macro per
                                       config (no Lower Bound bars), Upper
                                       Bound kept as a red dotted line.
  verifier_f1_progression.png       — Line chart of judged F1 vs. K, one
                                       line per model (same chart
                                       verifier-summary's own --k_pivots
                                       recording produces per-model, just
                                       all models overlaid on one axis).
  verifier_avg_time_per_response.png — Average seconds/response per config,
                                        read from each run's own checkpoint
                                        (avg_time_per_sample_s column).

No config list is hardcoded: every point plotted here is discovered by
reading outputs_final/verifier_k_sweep/*.csv, which already records each
point's own verifier_dir. This means running this script again after
copying over new results (from a longer local run, or a remote machine's
run) just picks up whatever configs exist now — no code change needed
for a new model, a new K, or a new --output_dir.

Usage (run any time after at least one `verifier-summary --k_pivots ...`
call has recorded a point):
    uv run python main.py verifier-scaling-summary
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.vision_eval import CKPT_FIELDS, compute_condition_stats, compute_delta_stats, load_ckpt  # noqa: E402
from src.verifier_cot import VERIFIER_CKPT_FIELDS  # noqa: E402

_K_SWEEP_DIR = Path("outputs_final/verifier_k_sweep")


def _display_label(model_safe: str) -> str:
    """Full model name (e.g. "gemma-4-E4B-it", "Qwen3.5-9B") — only the
    org prefix is dropped, matching src/linear_probe's backbone-label
    convention, so the same model reads identically across every figure."""
    return model_safe.replace("google_", "").replace("Qwen_", "")


def load_sweep_configs() -> list[dict]:
    """Every (model, K, pivots) point recorded so far, across every model's
    own outputs_final/verifier_k_sweep/<model>.csv — each row already
    carries its own verifier_dir (written by verifier-summary's
    record_k_sweep_point), so no separate directory-discovery pass is
    needed. Only judged points (non-empty judge_f1_macro) are returned."""
    configs = []
    if not _K_SWEEP_DIR.exists():
        return configs
    for f in sorted(_K_SWEEP_DIR.glob("*.csv")):
        model_safe = f.stem
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("judge_f1_macro"):
                    continue
                configs.append({
                    "model_safe": model_safe,
                    "label": _display_label(model_safe),
                    "verifier_dir": Path(r["verifier_dir"]),
                    "k_candidates": int(r["k_candidates"]),
                    "k_pivots": int(r["k_pivots"]),
                    "judge_f1_macro": float(r["judge_f1_macro"]),
                    "judge_ci_low": float(r["judge_ci_low"]),
                    "judge_ci_high": float(r["judge_ci_high"]),
                    "n": int(r["n"]),
                })
    return configs


def _load_verifier_rows(verifier_dir: Path, model_safe: str) -> list[dict]:
    ckpt = verifier_dir / model_safe / f"ckpt_{model_safe}.csv"
    rows, _ = load_ckpt(ckpt, fields=VERIFIER_CKPT_FIELDS)
    return [r for r in rows if r.get("judged_correct") not in (None, "")]


def _load_lower_bound_rows(model_safe: str, keys: set[tuple[str, str, str]]) -> list[dict]:
    ckpt = Path("outputs_final/vision_baseline") / model_safe / f"ckpt_{model_safe}.csv"
    rows, _ = load_ckpt(ckpt, fields=CKPT_FIELDS)
    return [r for r in rows if (r["agml_dataset"], r["class_label"], r["image_name"]) in keys
            and r.get("judged_correct") not in (None, "")]


_PALETTE = ["#e06b00", "#1258a0", "#2f7a3d", "#8c564b", "#9467bd", "#17a398"]


def _model_color_map(configs: list[dict]) -> dict[str, str]:
    """One color per model_safe, assigned in the same order/palette every
    chart in this module uses (first model encountered -> _PALETTE[0],
    etc.) — a single shared assignment so a model's color is identical
    across verifier_deltas_by_task.png, verifier_absolute_by_task.png,
    verifier_f1_progression.png, and verifier_avg_time_per_response.png,
    rather than each plot function re-deriving its own (previously
    plot_f1_progression used matplotlib's default color cycle instead of
    this palette, so its colors didn't match the other three charts)."""
    colors: dict[str, str] = {}
    for cfg in configs:
        if cfg["model_safe"] not in colors:
            colors[cfg["model_safe"]] = _PALETTE[len(colors) % len(_PALETTE)]
    return colors


def _load_upper_bound_rows(model_safe: str, keys: set[tuple[str, str, str]]) -> list[dict]:
    ckpt = Path("outputs_final/desc_augmented_vision") / model_safe / f"ckpt_{model_safe}.csv"
    rows, _ = load_ckpt(ckpt, fields=CKPT_FIELDS)
    return [r for r in rows if (r["agml_dataset"], r["class_label"], r["image_name"]) in keys
            and r.get("judged_correct") not in (None, "")]


def plot_deltas_by_task(configs: list[dict], out_dir: Path, n_boot: int, seed: int) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    model_colors = _model_color_map(configs)
    records = []
    for cfg in configs:
        vrows = _load_verifier_rows(cfg["verifier_dir"], cfg["model_safe"])
        if not vrows:
            continue
        keys = {(r["agml_dataset"], r["class_label"], r["image_name"]) for r in vrows}
        brows = _load_lower_bound_rows(cfg["model_safe"], keys)
        if not brows:
            print(f"  [skip] {cfg['label']} K={cfg['k_candidates']}: no matching Lower Bound rows")
            continue
        stats = compute_delta_stats(brows, vrows, n_boot=n_boot, seed=seed)
        # Upper Bound's own delta vs. the SAME Lower Bound rows/keys this
        # config used — drawn as a reference line, not a bar, mirroring
        # cot_compare.py's plot_gemma_cot_deltas convention.
        arows = _load_upper_bound_rows(cfg["model_safe"], keys)
        aug_stats = compute_delta_stats(brows, arows, n_boot=n_boot, seed=seed) if arows else None
        records.append((cfg, stats, aug_stats))

    if not records:
        print("  [skip] verifier_deltas_by_task: no configs with matching Lower Bound data")
        return

    task_types = sorted({tt for _c, st, _a in records for tt in st if tt != "overall"})
    scopes = task_types
    x = np.arange(len(scopes))
    n_bars = len(records)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scopes) * 1.9), 5))
    for i, (cfg, stats, aug_stats) in enumerate(records):
        means, lo_err, hi_err = [], [], []
        for scope in scopes:
            s = stats.get(scope)
            if s and s.get("delta_n_classes", 0) > 0:
                m, lo, hi = s["delta_f1_macro"], s["delta_ci_low"], s["delta_ci_high"]
            else:
                m = lo = hi = 0.0
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, means, width,
               label=f"{cfg['label']} (K={cfg['k_candidates']}, pivots={cfg['k_pivots']})",
               color=model_colors[cfg["model_safe"]],
               alpha=0.5 + 0.4 * min(cfg["k_candidates"] / 24, 1.0),
               edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.7})
        for bar, m, hi in zip(bars, means, hi_err):
            ax.text(bar.get_x() + bar.get_width() / 2, m + hi + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

        if aug_stats:
            xmin, xmax = x + offset - width / 2, x + offset + width / 2
            for xi_min, xi_max, scope in zip(xmin, xmax, scopes):
                s = aug_stats.get(scope)
                if s and s.get("delta_n_classes", 0) > 0:
                    ax.hlines(s["delta_f1_macro"], xi_min, xi_max, colors="#d62728",
                              linestyles=":", linewidth=1.8, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                        rotation=15, ha="right")
    ax.set_ylabel("Δ Judged F1 Macro (vs Lower Bound)")
    bound_handle = Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.8, label="Upper Bound")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bound_handle], labels + ["Upper Bound"], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out = out_dir / "verifier_deltas_by_task.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_absolute_by_task(configs: list[dict], out_dir: Path, n_boot: int, seed: int) -> None:
    """Absolute-score companion to plot_deltas_by_task: same orange/blue
    per-model palette, same task-only x-axis (no "overall", no Lower Bound
    bars) — just the Verifier's own raw judge_f1_macro per config, with
    Upper Bound kept as the same red dotted reference line."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    model_colors = _model_color_map(configs)
    records = []
    for cfg in configs:
        vrows = _load_verifier_rows(cfg["verifier_dir"], cfg["model_safe"])
        if not vrows:
            continue
        keys = {(r["agml_dataset"], r["class_label"], r["image_name"]) for r in vrows}
        stats = compute_condition_stats(vrows, n_boot=n_boot, seed=seed)
        arows = _load_upper_bound_rows(cfg["model_safe"], keys)
        aug_stats = compute_condition_stats(arows, n_boot=n_boot, seed=seed) if arows else None
        records.append((cfg, stats, aug_stats))

    if not records:
        print("  [skip] verifier_absolute_by_task: no configs with data")
        return

    task_types = sorted({tt for _c, st, _a in records for tt in st if tt != "overall"})
    scopes = task_types
    x = np.arange(len(scopes))
    n_bars = len(records)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scopes) * 1.9), 5))
    for i, (cfg, stats, aug_stats) in enumerate(records):
        means, lo_err, hi_err = [], [], []
        for scope in scopes:
            s = stats.get(scope)
            if s and s.get("judge_n_classes", 0) > 0:
                m, lo, hi = s["judge_f1_macro"], s["judge_ci_low"], s["judge_ci_high"]
            else:
                m = lo = hi = 0.0
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, means, width,
               label=f"{cfg['label']} (K={cfg['k_candidates']}, pivots={cfg['k_pivots']})",
               color=model_colors[cfg["model_safe"]],
               alpha=0.5 + 0.4 * min(cfg["k_candidates"] / 24, 1.0),
               edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.7})
        for bar, m, hi in zip(bars, means, hi_err):
            ax.text(bar.get_x() + bar.get_width() / 2, m + hi + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

        if aug_stats:
            xmin, xmax = x + offset - width / 2, x + offset + width / 2
            for xi_min, xi_max, scope in zip(xmin, xmax, scopes):
                s = aug_stats.get(scope)
                if s and s.get("judge_n_classes", 0) > 0:
                    ax.hlines(s["judge_f1_macro"], xi_min, xi_max, colors="#d62728",
                              linestyles=":", linewidth=1.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_identification", "").replace("_", " ") for s in scopes],
                        rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Judged F1 Macro")
    bound_handle = Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.8, label="Upper Bound")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bound_handle], labels + ["Upper Bound"], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out = out_dir / "verifier_absolute_by_task.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_f1_progression(configs: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[dict]] = {}
    for cfg in configs:
        by_model.setdefault(cfg["model_safe"], []).append(cfg)

    if not by_model:
        print("  [skip] verifier_f1_progression: no recorded sweep points")
        return

    model_colors = _model_color_map(configs)
    model_order = sorted(by_model)
    all_ks = [c["k_candidates"] for pts in by_model.values() for c in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_safe in model_order:
        pts = sorted(by_model[model_safe], key=lambda c: c["k_candidates"])
        ks = [c["k_candidates"] for c in pts]
        vals = [c["judge_f1_macro"] for c in pts]
        lo = [max(0.0, c["judge_f1_macro"] - c["judge_ci_low"]) for c in pts]
        hi = [max(0.0, c["judge_ci_high"] - c["judge_f1_macro"]) for c in pts]
        ax.errorbar(ks, vals, yerr=[lo, hi], marker="o", markersize=7, linewidth=1.8,
                    capsize=4, color=model_colors[model_safe], label=_display_label(model_safe))
        # E4B-it's labels sit below its line instead of above — its line
        # runs above 26B-A4B-it's, so this drops them into the gap between
        # the two lines. Vertical separation alone (no horizontal stagger
        # needed) since only these two lines exist.
        below = "e4b-it" in model_safe.lower()
        for c in pts:
            ax.annotate(f"K={c['k_candidates']}\npivots={c['k_pivots']}\n{c['judge_f1_macro']:.3f}",
                        (c["k_candidates"], c["judge_f1_macro"]), ha="center",
                        va="top" if below else "bottom", fontsize=7.5,
                        xytext=(0, -55 if below else 10), textcoords="offset points")

    ax.set_xlabel("Candidate pool size (K)")
    ax.set_ylabel("Judged F1 Macro")
    # Anchored at 0 (rather than autoscaled tight to the data) so the CI
    # error bars read at their true relative size instead of looking
    # inflated by a narrow y-range. Extra x-padding gives staggered
    # labels at the leftmost/rightmost K room without hitting the axes.
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(all_ks) - 1.2, max(all_ks) + 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "verifier_f1_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_avg_time_per_response(configs: list[dict], out_dir: Path) -> None:
    """Line chart, one line per model vs. K — same shape/palette as
    plot_f1_progression, so the two charts read as a matched pair (cost vs.
    accuracy at the same K points) rather than a bar chart with an
    unrelated color convention."""
    import statistics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[dict]] = {}
    for cfg in configs:
        ckpt = cfg["verifier_dir"] / cfg["model_safe"] / f"ckpt_{cfg['model_safe']}.csv"
        if not ckpt.exists():
            continue
        with open(ckpt, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        vals = [float(r["avg_time_per_sample_s"]) for r in rows if r.get("avg_time_per_sample_s")]
        if not vals:
            continue
        mean_t = sum(vals) / len(vals)
        std_t = statistics.stdev(vals) if len(vals) > 1 else 0.0
        by_model.setdefault(cfg["model_safe"], []).append({**cfg, "mean_t": mean_t, "std_t": std_t})

    if not by_model:
        print("  [skip] verifier_avg_time_per_response: no checkpoints with timing data")
        return

    model_colors = _model_color_map(configs)
    model_order = sorted(by_model)
    all_ks = [c["k_candidates"] for pts in by_model.values() for c in pts]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_safe in model_order:
        pts = sorted(by_model[model_safe], key=lambda c: c["k_candidates"])
        ks = [c["k_candidates"] for c in pts]
        vals = [c["mean_t"] for c in pts]
        errs = [c["std_t"] for c in pts]
        color = model_colors[model_safe]
        ax.errorbar(ks, vals, yerr=errs, marker="o", markersize=7, linewidth=1.8,
                    capsize=4, color=color, label=_display_label(model_safe))
        # E4B-it's labels sit below its line instead of above — its line
        # runs below 26B-A4B-it's, so this drops them into open space
        # instead of toward the other line.
        below = "e4b-it" in model_safe.lower()
        for c in pts:
            # Anchor to the error bar's own cap (not the bare point) so a
            # large std doesn't push the label under/over its own bar.
            anchor_y = c["mean_t"] - c["std_t"] if below else c["mean_t"] + c["std_t"]
            ax.annotate(f"pivots={c['k_pivots']}", (c["k_candidates"], anchor_y),
                        ha="center", va="top" if below else "bottom", fontsize=7.5,
                        xytext=(0, -6 if below else 6), textcoords="offset points")

    ax.set_xlabel("Candidate pool size (K)")
    ax.set_ylabel("Avg. time per response (s)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(all_ks) - 1.2, max(all_ks) + 1.2)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / "verifier_avg_time_per_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main(args: argparse.Namespace) -> None:
    configs = load_sweep_configs()
    if not configs:
        print(f"  No recorded points under {_K_SWEEP_DIR} — run `verifier-summary --k_pivots ...` "
              f"at least once first.")
        return
    print(f"  {len(configs)} (model, K, pivots) point(s) found:")
    for cfg in configs:
        print(f"    {cfg['label']:20s} K={cfg['k_candidates']:<3d} pivots={cfg['k_pivots']:<3d} "
              f"judge_f1_macro={cfg['judge_f1_macro']:.4f}  n={cfg['n']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_deltas_by_task(configs, out_dir, args.n_boot, args.seed)
    plot_absolute_by_task(configs, out_dir, args.n_boot, args.seed)
    plot_f1_progression(configs, out_dir)
    plot_avg_time_per_response(configs, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="outputs_final/verifier_scaling_collated/")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the delta 95%% CIs.")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
