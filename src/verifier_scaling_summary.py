"""
Stage 4-report — collated K/pivot scaling summary across every verifier-cot
config that has been run so far (no GPU, no model).

Unlike verifier-summary.py (one run, one model, apples-to-apples vs. the
bounds/CoT conditions), this script looks across EVERY (model, K, pivots)
config that has ever been recorded into outputs_final/verifier_k_sweep/
(written automatically by `verifier-summary --k_pivots ...`) and builds
three comparison charts from whatever's there right now:

  verifier_deltas_by_task.png       — Delta judged F1 vs. Lower Bound, per
                                       task type, one bar cluster per
                                       (model, K, pivots) config.
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

from src.common.vision_eval import CKPT_FIELDS, compute_delta_stats, load_ckpt  # noqa: E402
from src.verifier_cot import VERIFIER_CKPT_FIELDS  # noqa: E402

_K_SWEEP_DIR = Path("outputs_final/verifier_k_sweep")


def _display_label(model_safe: str) -> str:
    return model_safe.replace("google_gemma-4-", "").replace("Qwen_Qwen3.5-", "Qwen3.5-")


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

    model_colors = {}
    palette = ["#e06b00", "#1258a0", "#2f7a3d", "#8c564b", "#9467bd", "#17a398"]
    records = []
    for cfg in configs:
        if cfg["model_safe"] not in model_colors:
            model_colors[cfg["model_safe"]] = palette[len(model_colors) % len(palette)]
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
    scopes = task_types + ["overall"]
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
        ax.bar(x + offset, means, width,
               label=f"{cfg['label']} (K={cfg['k_candidates']}, pivots={cfg['k_pivots']})",
               color=model_colors[cfg["model_safe"]],
               alpha=0.5 + 0.4 * min(cfg["k_candidates"] / 24, 1.0),
               edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.7})

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
    ax.set_title("Verifier delta by task — every completed (model, K, pivots) config")
    bound_handle = Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.8, label="Upper Bound")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bound_handle], labels + ["Upper Bound"], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out = out_dir / "verifier_deltas_by_task.png"
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

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_safe in sorted(by_model):
        pts = sorted(by_model[model_safe], key=lambda c: c["k_candidates"])
        ks = [c["k_candidates"] for c in pts]
        vals = [c["judge_f1_macro"] for c in pts]
        lo = [max(0.0, c["judge_f1_macro"] - c["judge_ci_low"]) for c in pts]
        hi = [max(0.0, c["judge_ci_high"] - c["judge_f1_macro"]) for c in pts]
        ax.errorbar(ks, vals, yerr=[lo, hi], marker="o", markersize=7, linewidth=1.8,
                    capsize=4, label=_display_label(model_safe))
        for c in pts:
            ax.annotate(f"K={c['k_candidates']}\npivots={c['k_pivots']}\n{c['judge_f1_macro']:.3f}",
                        (c["k_candidates"], c["judge_f1_macro"]), ha="center", va="bottom",
                        fontsize=7.5, xytext=(0, 10), textcoords="offset points")

    ax.set_xlabel("Candidate pool size (K)")
    ax.set_ylabel("Judged F1 Macro")
    ax.set_title("Verifier scaling — judged F1 vs. K / pivots, per model")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "verifier_f1_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_avg_time_per_response(configs: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    timing = []
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
        # A config backfilled from a source other than a fresh single-round
        # verifier-cot run (e.g. extracted round-1 rows out of an old
        # multi-round checkpoint) has no compute_summary.json of its own —
        # flag it rather than silently mixing incomparable timing sources.
        has_compute_summary = (cfg["verifier_dir"] / cfg["model_safe"] / "compute_summary.json").exists()
        label = f"{cfg['label']}\nK={cfg['k_candidates']}, p={cfg['k_pivots']}"
        timing.append((label, mean_t, has_compute_summary))

    if not timing:
        print("  [skip] verifier_avg_time_per_response: no checkpoints with timing data")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(timing) * 1.6), 5))
    labels = [t[0] for t in timing]
    vals = [t[1] for t in timing]
    colors = ["#2f7a3d" if t[2] else "#b0b0b0" for t in timing]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}s", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Avg. time per response (s)")
    ax.set_title("Verifier compute cost per (model, K, pivots) config")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(handles=[
        Patch(facecolor="#2f7a3d", edgecolor="black", label="Fresh single-round run"),
        Patch(facecolor="#b0b0b0", edgecolor="black", label="Backfilled (no compute_summary.json)"),
    ], fontsize=8)
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
    plot_f1_progression(configs, out_dir)
    plot_avg_time_per_response(configs, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="outputs_final/verifier_scaling_collated/")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for the delta 95%% CIs.")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
