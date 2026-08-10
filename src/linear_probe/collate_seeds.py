"""
Stage 1b — Collate linear probe results across seeds, with 95% confidence
intervals.

probe.py's --seed controls which n_samples images of each class land in the
training split (see probe_config.yaml). Running probe.py multiple times with
different --seed values (same models, same --cache_dir so extraction is
never repeated — see docs/stage1_linear_probe.md) gives independent
replicate F1 scores per (task type, backbone). This script collates those
replicates into the same summary_by_task.png chart probe.py itself produces,
but with error bars showing the 95% CI of the mean across seeds (Student's
t, appropriate for the small number of replicates a multi-seed run
typically has) instead of a single point estimate.

Usage (after running probe.py once per seed into separate output dirs):
    uv run python main.py probe-collate-seeds \\
        --seed_dirs outputs_final/linear_probe_seed1/ \\
                    outputs_final/linear_probe_seed2/ \\
                    outputs_final/linear_probe_seed3/ \\
        --output_dir outputs_final/linear_probe_seeds_collated/

Outputs (into --output_dir):
  seed_collate_summary.csv    — task_type, backbone, mean, ci_low, ci_high, n_seeds
  seed_collate_summary.json   — same, structured, plus the input seed_dirs
  summary_by_task_seeds.png   — same chart as probe.py's summary_by_task.png,
                                 with 95% CI error bars across seeds
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as _t_dist

from src.common.plot_style import family_shaded_colors, is_vision_only


def _f1(entry) -> float:
    return entry if isinstance(entry, float) else entry.get("f1_macro", 0.0)


def load_by_task(results_path: Path) -> dict[str, dict[str, float]]:
    """{task_type: {backbone: f1_macro}} from one seed run's
    linear_probe_results.json — same by_task_lr_combined -> by_task_fc_combined
    -> by_task fallback chain probe.py's own plot_task_summary uses."""
    results = json.loads(results_path.read_text())
    by_task = (
        results.get("by_task_lr_combined")
        or results.get("by_task_fc_combined")
        or results.get("by_task", {})
    )
    return {tt: {b: _f1(v) for b, v in backbones.items()} for tt, backbones in by_task.items()}


def mean_ci(values: list[float]) -> dict:
    """Mean + 95% CI via Student's t (appropriate for a small number of real
    replicates, unlike bootstrap-over-classes which needs many resampling
    units). n=1 -> no interval (ci_low == ci_high == mean)."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"mean": round(mean, 4), "ci_low": round(mean, 4), "ci_high": round(mean, 4),
                "n_seeds": n, "values": [round(v, 4) for v in values]}
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    margin = float(_t_dist.ppf(0.975, df=n - 1)) * sem
    return {
        "mean": round(mean, 4),
        "ci_low": round(mean - margin, 4),
        "ci_high": round(mean + margin, 4),
        "n_seeds": n,
        "values": [round(v, 4) for v in values],
    }


def collate(seed_dirs: list[Path]) -> dict[str, dict[str, dict]]:
    """{task_type: {backbone: mean_ci(...)}} across every usable seed dir."""
    per_seed = []
    for d in seed_dirs:
        p = d / "linear_probe_results.json"
        if not p.exists():
            print(f"  [skip] {d} — no linear_probe_results.json")
            continue
        per_seed.append(load_by_task(p))
        print(f"  Loaded {p}")

    if not per_seed:
        return {}

    task_types = sorted({tt for by_task in per_seed for tt in by_task})
    result: dict[str, dict[str, dict]] = {}
    for tt in task_types:
        backbones = sorted({b for by_task in per_seed for b in by_task.get(tt, {})})
        result[tt] = {}
        for b in backbones:
            values = [by_task[tt][b] for by_task in per_seed if tt in by_task and b in by_task[tt]]
            if values:
                result[tt][b] = mean_ci(values)
    return result


def write_summary(output_dir: Path, seed_dirs: list[Path], collated: dict[str, dict[str, dict]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_collate_summary.json").write_text(json.dumps({
        "seed_dirs": [str(d) for d in seed_dirs],
        "by_task": collated,
    }, indent=2))

    with open(output_dir / "seed_collate_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "backbone", "mean", "ci_low", "ci_high", "n_seeds"])
        writer.writeheader()
        for tt, backbones in collated.items():
            for b, stat in backbones.items():
                writer.writerow({"task_type": tt, "backbone": b, "mean": stat["mean"],
                                  "ci_low": stat["ci_low"], "ci_high": stat["ci_high"],
                                  "n_seeds": stat["n_seeds"]})
    print(f"  Saved: {output_dir / 'seed_collate_summary.csv'}")
    print(f"  Saved: {output_dir / 'seed_collate_summary.json'}")


def plot_seed_collated(collated: dict[str, dict[str, dict]], output_dir: Path) -> None:
    task_types = list(collated.keys())
    if not task_types:
        print("  No data to plot.")
        return
    backbones = sorted({b for tt in collated.values() for b in tt})
    n_seeds_max = max((stat["n_seeds"] for tt in collated.values() for stat in tt.values()), default=0)

    x = np.arange(len(task_types))
    width = 0.8 / len(backbones)
    colors = family_shaded_colors(backbones)

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 2), 5))
    for i, b in enumerate(backbones):
        means, lo_err, hi_err = [], [], []
        for tt in task_types:
            stat = collated[tt].get(b)
            m = stat["mean"] if stat else 0.0
            lo = stat["ci_low"] if stat else 0.0
            hi = stat["ci_high"] if stat else 0.0
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))

        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, label=b.split("/")[-1],
            color=colors[b],
            hatch="//" if is_vision_only(b) else None,
            edgecolor="white" if is_vision_only(b) else "black",
            linewidth=0.5,
            yerr=[lo_err, hi_err], capsize=2,
            error_kw={"elinewidth": 0.8, "alpha": 0.7},
        )
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([tt.replace("_", " ") for tt in task_types], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"F1 Macro (95% CI across {n_seeds_max} seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(backbones), 3))
    plt.tight_layout()

    path = output_dir / "summary_by_task_seeds.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main(args: argparse.Namespace) -> None:
    seed_dirs = [Path(d) for d in args.seed_dirs]
    collated = collate(seed_dirs)
    if not collated:
        print("No usable linear_probe_results.json found in any --seed_dirs.")
        return

    output_dir = Path(args.output_dir)
    write_summary(output_dir, seed_dirs, collated)
    plot_seed_collated(collated, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed_dirs", nargs="+", required=True,
                         help="Paths to linear probe output dirs, one per seed run — each must "
                              "contain linear_probe_results.json (i.e. each was produced by "
                              "probe.py --seed <N> --output_dir <this dir>).")
    parser.add_argument("--output_dir", default="outputs_final/linear_probe_seeds_collated/")
    main(parser.parse_args())
