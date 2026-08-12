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
  top_movers.csv                 — every method's top-N gain/loss datasets
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

from src.common.vision_eval import collate_models

METHOD_DIRS_DEFAULT = {
    "desc-aug":  "outputs_final/desc_augmented_vision_v2/",
    "self-cot":  "outputs_final/self_cot/",
    "cot-yaml":  "outputs_final/cot_yaml/",
}
METHOD_COLOR = {"desc-aug": "#ff7f0e", "self-cot": "#2ca02c", "cot-yaml": "#9467bd"}


def _f1(m: dict) -> float | None:
    v = m.get("judge_f1_macro")
    if v is None:
        v = m.get("f1_macro")
    return float(v) if v is not None else None


def compute_deltas(baseline_by_ds: dict[str, dict], method_by_ds: dict[str, dict]) -> dict[str, dict]:
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
            "task_type": method_by_ds[ds].get("task_type", ""),
        }
    return out


def summarize_method(deltas: dict[str, dict], top_n: int) -> dict:
    if not deltas:
        return {"n_datasets": 0, "win_rate": None, "mean_gain": None, "mean_loss": None,
                "top_gains": [], "top_losses": []}
    wins = {ds: d for ds, d in deltas.items() if d["delta"] > 0}
    losses = {ds: d for ds, d in deltas.items() if d["delta"] < 0}
    ranked = sorted(deltas.items(), key=lambda kv: kv[1]["delta"], reverse=True)
    return {
        "n_datasets": len(deltas),
        "win_rate": round(len(wins) / len(deltas), 4),
        "mean_gain": round(sum(d["delta"] for d in wins.values()) / len(wins), 4) if wins else None,
        "mean_loss": round(sum(d["delta"] for d in losses.values()) / len(losses), 4) if losses else None,
        "top_gains": [{"agml_dataset": ds, **d} for ds, d in ranked[:top_n]],
        "top_losses": [{"agml_dataset": ds, **d} for ds, d in ranked[::-1][:top_n]],
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
        w.writerow(["model", "method", "direction", "rank", "agml_dataset", "task_type",
                     "baseline_f1", "method_f1", "delta"])
        for method, s in summaries.items():
            for direction, entries in (("gain", s["top_gains"]), ("loss", s["top_losses"])):
                for rank, e in enumerate(entries, start=1):
                    w.writerow([model, method, direction, rank, e["agml_dataset"], e["task_type"],
                                 e["baseline_f1"], e["method_f1"], e["delta"]])
    print(f"  Saved: {model_dir / 'top_movers.csv'}")


def plot_summary(model_dir: Path, model: str, summaries: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    methods = [m for m in summaries if summaries[m]["n_datasets"] > 0]
    if not methods:
        return
    colors = [METHOD_COLOR.get(m, "#7f7f7f") for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    win_rates = [summaries[m]["win_rate"] * 100 for m in methods]
    bars = ax1.bar(methods, win_rates, color=colors)
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
    ax2.set_xticklabels(methods)
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
    names = [e["agml_dataset"] for e in cot_yaml_summary["top_gains"]] + \
            [e["agml_dataset"] for e in cot_yaml_summary["top_losses"]]
    names = list(dict.fromkeys(names))  # de-dupe (possible if top_n > n_datasets/2), keep order
    path = model_dir / "cot_yaml_swap_datasets.txt"
    path.write_text(" ".join(names) + "\n")
    print(f"  Saved: {path}  ({len(names)} dataset(s) — feed straight into "
          f"`cot-yaml --swap_datasets $(cat {path})`)")


def main(args: argparse.Namespace) -> None:
    baseline_all = collate_models(Path(args.baseline_dir))
    if not baseline_all:
        print(f"No per-model runs found under --baseline_dir {args.baseline_dir} "
              f"(vision-baseline is the reference every method is compared against).")
        return

    method_dirs = {"desc-aug": args.aug_dir, "self-cot": args.self_cot_dir, "cot-yaml": args.cot_yaml_dir}
    method_all = {method: collate_models(Path(d)) for method, d in method_dirs.items()}

    output_dir = Path(args.output_dir)
    for model, baseline_metrics in baseline_all.items():
        summaries = {}
        for method, all_models in method_all.items():
            if model not in all_models:
                continue
            deltas = compute_deltas(baseline_metrics["by_dataset"], all_models[model]["by_dataset"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_dir", default="outputs_final/vision_baseline/",
                         help="vision_baseline.py's --output_dir (the lower bound every method is compared to).")
    parser.add_argument("--aug_dir", default=METHOD_DIRS_DEFAULT["desc-aug"])
    parser.add_argument("--self_cot_dir", default=METHOD_DIRS_DEFAULT["self-cot"])
    parser.add_argument("--cot_yaml_dir", default=METHOD_DIRS_DEFAULT["cot-yaml"])
    parser.add_argument("--output_dir", default="outputs_final/cot_compare/")
    parser.add_argument("--top_n", type=int, default=5, help="Top-N gains / top-N losses to report per method.")
    main(parser.parse_args())
