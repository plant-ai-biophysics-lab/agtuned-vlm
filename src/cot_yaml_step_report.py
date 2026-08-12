"""
Stage 2d-report — CoT-YAML per-step F1 report (no GPU, no model needed).

cot_yaml.py's --per_step / --swap_datasets write per-chain-position rows
(one prediction per step, not just the final one) to
ckpt_steps_<model>.csv / ckpt_swap_<model>.csv. This script reads those,
scores each step with the exact same shared metric functions
compute_model_metrics() uses for the final answer
(compute_step_metrics() in src/common/vision_eval.py — never a
step-analysis-specific reimplementation), and reports how F1 evolves as
reasoning accumulates.

Outputs (into --output_dir/<model>/, alongside the checkpoints it read from):
  step_f1_by_dataset.csv   — model, dataset, task_type, step_idx, f1_macro,
                              judge_f1_macro, judge_accuracy, n_judged, n_total
  step_f1_by_task.csv      — same, aggregated by task_type
  step_f1_overall.png      — F1 vs. step index, one line per task type + overall
  swap_step_f1.csv         — (--swap_datasets data only) rotation, step_idx,
                              question_id, task_type, f1_macro, judge_f1_macro
  swap_step_f1_<task>.png  — (--swap_datasets data only) one line per rotation,
                              per task type present in the swap data

Usage (run once per model has produced --per_step / --swap_datasets data):
    uv run python main.py cot-yaml-step-report
    uv run python main.py cot-yaml-step-report --output_dir outputs_final/cot_yaml/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
from collections import defaultdict

from src.common.vision_eval import compute_step_metrics, load_ckpt
from src.cot_yaml import STEP_CKPT_FIELDS

TASK_LABEL = {
    "disease": "Disease", "pest_damage": "Pest Damage",
    "species_identification": "Species ID", "weed_identification": "Weed ID",
    "quality_identification": "Quality ID",
}

_F1_KEY = "judge_f1_macro"  # falls back to f1_macro per-point if judging was skipped


def _f1(m: dict) -> float | None:
    v = m.get(_F1_KEY, m.get("f1_macro"))
    return float(v) if v is not None else None


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Saved: {path}")


def process_step_curve(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ds_task = {r["agml_dataset"]: r["task_type"] for r in rows}

    by_ds_step = compute_step_metrics(rows, ("agml_dataset", "step_idx"))
    ds_csv_rows = []
    for (ds, step), m in sorted(by_ds_step.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        ds_csv_rows.append([
            model_name, ds, ds_task.get(ds, ""), step, m.get("n_total"),
            m.get("f1_macro"), m.get("judge_f1_macro", ""), m.get("judge_accuracy", ""),
            m.get("n_judged", ""), m.get("rouge_ss", ""),
        ])
    _write_csv(
        out_dir / "step_f1_by_dataset.csv",
        ["model", "agml_dataset", "task_type", "step_idx", "n_total",
         "f1_macro", "judge_f1_macro", "judge_accuracy", "n_judged", "rouge_ss"],
        ds_csv_rows,
    )

    by_task_step = compute_step_metrics(rows, ("task_type", "step_idx"))
    task_csv_rows = []
    for (task, step), m in sorted(by_task_step.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        task_csv_rows.append([
            model_name, task, step, m.get("n_total"),
            m.get("f1_macro"), m.get("judge_f1_macro", ""), m.get("judge_accuracy", ""),
        ])
    _write_csv(
        out_dir / "step_f1_by_task.csv",
        ["model", "task_type", "step_idx", "n_total", "f1_macro", "judge_f1_macro", "judge_accuracy"],
        task_csv_rows,
    )

    by_overall_step = compute_step_metrics(rows, ("step_idx",))

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2ca02c", "#d62728", "#9467bd", "#1f77b4", "#ff7f0e"]
    for i, task in enumerate(sorted({t for t, _ in by_task_step})):
        steps = sorted({int(s) for t, s in by_task_step if t == task})
        ys = [_f1(by_task_step[(task, str(s))]) for s in steps]
        ax.plot(steps, ys, marker="o", linewidth=1.6, alpha=0.85,
                color=palette[i % len(palette)], label=TASK_LABEL.get(task, task))

    all_steps = sorted(int(s) for (s,) in by_overall_step)
    ys = [_f1(by_overall_step[(str(s),)]) for s in all_steps]
    ax.plot(all_steps, ys, marker="o", linewidth=2.4, color="black", label="overall", zorder=5)
    for s, y in zip(all_steps, ys):
        if y is not None:
            ax.annotate(f"{y:.2f}", (s, y), fontsize=7, ha="center",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_xticks(all_steps)
    ax.set_xlabel("Step (chain position)", fontsize=10)
    ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"CoT-YAML per-step F1 — {model_name}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = out_dir / "step_f1_overall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def process_swap_curve(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    by_key = compute_step_metrics(rows, ("task_type", "rotation", "step_idx"))
    qid_by_key: dict[tuple, str] = {}
    for r in rows:
        qid_by_key[(r["task_type"], r["rotation"], r["step_idx"])] = r["question_id"]

    csv_rows = []
    for (task, rot, step), m in sorted(by_key.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]))):
        csv_rows.append([
            model_name, task, rot, step, qid_by_key.get((task, rot, step), ""),
            m.get("n_total"), m.get("f1_macro"), m.get("judge_f1_macro", ""),
        ])
    _write_csv(
        out_dir / "swap_step_f1.csv",
        ["model", "task_type", "rotation", "step_idx", "question_id", "n_total", "f1_macro", "judge_f1_macro"],
        csv_rows,
    )

    tasks = sorted({t for t, _, _ in by_key})
    for task in tasks:
        rotations = sorted({int(rot) for t, rot, _ in by_key if t == task})
        steps = sorted({int(s) for t, _, s in by_key if t == task})
        if not rotations or not steps:
            continue
        cmap = cm.get_cmap("tab10", max(len(rotations), 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, rot in enumerate(rotations):
            ys = [_f1(by_key[(task, str(rot), str(s))]) if (task, str(rot), str(s)) in by_key else None
                  for s in steps]
            ax.plot(steps, ys, marker="o", linewidth=1.6, color=cmap(i), label=f"Q{rot + 1} first")
        ax.set_xticks(steps)
        ax.set_xlabel("Step (chain position)", fontsize=10)
        ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"CoT-YAML swap-order per-step F1 — {model_name} — "
                     f"{TASK_LABEL.get(task, task)}", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = out_dir / f"swap_step_f1_{task}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"{output_dir} does not exist.")
        return

    any_data = False
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        safe = model_dir.name
        step_path = model_dir / f"ckpt_steps_{safe}.csv"
        swap_path = model_dir / f"ckpt_swap_{safe}.csv"

        if step_path.exists():
            rows, _ = load_ckpt(step_path, fields=STEP_CKPT_FIELDS)
            rows = [r for r in rows if r.get("mode") == "cumulative"]
            if rows:
                any_data = True
                print(f"\n[{safe}] {len(rows)} per-step row(s)")
                process_step_curve(safe, rows, model_dir)

        if swap_path.exists():
            rows, _ = load_ckpt(swap_path, fields=STEP_CKPT_FIELDS)
            rows = [r for r in rows if r.get("mode") == "swap"]
            if rows:
                any_data = True
                print(f"\n[{safe}] {len(rows)} swap row(s)")
                process_swap_curve(safe, rows, model_dir)

    if not any_data:
        print(f"No ckpt_steps_*.csv or ckpt_swap_*.csv found under {output_dir} — "
              f"run `main.py cot-yaml --per_step` or `--swap_datasets ...` first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="outputs_final/cot_yaml/",
                         help="Same --output_dir cot_yaml.py was run with.")
    main(parser.parse_args())
