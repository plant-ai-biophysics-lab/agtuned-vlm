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
  anchor_recovery.csv          — (--swap_datasets data only) per task_type/rotation/
                                   step_position: anchor_rate (P(final correct |
                                   step-k correct)) and recovery_rate (P(final
                                   correct | step-k wrong))
  anchor_recovery_<task>.png   — (--swap_datasets data only) anchor + recovery
                                   rate vs. step position, one line per rotation
  diagnosis_stability.csv      — (--swap_datasets data only) per task_type/
                                   rotation/stability bucket (stable = predicted
                                   class never changed across the chain, unstable
                                   = changed at least once): final judged F1
  diagnosis_stability_<task>.png — (--swap_datasets data only) final judged F1,
                                   stable vs. unstable, per rotation

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


def _auto_ylim(ax, *value_lists) -> None:
    """Scale the y-axis to the actual plotted F1 values instead of a fixed
    [0, 1] — judged F1 on this pipeline's data clusters well under 1.0 (see
    stage2_f1_by_task.png's fix), so a fixed full-range axis wastes most of
    the plot and compresses exactly the step-to-step differences these
    charts exist to show into an illegibly thin band."""
    vals = [v for values in value_lists for v in values if v is not None]
    top = max(vals) if vals else 0.0
    ax.set_ylim(0, max(top * 1.15, 0.05))


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
    all_ys: list = []
    for i, task in enumerate(sorted({t for t, _ in by_task_step})):
        steps = sorted({int(s) for t, s in by_task_step if t == task})
        ys = [_f1(by_task_step[(task, str(s))]) for s in steps]
        all_ys.extend(ys)
        ax.plot(steps, ys, marker="o", linewidth=1.6, alpha=0.85,
                color=palette[i % len(palette)], label=TASK_LABEL.get(task, task))

    all_steps = sorted(int(s) for (s,) in by_overall_step)
    ys = [_f1(by_overall_step[(str(s),)]) for s in all_steps]
    all_ys.extend(ys)
    ax.plot(all_steps, ys, marker="o", linewidth=2.4, color="black", label="overall", zorder=5)
    for s, y in zip(all_steps, ys):
        if y is not None:
            ax.annotate(f"{y:.2f}", (s, y), fontsize=7, ha="center",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_xticks(all_steps)
    ax.set_xlabel("Step (chain position)", fontsize=10)
    ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
    _auto_ylim(ax, all_ys)
    ax.set_title(f"CoT-YAML per-step F1 — {model_name}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = out_dir / "step_f1_overall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_anchor_recovery(rows: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """{task_type: {rotation: {step_position: {anchor_rate, recovery_rate,
    n_correct, n_wrong}}}} — step_position runs 1..n_steps-1 (the final step
    is excluded; comparing it to itself is trivial).

    anchor_rate(pos)   = P(final correct | step-pos correct)  — does being
                         right early "anchor" the model to a right final answer?
    recovery_rate(pos) = P(final correct | step-pos wrong)    — can the model
                         still recover to a right final answer after an early
                         miss?

    Only uses rows with judged_correct set. Ports
    src/cot_step_analysis.py's compute_swap_step_anchor_rates onto the
    current STEP_CKPT_FIELDS row shape."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct", "") == "":
            continue
        key = (r["agml_dataset"], r["image_name"], r["class_label"], r["task_type"], r["rotation"])
        groups[key].append(r)

    stats: dict[tuple, dict] = defaultdict(
        lambda: {"correct": 0, "correct_final_correct": 0, "wrong": 0, "wrong_final_correct": 0}
    )
    for (_ds, _img, _cls, task, rot), grp in groups.items():
        grp.sort(key=lambda r: int(r["step_idx"]))
        if len(grp) < 2:
            continue
        final_ok = bool(int(grp[-1]["judged_correct"]))
        for pos in range(len(grp) - 1):
            step_ok = bool(int(grp[pos]["judged_correct"]))
            s = stats[(task, rot, pos + 1)]
            if step_ok:
                s["correct"] += 1
                s["correct_final_correct"] += int(final_ok)
            else:
                s["wrong"] += 1
                s["wrong_final_correct"] += int(final_ok)

    result: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (task, rot, pos), s in stats.items():
        result[task][rot][pos] = {
            "anchor_rate": (s["correct_final_correct"] / s["correct"]) if s["correct"] else None,
            "recovery_rate": (s["wrong_final_correct"] / s["wrong"]) if s["wrong"] else None,
            "n_correct": s["correct"], "n_wrong": s["wrong"],
        }
    return result


def write_anchor_recovery(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    by_task = compute_anchor_recovery(rows)
    if not by_task:
        return

    csv_rows = []
    for task, by_rot in by_task.items():
        for rot, by_pos in by_rot.items():
            for pos, s in sorted(by_pos.items()):
                csv_rows.append([model_name, task, rot, pos, s["anchor_rate"], s["recovery_rate"],
                                   s["n_correct"], s["n_wrong"]])
    _write_csv(
        out_dir / "anchor_recovery.csv",
        ["model", "task_type", "rotation", "step_position", "anchor_rate", "recovery_rate",
         "n_correct", "n_wrong"],
        csv_rows,
    )

    for task, by_rot in by_task.items():
        rotations = sorted(by_rot, key=int)
        positions = sorted({p for by_pos in by_rot.values() for p in by_pos})
        if not rotations or not positions:
            continue
        cmap = cm.get_cmap("tab10", max(len(rotations), 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for i, rot in enumerate(rotations):
            by_pos = by_rot[rot]
            anchor = [(by_pos.get(p) or {}).get("anchor_rate") for p in positions]
            recovery = [(by_pos.get(p) or {}).get("recovery_rate") for p in positions]
            anchor = [a * 100 if a is not None else None for a in anchor]
            recovery = [r * 100 if r is not None else None for r in recovery]
            color = cmap(i)
            label = f"Q{int(rot) + 1} first"
            ax1.plot(positions, anchor, marker="o", color=color, label=label)
            ax2.plot(positions, recovery, marker="o", color=color, label=label)

        for ax, title, ylabel in [
            (ax1, "Anchor rate", "Anchor rate (%)\nstep-k right → final right"),
            (ax2, "Recovery rate", "Recovery rate (%)\nstep-k wrong → final right"),
        ]:
            ax.set_xticks(positions)
            ax.set_xlabel("Step position in chain", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.25)
        ax2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle(f"CoT-YAML anchor/recovery — {model_name} — {TASK_LABEL.get(task, task)}", fontsize=12)
        plt.tight_layout()
        path = out_dir / f"anchor_recovery_{task}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def tag_diagnosis_stability(rows: list[dict]) -> list[dict]:
    """One row per (image, rotation) — its FINAL-step row, augmented with
    n_changes (how many times the predicted class flipped between
    consecutive steps across the whole chain) and a stable/unstable bucket.
    Lets diagnosis stability be related to the final judged outcome via
    compute_step_metrics() (grouping by task_type/rotation/stability),
    reusing the same shared scoring rather than a bespoke comparison."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["agml_dataset"], r["image_name"], r["class_label"], r["task_type"], r["rotation"])
        groups[key].append(r)

    tagged = []
    for grp in groups.values():
        grp.sort(key=lambda r: int(r["step_idx"]))
        preds = [(r.get("resolved_class") or r.get("prediction") or "").strip().lower() for r in grp]
        n_changes = sum(1 for a, b in zip(preds, preds[1:]) if a != b)
        final = dict(grp[-1])
        final["n_changes"] = n_changes
        final["stability"] = "stable" if n_changes == 0 else "unstable"
        tagged.append(final)
    return tagged


def write_diagnosis_stability(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    tagged = tag_diagnosis_stability(rows)
    if not tagged:
        return
    by_bucket = compute_step_metrics(tagged, ("task_type", "rotation", "stability"))

    csv_rows = []
    for (task, rot, bucket), m in sorted(by_bucket.items()):
        csv_rows.append([model_name, task, rot, bucket, m.get("n_total"),
                           m.get("f1_macro"), m.get("judge_f1_macro", ""), m.get("judge_accuracy", "")])
    _write_csv(
        out_dir / "diagnosis_stability.csv",
        ["model", "task_type", "rotation", "stability", "n_total", "f1_macro",
         "judge_f1_macro", "judge_accuracy"],
        csv_rows,
    )

    tasks = sorted({t for t, _, _ in by_bucket})
    for task in tasks:
        rotations = sorted({rot for t, rot, _ in by_bucket if t == task}, key=int)
        if not rotations:
            continue
        x = range(len(rotations))
        stable_ys = [_f1(by_bucket.get((task, rot, "stable"), {})) for rot in rotations]
        unstable_ys = [_f1(by_bucket.get((task, rot, "unstable"), {})) for rot in rotations]
        n_stable = [by_bucket.get((task, rot, "stable"), {}).get("n_total", 0) for rot in rotations]
        n_unstable = [by_bucket.get((task, rot, "unstable"), {}).get("n_total", 0) for rot in rotations]

        fig, ax = plt.subplots(figsize=(7, 5))
        w = 0.35
        ax.bar([i - w / 2 for i in x], stable_ys, width=w, color="#2ca02c",
               label="stable (prediction never changed)")
        ax.bar([i + w / 2 for i in x], unstable_ys, width=w, color="#d62728",
               label="unstable (prediction changed ≥1x)")
        for i, (sy, uy, ns, nu) in enumerate(zip(stable_ys, unstable_ys, n_stable, n_unstable)):
            if sy is not None:
                ax.annotate(f"{sy:.2f}\n(n={ns})", (i - w / 2, sy), fontsize=6.5, ha="center",
                            va="bottom", xytext=(0, 3), textcoords="offset points")
            if uy is not None:
                ax.annotate(f"{uy:.2f}\n(n={nu})", (i + w / 2, uy), fontsize=6.5, ha="center",
                            va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Q{int(r) + 1} first" for r in rotations], fontsize=8)
        _auto_ylim(ax, stable_ys, unstable_ys)
        ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
        ax.set_title(f"CoT-YAML diagnosis stability vs. final F1 — {model_name} — "
                     f"{TASK_LABEL.get(task, task)}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        path = out_dir / f"diagnosis_stability_{task}.png"
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
        all_ys: list = []
        for i, rot in enumerate(rotations):
            ys = [_f1(by_key[(task, str(rot), str(s))]) if (task, str(rot), str(s)) in by_key else None
                  for s in steps]
            all_ys.extend(ys)
            ax.plot(steps, ys, marker="o", linewidth=1.6, color=cmap(i), label=f"Q{rot + 1} first")
        ax.set_xticks(steps)
        ax.set_xlabel("Step (chain position)", fontsize=10)
        ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
        _auto_ylim(ax, all_ys)
        ax.set_title(f"CoT-YAML swap-order per-step F1 — {model_name} — "
                     f"{TASK_LABEL.get(task, task)}", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = out_dir / f"swap_step_f1_{task}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    write_anchor_recovery(model_name, rows, out_dir)
    write_diagnosis_stability(model_name, rows, out_dir)


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
