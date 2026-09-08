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
  answer_change_frequency.csv      — (--swap_datasets data only) per task_type/n_changes
                                       (# times the predicted class flipped across the
                                       whole chain, aggregated over rotations): n_images, fraction
  answer_change_frequency_<task>.png — (--swap_datasets data only) distribution of #
                                       answer changes per chain, per task type
  reasoning_dose_response.csv       — (--swap_datasets data only) per task_type/
                                        rotation/n_correct_steps: n_images, final_accuracy —
                                        does getting MORE of the intermediate steps right
                                        predict a correct final answer?
  reasoning_perfect_vs_imperfect.csv — (--swap_datasets data only) per task_type/rotation/
                                        {all_correct, any_wrong}: n_images, final_accuracy —
                                        the starkest version of the same question.
  reasoning_consistency_<task>.png   — (--swap_datasets data only) both of the above,
                                        side by side, one line/bar-pair per rotation

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
import json
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


def process_step_curve(
    model_name: str, rows: list[dict], out_dir: Path, final_metrics: dict | None = None,
) -> None:
    """final_metrics: the model's own metrics.json (compute_model_metrics'
    output — "overall" + "by_task"), from the REGULAR (non --per_step)
    ckpt_<model>.csv run — i.e. the official CoT-YAML number quoted
    everywhere else in this project (cot-compare's bars/violin/boxplot).
    --per_step's own step-N value is a separate inference call and can
    drift slightly from that official number even though both represent
    "classify after the full chain" — plotting it as an explicit extra
    point completes the curve against the number that actually gets
    reported, rather than leaving the reader to assume step-N already is
    that number."""
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
    all_steps_seen = sorted({int(s) for (s,) in by_overall_step})
    final_x = (all_steps_seen[-1] + 1) if all_steps_seen else 1

    def _plot_line(steps, ys, color, label, linewidth, zorder, final_y):
        ax.plot(steps, ys, marker="o", linewidth=linewidth, alpha=0.85 if zorder < 5 else 1.0,
                 color=color, label=label, zorder=zorder)
        if final_y is not None and steps:
            # Dashed segment + a distinct star marker for the OFFICIAL final
            # F1 (from the regular, non-per-step run) — visually separate
            # from the per-step chain itself, since it's a different
            # inference call, not literally "one more step."
            ax.plot([steps[-1], final_x], [ys[-1], final_y], linestyle="--",
                     linewidth=linewidth * 0.7, color=color, alpha=0.6, zorder=zorder)
            ax.plot([final_x], [final_y], marker="*", markersize=13,
                     color=color, markeredgecolor="black", markeredgewidth=0.6, zorder=zorder + 1)
            ax.annotate(f"{final_y:.2f}", (final_x, final_y), fontsize=7, ha="center",
                        xytext=(0, 8), textcoords="offset points")

    for i, task in enumerate(sorted({t for t, _ in by_task_step})):
        steps = sorted({int(s) for t, s in by_task_step if t == task})
        ys = [_f1(by_task_step[(task, str(s))]) for s in steps]
        all_ys.extend(ys)
        color = palette[i % len(palette)]
        final_y = _f1((final_metrics or {}).get("by_task", {}).get(task, {})) if final_metrics else None
        if final_y is not None:
            all_ys.append(final_y)
        _plot_line(steps, ys, color, TASK_LABEL.get(task, task), 1.6, 3, final_y)

    all_steps = sorted(int(s) for (s,) in by_overall_step)
    ys = [_f1(by_overall_step[(str(s),)]) for s in all_steps]
    all_ys.extend(ys)
    final_overall = _f1((final_metrics or {}).get("overall", {})) if final_metrics else None
    if final_overall is not None:
        all_ys.append(final_overall)
    _plot_line(all_steps, ys, "black", "overall", 2.4, 5, final_overall)
    for s, y in zip(all_steps, ys):
        if y is not None:
            ax.annotate(f"{y:.2f}", (s, y), fontsize=7, ha="center",
                        xytext=(0, 6), textcoords="offset points")

    xticks = list(all_steps)
    xticklabels = [str(s) for s in all_steps]
    if final_metrics is not None:
        xticks.append(final_x)
        xticklabels.append("Final")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Step (chain position)", fontsize=10)
    ax.set_ylabel(f"{_F1_KEY} (falls back to f1_macro if unjudged)", fontsize=9)
    _auto_ylim(ax, all_ys)
    title = f"CoT-YAML per-step F1 — {model_name}"
    if final_metrics is None:
        title += "\n(no metrics.json found — run the regular cot-yaml pass to add the Final marker)"
    ax.set_title(title, fontsize=11)
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
        final_x = positions[-1] + 1
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
            # "Final" — step-N compared to itself is trivial by definition
            # (anchor rate = 100%, recovery rate = 0%), which is exactly why
            # step-N itself is excluded from `positions` above. Plotting it
            # anyway as an explicit, visually-distinct reference point
            # completes the curve into its known endpoint instead of just
            # stopping short of it — same treatment as step_f1_overall.png's
            # "Final" marker.
            for ax, ys, final_y in ((ax1, anchor, 100.0), (ax2, recovery, 0.0)):
                if ys and ys[-1] is not None:
                    ax.plot([positions[-1], final_x], [ys[-1], final_y], linestyle="--",
                             linewidth=1.2, color=color, alpha=0.6, zorder=2)
                ax.plot([final_x], [final_y], marker="*", markersize=11, color=color,
                         markeredgecolor="black", markeredgewidth=0.5, zorder=3)

        for ax, title, ylabel in [
            (ax1, "Anchor rate", "Anchor rate (%)\nstep-k right → final right"),
            (ax2, "Recovery rate", "Recovery rate (%)\nstep-k wrong → final right"),
        ]:
            ax.set_xticks(positions + [final_x])
            ax.set_xticklabels([str(p) for p in positions] + ["Final"])
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


def compute_answer_change_frequency(rows: list[dict]) -> dict[str, dict[int, dict]]:
    """{task_type: {n_changes: {n_images, fraction}}} — aggregated across
    rotations. How often does the predicted class flip at all across a
    chain, and how many times? Reuses tag_diagnosis_stability's per-
    (image, rotation) n_changes rather than recomputing it, so this is
    exactly the same "changed >=1x" split as diagnosis_stability.csv,
    just shown as a full distribution instead of a stable/unstable binary."""
    tagged = tag_diagnosis_stability(rows)
    by_task: dict[str, list[int]] = defaultdict(list)
    for r in tagged:
        by_task[r["task_type"]].append(r["n_changes"])

    result: dict[str, dict[int, dict]] = {}
    for task, changes in by_task.items():
        total = len(changes)
        counts: dict[int, int] = defaultdict(int)
        for n in changes:
            counts[n] += 1
        result[task] = {
            n: {"n_images": c, "fraction": c / total}
            for n, c in sorted(counts.items())
        }
    return result


def write_answer_change_frequency(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    by_task = compute_answer_change_frequency(rows)
    if not by_task:
        return

    csv_rows = []
    for task, by_n in by_task.items():
        for n, s in sorted(by_n.items()):
            csv_rows.append([model_name, task, n, s["n_images"], s["fraction"]])
    _write_csv(
        out_dir / "answer_change_frequency.csv",
        ["model", "task_type", "n_changes", "n_images", "fraction"],
        csv_rows,
    )

    for task, by_n in by_task.items():
        ns = sorted(by_n)
        fracs = [by_n[n]["fraction"] for n in ns]
        counts = [by_n[n]["n_images"] for n in ns]
        total = sum(counts)
        mean_changes = sum(n * c for n, c in zip(ns, counts)) / total if total else 0.0
        pct_any_change = 1.0 - (by_n.get(0, {}).get("fraction") or 0.0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(ns, [f * 100 for f in fracs], color="#1f77b4")
        for n, f, c in zip(ns, fracs, counts):
            ax.annotate(f"{f * 100:.0f}%\n(n={c})", (n, f * 100), fontsize=8, ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(ns)
        ax.set_xlabel("# times predicted class changed across the chain", fontsize=10)
        ax.set_ylabel("% of images", fontsize=10)
        ax.set_title(f"CoT-YAML answer-change frequency — {model_name} — "
                     f"{TASK_LABEL.get(task, task)}\n"
                     f"mean={mean_changes:.2f} changes/chain, "
                     f"{pct_any_change * 100:.0f}% changed at least once", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        path = out_dir / f"answer_change_frequency_{task}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def compute_reasoning_consistency(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """{task_type: {rotation: {"by_n_correct": {n_correct_steps: {n_images,
    final_accuracy}}, "split": {"all_correct"/"any_wrong": {n_images,
    final_accuracy}}}}}

    Tests "if we reason through steps correctly, do we get the correct
    final answer" directly, unlike anchor_rate (which is per-step-position
    and marginal — doesn't ask about the chain as a whole) or diagnosis
    stability (which measures whether the predicted class ever changed,
    not whether it was ever right).

    by_n_correct — dose-response: bucket images by how MANY of the
    (n_steps - 1) intermediate steps were judged correct (order-agnostic —
    "3 right" counts the same whether steps 1,2,3 or 1,2,4 were the
    correct ones), then final_accuracy = fraction of that bucket whose
    final answer was also judged correct. A hypothesis-confirming result
    trends upward with n_correct_steps.

    split — the starkest version: "all_correct" (every intermediate step
    right) vs. "any_wrong" (at least one intermediate step wrong),
    final_accuracy compared directly between the two groups.

    Only uses rows with judged_correct set. Same (task, rot) grouping and
    "final = last step_idx" convention as compute_anchor_recovery."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct", "") == "":
            continue
        key = (r["agml_dataset"], r["image_name"], r["class_label"], r["task_type"], r["rotation"])
        groups[key].append(r)

    by_n_correct: dict[tuple, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_split: dict[tuple, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for (_ds, _img, _cls, task, rot), grp in groups.items():
        grp.sort(key=lambda r: int(r["step_idx"]))
        if len(grp) < 2:
            continue
        final_ok = bool(int(grp[-1]["judged_correct"]))
        step_oks = [bool(int(r["judged_correct"])) for r in grp[:-1]]
        n_correct = sum(step_oks)
        by_n_correct[(task, rot)][n_correct].append(final_ok)
        split = "all_correct" if all(step_oks) else "any_wrong"
        by_split[(task, rot)][split].append(final_ok)

    result: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for key in set(by_n_correct) | set(by_split):
        task, rot = key
        result[task][rot] = {
            "by_n_correct": {
                n: {"n_images": len(oks), "final_accuracy": sum(oks) / len(oks)}
                for n, oks in sorted(by_n_correct[key].items())
            },
            "split": {
                label: {"n_images": len(oks), "final_accuracy": sum(oks) / len(oks)}
                for label, oks in by_split[key].items()
            },
        }
    return result


def write_reasoning_consistency(model_name: str, rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    by_task = compute_reasoning_consistency(rows)
    if not by_task:
        return

    dose_rows, split_rows = [], []
    for task, by_rot in by_task.items():
        for rot, d in by_rot.items():
            for n, s in sorted(d["by_n_correct"].items()):
                dose_rows.append([model_name, task, rot, n, s["n_images"], s["final_accuracy"]])
            for label, s in d["split"].items():
                split_rows.append([model_name, task, rot, label, s["n_images"], s["final_accuracy"]])
    _write_csv(
        out_dir / "reasoning_dose_response.csv",
        ["model", "task_type", "rotation", "n_correct_steps", "n_images", "final_accuracy"],
        dose_rows,
    )
    _write_csv(
        out_dir / "reasoning_perfect_vs_imperfect.csv",
        ["model", "task_type", "rotation", "group", "n_images", "final_accuracy"],
        split_rows,
    )

    for task, by_rot in by_task.items():
        rotations = sorted(by_rot, key=int)
        n_values = sorted({n for d in by_rot.values() for n in d["by_n_correct"]})
        if not rotations or not n_values:
            continue
        cmap = cm.get_cmap("tab10", max(len(rotations), 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for i, rot in enumerate(rotations):
            by_n = by_rot[rot]["by_n_correct"]
            ys = [((by_n.get(n) or {}).get("final_accuracy", None)) for n in n_values]
            ys = [y * 100 if y is not None else None for y in ys]
            ns = [(by_n.get(n) or {}).get("n_images", 0) for n in n_values]
            color = cmap(i)
            label = f"Q{int(rot) + 1} first"
            ax1.plot(n_values, ys, marker="o", color=color, label=label)
            for n, y, cnt in zip(n_values, ys, ns):
                if y is not None:
                    ax1.annotate(f"n={cnt}", (n, y), fontsize=6, ha="center",
                                 xytext=(0, 6), textcoords="offset points", color=color)
        ax1.set_xticks(n_values)
        ax1.set_xlabel("# intermediate steps judged correct (order-agnostic)", fontsize=9)
        ax1.set_ylabel("Final judged accuracy (%)", fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.set_title("Dose-response", fontsize=11)
        ax1.grid(True, alpha=0.25)

        x = range(len(rotations))
        w = 0.35
        all_ys = [((by_rot[rot]["split"].get("all_correct") or {}).get("final_accuracy")) for rot in rotations]
        any_ys = [((by_rot[rot]["split"].get("any_wrong") or {}).get("final_accuracy")) for rot in rotations]
        all_n = [(by_rot[rot]["split"].get("all_correct") or {}).get("n_images", 0) for rot in rotations]
        any_n = [(by_rot[rot]["split"].get("any_wrong") or {}).get("n_images", 0) for rot in rotations]
        all_ys_pct = [y * 100 if y is not None else None for y in all_ys]
        any_ys_pct = [y * 100 if y is not None else None for y in any_ys]
        ax2.bar([i - w / 2 for i in x], all_ys_pct, width=w, color="#2ca02c", label="all steps correct")
        ax2.bar([i + w / 2 for i in x], any_ys_pct, width=w, color="#d62728", label="any step wrong")
        for i, (ay, ry, an, rn) in enumerate(zip(all_ys_pct, any_ys_pct, all_n, any_n)):
            if ay is not None:
                ax2.annotate(f"{ay:.0f}%\n(n={an})", (i - w / 2, ay), fontsize=6.5, ha="center",
                             va="bottom", xytext=(0, 3), textcoords="offset points")
            if ry is not None:
                ax2.annotate(f"{ry:.0f}%\n(n={rn})", (i + w / 2, ry), fontsize=6.5, ha="center",
                             va="bottom", xytext=(0, 3), textcoords="offset points")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([f"Q{int(r) + 1} first" for r in rotations], fontsize=8)
        ax2.set_ylabel("Final judged accuracy (%)", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_title("Perfect chain vs. any miss", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"CoT-YAML reasoning consistency — {model_name} — {TASK_LABEL.get(task, task)}",
                     fontsize=12)
        plt.tight_layout()
        path = out_dir / f"reasoning_consistency_{task}.png"
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
    write_answer_change_frequency(model_name, rows, out_dir)
    write_reasoning_consistency(model_name, rows, out_dir)


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
                metrics_path = model_dir / "metrics.json"
                final_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
                if final_metrics is None:
                    print(f"  [note] {metrics_path.name} not found — step_f1_overall.png will "
                          f"skip the Final marker (run the regular `cot-yaml` pass, not just "
                          f"--per_step, to produce it).")
                process_step_curve(safe, rows, model_dir, final_metrics=final_metrics)

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
