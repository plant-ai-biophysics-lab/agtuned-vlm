"""
Majority Vote Vision Assessment
=================================
Test-time compute scaling baseline: run the same classification prompt N times
per image with temperature > 0 and take the plurality vote.

Evaluates a sweep of N values (--n_votes 1 4 8 16) so you can plot F1 vs N
and compare against CoT YAML at a single forward pass budget.

Uses vLLM's native n= parameter in SamplingParams — one generate() call per
image produces N completions efficiently from the same KV-cache prefix.

Outputs per model:
  mv_cls_{model}.csv            — one row per (image, n_votes value)
  majority_vote_metrics.json    — F1 / judge F1 per n_votes × dataset

Usage:
    uv run python src/majority_vote.py \\
        --model      google/gemma-4-26B-A4B-it \\
        --output_dir outputs/majority_vote/ \\
        --test_split outputs/linear_probe/test_split.json \\
        --hard_only \\
        --n_samples  30 \\
        --n_votes    1 4 8 16 \\
        --temperature 0.8

    # Plots only:
    uv run python src/majority_vote.py \\
        --model      google/gemma-4-26B-A4B-it \\
        --output_dir outputs/majority_vote/ \\
        --plot_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.metrics import (
    apply_judge, compute_classification_metrics, contains_match,
    judged_summary, resolve_class, sample_class_images,
)
from src.common.dataset_config import datasets_by_name, clean_class_list
from src.common.difficulty_plots import plot_difficulty_correlation_conditions

# Re-use shared helpers from cot_vision
from src.cot_vision import (
    DATASET_TASK_TYPES, DEFAULT_DATASETS,
    _PROMPTS, _THINK_RE, _REPEAT_RE,
    _sys_msg, _cls_question,
    _build_prompt, _load_image, _clean,
    collect_work_items,
    load_vis_per_dataset, select_hard_datasets,
    load_clustering_per_dataset,
    load_aug_per_dataset,
    plot_hard_four_condition,
)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None          # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]


# ── Core logic ────────────────────────────────────────────────────────────────

def plurality_vote(resolved: list[str | None]) -> tuple[str | None, float]:
    """Return (winning_class, vote_fraction) from a list of resolved predictions.

    Ties return (None, fraction) — treated as unresolved.
    """
    valid = [r for r in resolved if r is not None]
    if not valid:
        return None, 0.0
    counts = Counter(valid)
    top_cls, top_n = counts.most_common(1)[0]
    # Tie: another class has the same count
    if len(counts) > 1 and counts.most_common(2)[1][1] == top_n:
        return None, top_n / len(valid)
    return top_cls, top_n / len(valid)


def run_majority_vote(
    items: list[dict],
    llm,
    tokenizer,
    n_votes: int,
    temperature: float,
) -> list[dict]:
    """One generate() call per image producing n_votes completions; plurality vote."""
    params = SamplingParams(
        n=n_votes,
        temperature=temperature,
        max_tokens=64,
        repetition_penalty=1.1,
    )

    print(f"\n[majority_vote] n={n_votes}  T={temperature}  ({len(items)} images)...")
    prompts = []
    for item in tqdm(items, desc="  Building prompts", unit="img", dynamic_ncols=True):
        cls_q = _cls_question(item["task_type"], item["crop"])
        msgs = _sys_msg() + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": cls_q},
        ]}]
        prompts.append(_build_prompt(msgs, tokenizer, _load_image(item)))

    print("  Generating...")
    outputs = llm.generate(prompts, params)
    del prompts

    rows = []
    for item, out in zip(items, outputs):
        raw_preds  = [_clean(c.text) for c in out.outputs]
        resolved   = [resolve_class(p, item["clean_classes"]) for p in raw_preds]
        winner, vote_frac = plurality_vote(resolved)

        # Build row — prediction = winning class text (or best raw pred if unresolved)
        pred_text  = winner if winner is not None else (raw_preds[0] if raw_preds else "")
        correct    = (winner or "").lower() == item["class_label"].lower()
        rows.append({
            "agml_dataset":   item["ds"],
            "class_label":    item["class_label"],
            "task_type":      item["task_type"],
            "crop":           item["crop"] or "",
            "image_name":     item["image_name"],
            "condition":      f"mv_{n_votes}",
            "n_votes":        n_votes,
            "temperature":    temperature,
            "prediction":     pred_text,
            "all_predictions": " | ".join(raw_preds),
            "resolved_class": winner or "",
            "ground_truth":   item["class_label"],
            "correct":        int(correct),
            "contains_match": int(contains_match(winner or pred_text, item["class_label"])),
            "vote_fraction":  round(vote_frac, 3),
            "n_resolved":     sum(1 for r in resolved if r is not None),
        })
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    from collections import defaultdict
    by_cond_ds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_ds[(r["condition"], r["agml_dataset"])].append(r)

    per_condition: dict[str, dict] = {}
    for (cond, ds), ds_rows in sorted(by_cond_ds.items()):
        refs  = [r["class_label"]    for r in ds_rows]
        preds = [r["resolved_class"] for r in ds_rows]
        m     = compute_classification_metrics(refs, preds)
        m.update(judged_summary(ds_rows))
        per_condition.setdefault(cond, {})[ds] = m

    overall: dict[str, dict] = {}
    for cond, per_ds in per_condition.items():
        f1s  = [v["f1_macro"]       for v in per_ds.values()]
        jf1s = [v["judge_f1_macro"] for v in per_ds.values() if "judge_f1_macro" in v]
        overall[cond] = {
            "f1_macro":       float(np.mean(f1s))  if f1s  else 0.0,
            "judge_f1_macro": float(np.mean(jf1s)) if jf1s else None,
            "per_dataset":    per_ds,
        }
    return overall


# ── CSV I/O ───────────────────────────────────────────────────────────────────

_CLS_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "condition", "n_votes", "temperature",
    "prediction", "all_predictions", "resolved_class", "ground_truth",
    "correct", "contains_match", "vote_fraction", "n_resolved",
    "judged_correct", "judge_reasoning",
]


def write_cls_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CLS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scaling_curve(
    metrics: dict,
    vis_per_ds: dict[str, float] | None,
    cot_metrics_path: Path | None,
    output_dir: Path,
) -> None:
    """F1 (and judge F1) vs N — the main majority vote result."""
    import matplotlib.pyplot as plt

    # Sort conditions by n_votes
    mv_conds = sorted(
        [(c, m) for c, m in metrics.items() if c.startswith("mv_")],
        key=lambda x: int(x[0].split("_")[1]),
    )
    if not mv_conds:
        return

    ns    = [int(c.split("_")[1]) for c, _ in mv_conds]
    f1s   = [m["f1_macro"]        for _, m in mv_conds]
    jf1s  = [m.get("judge_f1_macro") for _, m in mv_conds]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, f1s,  "o-", color="#1f77b4", label="Direct F1",   linewidth=1.8)
    if any(j is not None for j in jf1s):
        ax.plot(ns, [j or 0 for j in jf1s], "s--", color="#ff7f0e",
                label="Judged F1", linewidth=1.8)

    # Horizontal reference lines
    if vis_per_ds:
        baseline_f1 = float(np.mean(list(vis_per_ds.values())))
        ax.axhline(baseline_f1, color="#e67e22", linestyle=":", linewidth=1.2,
                   label=f"Vision Baseline (F1={baseline_f1:.3f})")

    # Load CoT YAML reference if available
    if cot_metrics_path and cot_metrics_path.exists():
        with open(cot_metrics_path) as f:
            cot = json.load(f)
        cot_conds = cot.get("conditions", cot)
        for cond_key, label, color in [
            ("cot_yaml",  "CoT YAML",  "#2ca02c"),
            ("self_cot",  "Self-CoT",  "#9467bd"),
        ]:
            if cond_key in cot_conds:
                ref_f1 = cot_conds[cond_key].get("judge_f1_macro") or cot_conds[cond_key].get("f1_macro")
                if ref_f1 is not None:
                    ax.axhline(ref_f1, color=color, linestyle="--", linewidth=1.2,
                               label=f"{label} (F1={ref_f1:.3f})")

    ax.set_xlabel("Number of votes (N)")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Majority Vote — F1 vs N (test-time compute scaling)")
    ax.set_xticks(ns)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "mv_scaling_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_vote_margin(metrics: dict, output_dir: Path) -> None:
    """Calibration: does higher vote fraction → higher accuracy?"""
    import matplotlib.pyplot as plt

    # Collect all rows from stored CSV if available — use per_dataset proxy
    # (full calibration requires raw rows; skip if not wired)
    pass  # placeholder — wired in main() via raw rows


def plot_vote_margin_from_rows(all_rows: list[dict], output_dir: Path) -> None:
    """Scatter: vote_fraction vs correct per image (calibration check)."""
    import matplotlib.pyplot as plt

    # Use highest-N condition only
    max_n = max((r["n_votes"] for r in all_rows), default=0)
    rows = [r for r in all_rows if r["n_votes"] == max_n]
    if not rows:
        return

    fracs   = [r["vote_fraction"] for r in rows]
    correct = [r["correct"]       for r in rows]

    # Bin by vote fraction
    bins = np.linspace(0, 1, 11)
    bin_acc = []
    bin_mid = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = [c for f, c in zip(fracs, correct) if lo <= f < hi]
        if bucket:
            bin_acc.append(np.mean(bucket))
            bin_mid.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fracs, correct, alpha=0.15, s=10, color="#1f77b4")
    ax.plot(bin_mid, bin_acc, "ro-", linewidth=1.5, label="binned accuracy")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect calibration")
    ax.set_xlabel(f"Vote fraction (N={max_n})")
    ax.set_ylabel("Correct (0/1)")
    ax.set_title("Vote Fraction vs Accuracy — Calibration")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "mv_vote_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majority Vote Vision Assessment — test-time compute scaling baseline"
    )
    parser.add_argument("--model",       default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output_dir",  default="outputs/majority_vote/")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max images per class (uses test_split when available)")
    parser.add_argument("--datasets",    nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--hard_only",   action="store_true",
                        help="Only run the 3 worst-performing datasets per task type")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_votes",     nargs="+", type=int, default=[1, 4, 8, 16],
                        help="Vote counts to sweep (e.g. --n_votes 1 4 8 16)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (>0 for diversity)")
    parser.add_argument("--use_4bit",    action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs",  type=int, default=32,
                        help="vLLM max concurrent sequences — keep low with large n_votes")
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/",
                        help="Vision assessment dir — for baseline reference in plots")
    parser.add_argument("--aug_dir",     default="outputs/desc_augmented_vision/",
                        help="Desc-aug output dir — for desc_aug bar in combined chart")
    parser.add_argument("--cot_dir",    default="outputs/cot_vision/",
                        help="CoT vision output dir — loads CoT metrics for scaling curve + combined chart")
    parser.add_argument("--clustering_metrics",
                        default="outputs/clustering_eval/clustering_metrics.json")
    parser.add_argument("--clustering_backbone",
                        default="dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip inference; regenerate plots from existing CSV + JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    test_split: dict | None = None
    split_path = Path(args.test_split)
    if split_path.exists():
        test_split = json.loads(split_path.read_text())
        print(f"[majority_vote] test_split loaded from {split_path}")
    else:
        print(f"[majority_vote] test_split not found — using all available images")

    # Resolve dataset list
    active_datasets = args.datasets
    vis_per_ds = load_vis_per_dataset(Path(args.vis_dir), args.model)
    if args.hard_only:
        print("\n[majority_vote] --hard_only: selecting 3 worst datasets per task type...")
        if not vis_per_ds:
            print("  [warn] No vision baseline found — falling back to all datasets")
        else:
            active_datasets = select_hard_datasets(vis_per_ds, n_per_task=3)
            print(f"  Selected {len(active_datasets)} datasets: {active_datasets}")

    metrics_path = output_dir / "majority_vote_metrics.json"
    cls_csv      = output_dir / f"mv_cls_{safe_model}.csv"

    if args.plot_only:
        if not metrics_path.exists():
            print(f"[error] --plot_only requires {metrics_path} — run inference first")
            return
        with open(metrics_path) as f:
            saved = json.load(f)
        metrics  = saved.get("conditions", saved)
        all_rows = []
        if cls_csv.exists():
            with open(cls_csv, newline="", encoding="utf-8") as f:
                all_rows = list(csv.DictReader(f))
            # Coerce types
            for r in all_rows:
                r["n_votes"]      = int(r.get("n_votes", 1))
                r["correct"]      = int(r.get("correct", 0))
                r["vote_fraction"] = float(r.get("vote_fraction", 0.0))
        print(f"[majority_vote] plot_only: loaded from {metrics_path}")
    else:
        # Collect work items once — reused across all n_votes
        print("\n=== Collecting work items ===")
        t0    = time.perf_counter()
        items = collect_work_items(active_datasets, args.n_samples, test_split)
        if not items:
            print("[error] No work items collected")
            return

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            quantization="bitsandbytes" if args.use_4bit else None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt={"image": 1},
        )
        tokenizer = llm.get_tokenizer()

        all_rows: list[dict] = []
        for n in sorted(args.n_votes):
            print(f"\n=== Phase: n_votes={n} ===")
            rows = run_majority_vote(items, llm, tokenizer, n, args.temperature)
            all_rows.extend(rows)

        # Judge all rows together
        print(f"\n=== Judging ({len(all_rows)} rows) ===")
        all_rows = apply_judge(
            rows=all_rows,
            llm=llm,
            tokenizer=tokenizer,
            system_prompt=_PROMPTS.get("judge_system_vision", ""),
            judge_template=_PROMPTS["judge_classification_template"],
            gt_key="ground_truth",
            pred_key="prediction",
        )

        elapsed = time.perf_counter() - t0
        print(f"\nTotal time: {elapsed:.1f}s  ({len(all_rows)} rows)")

        # Destroy vLLM before heavy plotting
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm, tokenizer
        gc.collect()
        import torch; torch.cuda.empty_cache()

        write_cls_csv(all_rows, cls_csv)
        metrics = compute_metrics(all_rows)
        with open(metrics_path, "w") as f:
            json.dump({"model": args.model, "n_samples": args.n_samples,
                       "n_votes": args.n_votes, "temperature": args.temperature,
                       "conditions": metrics}, f, indent=2)
        print(f"  Saved: {metrics_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Majority Vote Summary ─────────────────────────────")
    for cond in sorted(metrics, key=lambda c: int(c.split("_")[1]) if "_" in c else 0):
        m   = metrics[cond]
        jf1 = m.get("judge_f1_macro")
        jf1_str = f"  judge_F1={jf1:.3f}" if jf1 is not None else ""
        print(f"  {cond:<12}  F1={m['f1_macro']:.3f}{jf1_str}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    cot_metrics_path = Path(args.cot_dir) / "cot_vision_metrics.json"

    print("\n=== Generating plots ===")
    plot_scaling_curve(
        metrics, vis_per_ds,
        cot_metrics_path=cot_metrics_path,
        output_dir=output_dir,
    )
    if all_rows:
        plot_vote_margin_from_rows(all_rows, output_dir)

    # Difficulty correlation (clustering) vs judged F1 per n_votes condition
    def _judged_f1_per_ds(cond_key: str) -> dict[str, float]:
        per_ds = metrics.get(cond_key, {}).get("per_dataset", {})
        return {ds: m["judge_f1_macro"] for ds, m in per_ds.items()
                if "judge_f1_macro" in m}

    conditions_per_ds: dict[str, dict[str, float]] = {}
    if vis_per_ds:
        conditions_per_ds["Vision Baseline"] = vis_per_ds
    for cond in sorted(metrics, key=lambda c: int(c.split("_")[1]) if "_" in c else 0):
        conditions_per_ds[cond.replace("mv_", "N=")] = _judged_f1_per_ds(cond)

    if conditions_per_ds:
        plot_difficulty_correlation_conditions(
            conditions_per_ds,
            clustering_metrics_path=Path(args.clustering_metrics),
            output_dir=output_dir / "difficulty_correlation",
            backbone=args.clustering_backbone,
            filename="difficulty_vs_perf_majority_vote.png",
            title="Difficulty vs Judged F1 — Majority Vote (N sweep)",
        )

    # Combined hard-condition chart (MV + CoT + baseline + desc-aug)
    if args.hard_only and vis_per_ds:
        cot_metrics: dict | None = None
        if cot_metrics_path.exists():
            with open(cot_metrics_path) as f:
                cot_saved = json.load(f)
            cot_metrics = cot_saved.get("conditions", cot_saved)
            print(f"  Loaded CoT metrics from {cot_metrics_path}")
        else:
            print(f"  [info] No CoT metrics at {cot_metrics_path} — CoT bars omitted")

        aug_per_ds = load_aug_per_dataset(Path(args.aug_dir), args.model)

        mv_datasets = sorted({ds for cond in metrics.values()
                               for ds in cond.get("per_dataset", {})})

        plot_hard_four_condition(
            mv_datasets, vis_per_ds,
            aug_per_ds  = aug_per_ds,
            cot_metrics = cot_metrics or {},
            output_dir  = output_dir,
            vis_dir     = Path(args.vis_dir),
            aug_dir     = Path(args.aug_dir),
            model_id    = args.model,
            n_samples   = args.n_samples,
            mv_metrics  = metrics,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
