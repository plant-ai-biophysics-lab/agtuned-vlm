"""
Ablation stage — answers four reviewer questions about the verifier method
that the main pipeline's existing conditions can't isolate on their own:

  Q2 (--step selection)      Does the gain come from the rubric, from
                              sampling K candidates, or from the PPT
                              tournament's selection — confounded together
                              in the full method?
  Q3 (--step selection)      Is this genuine test-time scaling? Compares
                              verifier-selection against first-candidate,
                              random, majority-vote, and an oracle-in-pool
                              ceiling, plus F1/cost vs K.
  Q4 (--step calibration)    Is the verifier's top1_win_rate score usable
                              as a confidence signal — separated into
                              within-pool ranking accuracy vs. cross-image
                              calibration, with a selective-risk curve.
  Q5 (--step oracle-leakage) How much of the desc-aug "oracle upper bound"
                              is built from captions of the correct class's
                              own images (leak risk), and does that explain
                              cases where the verifier exceeds it?

--step selection and --step calibration are mostly reconstruction from data
already on disk (existing outputs_final/verifier_cot_k*/ runs' checkpoint
CSVs + verifier_log.jsonl sidecars) — no GPU needed. --step oracle-leakage
is a pure join against dataset/agml_classification.jsonl — no GPU needed
either. --step generate is the only step that loads a model (vLLM): it
produces the two new candidate-pool conditions selection/Q2 needs that
don't already exist anywhere in outputs_final/ (rubric-off K-candidate
sampling, and rubric-on single-candidate generation), restricted to the
same images an existing verifier_cot run already evaluated so every arm
stays apples-to-apples.

All output goes under --output_dir (default ablation/), one subfolder per
step. See the module docstrings of src.common.ablation_selection /
ablation_calibration / ablation_oracle_leakage for the actual logic — this
file is just CLI plumbing + I/O wiring across existing runs.

Usage:
    uv run python main.py ablation --step selection
    uv run python main.py ablation --step calibration
    uv run python main.py ablation --step oracle-leakage
    uv run python main.py ablation --step generate --model google/gemma-4-26B-A4B-it
    uv run python main.py ablation --step all --model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import random
import time

from PIL import Image as PILImage

from src.common.ablation_calibration import (
    calibration_curve, load_win_rate_rows, plot_calibration, ranking_accuracy,
    score_direction_check, selective_risk_curve,
)
from src.common.ablation_oracle_leakage import (
    f1_by_grounding_status, join_desc_aug_rows_with_grounding, load_grounding_status_map,
    rows_by_condition_for_chart, tally_grounding_status,
)
from src.common.ablation_selection import aggregate_arms, score_run
from src.common.dataset_config import datasets_by_name, load_manifest
from src.common.io import append_jsonl, load_prompts
from src.common.method_comparison import plot_overall_judge_comparison
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
# Reused directly, not reimplemented — same convention src/majority_vote.py
# already uses to import private helpers from src/cot_vision.py.
from src.verifier_cot import (
    _cls_question, _load_reference_images, _load_yaml, _sys_msg,
    extract_answer_tag, run_batch_n, safe_model_name,
)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]


# ── Step: generate (Q2's two new conditions) ────────────────────────────────

def run_generate(args: argparse.Namespace) -> None:
    """Two new candidate-pool conditions, restricted to the same
    (agml_dataset, class_label, image_name) keys already evaluated by an
    existing verifier_cot run (--reference_ckpt), so every arm downstream
    is compared on identical images:

      unaided_kN  — plain classification question (no rubric template),
                    K candidates via vLLM's native n= — isolates "just
                    sampling more" from the rubric's own effect.
      rubric_k1   — the rubric fill-in-template question, K=1 — isolates
                    the rubric's effect on generation alone, no pool.

    Checkpointed as JSONL (one line per image, all K raw candidate texts),
    not CSV, since a CSV row can't hold a variable-length candidate list —
    same shape as verifier_log.jsonl's "candidate" rows, so
    ablation_selection.load_candidate_pools can read either.
    """
    prompts = load_prompts(Path(args.prompts))
    vprompts = _load_yaml(Path(args.verifier_prompts))
    manifest = load_manifest(Path(args.manifest))
    ds_cfg = datasets_by_name(Path(args.config))

    reference_images = _load_reference_images(Path(args.reference_ckpt))
    if not reference_images:
        print(f"ERROR: no images found in --reference_ckpt {args.reference_ckpt}")
        sys.exit(1)
    print(f"  [reference] {sum(len(v) for v in reference_images.values())} image(s) across "
          f"{len(reference_images)} (dataset, class) group(s) from {args.reference_ckpt}")

    out_dir = Path(args.output_dir) / "candidates" / args.condition / safe_model_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.jsonl"

    seen: set[tuple[str, str, str]] = set()
    if out_path.exists() and not args.rerun:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen.add((rec["agml_dataset"], rec["class_label"], rec["image_name"]))
        print(f"  [resume] {len(seen)} image(s) already generated in {out_path}")
    elif out_path.exists() and args.rerun:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path.rename(out_path.with_suffix(f".{ts}.bak.jsonl"))

    from src.common.dataset_config import load_hf_samples, parse_class_label

    groups = []
    for (agml_name, clean_label), names in reference_images.items():
        ds = ds_cfg.get(agml_name)
        if ds is None:
            continue
        pending = [n for n in names if (agml_name, clean_label, n) not in seen]
        if not pending:
            continue
        try:
            samples = load_hf_samples(ds.agml_name, manifest, Path(args.images_root),
                                       excluded_classes=ds.excluded_classes)
        except KeyError:
            continue
        path_by_name = {}
        raw_label = None
        for s in samples:
            _, cl = parse_class_label(s["label"], ds.crop)
            if cl == clean_label:
                path_by_name[Path(s["path"]).name] = s["path"]
                raw_label = s["label"]
        if not path_by_name or raw_label is None:
            continue
        task_type = ds.effective_task_type(raw_label)
        groups.append((ds, clean_label, ds.crop, task_type, pending, path_by_name))

    if not groups:
        print("  Nothing to do — all reference images already generated (or none resolvable).")
        return

    n_pending = sum(len(g[4]) for g in groups)
    print(f"  {len(groups)} (dataset, class) group(s), {n_pending} pending image(s), "
          f"condition={args.condition}, k_candidates={args.k_candidates}")

    llm, tokenizer = load_llm(
        args.model, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget, caller_tag="ablation_generate",
    )

    gen_params = SamplingParams(
        temperature=args.gen_temperature, max_tokens=args.gen_max_tokens,
        repetition_penalty=1.1, n=args.k_candidates, seed=args.seed,
    )

    for ds, clean_label, crop_val, task_type, names, path_by_name in groups:
        images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]
        cls_q = _cls_question(task_type, crop_val, prompts)

        if args.condition == "rubric_k1":
            rubric_key = vprompts["task_type_to_rubric"].get(task_type, "disease_pest")
            gen_q = vprompts["structured_templates"][rubric_key].format(question=cls_q)
        else:  # unaided_kN
            gen_q = cls_q

        hist = [_sys_msg(prompts) + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": gen_q},
        ]}] for _ in images]

        gen_out = run_batch_n(llm, tokenizer, images, hist, gen_params, enable_thinking=False)
        for name, cands in zip(names, gen_out):
            for cand_idx, c in enumerate(cands):
                answer, _found = extract_answer_tag(c["text"])
                append_jsonl(out_path, {
                    "agml_dataset": ds.agml_name, "class_label": clean_label,
                    "image_name": name, "call_type": "candidate",
                    "candidate_index": cand_idx, "raw_text": c["text"],
                    "extracted_answer": answer,
                    "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
                })
        print(f"    {ds.agml_name} / {clean_label}: {len(names)} image(s) done")

    destroy_llm(llm)
    del tokenizer
    print(f"  Saved: {out_path}")


# ── Step: selection (Q2 + Q3) ───────────────────────────────────────────────

def _model_filter(args: argparse.Namespace) -> set[str] | None:
    """--models (safe-name-mapped) restricts every reconstruction step to a
    subset of models — e.g. an ablation run scoped to just one variant.
    None (the --models flag omitted/empty) means no restriction."""
    if not getattr(args, "models", None):
        return None
    return {safe_model_name(m) for m in args.models}


def run_selection(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_filter = _model_filter(args)

    all_rows: list[dict] = []
    for run_dir_str in args.verifier_runs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            print(f"  [skip] {run_dir} — not found")
            continue
        condition = run_dir.name
        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter is not None and model_dir.name not in model_filter:
                continue
            ckpt = model_dir / f"ckpt_{model_dir.name}.csv"
            log = model_dir / "verifier_log.jsonl"
            if not ckpt.exists() or not log.exists():
                continue
            rows = score_run(ckpt, log, Path(args.config), model_dir.name, condition,
                              n_random_seeds=args.n_random_seeds)
            print(f"  {condition}/{model_dir.name}: {len(rows)} arm-row(s) from "
                  f"{len({(r['agml_dataset'], r['class_label'], r['image_name']) for r in rows})} image(s)")
            all_rows.extend(rows)

    # New generate-step conditions, if present — scored against a trivial
    # tournament isn't run for these (no verifier_log "tournament_pair"
    # calls exist for a plain candidates.jsonl), so only the
    # pool-reconstructable arms (first/majority/random/oracle_in_pool) are
    # meaningful for them; "verifier" is intentionally absent from their rows.
    candidates_dir = Path(args.output_dir) / "candidates"
    if candidates_dir.exists():
        from src.common.ablation_selection import (
            first_candidate, majority_vote_candidate, oracle_in_pool, random_candidate, resolve_pool,
        )
        from src.common.dataset_config import datasets_by_name as _dbn
        ds_cfg = _dbn(Path(args.config))
        for condition_dir in sorted(candidates_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            for model_dir in sorted(condition_dir.iterdir()):
                if model_filter is not None and model_dir.name not in model_filter:
                    continue
                jsonl_path = model_dir / "candidates.jsonl"
                if not jsonl_path.exists():
                    continue
                pools: dict = {}
                gt_by_key: dict = {}
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        key = (rec["agml_dataset"], rec["class_label"], rec["image_name"])
                        pools.setdefault(key, {})[rec["candidate_index"]] = rec["raw_text"]
                        gt_by_key[key] = rec["class_label"]
                for key, texts in pools.items():
                    ds = ds_cfg.get(key[0])
                    if ds is None:
                        continue
                    ordered = [texts[i] for i in sorted(texts)]
                    resolved = resolve_pool(ordered, ds.clean_classes)
                    gt = gt_by_key[key]
                    base = {"model": model_dir.name, "condition": condition_dir.name,
                            "agml_dataset": key[0], "class_label": key[1], "image_name": key[2],
                            "task_type": ds.effective_task_type(key[1]), "pool_size": len(ordered)}

                    def _correct(pick):
                        return int(pick is not None and pick.strip().lower() == gt.strip().lower())

                    all_rows.append({**base, "arm": "first", "correct": _correct(first_candidate(resolved)),
                                      "n_calls": len(ordered)})
                    all_rows.append({**base, "arm": "majority", "correct": _correct(majority_vote_candidate(resolved)),
                                      "n_calls": len(ordered)})
                    rand_vals = [_correct(random_candidate(resolved, random.Random(s)))
                                 for s in range(args.n_random_seeds)]
                    all_rows.append({**base, "arm": "random", "correct": sum(rand_vals) / len(rand_vals),
                                      "n_calls": len(ordered)})
                    all_rows.append({**base, "arm": "oracle_in_pool", "correct": int(oracle_in_pool(resolved, gt)),
                                      "n_calls": len(ordered)})
                print(f"  [generate-step] {condition_dir.name}/{model_dir.name}: {len(pools)} pool(s) scored")

    if not all_rows:
        print("  Nothing scored — check --verifier_runs paths.")
        return

    agg = aggregate_arms(all_rows)
    agg_path = out_dir / "arms_by_model.csv"
    fields = ["model", "condition", "arm", "task_type", "pool_size", "n", "accuracy", "mean_n_calls"]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(agg, key=lambda r: (r["model"], r["condition"], r["task_type"], r["arm"])):
            w.writerow(r)
    print(f"  Saved: {agg_path}")

    _plot_f1_vs_k(agg, out_dir)
    _plot_cost_vs_k(agg, out_dir)

    summary_path = out_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(agg, indent=2))
    print(f"  Saved: {summary_path}")


def _plot_f1_vs_k(agg: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [r for r in agg if r["task_type"] == "ALL"]
    models = sorted({r["model"] for r in all_rows})
    arms = sorted({r["arm"] for r in all_rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        for arm in arms:
            pts = sorted(
                [(r["pool_size"], r["accuracy"]) for r in all_rows if r["model"] == model and r["arm"] == arm],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=arm)
        ax.set_xlabel("K (pool size)")
        ax.set_ylabel("Direct-match accuracy")
        ax.set_title(model)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "f1_vs_k.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def _plot_cost_vs_k(agg: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [r for r in agg if r["task_type"] == "ALL"]
    models = sorted({r["model"] for r in all_rows})
    arms = sorted({r["arm"] for r in all_rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        for arm in arms:
            pts = sorted(
                [(r["mean_n_calls"], r["accuracy"]) for r in all_rows if r["model"] == model and r["arm"] == arm],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=arm)
        ax.set_xlabel("Mean model calls (generation + comparison)")
        ax.set_ylabel("Direct-match accuracy")
        ax.set_title(model)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "cost_vs_k.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ── Step: calibration (Q4) ──────────────────────────────────────────────────

def run_calibration(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    direction = score_direction_check()
    (out_dir / "score_direction_check.json").write_text(json.dumps(direction, indent=2))
    print(f"  score_direction_check: direction_ok={direction['direction_ok']}")

    model_filter = _model_filter(args)
    summary: dict = {"score_direction_check": direction, "by_run": {}}
    for run_dir_str in args.verifier_runs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            continue
        condition = run_dir.name
        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter is not None and model_dir.name not in model_filter:
                continue
            ckpt = model_dir / f"ckpt_{model_dir.name}.csv"
            log = model_dir / "verifier_log.jsonl"
            if not ckpt.exists() or not log.exists():
                continue
            label = f"{condition}/{model_dir.name}"

            arm_rows = score_run(ckpt, log, Path(args.config), model_dir.name, condition,
                                  n_random_seeds=args.n_random_seeds)
            ranking = ranking_accuracy(arm_rows)

            wr_rows = load_win_rate_rows(ckpt)
            cal = calibration_curve(wr_rows, n_buckets=args.n_calibration_buckets)
            risk = selective_risk_curve(wr_rows, n_thresholds=args.n_risk_thresholds)

            run_out_dir = out_dir / label.replace("/", "__")
            plot_calibration(wr_rows, run_out_dir, label)

            summary["by_run"][label] = {
                "ranking_accuracy": ranking, "calibration_curve": cal, "selective_risk_curve": risk,
            }
            print(f"  {label}: ranking_accuracy={ranking['ranking_accuracy']}, "
                  f"oracle_in_pool_rate={ranking['oracle_in_pool_rate']}")

    summary_path = out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path}")


# ── Step: oracle-leakage (Q5) ───────────────────────────────────────────────

def run_oracle_leakage(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / "oracle_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    status_map = load_grounding_status_map(Path(args.classification_jsonl))
    tally = tally_grounding_status(status_map)
    tally_path = out_dir / "grounding_status_tally.csv"
    with open(tally_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["grounding_status", "n_classes", "fraction", "leaky"])
        w.writeheader()
        for r in tally:
            w.writerow(r)
    print(f"  Saved: {tally_path}")
    for r in tally:
        print(f"    {r['grounding_status']}: {r['n_classes']} ({r['fraction']:.1%}) leaky={r['leaky']}")

    model_filter = _model_filter(args)
    breakdown_rows = []
    for run_dir_str in args.desc_aug_runs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            continue
        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter is not None and model_dir.name not in model_filter:
                continue
            ckpt_candidates = list(model_dir.glob("ckpt_*.csv"))
            if not ckpt_candidates:
                continue
            rows = join_desc_aug_rows_with_grounding(ckpt_candidates[0], status_map)
            by_status = f1_by_grounding_status(rows)
            for status, stats in by_status.items():
                breakdown_rows.append({
                    "run": run_dir.name, "model": model_dir.name, "grounding_status": status,
                    "leaky": status in {"grounded_vision_only", "grounded_web_and_vision"},
                    **stats,
                })

            rows_by_cond = rows_by_condition_for_chart(rows)
            chart_dir = out_dir / f"{run_dir.name}__{model_dir.name}"
            chart_dir.mkdir(parents=True, exist_ok=True)
            chart_path = chart_dir / "f1_by_grounding_status.png"
            try:
                # condition_order must be passed explicitly — these two
                # labels aren't in method_comparison.CONDITION_STYLES'
                # fixed canonical list, and plot_overall_judge_comparison
                # silently drops any condition not in that order otherwise.
                plot_overall_judge_comparison(
                    rows_by_cond, chart_path,
                    title=f"Desc-Aug F1 by grounding leakage — {model_dir.name}",
                    condition_order=list(rows_by_cond.keys()),
                )
                if chart_path.exists():
                    print(f"  Saved: {chart_path}")
            except Exception as e:
                print(f"  [warn] chart failed for {run_dir.name}/{model_dir.name}: {e}")

    breakdown_path = out_dir / "exceeds_upper_bound_breakdown.csv"
    if breakdown_rows:
        fields = ["run", "model", "grounding_status", "leaky", "n", "accuracy", "judged_accuracy"]
        with open(breakdown_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in breakdown_rows:
                w.writerow(r)
        print(f"  Saved: {breakdown_path}")


# ── Step: all ────────────────────────────────────────────────────────────────

def run_all(args: argparse.Namespace) -> None:
    if args.model:
        for condition in ("unaided_kN", "rubric_k1"):
            args.condition = condition
            args.k_candidates = args.k_candidates if condition == "unaided_kN" else 1
            run_generate(args)
    else:
        print("  [skip generate] no --model given — using only pre-existing outputs_final/ runs")

    run_selection(args)
    run_calibration(args)
    run_oracle_leakage(args)

    out_dir = Path(args.output_dir)
    summary_lines = ["# Ablation appendix tables/figures\n"]
    for step_dir, label in [("selection", "Q2/Q3 — selection arms"),
                             ("calibration", "Q4 — calibration"),
                             ("oracle_leakage", "Q5 — oracle leakage")]:
        d = out_dir / step_dir
        if not d.exists():
            continue
        summary_lines.append(f"\n## {label}\n")
        for p in sorted(d.rglob("*")):
            if p.is_file():
                summary_lines.append(f"- `{p.relative_to(out_dir)}`")
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary" / "appendix_tables.md").write_text("\n".join(summary_lines) + "\n")
    print(f"  Saved: {out_dir / 'summary' / 'appendix_tables.md'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    step_fn = {
        "generate": run_generate,
        "selection": run_selection,
        "calibration": run_calibration,
        "oracle-leakage": run_oracle_leakage,
        "all": run_all,
    }[args.step]
    if args.step == "generate" and not args.model:
        print("ERROR: --step generate requires --model")
        sys.exit(1)
    step_fn(args)


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--ablation_config", default="src/config/ablation_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ablation_config", default=_pre_args.ablation_config,
        help="YAML file of defaults — edit it directly for routine changes instead of "
             "passing flags every time. Any flag also passed on the command line overrides it.",
    )
    parser.add_argument("--step", choices=["generate", "selection", "calibration", "oracle-leakage", "all"],
                         required=True)
    parser.add_argument("--output_dir", default="ablation/")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--classification_jsonl", default="dataset/agml_classification.jsonl")
    parser.add_argument("--verifier_runs", nargs="+",
                         default=["outputs_final/verifier_cot_k4", "outputs_final/verifier_cot_k8"])
    parser.add_argument("--desc_aug_runs", nargs="+",
                         default=["outputs_final/desc_augmented_vision", "outputs_final/desc_augmented_vision_v2"])
    parser.add_argument("--models", nargs="+", default=["google/gemma-4-E4B-it"],
                         help="Restricts selection/calibration/oracle-leakage to these models "
                              "(matched against each run dir's model subfolder via safe_model_name, "
                              "i.e. '/' -> '_'). Pass --models with a wider list (or set an empty "
                              "list in --ablation_config) to lift the restriction.")
    parser.add_argument("--n_random_seeds", type=int, default=5)
    parser.add_argument("--n_calibration_buckets", type=int, default=10)
    parser.add_argument("--n_risk_thresholds", type=int, default=20)

    # --step generate only
    parser.add_argument("--model", default=None, help="Required for --step generate.")
    parser.add_argument("--condition", choices=["unaided_kN", "rubric_k1"], default="unaided_kN")
    parser.add_argument("--reference_ckpt", default="outputs_final/verifier_cot_k4/google_gemma-4-26B-A4B-it/"
                                                      "ckpt_google_gemma-4-26B-A4B-it.csv",
                         help="Existing verifier_cot checkpoint whose (dataset, class, image) keys this "
                              "condition is restricted to, so every arm is compared on identical images.")
    parser.add_argument("--k_candidates", type=int, default=4)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_max_tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--verifier_prompts", default="src/config/verifier_prompts.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)

    if _pre_args.ablation_config and Path(_pre_args.ablation_config).exists():
        import yaml as _yaml
        with open(_pre_args.ablation_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests
        if _unknown:
            print(f"[warn] {_pre_args.ablation_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
