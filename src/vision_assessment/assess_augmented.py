"""
Stage 6: Augmented Vision Assessment.

Re-runs the vision assessment with two hint conditions to test alignment:

  cls_with_desc   — classification prompt includes a visual description alongside
                    the image. By default uses the model's own predicted description
                    from the baseline run (--baseline_dir). Pass --use_groundtruth_desc
                    to use the synthesised ground-truth description instead.

  desc_with_class — description prompt includes the ground-truth class name alongside
                    the image.

  both            — both conditions applied simultaneously.

Prediction: if cls_with_desc yields a large accuracy jump, the vision tower is
encoding features correctly but the language decoder cannot map them to class
names without a textual bridge — alignment bottleneck (H3).

Usage:
    python main.py vision-augmented \\
        --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \\
        --baseline_dir       outputs/vision_assessment/ \\
        --augment            both \\
        --models             google/gemma-4-26B-A4B-it \\
        --output_dir         outputs/vision_assessment_augmented/
"""

import argparse
import csv as _csv
import gc
import json
import multiprocessing
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from src.common.dataset_config import datasets_by_name
from src.common.io import (
    Tee, load_jsonl, load_prompts, save_json, write_csv,
)
from src.common.metrics import (
    aggregate_by_task, aggregate_rouge, apply_judge,
    compute_classification_metrics, contains_match, judged_summary,
    naive_classification_baseline, naive_rouge_baseline,
    resolve_class, rouge_scores_per_sample,
)
from src.common.metrics import model_color_map
from src.vision_assessment.assess import (
    DESC_PARAMS, CLS_PARAMS,
    _apply_healthy_override, _safe_name, _parse_model_quantization,
    _draw_judge_marker, _draw_naive_line, _hatch,
    build_multimodal_prompt, build_name_map, destroy_llm, load_llm,
    run_batch, _clean_response,
    compute_desc_metrics_per_dataset, compute_cls_metrics_per_dataset,
)


# ── Prompt templates for hint injection ───────────────────────────────────────

_CLS_WITH_DESC_TEMPLATE = (
    "Here is a visual description of what appears in this image:\n"
    "{description}\n\n"
    "{question}"
)

_DESC_WITH_CLASS_TEMPLATE = (
    "The plant condition shown in this image is: {class_label}\n\n"
    "{question}"
)


# ── Load baseline predicted descriptions ──────────────────────────────────────

def load_predicted_descriptions(baseline_dir: Path, model_id: str) -> dict[tuple, str]:
    """
    Load per-image predicted descriptions from a baseline desc_*.csv.
    Returns {(agml_dataset, class_label, image_name): prediction}.
    Falls back to per-class mean if image_name column is absent.
    """
    csv_path = baseline_dir / f"desc_{_safe_name(model_id)}.csv"
    if not csv_path.exists():
        print(f"  [warn] Baseline desc CSV not found: {csv_path}")
        return {}

    lookup: dict[tuple, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            ds  = row.get("agml_dataset", "")
            cls = row.get("class_label", "")
            img = row.get("image_name", "")
            pred = row.get("prediction", "")
            if img:
                lookup[(ds, cls, img)] = pred
            else:
                # No image_name column — store per class (last prediction wins)
                lookup[(ds, cls, "")] = pred
    print(f"  Loaded {len(lookup)} predicted descriptions from {csv_path.name}")
    return lookup


def _get_desc_for_image(
    lookup: dict[tuple, str],
    agml_dataset: str,
    class_label: str,
    image_name: str,
    fallback: str,
) -> str:
    """Return predicted description for this image; fall back to class-level or gt."""
    pred = lookup.get((agml_dataset, class_label, image_name))
    if pred:
        return pred
    pred = lookup.get((agml_dataset, class_label, ""))
    return pred if pred else fallback


# ── Augmented run_model ────────────────────────────────────────────────────────

def run_model_augmented(
    model_id: str,
    records: list[dict],
    datasets_cfg: dict,
    classification_suffix: str,
    desc_prompt_template: str,
    cls_prompt_template: str,
    augment: str,                    # "cls_with_desc" | "desc_with_class" | "both"
    desc_lookup: dict[tuple, str],   # predicted descriptions from baseline
    use_groundtruth_desc: bool,
    n_samples: int | None,
    output_dir: Path,
    quantization: str | None = None,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
) -> tuple[list[dict], list[dict]]:
    llm, tokenizer = load_llm(
        model_id, quantization=quantization,
        max_num_seqs=max_num_seqs, enforce_eager=enforce_eager,
    )

    desc_rows: list[dict] = []
    cls_rows:  list[dict] = []
    agml_cache: dict[str, dict[str, str]] = {}

    t_start = time.perf_counter()

    do_cls_hint  = augment in ("cls_with_desc", "both")
    do_desc_hint = augment in ("desc_with_class", "both")

    for record in records:
        meta        = record["metadata"]
        agml_name   = meta["agml_dataset"]
        class_label = meta["class_label"]
        task_type   = meta["task_type"]
        crop        = meta["crop"]
        image_names = meta["image_names"]
        synth_desc  = record["messages"][1]["content"]
        visual_desc_q = meta["visual_question"]
        visual_cls_q  = meta.get("visual_cls_question")

        if n_samples:
            image_names = image_names[:n_samples]

        if agml_name not in agml_cache:
            print(f"\n  Building image map: {agml_name}...")
            agml_cache[agml_name] = build_name_map(agml_name)
        name_map = agml_cache[agml_name]

        images, valid_names = [], []
        for name in image_names:
            if name in name_map:
                images.append(PILImage.open(name_map[name]).convert("RGB"))
                valid_names.append(name)
            else:
                print(f"    [warn] {name} not in {agml_name}")
        if not images:
            continue

        print(f"  {agml_name} / {class_label}  ({len(images)} images)")

        # ── Description task ──────────────────────────────────────────────────
        if do_desc_hint:
            # Provide class name hint alongside image
            hinted_desc_q = _DESC_WITH_CLASS_TEMPLATE.format(
                class_label=class_label,
                question=visual_desc_q,
            )
        else:
            hinted_desc_q = visual_desc_q

        wrapped_desc_q = desc_prompt_template.format(question=hinted_desc_q)
        desc_preds = run_batch(llm, tokenizer, images, wrapped_desc_q, DESC_PARAMS)

        for pred, img_name in zip(desc_preds, valid_names):
            scores = rouge_scores_per_sample(pred, synth_desc)
            desc_rows.append({
                "agml_dataset": agml_name,
                "class_label":  class_label,
                "task_type":    task_type,
                "crop":         crop,
                "image_name":   img_name,
                "prediction":   pred,
                "ground_truth": synth_desc,
                "desc_hint":    class_label if do_desc_hint else "",
                **scores,
            })

        # ── Classification task ───────────────────────────────────────────────
        if not visual_cls_q:
            continue

        ds_cfg = datasets_cfg.get(agml_name)
        clean_classes = ds_cfg.clean_classes if ds_cfg else []

        if do_cls_hint:
            # Build one augmented prompt per image (each uses its own predicted desc)
            cls_preds = []
            for img, img_name in zip(images, valid_names):
                if use_groundtruth_desc:
                    hint_desc = synth_desc
                else:
                    hint_desc = _get_desc_for_image(
                        desc_lookup, agml_name, class_label, img_name, synth_desc
                    )
                hinted_cls_q = _CLS_WITH_DESC_TEMPLATE.format(
                    description=hint_desc,
                    question=visual_cls_q,
                )
                wrapped_cls_q = cls_prompt_template.format(question=hinted_cls_q)
                from vllm import SamplingParams
                prompt = build_multimodal_prompt(tokenizer, img, wrapped_cls_q)
                out = llm.generate([prompt], sampling_params=CLS_PARAMS)
                cls_preds.append(_clean_response(out[0].outputs[0].text))
        else:
            wrapped_cls_q = cls_prompt_template.format(question=visual_cls_q)
            cls_preds = run_batch(llm, tokenizer, images, wrapped_cls_q, CLS_PARAMS)

        resolved_list = [resolve_class(p, clean_classes) for p in cls_preds]
        for pred, img_name, resolved in zip(cls_preds, valid_names, resolved_list):
            correct = resolved is not None and resolved.lower() == class_label.lower()
            c_match = contains_match(resolved or pred, class_label)
            # Retrieve the description used as hint (for logging)
            if do_cls_hint:
                if use_groundtruth_desc:
                    hint_used = synth_desc[:120]
                else:
                    hint_used = _get_desc_for_image(
                        desc_lookup, agml_name, class_label, img_name, synth_desc
                    )[:120]
            else:
                hint_used = ""
            cls_rows.append({
                "agml_dataset":  agml_name,
                "class_label":   class_label,
                "task_type":     task_type,
                "crop":          crop,
                "image_name":    img_name,
                "prediction":    pred,
                "resolved_class": resolved or "",
                "ground_truth":  class_label,
                "correct":       int(correct),
                "contains_match": int(c_match),
                "resolvable":    int(resolved is not None),
                "cls_hint":      hint_used,
            })

    elapsed = time.perf_counter() - t_start
    print(f"\nInference time: {elapsed:.1f}s")

    destroy_llm(llm)
    del tokenizer

    return desc_rows, cls_rows


# ── Subprocess worker ──────────────────────────────────────────────────────────

def _worker(kwargs: dict) -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.dataset_config import DatasetConfig

    datasets_cfg = {k: DatasetConfig(**v) for k, v in kwargs["datasets_cfg"].items()}
    desc_rows, cls_rows = run_model_augmented(
        model_id=kwargs["model_id"],
        records=kwargs["records"],
        datasets_cfg=datasets_cfg,
        classification_suffix=kwargs["classification_suffix"],
        desc_prompt_template=kwargs["desc_prompt_template"],
        cls_prompt_template=kwargs["cls_prompt_template"],
        augment=kwargs["augment"],
        desc_lookup={tuple(k): v for k, v in kwargs["desc_lookup"]},
        use_groundtruth_desc=kwargs["use_groundtruth_desc"],
        n_samples=kwargs["n_samples"],
        output_dir=Path(kwargs["output_dir"]),
        quantization=kwargs["quantization"],
        max_num_seqs=kwargs.get("max_num_seqs", 64),
        enforce_eager=kwargs.get("enforce_eager", False),
    )
    with open(kwargs["out_path"], "w") as f:
        json.dump({"desc_rows": desc_rows, "cls_rows": cls_rows}, f)


def _run_in_subprocess(fn, kwargs: dict) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=fn, args=(kwargs,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Worker process exited with code {p.exitcode}")


# ── Before / after comparison chart ───────────────────────────────────────────

def plot_before_after(
    baseline_metrics_path: Path,
    summary_metrics: dict,
    augment: str,
    output_dir: Path,
    multimodal_set: set | None = None,
    naive_cls_overall: dict | None = None,
    naive_ov_desc: dict | None = None,
) -> None:
    """
    Single figure, two panels (classification F1 + description ROUGE-SS).
    For each model: one bar group with before (baseline) and after (augmented).
    Judge score markers shown when available for either condition.
    Same naive baseline lines as the standard vision assessment charts.
    """
    if not baseline_metrics_path.exists():
        print(f"  [warn] Baseline metrics not found at {baseline_metrics_path} — skipping before/after chart")
        return

    baseline_data  = json.loads(baseline_metrics_path.read_text())
    baseline_models = baseline_data.get("models", {})
    multimodal_set  = multimodal_set or set()
    models          = list(summary_metrics.keys())
    if not models:
        return

    show_cls  = augment in ("cls_with_desc",  "both")
    show_desc = augment in ("desc_with_class", "both")
    n_panels  = int(show_cls) + int(show_desc)
    if n_panels == 0:
        return

    color_map = model_color_map(models)
    n_models  = len(models)
    group_w   = 0.35          # width of each bar
    gap       = 0.08          # gap between before/after within a group
    x         = np.arange(n_models)

    subplot_w = max(4, n_models * 2.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(subplot_w * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    def _panel(ax, get_before, get_after, get_judge_before, get_judge_after,
               naive_metrics, naive_key, ylabel, title):
        judge_label_drawn = False
        for xi, mid in enumerate(models):
            bval = get_before(mid)
            aval = get_after(mid)
            color = color_map[mid]
            hatch = _hatch(mid, multimodal_set)

            # Before bar — lighter alpha to distinguish
            b_bar = ax.bar(xi - (group_w / 2 + gap / 2), bval or 0, group_w,
                           color=color, alpha=0.45, hatch=hatch,
                           edgecolor="black", linewidth=0.5, label=f"{mid} before")[0]
            # After bar — solid
            a_bar = ax.bar(xi + (group_w / 2 + gap / 2), aval or 0, group_w,
                           color=color, alpha=0.87, hatch=hatch,
                           edgecolor="black", linewidth=0.5, label=f"{mid} after")[0]

            if bval is not None:
                ax.text(b_bar.get_x() + b_bar.get_width() / 2, bval + 0.008,
                        f"{bval:.3f}", ha="center", va="bottom", fontsize=7.5)
            if aval is not None:
                ax.text(a_bar.get_x() + a_bar.get_width() / 2, aval + 0.008,
                        f"{aval:.3f}", ha="center", va="bottom", fontsize=7.5)

            # Judge markers
            jb = get_judge_before(mid)
            if jb is not None:
                _draw_judge_marker(ax, b_bar, jb, label=(not judge_label_drawn))
                judge_label_drawn = True
            ja = get_judge_after(mid)
            if ja is not None:
                _draw_judge_marker(ax, a_bar, ja, label=(not judge_label_drawn))
                judge_label_drawn = True

        _draw_naive_line(ax, naive_metrics, naive_key, label=True)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{mid.split('/')[-1]}\n◀ before  after ▶" for mid in models],
            fontsize=8,
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if show_cls:
        def _avg_cls(per_ds):
            if not per_ds:
                return None
            vals = [v.get("f1_macro", 0.0) for v in per_ds.values()]
            return float(np.mean(vals)) if vals else None

        _panel(
            axes[ax_idx],
            get_before=lambda mid: _avg_cls(baseline_models.get(mid, {}).get("classification_per_dataset", {})),
            get_after=lambda mid: _avg_cls(summary_metrics[mid].get("classification_per_dataset", {})),
            get_judge_before=lambda mid: baseline_models.get(mid, {}).get("judge", {}).get("judge_f1_macro"),
            get_judge_after=lambda mid: summary_metrics[mid].get("judge", {}).get("judge_f1_macro"),
            naive_metrics=naive_cls_overall,
            naive_key="f1_macro",
            ylabel="F1 Macro",
            title="Classification F1 Macro — Before vs After (description hint)",
        )
        ax_idx += 1

    if show_desc:
        _panel(
            axes[ax_idx],
            get_before=lambda mid: baseline_models.get(mid, {}).get("description_overall", {}).get("rouge_ss"),
            get_after=lambda mid: summary_metrics[mid].get("description_overall", {}).get("rouge_ss"),
            get_judge_before=lambda _: None,
            get_judge_after=lambda _: None,
            naive_metrics=naive_ov_desc,
            naive_key="rouge_ss",
            ylabel="ROUGE-SS",
            title="Description ROUGE-SS — Before vs After (class-name hint)",
        )

    fig.suptitle("Hint Injection: Before vs After", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "before_after_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    classification_suffix  = prompts["classification_suffix"]
    desc_prompt_template   = prompts["assessment_descriptive"]
    cls_prompt_template    = prompts["assessment_classification"]

    datasets_cfg = datasets_by_name(Path(args.config))
    ds_to_task   = {name: ds.task_type for name, ds in datasets_cfg.items()}

    active_set = {name for name, ds in datasets_cfg.items() if not ds.skip}
    records = _apply_healthy_override([
        r for r in load_jsonl(Path(args.descriptions_jsonl), args.n_samples)
        if r.get("metadata", {}).get("agml_dataset") in active_set
    ])
    print(f"Records: {len(records)}  |  Active datasets: {len(active_set)}")
    print(f"Augment mode: {args.augment}")
    if args.augment in ("cls_with_desc", "both"):
        src = "ground-truth descriptions" if args.use_groundtruth_desc else f"predicted descriptions from {args.baseline_dir}"
        print(f"  cls hint source: {src}")

    per_model_quant = _parse_model_quantization(args.model_quantization or [])

    all_model_desc_metrics: dict[str, dict[str, dict]] = {}
    all_model_cls_metrics:  dict[str, dict[str, dict]] = {}
    summary_metrics: dict[str, dict] = {}

    for model_id in args.models:
        quant = per_model_quant.get(model_id, args.quantization)
        print(f"\n{'='*60}")
        print(f"Model: {model_id}" + (f"  [quantization={quant}]" if quant else ""))
        print(f"{'='*60}")

        # Load per-model baseline predicted descriptions
        desc_lookup: dict[tuple, str] = {}
        if args.augment in ("cls_with_desc", "both") and not args.use_groundtruth_desc:
            if not args.baseline_dir:
                print("  [warn] --baseline_dir not set; falling back to ground-truth descriptions")
            else:
                desc_lookup = load_predicted_descriptions(Path(args.baseline_dir), model_id)

        worker_out = output_dir / f"_worker_{_safe_name(model_id)}.json"
        _run_in_subprocess(_worker, dict(
            model_id=model_id,
            records=records,
            datasets_cfg={k: vars(v) for k, v in datasets_cfg.items()},
            classification_suffix=classification_suffix,
            desc_prompt_template=desc_prompt_template,
            cls_prompt_template=cls_prompt_template,
            augment=args.augment,
            # tuples can't be JSON keys — serialise as lists
            desc_lookup=[[list(k), v] for k, v in desc_lookup.items()],
            use_groundtruth_desc=args.use_groundtruth_desc,
            n_samples=args.n_samples,
            output_dir=str(output_dir),
            quantization=quant,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            out_path=str(worker_out),
        ))
        worker_result = json.loads(worker_out.read_text())
        worker_out.unlink()
        desc_rows = worker_result["desc_rows"]
        cls_rows  = worker_result["cls_rows"]

        desc_per_ds = compute_desc_metrics_per_dataset(desc_rows)
        cls_per_ds  = compute_cls_metrics_per_dataset(cls_rows)
        desc_overall = aggregate_rouge(desc_rows)
        cls_refs  = [r["class_label"]    for r in cls_rows]
        cls_preds = [r["resolved_class"] for r in cls_rows]
        cls_overall = compute_classification_metrics(cls_refs, cls_preds)
        desc_by_task = aggregate_by_task(desc_per_ds, ds_to_task)
        cls_by_task  = aggregate_by_task(cls_per_ds,  ds_to_task)

        all_model_desc_metrics[model_id] = desc_per_ds
        all_model_cls_metrics[model_id]  = cls_by_task
        summary_metrics[model_id] = {
            "description_overall":      desc_overall,
            "description_per_dataset":  desc_per_ds,
            "description_by_task":      desc_by_task,
            "classification_overall":   cls_overall,
            "classification_per_dataset": cls_per_ds,
            "classification_by_task":   cls_by_task,
        }

        write_csv(output_dir / f"desc_{_safe_name(model_id)}.csv", desc_rows)
        write_csv(output_dir / f"cls_{_safe_name(model_id)}.csv",  cls_rows)

    # Naive baselines
    naive_desc_per_ds: dict[str, list] = defaultdict(list)
    for record in records:
        agml_name = record.get("metadata", {}).get("agml_dataset", "")
        gt = record["messages"][1]["content"]
        naive_desc_per_ds[agml_name].append({"ground_truth": gt})
    naive_desc_metrics = {ds: naive_rouge_baseline(rows) for ds, rows in naive_desc_per_ds.items()}

    naive_cls_by_ds: dict[str, list[str]] = defaultdict(list)
    for r in records:
        agml_name = r.get("metadata", {}).get("agml_dataset", "")
        naive_cls_by_ds[agml_name].append(r.get("metadata", {}).get("class_label", ""))
    naive_cls_per_ds = {ds: naive_classification_baseline(refs) for ds, refs in naive_cls_by_ds.items()}

    # LLM judge (optional)
    if args.judge_model:
        print(f"\n=== LLM Judge: {args.judge_model} ===")
        judge_llm, judge_tok = load_llm(
            args.judge_model, quantization=args.judge_quantization,
            gpu_memory_utilization=0.9,
        )
        for model_id in args.models:
            cls_csv = output_dir / f"cls_{_safe_name(model_id)}.csv"
            if not cls_csv.exists():
                continue
            with open(cls_csv, newline="", encoding="utf-8") as f:
                cls_rows_j = list(_csv.DictReader(f))
            print(f"  Judging {model_id} ({len(cls_rows_j)} rows)...")
            cls_rows_j = apply_judge(
                rows=cls_rows_j, llm=judge_llm, tokenizer=judge_tok,
                system_prompt=prompts["judge_system_vision"],
                judge_template=prompts["judge_classification_template"],
                gt_key="class_label", pred_key="prediction",
                confidence_threshold=args.judge_confidence_threshold,
            )
            write_csv(cls_csv, cls_rows_j)
            summary_metrics[model_id]["judge"] = judged_summary(cls_rows_j)
        del judge_llm, judge_tok
        gc.collect()

    # Save metrics JSON
    full_results = {
        "augment": args.augment,
        "use_groundtruth_desc": args.use_groundtruth_desc,
        "naive": {"description": naive_desc_metrics, "classification": naive_cls_per_ds},
        "models": summary_metrics,
    }
    save_json(output_dir / "vision_assessment_metrics.json", full_results)

    # Summary
    tee = Tee(output_dir / "summary.txt")
    tee.print(f"\n=== Augmented Vision Assessment ({args.augment}) ===")
    tee.print(f"  {'Model':<45} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'R-SS':>6}")
    tee.print(f"  {'-'*66}")
    for model_id, m in summary_metrics.items():
        ov = m["description_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('rouge1',0):>6.3f} {ov.get('rouge2',0):>6.3f} "
                  f"{ov.get('rougeL',0):>6.3f} {ov.get('rouge_ss',0):>6.3f}")
    tee.print(f"\n  {'Model':<45} {'Acc':>6} {'Cont':>6} {'F1-mac':>8} {'Res/Tot':>9}")
    tee.print(f"  {'-'*77}")
    for model_id, m in summary_metrics.items():
        ov = m["classification_overall"]
        tee.print(f"  {model_id:<45} "
                  f"{ov.get('accuracy',0):>6.3f} "
                  f"{ov.get('contains_accuracy',0):>6.3f} "
                  f"{ov.get('f1_macro',0):>8.3f} "
                  f"{ov.get('n_resolvable',0):>4}/{ov.get('n_total',0):<4}")
    tee.close()

    multimodal_set = set(args.multimodal_models or [])
    naive_ov_desc  = {
        metric: float(np.mean([v[metric] for v in naive_desc_metrics.values() if metric in v]))
        for metric in ["rouge1", "rouge2", "rougeL", "rouge_ss"]
        if any(metric in v for v in naive_desc_metrics.values())
    } if naive_desc_metrics else {}
    naive_cls_all_refs = [r.get("metadata", {}).get("class_label", "") for r in records]
    naive_cls_overall  = naive_classification_baseline(naive_cls_all_refs)

    print("\n=== Generating plots ===")
    baseline_metrics_path = Path(args.baseline_dir) / "vision_assessment_metrics.json"
    plot_before_after(
        baseline_metrics_path, summary_metrics, args.augment, output_dir,
        multimodal_set=multimodal_set,
        naive_cls_overall=naive_cls_overall,
        naive_ov_desc=naive_ov_desc,
    )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augmented vision assessment — hint-injection experiment."
    )
    parser.add_argument("--descriptions_jsonl", default="outputs/data_generation/agml_descriptions.jsonl")
    parser.add_argument("--config",  default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--models",  nargs="+", required=True)
    parser.add_argument("--multimodal_models", nargs="*", default=[])
    parser.add_argument(
        "--augment", default="both",
        choices=["cls_with_desc", "desc_with_class", "both"],
        help=(
            "cls_with_desc  — add description hint to classification prompt; "
            "desc_with_class — add class name hint to description prompt; "
            "both           — apply both (default)"
        ),
    )
    parser.add_argument(
        "--baseline_dir", default=None,
        help="Directory containing desc_*.csv from the baseline vision assessment "
             "(used to load per-model predicted descriptions for the cls hint). "
             "Defaults to --output_dir/../vision_assessment/ if not set.",
    )
    parser.add_argument(
        "--use_groundtruth_desc", action="store_true",
        help="Use synthesised ground-truth descriptions as the cls hint instead of "
             "the model's own baseline predictions.",
    )
    parser.add_argument("--output_dir", default="outputs/vision_assessment_augmented/")
    parser.add_argument("--n_samples",  type=int, default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_quantization", default=None, choices=["awq", "gptq", "fp8", "bitsandbytes"])
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--quantization", default=None, choices=["awq", "gptq", "fp8", "bitsandbytes"])
    parser.add_argument("--model_quantization", nargs="*", default=[], metavar="MODEL_ID:METHOD")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()

    # Default baseline_dir
    if args.baseline_dir is None:
        args.baseline_dir = str(Path(args.output_dir).parent / "vision_assessment")

    main(args)
