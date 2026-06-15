"""
CoT Eval Experiment
===================
Tests whether a chain-of-thought visual analysis step prior to the classification
question improves F1 on agricultural image tasks.

Hypothesis:
  Prompting the model to reason through structured visual observations
  (symptoms, morphology, damage patterns) before the classification question
  anchors attention to task-relevant features and improves accuracy.

Pipeline (single phase — vLLM only):
  For each class in each dataset:
    Base pass:  [image] + classification_question → answer
    CoT pass:
      Step 1:  [image] + thinking_prompt → structured observations
      Step 2:  [image + thinking_prompt] + [observations] + classification_question → answer
  Compare F1 base vs CoT, with LLM-as-judge on both.

Thinking prompts are task-specific (disease_pest / weed_identification / crop_identification)
and stored in src/config/cot_prompts.yaml.
Base classification questions come from src/config/prompts.yaml.

Outputs (under output_dir/):
  results.json         — aggregate F1 base vs CoT, by task / dataset / class
  class_log.jsonl      — per-class records: responses + CoT observations
  checkpoint.json      — incremental checkpoint
  plots/
    f1_overall.png
    f1_by_task.png
    f1_by_dataset.png
    f1_per_class.png
    cot_length_vs_gain.png  — scatter: does longer analysis → larger F1 gain?
    gain_distribution.png
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import json
import random
import re
import time
from collections import defaultdict

import agml
import numpy as np
import yaml
from PIL import Image as PILImage

from src.common.dataset_config import datasets_by_name, clean_class_list, parse_class_label
from src.common.metrics import resolve_class, compute_classification_metrics, apply_judge

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_checkpoint(output_dir: Path) -> dict:
    p = output_dir / "checkpoint.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_checkpoint(output_dir: Path, ckpt: dict) -> None:
    (output_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── AgML image loading ────────────────────────────────────────────────────────

def load_agml_images(agml_name: str, n_samples: int, seed: int = 42) -> dict[str, list[str]]:
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    classes = loader.classes
    by_class: dict[str, list[str]] = defaultdict(list)
    for path, label_idx in contents.items():
        by_class[classes[int(label_idx)]].append(str(path))
    rng = random.Random(seed)
    return {cls: rng.sample(paths, min(n_samples, len(paths))) for cls, paths in by_class.items()}


# ── vLLM helpers ─────────────────────────────────────────────────────────────

def load_vlm(model_id: str, gpu_util: float, max_model_len: int):
    from vllm import LLM
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
    )
    return llm, llm.get_tokenizer()


def destroy_vlm(llm) -> None:
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[eval] vLLM engine destroyed")


def _apply_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _build_single_turn(tokenizer, image: PILImage.Image, question: str) -> dict:
    """Single-turn prompt: [image] + question."""
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    return {"prompt": _apply_template(tokenizer, msgs), "multi_modal_data": {"image": image}}


def _build_cot_classify_turn(
    tokenizer,
    image: PILImage.Image,
    thinking_prompt: str,
    cot_response: str,
    classification_question: str,
) -> dict:
    """
    Two-turn prompt:
      User:      [image] + thinking_prompt
      Assistant: cot_response  (the structured observations)
      User:      classification_question + suffix
    Image only appears in the first user turn.
    """
    msgs = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": thinking_prompt},
        ]},
        {"role": "assistant", "content": cot_response},
        {"role": "user", "content": [
            {"type": "text", "text": classification_question},
        ]},
    ]
    return {"prompt": _apply_template(tokenizer, msgs), "multi_modal_data": {"image": image}}


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ── Prompt builders ───────────────────────────────────────────────────────────

def get_thinking_prompt(task_type: str, cfg: dict) -> str:
    cot_key = cfg["task_type_to_cot"].get(task_type, "disease_pest")
    return cfg["thinking_prompts"][cot_key].strip()


def get_base_prompt(task_type: str, prompts_cfg: dict, cot_cfg: dict, crop: str | None) -> str:
    prompt_key = cot_cfg["task_type_to_base_prompt"].get(task_type, "visual_classification_disease")
    template = prompts_cfg[prompt_key]
    if "{crop}" in template and crop:
        return template.format(crop=crop)
    return template


# ── Per-class evaluation ──────────────────────────────────────────────────────

def evaluate_class(
    llm,
    tokenizer,
    image_paths: list[str],
    thinking_prompt: str,
    base_prompt: str,
    classification_question: str,
    class_label: str,
    all_classes: list[str],
    base_params,
    cot_params,
    cls_params,
    n_eval: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Three vLLM passes per class:
      1. Base — single-turn classification
      2. CoT thinking — structured visual analysis
      3. CoT classify — classify given the thinking context

    Returns (base_responses, cot_responses, cot_thoughts).
    """
    paths = image_paths[:n_eval]
    images: list[PILImage.Image] = []
    for p in paths:
        try:
            images.append(PILImage.open(p).convert("RGB"))
        except Exception as e:
            print(f"      [warn] load failed {p}: {e}")

    if not images:
        return [], [], []

    # Pass 1: base classification
    base_inputs = [_build_single_turn(tokenizer, img, base_prompt) for img in images]
    base_raw = llm.generate(base_inputs, base_params)
    base_responses = [_clean(o.outputs[0].text) for o in base_raw]

    # Pass 2: CoT thinking
    cot_inputs = [_build_single_turn(tokenizer, img, thinking_prompt) for img in images]
    cot_raw = llm.generate(cot_inputs, cot_params)
    cot_thoughts = [_clean(o.outputs[0].text) for o in cot_raw]

    # Pass 3: classify with CoT context
    cls_inputs = [
        _build_cot_classify_turn(tokenizer, img, thinking_prompt, thought, classification_question)
        for img, thought in zip(images, cot_thoughts)
    ]
    cls_raw = llm.generate(cls_inputs, cls_params)
    cot_responses = [_clean(o.outputs[0].text) for o in cls_raw]

    return base_responses, cot_responses, cot_thoughts


def _class_f1(responses: list[str], class_label: str, all_classes: list[str]) -> tuple[float, float]:
    """Returns (class_f1, resolvability_rate)."""
    from sklearn.metrics import f1_score
    preds = [resolve_class(r, all_classes) for r in responses]
    resolved = [p or "" for p in preds]
    resolvability = sum(1 for p in preds if p) / max(len(preds), 1)
    refs = [class_label] * len(resolved)
    f1 = f1_score(refs, resolved, labels=[class_label], average="macro", zero_division=0)
    return round(float(f1), 4), round(resolvability, 4)


# ── Evaluation phase ──────────────────────────────────────────────────────────

def run_evaluation(
    model_id: str,
    datasets_cfg: dict,
    dataset_names: list[str],
    cfg: dict,
    prompts_cfg: dict,
    checkpoint: dict,
    output_dir: Path,
    force: bool,
    seed: int,
) -> dict:
    from vllm import SamplingParams

    n_eval = int(cfg.get("n_eval", 10))
    temp = float(cfg.get("classifier_temperature", 0.0))
    max_cot = int(cfg.get("max_tokens_cot", 512))
    max_resp = int(cfg.get("max_tokens_response", 64))
    gpu_util = float(cfg.get("gpu_memory_utilization", 0.85))
    max_model_len = int(cfg.get("max_model_len", 8192))

    judge_system = prompts_cfg["judge_system_vision"]
    judge_template = prompts_cfg["judge_classification_template"]
    cls_suffix = prompts_cfg.get("classification_suffix", " Respond with only the class name, nothing else.")

    base_params = SamplingParams(temperature=temp, max_tokens=max_resp, repetition_penalty=1.1)
    cot_params  = SamplingParams(temperature=temp, max_tokens=max_cot,  repetition_penalty=1.05)
    cls_params  = SamplingParams(temperature=temp, max_tokens=max_resp, repetition_penalty=1.1)

    eval_results: dict = checkpoint.get("eval_results", {})
    log_path = output_dir / "class_log.jsonl"

    llm, tokenizer = load_vlm(model_id, gpu_util, max_model_len)

    for agml_name in dataset_names:
        ds_cfg = datasets_cfg.get(agml_name)
        if ds_cfg is None:
            continue

        print(f"\n{'='*60}")
        print(f"[eval] Dataset: {agml_name}  task={ds_cfg.task_type}")

        try:
            sampled = load_agml_images(agml_name, n_eval, seed=seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        all_classes = clean_class_list(ds_cfg.classes)
        thinking_prompt = get_thinking_prompt(ds_cfg.task_type, cfg)

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            if not force and cls_key in eval_results:
                print(f"  [skip] {clean_cls}")
                continue

            print(f"\n  Class: {clean_cls}")

            base_prompt = get_base_prompt(ds_cfg.task_type, prompts_cfg, cfg, crop)
            # Final classification question in CoT second turn includes suffix
            classification_question = base_prompt + cls_suffix

            base_responses, cot_responses, cot_thoughts = evaluate_class(
                llm, tokenizer, image_paths,
                thinking_prompt, base_prompt, classification_question,
                clean_cls, all_classes,
                base_params, cot_params, cls_params, n_eval,
            )

            if not base_responses:
                print(f"    [warn] no images loaded — skipping")
                continue

            base_f1, base_resolvable = _class_f1(base_responses, clean_cls, all_classes)
            cot_f1,  cot_resolvable  = _class_f1(cot_responses,  clean_cls, all_classes)
            print(f"    base  F1={base_f1:.3f}  resolvable={base_resolvable:.2f}")
            print(f"    cot   F1={cot_f1:.3f}  resolvable={cot_resolvable:.2f}  gain={cot_f1-base_f1:+.3f}")

            # Judge
            base_rows = [{"gt": clean_cls, "response": r} for r in base_responses]
            cot_rows  = [{"gt": clean_cls, "response": r} for r in cot_responses]
            base_judged = apply_judge(base_rows, llm, tokenizer, judge_system, judge_template, "gt", "response")
            cot_judged  = apply_judge(cot_rows,  llm, tokenizer, judge_system, judge_template, "gt", "response")
            base_judge = round(sum(r["judged_correct"] for r in base_judged) / max(len(base_judged), 1), 4)
            cot_judge  = round(sum(r["judged_correct"] for r in cot_judged)  / max(len(cot_judged),  1), 4)
            print(f"    judge base={base_judge:.3f}  cot={cot_judge:.3f}")

            mean_cot_len = round(float(np.mean([len(t.split()) for t in cot_thoughts])), 1) if cot_thoughts else 0.0

            entry = {
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "task_type": ds_cfg.task_type,
                "base_f1": base_f1,
                "cot_f1": cot_f1,
                "gain": round(cot_f1 - base_f1, 4),
                "base_resolvable": base_resolvable,
                "cot_resolvable": cot_resolvable,
                "base_judge_acc": base_judge,
                "cot_judge_acc": cot_judge,
                "mean_cot_length_words": mean_cot_len,
                "base_prompt": base_prompt,
                "classification_question": classification_question,
                "cot_prompt_key": cfg["task_type_to_cot"].get(ds_cfg.task_type, "disease_pest"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            eval_results[cls_key] = entry

            append_jsonl(log_path, {
                **entry,
                "base_responses": base_responses,
                "cot_responses": cot_responses,
                "cot_thoughts": cot_thoughts,
            })

            checkpoint["eval_results"] = eval_results
            save_checkpoint(output_dir, checkpoint)

    destroy_vlm(llm)
    return eval_results


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(eval_results: dict) -> dict:
    all_e = list(eval_results.values())
    if not all_e:
        return {}

    base_f1s = [e["base_f1"] for e in all_e]
    cot_f1s  = [e["cot_f1"]  for e in all_e]
    n_improved = sum(1 for e in all_e if e["cot_f1"] > e["base_f1"])

    base_judge = [e["base_judge_acc"] for e in all_e if "base_judge_acc" in e]
    cot_judge  = [e["cot_judge_acc"]  for e in all_e if "cot_judge_acc"  in e]

    by_task: dict[str, list] = defaultdict(lambda: [[], [], [], []])
    by_ds:   dict[str, list] = defaultdict(lambda: [[], [], [], []])
    for e in all_e:
        bj = e.get("base_judge_acc")
        cj = e.get("cot_judge_acc")
        if e.get("task_type"):
            by_task[e["task_type"]][0].append(e["base_f1"])
            by_task[e["task_type"]][1].append(e["cot_f1"])
            if bj is not None: by_task[e["task_type"]][2].append(bj)
            if cj is not None: by_task[e["task_type"]][3].append(cj)
        if e.get("dataset"):
            by_ds[e["dataset"]][0].append(e["base_f1"])
            by_ds[e["dataset"]][1].append(e["cot_f1"])
            if bj is not None: by_ds[e["dataset"]][2].append(bj)
            if cj is not None: by_ds[e["dataset"]][3].append(cj)

    def _mean_pair(v):
        d = {
            "base_f1_macro": round(float(np.mean(v[0])), 4),
            "cot_f1_macro":  round(float(np.mean(v[1])), 4),
            "gain":          round(float(np.mean(v[1])) - float(np.mean(v[0])), 4),
            "n_classes":     len(v[0]),
        }
        if v[2]: d["base_judge_acc_macro"] = round(float(np.mean(v[2])), 4)
        if v[3]: d["cot_judge_acc_macro"]  = round(float(np.mean(v[3])), 4)
        return d

    overall = {
        "base_f1_macro": round(float(np.mean(base_f1s)), 4),
        "cot_f1_macro":  round(float(np.mean(cot_f1s)),  4),
        "gain":          round(float(np.mean(cot_f1s)) - float(np.mean(base_f1s)), 4),
        "n_classes":     len(all_e),
        "n_improved":    n_improved,
        "pct_improved":  round(n_improved / max(len(all_e), 1), 4),
    }
    if base_judge: overall["base_judge_acc_macro"] = round(float(np.mean(base_judge)), 4)
    if cot_judge:  overall["cot_judge_acc_macro"]  = round(float(np.mean(cot_judge)),  4)

    return {
        "overall":    overall,
        "by_task":    {tt: _mean_pair(v) for tt, v in by_task.items()},
        "by_dataset": {ds: _mean_pair(v) for ds, v in by_ds.items()},
        "per_class":  eval_results,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_judge_line(ax, bar, value: float, label: bool = False) -> None:
    x0, w = bar.get_x(), bar.get_width()
    ax.plot([x0, x0 + w], [value, value], color="black", linewidth=2.5,
            solid_capstyle="butt", zorder=5,
            label="judge score" if label else "_nolegend_")
    ax.text(x0 + w / 2, value + 0.005, f"{value:.3f}",
            ha="center", va="bottom", fontsize=6, color="black", zorder=6)


def _bar_chart(
    labels: list[str],
    base_vals: list[float],
    cot_vals: list[float],
    title: str,
    out_path: Path,
    xlabel: str = "",
    rotate: int = 0,
    base_judge_vals: list[float] | None = None,
    judge_vals: list[float] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    x = np.arange(n)
    width = 0.38
    has_bj = base_judge_vals is not None and len(base_judge_vals) == n
    has_j  = judge_vals      is not None and len(judge_vals)      == n

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    def _annotate(bars, color):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color=color)

    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base (F1)", color="#5b8dd9", alpha=0.9)
    _annotate(bars_base, "#5b8dd9")

    bars_cot = ax.bar(x + width / 2, cot_vals, width, label="CoT (F1)", color="#e07b54", alpha=0.9)
    _annotate(bars_cot, "#c0522a")

    judge_label_drawn = False
    if has_bj:
        for bar, jval in zip(bars_base, base_judge_vals):
            _draw_judge_line(ax, bar, jval, label=not judge_label_drawn)
            judge_label_drawn = True
    if has_j:
        for bar, jval in zip(bars_cot, judge_vals):
            _draw_judge_line(ax, bar, jval, label=not judge_label_drawn)
            judge_label_drawn = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_ylabel("F1 / judge accuracy")
    ax.set_ylim(0, 1.12)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f1_overall(results: dict, plots_dir: Path) -> None:
    ov = results.get("overall", {})
    if not ov:
        return
    bj = ov.get("base_judge_acc_macro")
    cj = ov.get("cot_judge_acc_macro")
    _bar_chart(
        ["overall"],
        [ov.get("base_f1_macro", 0)],
        [ov.get("cot_f1_macro",  0)],
        title=(
            f"CoT Eval — Overall F1\n"
            f"n={ov.get('n_classes',0)} classes  "
            f"{ov.get('pct_improved',0):.0%} improved  "
            f"gain={ov.get('gain',0):+.4f}"
        ),
        out_path=plots_dir / "f1_overall.png",
        base_judge_vals=[bj] if bj is not None else None,
        judge_vals=[cj] if cj is not None else None,
    )


def plot_f1_by_task(results: dict, plots_dir: Path) -> None:
    by_task = results.get("by_task", {})
    if not by_task:
        return
    labels = sorted(by_task)
    bj = [by_task[t].get("base_judge_acc_macro") for t in labels]
    cj = [by_task[t].get("cot_judge_acc_macro")  for t in labels]
    has_bj = any(v is not None for v in bj)
    has_cj = any(v is not None for v in cj)
    _bar_chart(
        labels,
        [by_task[t]["base_f1_macro"] for t in labels],
        [by_task[t]["cot_f1_macro"]  for t in labels],
        title="CoT Eval — F1 by Task Type",
        out_path=plots_dir / "f1_by_task.png",
        base_judge_vals=[v or 0.0 for v in bj] if has_bj else None,
        judge_vals=[v or 0.0 for v in cj]       if has_cj else None,
    )


def plot_f1_by_dataset(results: dict, plots_dir: Path) -> None:
    by_ds = results.get("by_dataset", {})
    if not by_ds:
        return
    labels = sorted(by_ds)
    bj = [by_ds[d].get("base_judge_acc_macro") for d in labels]
    cj = [by_ds[d].get("cot_judge_acc_macro")  for d in labels]
    has_bj = any(v is not None for v in bj)
    has_cj = any(v is not None for v in cj)
    _bar_chart(
        labels,
        [by_ds[d]["base_f1_macro"] for d in labels],
        [by_ds[d]["cot_f1_macro"]  for d in labels],
        title="CoT Eval — F1 by Dataset",
        out_path=plots_dir / "f1_by_dataset.png",
        rotate=45,
        base_judge_vals=[v or 0.0 for v in bj] if has_bj else None,
        judge_vals=[v or 0.0 for v in cj]       if has_cj else None,
    )


def plot_f1_per_class(results: dict, plots_dir: Path) -> None:
    per_class = results.get("per_class", {})
    if not per_class:
        return
    items = sorted(per_class.items())
    bj = [v.get("base_judge_acc") for _, v in items]
    cj = [v.get("cot_judge_acc")  for _, v in items]
    has_bj = any(v is not None for v in bj)
    has_cj = any(v is not None for v in cj)
    _bar_chart(
        [v.get("class_label", k) for k, v in items],
        [v["base_f1"] for _, v in items],
        [v["cot_f1"]  for _, v in items],
        title="CoT Eval — F1 per Class",
        out_path=plots_dir / "f1_per_class.png",
        xlabel="class",
        rotate=45,
        base_judge_vals=[v or 0.0 for v in bj] if has_bj else None,
        judge_vals=[v or 0.0 for v in cj]       if has_cj else None,
    )


def plot_cot_length_vs_gain(eval_results: dict, plots_dir: Path) -> None:
    """Scatter: x = mean CoT observation length (words), y = F1 gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, labels, task_types = [], [], [], []
    for cls_key, e in eval_results.items():
        cot_len = e.get("mean_cot_length_words", 0.0)
        gain = e.get("gain", 0.0)
        xs.append(float(cot_len))
        ys.append(float(gain))
        labels.append(e.get("class_label", cls_key))
        task_types.append(e.get("task_type", "unknown"))

    if not xs:
        return

    task_palette = {"disease": "#e07b54", "pest_damage": "#9b59b6",
                    "weed_identification": "#2ecc71", "crop_identification": "#3498db"}
    colors = [task_palette.get(tt, "gray") for tt in task_types]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(xs, ys, c=colors, alpha=0.75, s=80, edgecolors="k", linewidths=0.5)

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5, alpha=0.8)

    ax.axhline(0, color="gray", lw=1.2, ls="--")

    if len(xs) > 2:
        try:
            from scipy import stats as sp_stats
            corr, p = sp_stats.pearsonr(xs, ys)
            ax.text(0.04, 0.96,
                    f"Pearson r = {corr:.3f}  (p={p:.3f})",
                    transform=ax.transAxes, fontsize=10, va="top",
                    color="darkgreen" if corr > 0 else "darkred",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        except ImportError:
            pass

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=col, label=tt)
                    for tt, col in task_palette.items() if tt in task_types]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, title="task type")

    ax.set_xlabel("Mean CoT analysis length (words)")
    ax.set_ylabel("F1 gain (CoT − base)")
    ax.set_title(
        "Does longer chain-of-thought analysis predict larger F1 gain?\n"
        "Each point = one class   |   Above zero line = CoT helps"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = plots_dir / "cot_length_vs_gain.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gain_distribution(eval_results: dict, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [e["gain"] for e in eval_results.values()]
    if not gains:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gains, bins=20, color="#5b8dd9", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="crimson", lw=2, ls="--", label="no gain baseline")
    ax.axvline(float(np.mean(gains)), color="darkorange", lw=2, ls="--",
               label=f"mean gain = {np.mean(gains):+.3f}")
    n_pos = sum(1 for g in gains if g > 0)
    n_neg = sum(1 for g in gains if g < 0)
    ax.set_xlabel("F1 gain (CoT − base)")
    ax.set_ylabel("Number of classes")
    ax.set_title(
        f"Distribution of F1 gains from CoT prior prompting\n"
        f"↑ improved: {n_pos}   ↓ degraded: {n_neg}   → no change: {len(gains)-n_pos-n_neg}"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out_path = plots_dir / "gain_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary_report(results: dict, eval_results: dict, output_dir: Path) -> None:
    ov = results.get("overall", {})
    lines = [
        "CoT Eval Experiment — Results Summary",
        "=" * 60,
        "",
        f"Classes evaluated : {ov.get('n_classes', 0)}",
        f"Base F1 (macro)   : {ov.get('base_f1_macro', 0):.4f}",
        f"CoT F1 (macro)    : {ov.get('cot_f1_macro',  0):.4f}",
        f"Mean gain         : {ov.get('gain', 0):+.4f}",
        f"Classes improved  : {ov.get('n_improved', 0)}  ({ov.get('pct_improved', 0):.1%})",
        "",
        "By task type:",
    ]
    for tt, v in sorted(results.get("by_task", {}).items()):
        lines.append(
            f"  {tt:25s}  base={v['base_f1_macro']:.4f}  cot={v['cot_f1_macro']:.4f}  "
            f"gain={v['gain']:+.4f}  n={v['n_classes']}"
        )

    lines += ["", "By dataset:"]
    for ds, v in sorted(results.get("by_dataset", {}).items()):
        lines.append(
            f"  {ds:40s}  base={v['base_f1_macro']:.4f}  cot={v['cot_f1_macro']:.4f}  "
            f"gain={v['gain']:+.4f}"
        )

    lines += ["", "Top-5 improved classes:"]
    top = sorted(eval_results.values(), key=lambda e: e.get("gain", 0), reverse=True)[:5]
    for e in top:
        lines.append(f"  {e['class_label']:30s}  gain={e['gain']:+.4f}  "
                     f"cot_len={e.get('mean_cot_length_words', 0):.0f} words")

    lines += ["", "Top-5 degraded classes:"]
    bot = sorted(eval_results.values(), key=lambda e: e.get("gain", 0))[:5]
    for e in bot:
        lines.append(f"  {e['class_label']:30s}  gain={e['gain']:+.4f}  "
                     f"cot_len={e.get('mean_cot_length_words', 0):.0f} words")

    report = "\n".join(lines) + "\n"
    path = output_dir / "results_summary.txt"
    path.write_text(report)
    print(f"  Saved: {path}")
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoT Eval — chain-of-thought prior prompting")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--config", default="src/config/cot_prompts.yaml")
    parser.add_argument("--datasets_config", default="src/config/datasets.yaml")
    parser.add_argument("--split_config", default="src/config/dataset_split.yaml")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--output_dir", default="outputs/cot_eval")
    parser.add_argument("--n_eval", type=int, default=None,
                        help="Override n_eval from config")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--plots_only", action="store_true",
                        help="Regenerate plots from existing results.json without re-running inference")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all classes even if already checkpointed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.config))
    if args.n_eval is not None:
        cfg["n_eval"] = args.n_eval
    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization

    prompts_cfg = yaml.safe_load(Path(cfg.get("prompts_config", "src/config/prompts.yaml")).read_text())

    if args.plots_only:
        results_path = output_dir / "results.json"
        if not results_path.exists():
            print(f"No results.json found at {results_path} — run inference first.")
            return
        results = json.loads(results_path.read_text())
        eval_results = results.get("per_class", {})
        print(f"Loaded {len(eval_results)} classes from {results_path}")
    else:
        datasets_cfg  = datasets_by_name(Path(args.datasets_config))
        split_raw     = yaml.safe_load(Path(args.split_config).read_text())

        if args.split == "train":
            dataset_names = split_raw.get("train", [])
        elif args.split == "test":
            dataset_names = split_raw.get("test", [])
        else:
            dataset_names = split_raw.get("train", []) + split_raw.get("test", [])

        dataset_names = [d for d in dataset_names if d in datasets_cfg and not datasets_cfg[d].skip]
        print(f"Datasets : {len(dataset_names)}")
        print(f"Config   : n_eval={cfg['n_eval']}  max_tokens_cot={cfg['max_tokens_cot']}  "
              f"gpu_util={cfg.get('gpu_memory_utilization', 0.85)}")

        checkpoint = load_checkpoint(output_dir)
        eval_results = run_evaluation(
            args.model, datasets_cfg, dataset_names, cfg, prompts_cfg,
            checkpoint, output_dir, args.force, args.seed,
        )

        if not eval_results:
            print("No evaluation results — exiting without plots")
            return

        results = aggregate_results(eval_results)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {output_dir / 'results.json'}")

    print("\nGenerating plots...")
    plot_f1_overall(results, plots_dir)
    plot_f1_by_task(results, plots_dir)
    plot_f1_by_dataset(results, plots_dir)
    plot_f1_per_class(results, plots_dir)
    plot_cot_length_vs_gain(eval_results, plots_dir)
    plot_gain_distribution(eval_results, plots_dir)

    print("\nSummary:")
    write_summary_report(results, eval_results, output_dir)


if __name__ == "__main__":
    main()
