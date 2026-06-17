"""
Prompt Optimisation — Stage X
Iteratively improves per-class classification prompts for VLMs on AgML datasets.

Pipeline
--------
Train loop (per class):
  1. Llama-3.2-1B generates N candidate prompts (closed-form, includes class list).
  2. VLM evaluates each candidate on sampled images → F1 score.
  3. Best prompt retained; knowledge-base summary updated after each class.

Test evaluation:
  - Apply best prompt from train (or base if no improvement).
  - Evaluate with open-ended prompt (no class list) for fair comparison.

Outputs (all under output_dir/prompt_optim/):
  run_log.jsonl          — one record per candidate evaluated
  best_prompts.json      — best prompt per (dataset, class) at run end
  summary_history.jsonl  — knowledge-base summary after each class completes
  results.json           — final F1: base vs optimised, per class and overall
  plots/                 — F1 comparison bar chart + t-SNE overlay
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import json
import re
import time
import random
from collections import defaultdict

import agml
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from vllm import LLM, SamplingParams

from src.common.dataset_config import datasets_by_name, clean_class_list, parse_class_label
from src.common.metrics import resolve_class, compute_classification_metrics

# ── regex helpers ────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*{1,3}([^*]*)\*{1,3}|^#{1,6}\s*|^[-*+]\s+", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r'\{([^{}]*)\}')
_KNOWN_PLACEHOLDERS = {"crop", "class_list", "cls"}


def _sanitize_template(template: str) -> str:
    """Escape any {placeholder} that isn't one of our known variables.

    Also escapes lone { or } (unmatched braces from model output) which would
    cause ValueError inside str.format().
    """
    def _replace(m: re.Match) -> str:
        return m.group(0) if m.group(1) in _KNOWN_PLACEHOLDERS else "{{" + m.group(1) + "}}"

    # First escape all complete {placeholder} pairs
    result = _PLACEHOLDER_RE.sub(_replace, template)

    # Then escape any remaining lone braces that str.format() would choke on.
    # We do this char-by-char so we don't disturb the {{ / }} we just wrote.
    out: list[str] = []
    i = 0
    while i < len(result):
        c = result[i]
        if c == '{':
            if i + 1 < len(result) and result[i + 1] == '{':
                out.append('{{')
                i += 2
            else:
                out.append('{{')   # lone '{' → escape it
                i += 1
        elif c == '}':
            if i + 1 < len(result) and result[i + 1] == '}':
                out.append('}}')
                i += 2
            else:
                out.append('}}')   # lone '}' → escape it
                i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)

# ── config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(split_config_path: Path) -> dict[str, list[str]]:
    """Load train/test dataset split from dataset_split.yaml."""
    with open(split_config_path) as f:
        return yaml.safe_load(f)


# ── text cleaning ─────────────────────────────────────────────────────────────

def _clean_response(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    if "<think>" in text.lower():
        text = text[:text.lower().find("<think>")].strip()
    text = _MARKDOWN_RE.sub(r"\1", text)
    text = " ".join(text.split())
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text.strip()


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def _apply_chat_template(tokenizer, prompt: str, system_prompt: str = "") -> str:
    messages = [{"role": "user", "content": prompt}]
    full = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    try:
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        sys_content = " ".join(filter(None, [system_prompt, "/no_think"]))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_content}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=True
        )


def build_multimodal_prompt(
    tokenizer, image: PILImage.Image, question: str, system_prompt: str = ""
) -> dict:
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [user_msg]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def _is_mamba(model_id: str) -> bool:
    name = model_id.lower().split("/")[-1]
    return any(name.startswith(p) for p in {"mamba", "jamba", "zamba", "qwen"})


def load_vlm(model_id: str, gpu_memory_utilization: float = 0.7, max_model_len: int = 6144) -> tuple[LLM, object]:
    max_num_seqs = 32 if _is_mamba(model_id) else 64
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
    )
    return llm, llm.get_tokenizer()


def destroy_llm(llm) -> None:
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
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ── Llama optimizer ───────────────────────────────────────────────────────────

def load_optimizer(model_id: str) -> tuple[LLM, object]:
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.25,
        max_num_seqs=4,
    )
    return llm, llm.get_tokenizer()


# ── AgML image loading ────────────────────────────────────────────────────────

def load_agml_images(
    agml_name: str, n_samples: int, seed: int = 42
) -> dict[str, list[str]]:
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    classes = loader.classes

    by_class: dict[str, list[str]] = defaultdict(list)
    for path, label_idx in contents.items():
        by_class[classes[int(label_idx)]].append(str(path))

    rng = random.Random(seed)
    sampled = {}
    for cls, paths in by_class.items():
        sampled[cls] = rng.sample(paths, min(n_samples, len(paths)))
    return sampled


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_prompt(
    llm: LLM,
    tokenizer,
    prompt_template: str,
    image_paths: list[str],
    crop: str,
    class_label: str,
    all_classes: list[str],
    class_list_str: str,
    cls_params: SamplingParams,
    n_eval: int | None = None,
    system_prompt: str = "",
) -> tuple[float, float, list[str]]:
    """
    Evaluate a prompt template on a set of images for one class.
    Returns (f1_macro_for_class, resolvability_rate, raw_responses).
    f1_macro_for_class is the F1 for the target class against all classes.
    """
    paths = image_paths[:n_eval] if n_eval else image_paths

    question = _sanitize_template(prompt_template).format(
        crop=crop, class_list=class_list_str, cls=class_label
    )

    prompts = []
    for p in paths:
        try:
            img = PILImage.open(p).convert("RGB")
            prompts.append(build_multimodal_prompt(tokenizer, img, question, system_prompt))
        except Exception as e:
            print(f"    [warn] Could not load {p}: {e}")

    if not prompts:
        return 0.0, 0.0, []

    outputs = llm.generate(prompts, sampling_params=cls_params)
    raw_responses = [_clean_response(o.outputs[0].text) for o in outputs]

    refs = [class_label] * len(raw_responses)
    preds = [resolve_class(r, all_classes) for r in raw_responses]
    resolved = [p if p is not None else "" for p in preds]

    metrics = compute_classification_metrics(refs, resolved)
    resolvability = metrics["n_resolvable"] / max(len(resolved), 1)

    # Per-class F1 for this specific class
    from sklearn.metrics import f1_score
    class_f1 = f1_score(
        refs, resolved,
        labels=[class_label], average="macro", zero_division=0
    )
    return round(float(class_f1), 4), round(resolvability, 4), raw_responses


# ── Optimizer candidate generation ───────────────────────────────────────────

def generate_candidates(
    opt_llm: LLM,
    opt_tokenizer,
    meta_system_prompt: str,
    crop: str,
    class_label: str,
    best_prompt: str | None,
    last_prompt: str | None,
    current_summary: str,
    n_candidates: int,
    opt_params: SamplingParams,
) -> list[str]:
    user_content = (
        f"Crop: {crop}\n"
        f"Target class: {class_label}\n\n"
        f"Cross-run insights:\n{current_summary or '(none yet)'}\n\n"
        f"Best prompt so far:\n{best_prompt or '(none)'}\n\n"
        f"Most recent prompt tested:\n{last_prompt or '(none)'}\n\n"
        f"Generate {n_candidates} new improved candidates."
    )
    system = meta_system_prompt.replace("{n_candidates}", str(n_candidates))
    formatted = _apply_chat_template(opt_tokenizer, user_content, system_prompt=system)
    outputs = opt_llm.generate([formatted], sampling_params=opt_params)
    raw = outputs[0].outputs[0].text.strip()

    # Parse JSON array
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            candidates = json.loads(raw[start:end])
            if isinstance(candidates, list):
                return [str(c) for c in candidates[:n_candidates]], raw
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback 1: extract quoted strings
    candidates = re.findall(r'"([^"]{20,})"', raw)
    if candidates:
        return candidates[:n_candidates], raw

    # Fallback 2: numbered list — capture each item including continuation lines
    # Split on "N." / "N)" markers, take everything until the next marker
    items = re.split(r'\n\s*\d+[\.\)]\s+', "\n" + raw)
    candidates = [c.strip() for c in items[1:] if len(c.strip()) >= 20]
    if candidates:
        return [c.replace("\n", " ") for c in candidates[:n_candidates]], raw

    print(f"  [warn] Could not parse candidates from optimizer output: {raw[:200]}")
    return [], raw


# ── Per-class summary + consolidation ────────────────────────────────────────

def generate_class_summary(
    opt_llm: LLM,
    opt_tokenizer,
    class_summary_system: str,
    class_summary_user_template: str,
    crop: str,
    cls: str,
    dataset: str,
    base_f1: float,
    best_f1: float,
    base_prompt: str,
    best_prompt: str,
    prompts_tested: list[dict],
    summary_params: SamplingParams,
) -> str:
    tested_lines = "\n".join(
        f"  iter={e['iteration']} c={e['candidate_idx']} f1={e['f1']:.3f}  {e['prompt'][:120]}"
        for e in prompts_tested
    )
    user_msg = class_summary_user_template.format(
        crop=crop,
        cls=cls,
        dataset=dataset,
        base_f1=round(base_f1, 3),
        best_f1=round(best_f1, 3),
        gain=round(best_f1 - base_f1, 3),
        base_prompt=base_prompt,
        best_prompt=best_prompt,
        prompts_tested=tested_lines or "(none)",
    )
    formatted = _apply_chat_template(opt_tokenizer, user_msg, system_prompt=class_summary_system)
    outputs = opt_llm.generate([formatted], sampling_params=summary_params)
    return _clean_response(outputs[0].outputs[0].text)


_MAX_CONSOLIDATION_CHARS = 2400  # leave room for system prompt + output within 4096 tokens


def consolidate_summaries(
    opt_llm: LLM,
    opt_tokenizer,
    consolidation_system: str,
    consolidation_user_template: str,
    per_class_summaries: list[dict],
    previous_consolidated: str,
    summary_params: SamplingParams,
) -> str:
    # Always include the most recent summaries; drop oldest if over budget
    entries = list(per_class_summaries)
    while len(entries) > 1:
        text = "\n\n---\n\n".join(f"[{e['cls_key']}]\n{e['summary']}" for e in entries)
        if len(text) <= _MAX_CONSOLIDATION_CHARS:
            break
        entries = entries[1:]  # drop oldest
    summaries_text = "\n\n---\n\n".join(
        f"[{e['cls_key']}]\n{e['summary']}"
        for e in entries
    )
    user_msg = consolidation_user_template.format(
        per_class_summaries=summaries_text or "(none yet)",
        previous_consolidated=previous_consolidated or "(empty)",
    )
    formatted = _apply_chat_template(opt_tokenizer, user_msg, system_prompt=consolidation_system)
    outputs = opt_llm.generate([formatted], sampling_params=summary_params)
    return _clean_response(outputs[0].outputs[0].text)


def generate_test_prompt(
    opt_llm: LLM,
    opt_tokenizer,
    test_prompt_system: str,
    test_prompt_user_template: str,
    consolidated_summary: str,
    crop: str,
    prompt_params: SamplingParams,
) -> str:
    user_msg = test_prompt_user_template.format(
        consolidated_summary=consolidated_summary or "(empty)",
        crop=crop,
    )
    formatted = _apply_chat_template(opt_tokenizer, user_msg, system_prompt=test_prompt_system)
    outputs = opt_llm.generate([formatted], sampling_params=prompt_params)
    prompt = _clean_response(outputs[0].outputs[0].text)
    # Explicitly strip any class list the model may have hallucinated in
    prompt = re.sub(r"\s*Choose from:[^.]*\.", "", prompt).strip()
    return prompt


def judge_responses(
    opt_llm: LLM,
    opt_tokenizer,
    responses: list[str],
    true_class: str,
    crop: str,
    judge_system: str,
    judge_user_template: str,
    judge_params: SamplingParams,
) -> float:
    """Ask Llama to judge each VLM response. Returns fraction judged correct."""
    if not responses:
        return 0.0
    prompts = []
    for resp in responses:
        user_msg = judge_user_template.format(
            true_class=true_class, crop=crop, response=resp
        )
        prompts.append(_apply_chat_template(opt_tokenizer, user_msg, system_prompt=judge_system))
    outputs = opt_llm.generate(prompts, sampling_params=judge_params)
    correct = 0
    for o in outputs:
        text = _clean_response(o.outputs[0].text).lower().strip()
        if text.startswith("yes") or "correct" in text:
            correct += 1
    return round(correct / len(responses), 4)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(output_dir: Path) -> dict:
    p = output_dir / "checkpoint.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_checkpoint(output_dir: Path, checkpoint: dict) -> None:
    p = output_dir / "checkpoint.json"
    p.write_text(json.dumps(checkpoint, indent=2))


# ── Logging helpers ───────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── t-SNE overlay plot ────────────────────────────────────────────────────────

def plot_tsne_overlay(
    coords_npz: Path,
    results: dict,
    output_dir: Path,
) -> None:
    """
    Overlay base-prompt and optimised-prompt class centroids on the existing
    linear probe t-SNE layout.

    coords_npz must be a *_coords.npz file saved by probe.py's plot_tsne(),
    e.g. outputs/linear_probe/tsne/tsne_combined_gemma-4-26B-A4B-it_lr_coords.npz
    It contains X_2d (N×2) and labels (N,) — the exact same coordinates used
    to render the linear probe t-SNE PNG, so the overlay aligns perfectly.

    For each class in results["per_class"]:
      ■ square at class centroid — base prompt
      ★ star   at class centroid — best optimised prompt
    Both markers use the same color as the class dots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [tsne-overlay] matplotlib not available — skipping")
        return

    if not coords_npz.exists():
        print(f"  [tsne-overlay] {coords_npz} not found — skipping overlay")
        return

    try:
        d = np.load(coords_npz, allow_pickle=True)
        X_2d = d["X_2d"]
        y_labels = [str(l) for l in d["labels"]]
    except Exception as e:
        print(f"  [tsne-overlay] Could not load {coords_npz}: {e}")
        return

    print(f"  [tsne-overlay] Loaded {len(X_2d)} points from {coords_npz.name}")

    unique_labels = sorted(set(y_labels))
    cmap = matplotlib.colormaps[
        "tab20" if len(unique_labels) <= 20 else "hsv"
    ].resampled(len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background dots — same style as probe.py's plot_tsne
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(y_labels) if l == lbl]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color_map[lbl]], s=12, alpha=0.7)

    # Centroid markers per class
    per_class = results.get("per_class", {})
    for cls_data in per_class.values():
        clean_cls = cls_data.get("class_label", "")
        mask = [i for i, l in enumerate(y_labels) if l == clean_cls]
        if not mask:
            continue
        centroid = X_2d[mask].mean(axis=0)
        color = color_map.get(clean_cls, (0.5, 0.5, 0.5, 1.0))

        ax.scatter(
            centroid[0], centroid[1],
            marker="s", s=140, c=[color], edgecolors="black", linewidths=0.8, zorder=5,
        )
        ax.scatter(
            centroid[0], centroid[1],
            marker="*", s=260, c=[color], edgecolors="white", linewidths=0.5, zorder=6,
        )

    # Marker legend (just the two marker types, not per-class)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="base prompt centroid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
               markeredgecolor="white", markersize=12, label="optimised prompt centroid"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax.set_title(f"t-SNE Embedding Space — prompt optimisation overlay", fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "tsne_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── F1 comparison bar chart ───────────────────────────────────────────────────

def _draw_judge_line(ax, bar, value: float, label: bool = False) -> None:
    """Draw a thick horizontal line across a bar at the judge score (vision-assessment style)."""
    x0, w = bar.get_x(), bar.get_width()
    ax.plot([x0, x0 + w], [value, value], color="black", linewidth=2.5,
            solid_capstyle="butt", zorder=5,
            label="judge score" if label else "_nolegend_")
    ax.text(x0 + w / 2, value + 0.005, f"{value:.3f}",
            ha="center", va="bottom", fontsize=6, color="black", zorder=6)


def _bar_chart(
    labels: list[str],
    base_vals: list[float],
    opt_vals: list[float],
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
    has_base_judge = base_judge_vals is not None and len(base_judge_vals) == n
    has_judge      = judge_vals is not None and len(judge_vals) == n

    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    def _annotate(bars, color):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color=color)

    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base (F1)",
                       color="#5b8dd9", alpha=0.9)
    _annotate(bars_base, "#5b8dd9")

    bars_opt = ax.bar(x + width / 2, opt_vals, width, label="Optimised (F1)",
                      color="#e07b54", alpha=0.9)
    _annotate(bars_opt, "#c0522a")

    # Judge scores as horizontal lines across their respective bars
    judge_label_drawn = False
    if has_base_judge:
        for bar, jval in zip(bars_base, base_judge_vals):
            _draw_judge_line(ax, bar, jval, label=not judge_label_drawn)
            judge_label_drawn = True
    if has_judge:
        for bar, jval in zip(bars_opt, judge_vals):
            _draw_judge_line(ax, bar, jval, label=not judge_label_drawn)
            judge_label_drawn = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_ylabel("F1 / accuracy")
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


def _judge_mean(entries: list[dict]) -> float | None:
    """Mean judge_acc across entries that have it; None if none have it."""
    vals = [v["judge_acc"] for v in entries if "judge_acc" in v]
    return float(np.mean(vals)) if vals else None


def plot_f1_overall(results: dict, output_dir: Path) -> None:
    ov = results.get("overall", {})
    tr = results.get("train_overall", {})
    te = results.get("test_overall", {})
    if not ov:
        return

    base_judge_test = te.get("base_judge_acc_macro") or None
    judge_test      = te.get("judge_acc_macro") or None
    has_judge = base_judge_test is not None or judge_test is not None

    rows = [
        ("overall", ov.get("base_f1_macro", 0), ov.get("best_f1_macro", 0), None, None),
        ("train",   tr.get("base_f1_macro", 0), tr.get("best_f1_macro", 0), None, None),
        ("test",    te.get("base_f1_macro", 0), te.get("best_f1_macro", 0), base_judge_test, judge_test),
    ]
    labels          = [r[0] for r in rows]
    base_vals       = [r[1] for r in rows]
    opt_vals        = [r[2] for r in rows]
    base_judge_vals = [r[3] or 0.0 for r in rows]
    judge_vals      = [r[4] or 0.0 for r in rows]

    _bar_chart(
        labels, base_vals, opt_vals,
        title="Prompt Optimisation — Overall F1",
        out_path=output_dir / "f1_overall.png",
        base_judge_vals=base_judge_vals if has_judge else None,
        judge_vals=judge_vals if has_judge else None,
    )


def plot_f1_by_task(results: dict, output_dir: Path) -> None:
    # Use test entries only — train uses class_list (closed-form) so mixing distorts the comparison
    by_task: dict[str, list] = defaultdict(lambda: [[], [], [], []])
    for data in results.get("test_results_raw", {}).values():
        tt = data.get("task_type")
        if not tt:
            continue
        by_task[tt][0].append(data.get("base_f1", 0.0))
        by_task[tt][1].append(data.get("best_f1", 0.0))
        if "base_judge_acc" in data:
            by_task[tt][2].append(data["base_judge_acc"])
        if "judge_acc" in data:
            by_task[tt][3].append(data["judge_acc"])
    if not by_task:
        return
    labels = sorted(by_task)
    base_vals       = [float(np.mean(by_task[t][0])) for t in labels]
    opt_vals        = [float(np.mean(by_task[t][1])) for t in labels]
    base_judge_vals = [float(np.mean(by_task[t][2])) if by_task[t][2] else None for t in labels]
    judge_vals      = [float(np.mean(by_task[t][3])) if by_task[t][3] else None for t in labels]
    has_bj    = any(v is not None for v in base_judge_vals)
    has_judge = any(v is not None for v in judge_vals)
    _bar_chart(
        labels, base_vals, opt_vals,
        title="Prompt Optimisation — F1 by Task Type",
        out_path=output_dir / "f1_by_task.png",
        base_judge_vals=[v or 0.0 for v in base_judge_vals] if has_bj else None,
        judge_vals=[v or 0.0 for v in judge_vals] if has_judge else None,
    )


def plot_f1_by_dataset(results: dict, output_dir: Path) -> None:
    # Use test entries only — train uses class_list (closed-form) so mixing distorts the comparison
    by_ds: dict[str, list] = defaultdict(lambda: [[], [], [], []])
    for data in results.get("test_results_raw", {}).values():
        ds = data.get("dataset")
        if not ds:
            continue
        by_ds[ds][0].append(data.get("base_f1", 0.0))
        by_ds[ds][1].append(data.get("best_f1", 0.0))
        if "base_judge_acc" in data:
            by_ds[ds][2].append(data["base_judge_acc"])
        if "judge_acc" in data:
            by_ds[ds][3].append(data["judge_acc"])
    if not by_ds:
        return
    labels = sorted(by_ds)
    base_vals       = [float(np.mean(by_ds[d][0])) for d in labels]
    opt_vals        = [float(np.mean(by_ds[d][1])) for d in labels]
    base_judge_vals = [float(np.mean(by_ds[d][2])) if by_ds[d][2] else None for d in labels]
    judge_vals      = [float(np.mean(by_ds[d][3])) if by_ds[d][3] else None for d in labels]
    has_bj    = any(v is not None for v in base_judge_vals)
    has_judge = any(v is not None for v in judge_vals)
    _bar_chart(
        labels, base_vals, opt_vals,
        title="Prompt Optimisation — F1 by Dataset",
        out_path=output_dir / "f1_by_dataset.png",
        rotate=45,
        base_judge_vals=[v or 0.0 for v in base_judge_vals] if has_bj else None,
        judge_vals=[v or 0.0 for v in judge_vals] if has_judge else None,
    )


def plot_f1_comparison(results: dict, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    per_class = results.get("per_class", {})
    if not per_class:
        return

    labels = []
    base_f1s = []
    opt_f1s = []

    for cls_key, data in sorted(per_class.items()):
        labels.append(data.get("class_label", cls_key))
        base_f1s.append(data.get("base_f1", 0.0))
        opt_f1s.append(data.get("best_f1", 0.0))

    x = np.arange(len(labels))
    width = 0.38

    _bar_chart(
        labels, base_f1s, opt_f1s,
        title="Prompt Optimisation — F1 per Class",
        out_path=output_dir / "f1_per_class.png",
        xlabel="class",
        rotate=45,
    )


# ── Main optimisation loop ────────────────────────────────────────────────────

def run_train_loop(
    vlm: LLM,
    vlm_tokenizer,
    opt_llm: LLM,
    opt_tokenizer,
    cfg: dict,
    datasets_cfg: dict,
    split_datasets: list[str],
    n_samples_per_class: int,
    n_eval_per_candidate: int,
    output_dir: Path,
    checkpoint: dict,
    force: bool,
    seed: int,
) -> tuple[dict, str]:
    """
    Run optimisation loop over train split.
    Returns (best_prompts dict, final consolidated summary).
    """
    best_prompts: dict = checkpoint.get("best_prompts", {})
    per_class_summaries: list[dict] = checkpoint.get("per_class_summaries", [])
    consolidated_summary: str = checkpoint.get("consolidated_summary", "")

    meta_system = cfg["meta_system_prompt"]
    class_summary_system = cfg["class_summary_system"]
    class_summary_user_tmpl = cfg["class_summary_user"]
    consolidation_system = cfg["consolidation_system"]
    consolidation_user_tmpl = cfg["consolidation_user"]
    base_prompt_tmpl = cfg["base_prompt"]
    closed_form_tmpl = cfg["closed_form_prompt"]
    n_candidates = int(cfg.get("n_candidates", 5))
    n_iterations = int(cfg.get("n_iterations", 10))

    cls_params = SamplingParams(
        temperature=float(cfg.get("classifier_temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens_response", 64)),
        repetition_penalty=1.1,
    )
    opt_params = SamplingParams(
        temperature=float(cfg.get("optimizer_temperature", 0.8)),
        max_tokens=int(cfg.get("max_tokens_optimizer", 512)),
    )
    summary_params = SamplingParams(
        temperature=float(cfg.get("optimizer_temperature", 0.8)),
        max_tokens=int(cfg.get("max_tokens_summary", 384)),
    )

    run_log_path = output_dir / "run_log.jsonl"
    per_class_summaries_path = output_dir / "per_class_summaries.jsonl"
    consolidated_history_path = output_dir / "consolidated_summary_history.jsonl"

    for agml_name in split_datasets:
        ds_cfg = datasets_cfg.get(agml_name)
        if ds_cfg is None:
            print(f"  [warn] {agml_name} not in datasets config — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {agml_name}")

        try:
            sampled = load_agml_images(agml_name, n_samples_per_class, seed=seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        all_classes = clean_class_list(ds_cfg.classes)
        class_list_str = ", ".join(all_classes)

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            ckpt_entry = checkpoint.get("classes", {}).get(cls_key, {})
            if not force and ckpt_entry.get("status") == "done":
                print(f"  [skip] {clean_cls} (cached)")
                continue

            print(f"\n  Class: {clean_cls} ({len(image_paths)} images)")

            # Evaluate base prompt
            base_f1, base_resolvable, _ = evaluate_prompt(
                vlm, vlm_tokenizer, base_prompt_tmpl, image_paths,
                crop, clean_cls, all_classes, class_list_str, cls_params,
                n_eval=n_eval_per_candidate,
            )
            print(f"    base F1={base_f1:.3f}  resolvable={base_resolvable:.2f}")

            best_f1 = base_f1
            best_prompt = closed_form_tmpl
            last_prompt = closed_form_tmpl

            for iteration in range(n_iterations):
                candidates, llama_raw = generate_candidates(
                    opt_llm, opt_tokenizer,
                    meta_system, crop, clean_cls,
                    best_prompt if best_prompt != closed_form_tmpl else None,
                    last_prompt,
                    consolidated_summary,
                    n_candidates,
                    opt_params,
                )

                if not candidates:
                    print(f"    iter {iteration+1}: no candidates generated")
                    continue

                iter_best_f1 = best_f1
                iter_best_prompt = best_prompt

                for ci, candidate in enumerate(candidates):
                    if "{class_list}" not in candidate:
                        candidate = candidate.rstrip() + " Choose from: {class_list}."

                    try:
                        f1, resolvable, responses = evaluate_prompt(
                            vlm, vlm_tokenizer, candidate, image_paths,
                            crop, clean_cls, all_classes, class_list_str, cls_params,
                            n_eval=n_eval_per_candidate,
                        )
                    except Exception as e:
                        print(f"    [warn] Candidate {ci} eval error: {e}")
                        continue

                    log_record = {
                        "dataset": agml_name,
                        "class_label": clean_cls,
                        "crop": crop,
                        "iteration": iteration,
                        "candidate_idx": ci,
                        "prompt": candidate,
                        "llama_raw_output": llama_raw,
                        "vlm_responses": responses,
                        "f1": f1,
                        "resolvable": resolvable,
                        "base_f1": base_f1,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    append_jsonl(run_log_path, log_record)

                    print(f"    iter {iteration+1} c{ci}: F1={f1:.3f}  resolvable={resolvable:.2f}")

                    if f1 > iter_best_f1:
                        iter_best_f1 = f1
                        iter_best_prompt = candidate

                    last_prompt = candidate

                if iter_best_f1 > best_f1:
                    best_f1 = iter_best_f1
                    best_prompt = iter_best_prompt
                    print(f"    New best: F1={best_f1:.3f}")

            # Store best prompt for this class
            best_prompts[cls_key] = {
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "best_prompt": best_prompt,
                "best_f1": best_f1,
                "base_f1": base_f1,
                "gain": round(best_f1 - base_f1, 4),
            }

            # Collect all candidates tested for this class from run_log
            class_candidates = [
                e for e in _read_jsonl(run_log_path)
                if e.get("dataset") == agml_name and e.get("class_label") == clean_cls
            ]

            # 1. Per-class summary
            try:
                cls_summary = generate_class_summary(
                    opt_llm, opt_tokenizer,
                    class_summary_system, class_summary_user_tmpl,
                    crop, clean_cls, agml_name,
                    base_f1, best_f1, base_prompt_tmpl, best_prompt,
                    class_candidates, summary_params,
                )
            except Exception as e:
                print(f"  [warn] Per-class summary failed: {e}")
                cls_summary = f"Optimised {clean_cls} on {agml_name}. Base F1={base_f1:.3f} → best F1={best_f1:.3f}."

            per_class_summaries.append({
                "cls_key": cls_key,
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "summary": cls_summary,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            append_jsonl(per_class_summaries_path, per_class_summaries[-1])
            print(f"    Per-class summary written.")

            # Checkpoint per-class progress (no consolidation yet)
            if "classes" not in checkpoint:
                checkpoint["classes"] = {}
            checkpoint["classes"][cls_key] = {"status": "done", "best_f1": best_f1}
            checkpoint["best_prompts"] = best_prompts
            checkpoint["per_class_summaries"] = per_class_summaries
            save_checkpoint(output_dir, checkpoint)

    # Consolidate once after all classes are done
    print(f"\nConsolidating {len(per_class_summaries)} per-class summaries...")
    try:
        consolidated_summary = consolidate_summaries(
            opt_llm, opt_tokenizer,
            consolidation_system, consolidation_user_tmpl,
            per_class_summaries, "",
            summary_params,
        )
    except Exception as e:
        print(f"  [warn] Consolidation failed: {e}")
        consolidated_summary = "\n\n".join(
            f"{s['crop']} / {s['class_label']}: {s['summary']}"
            for s in per_class_summaries
        )

    append_jsonl(consolidated_history_path, {
        "n_classes_seen": len(per_class_summaries),
        "consolidated_summary": consolidated_summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    checkpoint["consolidated_summary"] = consolidated_summary
    save_checkpoint(output_dir, checkpoint)

    return best_prompts, consolidated_summary


def run_test_evaluation(
    vlm: LLM,
    vlm_tokenizer,
    opt_llm: LLM,
    opt_tokenizer,
    cfg: dict,
    datasets_cfg: dict,
    split_datasets: list[str],
    consolidated_summary: str,
    n_eval_per_candidate: int,
    output_dir: Path,
    seed: int,
    checkpoint: dict | None = None,
    force: bool = False,
) -> dict:
    """
    Evaluate test split:
      - Base: open_ended_prompt only (no context)
      - Opt:  consolidated_summary as VLM system prompt + open_ended_prompt as question
              (hardcoded — no Llama generation)
    Returns per_class results dict.
    """
    open_ended_tmpl = cfg["open_ended_prompt"]
    judge_system = cfg.get("judge_system", "")
    judge_user_tmpl = cfg.get("judge_user", "")

    cls_params = SamplingParams(
        temperature=float(cfg.get("classifier_temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens_response", 64)),
        repetition_penalty=1.1,
    )
    judge_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(cfg.get("max_tokens_judge", 8)),
    )

    # Write summary to disk so it can be inspected / edited before re-runs
    summary_path = output_dir / "final_summary.txt"
    summary_path.write_text(consolidated_summary or "")
    print(f"  Final summary written to: {summary_path}")

    n_samples = max(n_eval_per_candidate * 2, 20)
    # Load any previously completed test results so re-runs don't overwrite them
    test_results: dict = (checkpoint or {}).get("test_results", {})
    test_log_path = output_dir / "test_log.jsonl"

    for agml_name in split_datasets:
        ds_cfg = datasets_cfg.get(agml_name)
        if ds_cfg is None:
            print(f"  [warn] {agml_name} not in datasets config — skipping")
            continue

        print(f"\n  Test: {agml_name}")

        try:
            sampled = load_agml_images(agml_name, n_samples, seed=seed)
        except Exception as e:
            print(f"  [error] {agml_name}: {e}")
            continue

        all_classes = clean_class_list(ds_cfg.classes)
        class_list_str = ", ".join(all_classes)

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            if cls_key in test_results and not force:
                print(f"    {clean_cls}: already done — skipping")
                continue

            # Base prompt (open-ended, no class list — matches production pipeline)
            base_f1, base_resolvable, base_responses = evaluate_prompt(
                vlm, vlm_tokenizer, open_ended_tmpl, image_paths,
                crop, clean_cls, all_classes, class_list_str, cls_params,
            )

            # Opt: consolidated summary as VLM system prompt + same base question
            # Structure is hardcoded: {summary} / {base_prompt} — no Llama involved
            opt_f1, opt_resolvable, opt_responses = evaluate_prompt(
                vlm, vlm_tokenizer, open_ended_tmpl, image_paths,
                crop, clean_cls, all_classes, class_list_str, cls_params,
                system_prompt=consolidated_summary,
            )
            opt_prompt = f"[system: consolidated_summary]\n{open_ended_tmpl}"

            # LLM-as-judge on both base and optimised responses
            base_judge_acc = 0.0
            judge_acc = 0.0
            if judge_system and judge_user_tmpl:
                try:
                    if base_responses:
                        base_judge_acc = judge_responses(
                            opt_llm, opt_tokenizer,
                            base_responses, clean_cls, crop,
                            judge_system, judge_user_tmpl, judge_params,
                        )
                    if opt_responses:
                        judge_acc = judge_responses(
                            opt_llm, opt_tokenizer,
                            opt_responses, clean_cls, crop,
                            judge_system, judge_user_tmpl, judge_params,
                        )
                except Exception as e:
                    print(f"  [warn] Judge failed for {clean_cls}: {e}")

            test_results[cls_key] = {
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "base_f1": base_f1,
                "base_resolvable": base_resolvable,
                "base_judge_acc": base_judge_acc,
                "best_f1": opt_f1,
                "best_resolvable": opt_resolvable,
                "judge_acc": judge_acc,
                "gain": round(opt_f1 - base_f1, 4),
                "judge_gain": round(judge_acc - base_judge_acc, 4),
                "best_prompt": opt_prompt,
            }

            append_jsonl(test_log_path, {
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "base_prompt": open_ended_tmpl,
                "base_vlm_responses": base_responses,
                "base_f1": base_f1,
                "base_resolvable": base_resolvable,
                "base_judge_acc": base_judge_acc,
                "opt_prompt": opt_prompt,
                "opt_vlm_responses": opt_responses,
                "opt_f1": opt_f1,
                "opt_resolvable": opt_resolvable,
                "judge_acc": judge_acc,
                "gain": round(opt_f1 - base_f1, 4),
                "judge_gain": round(judge_acc - base_judge_acc, 4),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            checkpoint["test_results"] = test_results
            save_checkpoint(output_dir, checkpoint)
            print(
                f"    {clean_cls}: base={base_f1:.3f}(j={base_judge_acc:.3f})  "
                f"opt={opt_f1:.3f}(j={judge_acc:.3f})  "
                f"gain={opt_f1 - base_f1:+.3f}"
            )

    return test_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt optimisation pipeline")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B",
                        help="VLM for classification evaluation")
    parser.add_argument("--config", default="src/config/datasets.yaml",
                        help="AgML datasets config (YAML)")
    parser.add_argument("--optim_config", default="src/config/prompt_optim.yaml",
                        help="Prompt optimisation config (YAML)")
    parser.add_argument("--n_samples_per_class", type=int, default=30,
                        help="Images sampled per class for optimisation")
    parser.add_argument("--n_eval_per_candidate", type=int, default=15,
                        help="Images used to score each candidate prompt")
    parser.add_argument("--output_dir", default="outputs/prompt_optim/",
                        help="Output directory")
    parser.add_argument("--max_model_len", type=int, default=6144,
                        help="Max sequence length for the VLM (lower = less KV cache memory)")
    parser.add_argument("--probe_dir", default="outputs/linear_probe/",
                        help="Linear probe output directory. Used to locate the t-SNE coords "
                             "file saved by probe.py (auto-derived from --model name).")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if checkpoint shows done")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.optim_config))
    datasets_cfg = datasets_by_name(Path(args.config))

    split_config_path = Path(cfg.get("split_config", "src/config/dataset_split.yaml"))
    split = load_split(split_config_path)
    train_split: list[str] = split.get("train", [])
    test_split: list[str] = split.get("test", [])

    optimizer_model = cfg.get("optimizer_model", "meta-llama/Llama-3.2-1B-Instruct")

    checkpoint = load_checkpoint(output_dir)

    print(f"VLM: {args.model}")
    print(f"Optimizer: {optimizer_model}")
    print(f"Output: {output_dir}")

    # Load both models once
    print("\nLoading VLM...")
    vlm, vlm_tokenizer = load_vlm(args.model, gpu_memory_utilization=0.65, max_model_len=args.max_model_len)

    print("Loading optimizer (Llama)...")
    opt_llm, opt_tokenizer = load_optimizer(optimizer_model)

    best_prompts: dict = checkpoint.get("best_prompts", {})
    consolidated_summary: str = checkpoint.get("consolidated_summary", "")

    # Train loop
    if args.split in ("train", "both"):
        print(f"\n{'#'*60}")
        print(f"TRAIN SPLIT ({len(train_split)} datasets)")
        best_prompts, consolidated_summary = run_train_loop(
            vlm, vlm_tokenizer, opt_llm, opt_tokenizer,
            cfg, datasets_cfg, train_split,
            n_samples_per_class=args.n_samples_per_class,
            n_eval_per_candidate=args.n_eval_per_candidate,
            output_dir=output_dir,
            checkpoint=checkpoint,
            force=args.force,
            seed=args.seed,
        )
        (output_dir / "best_prompts.json").write_text(
            json.dumps(best_prompts, indent=2)
        )

    # Test evaluation
    test_results: dict = {}
    if args.split in ("test", "both"):
        print(f"\n{'#'*60}")
        print(f"TEST SPLIT ({len(test_split)} datasets)")
        test_results = run_test_evaluation(
            vlm, vlm_tokenizer, opt_llm, opt_tokenizer,
            cfg, datasets_cfg, test_split,
            consolidated_summary=consolidated_summary,
            n_eval_per_candidate=args.n_eval_per_candidate,
            output_dir=output_dir,
            seed=args.seed,
            checkpoint=checkpoint,
            force=args.force,
        )

    destroy_llm(vlm)

    # Aggregate results
    all_per_class: dict = {}
    all_per_class.update(best_prompts)  # train results
    all_per_class.update(test_results)  # test results (overrides if overlapping)

    def _safe_mean(vals: list[float]) -> float:
        return round(float(np.mean(vals)), 4) if vals else 0.0

    base_f1s = [v["base_f1"] for v in all_per_class.values() if "base_f1" in v]
    best_f1s = [v["best_f1"] for v in all_per_class.values() if "best_f1" in v]
    train_entries = {k: v for k, v in best_prompts.items() if "best_f1" in v}
    test_entries = {k: v for k, v in test_results.items() if "best_f1" in v}

    # Annotate every entry with task_type for grouping in plots
    for entry in list(all_per_class.values()) + list(test_results.values()):
        ds_cfg = datasets_cfg.get(entry.get("dataset", ""))
        if ds_cfg:
            entry["task_type"] = ds_cfg.task_type

    results = {
        "model": args.model,
        "optimizer_model": optimizer_model,
        "overall": {
            "base_f1_macro": _safe_mean(base_f1s),
            "best_f1_macro": _safe_mean(best_f1s),
            "gain": round(_safe_mean(best_f1s) - _safe_mean(base_f1s), 4),
            "n_classes": len(all_per_class),
        },
        "train_overall": {
            "base_f1_macro": _safe_mean([v["base_f1"] for v in train_entries.values()]),
            "best_f1_macro": _safe_mean([v["best_f1"] for v in train_entries.values()]),
            "n_classes": len(train_entries),
        },
        "test_overall": {
            "base_f1_macro": _safe_mean([v["base_f1"] for v in test_entries.values()]),
            "best_f1_macro": _safe_mean([v["best_f1"] for v in test_entries.values()]),
            "base_judge_acc_macro": _safe_mean(
                [v["base_judge_acc"] for v in test_entries.values() if "base_judge_acc" in v]
            ),
            "judge_acc_macro": _safe_mean(
                [v["judge_acc"] for v in test_entries.values() if "judge_acc" in v]
            ),
            "n_classes": len(test_entries),
        },
        "per_class": all_per_class,
        "test_results_raw": test_results,
        "final_summary": consolidated_summary,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_path}")

    ov = results["overall"]
    print(
        f"\nOverall  base={ov['base_f1_macro']:.3f}  "
        f"best={ov['best_f1_macro']:.3f}  "
        f"gain={ov['gain']:+.3f}  "
        f"n_classes={ov['n_classes']}"
    )

    if not args.skip_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_f1_overall(results, plots_dir)
        plot_f1_by_task(results, plots_dir)
        plot_f1_by_dataset(results, plots_dir)
        plot_f1_comparison(results, plots_dir)

        safe_model = args.model.replace("/", "_").replace(":", "_").replace(" ", "_")
        coords_npz = Path(args.probe_dir) / "tsne" / f"tsne_combined_{safe_model}_lr_coords.npz"
        plot_tsne_overlay(coords_npz, results, plots_dir)


if __name__ == "__main__":
    main()
