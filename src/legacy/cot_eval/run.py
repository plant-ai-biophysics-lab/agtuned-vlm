"""
CoT Eval Experiment
===================
4-way ablation of chain-of-thought prompting strategies for agricultural
image classification.

Conditions (per image):
  base         — plain classification question, no thinking
  cot          — structured 5-point observation pass → classify with context
  thinking     — Gemma-4 native enable_thinking=True on base question
  cot_thinking — structured observation pass → classify with context + native thinking

This isolates: (1) does domain-specific prior help? (2) does native thinking help?
(3) do they compound?

Pipeline:
  Pass 1: base            [image] + base_prompt                        (thinking=False)
  Pass 2: observations    [image] + thinking_prompt                    (thinking=False)
  Pass 3: cot classify    [image + thinking_prompt] + obs + cls_q      (thinking=False)
  Pass 4: native thinking [image] + base_prompt                        (thinking=True)
  Pass 5: cot+thinking    [image + thinking_prompt] + obs + cls_q      (thinking=True)

Thinking prompts are task-specific (disease_pest / weed_identification) from
src/config/cot_prompts.yaml. Base classification prompts from prompts.yaml.

Outputs (under output_dir/):
  results.json         — aggregate metrics for all 4 conditions
  class_log.jsonl      — per-class records with all responses + CoT observations
  checkpoint.json      — incremental checkpoint
  plots/
    f1_overall.png
    f1_by_task.png
    f1_by_dataset.png
    f1_per_class.png
    cot_length_vs_gain.png   scatter: CoT observation length vs F1 gain (cot vs base)
    gain_distribution.png    histogram of gains for all non-base conditions
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
from src.common.metrics import resolve_class, apply_judge

# Matches both <think>...</think> and <thought>...</thought> (Gemma-4 uses <thought>)
_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)

CONDITIONS = ["base", "cot", "thinking", "cot_thinking"]


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


# ── vLLM helpers ──────────────────────────────────────────────────────────────

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


def _apply_template(tokenizer, messages: list[dict], enable_thinking: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _build_single_turn(
    tokenizer,
    image: PILImage.Image,
    question: str,
    enable_thinking: bool = False,
) -> dict:
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    return {
        "prompt": _apply_template(tokenizer, msgs, enable_thinking=enable_thinking),
        "multi_modal_data": {"image": image},
    }


def _build_cot_classify_turn(
    tokenizer,
    image: PILImage.Image,
    thinking_prompt: str,
    cot_response: str,
    classification_question: str,
    enable_thinking: bool = False,
) -> dict:
    """
    Two-turn: [image + thinking_prompt] → [observations] → [classification_question].
    Image only in the first user turn; enable_thinking controls the final answer pass.
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
    return {
        "prompt": _apply_template(tokenizer, msgs, enable_thinking=enable_thinking),
        "multi_modal_data": {"image": image},
    }


def _clean(text: str) -> str:
    """
    Strip complete <think>/<thought> blocks.
    Discard from any remaining unclosed opening tag onward (truncated output).
    """
    text = _THINK_RE.sub("", text)
    for tag in ("<think>", "<thought>"):
        idx = text.lower().find(tag)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _extract_thinking_answer(text: str) -> str:
    """
    Gemma-4 with enable_thinking=True renders thinking as:
      "thought\\n<reasoning>\\n...Final decision: X.X"
    The actual answer is always the last segment after the final period on the last line.
    Falls back to the full cleaned text if the pattern isn't found.
    """
    text = _clean(text)
    if not text.lower().startswith("thought"):
        return text
    # Take the last non-empty line of the reasoning block
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # Model repeats the answer: "Anthracnose.Anthracnose" or "healthy.healthy"
    # Take everything after the last period
    if "." in last:
        candidate = last.rsplit(".", 1)[1].strip()
        if candidate:
            return candidate
    return last


# ── Multi-turn CoT runner ─────────────────────────────────────────────────────

def _run_cot_multiturn(
    llm,
    tokenizer,
    images: list[PILImage.Image],
    steps: list[str],
    classification_q: str,
    step_params,
    cls_params,
    enable_thinking: bool = False,
) -> tuple[list[str], list[list[str]]]:
    """
    Run true multi-turn CoT: one vLLM batch per step, growing conversation per image.

    Turn structure for each image:
      User:  [image] <step 1 question>
      Asst:  <answer 1>
      User:  <step 2 question>          ← no image token; already in history
      Asst:  <answer 2>
      ...
      User:  <classification_q>
      Asst:  <final class name>

    For enable_thinking=True (CoT+thinking), the model reasons before each answer.
    Only the extracted final answer (not the full <thought> blob) is stored in
    history — keeps context length manageable across 5 steps.

    Returns:
      final_responses        — extracted class-name answer for each image
      step_responses_grid    — step_responses_grid[step_i][image_i] = answer text
    """
    n = len(images)
    # histories[i] = growing list of message dicts for image i
    histories: list[list[dict]] = [[] for _ in range(n)]
    step_responses_grid: list[list[str]] = []

    for step_idx, question in enumerate(steps):
        inputs = []
        for i, img in enumerate(images):
            if step_idx == 0:
                msgs = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]}]
            else:
                msgs = histories[i] + [{"role": "user", "content": [
                    {"type": "text", "text": question},
                ]}]
            inputs.append({
                "prompt": _apply_template(tokenizer, msgs, enable_thinking=enable_thinking),
                "multi_modal_data": {"image": img},
            })

        raw_outputs = llm.generate(inputs, step_params)
        step_answers = []
        for i, out in enumerate(raw_outputs):
            raw = out.outputs[0].text
            answer = _extract_thinking_answer(_clean(raw)) if enable_thinking else _clean(raw)
            step_answers.append(answer)

            if step_idx == 0:
                histories[i] = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": answer},
                ]
            else:
                histories[i] = histories[i] + [
                    {"role": "user", "content": [{"type": "text", "text": question}]},
                    {"role": "assistant", "content": answer},
                ]

        step_responses_grid.append(step_answers)

    # Final classification turn
    cls_inputs = []
    for i, img in enumerate(images):
        msgs = histories[i] + [{"role": "user", "content": [
            {"type": "text", "text": classification_q},
        ]}]
        cls_inputs.append({
            "prompt": _apply_template(tokenizer, msgs, enable_thinking=enable_thinking),
            "multi_modal_data": {"image": img},
        })

    cls_outputs = llm.generate(cls_inputs, cls_params)
    if enable_thinking:
        final_responses = [_extract_thinking_answer(_clean(o.outputs[0].text)) for o in cls_outputs]
    else:
        final_responses = [_clean(o.outputs[0].text) for o in cls_outputs]

    return final_responses, step_responses_grid


# ── Prompt builders ───────────────────────────────────────────────────────────

def get_thinking_steps(task_type: str, cfg: dict) -> list[str]:
    cot_key = cfg["task_type_to_cot"].get(task_type, "disease_pest")
    return cfg["thinking_prompts"][cot_key]["steps"]


def get_base_prompt(task_type: str, prompts_cfg: dict, cot_cfg: dict, crop: str | None) -> str:
    prompt_key = cot_cfg["task_type_to_base_prompt"].get(task_type, "visual_classification_disease")
    template = prompts_cfg[prompt_key]
    if "{crop}" in template and crop:
        return template.format(crop=crop)
    return template


# ── Per-class 4-way evaluation ────────────────────────────────────────────────

def evaluate_class(
    llm,
    tokenizer,
    image_paths: list[str],
    steps: list[str],
    base_classification_q: str,
    cot_classification_q: str,
    base_params,
    step_params,
    cls_params,
    thinking_cls_params,
    thinking_step_params,
    n_eval: int,
) -> tuple[list[str], list[str], list[str], list[str], list[list[str]], list[list[str]], list[str], list[str]]:
    """
    4 conditions, true multi-turn CoT:

      base         — [image] + base_classification_q                              (thinking=False)
      cot          — multi-turn: step 1→answer 1→step 2→answer 2→...→classify    (thinking=False)
      thinking     — [image] + base_classification_q                              (thinking=True)
      cot_thinking — same multi-turn, thinking=True on every step + final         (thinking=True)

    CoT uses _run_cot_multiturn: each bullet point is its own vLLM batch with the
    growing conversation as context. For cot_thinking, the model reasons before each
    step answer, but only the extracted final answer enters the history.

    Returns:
      base_responses, cot_responses, thinking_responses, cot_thinking_responses,
      cot_step_grid, cot_thinking_step_grid,   ← per-step answers [step][image]
      thinking_full, cot_thinking_full          ← raw reasoning blobs for logging
    """
    paths = image_paths[:n_eval]
    images: list[PILImage.Image] = []
    for p in paths:
        try:
            images.append(PILImage.open(p).convert("RGB"))
        except Exception as e:
            print(f"      [warn] {p}: {e}")

    if not images:
        return [], [], [], [], [], [], [], []

    # Pass 1 — base
    base_inputs = [_build_single_turn(tokenizer, img, base_classification_q, enable_thinking=False)
                   for img in images]
    base_responses = [_clean(o.outputs[0].text)
                      for o in llm.generate(base_inputs, base_params)]

    # Passes 2+3 — multi-turn CoT (thinking=False, one step per turn)
    print(f"      [cot] {len(steps)} steps × {len(images)} images ...")
    cot_responses, cot_step_grid = _run_cot_multiturn(
        llm, tokenizer, images, steps, cot_classification_q,
        step_params, cls_params, enable_thinking=False,
    )

    # Pass 4 — native thinking on base question (budget = thinking + answer headroom)
    think_inputs   = [_build_single_turn(tokenizer, img, base_classification_q, enable_thinking=True)
                      for img in images]
    thinking_raw   = llm.generate(think_inputs, thinking_cls_params)
    thinking_full  = [_clean(o.outputs[0].text) for o in thinking_raw]
    thinking_responses = [_extract_thinking_answer(t) for t in thinking_full]

    # Passes 5+6 — multi-turn CoT + native thinking (thinking=True on every turn)
    # step budget = thinking + step headroom; final cls budget = thinking + response headroom
    print(f"      [cot+thinking] {len(steps)} steps × {len(images)} images ...")
    cot_thinking_responses, cot_thinking_step_grid = _run_cot_multiturn(
        llm, tokenizer, images, steps, cot_classification_q,
        thinking_step_params, thinking_cls_params, enable_thinking=True,
    )
    # Flatten reasoning blobs from the last step for logging (approximation)
    cot_thinking_full = [
        " | ".join(cot_thinking_step_grid[s][i] for s in range(len(steps)))
        for i in range(len(images))
    ]

    return (base_responses, cot_responses,
            thinking_responses, cot_thinking_responses,
            cot_step_grid, cot_thinking_step_grid,
            thinking_full, cot_thinking_full)


def _class_f1(responses: list[str], class_label: str, all_classes: list[str]) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    preds = [resolve_class(r, all_classes) for r in responses]
    resolved = [p or "" for p in preds]
    resolvability = sum(1 for p in preds if p) / max(len(preds), 1)
    refs = [class_label] * len(resolved)
    f1 = f1_score(refs, resolved, labels=[class_label], average="macro", zero_division=0)
    return round(float(f1), 4), round(resolvability, 4)


def _run_judge(
    rows: list[dict],
    llm,
    tokenizer,
    judge_system: str,
    judge_template: str,
) -> tuple[float, list[dict]]:
    """Run judge; return (accuracy, judged_rows) so callers can log details."""
    judged = apply_judge(rows, llm, tokenizer, judge_system, judge_template, "gt", "response")
    acc = round(sum(r["judged_correct"] for r in judged) / max(len(judged), 1), 4)
    return acc, judged


def _log_judge(path: Path, cls_key: str, condition: str, judged_rows: list[dict]) -> None:
    """Append one record per prediction to judge_log.jsonl for debugging."""
    with open(path, "a") as f:
        for i, r in enumerate(judged_rows):
            f.write(json.dumps({
                "cls_key":        cls_key,
                "condition":      condition,
                "sample_idx":     i,
                "gt":             r.get("gt", ""),
                "response":       r.get("response", ""),
                "judged_correct": r.get("judged_correct", 0),
                "judge_reasoning": r.get("judge_reasoning", ""),
            }) + "\n")


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

    n_eval        = int(cfg.get("n_eval", 30))
    temp          = float(cfg.get("classifier_temperature", 0.0))
    max_step      = int(cfg.get("max_tokens_step", 256))
    max_resp      = int(cfg.get("max_tokens_response", 128))
    max_thinking  = int(cfg.get("max_tokens_thinking", 4096))
    gpu_util      = float(cfg.get("gpu_memory_utilization", 0.85))
    max_model_len = int(cfg.get("max_model_len", 12288))

    judge_system   = prompts_cfg["judge_system_vision"]
    judge_template = prompts_cfg["judge_classification_template"]
    base_suffix = cfg.get("base_classification_suffix",
                          " Answer with the specific disease, pest, or stress name only (if any - healthy otherwise). Do not describe — just name it.")
    cot_suffix  = cfg.get("cot_classification_suffix",
                          " Based on your analysis above, what is the specific name of the condition shown (if any - healthy otherwise)? Answer with the specific name only.")

    base_params          = SamplingParams(temperature=temp, max_tokens=max_resp,                  repetition_penalty=1.1)
    step_params          = SamplingParams(temperature=temp, max_tokens=max_step,                  repetition_penalty=1.05)
    cls_params           = SamplingParams(temperature=temp, max_tokens=max_resp,                  repetition_penalty=1.1)
    # thinking=True: budget = thinking block + answer headroom
    thinking_cls_params  = SamplingParams(temperature=temp, max_tokens=max_thinking + max_resp,   repetition_penalty=1.05)
    thinking_step_params = SamplingParams(temperature=temp, max_tokens=max_thinking + max_step,   repetition_penalty=1.05)

    eval_results: dict = checkpoint.get("eval_results", {})
    log_path       = output_dir / "class_log.jsonl"
    judge_log_path = output_dir / "judge_log.jsonl"

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
        steps = get_thinking_steps(ds_cfg.task_type, cfg)

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            if not force and cls_key in eval_results:
                print(f"  [skip] {clean_cls}")
                continue

            print(f"\n  Class: {clean_cls}")

            base_prompt             = get_base_prompt(ds_cfg.task_type, prompts_cfg, cfg, crop)
            # base/thinking: direct name instruction
            base_classification_q   = base_prompt + base_suffix
            # cot passes: bridge from structured analysis to specific name
            cot_classification_q    = base_prompt + cot_suffix

            (base_responses, cot_responses,
             thinking_responses, cot_thinking_responses,
             cot_step_grid, cot_thinking_step_grid,
             thinking_full, cot_thinking_full) = evaluate_class(
                llm, tokenizer, image_paths,
                steps,
                base_classification_q, cot_classification_q,
                base_params, step_params, cls_params, thinking_cls_params, thinking_step_params, n_eval,
            )

            if not base_responses:
                print(f"    [warn] no images loaded — skipping")
                continue

            base_f1,         base_res  = _class_f1(base_responses,         clean_cls, all_classes)
            cot_f1,          cot_res   = _class_f1(cot_responses,          clean_cls, all_classes)
            thinking_f1,     think_res = _class_f1(thinking_responses,     clean_cls, all_classes)
            cot_thinking_f1, ct_res    = _class_f1(cot_thinking_responses, clean_cls, all_classes)

            print(f"    base            F1={base_f1:.3f}  res={base_res:.2f}")
            print(f"    cot             F1={cot_f1:.3f}  res={cot_res:.2f}  gain={cot_f1-base_f1:+.3f}")
            print(f"    thinking        F1={thinking_f1:.3f}  res={think_res:.2f}  gain={thinking_f1-base_f1:+.3f}")
            print(f"    cot+thinking    F1={cot_thinking_f1:.3f}  res={ct_res:.2f}  gain={cot_thinking_f1-base_f1:+.3f}")

            # Judge all 4 conditions with the same vLLM instance
            def _rows(responses):
                return [{"gt": clean_cls, "response": r} for r in responses]

            base_judge,         base_judged         = _run_judge(_rows(base_responses),         llm, tokenizer, judge_system, judge_template)
            cot_judge,          cot_judged          = _run_judge(_rows(cot_responses),          llm, tokenizer, judge_system, judge_template)
            thinking_judge,     thinking_judged     = _run_judge(_rows(thinking_responses),     llm, tokenizer, judge_system, judge_template)
            cot_thinking_judge, cot_thinking_judged = _run_judge(_rows(cot_thinking_responses), llm, tokenizer, judge_system, judge_template)

            print(f"    judge  base={base_judge:.3f}  cot={cot_judge:.3f}  "
                  f"thinking={thinking_judge:.3f}  cot+thinking={cot_thinking_judge:.3f}")

            # Write per-prediction judge details to judge_log.jsonl
            _log_judge(judge_log_path, cls_key, "base",         base_judged)
            _log_judge(judge_log_path, cls_key, "cot",          cot_judged)
            _log_judge(judge_log_path, cls_key, "thinking",     thinking_judged)
            _log_judge(judge_log_path, cls_key, "cot_thinking", cot_thinking_judged)

            # Mean words per image across all CoT steps
            all_step_words = [
                len(cot_step_grid[s][i].split())
                for s in range(len(steps))
                for i in range(len(cot_step_grid[0]) if cot_step_grid else 0)
            ]
            mean_cot_len = round(float(np.mean(all_step_words)), 1) if all_step_words else 0.0

            entry = {
                "dataset":    agml_name,
                "class_label": clean_cls,
                "crop":       crop,
                "task_type":  ds_cfg.task_type,
                "base_classification_q": base_classification_q,
                "cot_classification_q":  cot_classification_q,
                # F1
                "base_f1":          base_f1,
                "cot_f1":           cot_f1,
                "thinking_f1":      thinking_f1,
                "cot_thinking_f1":  cot_thinking_f1,
                # gains vs base
                "cot_gain":         round(cot_f1         - base_f1, 4),
                "thinking_gain":    round(thinking_f1    - base_f1, 4),
                "cot_thinking_gain": round(cot_thinking_f1 - base_f1, 4),
                # resolvability
                "base_resolvable":         base_res,
                "cot_resolvable":          cot_res,
                "thinking_resolvable":     think_res,
                "cot_thinking_resolvable": ct_res,
                # judge
                "base_judge_acc":         base_judge,
                "cot_judge_acc":          cot_judge,
                "thinking_judge_acc":     thinking_judge,
                "cot_thinking_judge_acc": cot_thinking_judge,
                # metadata
                "mean_cot_length_words":  mean_cot_len,
                "base_prompt":            base_prompt,
                "cot_prompt_key":         cfg["task_type_to_cot"].get(ds_cfg.task_type, "disease_pest"),
                "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            eval_results[cls_key] = entry

            append_jsonl(log_path, {
                **entry,
                "base_responses":            base_responses,
                "cot_responses":             cot_responses,
                "cot_step_grid":             cot_step_grid,             # [step][image] answers
                "thinking_responses":        thinking_responses,        # extracted final answer
                "thinking_full":             thinking_full,             # full reasoning blob
                "cot_thinking_responses":    cot_thinking_responses,    # extracted final answer
                "cot_thinking_step_grid":    cot_thinking_step_grid,   # [step][image] answers
                "cot_thinking_full":         cot_thinking_full,
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

    def _macro(key):
        vals = [e[key] for e in all_e if key in e]
        return round(float(np.mean(vals)), 4) if vals else None

    def _n_improved(f1_key):
        return sum(1 for e in all_e if e.get(f1_key, 0) > e.get("base_f1", 0))

    by_task: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    by_ds:   dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    numeric_keys = [
        "base_f1", "cot_f1", "thinking_f1", "cot_thinking_f1",
        "base_judge_acc", "cot_judge_acc", "thinking_judge_acc", "cot_thinking_judge_acc",
    ]

    for e in all_e:
        tt = e.get("task_type")
        ds = e.get("dataset")
        for k in numeric_keys:
            if k in e:
                if tt: by_task[tt][k].append(e[k])
                if ds: by_ds[ds][k].append(e[k])

    def _mean_group(d):
        out = {k: round(float(np.mean(v)), 4) for k, v in d.items() if v}
        out["n_classes"] = len(d.get("base_f1", []))
        for f1_key in ("cot_f1", "thinking_f1", "cot_thinking_f1"):
            gain_key = f1_key.replace("_f1", "_gain")
            if "base_f1" in out and f1_key in out:
                out[gain_key] = round(out[f1_key] - out["base_f1"], 4)
        return out

    n = len(all_e)
    overall = {
        "n_classes": n,
    }
    for k in numeric_keys:
        v = _macro(k)
        if v is not None:
            overall[k] = v
    for f1_key in ("cot_f1", "thinking_f1", "cot_thinking_f1"):
        if "base_f1" in overall and f1_key in overall:
            overall[f1_key.replace("_f1", "_gain")] = round(overall[f1_key] - overall["base_f1"], 4)
    overall["cot_n_improved"]         = _n_improved("cot_f1")
    overall["thinking_n_improved"]    = _n_improved("thinking_f1")
    overall["cot_thinking_n_improved"] = _n_improved("cot_thinking_f1")

    return {
        "overall":    overall,
        "by_task":    {tt: _mean_group(d) for tt, d in by_task.items()},
        "by_dataset": {ds: _mean_group(d) for ds, d in by_ds.items()},
        "per_class":  eval_results,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLORS = {
    "base":         "#5b8dd9",
    "cot":          "#e07b54",
    "thinking":     "#2ecc71",
    "cot_thinking": "#9b59b6",
}
_LABELS = {
    "base":         "Base",
    "cot":          "Structured CoT",
    "thinking":     "Native Thinking",
    "cot_thinking": "CoT + Thinking",
}


def _draw_judge_line(ax, bar, value: float, label: bool = False) -> None:
    x0, w = bar.get_x(), bar.get_width()
    ax.plot([x0, x0 + w], [value, value], color="black", linewidth=2.0,
            solid_capstyle="butt", zorder=5,
            label="judge score" if label else "_nolegend_")
    ax.text(x0 + w / 2, value + 0.005, f"{value:.2f}",
            ha="center", va="bottom", fontsize=5.5, color="black", zorder=6)


def _bar_chart_4(
    labels: list[str],
    vals: dict[str, list[float]],          # condition → values
    judge_vals: dict[str, list[float]],    # condition → judge values (may be empty)
    title: str,
    out_path: Path,
    xlabel: str = "",
    rotate: int = 0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    x = np.arange(n)
    width = 0.2
    offsets = {"base": -1.5, "cot": -0.5, "thinking": 0.5, "cot_thinking": 1.5}

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))

    judge_label_drawn = False
    for cond in CONDITIONS:
        v = vals.get(cond, [0.0] * n)
        offset = offsets[cond] * width
        bars = ax.bar(x + offset, v, width, label=_LABELS[cond],
                      color=_COLORS[cond], alpha=0.88)
        for bar, h in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6, color=_COLORS[cond])

        jv = judge_vals.get(cond, [])
        if len(jv) == n:
            for bar, jval in zip(bars, jv):
                _draw_judge_line(ax, bar, jval, label=not judge_label_drawn)
                judge_label_drawn = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_ylabel("F1 / judge accuracy")
    ax.set_ylim(0, 1.18)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _extract_vals(source: dict, f1_keys: dict[str, str], judge_keys: dict[str, str]) \
        -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Pull ordered F1 and judge value lists from an ordered dict of group entries."""
    group_labels = list(source.keys())
    vals, jvals = {}, {}
    for cond in CONDITIONS:
        vals[cond]  = [source[g].get(f1_keys[cond],    0.0) for g in group_labels]
        jvals[cond] = [source[g].get(judge_keys[cond], 0.0) for g in group_labels]
    return vals, jvals


_F1_KEYS   = {c: f"{c}_f1"        for c in CONDITIONS}
_JUDGE_KEYS = {c: f"{c}_judge_acc" for c in CONDITIONS}


def plot_f1_overall(results: dict, plots_dir: Path) -> None:
    ov = results.get("overall", {})
    if not ov:
        return
    vals  = {c: [ov.get(f"{c}_f1", 0.0)]        for c in CONDITIONS}
    jvals = {c: [ov.get(f"{c}_judge_acc", 0.0)]  for c in CONDITIONS}
    cot_gain = ov.get("cot_gain", 0.0)
    _bar_chart_4(
        ["overall"], vals, jvals,
        title=(
            f"CoT Eval — Overall F1  (n={ov.get('n_classes',0)} classes)\n"
            f"cot gain={cot_gain:+.4f}  "
            f"thinking gain={ov.get('thinking_gain',0):+.4f}  "
            f"cot+thinking gain={ov.get('cot_thinking_gain',0):+.4f}"
        ),
        out_path=plots_dir / "f1_overall.png",
    )


def plot_f1_by_task(results: dict, plots_dir: Path) -> None:
    by_task = results.get("by_task", {})
    if not by_task:
        return
    labels = sorted(by_task)
    vals  = {c: [by_task[t].get(f"{c}_f1",        0.0) for t in labels] for c in CONDITIONS}
    jvals = {c: [by_task[t].get(f"{c}_judge_acc",  0.0) for t in labels] for c in CONDITIONS}
    _bar_chart_4(labels, vals, jvals, title="CoT Eval — F1 by Task Type",
                 out_path=plots_dir / "f1_by_task.png")


def plot_f1_by_dataset(results: dict, plots_dir: Path) -> None:
    by_ds = results.get("by_dataset", {})
    if not by_ds:
        return
    labels = sorted(by_ds)
    vals  = {c: [by_ds[d].get(f"{c}_f1",        0.0) for d in labels] for c in CONDITIONS}
    jvals = {c: [by_ds[d].get(f"{c}_judge_acc",  0.0) for d in labels] for c in CONDITIONS}
    _bar_chart_4(labels, vals, jvals, title="CoT Eval — F1 by Dataset",
                 out_path=plots_dir / "f1_by_dataset.png", rotate=45)


def plot_f1_per_class(results: dict, plots_dir: Path) -> None:
    per_class = results.get("per_class", {})
    if not per_class:
        return
    items = sorted(per_class.items())
    labels = [v.get("class_label", k) for k, v in items]
    vals  = {c: [v.get(f"{c}_f1",        0.0) for _, v in items] for c in CONDITIONS}
    jvals = {c: [v.get(f"{c}_judge_acc",  0.0) for _, v in items] for c in CONDITIONS}
    _bar_chart_4(labels, vals, jvals, title="CoT Eval — F1 per Class",
                 out_path=plots_dir / "f1_per_class.png", xlabel="class", rotate=45)


def plot_cot_length_vs_gain(eval_results: dict, plots_dir: Path) -> None:
    """Scatter: x = mean CoT observation length (words), y = F1 gain (cot vs base)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, labels, task_types = [], [], [], []
    for cls_key, e in eval_results.items():
        xs.append(float(e.get("mean_cot_length_words", 0.0)))
        ys.append(float(e.get("cot_gain", 0.0)))
        labels.append(e.get("class_label", cls_key))
        task_types.append(e.get("task_type", "unknown"))

    if not xs:
        return

    task_palette = {"disease": "#e07b54", "pest_damage": "#9b59b6", "weed_identification": "#2ecc71"}
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
            ax.text(0.04, 0.96, f"Pearson r = {corr:.3f}  (p={p:.3f})",
                    transform=ax.transAxes, fontsize=10, va="top",
                    color="darkgreen" if corr > 0 else "darkred",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        except ImportError:
            pass

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=col, label=tt)
                        for tt, col in task_palette.items() if tt in task_types],
              loc="lower right", fontsize=8, title="task type")

    ax.set_xlabel("Mean CoT analysis length (words)")
    ax.set_ylabel("F1 gain — Structured CoT vs Base")
    ax.set_title("Does longer structured analysis predict larger F1 gain?\n"
                 "Each point = one class   |   Above zero = CoT helps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = plots_dir / "cot_length_vs_gain.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gain_distribution(eval_results: dict, plots_dir: Path) -> None:
    """Overlapping histograms of F1 gains for the 3 non-base conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    non_base = {
        "cot":          ("cot_gain",         _COLORS["cot"],          "Structured CoT"),
        "thinking":     ("thinking_gain",     _COLORS["thinking"],     "Native Thinking"),
        "cot_thinking": ("cot_thinking_gain", _COLORS["cot_thinking"], "CoT + Thinking"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for cond, (gain_key, color, label) in non_base.items():
        gains = [e[gain_key] for e in eval_results.values() if gain_key in e]
        if gains:
            ax.hist(gains, bins=15, color=color, alpha=0.55, label=f"{label} (mean={np.mean(gains):+.3f})",
                    edgecolor="white")

    ax.axvline(0, color="crimson", lw=2, ls="--", label="no gain baseline")
    ax.set_xlabel("F1 gain vs base")
    ax.set_ylabel("Number of classes")
    ax.set_title("Distribution of F1 gains by prompting condition")
    ax.legend(fontsize=8)
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
        "",
        f"{'Condition':<20} {'F1':>6}  {'Judge':>6}  {'Gain':>7}  {'Improved':>9}",
        "-" * 55,
    ]
    for cond, label in _LABELS.items():
        f1    = ov.get(f"{cond}_f1", 0.0)
        judge = ov.get(f"{cond}_judge_acc", 0.0)
        gain  = ov.get(f"{cond}_gain", 0.0) if cond != "base" else 0.0
        n_imp = ov.get(f"{cond}_n_improved", "-") if cond != "base" else "-"
        lines.append(f"  {label:<18} {f1:>6.4f}  {judge:>6.4f}  {gain:>+7.4f}  {str(n_imp):>9}")

    lines += ["", "By task type:"]
    for tt, v in sorted(results.get("by_task", {}).items()):
        lines.append(f"  {tt}")
        for cond in CONDITIONS:
            f1   = v.get(f"{cond}_f1",  0.0)
            gain = round(f1 - v.get("base_f1", 0.0), 4) if cond != "base" else 0.0
            lines.append(f"    {_LABELS[cond]:<20} F1={f1:.4f}  gain={gain:+.4f}")

    lines += ["", "Top-5 classes by cot_gain:"]
    top = sorted(eval_results.values(), key=lambda e: e.get("cot_gain", 0), reverse=True)[:5]
    for e in top:
        lines.append(f"  {e['class_label']:30s}  cot_gain={e.get('cot_gain',0):+.4f}  "
                     f"thinking_gain={e.get('thinking_gain',0):+.4f}  "
                     f"cot_len={e.get('mean_cot_length_words',0):.0f}w")

    lines += ["", "Top-5 classes by thinking_gain:"]
    top2 = sorted(eval_results.values(), key=lambda e: e.get("thinking_gain", 0), reverse=True)[:5]
    for e in top2:
        lines.append(f"  {e['class_label']:30s}  thinking_gain={e.get('thinking_gain',0):+.4f}")

    report = "\n".join(lines) + "\n"
    path = output_dir / "results_summary.txt"
    path.write_text(report)
    print(f"  Saved: {path}")
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoT Eval — 4-way ablation of CoT prompting strategies")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--config", default="src/config/cot_prompts.yaml")
    parser.add_argument("--datasets_config", default="src/config/datasets.yaml")
    parser.add_argument("--split_config", default="src/config/dataset_split.yaml")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--output_dir", default="outputs/cot_eval")
    parser.add_argument("--n_eval", type=int, default=None)
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

    prompts_cfg = yaml.safe_load(
        Path(cfg.get("prompts_config", "src/config/prompts.yaml")).read_text()
    )

    if args.plots_only:
        results_path = output_dir / "results.json"
        if not results_path.exists():
            print(f"No results.json at {results_path} — run inference first.")
            return
        results = json.loads(results_path.read_text())
        eval_results = results.get("per_class", {})
        print(f"Loaded {len(eval_results)} classes from {results_path}")
    else:
        datasets_cfg = datasets_by_name(Path(args.datasets_config))
        split_raw    = yaml.safe_load(Path(args.split_config).read_text())

        if args.split == "train":
            dataset_names = split_raw.get("train", [])
        elif args.split == "test":
            dataset_names = split_raw.get("test", [])
        else:
            dataset_names = split_raw.get("train", []) + split_raw.get("test", [])

        dataset_names = [d for d in dataset_names if d in datasets_cfg and not datasets_cfg[d].skip]
        print(f"Datasets : {len(dataset_names)}")
        print(f"Config   : n_eval={cfg['n_eval']}  max_tokens_step={cfg.get('max_tokens_step', 256)}  "
              f"max_tokens_thinking={cfg.get('max_tokens_thinking', 4096)}  "
              f"gpu_util={cfg.get('gpu_memory_utilization', 0.85)}")

        checkpoint   = load_checkpoint(output_dir)
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
