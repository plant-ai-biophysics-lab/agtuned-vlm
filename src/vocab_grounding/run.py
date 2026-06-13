"""
Vocab Grounding Experiment
==========================
Tests whether injecting class-discriminative tokens from the VLM's own embedding
space into classification prompts improves F1.

Hypothesis (from gemma4_disease_grounding.ipynb):
  Diagnostic vocabulary tokens like "disease", "pest", "stress" are weakly grounded
  in visual space — they attend weakly to image tokens. But the VLM's embedding
  table contains vocabulary whose embeddings are geometrically close to specific
  disease-class visual representations. Injecting those tokens into the prompt
  should anchor attention to the right visual features, improving F1.

Pipeline
--------
Phase 1 — Grounding analysis (HuggingFace transformers, eager inference):
  For each class, run N images through the model with output_hidden_states=True.
  Compute mean image-token hidden-state centroid per class. Then, for each class,
  find vocabulary tokens with highest (sim_to_target - mean_sim_to_others),
  i.e. maximally discriminative across classes.
  → Saves discriminative_vocab.json

Phase 2 — Evaluation (vLLM, fast batch inference):
  Compare two prompt types per class:
    base:        closed-form "{crop}, choose from {class_list}"
    contrastive: same base + prepended discriminative token descriptors
  Measure F1 for each class under both conditions.
  → Saves results.json + class_log.jsonl

Outputs (under output_dir/):
  discriminative_vocab.json  — per-class discriminative tokens + scores
  results.json               — F1 base vs contrastive, per class/dataset/task
  class_log.jsonl            — per-class evaluation records
  checkpoint.json            — incremental checkpoint
  plots/
    f1_overall.png
    f1_by_task.png
    f1_by_dataset.png
    f1_per_class.png
    token_grounding_{dataset}.png  — discriminability heatmap per dataset
    discriminability_vs_gain.png   — scatter: does disc score predict F1 gain?
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
import torch
import yaml
from PIL import Image as PILImage

from src.common.dataset_config import datasets_by_name, clean_class_list, parse_class_label
from src.common.metrics import resolve_class, compute_classification_metrics

# ── regex ─────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ── checkpointing ─────────────────────────────────────────────────────────────

def load_checkpoint(output_dir: Path) -> dict:
    p = output_dir / "checkpoint.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_checkpoint(output_dir: Path, ckpt: dict) -> None:
    (output_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))


# ── logging ───────────────────────────────────────────────────────────────────

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


# ── Phase 1: Grounding analysis (HuggingFace transformers) ───────────────────

def load_hf_model(model_id: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"[phase1] Loading {model_id} via transformers ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[phase1] Model loaded  dtype={next(model.parameters()).dtype}")
    return model, processor


def unload_hf_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[phase1] HF model unloaded")


def _image_token_id(processor) -> int:
    if hasattr(processor, "image_token_id"):
        return int(processor.image_token_id)
    tok = getattr(processor.tokenizer, "image_token", "<image>")
    return int(processor.tokenizer.convert_tokens_to_ids(tok))


def _build_hf_inputs(processor, image: PILImage.Image, prompt: str, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    try:
        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            return_tensors=None,
        )
    return processor(text=text_input, images=image, return_tensors="pt").to(device)


def compute_image_centroid(
    model,
    processor,
    image_paths: list[str],
    prompt: str,
    img_tok_id: int,
    max_images: int = 10,
) -> np.ndarray | None:
    """
    Forward N images → compute mean of image-token hidden states at final layer.
    Returns L2-normalized centroid, or None if all images fail.
    """
    device = next(model.parameters()).device
    centroids = []

    for path in image_paths[:max_images]:
        try:
            img = PILImage.open(path).convert("RGB")
            inputs = _build_hf_inputs(processor, img, prompt, device)
        except Exception as e:
            print(f"      [warn] load/preprocess failed: {path} — {e}")
            continue

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            except Exception as e:
                print(f"      [warn] forward pass failed: {path} — {e}")
                continue

        input_ids = inputs["input_ids"][0]
        img_positions = (input_ids == img_tok_id).nonzero(as_tuple=True)[0]
        if len(img_positions) == 0:
            continue

        last_h = outputs.hidden_states[-1][0].float().cpu()
        img_h = last_h[img_positions.cpu()].mean(0)
        img_h = img_h / (img_h.norm() + 1e-6)
        centroids.append(img_h.numpy())

    if not centroids:
        return None

    centroid = np.mean(centroids, axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    return centroid / (norm + 1e-6)


def find_discriminative_tokens(
    model,
    processor,
    target_centroid: np.ndarray,
    other_centroids: dict[str, np.ndarray],
    top_k: int = 15,
    min_len: int = 3,
) -> list[dict]:
    """
    Search full vocab embedding table for tokens with highest
    (sim_to_target - mean_sim_to_others).
    Only alphabetic tokens of length >= min_len are returned.
    """
    embed_w = model.get_input_embeddings().weight.detach().float().cpu()
    norms = embed_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    embed_norm = embed_w / norms  # (vocab_size, d_model)

    target_t = torch.tensor(target_centroid, dtype=torch.float32)
    target_t = target_t / (target_t.norm() + 1e-6)
    sim_target = embed_norm @ target_t  # (vocab_size,)

    if other_centroids:
        other_sims = []
        for centroid in other_centroids.values():
            c = torch.tensor(centroid, dtype=torch.float32)
            c = c / (c.norm() + 1e-6)
            other_sims.append(embed_norm @ c)
        sim_others_mean = torch.stack(other_sims).mean(0)  # (vocab_size,)
        discriminability = sim_target - sim_others_mean
    else:
        discriminability = sim_target

    # Take a large pool before filtering to readable tokens
    pool_size = top_k * 50
    top_ids = discriminability.argsort(descending=True)[:pool_size]
    top_tokens = processor.tokenizer.convert_ids_to_tokens(top_ids.tolist())

    result = []
    for tok, tok_id, disc, sim in zip(
        top_tokens,
        top_ids.tolist(),
        discriminability[top_ids].tolist(),
        sim_target[top_ids].tolist(),
    ):
        clean = tok.replace("▁", "").replace("Ġ", "").replace("##", "").strip()
        if clean.isalpha() and len(clean) >= min_len:
            result.append({
                "token": clean,
                "raw_token": tok,
                "token_id": tok_id,
                "discriminability": round(float(disc), 5),
                "sim_to_class": round(float(sim), 5),
            })
        if len(result) >= top_k:
            break

    return result


def run_grounding_phase(
    model_id: str,
    datasets_cfg: dict,
    dataset_names: list[str],
    cfg: dict,
    checkpoint: dict,
    output_dir: Path,
    force: bool,
    seed: int,
) -> dict:
    """
    Phase 1: Compute image centroids + discriminative vocab per class.
    Returns disc_vocab dict and updates checkpoint in place.
    """
    n_grounding = int(cfg.get("n_grounding", 10))
    top_k = int(cfg.get("top_k_tokens", 15))
    min_len = int(cfg.get("min_token_len", 3))
    base_prompt_tmpl = cfg["base_prompt"]

    disc_vocab: dict = checkpoint.get("disc_vocab", {})

    model, processor = load_hf_model(model_id)
    img_tok_id = _image_token_id(processor)

    for agml_name in dataset_names:
        ds_cfg = datasets_cfg.get(agml_name)
        if ds_cfg is None:
            print(f"  [warn] {agml_name} not in datasets config — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[phase1] Dataset: {agml_name}")

        try:
            sampled = load_agml_images(agml_name, n_grounding, seed=seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        all_classes = clean_class_list(ds_cfg.classes)

        # Check if all classes already done
        class_keys = []
        for raw_cls in sampled:
            _, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            class_keys.append(f"{agml_name}/{clean_cls}")
        if not force and all(k in disc_vocab for k in class_keys):
            print("  [skip] all classes already grounded")
            continue

        # Pass 1: compute centroids for all classes in this dataset
        centroids: dict[str, np.ndarray] = {}
        crop_map: dict[str, str] = {}

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            crop_map[clean_cls] = crop
            prompt = base_prompt_tmpl.format(crop=crop)
            print(f"    centroid → {clean_cls} ({len(image_paths)} imgs)")
            centroid = compute_image_centroid(
                model, processor, image_paths, prompt, img_tok_id, max_images=n_grounding
            )
            if centroid is not None:
                centroids[clean_cls] = centroid
            else:
                print(f"    [warn] no centroid for {clean_cls}")

        # Pass 2: find discriminative tokens using cross-class discriminability
        for raw_cls, image_paths in sampled.items():
            _, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            if not force and cls_key in disc_vocab:
                print(f"    [skip] {clean_cls}")
                continue

            if clean_cls not in centroids:
                print(f"    [skip] {clean_cls} — no centroid")
                continue

            other_centroids = {k: v for k, v in centroids.items() if k != clean_cls}
            disc_tokens = find_discriminative_tokens(
                model, processor,
                centroids[clean_cls],
                other_centroids,
                top_k=top_k,
                min_len=min_len,
            )

            disc_vocab[cls_key] = {
                "crop": crop_map[clean_cls],
                "class_label": clean_cls,
                "dataset": agml_name,
                "tokens": disc_tokens,
            }
            top_str = ", ".join(t["token"] for t in disc_tokens[:5])
            print(f"    disc vocab → {clean_cls}: [{top_str}]")

        checkpoint["disc_vocab"] = disc_vocab
        save_checkpoint(output_dir, checkpoint)

    unload_hf_model(model)

    vocab_path = output_dir / "discriminative_vocab.json"
    vocab_path.write_text(json.dumps(disc_vocab, indent=2))
    print(f"\n[phase1] Saved → {vocab_path}  ({len(disc_vocab)} classes)")
    return disc_vocab


# ── Phase 2: Evaluation (vLLM) ────────────────────────────────────────────────

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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[phase2] vLLM engine destroyed")


def _build_vllm_prompt(tokenizer, image: PILImage.Image, question: str) -> dict:
    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    try:
        text = tokenizer.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def evaluate_prompt(
    llm,
    tokenizer,
    prompt: str,
    image_paths: list[str],
    class_label: str,
    all_classes: list[str],
    cls_params,
    n_eval: int | None = None,
) -> tuple[float, float, list[str]]:
    """
    Evaluate prompt on images for one class.
    Returns (class_f1, resolvability_rate, raw_responses).
    """
    from sklearn.metrics import f1_score

    paths = image_paths[:n_eval] if n_eval else image_paths
    prompts = []
    for p in paths:
        try:
            img = PILImage.open(p).convert("RGB")
            prompts.append(_build_vllm_prompt(tokenizer, img, prompt))
        except Exception as e:
            print(f"      [warn] {p}: {e}")

    if not prompts:
        return 0.0, 0.0, []

    outputs = llm.generate(prompts, sampling_params=cls_params)
    raw = [_clean(o.outputs[0].text) for o in outputs]

    refs = [class_label] * len(raw)
    preds = [resolve_class(r, all_classes) for r in raw]
    resolved = [p or "" for p in preds]

    resolvability = sum(1 for p in preds if p) / max(len(preds), 1)
    class_f1 = f1_score(refs, resolved, labels=[class_label], average="macro", zero_division=0)
    return round(float(class_f1), 4), round(resolvability, 4), raw


def build_contrastive_prompt(
    disc_tokens: list[dict],
    n_tokens: int,
    min_gap: float,
    template: str,
    crop: str,
) -> tuple[str | None, list[dict]]:
    """
    Build contrastive prompt with discriminative tokens above min_gap threshold.
    Returns (prompt or None, tokens_used).
    None → no tokens passed the threshold; caller should fall back to base.
    """
    filtered = [t for t in disc_tokens if t["discriminability"] >= min_gap]
    if not filtered:
        return None, []
    used = filtered[:n_tokens]
    token_str = ", ".join(t["token"] for t in used)
    prompt = template.format(crop=crop, disc_tokens=token_str)
    return prompt, used


def run_evaluation_phase(
    model_id: str,
    disc_vocab: dict,
    datasets_cfg: dict,
    dataset_names: list[str],
    cfg: dict,
    checkpoint: dict,
    output_dir: Path,
    force: bool,
    seed: int,
) -> dict:
    """Phase 2: Evaluate base vs contrastive prompts via vLLM."""
    from vllm import SamplingParams

    n_eval = int(cfg.get("n_eval", 10))
    min_gap = float(cfg.get("min_gap", 0.003))
    n_disc_tokens = int(cfg.get("n_disc_tokens", 3))
    base_tmpl = cfg["base_prompt"]
    cont_tmpl = cfg["contrastive_prompt"]
    gpu_util = float(cfg.get("gpu_memory_utilization", 0.55))
    max_model_len = int(cfg.get("max_model_len", 6144))

    cls_params = SamplingParams(
        temperature=float(cfg.get("classifier_temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens_response", 64)),
        repetition_penalty=1.1,
    )

    eval_results: dict = checkpoint.get("eval_results", {})
    log_path = output_dir / "class_log.jsonl"

    llm, tokenizer = load_vlm(model_id, gpu_util, max_model_len)

    for agml_name in dataset_names:
        ds_cfg = datasets_cfg.get(agml_name)
        if ds_cfg is None:
            continue

        print(f"\n{'='*60}")
        print(f"[phase2] Dataset: {agml_name}")

        try:
            sampled = load_agml_images(agml_name, n_eval, seed=seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        all_classes = clean_class_list(ds_cfg.classes)

        for raw_cls, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_cls, ds_cfg.crop)
            cls_key = f"{agml_name}/{clean_cls}"

            if not force and cls_key in eval_results:
                print(f"  [skip] {clean_cls}")
                continue

            print(f"\n  Class: {clean_cls}")

            # Base prompt — open-ended, no class list
            base_prompt = base_tmpl.format(crop=crop)
            base_f1, base_resolvable, base_responses = evaluate_prompt(
                llm, tokenizer, base_prompt, image_paths,
                clean_cls, all_classes, cls_params, n_eval,
            )
            print(f"    base       F1={base_f1:.3f}  resolvable={base_resolvable:.2f}")

            # Contrastive prompt — disc tokens prepended, then same OEQ
            disc_entry = disc_vocab.get(cls_key, {})
            disc_tokens = disc_entry.get("tokens", [])
            cont_prompt, tokens_used = build_contrastive_prompt(
                disc_tokens, n_disc_tokens, min_gap, cont_tmpl, crop
            )

            if cont_prompt is None:
                print(f"    [info] no tokens above gap={min_gap} — contrastive = base")
                cont_prompt = base_prompt
                tokens_used = []

            else:
                used_str = ", ".join(t["token"] for t in tokens_used)
                print(f"    injected:  [{used_str}]")

            cont_f1, cont_resolvable, cont_responses = evaluate_prompt(
                llm, tokenizer, cont_prompt, image_paths,
                clean_cls, all_classes, cls_params, n_eval,
            )
            print(f"    contrastive F1={cont_f1:.3f}  resolvable={cont_resolvable:.2f}  gain={cont_f1-base_f1:+.3f}")

            entry = {
                "dataset": agml_name,
                "class_label": clean_cls,
                "crop": crop,
                "task_type": ds_cfg.task_type,
                "base_f1": base_f1,
                "cont_f1": cont_f1,
                "gain": round(cont_f1 - base_f1, 4),
                "base_resolvable": base_resolvable,
                "cont_resolvable": cont_resolvable,
                "base_prompt": base_prompt,
                "cont_prompt": cont_prompt,
                "disc_tokens_used": [t["token"] for t in tokens_used],
                "disc_token_details": tokens_used,
                "n_disc_tokens_used": len(tokens_used),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            eval_results[cls_key] = entry

            # Log with responses separately (heavier)
            append_jsonl(log_path, {
                **entry,
                "base_responses": base_responses,
                "cont_responses": cont_responses,
            })

            checkpoint["eval_results"] = eval_results
            save_checkpoint(output_dir, checkpoint)

    destroy_vlm(llm)
    return eval_results


# ── Results aggregation ───────────────────────────────────────────────────────

def aggregate_results(eval_results: dict) -> dict:
    all_e = list(eval_results.values())
    if not all_e:
        return {}

    base_f1s = [e["base_f1"] for e in all_e]
    cont_f1s = [e["cont_f1"] for e in all_e]
    n_improved = sum(1 for e in all_e if e["cont_f1"] > e["base_f1"])

    by_task: dict[str, list] = defaultdict(lambda: [[], []])
    by_ds: dict[str, list] = defaultdict(lambda: [[], []])
    for e in all_e:
        if e.get("task_type"):
            by_task[e["task_type"]][0].append(e["base_f1"])
            by_task[e["task_type"]][1].append(e["cont_f1"])
        if e.get("dataset"):
            by_ds[e["dataset"]][0].append(e["base_f1"])
            by_ds[e["dataset"]][1].append(e["cont_f1"])

    def _mean_pair(v):
        return {
            "base_f1_macro": round(float(np.mean(v[0])), 4),
            "cont_f1_macro": round(float(np.mean(v[1])), 4),
            "gain": round(float(np.mean(v[1])) - float(np.mean(v[0])), 4),
            "n_classes": len(v[0]),
        }

    return {
        "overall": {
            "base_f1_macro": round(float(np.mean(base_f1s)), 4),
            "cont_f1_macro": round(float(np.mean(cont_f1s)), 4),
            "gain": round(float(np.mean(cont_f1s)) - float(np.mean(base_f1s)), 4),
            "n_classes": len(all_e),
            "n_improved": n_improved,
            "pct_improved": round(n_improved / max(len(all_e), 1), 4),
        },
        "by_task": {tt: _mean_pair(v) for tt, v in by_task.items()},
        "by_dataset": {ds: _mean_pair(v) for ds, v in by_ds.items()},
        "per_class": eval_results,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _bar_chart(
    labels: list[str],
    base_vals: list[float],
    cont_vals: list[float],
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
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    def _annotate(bars, color):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color=color)

    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base (F1)", color="#5b8dd9", alpha=0.9)
    _annotate(bars_base, "#5b8dd9")

    bars_cont = ax.bar(x + width / 2, cont_vals, width, label="Contrastive (F1)", color="#e07b54", alpha=0.9)
    _annotate(bars_cont, "#c0522a")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_ylabel("F1")
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
    _bar_chart(
        ["overall"],
        [ov.get("base_f1_macro", 0)],
        [ov.get("cont_f1_macro", 0)],
        title=(
            f"Vocab Grounding — Overall F1\n"
            f"n={ov.get('n_classes',0)} classes  "
            f"{ov.get('pct_improved',0):.0%} improved  "
            f"gain={ov.get('gain',0):+.4f}"
        ),
        out_path=plots_dir / "f1_overall.png",
    )


def plot_f1_by_task(results: dict, plots_dir: Path) -> None:
    by_task = results.get("by_task", {})
    if not by_task:
        return
    labels = sorted(by_task)
    _bar_chart(
        labels,
        [by_task[t]["base_f1_macro"] for t in labels],
        [by_task[t]["cont_f1_macro"] for t in labels],
        title="Vocab Grounding — F1 by Task Type",
        out_path=plots_dir / "f1_by_task.png",
    )


def plot_f1_by_dataset(results: dict, plots_dir: Path) -> None:
    by_ds = results.get("by_dataset", {})
    if not by_ds:
        return
    labels = sorted(by_ds)
    _bar_chart(
        labels,
        [by_ds[d]["base_f1_macro"] for d in labels],
        [by_ds[d]["cont_f1_macro"] for d in labels],
        title="Vocab Grounding — F1 by Dataset",
        out_path=plots_dir / "f1_by_dataset.png",
        rotate=45,
    )


def plot_f1_per_class(results: dict, plots_dir: Path) -> None:
    per_class = results.get("per_class", {})
    if not per_class:
        return
    items = sorted(per_class.items())
    _bar_chart(
        [v.get("class_label", k) for k, v in items],
        [v["base_f1"] for _, v in items],
        [v["cont_f1"] for _, v in items],
        title="Vocab Grounding — F1 per Class",
        out_path=plots_dir / "f1_per_class.png",
        xlabel="class",
        rotate=45,
    )


def plot_token_grounding_heatmap(disc_vocab: dict, plots_dir: Path) -> None:
    """
    One heatmap per dataset: rows = top discriminative tokens across all classes,
    cols = disease classes. Cell value = discriminability score.
    Colored labels indicate which class each token was most discriminative for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_dataset: dict[str, dict] = defaultdict(dict)
    for cls_key, entry in disc_vocab.items():
        ds = entry.get("dataset", "unknown")
        cls = entry.get("class_label", cls_key)
        by_dataset[ds][cls] = entry

    for ds_name, class_entries in by_dataset.items():
        classes = sorted(class_entries)
        if not classes:
            continue

        # Collect top-3 tokens per class (unique union)
        tok_order: list[str] = []
        tok_source: dict[str, str] = {}  # tok → class that nominated it
        for cls in classes:
            for t in class_entries[cls].get("tokens", [])[:3]:
                tok = t["token"]
                if tok not in tok_order:
                    tok_order.append(tok)
                    tok_source[tok] = cls

        if not tok_order:
            continue

        # Matrix: rows=tokens, cols=classes
        tok_disc_map: dict[str, dict[str, float]] = defaultdict(dict)
        for cls in classes:
            for t in class_entries[cls].get("tokens", []):
                tok_disc_map[t["token"]][cls] = t["discriminability"]

        matrix = np.array([
            [tok_disc_map[tok].get(cls, 0.0) for cls in classes]
            for tok in tok_order
        ])

        n_rows, n_cols = len(tok_order), len(classes)
        fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.3), max(5, n_rows * 0.55)))

        vmin, vmax = matrix.min(), matrix.max()
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(tok_order, fontsize=8)

        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)

        # Color y-tick labels by their nominating class (tab10 palette)
        palette = plt.cm.tab10.colors
        cls_color = {cls: palette[i % 10] for i, cls in enumerate(classes)}
        for ytick, tok in zip(ax.get_yticklabels(), tok_order):
            src = tok_source.get(tok)
            if src and src in cls_color:
                ytick.set_color(cls_color[src])
                ytick.set_fontweight("bold")

        plt.colorbar(im, ax=ax, label="discriminability (sim_to_class − mean_sim_to_others)")
        ax.set_title(
            f"{ds_name} — Discriminative Token Grounding\n"
            "Y-label color = class that nominated the token  |  "
            "Green = strongly discriminative"
        )
        fig.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9_]", "_", ds_name)
        out_path = plots_dir / f"token_grounding_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_discriminability_vs_gain(disc_vocab: dict, eval_results: dict, plots_dir: Path) -> None:
    """
    Scatter: x = mean discriminability of injected tokens, y = F1 gain.
    Tests the core hypothesis: does higher discriminability → larger gain?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, labels, task_types = [], [], [], []
    for cls_key, entry in eval_results.items():
        used = entry.get("disc_token_details", [])
        if not used:
            used = [{"discriminability": 0.0}]
        mean_disc = float(np.mean([t["discriminability"] for t in used]))
        gain = entry.get("gain", 0.0)
        xs.append(mean_disc)
        ys.append(float(gain))
        labels.append(entry.get("class_label", cls_key))
        task_types.append(entry.get("task_type", "unknown"))

    if not xs:
        return

    task_palette = {"disease": "#e07b54", "pest_damage": "#9b59b6",
                    "weed_identification": "#2ecc71", "crop_identification": "#3498db"}
    colors = [task_palette.get(tt, "gray") for tt in task_types]

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(xs, ys, c=colors, alpha=0.75, s=80, edgecolors="k", linewidths=0.5)

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5, alpha=0.8)

    ax.axhline(0, color="gray", lw=1.2, ls="--")

    # Pearson correlation
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

    # Legend for task types
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=col, label=tt)
        for tt, col in task_palette.items()
        if tt in task_types
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, title="task type")

    ax.set_xlabel("Mean discriminability of injected tokens\n(sim_to_class − mean_sim_to_others)")
    ax.set_ylabel("F1 gain (contrastive − base)")
    ax.set_title(
        "Does higher token discriminability predict larger F1 gain?\n"
        "Each point = one class   |   Above zero line = contrastive helps"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = plots_dir / "discriminability_vs_gain.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gain_distribution(eval_results: dict, plots_dir: Path) -> None:
    """Histogram of F1 gains across all classes."""
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
    ax.set_xlabel("F1 gain (contrastive − base)")
    ax.set_ylabel("Number of classes")
    ax.set_title(
        f"Distribution of F1 gains from vocab grounding injection\n"
        f"↑ improved: {n_pos}   ↓ degraded: {n_neg}   → no change: {len(gains)-n_pos-n_neg}"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out_path = plots_dir / "gain_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_threshold_sensitivity(disc_vocab: dict, eval_results: dict, plots_dir: Path) -> None:
    """
    Shows how many classes have at least K tokens above each gap threshold,
    and what fraction use the fallback (base) prompt.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [0.000, 0.001, 0.002, 0.003, 0.005, 0.010]
    n_total = len(eval_results)

    coverage = []
    for thresh in thresholds:
        n_covered = 0
        for cls_key in eval_results:
            entry = disc_vocab.get(cls_key, {})
            toks = [t for t in entry.get("tokens", []) if t["discriminability"] >= thresh]
            if toks:
                n_covered += 1
        coverage.append(n_covered / max(n_total, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, coverage, marker="o", color="#5b8dd9", linewidth=2.5)
    ax.axvline(0.003, color="crimson", lw=1.5, ls="--", label="default min_gap=0.003")
    ax.set_xlabel("min_gap threshold (discriminability)")
    ax.set_ylabel("Fraction of classes with ≥1 token above threshold")
    ax.set_title(
        "Threshold sensitivity: coverage vs discriminability filter\n"
        "Lower threshold = more classes get contrastive tokens"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = plots_dir / "threshold_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary_report(results: dict, disc_vocab: dict, eval_results: dict, output_dir: Path) -> None:
    """Write a human-readable text summary to results_summary.txt."""
    ov = results.get("overall", {})
    lines = [
        "Vocab Grounding Experiment — Results Summary",
        "=" * 60,
        "",
        f"Classes evaluated : {ov.get('n_classes', 0)}",
        f"Base F1 (macro)   : {ov.get('base_f1_macro', 0):.4f}",
        f"Contrastive F1    : {ov.get('cont_f1_macro', 0):.4f}",
        f"Mean gain         : {ov.get('gain', 0):+.4f}",
        f"Classes improved  : {ov.get('n_improved', 0)}  ({ov.get('pct_improved', 0):.1%})",
        "",
        "By task type:",
    ]
    for tt, v in sorted(results.get("by_task", {}).items()):
        lines.append(
            f"  {tt:25s}  base={v['base_f1_macro']:.4f}  cont={v['cont_f1_macro']:.4f}  "
            f"gain={v['gain']:+.4f}  n={v['n_classes']}"
        )

    lines += ["", "By dataset:"]
    for ds, v in sorted(results.get("by_dataset", {}).items()):
        lines.append(
            f"  {ds:40s}  base={v['base_f1_macro']:.4f}  cont={v['cont_f1_macro']:.4f}  "
            f"gain={v['gain']:+.4f}"
        )

    lines += ["", "Top-5 improved classes:"]
    top = sorted(eval_results.values(), key=lambda e: e.get("gain", 0), reverse=True)[:5]
    for e in top:
        toks = ", ".join(e.get("disc_tokens_used", []))
        lines.append(f"  {e['class_label']:25s}  gain={e['gain']:+.4f}  tokens=[{toks}]")

    lines += ["", "Top-5 degraded classes:"]
    bot = sorted(eval_results.values(), key=lambda e: e.get("gain", 0))[:5]
    for e in bot:
        toks = ", ".join(e.get("disc_tokens_used", []))
        lines.append(f"  {e['class_label']:25s}  gain={e['gain']:+.4f}  tokens=[{toks}]")

    report = "\n".join(lines) + "\n"
    path = output_dir / "results_summary.txt"
    path.write_text(report)
    print(f"  Saved: {path}")
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vocab Grounding Experiment")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it",
                        help="HuggingFace model ID (used for both phases)")
    parser.add_argument("--config", default="src/config/vocab_grounding.yaml")
    parser.add_argument("--datasets_config", default="src/config/datasets.yaml")
    parser.add_argument("--split_config", default="src/config/dataset_split.yaml")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test",
                        help="Dataset split to evaluate (default: test for generalization)")
    parser.add_argument("--output_dir", default="outputs/vocab_grounding")

    parser.add_argument("--n_grounding", type=int, default=None,
                        help="Override n_grounding from config")
    parser.add_argument("--n_eval", type=int, default=None,
                        help="Override n_eval from config")
    parser.add_argument("--min_gap", type=float, default=None,
                        help="Override min_gap threshold from config")
    parser.add_argument("--n_disc_tokens", type=int, default=None,
                        help="Override n_disc_tokens from config")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)

    parser.add_argument("--skip_grounding", action="store_true",
                        help="Skip Phase 1 — load disc vocab from checkpoint/discriminative_vocab.json")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip Phase 2 — load eval results from checkpoint, only replot")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all classes even if already checkpointed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load and patch config from CLI overrides
    cfg = load_config(Path(args.config))
    if args.n_grounding is not None:
        cfg["n_grounding"] = args.n_grounding
    if args.n_eval is not None:
        cfg["n_eval"] = args.n_eval
    if args.min_gap is not None:
        cfg["min_gap"] = args.min_gap
    if args.n_disc_tokens is not None:
        cfg["n_disc_tokens"] = args.n_disc_tokens
    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization

    datasets_cfg = datasets_by_name(Path(args.datasets_config))
    split_raw = yaml.safe_load(Path(args.split_config).read_text())

    if args.split == "train":
        dataset_names = split_raw.get("train", [])
    elif args.split == "test":
        dataset_names = split_raw.get("test", [])
    else:
        dataset_names = split_raw.get("train", []) + split_raw.get("test", [])

    dataset_names = [d for d in dataset_names if d in datasets_cfg and not datasets_cfg[d].skip]
    print(f"Datasets: {len(dataset_names)}")
    print(f"Config:   n_grounding={cfg['n_grounding']}  n_eval={cfg['n_eval']}  "
          f"min_gap={cfg['min_gap']}  n_disc_tokens={cfg['n_disc_tokens']}")

    checkpoint = load_checkpoint(output_dir)

    # ── Phase 1: Grounding ────────────────────────────────────────────────────
    if not args.skip_grounding:
        disc_vocab = run_grounding_phase(
            args.model, datasets_cfg, dataset_names, cfg,
            checkpoint, output_dir, args.force, args.seed,
        )
    else:
        vocab_path = output_dir / "discriminative_vocab.json"
        if vocab_path.exists():
            disc_vocab = json.loads(vocab_path.read_text())
        else:
            disc_vocab = checkpoint.get("disc_vocab", {})
        print(f"[phase1] Skipped — loaded {len(disc_vocab)} classes from cache")

    # ── Phase 2: Evaluation ───────────────────────────────────────────────────
    if not args.skip_eval:
        eval_results = run_evaluation_phase(
            args.model, disc_vocab, datasets_cfg, dataset_names, cfg,
            checkpoint, output_dir, args.force, args.seed,
        )
    else:
        eval_results = checkpoint.get("eval_results", {})
        print(f"[phase2] Skipped — {len(eval_results)} classes from checkpoint")

    if not eval_results:
        print("No evaluation results — exiting without plots")
        return

    # ── Aggregate + save ──────────────────────────────────────────────────────
    results = aggregate_results(eval_results)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {output_dir / 'results.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_f1_overall(results, plots_dir)
    plot_f1_by_task(results, plots_dir)
    plot_f1_by_dataset(results, plots_dir)
    plot_f1_per_class(results, plots_dir)
    plot_token_grounding_heatmap(disc_vocab, plots_dir)
    plot_discriminability_vs_gain(disc_vocab, eval_results, plots_dir)
    plot_gain_distribution(eval_results, plots_dir)
    plot_threshold_sensitivity(disc_vocab, eval_results, plots_dir)

    # ── Summary report ────────────────────────────────────────────────────────
    print("\nSummary:")
    write_summary_report(results, disc_vocab, eval_results, output_dir)


if __name__ == "__main__":
    main()
