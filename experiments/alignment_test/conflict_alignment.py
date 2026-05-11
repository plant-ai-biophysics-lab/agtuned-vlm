"""
Cross-Modality Knowledge Conflict — Agricultural Alignment Test
==============================================================
Adapted from: "Unraveling Cross-Modality Knowledge Conflicts in Large
Vision-Language Models" (arXiv 2410.03659).

Tests whether Gemma-4 gives consistent answers when shown:
  - Visual branch  : the actual image + question
  - Textual branch : the AgriCoT reasoning description + question (no image)

Metrics (matching the paper):
  - Accuracy          : % correct on each branch
  - Flip Rate (FR)    : % samples where visual_answer ≠ textual_answer
  - Conflict Rate (CR): FR - ΔAcc  (isolates true conflicts from accuracy gaps)
  - CD Score          : |log(P_visual[ans] / P_text[ans])|  per sample
  - Mean CD           : overall, conflict-only, non-conflict-only

Dataset: wenyb/AgriCoT (loaded from AgriCoT.zip via HuggingFace Hub)
  question  → shared question for both branches
  image     → visual branch input
  reasoning → textual branch input (symptom description)
  answer    → ground truth (A/B/C/D)

Usage:
    # Base model (default: google/gemma-4-E4B-it)
    python conflict_alignment.py

    # Specific model with sample limit
    python conflict_alignment.py --model google/gemma-4-E4B-it --n_samples 200

    # Compare fine-tuned vs base
    python conflict_alignment.py \\
        --model /path/to/finetuned \\
        --baseline_model google/gemma-4-E4B-it \\
        --n_samples 200

    # Skip inference, just re-plot saved results
    python conflict_alignment.py --plot_only --output_dir results/conflict
"""

import argparse
import io
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download


# ── Prompt templates ──────────────────────────────────────────────────────────

# MCQ prompts (AgriCoT)
MCQ_SYSTEM_PROMPT = (
    "You are an expert agricultural scientist. "
    "Answer the following multiple-choice question. "
    "Respond with ONLY the option letter: A, B, C, or D."
)

def visual_prompt(question: str, options: dict) -> str:
    opts = "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))
    return f"<image>\n{question}\n{opts}"


def textual_prompt(description: str, question: str, options: dict) -> str:
    opts = "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))
    return (
        f"Visual description of an agricultural image:\n{description}\n\n"
        f"{question}\n{opts}"
    )


# OEQ prompts (IDtools)
OEQ_SYSTEM_PROMPT = (
    "You are an expert agricultural scientist. "
    "Answer the following question about the visual appearance of the agricultural "
    "subject shown. Be specific: include colors, textures, shapes, and affected plant parts."
)

def oeq_visual_prompt(question: str) -> str:
    return f"<image>\n{question}"


def oeq_textual_prompt(description: str, question: str) -> str:
    return (
        f"Visual description of an agricultural subject:\n{description}\n\n"
        f"{question}"
    )


# Compatibility alias kept for MCQ
SYSTEM_PROMPT = MCQ_SYSTEM_PROMPT


# ── AgriCoT reasoning cleaner ────────────────────────────────────────────────

def clean_reasoning(text: str) -> str:
    """
    Remove answer-leaking content from AgriCoT reasoning text.

    Handles two leak patterns:
      1. "Final Answer: A" (and markdown variants like **Final Answer**)
         — removes the entire line.
      2. "Option A, [confirming/eliminating text]"
         — truncates after the comma so only the option label remains.

    Also strips generic "X is correct/wrong" sentences that reveal the answer.
    """
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()

        # Drop lines that explicitly state the chosen answer
        if re.search(r"final\s+answer", stripped, re.IGNORECASE):
            continue
        if re.search(
            r"\b(the\s+answer\s+is|correct\s+answer\s*:|answer\s*:\s*[A-E])\b",
            stripped, re.IGNORECASE,
        ):
            continue

        # "Option A" or "Option A (label)", [confirming/eliminating text] → "Option A (label)"
        line = re.sub(
            r"(Option\s+[A-E][^,\n]*)[,;]\s*[^\n]+",
            r"\1",
            line,
            flags=re.IGNORECASE,
        )

        # "A. [label], is correct/incorrect/matches..." → remove confirming clause
        line = re.sub(
            r"(\b[A-E]\.\s*[^,\n]{1,60}),\s*(is\s+correct|is\s+incorrect|matches|does\s+not\s+match|is\s+the\s+right|is\s+wrong)[^\n]*",
            r"\1",
            line,
            flags=re.IGNORECASE,
        )

        lines.append(line)

    cleaned = "\n".join(lines)
    # Collapse runs of blank lines left by removed lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── AgriCoT loader ────────────────────────────────────────────────────────────

def _is_qa_record(r: dict) -> bool:
    if not isinstance(r, dict):
        return False
    q = r.get("question", "")
    if not isinstance(q, str) or len(q) < 10:
        return False
    ans = r.get("answer")
    if ans is not None and not isinstance(ans, str):
        return False
    return (
        isinstance(r.get("reasoning"), str) and len(r.get("reasoning", "")) > 0
        or r.get("options") is not None
        or r.get("dimension_id") is not None
    )


def _find_qa_json(zf: zipfile.ZipFile, hint: str | None) -> list[dict]:
    json_files = [n for n in zf.namelist() if n.endswith(".json")]
    print(f"  ZIP JSON files: {json_files}")

    ordered = sorted(json_files, key=lambda n: (
        0 if (hint and hint.lower() in n.lower()) else
        0 if any(k in n.lower() for k in ("vqra", "qa", "question", "cot")) else
        1 if any(k in n.lower() for k in ("data", "meta", "sample")) else 2,
        n,
    ))

    for name in ordered:
        info = zf.getinfo(name)
        if info.file_size > 500 * 1024 * 1024:
            continue
        try:
            with zf.open(name) as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, dict):
            for key in ("data", "samples", "items", "records", "questions"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                if len(data) == 1:
                    data = list(data.values())[0]

        if not isinstance(data, list) or len(data) < 5:
            continue
        if any(_is_qa_record(r) for r in data[:5]):
            print(f"  ✓ Found {len(data):,} QA records in {name}")
            return data

    raise RuntimeError(
        f"No QA JSON found. Files: {json_files}. "
        "Try --json_hint <filename substring>."
    )


def load_agricot(n_samples: int | None,
                 zip_file: str | None = None,
                 json_hint: str | None = None) -> list[dict]:
    candidates = (zip_file,) if zip_file else ("AgriCoT_2.zip", "AgriCoT.zip")
    for filename in candidates:
        print(f"Downloading {filename} ...")
        try:
            zip_path = hf_hub_download(
                repo_id="wenyb/AgriCoT", filename=filename, repo_type="dataset"
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                records = _find_qa_json(zf, hint=json_hint)

                # Also build image lookup: image_path → raw bytes
                image_index: dict[str, bytes] = {}
                for member in zf.namelist():
                    if member.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        stem = Path(member).name
                        image_index[stem] = zf.read(member)

            break
        except RuntimeError as e:
            print(f"  {e}")
            continue
    else:
        raise RuntimeError("Could not load AgriCoT from any ZIP archive.")

    # Filter: need question + options + answer + reasoning + image_path
    valid = []
    for r in records:
        if not _is_qa_record(r):
            continue
        if not r.get("options"):
            continue
        if not r.get("reasoning"):
            continue
        ip = r.get("image_path", "")
        stem = Path(ip).name if ip else ""
        if stem not in image_index:
            continue
        r["_image_bytes"] = image_index[stem]
        r["reasoning"] = clean_reasoning(r["reasoning"])
        r.setdefault("item_id", len(valid))
        valid.append(r)

    print(f"  {len(valid):,} valid samples loaded")

    if n_samples and n_samples < len(valid):
        valid = valid[:n_samples]

    return valid


# ── IDtools loader ────────────────────────────────────────────────────────────

def load_idtools(jsonl_path: str, n_samples: int | None = None) -> list[dict]:
    """
    Load records from scrape_idtools_with_images.py output JSONL.
    Normalises to the same internal schema used by load_agricot():
      _image_bytes, question, options, answer, description
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"IDtools JSONL not found: {jsonl_path}")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Must have at least one image
            image_paths = r.get("image_paths", [])
            if not image_paths:
                continue

            # Load bytes from the first available image file
            img_bytes = None
            for p in image_paths:
                if os.path.exists(p):
                    with open(p, "rb") as img_f:
                        img_bytes = img_f.read()
                    break
            if img_bytes is None:
                continue

            r["_image_bytes"] = img_bytes
            records.append(r)

    print(f"  Loaded {len(records):,} IDtools records with images from {jsonl_path}")

    if n_samples and n_samples < len(records):
        records = records[:n_samples]

    return records


# ── vLLM inference ────────────────────────────────────────────────────────────

OPTION_LETTERS = ["A", "B", "C", "D"]


def load_vllm_model(model_name: str,
                    gpu_mem: float = 0.90,
                    max_model_len: int = 4096,
                    mode: str = "mcq"):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    if mode == "oeq":
        # Free-text generation; use mean token logprob as confidence proxy
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            logprobs=1,
        )
    else:
        # MCQ: greedy + top-20 logprobs to extract A/B/C/D probs
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4,
            logprobs=20,
        )

    return llm, tokenizer, sampling_params


def _make_visual_messages(tokenizer, record: dict) -> str:
    """MCQ: chat prompt with embedded image."""
    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user",   "content": visual_prompt(
            record["question"],
            record["options"] if isinstance(record["options"], dict)
            else json.loads(record["options"])
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_textual_messages(tokenizer, record: dict) -> str:
    """MCQ: chat prompt with description text only (no image)."""
    opts = record["options"] if isinstance(record["options"], dict) \
           else json.loads(record["options"])
    desc = record.get("description") or record.get("reasoning", "")
    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user",   "content": textual_prompt(desc, record["question"], opts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_visual_messages_oeq(tokenizer, record: dict) -> str:
    """OEQ: chat prompt with embedded image, no options."""
    messages = [
        {"role": "system", "content": OEQ_SYSTEM_PROMPT},
        {"role": "user",   "content": oeq_visual_prompt(record["question"])},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_textual_messages_oeq(tokenizer, record: dict) -> str:
    """OEQ: chat prompt with description text only (no image, no options)."""
    desc = record.get("description") or record.get("reasoning", "")
    messages = [
        {"role": "system", "content": OEQ_SYSTEM_PROMPT},
        {"role": "user",   "content": oeq_textual_prompt(desc, record["question"])},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _extract_answer_and_logprob(output, tokenizer) -> tuple[str, float]:
    """
    MCQ: extract predicted option letter (A/B/C/D) + its log-probability.
    """
    text = output.outputs[0].text.strip().upper()

    answer = "?"
    for ch in text:
        if ch in OPTION_LETTERS:
            answer = ch
            break

    logprob = 0.0
    logprobs_list = output.outputs[0].logprobs
    if logprobs_list:
        first_token_logprobs = logprobs_list[0]
        option_logprobs: dict[str, float] = {}
        for token_id, lp_obj in first_token_logprobs.items():
            decoded = tokenizer.decode([token_id]).strip().upper()
            if decoded in OPTION_LETTERS:
                option_logprobs[decoded] = lp_obj.logprob

        if answer in option_logprobs:
            logprob = option_logprobs[answer]
        elif option_logprobs:
            answer = max(option_logprobs, key=option_logprobs.get)
            logprob = option_logprobs[answer]

    return answer, logprob


def _extract_answer_oeq(output) -> tuple[str, float]:
    """
    OEQ: return generated text + mean token log-probability as confidence proxy.
    """
    text = output.outputs[0].text.strip()
    logprobs_list = output.outputs[0].logprobs
    if logprobs_list:
        per_token = [
            max(lp.values(), key=lambda x: x.logprob).logprob
            for lp in logprobs_list if lp
        ]
        mean_lp = float(np.mean(per_token)) if per_token else 0.0
    else:
        mean_lp = 0.0
    return text, mean_lp


_OEQ_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are", "have"}

def _oeq_match(pred: str, gt: str, threshold: float = 0.25) -> bool:
    """Keyword-overlap accuracy for OEQ: True if ≥threshold of gt key terms appear in pred."""
    gt_words   = {w for w in re.findall(r'\b[a-z]{4,}\b', gt.lower())
                  if w not in _OEQ_STOPWORDS}
    pred_words = {w for w in re.findall(r'\b[a-z]{4,}\b', pred.lower())
                  if w not in _OEQ_STOPWORDS}
    if not gt_words:
        return False
    return len(gt_words & pred_words) / len(gt_words) >= threshold


def run_branch(llm, tokenizer, sampling_params,
               records: list[dict],
               branch: str,  # "visual" or "textual"
               batch_size: int = 8,
               mode: str = "mcq") -> list[dict]:
    """
    Run inference for one branch. Returns list of
    {item_id, answer, logprob, ground_truth}.
    mode="mcq" uses A/B/C/D logprob extraction; mode="oeq" uses free-text generation.
    """
    from PIL import Image as PILImage

    results = []
    total   = len(records)

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        prompts = []

        for record in batch:
            if branch == "visual":
                if mode == "oeq":
                    text = _make_visual_messages_oeq(tokenizer, record)
                else:
                    text = _make_visual_messages(tokenizer, record)
                img  = PILImage.open(io.BytesIO(record["_image_bytes"])).convert("RGB")
                prompts.append({"prompt": text, "multi_modal_data": {"image": img}})
            else:
                if mode == "oeq":
                    text = _make_textual_messages_oeq(tokenizer, record)
                else:
                    text = _make_textual_messages(tokenizer, record)
                prompts.append({"prompt": text})

        outputs = llm.generate(prompts, sampling_params)

        for record, output in zip(batch, outputs):
            if mode == "oeq":
                ans, lp = _extract_answer_oeq(output)
                gt = str(record.get("answer", ""))
            else:
                ans, lp = _extract_answer_and_logprob(output, tokenizer)
                gt = str(record.get("answer", "")).strip().upper()

            results.append({
                "item_id":      record.get("item_id", -1),
                "answer":       ans,
                "logprob":      lp,
                "ground_truth": gt,
            })

        done = min(start + batch_size, total)
        print(f"  [{branch}] {done}/{total}")

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(visual_results: list[dict],
                    textual_results: list[dict],
                    mode: str = "mcq") -> dict:
    assert len(visual_results) == len(textual_results)
    n = len(visual_results)

    vis_correct  = 0
    text_correct = 0
    flip_count   = 0
    cd_all, cd_conflict, cd_no_conflict = [], [], []

    per_sample = []

    for v, t in zip(visual_results, textual_results):
        gt        = v["ground_truth"]
        vis_ans   = v["answer"]
        text_ans  = t["answer"]
        vis_lp    = v["logprob"]
        text_lp   = t["logprob"]

        if mode == "oeq":
            vis_ok  = _oeq_match(vis_ans,  gt)
            text_ok = _oeq_match(text_ans, gt)
            # Flip: answers are semantically divergent (low mutual overlap)
            flipped = not _oeq_match(vis_ans, text_ans, threshold=0.30)
        else:
            vis_ok  = vis_ans  == gt
            text_ok = text_ans == gt
            flipped = vis_ans  != text_ans

        vis_correct  += int(vis_ok)
        text_correct += int(text_ok)
        flip_count   += int(flipped)

        # CD score: |log(p_visual / p_text)| = |vis_lp - text_lp|
        # For OEQ this is |mean_seq_logprob_visual - mean_seq_logprob_textual|
        cd = abs(vis_lp - text_lp)
        cd_all.append(cd)
        (cd_conflict if flipped else cd_no_conflict).append(cd)

        per_sample.append({
            "item_id":      v["item_id"],
            "ground_truth": gt,
            "visual_answer":  vis_ans,
            "textual_answer": text_ans,
            "visual_correct":  vis_ok,
            "textual_correct": text_ok,
            "flipped":  flipped,
            "cd_score": round(cd, 4),
            "visual_logprob":  round(vis_lp,  4),
            "textual_logprob": round(text_lp, 4),
        })

    vis_acc  = vis_correct  / n
    text_acc = text_correct / n
    fr       = flip_count   / n
    delta    = abs(vis_acc - text_acc)
    cr       = fr - delta  # Conflict Rate = FR - |ΔAcc|

    return {
        "n_samples":        n,
        "mode":             mode,
        "visual_accuracy":  round(vis_acc,  4),
        "textual_accuracy": round(text_acc, 4),
        "delta_acc":        round(delta,    4),
        "flip_rate":        round(fr,       4),
        "conflict_rate":    round(cr,       4),
        "cd_mean":              round(float(np.mean(cd_all)),          4),
        "cd_mean_conflict":     round(float(np.mean(cd_conflict)),     4) if cd_conflict     else None,
        "cd_mean_no_conflict":  round(float(np.mean(cd_no_conflict)),  4) if cd_no_conflict  else None,
        "per_sample": per_sample,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_summary(metrics_by_model: dict, output_dir: str):
    """Bar chart comparing key metrics across models."""
    models  = list(metrics_by_model.keys())
    keys    = ["visual_accuracy", "textual_accuracy", "flip_rate", "conflict_rate"]
    labels  = ["Visual Acc", "Textual Acc", "Flip Rate", "Conflict Rate"]
    colors  = ["steelblue", "darkorange", "tomato", "mediumpurple"]

    x     = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.5 + 2), 5))

    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals   = [metrics_by_model[m].get(key, 0) for m in models]
        offset = (i - (len(keys) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate / Accuracy")
    ax.set_title("Cross-Modality Knowledge Conflict — Agricultural (AgriCoT)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(output_dir, "conflict_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cd_distribution(metrics: dict, model_label: str, output_dir: str):
    """Histogram of CD scores split by conflict vs non-conflict."""
    per = metrics["per_sample"]
    cd_conflict    = [s["cd_score"] for s in per if s["flipped"]]
    cd_no_conflict = [s["cd_score"] for s in per if not s["flipped"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(max(cd_conflict or [0]), max(cd_no_conflict or [0])) + 0.5, 40)

    if cd_no_conflict:
        ax.hist(cd_no_conflict, bins=bins, alpha=0.6,
                color="steelblue", label=f"No conflict (n={len(cd_no_conflict)})")
    if cd_conflict:
        ax.hist(cd_conflict, bins=bins, alpha=0.6,
                color="tomato",    label=f"Conflict (n={len(cd_conflict)})")

    ax.set_xlabel("|log(P_visual / P_textual)|")
    ax.set_ylabel("Count")
    ax.set_title(f"CD Score Distribution — {model_label}")
    ax.legend()
    plt.tight_layout()

    safe  = model_label.replace("/", "_").replace(" ", "_")
    path  = os.path.join(output_dir, f"cd_distribution_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_answer_heatmap(metrics: dict, model_label: str, output_dir: str):
    """Confusion-style heatmap: visual_answer vs textual_answer (MCQ only)."""
    if metrics.get("mode") == "oeq":
        # Free-text answers can't be binned into a small letter matrix — skip
        return

    opts    = OPTION_LETTERS + ["?"]
    matrix  = defaultdict(int)
    for s in metrics["per_sample"]:
        matrix[(s["visual_answer"], s["textual_answer"])] += 1

    data = np.array([[matrix[(v, t)] for t in opts] for v in opts])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(opts))); ax.set_xticklabels(opts)
    ax.set_yticks(range(len(opts))); ax.set_yticklabels(opts)
    ax.set_xlabel("Textual answer");  ax.set_ylabel("Visual answer")
    ax.set_title(f"Visual vs Textual Answers — {model_label}")
    plt.colorbar(im, ax=ax)
    for i, v in enumerate(opts):
        for j, t in enumerate(opts):
            val = data[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if val > data.max() * 0.6 else "black")
    plt.tight_layout()

    safe = model_label.replace("/", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"answer_heatmap_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name: str, records: list[dict], args, mode: str = "mcq") -> dict:
    llm, tokenizer, sampling_params = load_vllm_model(
        model_name, gpu_mem=args.gpu_mem, max_model_len=args.max_model_len, mode=mode
    )

    print(f"\nRunning visual branch ({mode}) ...")
    visual_results = run_branch(llm, tokenizer, sampling_params,
                                records, "visual", args.batch_size, mode=mode)

    print(f"\nRunning textual branch ({mode}) ...")
    textual_results = run_branch(llm, tokenizer, sampling_params,
                                 records, "textual", args.batch_size, mode=mode)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return compute_metrics(visual_results, textual_results, mode=mode)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-modality knowledge conflict test on AgriCoT"
    )
    parser.add_argument("--model",          type=str, default="google/gemma-4-E4B-it",
        help="Fine-tuned or base Gemma-4 model (HF ID or local path)")
    parser.add_argument("--baseline_model", type=str, default=None,
        help="Optional second model to compare (e.g. a fine-tuned checkpoint)")
    parser.add_argument("--output_dir",     type=str, default="conflict_results")
    parser.add_argument("--n_samples",      type=int, default=None)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--gpu_mem",        type=float, default=0.90)
    parser.add_argument("--max_model_len",  type=int, default=4096)
    parser.add_argument("--source",         type=str, default="agricot",
        choices=["agricot", "idtools"],
        help="Dataset source: 'agricot' (default) or 'idtools'")
    parser.add_argument("--idtools_jsonl",  type=str, default="idtools_oeq.jsonl",
        help="Path to scrape_idtools_with_images.py output (used when --source idtools)")
    parser.add_argument("--zip",            type=str, default=None,
        help="Force specific ZIP file, e.g. AgriCoT.zip (agricot source only)")
    parser.add_argument("--json_hint",      type=str, default=None)

    parser.add_argument("--plot_only",      action="store_true",
        help="Skip inference, re-plot from saved results JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "conflict_results.json")

    if args.plot_only:
        with open(results_path) as f:
            all_metrics = json.load(f)
        print("Loaded saved results — re-plotting ...")
    else:
        if args.source == "idtools":
            records = load_idtools(args.idtools_jsonl, n_samples=args.n_samples)
            mode = "oeq"
        else:
            records = load_agricot(args.n_samples,
                                   zip_file=args.zip,
                                   json_hint=args.json_hint)
            mode = "mcq"
        print(f"\nRunning on {len(records)} samples  [mode={mode}]\n")

        all_metrics: dict = {}

        for label, model_name in [
            ("finetuned", args.model),
            ("baseline",  args.baseline_model),
        ]:
            if not model_name:
                continue
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}  ({label})")
            print(f"{'='*60}")
            metrics = run_model(model_name, records, args, mode=mode)
            all_metrics[label] = metrics
            print(f"\n  Visual Acc  : {metrics['visual_accuracy']:.4f}")
            print(f"  Textual Acc : {metrics['textual_accuracy']:.4f}")
            print(f"  Flip Rate   : {metrics['flip_rate']:.4f}")
            print(f"  Conflict Rate: {metrics['conflict_rate']:.4f}")
            print(f"  CD Mean     : {metrics['cd_mean']:.4f}")

        with open(results_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved: {results_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    # Summary bar chart across models
    summary_input = {k: v for k, v in all_metrics.items() if "per_sample" in v}
    if len(summary_input) > 0:
        plot_summary(summary_input, args.output_dir)

    for label, metrics in all_metrics.items():
        if "per_sample" not in metrics:
            continue
        plot_cd_distribution(metrics, label, args.output_dir)
        plot_answer_heatmap(metrics, label, args.output_dir)

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
