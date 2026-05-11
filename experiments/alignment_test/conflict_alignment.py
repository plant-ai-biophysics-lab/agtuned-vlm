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


# ── Task metadata ─────────────────────────────────────────────────────────────

DIMENSION_NAMES: dict[int, str] = {
    1: "Produce Counting",
    2: "Farm Management",
    3: "Crop Species ID",
    4: "Pest and Disease",
    5: "Weed Management",
}

# ── Prompt templates ──────────────────────────────────────────────────────────

# MCQ prompts (AgriCoT)
MCQ_SYSTEM_PROMPT = (
    "You are an expert agricultural scientist. "
    "Answer the following multiple-choice question. "
    "Respond with ONLY the option letter: A, B, C, or D."
)

def visual_prompt(question: str, options: dict) -> str:
    opts = "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))
    return f"<|image|>\n{question}\n{opts}"


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
    return f"<|image|>\n{question}"


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
        1 if any(k in n.lower() for k in ("meta", "sample")) else 2,
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
        if any(_is_qa_record(r) for r in data[:5]) and any(r.get("reasoning") for r in data[:5]):
            print(f"  ✓ Found {len(data):,} QA records in {name}")
            return data

    raise RuntimeError(
        f"No QA JSON found. Files: {json_files}. "
        "Try --json_hint <filename substring>."
    )


def load_agricot_by_task(n_samples: int | None,
                         zip_file: str | None = None,
                         json_hint: str | None = None) -> dict[str, list[dict]]:
    """Load AgriCoT and return records grouped by task name (dimension_id)."""
    candidates = (zip_file,) if zip_file else ("AgriCoT.zip", "AgriCoT_2.zip")
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

    # Group valid MCQ records by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not _is_qa_record(r):
            continue
        if not r.get("options") or not r.get("reasoning"):
            continue
        if not isinstance(r.get("answer"), str):
            continue
        ip = r.get("image_path", "")
        stem = Path(ip).name if ip else ""
        if stem not in image_index:
            continue
        r["_image_bytes"] = image_index[stem]
        r["reasoning"] = clean_reasoning(r["reasoning"])
        r.setdefault("item_id", len(by_task))
        dim_id = r.get("dimension_id")
        task_name = DIMENSION_NAMES.get(dim_id, f"task_{dim_id}")
        by_task[task_name].append(r)

    # Apply n_samples cap per task
    result: dict[str, list[dict]] = {}
    for task_name, task_records in by_task.items():
        capped = task_records[:n_samples] if n_samples else task_records
        result[task_name] = capped
        print(f"  {task_name}: {len(capped):,} samples")

    return result


def load_preprocessed_by_task(preprocessed_json: str,
                               zip_file: str | None,
                               n_samples: int | None) -> dict[str, list[dict]]:
    """Load from preprocess_reasoning.py output, attach image bytes, group by task."""
    with open(preprocessed_json) as f:
        records = json.load(f)

    # Download zip just for images
    candidates = (zip_file,) if zip_file else ("AgriCoT.zip", "AgriCoT_2.zip")
    image_index: dict[str, bytes] = {}
    for filename in candidates:
        try:
            zip_path = hf_hub_download(
                repo_id="wenyb/AgriCoT", filename=filename, repo_type="dataset"
            )
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        image_index[Path(member).name] = zf.read(member)
            break
        except Exception:
            continue

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        stem = Path(r.get("image_path", "")).name
        if stem not in image_index:
            continue
        r["_image_bytes"] = image_index[stem]
        # ensure fallback question fields exist
        r.setdefault("question", r.get("question_visual", ""))
        task = r.get("task") or DIMENSION_NAMES.get(r.get("dimension_id"), "unknown")
        by_task[task].append(r)

    result: dict[str, list[dict]] = {}
    for task_name, task_records in by_task.items():
        capped = task_records[:n_samples] if n_samples else task_records
        result[task_name] = capped
        print(f"  {task_name}: {len(capped):,} samples")

    return result


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


def _opts(record: dict) -> dict:
    o = record["options"]
    return o if isinstance(o, dict) else json.loads(o)


def _make_visual_messages(tokenizer, record: dict) -> str:
    """MCQ visual branch: image + question_visual (mentions 'this image')."""
    q = record.get("question_visual") or record["question"]
    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user",   "content": visual_prompt(q, _opts(record))},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_textual_messages(tokenizer, record: dict) -> str:
    """MCQ textual branch: visual_description + question_textual (no image reference)."""
    opts = _opts(record)
    desc = record.get("visual_description") \
        or record.get("description") \
        or record.get("reasoning", "")
    q    = record.get("question_textual") or record["question"]
    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user",   "content": textual_prompt(desc, q, opts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_visual_messages_oeq(tokenizer, record: dict) -> str:
    """OEQ visual branch: image + question_visual."""
    q = record.get("question_visual") or record["question"]
    messages = [
        {"role": "system", "content": OEQ_SYSTEM_PROMPT},
        {"role": "user",   "content": oeq_visual_prompt(q)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _make_textual_messages_oeq(tokenizer, record: dict) -> str:
    """OEQ textual branch: visual_description + question_textual."""
    desc = record.get("visual_description") \
        or record.get("description") \
        or record.get("reasoning", "")
    q = record.get("question_textual") or record["question"]
    messages = [
        {"role": "system", "content": OEQ_SYSTEM_PROMPT},
        {"role": "user",   "content": oeq_textual_prompt(desc, q)},
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
                "question":     record.get("question", ""),
                "reasoning":    record.get("reasoning", ""),
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
            "item_id":        v["item_id"],
            "question":       v.get("question", ""),
            "reasoning":      v.get("reasoning", ""),
            "ground_truth":   gt,
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
    delta    = text_acc - vis_acc   # ΔAcc = Acc_textual − Acc_visual (signed, per paper eq.5)
    cr       = fr - delta           # CR = FR − ΔAcc

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


def plot_task_summary(task_metrics: dict[str, dict], model_label: str, output_dir: str):
    """Grouped bar chart: one group per task, bars = vis_acc / text_acc / FR / CR."""
    tasks  = list(task_metrics.keys())
    keys   = ["visual_accuracy", "textual_accuracy", "flip_rate", "conflict_rate"]
    labels = ["Visual Acc", "Textual Acc", "Flip Rate", "Conflict Rate"]
    colors = ["steelblue", "darkorange", "tomato", "mediumpurple"]

    x     = np.arange(len(tasks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2.5 + 2), 5))

    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals   = [task_metrics[t].get(key, 0) or 0 for t in tasks]
        offset = (i - (len(keys) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Rate / Accuracy")
    ax.set_title(f"Per-Task Conflict Analysis — {model_label}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    safe = model_label.replace("/", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"task_summary_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_task_table(task_metrics: dict[str, dict], model_label: str):
    """Print a compact summary table to stdout."""
    cols = ["n_samples", "visual_accuracy", "textual_accuracy", "flip_rate", "conflict_rate"]
    header = f"{'Task':<25} {'N':>5} {'VisAcc':>8} {'TxtAcc':>8} {'FR':>8} {'CR':>8}"
    print(f"\n  {model_label}")
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for task, m in task_metrics.items():
        n   = m.get("n_samples", 0)
        va  = m.get("visual_accuracy",  0) or 0
        ta  = m.get("textual_accuracy", 0) or 0
        fr  = m.get("flip_rate",        0) or 0
        cr  = m.get("conflict_rate",    0) or 0
        print(f"  {task:<25} {n:>5} {va:>8.4f} {ta:>8.4f} {fr:>8.4f} {cr:>8.4f}")
    print("  " + "-" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_samples(per_sample: list[dict], task_name: str, n: int = 3):
    """Print n sample records per task to aid reasoning-leakage inspection."""
    samples = per_sample[:n]
    print(f"\n  ── Sample outputs: {task_name} ──")
    for s in samples:
        flip_tag = " [FLIP]" if s["flipped"] else ""
        print(f"  item {s['item_id']}{flip_tag}")
        print(f"    Q        : {s['question']}")
        print(f"    Reasoning: {s['reasoning'][:300].strip()!r}")
        print(f"    GT={s['ground_truth']}  Visual={s['visual_answer']}(lp={s['visual_logprob']:.3f})"
              f"  Textual={s['textual_answer']}(lp={s['textual_logprob']:.3f})")


def run_model(model_name: str,
              tasks: dict[str, list[dict]],
              args,
              mode: str = "mcq") -> dict[str, dict]:
    """Load model once, run all tasks, return {task_name: metrics}."""
    llm, tokenizer, sampling_params = load_vllm_model(
        model_name, gpu_mem=args.gpu_mem, max_model_len=args.max_model_len, mode=mode
    )

    task_metrics: dict[str, dict] = {}
    for task_name, records in tasks.items():
        if not records:
            continue
        print(f"\n  ── Task: {task_name} ({len(records)} samples) ──")

        print(f"  Running visual branch ...")
        visual_results = run_branch(llm, tokenizer, sampling_params,
                                    records, "visual", args.batch_size, mode=mode)

        print(f"  Running textual branch ...")
        textual_results = run_branch(llm, tokenizer, sampling_params,
                                     records, "textual", args.batch_size, mode=mode)

        metrics = compute_metrics(visual_results, textual_results, mode=mode)
        metrics["task"] = task_name
        task_metrics[task_name] = metrics

        _print_samples(metrics["per_sample"], task_name, n=3)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return task_metrics


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
    parser.add_argument("--json_hint",        type=str, default=None)
    parser.add_argument("--preprocessed_json", type=str, default=None,
        help="Path to output of preprocess_reasoning.py (skips raw AgriCoT loading)")

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
            tasks   = {"idtools": records}
            mode    = "oeq"
        elif args.preprocessed_json:
            print(f"Loading preprocessed records from {args.preprocessed_json} ...")
            tasks = load_preprocessed_by_task(args.preprocessed_json,
                                               zip_file=args.zip,
                                               n_samples=args.n_samples)
            mode  = "mcq"
        else:
            tasks = load_agricot_by_task(args.n_samples,
                                         zip_file=args.zip,
                                         json_hint=args.json_hint)
            mode  = "mcq"

        total = sum(len(v) for v in tasks.values())
        print(f"\nRunning on {total} samples across {len(tasks)} tasks  [mode={mode}]\n")

        all_metrics: dict = {}

        def _model_label(model_path: str) -> str:
            name = Path(model_path).name if "/" not in model_path else model_path.split("/")[-1]
            return name

        models_to_run = [(_model_label(args.model), args.model)]
        if args.baseline_model:
            models_to_run.append((_model_label(args.baseline_model), args.baseline_model))

        for label, model_name in models_to_run:
            if not model_name:
                continue
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}  ({label})")
            print(f"{'='*60}")
            task_metrics = run_model(model_name, tasks, args, mode=mode)
            all_metrics[label] = task_metrics
            print_task_table(task_metrics, label)

        with open(results_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved: {results_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    for label, task_metrics in all_metrics.items():
        if not isinstance(task_metrics, dict):
            continue
        # Per-task summary bar chart
        plot_task_summary(task_metrics, label, args.output_dir)
        # Per-task CD distribution and answer heatmap
        for task_name, metrics in task_metrics.items():
            if "per_sample" not in metrics:
                continue
            plot_cd_distribution(metrics, f"{label}_{task_name}", args.output_dir)
            plot_answer_heatmap(metrics, f"{label}_{task_name}", args.output_dir)

    # Cross-model summary (if multiple models)
    model_level = {label: list(tm.values())[0]
                   for label, tm in all_metrics.items()
                   if isinstance(tm, dict) and tm}
    if len(model_level) > 1:
        plot_summary(model_level, args.output_dir)

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
