"""
H2: Text-only QA — Agricultural Language Knowledge
===================================================
Tests whether the model can answer agricultural questions from text alone
(no images) using the KisanVaani agriculture QA benchmark.

If text-only performance is weak → the language model lacks agricultural
domain knowledge independently of any vision issues → H2 confirmed.
If text-only performance is strong → the bottleneck is vision, not language.

Dataset: KisanVaani/agriculture-qa-english-only
  Fields: question (str), answers (str)
  Split: train (22615 rows)

Backbones compared:
  --lang_model      Language-only model, e.g. gemma-3-4b-it (on by default)
  --lang_model2     Second language-only model for comparison (optional)
  --model           Gemma4 base/multimodal (on by default)
  --finetuned_model Fine-tuned Gemma4 (optional)

Usage:
    python text_qa.py
    python text_qa.py --model google/gemma-4-E2B-it --n_samples 500
    python text_qa.py --model google/gemma-4-E2B-it \\
                      --finetuned_model outputs/vllm_model \\
                      --dataset_path data/agri_qa
"""

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from rouge_score import rouge_scorer as rs_scorer
from sklearn.metrics import accuracy_score, f1_score
from vllm import LLM, SamplingParams

rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")

PROMPT_TEMPLATE = (
    "Answer the following agricultural question in one or two sentences.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Used when evaluating agml_scraper output — questions are already descriptive,
# so no preamble is needed; just pass the question directly.
DESCRIPTIVE_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer:"
)


# ── Classification helpers (shared with agml_inference.py) ───────────────────

def load_datasets_config(config_path: Path) -> dict[str, dict]:
    """Returns {agml_name: {task_type, classes (cleaned), ...}} from datasets.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    result = {}
    for d in cfg["datasets"]:
        result[d["agml_name"]] = {
            **d,
            "classes": clean_class_list(d.get("classes", [])),
        }
    return result


def clean_class_list(raw_classes: list[str]) -> list[str]:
    cleaned = []
    for c in raw_classes:
        if "___" in c:
            cleaned.append(c.split("___", 1)[1].replace("_", " ").strip())
        else:
            cleaned.append(c.replace("_", " ").strip())
    return cleaned


def resolve_class(response: str, classes: list[str]) -> str | None:
    """Case-insensitive exact match, then single-match substring fallback."""
    resp = response.lower().strip()
    for cls in classes:
        if cls.lower() == resp:
            return cls
    matches = [cls for cls in classes if cls.lower() in resp]
    return matches[0] if len(matches) == 1 else None


def score_classification(
    samples: list[dict],
    predictions: list[str],
    datasets_cfg: dict[str, list[str]],
) -> dict:
    """
    Compute accuracy, F1-macro, F1-weighted for classification JSONL.
    Groups by agml_dataset, resolves each prediction against known classes,
    then aggregates. Unresolvable predictions are excluded from metrics.
    """
    by_dataset: dict[str, list] = defaultdict(list)
    for sample, pred in zip(samples, predictions):
        agml_name = sample.get("agml_dataset")
        if not agml_name or agml_name not in datasets_cfg:
            continue
        classes  = datasets_cfg[agml_name]["classes"]
        resolved = resolve_class(pred, classes)
        by_dataset[agml_name].append({
            "class_label":    sample["answer"],
            "resolved_class": resolved or "",
            "resolvable":     resolved is not None,
        })

    per_dataset = {}
    all_refs, all_preds = [], []

    for ds_name, rows in by_dataset.items():
        resolvable = [r for r in rows if r["resolvable"]]
        if not resolvable:
            per_dataset[ds_name] = {
                "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
                "n_total": len(rows), "n_resolvable": 0,
            }
            continue
        refs   = [r["class_label"]    for r in resolvable]
        preds  = [r["resolved_class"] for r in resolvable]
        labels = sorted(set(refs) | set(preds))
        per_dataset[ds_name] = {
            "accuracy":     round(accuracy_score(refs, preds), 4),
            "f1_macro":     round(f1_score(refs, preds, average="macro",
                                           labels=labels, zero_division=0), 4),
            "f1_weighted":  round(f1_score(refs, preds, average="weighted",
                                           labels=labels, zero_division=0), 4),
            "n_total":      len(rows),
            "n_resolvable": len(resolvable),
        }
        all_refs.extend(refs)
        all_preds.extend(preds)

    overall = {}
    if all_refs:
        labels = sorted(set(all_refs) | set(all_preds))
        overall = {
            "accuracy":     round(accuracy_score(all_refs, all_preds), 4),
            "f1_macro":     round(f1_score(all_refs, all_preds, average="macro",
                                           labels=labels, zero_division=0), 4),
            "f1_weighted":  round(f1_score(all_refs, all_preds, average="weighted",
                                           labels=labels, zero_division=0), 4),
            "n_total":      len(all_refs),
            "n_resolvable": len(all_refs),
        }

    return {"overall": overall, "per_dataset": per_dataset}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_agri_qa(n_samples: int | None, dataset_path: str | None = None) -> list[dict]:
    if dataset_path:
        from datasets import load_from_disk
        print(f"Loading dataset from disk: {dataset_path}")
        ds = load_from_disk(dataset_path)
    else:
        print("Loading KisanVaani/agriculture-qa-english-only...")
        ds = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
    if n_samples:
        ds = ds.select(range(n_samples))
    samples = [{"question": row["question"], "answer": row["answers"]} for row in ds]
    print(f"  {len(samples)} samples")
    return samples


def load_idtools_qa(jsonl_path: str, n_samples: int | None = None) -> list[dict]:
    """
    Load IDtools scraper output (messages format) as flat question/answer pairs.
    Each JSONL line: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], ...}
    """
    print(f"Loading IDtools dataset: {jsonl_path}")
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = record.get("messages", [])
            if len(msgs) < 2:
                continue
            question = msgs[0].get("content", "").strip()
            answer   = msgs[1].get("content", "").strip()
            if question and answer:
                meta = record.get("metadata", {})
                samples.append({
                    "question":    question,
                    "answer":      answer,
                    "agml_dataset": meta.get("agml_dataset"),
                })
    if n_samples:
        samples = samples[:n_samples]
    print(f"  {len(samples)} samples")
    return samples


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_sample_rougeL(predictions: list[str], references: list[str]) -> list[float]:
    scorer = rs_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    em_scores = exact_match.compute(
        predictions=[p.lower() for p in predictions],
        references=[r.lower() for r in references],
    )
    contains_match = sum(
        r.lower() in p.lower() for p, r in zip(predictions, references)
    ) / len(predictions)

    return {
        "contains_match": round(contains_match, 4),
        "exact_match": round(em_scores["exact_match"], 4),
        "rouge1_raw": round(rouge_scores["rouge1"], 4),
        "rouge2_raw": round(rouge_scores["rouge2"], 4),
        "rougeL_raw": round(rouge_scores["rougeL"], 4),
    }


def score(responses: list[str], samples: list[dict]) -> tuple[dict, list[dict]]:
    predictions = [r.strip() for r in responses]
    references = [s["answer"] for s in samples]

    metrics = compute_metrics(predictions, references)
    rougeL_per_sample = per_sample_rougeL(predictions, references)

    rows = [
        {
            "question": s["question"],
            "ground_truth": s["answer"],
            "response": pred,
            "contains_match": int(s["answer"].lower() in pred.lower()),
            "rougeL": round(rl, 4),
        }
        for s, pred, rl in zip(samples, predictions, rougeL_per_sample)
    ]

    return metrics, rows


# ── Naive baseline ────────────────────────────────────────────────────────────

def compute_naive_baseline(samples: list[dict]) -> dict:
    """Randomly shuffle answers across questions — sets the floor for all metrics."""
    references = [s["answer"] for s in samples]
    rng = np.random.default_rng(42)
    shuffled = references.copy()
    rng.shuffle(shuffled)
    return compute_metrics(shuffled, references)


# ── Inference ─────────────────────────────────────────────────────────────────

_SAMPLING = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    repetition_penalty=1.1,
    stop=["Question:", "\n\nQ:", "\n\nAnswer:", "[Answer"],
)

_REPEAT_RE  = re.compile(r'(\b\w+\b)(\s+\1){4,}', re.IGNORECASE)
_THINK_RE   = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def _clean_response(text: str) -> str:
    """Strip degenerate and thinking outputs from instruct model responses."""
    # Remove complete <think>...</think> blocks
    text = _THINK_RE.sub('', text).strip()
    # Remove incomplete <think> block (no closing tag — truncate at opening)
    think_start = text.lower().find('<think>')
    if think_start != -1:
        text = text[:think_start].strip()
    # Truncate at recursive bracket patterns like "[Answer: [Answer: ..."
    bracket = text.find("[Answer")
    if bracket != -1:
        text = text[:bracket].strip()
    # Truncate at first run of 5+ repeated words
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text


def _apply_chat_template(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
    # Try with enable_thinking=False (Qwen3 and similar)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    # Fallback: add /no_think via system prompt for models that use that convention
    try:
        msgs_with_sys = [
            {"role": "system", "content": "/no_think"},
            *messages,
        ]
        return tokenizer.apply_chat_template(
            msgs_with_sys, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def run_inference(llm: LLM, prompts: list[str], label: str,
                  tokenizer=None) -> tuple[list[str], float, int]:
    # Use chat template for instruct models when tokenizer is available
    if tokenizer is not None:
        prompts = [_apply_chat_template(tokenizer, p) for p in prompts]

    print(f"\nRunning inference [{label}]...")
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=_SAMPLING)
    elapsed = time.perf_counter() - t_start
    responses = [_clean_response(o.outputs[0].text) for o in outputs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {elapsed:.1f}s | {total_tokens / elapsed:.0f} tok/s")
    return responses, elapsed, total_tokens


def load_llm(model_id: str):
    from transformers import AutoTokenizer
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return llm, tokenizer


# ── Plots ─────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "contains_match": "Contains Match",
    "exact_match": "Exact Match",
    "rouge1_raw": "ROUGE-1",
    "rouge2_raw": "ROUGE-2",
    "rougeL_raw": "ROUGE-L",
}


def _safe_name(model_id: str) -> str:
    """Filesystem-safe version of a model ID for filenames."""
    return model_id.replace("/", "_").replace(" ", "_")


def plot_metrics_comparison(results: dict, output_dir: Path):
    models = list(results["models"].keys())
    metrics = list(METRIC_LABELS.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(models):
        vals = [results["models"][key][m] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        is_naive = key == "naive"
        bars = ax.bar(
            x + offset, vals, width, label=key,
            color="#aaaaaa" if is_naive else None,
            hatch="///" if is_naive else None,
            edgecolor="white" if is_naive else None,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Text QA — Model Comparison")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rougeL_distribution(all_rows: dict[str, list[dict]], output_dir: Path):
    models = list(all_rows.keys())
    data = [[row["rougeL"] for row in all_rows[k]] for k in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    parts = ax.violinplot(data, positions=range(len(models)), showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROUGE-L per sample")
    ax.set_title("ROUGE-L Score Distribution per Model")

    for i, d in enumerate(data):
        ax.text(i, np.median(d) + 0.02, f"med={np.median(d):.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "rougeL_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_contains_match_overlap(all_rows: dict[str, list[dict]], output_dir: Path):
    """Bar chart showing per-model contains_match rate, with a stacked breakdown
    of questions answered correctly by all models vs only some."""
    models = list(all_rows.keys())
    n = len(next(iter(all_rows.values())))

    # Per-question correct count across models
    per_q = np.array([[row["contains_match"] for row in all_rows[k]] for k in models])
    total_correct = per_q.sum(axis=0)  # shape (n,)

    all_correct = (total_correct == len(models)).sum()
    none_correct = (total_correct == 0).sum()
    some_correct = n - all_correct - none_correct

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-model contains_match rate
    rates = [np.mean(per_q[i]) for i in range(len(models))]
    bars = axes[0].bar(models, rates, color=plt.cm.tab10(np.linspace(0, 0.5, len(models))))
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{rate:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Contains Match Rate")
    axes[0].set_title("Contains Match Rate per Model")

    # Right: question-level overlap breakdown
    counts = [all_correct, some_correct, none_correct]
    labels = ["All models correct", "Some models correct", "No model correct"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[1].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Question-level Agreement Across Models")

    plt.tight_layout()
    path = output_dir / "contains_match_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_response_length(all_rows: dict[str, list[dict]], output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, rows in all_rows.items():
        lengths = [len(row["response"].split()) for row in rows]
        ax.hist(lengths, bins=30, alpha=0.6, label=key)
    ax.set_xlabel("Response length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Response Length Distribution")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "response_length.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cls_summary_per_dataset(
    all_cls_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: Accuracy / F1-macro / F1-weighted per AgML dataset, one group per model."""
    # Collect all dataset names across all models
    all_datasets = []
    for cls_results in all_cls_results.values():
        for ds in cls_results["per_dataset"]:
            if ds not in all_datasets:
                all_datasets.append(ds)
    if not all_datasets:
        return

    models  = list(all_cls_results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(models)))
    x       = np.arange(len(all_datasets))
    width   = 0.8 / (len(models) * len(metrics))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(10, len(all_datasets) * 0.8), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            per_ds = all_cls_results[model_key]["per_dataset"]
            vals   = [per_ds.get(ds, {}).get(metric, 0.0) for ds in all_datasets]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_classification", "").replace("_", " ") for d in all_datasets],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Text Classification — Per Dataset", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cls_summary_per_task(
    all_cls_results: dict[str, dict],
    datasets_cfg_full: dict[str, dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: Accuracy / F1-macro / F1-weighted per task_type, one group per model."""
    # Map dataset → task_type
    ds_to_task = {name: cfg["task_type"] for name, cfg in datasets_cfg_full.items()}
    task_types = sorted({t for t in ds_to_task.values()})
    if not task_types:
        return

    models  = list(all_cls_results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(models)))
    x       = np.arange(len(task_types))
    width   = 0.8 / (len(models) * len(metrics))

    # Aggregate per task_type per model
    task_scores: dict[str, dict[str, dict[str, list]]] = {
        model_key: {tt: {m: [] for m in metrics} for tt in task_types}
        for model_key in models
    }
    for model_key, cls_results in all_cls_results.items():
        for ds, m in cls_results["per_dataset"].items():
            tt = ds_to_task.get(ds)
            if tt:
                for metric in metrics:
                    task_scores[model_key][tt][metric].append(m.get(metric, 0.0))

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(8, len(task_types) * 1.5), 5), sharey=True)
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for i, (model_key, color) in enumerate(zip(models, colors)):
            vals = [
                float(np.mean(task_scores[model_key][tt][metric]))
                if task_scores[model_key][tt][metric] else 0.0
                for tt in task_types
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model_key, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([tt.replace("_", " ") for tt in task_types], rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score" if mi == 0 else "")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Text Classification — Per Task Type", fontsize=11)
    plt.tight_layout()
    path = output_dir / "cls_summary_per_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    datasets_cfg = None
    if args.datasets_config:
        datasets_cfg = load_datasets_config(Path(args.datasets_config))
        print(f"Datasets config: {args.datasets_config} ({len(datasets_cfg)} datasets)")

    if args.idtools_jsonl:
        samples = load_idtools_qa(args.idtools_jsonl, args.n_samples)
        dataset_label = f"IDtools:{args.idtools_jsonl}"
        prompt_template = DESCRIPTIVE_PROMPT_TEMPLATE
    else:
        samples = load_agri_qa(args.n_samples, args.dataset_path)
        dataset_label = args.dataset_path or "KisanVaani/agriculture-qa-english-only"
        prompt_template = PROMPT_TEMPLATE

    prompts = [prompt_template.format(question=s["question"]) for s in samples]

    print(f"\nPrompt template: {'descriptive' if args.idtools_jsonl else 'standard'}")
    print("\nSample prompt:")
    print(prompts[0])
    print(f"  Expected answer: {samples[0]['answer']}\n")

    results = {
        "dataset": dataset_label,
        "n_samples": len(samples),
        "models": {
            "naive": {"model_id": "naive_random", **compute_naive_baseline(samples)},
        },
    }
    all_rows: dict[str, list[dict]] = {}

    # --- Language-only model ---
    if args.lang_model:
        print(f"\n=== Language Model: {args.lang_model} ===")
        llm, tokenizer = load_llm(args.lang_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.lang_model, tokenizer)
        metrics, rows = score(responses, samples)
        results["models"][args.lang_model] = {
            "model_id": args.lang_model,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows[args.lang_model] = rows
        del llm, tokenizer

    # --- Second language-only model ---
    if args.lang_model2:
        print(f"\n=== Language Model 2: {args.lang_model2} ===")
        llm, tokenizer = load_llm(args.lang_model2)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.lang_model2, tokenizer)
        metrics, rows = score(responses, samples)
        results["models"][args.lang_model2] = {
            "model_id": args.lang_model2,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows[args.lang_model2] = rows
        del llm, tokenizer

    # --- Gemma4 base ---
    print(f"\n=== Gemma4 Base: {args.model} ===")
    llm, tokenizer = load_llm(args.model)
    responses, elapsed, total_tokens = run_inference(llm, prompts, args.model, tokenizer)
    metrics, rows = score(responses, samples)
    results["models"][args.model] = {
        "model_id": args.model,
        "total_time_s": round(elapsed, 3),
        "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
        **metrics,
    }
    all_rows[args.model] = rows
    del llm, tokenizer

    # --- Gemma4 fine-tuned (optional) ---
    if args.finetuned_model:
        print(f"\n=== Gemma4 Fine-tuned: {args.finetuned_model} ===")
        llm, tokenizer = load_llm(args.finetuned_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.finetuned_model, tokenizer)
        metrics, rows = score(responses, samples)
        results["models"][args.finetuned_model] = {
            "model_id": args.finetuned_model,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows[args.finetuned_model] = rows
        del llm, tokenizer

    # --- Classification metrics (when datasets_config provided) ---
    if datasets_cfg and args.idtools_jsonl:
        print("\n=== Classification Metrics ===")
        for key, rows in all_rows.items():
            preds = [row["response"] for row in rows]
            cls_results = score_classification(samples, preds, datasets_cfg)
            results["models"][key]["classification"] = cls_results
            ov = cls_results.get("overall", {})
            print(f"\n  {key}")
            print(f"    Overall  Acc={ov.get('accuracy', 0):.3f}  "
                  f"F1-macro={ov.get('f1_macro', 0):.3f}  "
                  f"F1-weighted={ov.get('f1_weighted', 0):.3f}  "
                  f"({ov.get('n_resolvable', 0)}/{ov.get('n_total', 0)} resolvable)")
            print(f"  {'Dataset':<45} {'Acc':>6} {'F1-mac':>8} {'F1-wt':>7} {'Res/Tot':>10}")
            print(f"  {'-'*80}")
            for ds, m in cls_results["per_dataset"].items():
                print(f"  {ds:<45} {m['accuracy']:>6.3f} {m['f1_macro']:>8.3f} "
                      f"{m['f1_weighted']:>7.3f} "
                      f"{m['n_resolvable']:>4}/{m['n_total']:<4}")

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"  {'Model':<40} {'Contains':>10} {'ExactMatch':>12} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
    print(f"  {'-'*94}")
    for key, m in results["models"].items():
        print(f"  {key:<40} {m['contains_match']:>10.4f} {m['exact_match']:>12.4f} {m['rouge1_raw']:>10.4f} {m['rouge2_raw']:>10.4f} {m['rougeL_raw']:>10.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, rows in all_rows.items():
        csv_path = output_dir / f"text_qa_{_safe_name(key)}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["question", "ground_truth", "response", "contains_match", "rougeL"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")

    metrics_path = output_dir / "text_qa_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # --- Plots ---
    print("\n=== Generating plots ===")
    plot_metrics_comparison(results, output_dir)
    plot_rougeL_distribution(all_rows, output_dir)
    plot_contains_match_overlap(all_rows, output_dir)
    plot_response_length(all_rows, output_dir)

    if datasets_cfg and args.idtools_jsonl:
        all_cls_results = {
            key: results["models"][key]["classification"]
            for key in all_rows
            if "classification" in results["models"].get(key, {})
        }
        if all_cls_results:
            plot_cls_summary_per_dataset(all_cls_results, output_dir)
            plot_cls_summary_per_task(all_cls_results, datasets_cfg, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="Gemma4 base model (HF ID or local path)")
    parser.add_argument("--finetuned_model", type=str, default=None,
                        help="Fine-tuned Gemma4 (optional)")
    parser.add_argument("--lang_model", type=str, default="google/gemma-3-4b-it",
                        help="Language-only baseline. Pass 'none' to skip.")
    parser.add_argument("--lang_model2", type=str, default=None,
                        help="Second language-only model for comparison (e.g. meta-llama/Llama-3.2-3B-Instruct).")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset saved with download_data.py (for offline cluster use)")
    parser.add_argument("--idtools_jsonl", type=str, default=None,
                        help="Path to scrape_idtools.py JSONL output — uses IDtools visual-descriptor QA instead of KisanVaani")
    parser.add_argument("--datasets_config", type=str, default=None,
                        help="Path to datasets.yaml — enables accuracy/F1 classification metrics when used with --idtools_jsonl")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of samples (default: all 22615)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.lang_model and args.lang_model.lower() == "none":
        args.lang_model = None
    if args.lang_model2 and args.lang_model2.lower() == "none":
        args.lang_model2 = None

    main(args)
