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
import time
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer as rs_scorer
from vllm import LLM, SamplingParams

rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")

PROMPT_TEMPLATE = (
    "Answer the following agricultural question in one or two sentences.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


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

def run_inference(llm: LLM, prompts: list[str], label: str) -> tuple[list[str], float, int]:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    print(f"\nRunning inference [{label}]...")
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - t_start
    responses = [o.outputs[0].text.strip() for o in outputs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {elapsed:.1f}s | {total_tokens / elapsed:.0f} tok/s")
    return responses, elapsed, total_tokens


def load_llm(model_id: str) -> LLM:
    return LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "contains_match": "Contains Match",
    "exact_match": "Exact Match",
    "rouge1_raw": "ROUGE-1",
    "rouge2_raw": "ROUGE-2",
    "rougeL_raw": "ROUGE-L",
}


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    samples = load_agri_qa(args.n_samples, args.dataset_path)
    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in samples]

    print("\nSample prompt:")
    print(prompts[0])
    print(f"  Expected answer: {samples[0]['answer']}\n")

    results = {
        "dataset": "KisanVaani/agriculture-qa-english-only",
        "n_samples": len(samples),
        "models": {
            "naive": {"model_id": "naive_random", **compute_naive_baseline(samples)},
        },
    }
    all_rows: dict[str, list[dict]] = {}

    # --- Language-only model ---
    if args.lang_model:
        print(f"\n=== Language Model: {args.lang_model} ===")
        llm = load_llm(args.lang_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.lang_model)
        metrics, rows = score(responses, samples)
        results["models"]["lang_model"] = {
            "model_id": args.lang_model,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows["lang_model"] = rows
        del llm

    # --- Second language-only model ---
    if args.lang_model2:
        print(f"\n=== Language Model 2: {args.lang_model2} ===")
        llm = load_llm(args.lang_model2)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.lang_model2)
        metrics, rows = score(responses, samples)
        results["models"]["lang_model2"] = {
            "model_id": args.lang_model2,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows["lang_model2"] = rows
        del llm

    # --- Gemma4 base ---
    print(f"\n=== Gemma4 Base: {args.model} ===")
    llm = load_llm(args.model)
    responses, elapsed, total_tokens = run_inference(llm, prompts, args.model)
    metrics, rows = score(responses, samples)
    results["models"]["gemma4_base"] = {
        "model_id": args.model,
        "total_time_s": round(elapsed, 3),
        "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
        **metrics,
    }
    all_rows["gemma4_base"] = rows
    del llm

    # --- Gemma4 fine-tuned (optional) ---
    if args.finetuned_model:
        print(f"\n=== Gemma4 Fine-tuned: {args.finetuned_model} ===")
        llm = load_llm(args.finetuned_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.finetuned_model)
        metrics, rows = score(responses, samples)
        results["models"]["gemma4_finetuned"] = {
            "model_id": args.finetuned_model,
            "total_time_s": round(elapsed, 3),
            "throughput_tokens_per_s": round(total_tokens / elapsed, 1),
            **metrics,
        }
        all_rows["gemma4_finetuned"] = rows
        del llm

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"  {'Model':<25} {'Contains':>10} {'ExactMatch':>12} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
    print(f"  {'-'*79}")
    for key, m in results["models"].items():
        print(f"  {key:<25} {m['contains_match']:>10.4f} {m['exact_match']:>12.4f} {m['rouge1_raw']:>10.4f} {m['rouge2_raw']:>10.4f} {m['rougeL_raw']:>10.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, rows in all_rows.items():
        csv_path = output_dir / f"text_qa_{key}.csv"
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
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of samples (default: all 22615)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.lang_model and args.lang_model.lower() == "none":
        args.lang_model = None
    if args.lang_model2 and args.lang_model2.lower() == "none":
        args.lang_model2 = None

    main(args)
