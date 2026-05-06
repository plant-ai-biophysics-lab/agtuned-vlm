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
  --lang_model     Language-only model, e.g. gemma-3-4b-it (on by default)
  --model          Gemma4 base/multimodal (on by default)
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
from datasets import load_dataset
from vllm import LLM, SamplingParams

rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")

PROMPT_TEMPLATE = (
    "Answer the following agricultural question in one or two sentences.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


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

    rows = [
        {
            "question": s["question"],
            "ground_truth": s["answer"],
            "response": pred,
            "contains_match": int(s["answer"].lower() in pred.lower()),
        }
        for s, pred in zip(samples, predictions)
    ]

    return metrics, rows


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


def main(args):
    samples = load_agri_qa(args.n_samples, args.dataset_path)

    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in samples]

    print("\nSample prompt:")
    print(prompts[0])
    print(f"  Expected answer: {samples[0]['answer']}\n")

    results = {
        "dataset": "KisanVaani/agriculture-qa-english-only",
        "n_samples": len(samples),
        "models": {},
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
                f, fieldnames=["question", "ground_truth", "response", "contains_match"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")

    metrics_path = output_dir / "text_qa_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="Gemma4 base model (HF ID or local path)")
    parser.add_argument("--finetuned_model", type=str, default=None,
                        help="Fine-tuned Gemma4 (optional)")
    parser.add_argument("--lang_model", type=str, default="google/gemma-3-4b-it",
                        help="Language-only baseline. Pass 'none' to skip.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset saved with download_data.py (for offline cluster use)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of samples (default: all 22615)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.lang_model and args.lang_model.lower() == "none":
        args.lang_model = None

    main(args)
