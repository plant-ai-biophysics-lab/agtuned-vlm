"""
H2: Text-only QA — Agricultural Language Knowledge
===================================================
Tests whether the model can answer agricultural questions from text alone
(no images) using the AgroCoT benchmark.

If text-only performance is weak → the language model lacks agricultural
domain knowledge independently of any vision issues → H2 confirmed.
If text-only performance is strong → the bottleneck is vision, not language.

Dataset: AgroCoT (huggingface: AgriCulturalQA/AgroCoT)

Usage:
    python text_qa.py --model google/gemma-4-E2B-it
    python text_qa.py --model /path/to/vllm_model --n_samples 200
"""

import argparse
import csv
import json
import time
from pathlib import Path

import evaluate
from datasets import load_dataset
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = (
    "Answer the following agricultural question.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")


def load_agrocot(n_samples: int | None) -> list[dict]:
    ds = load_dataset("AgriCulturalQA/AgroCoT", split="test")
    samples = []
    for row in ds:
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
        })
    if n_samples:
        samples = samples[:n_samples]
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
        "rouge1": round(rouge_scores["rouge1"], 4),
        "rouge2": round(rouge_scores["rouge2"], 4),
        "rougeL": round(rouge_scores["rougeL"], 4),
    }


def main(args):
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=256,
    )

    print("Loading AgroCoT dataset...")
    samples = load_agrocot(args.n_samples)
    print(f"  {len(samples)} samples")

    print("\nSample prompt:")
    print(PROMPT_TEMPLATE.format(question=samples[0]["question"]))
    print(f"  Expected answer: {samples[0]['answer']}\n")

    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in samples]

    print("Running text-only inference...")
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - t_start

    predictions, references = [], []
    rows = []
    for sample, output in zip(samples, outputs):
        prediction = output.outputs[0].text.strip()
        answer = sample["answer"]
        predictions.append(prediction)
        references.append(answer)

        contains = answer.lower() in prediction.lower()
        rows.append({
            "question": sample["question"],
            "ground_truth": answer,
            "response": prediction,
            "contains_match": int(contains),
        })

    metrics = compute_metrics(predictions, references)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    metrics["total_samples"] = len(samples)
    metrics["total_time_s"] = round(elapsed, 3)
    metrics["latency_per_sample_s"] = round(elapsed / len(samples), 3)
    metrics["throughput_tokens_per_s"] = round(total_tokens / elapsed, 1)

    print("=== Text QA Metrics (AgroCoT) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "ground_truth", "response", "contains_match"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_csv}")

    metrics_path = output_csv.with_name(output_csv.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, "dataset": "AgroCoT", **metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="HuggingFace model ID or path to fine-tuned vLLM model")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--output_csv", type=str, default="results/text_qa_results.csv")
    args = parser.parse_args()
    main(args)
