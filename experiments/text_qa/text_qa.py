"""
H2: Text-only QA — Agricultural Language Knowledge
===================================================
Tests whether the model can answer agricultural questions from text alone
(no images) using the AgroCoT benchmark (multiple-choice).

If text-only performance is weak → the language model lacks agricultural
domain knowledge independently of any vision issues → H2 confirmed.
If text-only performance is strong → the bottleneck is vision, not language.

Dataset: AgroCoT (huggingface: wenyb/AgriCoT)
  Fields: question, options (list), answer (letter A/B/C/D)
  Split: train only (4821 rows)

Backbones compared:
  --model          Gemma4 base/multimodal (required)
  --lang_model     Language-only model, e.g. gemma-3-4b-it (optional, on by default)
  --finetuned_model Fine-tuned Gemma4 (optional)
  Random chance baseline is always computed.

Usage:
    python text_qa.py --model google/gemma-4-E2B-it
    python text_qa.py --model google/gemma-4-E2B-it \\
                      --finetuned_model /path/to/vllm_model \\
                      --n_samples 500
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

PROMPT_TEMPLATE = (
    "Answer the following agricultural multiple-choice question. "
    "Reply with only the letter of the correct answer (A, B, C, or D).\n\n"
    "Question: {question}\n\n"
    "{options_block}\n\n"
    "Answer:"
)


def format_options(options) -> str:
    if isinstance(options, dict):
        items = options.items()
    else:
        items = zip(OPTION_LETTERS, options)
    return "\n".join(f"{letter}. {text}" for letter, text in items)


def extract_predicted_letter(response: str) -> str | None:
    m = re.search(r"\b([A-E])\b", response.strip())
    return m.group(1) if m else None


def load_agrocot(n_samples: int | None, dataset_path: str | None = None) -> list[dict]:
    if dataset_path:
        from datasets import load_from_disk
        print(f"Loading AgroCoT from disk: {dataset_path}")
        ds = load_from_disk(dataset_path)
    else:
        print("Loading AgroCoT dataset (wenyb/AgriCoT, train split)...")
        ds = load_dataset("wenyb/AgriCoT", split="train")
    samples = []
    for row in ds:
        samples.append({
            "question": row["question"],
            "options": row["options"],
            "answer": str(row["answer"]).strip().upper(),
        })
    if n_samples:
        samples = samples[:n_samples]
    print(f"  {len(samples)} samples")
    return samples


def run_inference(llm: LLM, prompts: list[str], label: str) -> list[str]:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    print(f"\nRunning inference [{label}]...")
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - t_start
    responses = [o.outputs[0].text.strip() for o in outputs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {elapsed:.1f}s | {total_tokens / elapsed:.0f} tok/s")
    return responses, elapsed, total_tokens


def score(responses: list[str], samples: list[dict]) -> dict:
    n_correct = 0
    n_parsed = 0
    rows = []
    for sample, response in zip(samples, responses):
        predicted = extract_predicted_letter(response)
        gt = sample["answer"]
        correct = predicted == gt if predicted is not None else False
        if predicted is not None:
            n_parsed += 1
        if correct:
            n_correct += 1
        rows.append({
            "question": sample["question"],
            "ground_truth": gt,
            "raw_response": response,
            "predicted_letter": predicted or "",
            "correct": int(correct),
        })
    n = len(samples)
    return {
        "accuracy": round(n_correct / n, 4),
        "parse_rate": round(n_parsed / n, 4),
        "n_correct": n_correct,
        "n_total": n,
    }, rows


def random_baseline_accuracy(samples: list[dict]) -> float:
    import random
    random.seed(42)
    n_correct = sum(
        random.choice(OPTION_LETTERS[:4]) == s["answer"] for s in samples
    )
    return round(n_correct / len(samples), 4)


def load_llm(model_id: str) -> LLM:
    return LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )


def main(args):
    samples = load_agrocot(args.n_samples, args.dataset_path)

    prompts = [
        PROMPT_TEMPLATE.format(
            question=s["question"],
            options_block=format_options(s["options"]),
        )
        for s in samples
    ]

    print("\nSample prompt:")
    print(prompts[0])
    print(f"  Expected answer: {samples[0]['answer']}")

    results = {
        "dataset": "AgroCoT (wenyb/AgriCoT)",
        "n_samples": len(samples),
        "models": {},
    }
    all_rows: dict[str, list[dict]] = {}

    # --- Random baseline ---
    rand_acc = random_baseline_accuracy(samples)
    print(f"\n=== Random Baseline ===")
    print(f"  Accuracy: {rand_acc:.4f}  (expected ~0.25 for 4-option MCQ)")
    results["models"]["random"] = {"accuracy": rand_acc, "parse_rate": 1.0}

    # --- Language-only model ---
    if args.lang_model:
        print(f"\n=== Language Model: {args.lang_model} ===")
        llm = load_llm(args.lang_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.lang_model)
        metrics, rows = score(responses, samples)
        metrics["model_id"] = args.lang_model
        metrics["total_time_s"] = round(elapsed, 3)
        metrics["throughput_tokens_per_s"] = round(total_tokens / elapsed, 1)
        results["models"]["lang_model"] = metrics
        all_rows["lang_model"] = rows
        del llm

    # --- Gemma4 base ---
    print(f"\n=== Gemma4 Base: {args.model} ===")
    llm = load_llm(args.model)
    responses, elapsed, total_tokens = run_inference(llm, prompts, args.model)
    metrics, rows = score(responses, samples)
    metrics["model_id"] = args.model
    metrics["total_time_s"] = round(elapsed, 3)
    metrics["throughput_tokens_per_s"] = round(total_tokens / elapsed, 1)
    results["models"]["gemma4_base"] = metrics
    all_rows["gemma4_base"] = rows
    del llm

    # --- Gemma4 fine-tuned (optional) ---
    if args.finetuned_model:
        print(f"\n=== Gemma4 Fine-tuned: {args.finetuned_model} ===")
        llm = load_llm(args.finetuned_model)
        responses, elapsed, total_tokens = run_inference(llm, prompts, args.finetuned_model)
        metrics, rows = score(responses, samples)
        metrics["model_id"] = args.finetuned_model
        metrics["total_time_s"] = round(elapsed, 3)
        metrics["throughput_tokens_per_s"] = round(total_tokens / elapsed, 1)
        results["models"]["gemma4_finetuned"] = metrics
        all_rows["gemma4_finetuned"] = rows
        del llm

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Parse Rate':>12}")
    print(f"  {'-'*49}")
    for key, m in results["models"].items():
        print(f"  {key:<25} {m['accuracy']:>10.4f} {m.get('parse_rate', 1.0):>12.4f}")

    output_csv = Path(args.output_dir)
    output_csv.mkdir(parents=True, exist_ok=True)

    for key, rows in all_rows.items():
        csv_path = output_csv / f"text_qa_{key}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["question", "ground_truth", "raw_response", "predicted_letter", "correct"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")

    metrics_path = output_csv / "text_qa_metrics.json"
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
                        help="Language-only baseline model. Pass 'none' to skip.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of samples (default: all 4821)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset saved with download_data.py (use when compute nodes have no internet)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.lang_model and args.lang_model.lower() == "none":
        args.lang_model = None

    main(args)
