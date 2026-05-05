import argparse
import csv
import json
import time
from pathlib import Path

import agml
import evaluate
from PIL import Image as PILImage
from sklearn.metrics import accuracy_score, f1_score, classification_report
from vllm import LLM, SamplingParams

PROMPT = "Identify the plant and any disease present in this image."

rouge = evaluate.load("rouge")
exact_match = evaluate.load("exact_match")


def load_samples(n_samples: int):
    loader = agml.data.AgMLDataLoader("plant_village_classification")
    loader.split(train=0.8, val=0.1, test=0.1)

    contents = loader.test_data.export_contents()
    class_names = loader.classes
    cleaned_class_names = [c.replace("___", " ").replace("_", " ") for c in class_names]

    samples = []
    for path, label_idx in list(contents.items())[:n_samples]:
        cleaned_label = cleaned_class_names[int(label_idx)]
        image = PILImage.open(path).convert("RGB")
        samples.append({"image": image, "path": path, "ground_truth": cleaned_label})

    return samples, cleaned_class_names


def build_prompt(processor, image: PILImage.Image) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text, "multi_modal_data": {"image": image}}


def resolve_predicted_class(response: str, class_names: list[str]) -> str | None:
    """Map free-text response to a class name. Returns None if zero or multiple classes match."""
    response_lower = response.lower()
    matches = [c for c in class_names if c.lower() in response_lower]
    if len(matches) != 1:
        return None
    return matches[0]


def compute_metrics(
    predictions: list[str],
    references: list[str],
    pred_classes: list[str],
) -> tuple[dict, str]:
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    em_scores = exact_match.compute(
        predictions=[p.lower() for p in predictions],
        references=[r.lower() for r in references],
    )
    contains_match = sum(
        r.lower() in p.lower() for p, r in zip(predictions, references)
    ) / len(predictions)

    resolved = [(p, r) for p, r in zip(pred_classes, references) if p is not None]
    resolvable = len(resolved)
    if resolved:
        res_preds, res_refs = zip(*resolved)
        all_labels = sorted(set(res_refs) | set(res_preds))
        accuracy = accuracy_score(res_refs, res_preds)
        f1_macro = f1_score(res_refs, res_preds, average="macro", labels=all_labels, zero_division=0)
        f1_weighted = f1_score(res_refs, res_preds, average="weighted", labels=all_labels, zero_division=0)
        report = classification_report(res_refs, res_preds, labels=all_labels, zero_division=0)
    else:
        accuracy = f1_macro = f1_weighted = 0.0
        report = "No resolvable predictions."

    metrics = {
        "contains_match": round(contains_match, 4),
        "exact_match": round(em_scores["exact_match"], 4),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "rouge1": round(rouge_scores["rouge1"], 4),
        "rouge2": round(rouge_scores["rouge2"], 4),
        "rougeL": round(rouge_scores["rougeL"], 4),
        "resolvable_samples": resolvable,
    }
    return metrics, report


def main(args):
    print(f"Loading model from: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
    )

    processor = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=64,
    )

    print(f"Loading {args.n_samples} test samples...")
    samples, class_names = load_samples(args.n_samples)

    inputs = [build_prompt(processor, s["image"]) for s in samples]

    print("Running inference...")
    t_start = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    elapsed = time.perf_counter() - t_start

    predictions, references, pred_classes = [], [], []
    rows = []
    for sample, output in zip(samples, outputs):
        prediction = output.outputs[0].text.strip()
        gt = sample["ground_truth"]
        pred_class = resolve_predicted_class(prediction, class_names)

        predictions.append(prediction)
        references.append(gt)
        pred_classes.append(pred_class)

        contains = gt.lower() in prediction.lower()
        rows.append({
            "image": sample["path"],
            "ground_truth": gt,
            "predicted_class": pred_class,
            "response": prediction,
            "contains_match": int(contains),
        })
        print(f"  Image      : {Path(sample['path']).name}")
        print(f"  GT         : {gt}")
        print(f"  Pred class : {pred_class}")
        print(f"  Response   : {prediction}")
        print(f"  Match      : {'✓' if contains else '✗'}")
        print()

    metrics, report = compute_metrics(predictions, references, pred_classes)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    metrics["total_samples"] = len(samples)
    metrics["total_time_s"] = round(elapsed, 3)
    metrics["latency_per_sample_s"] = round(elapsed / len(samples), 3)
    metrics["throughput_tokens_per_s"] = round(total_tokens / elapsed, 1)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Per-class Report ===")
    print(report)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "ground_truth", "predicted_class", "response", "contains_match"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_csv}")

    metrics_path = output_csv.with_name(output_csv.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, **metrics, "per_class_report": report}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-4-E2B-it",
        help="Path to merged vLLM model or HuggingFace model ID",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of test samples to run inference on",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/inference_results.csv",
        help="Path to save the output CSV",
    )
    args = parser.parse_args()
    main(args)
