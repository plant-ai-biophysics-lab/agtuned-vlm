import argparse
import csv
from pathlib import Path

import agml
from PIL import Image as PILImage
from vllm import LLM, SamplingParams

PROMPT = "Identify the plant and any disease present in this image."


def load_samples(n_samples: int):
    loader = agml.data.AgMLDataLoader("plant_village_classification")
    loader.split(train=0.8, val=0.1, test=0.1)

    contents = loader.test_data.export_contents()
    class_names = loader.classes

    samples = []
    for path, label_idx in list(contents.items())[:n_samples]:
        label_text = class_names[int(label_idx)]
        cleaned_label = label_text.replace("___", " ").replace("_", " ")
        image = PILImage.open(path).convert("RGB")
        samples.append({"image": image, "path": path, "ground_truth": cleaned_label})

    return samples


def build_prompt(processor, image: PILImage.Image) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "image": image},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": text, "multi_modal_data": {"image": image}}


def main(args):
    print(f"Loading model from: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.4,
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
    samples = load_samples(args.n_samples)

    inputs = [build_prompt(processor, s["image"]) for s in samples]

    print("Running inference...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    correct = 0
    rows = []
    for sample, output in zip(samples, outputs):
        prediction = output.outputs[0].text.strip()
        gt = sample["ground_truth"]
        match = gt.lower() in prediction.lower()
        if match:
            correct += 1
        rows.append({"image": sample["path"], "ground_truth": gt, "response": prediction})
        print(f"  Image : {Path(sample['path']).name}")
        print(f"  GT    : {gt}")
        print(f"  Pred  : {prediction}")
        print(f"  Match : {'✓' if match else '✗'}")
        print()

    print(f"Accuracy: {correct}/{len(samples)} ({100 * correct / len(samples):.1f}%)")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "response"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_csv}")


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
