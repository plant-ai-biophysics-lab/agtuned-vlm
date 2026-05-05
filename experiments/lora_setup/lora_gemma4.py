from typing import Any, Dict, cast
import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path

import agml
import torch
from PIL import Image as PILImage
from unsloth import FastVisionModel, is_bf16_supported, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

GEMMA_4_MODELS = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth
PROMPT = "Identify the plant and any disease present in this image. Respond in one sentence: "
LABEL_TEMPLATE = "This image shows {label_text}."


def create_model(model_type: str):

    assert model_type in GEMMA_4_MODELS

    # 2B bnb-4bit checkpoint from unsloth is broken; 2B fits in bf16 on most GPUs
    load_in_4bit = "E2B" not in model_type

    model, processor = FastVisionModel.from_pretrained(
        model_type,
        load_in_4bit = False,
        use_gradient_checkpointing = "unsloth"
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = False,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    return model, processor


def make_sample(image: PILImage.Image, response: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]
    }


def convert_to_multimodal_hf(loader: Any) -> list[dict]:
    contents = cast(Dict[str, Any], loader.export_contents())
    if contents is None:
        raise ValueError("Failed to export contents from the AgML loader.")

    class_names = cast(list[str], loader.classes)
    formatted_data = []
    for path, label_idx in contents.items():
        label_text = class_names[int(cast(Any, label_idx))]
        parts = label_text.split('___')
        crop = parts[0].replace('_', ' ')
        cleaned_label = f"{crop} with {parts[1].replace('_', ' ')}" if len(parts) > 1 else crop
        image = PILImage.open(path).convert("RGB")
        formatted_data.append(make_sample(image, LABEL_TEMPLATE.format(label_text=cleaned_label)))

    return formatted_data


def parse_dg_label(folder_name: str) -> str:
    """Convert 'tomato__early_blight__thrips' → 'tomato early blight, thrips'."""
    parts = folder_name.split("__")
    crop = parts[0].replace("_", " ")
    if len(parts) == 1:
        return crop
    conditions = ", ".join(p.replace("_", " ") for p in parts[1:])
    return f"{crop} with {conditions}"


def load_digital_green(data_dir: str) -> list[dict]:
    data_dir = Path(data_dir)
    captions_tsv = data_dir / "captions.tsv"
    image_dir = data_dir / "digital_green_crop_diagnosis_classification"

    # Build {filename → [caption, ...]} lookup — TSV paths use old Mac paths,
    # so match by filename only.
    caption_map: dict[str, list[str]] = {}
    with open(captions_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            filename = Path(row["image_path"]).name
            caption = row["caption"].strip()
            if caption:
                caption_map.setdefault(filename, []).append(caption)

    formatted_data = []
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = parse_dg_label(class_dir.name)

        for img_path in sorted(class_dir.glob("*.jpg")):
            image = PILImage.open(img_path).convert("RGB")
            captions = caption_map.get(img_path.name, [])

            if captions:
                # One training sample per caption so the model sees all explanations.
                for caption in captions:
                    response = f"{LABEL_TEMPLATE.format(label_text=label)} {caption}"
                    formatted_data.append(make_sample(image, response))
            else:
                formatted_data.append(make_sample(image, LABEL_TEMPLATE.format(label_text=label)))

    return formatted_data


def stratified_sample(data: list[dict], fraction: float) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for sample in data:
        label = sample["messages"][1]["content"][0]["text"]
        groups.setdefault(label, []).append(sample)
    result = []
    for samples in groups.values():
        k = max(1, int(len(samples) * fraction))
        result.extend(random.sample(samples, k))
    return result


def split_dataset(data: list[dict], train: float = 0.8, val: float = 0.1) -> tuple[list, list, list]:
    shuffled = data.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def print_sample(sample: dict, label: str):
    messages = sample["messages"]
    user_text = next(c["text"] for c in messages[0]["content"] if c["type"] == "text")
    assistant_text = messages[1]["content"][0]["text"]
    print(f"  [{label}]")
    print(f"  Prompt   : {user_text}")
    print(f"  Response : {assistant_text}")
    print()


def main(args):

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    model, processor = create_model(args.model)

    # PlantVillage
    loader = agml.data.AgMLDataLoader("plant_village_classification")
    loader.split(train=0.8, val=0.1, test=0.1)
    pv_train = convert_to_multimodal_hf(loader.train_data)
    pv_val   = convert_to_multimodal_hf(loader.val_data)
    pv_test  = convert_to_multimodal_hf(loader.test_data)

    if args.pv_downsample < 1.0:
        pv_train = stratified_sample(pv_train, args.pv_downsample)

    train_ds = pv_train
    val_ds   = pv_val
    test_ds  = pv_test

    print("=== PlantVillage ===")
    print(f"  Train: {len(pv_train)}, Val: {len(pv_val)}, Test: {len(pv_test)}")
    print("  Sample:")
    print_sample(pv_train[0], "PlantVillage")

    # Digital Green (optional)
    if args.dg_data_dir:
        dg_all = load_digital_green(args.dg_data_dir)
        dg_train, dg_val, dg_test = split_dataset(dg_all)

        print("=== Digital Green ===")
        print(f"  Total samples (incl. per-caption): {len(dg_all)}")
        print(f"  Train: {len(dg_train)}, Val: {len(dg_val)}, Test: {len(dg_test)}")
        print(f"  Upsample factor: {args.dg_upsample}x → {len(dg_train) * args.dg_upsample} train samples")
        print("  Sample:")
        print_sample(dg_train[0], "Digital Green")

        train_ds = pv_train + dg_train * args.dg_upsample
        val_ds   = pv_val + dg_val
        test_ds  = pv_test + dg_test
        random.shuffle(train_ds)

    print("=== Combined ===")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # set val rate
    eval_steps = int(0.2 * len(train_ds))
    print(f"  Eval steps: {eval_steps}")

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        processing_class = processor,
        data_collator = UnslothVisionDataCollator(model, processor),
        args = SFTConfig(
            per_device_train_batch_size = 12,
            gradient_accumulation_steps = 1,
            max_grad_norm = 0.3,
            warmup_ratio = 0.03,
            # max_steps = 1,
            num_train_epochs = 10,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = eval_steps,
            save_strategy = "steps",
            save_steps = eval_steps,
            save_total_limit = 1,
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = args.output_dir if args.output_dir is not None else "outputs",
            report_to = "trackio",

            # Required for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        )
    )

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    train_runtime = trainer_stats.metrics["train_runtime"]

    gpu_stats_dict = {
        "train_runtime_s": round(train_runtime, 2),
        "train_runtime_min": round(train_runtime / 60, 2),
        "gpu_name": gpu_stats.name,
        "gpu_total_memory_gb": max_memory,
        "peak_reserved_memory_gb": used_memory,
        "peak_reserved_for_lora_gb": used_memory_for_lora,
        "peak_reserved_pct": used_percentage,
        "peak_reserved_for_lora_pct": lora_percentage,
    }

    for k, v in gpu_stats_dict.items():
        print(f"  {k}: {v}")

    output_dir = args.output_dir if args.output_dir is not None else "outputs"

    model.save_pretrained(f"{output_dir}/lora_adapters")
    processor.save_pretrained(f"{output_dir}/lora_adapters")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/gpu_stats.json", "w") as f:
        json.dump(gpu_stats_dict, f, indent=2)
    print(f"GPU stats saved to {output_dir}/gpu_stats.json")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    model.save_pretrained_merged(f"{output_dir}/vllm_model", processor, save_method="merged_16bit")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dg_data_dir", type=str, default=None,
                        help="Path to digital_green/ dataset root. Omit to train on PlantVillage only.")
    parser.add_argument("--dg_upsample", type=int, default=1,
                        help="Repeat Digital Green train samples this many times to balance with PlantVillage.")
    parser.add_argument("--pv_downsample", type=float, default=1.0,
                        help="Fraction of PlantVillage train samples to keep (e.g. 0.1 = 10%%). Val/test unaffected.")
    args = parser.parse_args()

    main(args)
