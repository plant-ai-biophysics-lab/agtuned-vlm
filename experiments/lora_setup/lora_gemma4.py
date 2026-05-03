from typing import Any, Dict, cast
import argparse

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
PROMPT = "Identify the plant and any disease present in this image."
LABEL_TEMPLATE = "This image shows {label_text}."

def create_model(model_type: str):

    assert model_type in GEMMA_4_MODELS

    model, processor = FastVisionModel.from_pretrained(
        model_type,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth"
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 32,                           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 32,                  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,               # We support rank stabilized LoRA
        loftq_config = None,               # And LoftQ
        # target_modules = "all-linear",    # Optional now! Can specify a list if needed
    )

    return model, processor

def convert_to_multimodal_hf(loader: Any):
    contents = cast(Dict[str, Any], loader.export_contents())
    if contents is None:
        raise ValueError("Failed to export contents from the AgML loader.")

    paths = list(contents.keys())
    labels = [contents[p] for p in paths]
    class_names = cast(list[str], loader.classes)

    formatted_data = []
    for path, label_idx in zip(paths, labels):

        label_idx = int(cast(Any, label_idx))
        label_text = class_names[label_idx]
        cleaned_label = label_text.replace('___', ' ').replace('_', ' ')

        image = PILImage.open(path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image", "image": image},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": LABEL_TEMPLATE.format(label_text=cleaned_label)}
                ]
            }
        ]

        formatted_data.append({"messages": messages})

    return formatted_data

def main(args):

    model, processor = create_model(args.model)

    # agml data loading
    loader = agml.data.AgMLDataLoader("plant_village_classification")
    loader.split(train=0.8, val=0.1, test=0.1)

    train_ds = convert_to_multimodal_hf(loader.train_data)
    val_ds = convert_to_multimodal_hf(loader.val_data)
    test_ds = convert_to_multimodal_hf(loader.test_data)

    print(f"Dataset sizes: Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print("Example sample:")
    print(train_ds[0]["messages"])

    # set val rate
    eval_steps = int(0.2 * len(train_ds))
    print(f"Using eval steps: {eval_steps}")

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        processing_class = processor,
        data_collator = UnslothVisionDataCollator(model, processor),
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            max_grad_norm = 0.3,
            warmup_ratio = 0.03,
            # max_steps = 10,
            num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = eval_steps,
            save_strategy = "steps",
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = args.output_dir if args.output_dir is not None else "outputs",
            report_to = "trackio", # For Weights and Biases or others

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        )
    )

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # save model
    model.save_pretrained("gemma_4_lora")  # Local saving
    processor.save_pretrained("gemma_4_lora")
    # model.push_to_hub("your_name/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
    # processor.push_to_hub("your_name/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving

    # Select ONLY 1 to save! (Both not needed!)

    # Save locally to 16bit
    if True: model.save_pretrained_merged("unsloth_finetune", processor,)

    # To export and save to your Hugging Face account
    if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", processor, token = "YOUR_HF_TOKEN")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
