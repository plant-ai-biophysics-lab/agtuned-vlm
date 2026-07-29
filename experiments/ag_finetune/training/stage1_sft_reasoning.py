"""
SFT on the consolidated Q&A data (build_sft_data.py) with LoRA (Unsloth).

Plain open-ended {question, answer} — no <think> chain wrapper (that was
the old stage4-based format; this simplified path drops it, see
build_sft_data.py's docstring).

IMPORTANT — base model must be vision-capable and match what
experiments/verifier_ppt/run_verifier_ppt.py loads: --model has no default
on purpose. This trains via unsloth's FastVisionModel (not FastLanguageModel)
on the SAME Gemma-4 family run_verifier_ppt.py uses, with only the language
layers unfrozen (finetune_vision_layers=False) since this data is text-only —
the opposite split from experiments/lora_setup/lora_gemma4.py, which tunes
vision layers for image diagnosis and freezes language. Using FastLanguageModel
+ a text-only model (e.g. Qwen) here would produce a checkpoint with no vision
tower at all — incompatible with verifier_ppt's image-grounded verifier prompt,
which is the whole point of checking the score distribution afterward.

Saves both outputs, matching lora_gemma4.py's convention:
  <output_dir>/lora_adapters/  — raw PEFT adapters
  <output_dir>/vllm_model/     — merged 16-bit weights, directly loadable by
                                  vLLM: run_verifier_ppt.py --model <this path>

Usage:
  uv run python experiments/ag_finetune/training/stage1_sft_reasoning.py \
    --model unsloth/gemma-4-E4B-it \
    --data  outputs/sft_data.jsonl \
    --output_dir outputs/checkpoints/stage1_sft
"""

import argparse
import json
from pathlib import Path


def make_sample(question: str, answer: str) -> dict:
    """Text-only turn in the vision-model content-list convention (matches
    lora_gemma4.py's make_sample / this repo's documented multimodal sample
    format) — no image entries, since this data has none."""
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=True,
                        help="Vision-capable base model — must match what "
                             "run_verifier_ppt.py --model loads, e.g. "
                             "unsloth/gemma-4-E4B-it")
    parser.add_argument("--data",          default="outputs/sft_data.jsonl")
    parser.add_argument("--output_dir",    default="outputs/checkpoints/stage1_sft")
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--max_seq_len",   type=int,   default=2048)
    args = parser.parse_args()

    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run build_sft_data.py first. Missing: {data_path}")
        return

    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} training examples")

    dataset = [make_sample(r["question"], r["answer"]) for r in rows]

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # text-only data — leave the vision tower alone
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_8bit",
            seed=3407,
            # Required for vision finetuning (see lora_gemma4.py):
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_len,
        ),
    )
    trainer.train()

    model.save_pretrained(f"{args.output_dir}/lora_adapters")
    processor.save_pretrained(f"{args.output_dir}/lora_adapters")
    model.save_pretrained_merged(f"{args.output_dir}/vllm_model", processor, save_method="merged_16bit")
    print(f"\nSaved adapters to {args.output_dir}/lora_adapters")
    print(f"Saved merged 16-bit weights to {args.output_dir}/vllm_model "
          f"(pass this path to run_verifier_ppt.py --model)")


if __name__ == "__main__":
    main()
