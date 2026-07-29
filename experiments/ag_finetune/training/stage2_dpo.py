"""
DPO alignment on the consolidated preference pairs (build_dpo_data.py), with
LoRA (Unsloth). Warm-starts from a Stage 1 SFT checkpoint by default.

Plain {prompt, chosen, rejected} — no verifier JSON schema, no confidence
field (that's stage7_verifier_dpo_pairs.jsonl's job, left untouched).

IMPORTANT — same vision-capable-base requirement as stage1_sft_reasoning.py:
uses unsloth's FastVisionModel (not FastLanguageModel), text-only data so
only language layers are unfrozen. See that script's docstring for why this
matters for eventually loading the result into run_verifier_ppt.py.

Saves both outputs, matching stage1_sft_reasoning.py's convention:
  <output_dir>/lora_adapters/  — raw PEFT adapters
  <output_dir>/vllm_model/     — merged 16-bit weights, directly loadable by
                                  vLLM: run_verifier_ppt.py --model <this path>

Usage:
  uv run python experiments/ag_finetune/training/stage2_dpo.py \
    --model outputs/checkpoints/stage1_sft/vllm_model \
    --data  outputs/dpo_pairs.jsonl \
    --output_dir outputs/checkpoints/stage2_dpo
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="outputs/checkpoints/stage1_sft/vllm_model",
                        help="Vision-capable base — a Stage 1 SFT merged checkpoint by "
                             "default, or pass a fresh base model to skip SFT warm-start.")
    parser.add_argument("--data",        default="outputs/dpo_pairs.jsonl")
    parser.add_argument("--output_dir",  default="outputs/checkpoints/stage2_dpo")
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=16)
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--max_seq_len", type=int,   default=1024)
    args = parser.parse_args()

    from unsloth import FastVisionModel, is_bf16_supported
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run build_dpo_data.py first. Missing: {data_path}")
        return

    pairs = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs):,} DPO pairs")

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

    dataset = Dataset.from_dict({
        "prompt":   [p["prompt"]   for p in pairs],
        "chosen":   [p["chosen"]   for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=processor,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=20,
            save_steps=200,
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
