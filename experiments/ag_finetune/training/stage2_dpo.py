"""
Stage 4c — DPO alignment on Chi-Metrics preference pairs.

Warm-starts from the Stage 1 SFT checkpoint.

Usage:
  uv run python experiments/ag_finetune/training/stage2_dpo.py \
    --model outputs/checkpoints/stage1_sft \
    --data  outputs/stage2_chi_dpo_pairs.jsonl \
    --output_dir outputs/checkpoints/stage2_dpo
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="outputs/checkpoints/stage1_sft")
    parser.add_argument("--data",        default="outputs/stage2_chi_dpo_pairs.jsonl")
    parser.add_argument("--output_dir",  default="outputs/checkpoints/stage2_dpo")
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--max_seq_len", type=int,   default=1024)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run 02_filter_chi_metrics.py first. Missing: {data_path}")
        return

    pairs = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs):,} DPO pairs")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_training(model)

    dataset = Dataset.from_dict({
        "prompt":   [p["chosen_prompt"]    for p in pairs],
        "chosen":   [p["chosen_response"]  for p in pairs],
        "rejected": [p["rejected_response"] for p in pairs],
    })

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=20,
            save_steps=200,
            max_length=args.max_seq_len,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
