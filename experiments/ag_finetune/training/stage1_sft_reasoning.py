"""
Stage 4a — SFT on reasoning chains with LoRA (Unsloth).

Trains Qwen2.5-7B-Instruct on bootstrapped CoT chains from stage 4.
Format: <|user|>{question}<|assistant|><think>{chain}</think>{answer}

Usage:
  uv run python experiments/ag_finetune/training/stage1_sft_reasoning.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data  outputs/stage4_reasoning_chains.jsonl \
    --output_dir outputs/checkpoints/stage1_sft
"""

import argparse
import json
from pathlib import Path


def format_example(row: dict, tokenizer) -> dict:
    system = row.get("system", "You are an expert agricultural advisor.")
    question = row["question"]
    chain    = row.get("chain", "")
    answer   = row["answer"]

    msgs = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": f"<think>\n{chain}\n</think>\n\n{answer}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data",          default="outputs/stage4_reasoning_chains.jsonl")
    parser.add_argument("--output_dir",    default="outputs/checkpoints/stage1_sft")
    parser.add_argument("--lora_r",        type=int,   default=64)
    parser.add_argument("--lora_alpha",    type=int,   default=128)
    parser.add_argument("--epochs",        type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--max_seq_len",   type=int,   default=2048)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run 04_bootstrap_reasoning.py first. Missing: {data_path}")
        return

    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} training examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    texts = [format_example(r, tokenizer) for r in rows]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
