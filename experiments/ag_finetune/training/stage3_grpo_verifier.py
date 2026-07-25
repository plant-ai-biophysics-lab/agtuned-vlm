"""
Stage 4d — GRPO verifier RL training.

The verifier model reads a (question, wrong prediction) pair and must produce
a correction. Reward = 1.0 if the corrected answer matches ground truth,
0.0 otherwise (with partial credit for contains-match).

Usage:
  uv run python experiments/ag_finetune/training/stage3_grpo_verifier.py \
    --model outputs/checkpoints/stage2_dpo \
    --data  outputs/stage5_verifier_data.jsonl \
    --output_dir outputs/checkpoints/stage3_grpo
"""

import argparse
import json
import re
from pathlib import Path


def extract_answer(text: str) -> str:
    """Pull out the 'Correct answer: ...' line from verifier output."""
    m = re.search(r"[Cc]orrect answer[:\s]+(.+)", text)
    if m:
        return m.group(1).strip().lower()
    return text.strip().lower()


def reward_fn(completions: list[str], ground_truths: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred = extract_answer(completion)
        gt_lower = gt.strip().lower()
        if pred == gt_lower:
            rewards.append(1.0)
        elif gt_lower in pred or pred in gt_lower:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="outputs/checkpoints/stage2_dpo")
    parser.add_argument("--data",         default="outputs/stage5_verifier_data.jsonl")
    parser.add_argument("--output_dir",   default="outputs/checkpoints/stage3_grpo")
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--n_completions",type=int,   default=4,  help="Samples per prompt for GRPO")
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--kl_coeff",     type=float, default=0.01)
    parser.add_argument("--max_seq_len",  type=int,   default=1024)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run 05_build_verifier_data.py first. Missing: {data_path}")
        return

    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} verifier examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_training(model)

    # Build prompts: show the question + wrong prediction, ask for correction
    prompts, ground_truths = [], []
    for row in rows:
        msgs = row.get("messages", [])
        user_content = msgs[0]["content"] if msgs else f"Question: {row['question']}\nPrediction: {row['wrong_pred']}"
        prompts.append(user_content)
        ground_truths.append(row["correct"])

    dataset = Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            num_generations=args.n_completions,
            learning_rate=args.lr,
            kl_coeff=args.kl_coeff,
            bf16=True,
            logging_steps=20,
            save_steps=200,
            max_completion_length=256,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
