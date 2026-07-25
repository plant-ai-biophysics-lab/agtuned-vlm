"""
Evaluate a fine-tuned model on agrillm-qa-eval-800.

Computes exact match, contains match, and ROUGE-L vs reference answers.

Usage:
  uv run python experiments/ag_finetune/evaluation/eval_agrillm.py \
    --model outputs/checkpoints/stage3_grpo \
    --output_dir outputs/eval
"""

import argparse
import json
from pathlib import Path


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    exact, contains, rouge_scores = 0, 0, []
    for pred, ref in zip(predictions, references):
        p, r = pred.strip().lower(), ref.strip().lower()
        if p == r:
            exact += 1
        if r in p or p in r:
            contains += 1
        rouge_scores.append(scorer.score(r, p)["rougeL"].fmeasure)

    n = len(predictions)
    return {
        "exact_match":    round(exact / n, 4),
        "contains_match": round(contains / n, 4),
        "rougeL":         round(sum(rouge_scores) / n, 4),
        "n":              n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="outputs/checkpoints/stage3_grpo")
    parser.add_argument("--output_dir",  default="outputs/eval")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=8)
    args = parser.parse_args()

    from datasets import load_dataset
    from unsloth import FastLanguageModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading agrillm-qa-eval-800...")
    eval_ds = list(load_dataset("AI71ai/agrillm-qa-eval-800", split="test"))
    print(f"  {len(eval_ds):,} eval examples")

    print(f"Loading model {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    predictions, references, rows_out = [], [], []
    for row in eval_ds:
        question = row.get("question", "")
        reference = row.get("answer", "")
        # Strip prompt scaffolding
        import re
        q_clean = re.sub(r"###\s*QUESTION:\s*", "", question)
        q_clean = re.sub(r"(Formulate|Compose|Please create)[^#\n]*\n+", "", q_clean)
        q_clean = re.sub(r"###\s*An.*$", "", q_clean, flags=re.MULTILINE).strip()

        msgs = [{"role": "user", "content": q_clean}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predictions.append(pred)
        references.append(reference)
        rows_out.append({"question": q_clean, "reference": reference, "prediction": pred})

    metrics = compute_metrics(predictions, references)
    print("\nResults:", json.dumps(metrics, indent=2))

    (out_dir / "eval_agrillm_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "eval_agrillm_predictions.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
