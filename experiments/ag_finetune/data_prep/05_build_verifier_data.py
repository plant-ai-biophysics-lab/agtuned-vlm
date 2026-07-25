"""
Stage 3 — Build verifier (critic) training data from wrong predictions.

For each wrong prediction, the model writes a correction explaining the error
and suggesting the right answer. Output is a multi-turn constitutional self-critique.

Uses a local model via vLLM — no API key needed.

Inputs (any combination):
  - outputs/stage1_agrillm_filtered.jsonl  (hard negatives from seed data)
  - A CSV of wrong predictions (e.g. from self-cot eval runs)

Outputs:
  outputs/stage5_verifier_data.jsonl
  outputs/stage5_verifier_stats.json

Usage:
  # Hard negatives only (no wrong_csv yet)
  uv run python experiments/ag_finetune/data_prep/05_build_verifier_data.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --n_hardneg 2000

  # With wrong predictions from a self-cot run
  uv run python experiments/ag_finetune/data_prep/05_build_verifier_data.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --wrong_csv outputs/self_cot_regressions.csv \
    --n_hardneg 2000
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"

CORRECTION_SYSTEM = (
    "You are an expert agricultural verifier. "
    "Given a question, the model's wrong prediction, and the correct answer, "
    "write a brief correction (2–4 sentences) that: "
    "(1) explains specifically why the prediction is wrong, "
    "(2) provides the correct answer with the agricultural reasoning. "
    "Do NOT just state the correct label — explain the mechanism."
)

CORRECTION_TEMPLATE = """\
Question: {question}
Model predicted: {wrong}
Correct answer: {correct}

Write the correction:"""

HARDNEG_SYSTEM = (
    "You are an agricultural expert. Given a correct Q&A pair, "
    "generate {n} plausible-but-wrong alternative answers that a model might hallucinate. "
    "Each wrong answer should sound agricultural but be factually incorrect. "
    "Return ONLY a JSON array of strings: [\"wrong1\", \"wrong2\", ...]"
)


def _load_llm(args):
    from vllm import LLM, SamplingParams
    print(f"Loading {args.model} (quantization={args.quantization or 'none'})...")
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    print("Model loaded.\n")
    return llm, SamplingParams


def batch_generate(llm, SamplingParams, messages_list: list[list[dict]],
                   max_tokens: int, temperature: float = 0.3) -> list[str]:
    """Run llm.chat on a list of message-lists, return list of text outputs."""
    if not messages_list:
        return []
    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.9)
    outputs = llm.chat(messages_list, sampling_params=sampling, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


def parse_json_list(text: str) -> list[str]:
    """Extract a JSON list from model output, tolerating surrounding prose."""
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return [s for s in json.loads(m.group()) if isinstance(s, str) and s.strip()]
        except Exception:
            pass
    return []


def format_verifier_entry(question: str, wrong: str, correct: str, correction: str) -> dict:
    return {
        "stage": "verifier",
        "source": "verifier_data",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Model prediction: {wrong}\n\n"
                    "Is this prediction correct? If not, explain and suggest the right answer."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Let me verify this prediction.\n\n{correction}\n\n"
                    f"Correct answer: {correct}"
                ),
            },
        ],
        "question":   question,
        "wrong_pred": wrong,
        "correct":    correct,
        "correction": correction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong_csv",    default=None,  help="CSV with wrong predictions")
    parser.add_argument("--n_hardneg",    type=int, default=1000,
                        help="Number of hard-negative triples to generate from agrillm seed")
    parser.add_argument("--n_per_neg",    type=int, default=2,
                        help="Wrong alternatives per seed Q (total = n_hardneg // n_per_neg seeds)")
    parser.add_argument("--model",        default="Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--quantization", default=None,
                        help="awq, awq_marlin, gptq, bitsandbytes, or None for bf16")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_util",     type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    try:
        llm, SamplingParams = _load_llm(args)
    except ImportError:
        print("ERROR: pip install vllm")
        return

    out_path = OUT_DIR / "stage5_verifier_data.jsonl"
    done, failed = 0, 0
    t0 = time.time()

    with open(out_path, "a") as f:

        # ── From existing wrong-predictions CSV ───────────────────────────────
        if args.wrong_csv:
            csv_path = Path(args.wrong_csv)
            if not csv_path.exists():
                print(f"WARNING: wrong_csv not found: {csv_path}")
            else:
                print(f"Processing wrong predictions from {csv_path}...")
                triples: list[tuple[str, str, str]] = []
                with open(csv_path) as cf:
                    for row in csv.DictReader(cf):
                        question = row.get("class_label") or row.get("question", "")
                        wrong    = row.get("prediction")  or row.get("resolved_class", "")
                        correct  = row.get("ground_truth") or row.get("class_label", "")
                        if question and wrong and correct and wrong != correct:
                            triples.append((question, wrong, correct))

                print(f"  {len(triples)} wrong-prediction triples")
                for batch_start in range(0, len(triples), args.batch_size):
                    batch = triples[batch_start : batch_start + args.batch_size]
                    msgs = [
                        [{"role": "system", "content": CORRECTION_SYSTEM},
                         {"role": "user",   "content": CORRECTION_TEMPLATE.format(
                             question=q, wrong=w, correct=c)}]
                        for q, w, c in batch
                    ]
                    try:
                        corrections = batch_generate(llm, SamplingParams, msgs, max_tokens=256)
                        for (q, w, c), corr in zip(batch, corrections):
                            f.write(json.dumps(format_verifier_entry(q, w, c, corr)) + "\n")
                            done += 1
                    except Exception as e:
                        print(f"  Batch {batch_start} failed: {e}")
                        failed += len(batch)
                    elapsed = time.time() - t0
                    print(f"  CSV: {done} done ({failed} failed) [{elapsed:.0f}s]")

        # ── Hard negatives from agrillm seed ─────────────────────────────────
        seed_path = OUT_DIR / "stage1_agrillm_filtered.jsonl"
        if seed_path.exists() and args.n_hardneg > 0:
            seed_rows = [json.loads(l) for l in seed_path.read_text().splitlines() if l.strip()]
            n_seeds = args.n_hardneg // args.n_per_neg
            seed_rows = seed_rows[:n_seeds]
            print(f"\nGenerating hard negatives: {n_seeds} seeds × {args.n_per_neg} = "
                  f"~{args.n_hardneg} target triples...")

            # Pass 1: generate wrong alternatives for each seed
            neg_triples: list[tuple[str, str, str]] = []
            for batch_start in range(0, len(seed_rows), args.batch_size):
                batch = seed_rows[batch_start : batch_start + args.batch_size]
                msgs = [
                    [{"role": "system",
                      "content": HARDNEG_SYSTEM.format(n=args.n_per_neg)},
                     {"role": "user",
                      "content": f"Q: {r['question']}\nA: {r['answer']}"}]
                    for r in batch
                ]
                try:
                    results = batch_generate(llm, SamplingParams, msgs,
                                             max_tokens=256, temperature=0.7)
                    for row, text in zip(batch, results):
                        for wrong in parse_json_list(text)[: args.n_per_neg]:
                            neg_triples.append((row["question"], wrong, row["answer"]))
                except Exception as e:
                    print(f"  Hard-neg batch {batch_start} failed: {e}")
                    failed += len(batch)

                print(f"  Pass 1: {batch_start + len(batch)}/{len(seed_rows)} seeds → "
                      f"{len(neg_triples)} triples so far")

            print(f"\nPass 2: generating corrections for {len(neg_triples)} triples...")

            # Pass 2: generate corrections for all triples
            for batch_start in range(0, len(neg_triples), args.batch_size):
                batch = neg_triples[batch_start : batch_start + args.batch_size]
                msgs = [
                    [{"role": "system", "content": CORRECTION_SYSTEM},
                     {"role": "user",   "content": CORRECTION_TEMPLATE.format(
                         question=q, wrong=w, correct=c)}]
                    for q, w, c in batch
                ]
                try:
                    corrections = batch_generate(llm, SamplingParams, msgs, max_tokens=256)
                    for (q, w, c), corr in zip(batch, corrections):
                        f.write(json.dumps(format_verifier_entry(q, w, c, corr)) + "\n")
                        done += 1
                except Exception as e:
                    print(f"  Correction batch {batch_start} failed: {e}")
                    failed += len(batch)

                elapsed = time.time() - t0
                print(f"  Pass 2: {batch_start + len(batch)}/{len(neg_triples)} "
                      f"({failed} failed) [{elapsed:.0f}s]")

    stats = {"done": done, "failed": failed, "model": args.model}
    (OUT_DIR / "stage5_verifier_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. {done:,} verifier examples → {out_path}")


if __name__ == "__main__":
    main()
