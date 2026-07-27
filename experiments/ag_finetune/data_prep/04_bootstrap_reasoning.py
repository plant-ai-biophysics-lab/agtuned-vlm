"""
Stage 2a — Bootstrap reasoning chains via local Qwen3.5 (vLLM).

Reads outputs/stage1_agrillm_filtered.jsonl and generates a CoT reasoning
chain for each Q&A pair using a local model served via vLLM.

Outputs:
  outputs/stage4_reasoning_chains.jsonl   — (question, answer, chain) triples
  outputs/stage4_reasoning_stats.json

Usage:
  # Full run on 80GB GPU
  uv run python experiments/ag_finetune/data_prep/04_bootstrap_reasoning.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --n_samples 5000 \
    --batch_size 64

  # Smaller model (fits more easily, faster)
  uv run python experiments/ag_finetune/data_prep/04_bootstrap_reasoning.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --quantization bitsandbytes \
    --n_samples 5000

  # Pre-quantized AWQ variant (recommended for 122B on 80GB)
  uv run python experiments/ag_finetune/data_prep/04_bootstrap_reasoning.py \
    --model Qwen/Qwen3.5-122B-A10B-AWQ \
    --quantization awq_marlin
"""

import argparse
import json
import time
from pathlib import Path

from _progress import fmt_duration, stage_banner

OUT_DIR = Path(__file__).parent.parent / "outputs"

SYSTEM_PROMPT = (
    "You are an expert agricultural scientist and advisor. "
    "When given a farming question and its correct answer, "
    "write a concise step-by-step reasoning chain (3–6 steps) that logically "
    "arrives at that answer. Be specific about agricultural mechanisms. "
    "End with: 'Answer: <answer>'"
)

USER_TEMPLATE = """\
Question: {question}

Correct answer: {answer}

Write the reasoning chain:"""


def build_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, answer=answer)},
    ]


def main() -> None:
    stage_banner(4, "Bootstrap CoT reasoning chains")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",     type=int,   default=5000)
    parser.add_argument("--model",         default="Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--quantization",  default=None,
                        help="awq, awq_marlin, gptq, bitsandbytes, or None for bf16")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--max_tokens",    type=int,   default=512)
    parser.add_argument("--max_model_len", type=int,   default=4096)
    parser.add_argument("--gpu_util",      type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (multi-GPU setups)")
    parser.add_argument("--input",         default=str(OUT_DIR / "stage1_agrillm_filtered.jsonl"))
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: pip install vllm")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Run 01_filter_agrillm.py first. Missing: {input_path}")
        return

    rows = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    rows = rows[: args.n_samples]
    print(f"Loaded {len(rows):,} samples from {input_path.name}")

    # Resume from checkpoint — track by _idx field
    out_path = OUT_DIR / "stage4_reasoning_chains.jsonl"
    done_ids: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["_idx"])
                except Exception:
                    pass
        if done_ids:
            print(f"  Resuming: {len(done_ids)} already done, {len(rows) - len(done_ids)} remaining")

    pending = [(i, r) for i, r in enumerate(rows) if i not in done_ids]
    if not pending:
        print("All samples already generated.")
        return

    print(f"Loading {args.model} (quantization={args.quantization or 'none'})...")
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.3, top_p=0.9)
    print("Model loaded.\n")

    done, failed = len(done_ids), 0
    t0 = time.time()

    with open(out_path, "a") as f:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            messages_batch = [build_messages(r["question"], r["answer"]) for _, r in batch]

            try:
                outputs = llm.chat(messages_batch, sampling_params=sampling, use_tqdm=False)
            except Exception as e:
                print(f"  Batch {batch_start}–{batch_start + len(batch)} failed: {e}")
                failed += len(batch)
                continue

            for (idx, row), output in zip(batch, outputs):
                chain = output.outputs[0].text.strip()
                entry = {
                    "_idx": idx,
                    "stage": "sft_reasoning",
                    "source": row.get("source", ""),
                    "system": row.get("system", ""),
                    "question": row["question"],
                    "answer": row["answer"],
                    "chain": chain,
                }
                f.write(json.dumps(entry) + "\n")
                done += 1

            elapsed = time.time() - t0
            n_done_this_run = done - len(done_ids)
            rate = n_done_this_run / elapsed if elapsed > 0 else 0
            eta_s = (len(pending) - n_done_this_run) / rate if rate > 0 else 0
            pct = 100 * done / len(rows)
            print(f"  [chains] {done:,}/{len(rows):,} ({pct:.1f}%) done  ({failed:,} failed)  "
                  f"| elapsed {fmt_duration(elapsed)} | rate {rate:.1f}/s "
                  f"| ETA {fmt_duration(eta_s)}")

    stats = {
        "done":      done,
        "failed":    failed,
        "model":     args.model,
        "n_samples": len(rows),
    }
    (OUT_DIR / "stage4_reasoning_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone in {fmt_duration(time.time() - t0)}. {done:,} chains → {out_path}")


if __name__ == "__main__":
    main()
