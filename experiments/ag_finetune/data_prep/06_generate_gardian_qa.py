"""
Stage 2b — Generate Q&A pairs from CGIAR document passages (vLLM).

Reads outputs/stage3_gardian_passages.jsonl (produced by 03_chunk_gardian.py)
and prompts a local Qwen model to generate 2–3 question-answer pairs per passage.

Outputs:
  outputs/stage6_gardian_qa.jsonl
  outputs/stage6_gardian_qa_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/06_generate_gardian_qa.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --n_qa 2 \
    --batch_size 32

  # Cap samples for a quick test
  uv run python experiments/ag_finetune/data_prep/06_generate_gardian_qa.py \
    --model Qwen/Qwen3.5-122B-A10B \
    --n_samples 500
"""

import argparse
import json
import re
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"

SYSTEM_PROMPT = (
    "You are an expert agricultural scientist. "
    "Given a passage from an agricultural research document, generate {n} clear "
    "question-answer pairs that test understanding of specific facts in the passage. "
    "Questions must be answerable from the passage alone — do not require outside knowledge. "
    "Each answer should be 1–3 sentences. "
    "Return ONLY a JSON array: "
    '[{{"question": "...", "answer": "..."}}, ...]'
)

USER_TEMPLATE = "Passage:\n{passage}\n\nGenerate {n} question-answer pairs:"


def build_messages(passage: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
        {"role": "user",   "content": USER_TEMPLATE.format(passage=passage[:2000], n=n)},
    ]


def parse_qa_list(text: str) -> list[dict]:
    """Extract JSON Q&A list from model output, tolerating surrounding prose."""
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group())
        return [
            {"question": str(i.get("question", "")).strip(),
             "answer":   str(i.get("answer",   "")).strip()}
            for i in items
            if isinstance(i, dict)
            and i.get("question", "").strip()
            and i.get("answer",   "").strip()
        ]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(OUT_DIR / "stage3_gardian_passages.jsonl"))
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="Limit passages to process (default: all)")
    parser.add_argument("--n_qa",         type=int, default=2,
                        help="Q&A pairs to generate per passage")
    parser.add_argument("--model",        default="Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--quantization", default=None,
                        help="awq, awq_marlin, gptq, bitsandbytes, or None for bf16")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_tokens",   type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=3072)
    parser.add_argument("--gpu_util",     type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: pip install vllm")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Run 03_chunk_gardian.py first. Missing: {input_path}")
        return

    passages = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    if args.n_samples:
        passages = passages[: args.n_samples]
    print(f"Loaded {len(passages):,} passages from {input_path.name}")

    # Resume via _idx
    out_path = OUT_DIR / "stage6_gardian_qa.jsonl"
    done_ids: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["_idx"])
                except Exception:
                    pass
        if done_ids:
            print(f"  Resuming: {len(done_ids)} passages already processed")

    pending = [(i, p) for i, p in enumerate(passages) if i not in done_ids]
    if not pending:
        print("All passages already processed.")
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
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.4, top_p=0.9)
    print("Model loaded.\n")

    done_passages = len(done_ids)
    total_qa = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "a") as f:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            messages_batch = [build_messages(p["passage"], args.n_qa) for _, p in batch]

            try:
                outputs = llm.chat(messages_batch, sampling_params=sampling, use_tqdm=False)
            except Exception as e:
                print(f"  Batch {batch_start} failed: {e}")
                failed += len(batch)
                continue

            for (idx, passage), output in zip(batch, outputs):
                pairs = parse_qa_list(output.outputs[0].text)
                if not pairs:
                    failed += 1
                    continue
                for pair in pairs:
                    entry = {
                        "_idx":     idx,
                        "stage":    "rag_qa",
                        "source":   passage.get("source", ""),
                        "chunk_id": passage.get("chunk_id", ""),
                        "title":    passage.get("title", ""),
                        "passage":  passage["passage"],
                        "question": pair["question"],
                        "answer":   pair["answer"],
                    }
                    f.write(json.dumps(entry) + "\n")
                    total_qa += 1
                done_passages += 1

            elapsed = time.time() - t0
            rate = (done_passages - len(done_ids)) / elapsed if elapsed > 0 else 0
            eta_h = (len(pending) - (done_passages - len(done_ids))) / rate / 3600 if rate > 0 else 0
            print(f"  {done_passages}/{len(passages)} passages  →  {total_qa} Q&A pairs  "
                  f"({failed} failed)  [{rate:.1f} pass/s  ETA {eta_h:.1f}h]")

    stats = {
        "passages_processed": done_passages,
        "qa_pairs_generated": total_qa,
        "failed_passages":    failed,
        "model":              args.model,
        "n_qa_per_passage":   args.n_qa,
    }
    (OUT_DIR / "stage6_gardian_qa_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDone. {total_qa:,} Q&A pairs from {done_passages:,} passages → {out_path}")


if __name__ == "__main__":
    main()
