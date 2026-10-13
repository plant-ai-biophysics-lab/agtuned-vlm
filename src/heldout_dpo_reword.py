"""
Rewrites the reasoning PREAMBLE of every `chosen` completion in
build_dpo_unified_data.py's own output JSONL from third-person evaluator
voice into first-person diagnostic voice matching the policy's own natural
completion register. Deliberately NOT done by editing
heldout_explain_miss_prompt/heldout_explain_correct_prompt in place — those
are shared with heldout_summarize.py's skill-memory pipeline, which
genuinely wants that evaluative register for its own advisory bullets; this
operates as a separate, later stage on the DPO data only.

Text-only — no image needed, this is a pure REWRITE of already-written
text, not a re-examination of the image (vision_token_budget=None, same
convention heldout_summarize.py's own text-only map/reduce calls use).

Splits each row's `chosen` field on the LAST "\\n\\n" — the preamble
(everything before) gets reworded, the suffix (everything after — a
classification `<answer>...</answer>` tag, or a detection `box_2d` JSON
array) is passed through UNCHANGED, since neither of those can be
paraphrased without breaking their format.

Usage:
    uv run python main.py heldout-dpo-reword \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --input outputs/dpo_unified_pairs.jsonl \\
        --output outputs/dpo_unified_pairs_reworded.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_thinking
from src.common.vlm_load import destroy_llm, load_llm
from src.common.progress import ProgressBar

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def split_chosen(chosen: str) -> tuple[str, str]:
    """(preamble, suffix) — suffix is everything after the LAST blank-line
    break (the <answer> tag for classification, or the box_2d JSON array
    for detection), passed through untouched; preamble is the reasoning
    text that gets reworded. A chosen field with no "\\n\\n" at all (should
    not happen given build_dpo_unified_data.py's own construction) returns
    (chosen, "") — the whole thing gets reworded, nothing preserved."""
    if "\n\n" not in chosen:
        return chosen, ""
    preamble, _, suffix = chosen.rpartition("\n\n")
    return preamble, suffix


def run_model(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.heldout_prompts))
    template = hprompts["heldout_dpo_chosen_rewrite_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"ERROR: {args.input} has no rows.")

    tasks_to_reword = set(args.tasks.split(",")) if args.tasks else None
    pending_idx = [
        i for i, r in enumerate(rows)
        if (tasks_to_reword is None or r["task"] in tasks_to_reword)
    ]
    print(f"  {len(rows)} total row(s), {len(pending_idx)} to reword "
          f"(tasks filter: {tasks_to_reword or 'all'})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.with_suffix(".cache.json")
    if args.rerun and cache_path.exists():
        cache_path.unlink()
    cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # Cache key: row index as a string — input JSONL order is deterministic
    # given the same upstream checkpoints, same resumability convention as
    # heldout_summarize.py's own chunk_summaries.json cache.
    pending = [i for i in pending_idx if str(i) not in cache]
    print(f"  [resume] {len(cache)} already reworded, {len(pending)} pending")

    if pending:
        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=None,  # text-only — no image needed for a pure rewrite
            kv_cache_dtype=args.kv_cache_dtype,
            caller_tag="heldout_dpo_reword",
        )
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=args.gen_max_tokens, seed=args.seed,
            skip_special_tokens=True,
        )
        pbar = ProgressBar(total=len(pending), desc="Rewording")
        for i in pending:
            preamble, _suffix = split_chosen(rows[i]["chosen"])
            text = template.format(analysis=preamble)
            hist = [{"role": "user", "content": text}]
            prompt = render_chat_template(tokenizer, hist, enable_thinking=False, chat_template=chat_template)
            out = llm.generate([prompt], sampling_params=gen_params)[0]
            raw = out.outputs[0].text
            _, cleaned = extract_thinking(raw)
            cache[str(i)] = cleaned.strip()
            if len(cache) % args.save_every == 0:
                cache_path.write_text(json.dumps(cache))
            pbar.update(1)
        cache_path.write_text(json.dumps(cache))
        destroy_llm(llm)

    n_reworded = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            out_row = dict(row)
            key = str(i)
            if key in cache:
                _preamble, suffix = split_chosen(row["chosen"])
                out_row["chosen"] = f"{cache[key]}\n\n{suffix}" if suffix else cache[key]
                n_reworded += 1
            f.write(json.dumps(out_row) + "\n")

    print(f"\n  Reworded {n_reworded}/{len(rows)} row(s) (rest passed through unchanged)")
    print(f"  Saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--input", default="outputs/dpo_unified_pairs.jsonl",
                         help="Output of build_dpo_unified_data.py.")
    parser.add_argument("--output", default="outputs/dpo_unified_pairs_reworded.jsonl")
    parser.add_argument("--tasks", default="disease_pest,quality,species",
                         help="Comma-separated task tags to reword — classification only by default. "
                              "Pass e.g. 'disease_pest,quality,species,detection' to also reword "
                              "detection's collated-explanation preamble.")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=400,
                         help="A 2-4 sentence rewrite needs far less than explain-traces' own budget.")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192,
                         help="Text-only, short input/output — far below the vision stages' own "
                              "16384-24576 range.")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)

    main(parser.parse_args())
