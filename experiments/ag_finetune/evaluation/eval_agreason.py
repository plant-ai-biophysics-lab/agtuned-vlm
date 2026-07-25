"""
Per-stage evaluation on BGLab/AgReason (100 questions) and BGLab/AgThoughts (44k).

AgReason  — quick check after every training stage (~2 min)
AgThoughts — full held-out eval at end of each major stage (~90 min)

Results are saved per checkpoint so you can track improvement across stages.

Usage:
  # Quick check (AgReason only)
  uv run python experiments/ag_finetune/evaluation/eval_agreason.py \
    --model outputs/checkpoints/stage1_sft \
    --stage stage1_sft

  # Full eval (AgReason + AgThoughts)
  uv run python experiments/ag_finetune/evaluation/eval_agreason.py \
    --model outputs/checkpoints/stage1_sft \
    --stage stage1_sft \
    --full

  # Compare all saved results
  uv run python experiments/ag_finetune/evaluation/eval_agreason.py --compare
"""

import argparse
import json
import re
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs" / "eval"

QUESTION_TYPES = [
    "Biotic Diseases Questions",
    "Biotic Insects Questions",
    "Biotic Weeds Questions",
    "Abiotic Soil Questions",
    "Abiotic Weather Questions",
    "Abiotic Harvest Questions",
    "Crop Management Questions",
    "Crop Inputs Questions",
    "Cover Crop Questions",
    "Plant and Seed Health Questions",
]

SYSTEM_PROMPT = (
    "You are an expert agricultural scientist and advisor. "
    "Think through problems step by step before giving your final answer."
)


# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    """Strip <think>...</think> block and return the final answer text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def score_pair(pred: str, ref: str) -> dict:
    pred_clean = extract_final_answer(pred).lower()
    ref_lower  = ref.strip().lower()

    # Key noun/phrase overlap — split ref into meaningful chunks
    ref_words = set(w for w in re.findall(r"\b\w{4,}\b", ref_lower))
    pred_words = set(w for w in re.findall(r"\b\w{4,}\b", pred_clean))
    overlap = len(ref_words & pred_words) / len(ref_words) if ref_words else 0.0

    return {
        "contains_match": int(ref_lower[:80] in pred_clean or pred_clean[:80] in ref_lower),
        "keyword_overlap": round(overlap, 3),
    }


def aggregate_scores(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {}
    contains  = sum(r["contains_match"]  for r in results) / n
    kw_overlap = sum(r["keyword_overlap"] for r in results) / n

    by_type: dict[str, list] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(r["keyword_overlap"])
    per_type = {qt: round(sum(v) / len(v), 3) for qt, v in by_type.items()}

    return {
        "n":               n,
        "contains_match":  round(contains, 3),
        "keyword_overlap": round(kw_overlap, 3),
        "per_question_type": per_type,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, questions: list[str], max_new_tokens: int = 512) -> list[str]:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    preds = []
    for q in questions:
        msgs   = [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": q}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        pred   = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(pred)
    return preds


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_agreason() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("BGLab/AgReason", split="train")
    return [{"question": r["Question"], "answer": r["Answer"],
             "question_type": r["Question Type"]} for r in ds]


def load_agthoughts(max_samples: int | None = None, exclude_questions: set[str] | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("BGLab/AgThoughts", split="train")
    rows = [{"question": r["Question"], "answer": r["Answer"],
             "question_type": r["Question Type"]} for r in ds]
    # Remove questions that appear in AgReason to keep benchmarks independent
    if exclude_questions:
        before = len(rows)
        rows = [r for r in rows if r["question"] not in exclude_questions]
        print(f"  AgThoughts: removed {before - len(rows)} questions overlapping with AgReason")
    if max_samples:
        rows = rows[:max_samples]
    return rows


# ── Eval runner ───────────────────────────────────────────────────────────────

def eval_dataset(
    model, tokenizer,
    rows: list[dict],
    dataset_name: str,
    stage: str,
    max_new_tokens: int = 512,
) -> dict:
    print(f"\n  Running {dataset_name} ({len(rows)} questions)...")
    questions = [r["question"] for r in rows]
    preds     = run_inference(model, tokenizer, questions, max_new_tokens)

    results = []
    for row, pred in zip(rows, preds):
        scores = score_pair(pred, row["answer"])
        results.append({
            "question":      row["question"],
            "reference":     row["answer"],
            "prediction":    pred,
            "question_type": row["question_type"],
            **scores,
        })

    metrics = aggregate_scores(results)
    metrics["stage"]   = stage
    metrics["dataset"] = dataset_name

    print(f"  Contains match:  {metrics['contains_match']:.1%}")
    print(f"  Keyword overlap: {metrics['keyword_overlap']:.1%}")
    print("  Per type:")
    for qt, score in sorted(metrics["per_question_type"].items(), key=lambda x: -x[1]):
        print(f"    {qt:<40} {score:.3f}")

    return {"metrics": metrics, "predictions": results}


# ── Compare across stages ─────────────────────────────────────────────────────

def compare_all() -> None:
    files = sorted(OUT_DIR.glob("*.json"))
    if not files:
        print(f"No eval results found in {OUT_DIR}")
        return

    records: list[dict] = []
    for f in files:
        data = json.loads(f.read_text())
        m = data.get("metrics", {})
        if m:
            records.append(m)

    records.sort(key=lambda r: (r.get("dataset", ""), r.get("stage", "")))

    print(f"\n{'Stage':<25} {'Dataset':<15} {'Contains':>10} {'KW Overlap':>12}")
    print("-" * 65)
    for r in records:
        print(f"  {r.get('stage','?'):<23} {r.get('dataset','?'):<15} "
              f"{r.get('contains_match', 0):>9.1%} {r.get('keyword_overlap', 0):>11.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="outputs/checkpoints/stage1_sft")
    parser.add_argument("--stage",          default="unknown", help="Label for this checkpoint")
    parser.add_argument("--full",           action="store_true", help="Also run AgThoughts (44k)")
    parser.add_argument("--agthoughts_max", type=int, default=None,
                        help="Cap AgThoughts samples (default: all 44k)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--compare",        action="store_true",
                        help="Print comparison table of all saved results and exit")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare_all()
        return

    from unsloth import FastLanguageModel
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    # ── AgReason (always) ────────────────────────────────────────────────────
    agreason_rows = load_agreason()
    ar_out = eval_dataset(model, tokenizer, agreason_rows,
                          "AgReason", args.stage, args.max_new_tokens)
    ar_path = OUT_DIR / f"{args.stage}_agreason.json"
    ar_path.write_text(json.dumps(ar_out, indent=2))
    print(f"\n  Saved → {ar_path}")

    # ── AgThoughts (optional full eval) ──────────────────────────────────────
    if args.full:
        agreason_qs = {r["question"] for r in agreason_rows}
        at_rows = load_agthoughts(args.agthoughts_max, exclude_questions=agreason_qs)
        at_out  = eval_dataset(model, tokenizer, at_rows,
                               "AgThoughts", args.stage, args.max_new_tokens)
        at_path = OUT_DIR / f"{args.stage}_agthoughts.json"
        at_path.write_text(json.dumps(at_out, indent=2))
        print(f"  Saved → {at_path}")

    print("\nDone. Run with --compare to see all stages side by side.")


if __name__ == "__main__":
    main()
