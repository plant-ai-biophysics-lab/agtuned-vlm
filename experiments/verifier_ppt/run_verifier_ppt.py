"""
Verifier-PPT feasibility test — can the verifier's confidence score be
replaced with a logprob-derived fine-grained score, the way
https://github.com/llm-as-a-verifier/llm-as-a-verifier does it?

This is deliberately narrow, matching the actual open question: not "build
the full PPT pipeline" (that already exists, see src/verifier_lmv.py, which
does pairwise N-candidate tournament ranking with criteria decomposition) —
just "does a SINGLE-statement, SINGLE-round verifier reliably emit a scoreable
letter token, and does the resulting score distribution look reasonable?"

What this deliberately does NOT do, on purpose:
  - No pairwise comparison (no candidate A vs candidate B) — verifier_cot's
    existing design verifies ONE statement at a time, not a ranked list.
  - No criteria decomposition (the paper's C axis) — one holistic verifier
    judgment per statement, same as verifier_cot's existing prompt.
  - No multi-round loop — one VLM statement (self_cot-style opening turn),
    one verification, done. If the scoring mechanism looks sound, extending
    it back into verifier_cot's multi-round loop is a natural follow-up, not
    this script's job.
  - No classification / no judging against ground truth — the only question
    here is "does scoring work at all," not "is this condition more accurate."

What it DOES test, per statement, K times ("repeated verifications"):
  - Does the model emit a <score>LETTER</score> tag the token-scanner can
    actually find? (extraction success rate, logged explicitly)
  - Fine-grained score: expected value over the logprob distribution at that
    tag position (llm_verifier.fine_grained_reward.extract_score — the exact
    function src/verifier_lmv.py's PPT tournament already uses successfully
    in this repo, just with a single <score> tag instead of pairwise
    <score_A>/<score_B>).
  - Naive score: literal text value of the letter actually written, with no
    logprob weighting — the number you'd get from parsing verifier_cot's old
    "confidence": <int> field, for direct comparison against the fine-grained
    version on the exact same statements.

Verifier prompt: experiments/verifier_ppt/verifier_prompt.yaml — a copy of
verifier_cot's template with the confidence integer replaced by the letter tag.

Usage:
  uv run python experiments/verifier_ppt/run_verifier_ppt.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_ppt/ \\
    --n_datasets 1 --n_samples 10 --n_evals 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "verifier_cot"))

from PIL import Image as PILImage

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

try:
    from llm_verifier.fine_grained_reward import SCALE, extract_score
    _HAS_LMV = True
except ImportError:
    _HAS_LMV = False
    SCALE = extract_score = None

from src.common.progress import Progress, stage_banner, sub_banner

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_SCORE_TAG_RE = re.compile(r"<score>\s*(.+?)\s*</score>", re.IGNORECASE)

CSV_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "vlm_statement", "verdict", "verifier_response", "raw_text_sample",
    "n_evals", "n_logprob_hits", "n_text_fallback", "n_extraction_failed",
    "fine_grained_score", "naive_score",
    "raw_evals_json",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def _load_verifier_prompts() -> dict:
    import yaml
    with open(Path(__file__).parent / "verifier_prompt.yaml") as f:
        return yaml.safe_load(f)


_VERIFIER_PROMPTS = _load_verifier_prompts()


def _verifier_system(rubric: str) -> str:
    return _VERIFIER_PROMPTS["verifier_system"].replace("{rubric}", rubric)


def _verifier_user_turn(verified_history: str, statement: str) -> str:
    return (
        _VERIFIER_PROMPTS["verifier_user_turn"]
        .replace("{verified_history}", verified_history or "(none yet)")
        .replace("{statement}", statement)
    )


def _extract_first_json_object(text: str) -> dict | None:
    """Same balanced-brace scan as run_verifier_cot.py — duplicated (not
    imported) to keep this experiment self-contained per repo convention."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
    return None


def _vllm_to_score_format(out) -> tuple[str, list[str], list[list[tuple]]]:
    """Convert a vLLM RequestOutput into the (text, tokens, position_logprobs)
    shape llm_verifier.fine_grained_reward.extract_score expects. Identical
    logic to src/verifier_lmv.py's helper of the same name — duplicated, not
    imported, since verifier_lmv.py isn't designed as a shared library module."""
    text = out.outputs[0].text
    token_ids = out.outputs[0].token_ids
    logprobs_list = out.outputs[0].logprobs or []

    tokens: list[str] = []
    position_logprobs: list[list[tuple]] = []
    for i, tid in enumerate(token_ids):
        lp_dict = logprobs_list[i] if i < len(logprobs_list) else {}
        tok_str = lp_dict[tid].decoded_token if (lp_dict and tid in lp_dict) else ""
        tokens.append(tok_str)
        alts = [(lp.decoded_token, lp.logprob) for lp in (lp_dict or {}).values()]
        position_logprobs.append(alts)
    return text, tokens, position_logprobs


def naive_score_from_text(text: str) -> float | None:
    """Literal text value of the letter actually written, no logprob
    weighting — what you'd get parsing the old confidence field's approach.
    Returns None if no <score> tag/valid letter found at all."""
    m = _SCORE_TAG_RE.search(text or "")
    if not m:
        return None
    tok = m.group(1).strip()
    valid = SCALE["valid_tokens"]
    raw_val = valid.get(tok)
    if raw_val is None:
        for vt, val in valid.items():
            if tok.lower() == vt.lower():
                raw_val = val
                break
    if raw_val is None:
        return None
    vals = sorted(set(valid.values()))
    lo, hi = min(vals), max(vals)
    return (raw_val - lo) / (hi - lo) if hi > lo else 0.5


def parse_verifier_output(out) -> dict:
    """One K-repeat's full result: verdict/response (JSON) + both score variants."""
    text, tokens, position_logprobs = _vllm_to_score_format(out)
    clean_text = _THINK_RE.sub("", text).strip()

    data = _extract_first_json_object(clean_text) or {}
    verdict = str(data.get("verdict", "UNSUPPORTED")).strip().upper()
    if verdict not in ("SUPPORTED", "UNSUPPORTED"):
        verdict = "UNSUPPORTED"
    response = str(data.get("response", ""))[:500]

    # Did we actually find usable logprobs at the tag, or fall back?
    from llm_verifier.fine_grained_reward import _find_tag_logprobs
    tag_lp = _find_tag_logprobs(tokens, position_logprobs, "<score>")
    had_logprob_hit = bool(tag_lp) and any(
        tok.strip() in SCALE["valid_tokens"] for tok, _ in tag_lp
    )

    fine_grained = extract_score(clean_text, tokens, position_logprobs, "<score>")
    naive = naive_score_from_text(clean_text)

    return {
        "verdict": verdict,
        "response": response,
        "raw_text": clean_text,  # full generated text, unstripped by the 500-char JSON
                                 # response truncation — lets you check what the model
                                 # actually wrote when JSON parsing or score extraction
                                 # didn't behave as expected, instead of only ever seeing
                                 # the parser's (possibly wrong) interpretation of it
        "fine_grained_score": fine_grained,
        "naive_score": naive,
        "extraction": "logprob" if had_logprob_hit else ("text_fallback" if naive is not None else "failed"),
    }


def run_feasibility_test(
    items: list[dict],
    llm,
    tokenizer,
    n_evals: int = 4,
    batch_size: int = 64,
    ckpt_path: Path | None = None,
) -> list[dict]:
    from src.verifier_vision import _RUBRIC_DISEASE, _RUBRIC_WEED
    from src.cot_vision import _sys_msg, _cls_question, _desc_question, _build_prompt, _clean

    _VLM_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _VER_P = SamplingParams(temperature=1.0, max_tokens=150, logprobs=20)

    n = len(items)

    # ── Round 1: initial free-form description (self_cot-style, same as verifier_cot) ──
    sub_banner(f"Generating statements ({n} images)")
    statements: list[str] = [""] * n
    prog = Progress(total=n, label="Statement generation", every=max(batch_size, 1))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        prompts = []
        for i in batch_idx:
            item = items[i]
            desc_q = _desc_question(item["task_type"], item["crop"])
            h = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc_q},
            ]}]
            prompts.append(_build_prompt(h, tokenizer, PILImage.open(item["image_path"]).convert("RGB")))
        outputs = llm.generate(prompts, _VLM_P)
        for i, out in zip(batch_idx, outputs):
            statements[i] = _clean(out.outputs[0].text)
        del prompts
        prog.update(len(batch_idx))

    # ── Verify each statement K times ───────────────────────────────────────
    sub_banner(f"Verifying ({n} statements x {n_evals} repeated evaluations = {n * n_evals} calls)")
    all_evals: list[list[dict]] = [[] for _ in range(n)]
    total_calls = n * n_evals
    prog = Progress(total=total_calls, label="Verification", every=max(batch_size, 1))
    for k in range(n_evals):
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            prompts = []
            for i in batch_idx:
                item = items[i]
                rubric = _RUBRIC_WEED if item["task_type"] == "weed_identification" else _RUBRIC_DISEASE
                sys_p = _verifier_system(rubric)
                user_p = _verifier_user_turn("(none yet)", statements[i])
                msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": user_p},
                ]}]
                prompts.append(_build_prompt(msgs, tokenizer, PILImage.open(item["image_path"]).convert("RGB")))
            outputs = llm.generate(prompts, _VER_P)
            del prompts
            for i, out in zip(batch_idx, outputs):
                all_evals[i].append(parse_verifier_output(out))
            prog.update(len(batch_idx))

    # ── Aggregate per statement ──────────────────────────────────────────────
    rows = []
    for i, item in enumerate(items):
        evals = all_evals[i]
        fg_scores = [e["fine_grained_score"] for e in evals if e["fine_grained_score"] is not None]
        nv_scores = [e["naive_score"] for e in evals if e["naive_score"] is not None]
        n_logprob = sum(1 for e in evals if e["extraction"] == "logprob")
        n_fallback = sum(1 for e in evals if e["extraction"] == "text_fallback")
        n_failed = sum(1 for e in evals if e["extraction"] == "failed")
        majority_verdict = max(set(e["verdict"] for e in evals), key=lambda v: sum(1 for e in evals if e["verdict"] == v))
        rows.append({
            "agml_dataset": item["ds"],
            "class_label":  item["cls"],
            "task_type":    item["task_type"],
            "crop":         item.get("crop") or "",
            "image_name":   item["image_name"],
            "vlm_statement": statements[i],
            "verdict":      majority_verdict,
            "verifier_response": evals[0]["response"] if evals else "",
            "raw_text_sample": evals[0]["raw_text"] if evals else "",
            "n_evals":      len(evals),
            "n_logprob_hits": n_logprob,
            "n_text_fallback": n_fallback,
            "n_extraction_failed": n_failed,
            "fine_grained_score": sum(fg_scores) / len(fg_scores) if fg_scores else None,
            "naive_score": sum(nv_scores) / len(nv_scores) if nv_scores else None,
            "raw_evals_json": json.dumps(evals),
        })

    if ckpt_path:
        _write_csv(rows, ckpt_path)
    return rows


def make_charts(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fg = [r["fine_grained_score"] for r in rows if r["fine_grained_score"] is not None]
    nv = [r["naive_score"] for r in rows if r["naive_score"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = [i / 20 for i in range(21)]
    ax1.hist(fg, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title(f"Fine-grained score (logprob-weighted) — n={len(fg)}")
    ax1.set_xlabel("Score [0,1] (1 = fully supported)")
    ax1.set_ylabel("Number of statements")
    ax1.set_xlim(0, 1)
    ax1.yaxis.grid(True, alpha=0.3)

    ax2.hist(nv, bins=bins, color="#C44E52", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title(f"Naive score (literal letter parsed) — n={len(nv)}")
    ax2.set_xlabel("Score [0,1] (1 = fully supported)")
    ax2.set_ylabel("Number of statements")
    ax2.set_xlim(0, 1)
    ax2.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    p = output_dir / "score_distribution_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

    total_evals = sum(r["n_evals"] for r in rows)
    total_logprob = sum(r["n_logprob_hits"] for r in rows)
    total_fallback = sum(r["n_text_fallback"] for r in rows)
    total_failed = sum(r["n_extraction_failed"] for r in rows)

    import numpy as np
    summary = {
        "n_statements": len(rows),
        "total_evals": total_evals,
        "logprob_hit_rate": total_logprob / total_evals if total_evals else None,
        "text_fallback_rate": total_fallback / total_evals if total_evals else None,
        "extraction_failed_rate": total_failed / total_evals if total_evals else None,
        "fine_grained_mean": float(np.mean(fg)) if fg else None,
        "fine_grained_std": float(np.std(fg)) if fg else None,
        "naive_mean": float(np.mean(nv)) if nv else None,
        "naive_std": float(np.std(nv)) if nv else None,
    }
    (output_dir / "score_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"  Saved: {output_dir / 'score_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier-PPT feasibility test — fine-grained logprob scoring")
    parser.add_argument("--model",       required=True)
    parser.add_argument("--output_dir",  default="outputs/verifier_ppt/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_samples",   type=int, default=10, help="Max images per class")
    parser.add_argument("--n_datasets",  type=int, default=1, help="Datasets per task (kept small — feasibility test)")
    parser.add_argument("--n_evals",     type=int, default=4, help="K: repeated verifications per statement")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--plot_only",   action="store_true")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    if not _HAS_LMV:
        print("ERROR: llm-verifier package not installed. Run: uv add llm-verifier")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "ckpt_ppt_scores.csv"

    if ckpt_path.exists() and not (args.plot_only or args.overwrite):
        print(f"ERROR: {ckpt_path} already exists. Pass --overwrite to replace it, "
              f"or --plot_only to just re-chart it.")
        sys.exit(1)

    if args.plot_only:
        stage_banner(1, "Plot only", total_stages=1)
        rows = list(csv.DictReader(open(ckpt_path, encoding="utf-8")))
        make_charts(rows, output_dir)
        print("\nDone.")
        return

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    test_split = None
    ts_path = Path(args.test_split)
    if ts_path.exists():
        test_split = json.loads(ts_path.read_text())
        print(f"Loaded test_split: {ts_path} ({len(test_split)} datasets)")

    from src.verifier_vision import select_datasets, load_items
    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed, config_path=args.config)
    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")

    all_items: list[dict] = []
    for task, ds_names in selected.items():
        for ds in ds_names:
            info = ds_info.get(ds, {})
            crop = info.get("crop", "plant")
            class_overrides = info.get("class_overrides", {})
            items = load_items(ds, task, crop, n_samples=args.n_samples,
                               class_overrides=class_overrides, seed=args.seed,
                               test_split=test_split)
            all_items.extend(items)
            print(f"  {ds} ({task}): {len(items)} items")
    print(f"Total items: {len(all_items)}")

    stage_banner(1, f"Verifier-PPT feasibility test ({len(all_items)} images, K={args.n_evals})",
                total_stages=2)

    from src.cot_vision import load_llm, destroy_llm
    print(f"\n=== Loading model: {args.model} ===")
    llm, tokenizer = load_llm(args.model)

    rows = run_feasibility_test(all_items, llm, tokenizer, n_evals=args.n_evals,
                                batch_size=args.batch_size, ckpt_path=ckpt_path)

    destroy_llm(llm)
    del tokenizer, llm
    import gc
    gc.collect()

    stage_banner(2, "Score distribution + summary", total_stages=2)
    make_charts(rows, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
