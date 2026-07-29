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

Scoring mechanism — two-pass prefill (mirrors llm_verifier's own
_score_tags_by_prefill, not a single free-generation call):
  Pass 1 generates the response JSON only (temperature=1.0,
  max_tokens=150, no logprobs needed — we don't read this call's tag).
  Pass 2 continues that EXACT prefix with "<score>" appended and the very
  next token constrained via SamplingParams(structured_outputs=
  StructuredOutputsParams(choice=_SCORE_LETTERS)) to just the 20 scale
  letters (+ their leading-space tokenization variants). The resulting
  top-20 logprobs are then a distribution already renormalized over the
  scale itself, not competing against the model's full vocabulary.
  This exists because an earlier single-call version (ask for the tag inline
  and read whatever top-20 logprobs happened to land at that free-generation
  position) collapsed onto "A" 95.8% of the time on a real 160-statement run
  — the model's unconstrained top-20 at that position simply wasn't
  dominated by the 20 target letters. Confirmed against vLLM 0.20.1's
  source (this repo's pinned version) that StructuredOutputsParams.choice
  coexists with logprobs + max_tokens=1 before building this.

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
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "verifier_cot"))

from PIL import Image as PILImage

try:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    StructuredOutputsParams = None  # type: ignore[assignment,misc]

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
    "vlm_statement", "prediction", "ground_truth", "judged_correct", "judge_reasoning",
    "verdict", "verifier_response", "raw_text_sample",
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


# The 20-letter scale plus a leading-space variant of each letter — mirrors
# llm_verifier's own _score_tags_by_prefill comment: "Some models (e.g. Qwen
# VL) put nearly all their mass on the letter WITH a leading space after the
# prefilled '>'; allow both spellings so the grammar mask keeps the real
# distribution." Constrains pass 2's single generated token to exactly these
# 40 candidates via SamplingParams(structured_outputs=...).
_SCORE_LETTERS = [chr(65 + i) for i in range(20)] + [f" {chr(65 + i)}" for i in range(20)]


def _build_score_prompt(base_prompt: dict, analysis: str) -> dict:
    """Continue `base_prompt`'s exact text with the verifier's own analysis
    plus a prefilled '<score>' tag — the offline-vLLM equivalent of the
    OpenAI-server's continue_final_message=True, since _build_prompt already
    stops right where the assistant turn begins (add_generation_prompt=True)."""
    return {
        "prompt": base_prompt["prompt"] + analysis + "\n<score>",
        "multi_modal_data": base_prompt["multi_modal_data"],
    }


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


def parse_verifier_output_prefill(analysis: str, second_out) -> dict:
    """One K-repeat's full result under the two-pass prefill scheme (see
    module docstring). `analysis` is pass 1's cleaned text up to (not
    including) the '<score>' tag; `second_out` is pass 2's single
    grammar-constrained token generated right after a prefilled '<score>'."""
    data = _extract_first_json_object(analysis) or {}
    response = str(data.get("response", ""))[:500]

    letter_raw = second_out.outputs[0].text
    logprobs_list = second_out.outputs[0].logprobs or []
    lp_dict = logprobs_list[0] if logprobs_list else {}
    alts = [(lp.decoded_token, lp.logprob) for lp in lp_dict.values()] if lp_dict \
        else [(letter_raw, 0.0)]

    # Reconstruct the (text, tokens, position_logprobs) shape extract_score
    # expects, matching llm_verifier's own _score_tags_by_prefill return
    # convention exactly, so the shared library function needs no changes.
    synth_text = f"{analysis}\n<score>{letter_raw}</score>"
    synth_tokens = ["\n<score>", letter_raw, "</score>"]
    synth_position_logprobs = [[("\n<score>", 0.0)], alts, [("</score>", 0.0)]]

    had_logprob_hit = any(tok.strip() in SCALE["valid_tokens"] for tok, _ in alts)
    fine_grained = extract_score(synth_text, synth_tokens, synth_position_logprobs, "<score>")
    naive = naive_score_from_text(synth_text)

    # Renormalized probability actually assigned to each candidate letter
    # after grammar masking — kept explicitly (not just the final expected
    # value) so a real distribution can be inspected directly instead of
    # inferred from aggregate stats. Only the valid-letter subset of `alts`,
    # since off-candidate tokens shouldn't appear under the mask anyway.
    letter_probs = {}
    for tok, logprob in alts:
        t = tok.strip()
        if t in SCALE["valid_tokens"]:
            letter_probs[t] = max(letter_probs.get(t, 0.0), math.exp(logprob))
    total_p = sum(letter_probs.values()) or 1.0
    score_alts = sorted(
        [(t, p / total_p) for t, p in letter_probs.items()],
        key=lambda x: -x[1],
    )

    # No separate verdict is asked for in the prompt anymore (see
    # verifier_prompt.yaml's header comment on why) — derive one from the
    # score itself, thresholded at the scale's midpoint, purely so the
    # existing CSV "verdict" column / majority-verdict aggregation keep working.
    verdict = "SUPPORTED" if fine_grained >= 0.5 else "UNSUPPORTED"

    return {
        "verdict": verdict,
        "response": response,
        "raw_text": synth_text,  # analysis + the prefilled tag/letter, so you can see
                                 # exactly what pass 2 was constrained to continue from
        "score_alts": score_alts,  # [(letter, renormalized_prob), ...] sorted desc —
                                   # the actual post-mask distribution behind fine_grained_score
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

    # 512 not 256 — this model consistently writes longer multi-section
    # descriptions than 256 tokens allows (same truncation issue found
    # elsewhere in this session: verifier_cot's response field, the
    # ag_finetune scripts). A truncated statement gets legitimately flagged
    # as "incomplete" by the verifier, which is a real but different signal
    # than genuine diagnostic disagreement — worth removing as a confound
    # before drawing conclusions from the score distribution.
    _VLM_P = SamplingParams(temperature=0.0, max_tokens=512, repetition_penalty=1.1)
    # Pass 1: response JSON only — no logprobs needed, we never read
    # this call's own (unconstrained) attempt at the <score> tag.
    _VER_P = SamplingParams(temperature=1.0, max_tokens=150)
    # Pass 2: one grammar-constrained token continuing "<analysis>\n<score>",
    # restricted to the 20 scale letters (+ leading-space variants) — see
    # module docstring for why this replaces reading the free-generation
    # top-20 logprobs directly.
    _SCORE_P = SamplingParams(temperature=1.0, max_tokens=1, logprobs=20,
                              structured_outputs=StructuredOutputsParams(choice=_SCORE_LETTERS))

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
            base_prompts = []
            for i in batch_idx:
                item = items[i]
                rubric = _RUBRIC_WEED if item["task_type"] == "weed_identification" else _RUBRIC_DISEASE
                sys_p = _verifier_system(rubric)
                user_p = _verifier_user_turn("(none yet)", statements[i])
                msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": user_p},
                ]}]
                base_prompts.append(_build_prompt(msgs, tokenizer, PILImage.open(item["image_path"]).convert("RGB")))

            first_outputs = llm.generate(base_prompts, _VER_P)

            analyses = []
            score_prompts = []
            for bp, out in zip(base_prompts, first_outputs):
                clean_text = _THINK_RE.sub("", out.outputs[0].text).strip()
                idx = clean_text.lower().find("<score>")
                analysis = clean_text[:idx].rstrip() if idx != -1 else clean_text
                analyses.append(analysis)
                score_prompts.append(_build_score_prompt(bp, analysis))
            del base_prompts

            second_outputs = llm.generate(score_prompts, _SCORE_P)
            del score_prompts

            for i, analysis, second_out in zip(batch_idx, analyses, second_outputs):
                all_evals[i].append(parse_verifier_output_prefill(analysis, second_out))
            prog.update(len(batch_idx))

    # ── Final classification, so we have something to correlate scores against ──
    # (matches self_cot/verifier_cot's ending: same statement history + the
    # standard classification question, so accuracy is comparable to those
    # conditions and not just an internal-only signal.)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)
    sub_banner(f"Final classification ({n} images)")
    predictions: list[str] = [""] * n
    prog = Progress(total=n, label="Classification", every=max(batch_size, 1))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        prompts = []
        for i in batch_idx:
            item = items[i]
            desc_q = _desc_question(item["task_type"], item["crop"])
            h = _sys_msg() + [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": desc_q}]},
                {"role": "assistant", "content": statements[i]},
                {"role": "user", "content": _cls_question(item["task_type"], item["crop"])},
            ]
            prompts.append(_build_prompt(h, tokenizer, PILImage.open(item["image_path"]).convert("RGB")))
        outputs = llm.generate(prompts, _CLS_P)
        for i, out in zip(batch_idx, outputs):
            predictions[i] = _clean(out.outputs[0].text)
        del prompts
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
            "prediction":   predictions[i],
            "ground_truth": item["cls"],
            "judged_correct": "",
            "judge_reasoning": "",
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

    # ── Judge predictions against ground truth (same shared judge as every
    # other condition in this repo, so accuracy numbers are comparable) ──────
    from src.common.metrics import apply_judge
    import yaml as _yaml
    _p = _yaml.safe_load((Path(__file__).parent.parent.parent / "src" / "config" / "prompts.yaml").read_text())
    sub_banner(f"Judging ({len(rows)} predictions)")
    rows = apply_judge(
        rows, llm, tokenizer,
        system_prompt=_p["judge_system_vision"],
        judge_template=_p["judge_classification_template"],
        gt_key="ground_truth", pred_key="prediction",
    )

    if ckpt_path:
        _write_csv(rows, ckpt_path)
    return rows


def _as_float(v) -> float | None:
    """Handles both native floats/None (fresh run, in-memory) and strings/""
    (rows reloaded from CSV via --plot_only, where csv.DictReader returns
    every cell as a string and empty cells as "" rather than None)."""
    if v is None or v == "":
        return None
    return float(v)


def make_charts(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    fg = [x for x in (_as_float(r["fine_grained_score"]) for r in rows) if x is not None]
    nv = [x for x in (_as_float(r["naive_score"]) for r in rows) if x is not None]

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

    # ── Accuracy vs score — does the score actually predict correctness? ────
    def _acc_by_bucket(score_key: str) -> dict[float, dict]:
        by_bucket: dict[float, list[int]] = {}
        for r in rows:
            jc = r.get("judged_correct", "")
            score = _as_float(r.get(score_key))
            if jc == "" or score is None:
                continue
            bucket = round(score, 1)  # 11 buckets: 0.0, 0.1, ..., 1.0
            by_bucket.setdefault(bucket, []).append(int(jc))
        return {b: {"n": len(v), "accuracy": sum(v) / len(v)} for b, v in sorted(by_bucket.items())}

    acc_fg = _acc_by_bucket("fine_grained_score")
    acc_nv = _acc_by_bucket("naive_score")

    if acc_fg or acc_nv:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, acc, title, color in [
            (ax1, acc_fg, "Fine-grained score", "#4C72B0"),
            (ax2, acc_nv, "Naive score", "#C44E52"),
        ]:
            buckets = sorted(acc)
            accs = [acc[b]["accuracy"] * 100 for b in buckets]
            ns = [acc[b]["n"] for b in buckets]
            bars = ax.bar([f"{b:.1f}" for b in buckets], accs, color=color, alpha=0.85,
                         edgecolor="black", linewidth=0.5)
            for bar, n in zip(bars, ns):
                ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                           ha="center", fontsize=7)
            ax.set_xlabel(f"{title} bucket")
            ax.set_ylabel("Judged accuracy (%)")
            ax.set_ylim(0, 105)
            ax.set_title(f"Accuracy vs {title.lower()}")
            ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "accuracy_vs_score.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")
    else:
        print("  [skip] accuracy_vs_score.png — no judged_correct data (rerun with the "
              "current script to get judging, older checkpoints predate it)")

    total_evals = sum(int(r["n_evals"]) for r in rows)
    total_logprob = sum(int(r["n_logprob_hits"]) for r in rows)
    total_fallback = sum(int(r["n_text_fallback"]) for r in rows)
    total_failed = sum(int(r["n_extraction_failed"]) for r in rows)

    all_correct = [int(r["judged_correct"]) for r in rows if r.get("judged_correct", "") != ""]
    accuracy = sum(all_correct) / len(all_correct) if all_correct else None
    fg_paired = [(x, int(r["judged_correct"])) for r, x in
                ((r, _as_float(r["fine_grained_score"])) for r in rows)
                if x is not None and r.get("judged_correct", "") != ""]
    corr_fg = (float(np.corrcoef([a for a, _ in fg_paired], [b for _, b in fg_paired])[0, 1])
              if len(fg_paired) > 2 and np.std([a for a, _ in fg_paired]) > 0 else None)

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
        "accuracy": accuracy,
        "correlation_fine_grained_score_vs_correct": corr_fg,
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
