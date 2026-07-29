"""
Verifier-CoT — prompt-only verifier-in-the-loop experiment (no fine-tuning).

Combines self_cot's free-form single description turn with cot_yaml's
multi-turn structure: instead of a fixed step list, the SAME model plays a
verifier role each round (different system prompt, same weights) and decides
whether the VLM's latest statement is SUPPORTED (end the chain) or
UNSUPPORTED (send a follow-up back to the VLM asking for more evidence,
try again). cot_yaml's step content is reused as the verifier's diagnostic
reference rubric (see src/verifier_vision.py's _RUBRIC_DISEASE/_RUBRIC_WEED,
already derived from cot_prompts.yaml) — the VLM being tested never sees it.

This is a throwaway validation experiment, not a fine-tuning pipeline: if
prompting alone recovers a meaningful chunk of the CoT failure modes
documented in outputs/verifier_analysis/verifier_findings.pdf, that changes
the cost/benefit case for the fine-tuned per-statement verifier
(experiments/ag_finetune/). If it doesn't, that's useful negative evidence.

Reuses (not reimplements):
  - src/verifier_vision.py: select_datasets, load_items, load_existing_rows,
    _RUBRIC_DISEASE, _RUBRIC_WEED
  - src/cot_vision.py: _sys_msg, _cls_question, _desc_question, _build_prompt,
    _clean, load_llm, destroy_llm
  - src/common/metrics.py: apply_judge
  - src/common/method_comparison.py: plot_overall_judge_comparison,
    plot_per_task_judge_comparison ("Verifier-CoT" condition registered there)

Usage:
  uv run python experiments/verifier_cot/run_verifier_cot.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/verifier_cot/ \\
    --vis_dir outputs/vision_assessment/ \\
    --cot_dir outputs/cot_vision/ \\
    --aug_dir outputs/desc_augmented_vision/ \\
    --n_samples 50 --n_datasets 3 --max_rounds 5 --batch_size 64

  # Resume from checkpoints:
  uv run python experiments/verifier_cot/run_verifier_cot.py ... --from_ckpt

  # Plot only (all checkpoints must exist):
  uv run python experiments/verifier_cot/run_verifier_cot.py ... --plot_only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image as PILImage

from src.common.progress import Progress, fmt_duration, stage_banner, sub_banner

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_ROUNDS_DEFAULT = 5
N_DATASETS_PER_TASK = 3
SEED = 42

# Matches src/verifier_vision.py's JUDGED_FIELDS shape (no resolved_class/
# correct/contains_match — not needed by the plotting functions this
# experiment uses, and verifier_vision.py's own load_items() item schema
# doesn't carry a clean_classes list the way cot_vision.py's does, so
# cot_vision._make_row isn't a drop-in fit here).
CLS_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "condition", "prediction", "reasoning", "ground_truth",
    "judged_correct", "judge_reasoning",
    "n_rounds", "final_verifier_verdict", "final_verifier_confidence",
    "transcript_json",
]

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)


# ── Verifier prompt template — loaded from verifier_prompt.yaml, not hardcoded,
# so it's editable without touching Python. Substitution is plain string
# .replace(), not str.format(), so the JSON schema's literal { } don't need
# escaping when editing the YAML. ──────────────────────────────────────────────

_VERIFIER_PROMPT_YAML = Path(__file__).parent / "verifier_prompt.yaml"


def _load_verifier_prompts() -> dict:
    import yaml
    with open(_VERIFIER_PROMPT_YAML) as f:
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
    """Find and parse the first balanced {...} object in text, tracking brace
    depth rather than naively spanning first-'{' to last-'}' — the naive span
    breaks whenever the model wraps its JSON in a markdown code fence or the
    response text itself contains other brace-like content, silently grabbing
    the wrong substring instead of failing loudly."""
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
                        start = None  # keep scanning for the next balanced pair
    return None


def _parse_verifier_json(text: str) -> tuple[str, str, int]:
    """Extract (verdict, response, confidence) from the verifier's JSON reply,
    tolerating surrounding prose, code fences, and stray <think> blocks."""
    text = _THINK_RE.sub("", text).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)  # strip markdown code fences

    data = _extract_first_json_object(text)
    if data is not None:
        verdict = str(data.get("verdict", "UNSUPPORTED")).strip().upper()
        if verdict not in ("SUPPORTED", "UNSUPPORTED"):
            verdict = "UNSUPPORTED"
        response = str(data.get("response", ""))[:500]
        try:
            confidence = max(1, min(20, int(data.get("confidence", 1))))
        except (ValueError, TypeError):
            confidence = 1
        return verdict, response, confidence

    # Only reached if no balanced, parseable {...} exists at all (e.g. output
    # was truncated mid-JSON by max_tokens before the closing brace).
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', text, re.IGNORECASE)
    verdict = m.group(1).upper() if m else "UNSUPPORTED"
    if verdict not in ("SUPPORTED", "UNSUPPORTED"):
        verdict = "UNSUPPORTED"
    m2 = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    response = m2.group(1)[:500] if m2 else text[:500]
    m3 = re.search(r'"confidence"\s*:\s*(\d+)', text)
    confidence = max(1, min(20, int(m3.group(1)))) if m3 else 1
    return verdict, response, confidence


# ── CSV helpers (matches src/verifier_vision.py's local pattern) ──────────────

def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _import_helpers():
    from src.cot_vision import (
        _sys_msg, _cls_question, _desc_question, _build_prompt, _clean,
    )
    return _sys_msg, _cls_question, _desc_question, _build_prompt, _clean


# ── Sampling: guarantee overlap with existing self_cot/cot_yaml, don't hope for it ──
#
# sample_class_images(names, n, seed) is NOT nested across different n for the
# same seed (np.random.default_rng(seed).choice(N, size=n) picks different
# specific indices for n=5 vs n=50) — so independently re-sampling here and
# only filtering existing self_cot/cot_yaml rows down to the intersection
# afterward (verifier_vision.py's load_existing_rows pattern) can silently
# yield ZERO overlap if this run's --n_samples differs from whatever the
# original self_cot/cot_yaml run used. Fix: restrict the candidate pool to
# images already covered by the existing CSV BEFORE sampling, so overlap is
# guaranteed regardless of n_samples mismatches between runs.

def existing_cot_image_names(cot_dir: Path, model_safe: str) -> dict[tuple, set[str]]:
    """Return {(agml_dataset, class_label): {image_name, ...}} already covered
    by cot_cls_{model}.csv (both self_cot and cot_yaml rows — same file)."""
    cot_csv = cot_dir / f"cot_cls_{model_safe}.csv"
    covered: dict[tuple, set[str]] = {}
    if not cot_csv.exists():
        print(f"  [warn] {cot_csv} not found — cannot restrict sampling to overlap with it")
        return covered
    for r in _read_csv(cot_csv):
        key = (r.get("agml_dataset", ""), r.get("class_label", ""))
        covered.setdefault(key, set()).add(r.get("image_name", ""))
    return covered


def load_items_matching_existing(
    ds_name: str,
    task_type: str,
    crop: str | None,
    n_samples: int | None,
    class_overrides: dict[str, str] | None,
    seed: int,
    test_split: dict | None,
    covered: dict[tuple, set[str]],
) -> list[dict]:
    """Same behavior/shape as src.verifier_vision.load_items, but restricts
    each class's candidate pool to images already in `covered` (if any exist
    for that dataset/class) before applying the --n_samples cap, so the
    resulting sample is guaranteed to overlap with the existing self_cot/
    cot_yaml CSV instead of hoping two independent random draws collide."""
    import random
    from src.cot_vision import load_dataset_classes, DATASET_TASK_TYPES
    from src.common.metrics import sample_class_images

    by_class = load_dataset_classes(ds_name, None, test_split)  # uncapped — full per-class pool
    primary_task = DATASET_TASK_TYPES.get(ds_name, (task_type, crop))[0]
    items: list[dict] = []
    for cls, paths in by_class.items():
        effective = (class_overrides or {}).get(cls, primary_task)
        if effective != task_type:
            continue
        cov = covered.get((ds_name, cls))
        candidates = [p for p in paths if p.name in cov] if cov else paths
        if cov and not candidates:
            print(f"    [warn] {ds_name}/{cls}: no overlap with existing self_cot/cot_yaml "
                  f"images — falling back to full pool ({len(paths)} images), comparison "
                  f"charts for this class will be missing Self-CoT/CoT YAML")
            candidates = paths
        names = sample_class_images([p.name for p in candidates], n_samples, seed=seed)
        name_set = set(names)
        for p in candidates:
            if p.name in name_set:
                items.append({
                    "ds": ds_name, "task_type": task_type, "crop": crop, "cls": cls,
                    "image_path": str(p), "image_name": p.name,
                })
    random.Random(seed).shuffle(items)
    return items


# ── Verifier-in-the-loop generation ────────────────────────────────────────────

def run_verifier_loop(
    items: list[dict],
    llm,
    tokenizer,
    max_rounds: int = MAX_ROUNDS_DEFAULT,
    batch_size: int = 64,
    ckpt_path: Path | None = None,
    force_max_rounds: bool = False,
) -> list[dict]:
    """Round-loop: VLM free-form statement -> verifier accept/reject -> repeat
    (up to max_rounds) -> final classification turn for every image.

    force_max_rounds=True ignores SUPPORTED verdicts as a stopping condition
    and always runs every image to exactly max_rounds — the verifier still
    scores and responds every round exactly as normal (nothing about the
    prompt or generation changes), it just never gets to end the chain early.
    Useful for isolating "what does round N look like" from "which images
    survive to round N", since early-stopping normally shrinks the active set
    fast (e.g. 1927 -> 231 -> 55 -> 17 -> 6 across 5 rounds) and later rounds
    become too small a sample to read anything into.
    """
    from src.verifier_vision import _RUBRIC_DISEASE, _RUBRIC_WEED

    _sys_msg, _cls_question, _desc_question, _build_prompt, _clean = _import_helpers()

    _VLM_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    # 512 not 256 — the "response" field has no length cap in the prompt, and
    # this model tends to write multi-paragraph analyses that can otherwise
    # truncate mid-JSON before the closing "confidence" field is ever reached.
    _VER_P = SamplingParams(temperature=0.0, max_tokens=512)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)

    n = len(items)
    hist: list[list[dict]] = [[] for _ in range(n)]
    n_rounds = [0] * n
    final_statement = [""] * n
    final_verdict = [""] * n
    final_confidence = [0] * n
    # Full round-by-round record per item, for quality review — each entry is
    # {"round", "vlm_statement", "verifier_verdict", "verifier_response",
    # "verifier_confidence"}. Not just the final accepted statement.
    transcript: list[list[dict]] = [[] for _ in range(n)]

    def _img(i: int) -> PILImage.Image:
        return PILImage.open(items[i]["image_path"]).convert("RGB")

    # ── Round 1: initial free-form description (self_cot-style) ────────────────
    sub_banner(f"Round 1/{max_rounds}: initial description ({n} images)")
    prog = Progress(total=n, label="Round 1 description", every=max(batch_size, 1))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        prompts = []
        for i in batch_idx:
            item = items[i]
            desc_q = _desc_question(item["task_type"], item["crop"])
            h = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": desc_q},
            ]}]
            hist[i] = h
            prompts.append(_build_prompt(h, tokenizer, _img(i)))
        outputs = llm.generate(prompts, _VLM_P)
        for i, out in zip(batch_idx, outputs):
            hist[i].append({"role": "assistant", "content": _clean(out.outputs[0].text)})
        del prompts
        prog.update(len(batch_idx))

    active_idx = list(range(n))

    for rnd in range(1, max_rounds + 1):
        if not active_idx:
            break
        sub_banner(f"Verifying round {rnd}/{max_rounds}: {len(active_idx)} active images")

        still_active = []
        n_supported_this_round = 0
        prog = Progress(total=len(active_idx), label=f"Verify round {rnd}", every=max(batch_size, 1))
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            ver_prompts = []
            cls_prompts = []  # round-level classification snapshot, for a future per-round accuracy curve
            for i in batch_idx:
                item = items[i]
                statement = hist[i][-1]["content"]
                rubric = _RUBRIC_WEED if item["task_type"] == "weed_identification" else _RUBRIC_DISEASE
                sys_p = _verifier_system(rubric)
                user_p = _verifier_user_turn("(none yet)", statement)
                msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": user_p},
                ]}]
                ver_prompts.append(_build_prompt(msgs, tokenizer, _img(i)))
                cls_h = hist[i] + [{"role": "user", "content": _cls_question(item["task_type"], item["crop"])}]
                cls_prompts.append(_build_prompt(cls_h, tokenizer, _img(i)))
            ver_outputs = llm.generate(ver_prompts, _VER_P)
            cls_outputs = llm.generate(cls_prompts, _CLS_P)
            del ver_prompts, cls_prompts

            for i, out, cls_out in zip(batch_idx, ver_outputs, cls_outputs):
                verdict, response, confidence = _parse_verifier_json(out.outputs[0].text)
                round_prediction = _clean(cls_out.outputs[0].text)
                n_rounds[i] = rnd
                statement_this_round = hist[i][-1]["content"]
                transcript[i].append({
                    "round": rnd,
                    "vlm_statement": statement_this_round,
                    "verifier_verdict": verdict,
                    "verifier_response": response,
                    "verifier_confidence": confidence,
                    "round_prediction": round_prediction,
                })
                stop_here = rnd == max_rounds or (verdict == "SUPPORTED" and not force_max_rounds)
                if stop_here:
                    final_statement[i] = statement_this_round
                    final_verdict[i] = "SUPPORTED" if verdict == "SUPPORTED" else "UNSUPPORTED_MAXROUNDS"
                    final_confidence[i] = confidence
                    if verdict == "SUPPORTED":
                        n_supported_this_round += 1
                else:
                    hist[i].append({"role": "user", "content": response})
                    still_active.append(i)
            prog.update(len(batch_idx), extra=f"{n_supported_this_round} supported so far")

        active_idx = still_active
        if not active_idx or rnd == max_rounds:
            break

        sub_banner(f"Generating round {rnd + 1} statements: {len(active_idx)} images")
        prog = Progress(total=len(active_idx), label=f"Round {rnd + 1} generation", every=max(batch_size, 1))
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            prompts = [_build_prompt(hist[i], tokenizer, _img(i)) for i in batch_idx]
            outputs = llm.generate(prompts, _VLM_P)
            for i, out in zip(batch_idx, outputs):
                hist[i].append({"role": "assistant", "content": _clean(out.outputs[0].text)})
            del prompts
            prog.update(len(batch_idx))

    # ── Final classification turn for every image (matches self_cot/cot_yaml) ──
    sub_banner(f"Final classification: {n} images")
    predictions: list[str] = [""] * n
    prog = Progress(total=n, label="Final classification", every=max(batch_size, 1))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        prompts = []
        for i in batch_idx:
            item = items[i]
            h = hist[i] + [{"role": "user", "content": _cls_question(item["task_type"], item["crop"])}]
            prompts.append(_build_prompt(h, tokenizer, _img(i)))
        outputs = llm.generate(prompts, _CLS_P)
        for i, out in zip(batch_idx, outputs):
            predictions[i] = _clean(out.outputs[0].text)
        del prompts
        prog.update(len(batch_idx))

    rows = []
    for i, item in enumerate(items):
        fallback_statement = hist[i][-1]["content"] if hist[i] and hist[i][-1]["role"] == "assistant" else ""
        reasoning = "\n\n".join(
            f"Round {t['round']}:\n"
            f"  VLM: {t['vlm_statement']}\n"
            f"  Verifier: {t['verifier_verdict']} (confidence {t['verifier_confidence']})\n"
            f"  Verifier response: {t['verifier_response']}"
            for t in transcript[i]
        )
        rows.append({
            "agml_dataset": item["ds"],
            "class_label":  item["cls"],
            "task_type":    item["task_type"],
            "crop":         item.get("crop") or "",
            "image_name":   item["image_name"],
            "condition":    "verifier_cot",
            "prediction":   predictions[i],
            "reasoning":    reasoning or final_statement[i] or fallback_statement,
            "ground_truth": item["cls"],
            "judged_correct": "",
            "judge_reasoning": "",
            "n_rounds": n_rounds[i],
            "final_verifier_verdict": final_verdict[i],
            "final_verifier_confidence": final_confidence[i],
            "transcript_json": json.dumps(transcript[i]),
        })

    if ckpt_path:
        _write_csv(rows, ckpt_path, CLS_FIELDS)
    return rows


# ── Plots ──────────────────────────────────────────────────────────────────────

def build_rows_by_condition(verifier_rows: list[dict], existing: dict[str, list[dict]]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {"Verifier-CoT": verifier_rows}
    for label, rows in existing.items():
        if rows:
            result[label] = rows
    return result


def plot_all(judged: list[dict], existing: dict[str, list[dict]], output_dir: Path) -> None:
    from src.common.method_comparison import plot_overall_judge_comparison, plot_per_task_judge_comparison
    from src.verifier_vision import plot_per_dataset

    rows_by_cond = build_rows_by_condition(judged, existing)
    present = [c for c in rows_by_cond if rows_by_cond[c]]
    print(f"\n[plot] conditions: {present}")

    plot_overall_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_overall.png",
        title="Verifier-CoT vs Baselines — Overall Mean Judged F1",
    )
    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_per_task.png",
        title="Verifier-CoT vs Baselines — Mean Judged F1 by Task",
    )
    plot_per_dataset(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_per_dataset.png",
    )

    verdicts = [r["final_verifier_verdict"] for r in judged]
    from collections import Counter
    print(f"\n[verifier verdict distribution] {dict(Counter(verdicts))}")

    # Confidence analysis (confidence_by_round.png, final_confidence_distribution.png,
    # accuracy_vs_confidence.png, confidence_analysis_summary.json) — same directory
    # as everything else here, run automatically rather than requiring a separate
    # manual invocation of analyze_confidence.py after the fact.
    print("\n[plot] confidence analysis...")
    from analyze_confidence import make_charts as make_confidence_charts
    confidence_summary = make_confidence_charts(judged, output_dir)
    confidence_summary_path = output_dir / "confidence_analysis_summary.json"
    confidence_summary_path.write_text(json.dumps(confidence_summary, indent=2))
    print(f"  Saved: {confidence_summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier-CoT — prompt-only verifier-in-the-loop experiment")
    parser.add_argument("--model",       required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir",  default="outputs/verifier_cot/")
    parser.add_argument("--vis_dir",     default="outputs/vision_assessment/")
    parser.add_argument("--cot_dir",     default="outputs/cot_vision/")
    parser.add_argument("--aug_dir",     default="outputs/desc_augmented_vision/")
    parser.add_argument("--config",      default="src/config/datasets.yaml")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json")
    parser.add_argument("--n_samples",   type=int, default=50, help="Max images per class")
    parser.add_argument("--n_datasets",  type=int, default=N_DATASETS_PER_TASK, help="Datasets per task")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Use every active dataset instead of sampling --n_datasets per task "
                             "(--n_datasets is ignored if this is set)")
    parser.add_argument("--max_rounds",  type=int, default=MAX_ROUNDS_DEFAULT)
    parser.add_argument("--force_max_rounds", action="store_true",
                        help="Ignore SUPPORTED verdicts as a stopping condition — every image "
                             "runs the full --max_rounds regardless. Useful for getting a full "
                             "sample size at every round instead of the active set shrinking "
                             "fast under normal early-stopping.")
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--from_ckpt",   action="store_true",
                        help="Skip phases if their checkpoint CSV exists")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip all inference; load checkpoints and plot")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Required to proceed if --output_dir already has a "
                             "ckpt_verifier_cot.csv and this run isn't --from_ckpt/--plot_only "
                             "(i.e. would silently overwrite existing results)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_verifier_cot = output_dir / "ckpt_verifier_cot.csv"
    ckpt_judged = output_dir / "ckpt_judged.csv"

    # Hard stop, not a print-and-continue warning — this is a long-running
    # (potentially hours, on a remote machine you may not be watching in real
    # time) destructive operation, and a warning buried mid-log is not a real
    # safeguard against silently overwriting a previous run's results.
    if ckpt_verifier_cot.exists() and not (args.from_ckpt or args.plot_only or args.overwrite):
        print(f"ERROR: {ckpt_verifier_cot} already exists and this run is not "
              f"--from_ckpt/--plot_only, so it would overwrite existing results.\n"
              f"  Either point --output_dir somewhere new (e.g. matching the "
              f"'_forced_{{max_rounds}}' convention if --force_max_rounds is set), "
              f"or pass --overwrite to confirm you want to replace what's there.")
        sys.exit(1)

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    test_split = None
    ts_path = Path(args.test_split)
    if ts_path.exists():
        test_split = json.loads(ts_path.read_text())
        print(f"Loaded test_split: {ts_path} ({len(test_split)} datasets)")
    else:
        print(f"[warn] test_split not found at {ts_path} — using all images")

    from src.verifier_vision import select_datasets, load_existing_rows

    model_safe = args.model.replace("/", "_")

    if args.all_datasets and args.n_datasets != N_DATASETS_PER_TASK:
        print(f"[warn] --all_datasets set — ignoring --n_datasets={args.n_datasets}")
    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed,
                               config_path=args.config, select_all=args.all_datasets)
    print(f"\n=== Selected datasets ({'ALL' if args.all_datasets else f'{args.n_datasets}/task'}) ===")
    total_selected = 0
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")
        total_selected += len(names)
    print(f"  Total: {total_selected} dataset selections across all tasks")

    skip_generate = args.plot_only or (args.from_ckpt and ckpt_verifier_cot.exists())
    skip_judge = args.plot_only or (args.from_ckpt and ckpt_judged.exists())

    need_model = not (skip_generate and skip_judge)
    llm = tokenizer = None
    t0 = time.perf_counter()

    if need_model:
        from src.cot_vision import load_llm
        print(f"\n=== Loading model: {args.model} ===")
        llm, tokenizer = load_llm(args.model)
        t0 = time.perf_counter()

    if skip_generate:
        stage_banner(1, "Verifier loop [SKIPPED]", total_stages=3)
        verifier_rows = _read_csv(ckpt_verifier_cot)
    else:
        covered = existing_cot_image_names(Path(args.cot_dir), model_safe)
        print(f"  Existing self_cot/cot_yaml coverage: {len(covered)} (dataset, class) groups")

        all_items: list[dict] = []
        for task, ds_names in selected.items():
            for ds in ds_names:
                info = ds_info.get(ds, {})
                crop = info.get("crop", "plant")
                class_overrides = info.get("class_overrides", {})
                items = load_items_matching_existing(
                    ds, task, crop, n_samples=args.n_samples,
                    class_overrides=class_overrides, seed=args.seed,
                    test_split=test_split, covered=covered,
                )
                all_items.extend(items)
                print(f"  {ds} ({task}): {len(items)} items")
        print(f"Total items: {len(all_items)}")

        mode_label = "FORCED " if args.force_max_rounds else ""
        stage_banner(1, f"{mode_label}Verifier loop ({len(all_items)} images, max {args.max_rounds} rounds)",
                    total_stages=3)
        if args.force_max_rounds:
            print(f"  --force_max_rounds set: every image will run all {args.max_rounds} rounds "
                  f"regardless of verdict.")
            if "_forced_" not in str(output_dir):
                print(f"  [reminder] --output_dir does not contain '_forced_{args.max_rounds}' — "
                      f"make sure this isn't overwriting a normal (early-stopping) run's results.")
        t_stage1 = time.perf_counter()
        verifier_rows = run_verifier_loop(all_items, llm, tokenizer, max_rounds=args.max_rounds,
                                          batch_size=args.batch_size, ckpt_path=ckpt_verifier_cot,
                                          force_max_rounds=args.force_max_rounds)
        print(f"\nStage 1 done in {fmt_duration(time.perf_counter() - t_stage1)}")

    if skip_judge:
        stage_banner(2, "Judge [SKIPPED]", total_stages=3)
        judged = _read_csv(ckpt_judged)
    else:
        from src.common.metrics import apply_judge
        prompts_yaml = Path(__file__).parent.parent.parent / "src" / "config" / "prompts.yaml"
        _p = yaml.safe_load(prompts_yaml.read_text())
        stage_banner(2, f"Judge ({len(verifier_rows)} rows)", total_stages=3)
        t_stage2 = time.perf_counter()
        judged = apply_judge(
            verifier_rows, llm, tokenizer,
            system_prompt=_p["judge_system_vision"],
            judge_template=_p["judge_classification_template"],
            gt_key="ground_truth", pred_key="prediction",
        )
        _write_csv(judged, ckpt_judged, CLS_FIELDS)
        print(f"\nStage 2 done in {fmt_duration(time.perf_counter() - t_stage2)}")

    if need_model:
        from src.cot_vision import destroy_llm
        elapsed = time.perf_counter() - t0
        print(f"\nInference done in {fmt_duration(elapsed)}")
        destroy_llm(llm)
        del tokenizer, llm
        gc.collect()

    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    stage_banner(3, f"Plot ({len(image_keys)} unique images)", total_stages=3)

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        model_safe=model_safe,
    )

    plot_all(judged, existing, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
