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


def _parse_verifier_json(text: str) -> tuple[str, str, int]:
    """Extract (verdict, response, confidence) from the verifier's JSON reply,
    tolerating surrounding prose / stray <think> blocks."""
    text = _THINK_RE.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            verdict = str(data.get("verdict", "UNSUPPORTED")).strip().upper()
            if verdict not in ("SUPPORTED", "UNSUPPORTED"):
                verdict = "UNSUPPORTED"
            response = str(data.get("response", ""))[:500]
            confidence = max(1, min(20, int(data.get("confidence", 1))))
            return verdict, response, confidence
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
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


# ── Verifier-in-the-loop generation ────────────────────────────────────────────

def run_verifier_loop(
    items: list[dict],
    llm,
    tokenizer,
    max_rounds: int = MAX_ROUNDS_DEFAULT,
    batch_size: int = 64,
    ckpt_path: Path | None = None,
) -> list[dict]:
    """Round-loop: VLM free-form statement -> verifier accept/reject -> repeat
    (up to max_rounds) -> final classification turn for every image."""
    from src.verifier_vision import _RUBRIC_DISEASE, _RUBRIC_WEED

    _sys_msg, _cls_question, _desc_question, _build_prompt, _clean = _import_helpers()

    _VLM_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _VER_P = SamplingParams(temperature=0.0, max_tokens=256)
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)

    n = len(items)
    hist: list[list[dict]] = [[] for _ in range(n)]
    n_rounds = [0] * n
    final_statement = [""] * n
    final_verdict = [""] * n
    final_confidence = [0] * n

    def _img(i: int) -> PILImage.Image:
        return PILImage.open(items[i]["image_path"]).convert("RGB")

    # ── Round 1: initial free-form description (self_cot-style) ────────────────
    print(f"\n[verifier_cot] Round 1/{max_rounds}: initial description ({n} images)")
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

    active_idx = list(range(n))

    for rnd in range(1, max_rounds + 1):
        if not active_idx:
            break
        print(f"\n[verifier_cot] Verifying round {rnd}/{max_rounds}: {len(active_idx)} active images")

        ver_prompts = []
        for i in active_idx:
            item = items[i]
            statement = hist[i][-1]["content"]
            rubric = _RUBRIC_WEED if item["task_type"] == "weed_identification" else _RUBRIC_DISEASE
            sys_p = _verifier_system(rubric)
            user_p = _verifier_user_turn("(none yet)", statement)
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": user_p},
            ]}]
            ver_prompts.append(_build_prompt(msgs, tokenizer, _img(i)))
        ver_outputs = llm.generate(ver_prompts, _VER_P)
        del ver_prompts

        still_active = []
        for i, out in zip(active_idx, ver_outputs):
            verdict, response, confidence = _parse_verifier_json(out.outputs[0].text)
            n_rounds[i] = rnd
            if verdict == "SUPPORTED" or rnd == max_rounds:
                final_statement[i] = hist[i][-1]["content"]
                final_verdict[i] = "SUPPORTED" if verdict == "SUPPORTED" else "UNSUPPORTED_MAXROUNDS"
                final_confidence[i] = confidence
            else:
                hist[i].append({"role": "user", "content": response})
                still_active.append(i)

        active_idx = still_active
        if not active_idx or rnd == max_rounds:
            break

        print(f"[verifier_cot] Generating round {rnd + 1} statements: {len(active_idx)} images")
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            prompts = [_build_prompt(hist[i], tokenizer, _img(i)) for i in batch_idx]
            outputs = llm.generate(prompts, _VLM_P)
            for i, out in zip(batch_idx, outputs):
                hist[i].append({"role": "assistant", "content": _clean(out.outputs[0].text)})
            del prompts

    # ── Final classification turn for every image (matches self_cot/cot_yaml) ──
    print(f"\n[verifier_cot] Final classification: {n} images")
    predictions: list[str] = [""] * n
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

    rows = []
    for i, item in enumerate(items):
        fallback_statement = hist[i][-1]["content"] if hist[i] and hist[i][-1]["role"] == "assistant" else ""
        rows.append({
            "agml_dataset": item["ds"],
            "class_label":  item["cls"],
            "task_type":    item["task_type"],
            "crop":         item.get("crop") or "",
            "image_name":   item["image_name"],
            "condition":    "verifier_cot",
            "prediction":   predictions[i],
            "reasoning":    final_statement[i] or fallback_statement,
            "ground_truth": item["cls"],
            "judged_correct": "",
            "judge_reasoning": "",
            "n_rounds": n_rounds[i],
            "final_verifier_verdict": final_verdict[i],
            "final_verifier_confidence": final_confidence[i],
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
    parser.add_argument("--max_rounds",  type=int, default=MAX_ROUNDS_DEFAULT)
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--from_ckpt",   action="store_true",
                        help="Skip phases if their checkpoint CSV exists")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip all inference; load checkpoints and plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_verifier_cot = output_dir / "ckpt_verifier_cot.csv"
    ckpt_judged = output_dir / "ckpt_judged.csv"

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

    from src.verifier_vision import select_datasets, load_items, load_existing_rows

    selected = select_datasets(n_per_task=args.n_datasets, seed=args.seed, config_path=args.config)
    print("\n=== Selected datasets ===")
    for task, names in selected.items():
        print(f"  {task} ({len(names)}): {names}")

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
        print("\n=== Phase 1: verifier loop [SKIPPED] ===")
        verifier_rows = _read_csv(ckpt_verifier_cot)
    else:
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

        print(f"\n=== Phase 1: verifier loop ({len(all_items)} images, max {args.max_rounds} rounds) ===")
        verifier_rows = run_verifier_loop(all_items, llm, tokenizer, max_rounds=args.max_rounds,
                                          batch_size=args.batch_size, ckpt_path=ckpt_verifier_cot)

    if skip_judge:
        print("\n=== Phase 2: judge [SKIPPED] ===")
        judged = _read_csv(ckpt_judged)
    else:
        from src.common.metrics import apply_judge
        prompts_yaml = Path(__file__).parent.parent.parent / "src" / "config" / "prompts.yaml"
        _p = yaml.safe_load(prompts_yaml.read_text())
        print(f"\n=== Phase 2: judge ({len(verifier_rows)} rows) ===")
        judged = apply_judge(
            verifier_rows, llm, tokenizer,
            system_prompt=_p["judge_system_vision"],
            judge_template=_p["judge_classification_template"],
            gt_key="ground_truth", pred_key="prediction",
        )
        _write_csv(judged, ckpt_judged, CLS_FIELDS)

    if need_model:
        from src.cot_vision import destroy_llm
        elapsed = time.perf_counter() - t0
        print(f"\nInference done in {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer, llm
        gc.collect()

    model_safe = args.model.replace("/", "_")
    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    print(f"\n=== Phase 3: plot ({len(image_keys)} unique images) ===")

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
