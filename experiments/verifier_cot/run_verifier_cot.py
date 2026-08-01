"""
Verifier-CoT — prompt-only verifier-in-the-loop experiment (no fine-tuning).

Combines self_cot's free-form single description turn with cot_yaml's
multi-turn structure: instead of a fixed step list, a verifier model plays a
verifier role each round (different system prompt) and scores the VLM's
latest statement on the same fine-grained logprob letter-scale as
experiments/verifier_ppt (see that experiment's module docstring for the
two-pass prefill mechanism) — the chain ends early once that score clears
--score_threshold (default 0.9), or unconditionally at --max_rounds.
cot_yaml's step content is reused as the verifier's diagnostic reference
rubric (see src/verifier_vision.py's _RUBRIC_DISEASE/_RUBRIC_WEED, already
derived from cot_prompts.yaml) — the VLM being tested never sees it.

Verifier can be the SAME model as the general/diagnostic model (original
design, same weights, different system prompt) or a SEPARATE --verifier_model
(e.g. a smaller dedicated model). Because verification and generation
alternate every round (verify → round-classification snapshot → next-round
generation → verify again...), a cross-model run can't do one clean swap the
way experiments/verifier_ppt does — see ModelSwitcher below, which swaps at
each round boundary rather than keeping both models resident (full-weight
MoE + a second dense model doesn't reliably fit an 80GB GPU together).

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
    plot_per_task_judge_comparison ("Verifier-CoT (...)" conditions registered
    there — one per --condition_label, so separate runs with different
    --verifier_model values land as distinct bars in the same chart)
  - llm_verifier.fine_grained_reward: SCALE, extract_score (score extraction
    duplicated from experiments/verifier_ppt/run_verifier_ppt.py, not
    imported, per this repo's self-contained-experiment convention)

Usage:
  # Gemma verifies itself (original design):
  uv run python experiments/verifier_cot/run_verifier_cot.py \\
    --model google/gemma-4-26B-A4B-it \\
    --condition_label "Verifier-CoT (Gemma-self)" \\
    --output_dir outputs/verifier_cot_gemma_self/ \\
    --vis_dir outputs/vision_assessment/ \\
    --cot_dir outputs/cot_vision/ \\
    --aug_dir outputs/desc_augmented_vision/ \\
    --n_samples 50 --n_datasets 3 --max_rounds 5 --score_threshold 0.9 --batch_size 64

  # Dedicated verifier model instead:
  uv run python experiments/verifier_cot/run_verifier_cot.py \\
    --model google/gemma-4-26B-A4B-it \\
    --verifier_model Qwen/Qwen3.5-9B \\
    --condition_label "Verifier-CoT (Qwen-base)" \\
    --output_dir outputs/verifier_cot_qwen_base/ \\
    --n_samples 50 --n_datasets 3 --max_rounds 5 --score_threshold 0.9

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
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image as PILImage

from src.common.progress import Progress, fmt_duration, stage_banner, sub_banner

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

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_ROUNDS_DEFAULT = 5
SCORE_THRESHOLD_DEFAULT = 0.9
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
    "n_rounds", "final_verifier_verdict",
    "final_verifier_fine_grained_score", "final_verifier_naive_score",
    "transcript_json",
    "full_history_json",  # exact verbatim message list sent to the model at
                          # every point (system prompts included) — the
                          # distilled per-round transcript_json above is
                          # derived from this but easier to skim; this field
                          # is for full-fidelity reconstruction of any
                          # intermediate conversation state.
]

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_SCORE_TAG_RE = re.compile(r"<score>\s*(.+?)\s*</score>", re.IGNORECASE)

# Same 20-letter scale + leading-space variants as verifier_ppt — see that
# script's comment on the same constant for why the space variants matter.
_SCORE_LETTERS = [chr(65 + i) for i in range(20)] + [f" {chr(65 + i)}" for i in range(20)]


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


def _build_score_prompt(base_prompt: dict, analysis: str) -> dict:
    """Continue `base_prompt`'s exact text with the verifier's own analysis
    plus a prefilled '<score>' tag — see verifier_ppt.py's function of the
    same name for the full rationale (offline-vLLM equivalent of
    continue_final_message=True)."""
    return {
        "prompt": base_prompt["prompt"] + analysis + "\n<score>",
        "multi_modal_data": base_prompt["multi_modal_data"],
    }


def naive_score_from_text(text: str) -> float | None:
    """Literal text value of the letter actually written, no logprob
    weighting. Returns None if no <score> tag/valid letter found."""
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
    """One round's verifier result under the two-pass prefill scheme — see
    experiments/verifier_ppt/run_verifier_ppt.py's function of the same name
    for the full mechanism. `analysis` is pass 1's cleaned text up to (not
    including) the '<score>' tag; `second_out` is pass 2's single
    grammar-constrained token generated right after a prefilled '<score>'."""
    data = _extract_first_json_object(analysis) or {}
    response = str(data.get("response", ""))[:500]

    letter_raw = second_out.outputs[0].text
    logprobs_list = second_out.outputs[0].logprobs or []
    lp_dict = logprobs_list[0] if logprobs_list else {}
    alts = [(lp.decoded_token, lp.logprob) for lp in lp_dict.values()] if lp_dict \
        else [(letter_raw, 0.0)]

    synth_text = f"{analysis}\n<score>{letter_raw}</score>"
    synth_tokens = ["\n<score>", letter_raw, "</score>"]
    synth_position_logprobs = [[("\n<score>", 0.0)], alts, [("</score>", 0.0)]]

    had_logprob_hit = any(tok.strip() in SCALE["valid_tokens"] for tok, _ in alts)
    fine_grained = extract_score(synth_text, synth_tokens, synth_position_logprobs, "<score>")
    naive = naive_score_from_text(synth_text)

    return {
        "response": response,
        "raw_text": synth_text,
        "fine_grained_score": fine_grained,
        "naive_score": naive,
        "extraction": "logprob" if had_logprob_hit else ("text_fallback" if naive is not None else "failed"),
    }


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


# ── Cross-model round-boundary swapping ─────────────────────────────────────────
#
# Verification and generation alternate every round, so a separate
# --verifier_model can't do one clean swap like experiments/verifier_ppt does
# — it needs to swap at each round boundary instead. Keeping both models
# resident simultaneously isn't attempted: a full-weight MoE general model
# (all experts resident even though few activate per token) plus a second
# dense verifier model doesn't reliably fit an 80GB GPU together. When
# verifier_model is None or equals the general model, this never reloads —
# .get() just returns the single loaded model every time, identical to the
# original single-model behavior.
#
# Repeated reloads within one process hit a real bug on farm compute nodes
# without reliable outbound internet: transformers/huggingface_hub still
# issues a HEAD request on every from_pretrained call to check cache
# freshness, even when every needed file is already local. That request
# fails ("Network is unreachable"), and the retry path leaves the
# underlying httpx client closed, crashing the NEXT load attempt with
# "Cannot send a request, as the client has been closed" — hit in practice
# reloading Gemma back after a Qwen verifier round (both already proven
# cached earlier in the same run). HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE skip
# that check entirely and trust the local cache — safe here since every
# model this switcher loads (both --model and --verifier_model, or a local
# checkpoint path) must already be fully cached to have loaded once, which
# happens before any reload ever occurs. setdefault() so an operator's own
# explicit choice (e.g. deliberately online for a not-yet-cached model) isn't
# silently overridden.

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ModelSwitcher:
    def __init__(self, general_model_id: str, verifier_model_id: str | None,
                max_model_len: int):
        self.general_id = general_model_id
        self.verifier_id = verifier_model_id or general_model_id
        self.same = self.verifier_id == self.general_id
        self.max_model_len = max_model_len
        self.llm = None
        self.tokenizer = None
        self.current_role: str | None = None
        self.n_reloads = 0

    def get(self, role: str):
        """role: 'general' or 'verifier'. Returns (llm, tokenizer)."""
        assert role in ("general", "verifier")
        target_id = self.general_id if role == "general" else self.verifier_id
        if self.same:
            if self.llm is None:
                self._load(target_id)
            return self.llm, self.tokenizer
        if self.current_role != role:
            self._reload(target_id)
            self.current_role = role
        return self.llm, self.tokenizer

    def _load(self, model_id: str) -> None:
        from src.cot_vision import load_llm
        print(f"\n=== Loading model: {model_id} ===")
        self.llm, self.tokenizer = load_llm(model_id, max_model_len=self.max_model_len)

    def _reload(self, model_id: str) -> None:
        was_loaded = self.llm is not None
        self._destroy()
        self._load(model_id)
        if was_loaded:  # don't count the very first load as a "reload"
            self.n_reloads += 1

    def _destroy(self) -> None:
        if self.llm is None:
            return
        from src.cot_vision import destroy_llm
        destroy_llm(self.llm)
        del self.tokenizer, self.llm
        self.llm = self.tokenizer = None
        gc.collect()

    def destroy(self) -> None:
        self._destroy()
        self.current_role = None


# ── Verifier-in-the-loop generation ────────────────────────────────────────────

def run_verifier_loop(
    items: list[dict],
    switcher: ModelSwitcher,
    max_rounds: int = MAX_ROUNDS_DEFAULT,
    score_threshold: float = SCORE_THRESHOLD_DEFAULT,
    batch_size: int = 64,
    ckpt_path: Path | None = None,
    force_max_rounds: bool = False,
    condition_label: str = "Verifier-CoT",
) -> list[dict]:
    """Round-loop: VLM free-form statement -> verifier scores it on the
    fine-grained letter scale -> repeat (up to max_rounds, or until the score
    clears score_threshold) -> final classification turn for every image.

    force_max_rounds=True ignores the score threshold as a stopping condition
    and always runs every image to exactly max_rounds — the verifier still
    scores and responds every round exactly as normal (nothing about the
    prompt or generation changes), it just never gets to end the chain early.
    Useful for isolating "what does round N look like" from "which images
    survive to round N", since early-stopping normally shrinks the active set
    fast and later rounds become too small a sample to read anything into.
    """
    from src.verifier_vision import _RUBRIC_DISEASE, _RUBRIC_WEED

    _sys_msg, _cls_question, _desc_question, _build_prompt, _clean = _import_helpers()

    _VLM_P = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    # Pass 1: response JSON only — no logprobs needed, matches verifier_ppt.py.
    _VER_P = SamplingParams(temperature=1.0, max_tokens=512)
    # Pass 2: one grammar-constrained token continuing "<analysis>\n<score>" —
    # see verifier_ppt.py's module docstring for why this replaces reading
    # free-generation top-20 logprobs directly.
    _SCORE_P = SamplingParams(temperature=1.0, max_tokens=1, logprobs=20,
                              structured_outputs=StructuredOutputsParams(choice=_SCORE_LETTERS))
    _CLS_P = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)

    n = len(items)
    hist: list[list[dict]] = [[] for _ in range(n)]
    n_rounds = [0] * n
    final_statement = [""] * n
    final_verdict = [""] * n
    final_fine_grained_score: list[float | None] = [None] * n
    final_naive_score: list[float | None] = [None] * n
    # Full round-by-round record per item, for quality review — each entry is
    # {"round", "vlm_statement", "verifier_fine_grained_score",
    # "verifier_naive_score", "verifier_response", "verifier_extraction",
    # "round_prediction"}. Not just the final accepted statement.
    transcript: list[list[dict]] = [[] for _ in range(n)]

    def _img(i: int) -> PILImage.Image:
        return PILImage.open(items[i]["image_path"]).convert("RGB")

    # ── Round 1: initial free-form description (self_cot-style) ────────────────
    general_llm, general_tokenizer = switcher.get("general")
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
            prompts.append(_build_prompt(h, general_tokenizer, _img(i)))
        outputs = general_llm.generate(prompts, _VLM_P)
        for i, out in zip(batch_idx, outputs):
            hist[i].append({"role": "assistant", "content": _clean(out.outputs[0].text)})
        del prompts
        prog.update(len(batch_idx))

    active_idx = list(range(n))

    for rnd in range(1, max_rounds + 1):
        if not active_idx:
            break

        # ── Verify phase (verifier role, two-pass letter-scale scoring) ────────
        verifier_llm, verifier_tokenizer = switcher.get("verifier")
        sub_banner(f"Verifying round {rnd}/{max_rounds}: {len(active_idx)} active images")
        round_results: dict[int, dict] = {}
        prog = Progress(total=len(active_idx), label=f"Verify round {rnd}", every=max(batch_size, 1))
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            base_prompts = []
            for i in batch_idx:
                item = items[i]
                statement = hist[i][-1]["content"]
                rubric = _RUBRIC_WEED if item["task_type"] == "weed_identification" else _RUBRIC_DISEASE
                sys_p = _verifier_system(rubric)
                user_p = _verifier_user_turn("(none yet)", statement)
                msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": user_p},
                ]}]
                base_prompts.append(_build_prompt(msgs, verifier_tokenizer, _img(i)))

            first_outputs = verifier_llm.generate(base_prompts, _VER_P)

            analyses = []
            score_prompts = []
            for bp, out in zip(base_prompts, first_outputs):
                clean_text = _THINK_RE.sub("", out.outputs[0].text).strip()
                tag_idx = clean_text.lower().find("<score>")
                analysis = clean_text[:tag_idx].rstrip() if tag_idx != -1 else clean_text
                analyses.append(analysis)
                score_prompts.append(_build_score_prompt(bp, analysis))
            del base_prompts

            second_outputs = verifier_llm.generate(score_prompts, _SCORE_P)
            del score_prompts

            for i, analysis, second_out in zip(batch_idx, analyses, second_outputs):
                round_results[i] = parse_verifier_output_prefill(analysis, second_out)
            prog.update(len(batch_idx))

        # ── Round-classification snapshot (general role) ───────────────────────
        general_llm, general_tokenizer = switcher.get("general")
        round_predictions: dict[int, str] = {}
        prog = Progress(total=len(active_idx), label=f"Round {rnd} snapshot", every=max(batch_size, 1))
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            cls_prompts = [
                _build_prompt(
                    hist[i] + [{"role": "user", "content": _cls_question(items[i]["task_type"], items[i]["crop"])}],
                    general_tokenizer, _img(i),
                )
                for i in batch_idx
            ]
            cls_outputs = general_llm.generate(cls_prompts, _CLS_P)
            del cls_prompts
            for i, cls_out in zip(batch_idx, cls_outputs):
                round_predictions[i] = _clean(cls_out.outputs[0].text)
            prog.update(len(batch_idx))

        # ── Decide stop/continue per item ───────────────────────────────────────
        still_active = []
        n_ended_this_round = 0
        for i in active_idx:
            result = round_results[i]
            n_rounds[i] = rnd
            statement_this_round = hist[i][-1]["content"]
            fg = result["fine_grained_score"]
            transcript[i].append({
                "round": rnd,
                "vlm_statement": statement_this_round,
                "verifier_fine_grained_score": fg,
                "verifier_naive_score": result["naive_score"],
                "verifier_response": result["response"],
                "verifier_extraction": result["extraction"],
                "round_prediction": round_predictions[i],
            })
            score_ok = fg is not None and fg >= score_threshold
            stop_here = rnd == max_rounds or (score_ok and not force_max_rounds)
            if stop_here:
                final_statement[i] = statement_this_round
                final_verdict[i] = "SUPPORTED" if score_ok else "UNSUPPORTED_MAXROUNDS"
                final_fine_grained_score[i] = fg
                final_naive_score[i] = result["naive_score"]
                if score_ok:
                    n_ended_this_round += 1
            else:
                hist[i].append({"role": "user", "content": result["response"]})
                still_active.append(i)

        active_idx = still_active
        print(f"  {n_ended_this_round} ended this round (score >= {score_threshold})")
        if not active_idx or rnd == max_rounds:
            break

        # ── Generate round+1 statements (general role, already loaded above) ───
        sub_banner(f"Generating round {rnd + 1} statements: {len(active_idx)} images")
        prog = Progress(total=len(active_idx), label=f"Round {rnd + 1} generation", every=max(batch_size, 1))
        for start in range(0, len(active_idx), batch_size):
            batch_idx = active_idx[start:start + batch_size]
            prompts = [_build_prompt(hist[i], general_tokenizer, _img(i)) for i in batch_idx]
            outputs = general_llm.generate(prompts, _VLM_P)
            for i, out in zip(batch_idx, outputs):
                hist[i].append({"role": "assistant", "content": _clean(out.outputs[0].text)})
            del prompts
            prog.update(len(batch_idx))

    # ── Final classification turn for every image (matches self_cot/cot_yaml) ──
    general_llm, general_tokenizer = switcher.get("general")
    sub_banner(f"Final classification: {n} images")
    predictions: list[str] = [""] * n
    prog = Progress(total=n, label="Final classification", every=max(batch_size, 1))
    for start in range(0, n, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n)))
        prompts = []
        for i in batch_idx:
            item = items[i]
            h = hist[i] + [{"role": "user", "content": _cls_question(item["task_type"], item["crop"])}]
            prompts.append(_build_prompt(h, general_tokenizer, _img(i)))
        outputs = general_llm.generate(prompts, _CLS_P)
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
            f"  Verifier score: fine_grained={t['verifier_fine_grained_score']}, "
            f"naive={t['verifier_naive_score']} ({t['verifier_extraction']})\n"
            f"  Verifier response: {t['verifier_response']}"
            for t in transcript[i]
        )
        rows.append({
            "agml_dataset": item["ds"],
            "class_label":  item["cls"],
            "task_type":    item["task_type"],
            "crop":         item.get("crop") or "",
            "image_name":   item["image_name"],
            "condition":    condition_label,
            "prediction":   predictions[i],
            "reasoning":    reasoning or final_statement[i] or fallback_statement,
            "ground_truth": item["cls"],
            "judged_correct": "",
            "judge_reasoning": "",
            "n_rounds": n_rounds[i],
            "final_verifier_verdict": final_verdict[i],
            "final_verifier_fine_grained_score": final_fine_grained_score[i],
            "final_verifier_naive_score": final_naive_score[i],
            "transcript_json": json.dumps(transcript[i]),
            "full_history_json": json.dumps(hist[i]),
        })

    if ckpt_path:
        _write_csv(rows, ckpt_path, CLS_FIELDS)
    return rows


# ── Plots ──────────────────────────────────────────────────────────────────────

def build_rows_by_condition(
    verifier_rows: list[dict], existing: dict[str, list[dict]], condition_label: str = "Verifier-CoT",
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {condition_label: verifier_rows}
    for label, rows in existing.items():
        if rows:
            result[label] = rows
    return result


def plot_all(
    judged: list[dict], existing: dict[str, list[dict]], output_dir: Path, condition_label: str = "Verifier-CoT",
) -> None:
    from src.common.method_comparison import plot_overall_judge_comparison, plot_per_task_judge_comparison
    from src.verifier_vision import plot_per_dataset

    rows_by_cond = build_rows_by_condition(judged, existing, condition_label=condition_label)
    present = [c for c in rows_by_cond if rows_by_cond[c]]
    print(f"\n[plot] conditions: {present}")

    plot_overall_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_overall.png",
        title=f"{condition_label} vs Baselines — Overall Mean Judged F1",
    )
    plot_per_task_judge_comparison(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_per_task.png",
        title=f"{condition_label} vs Baselines — Mean Judged F1 by Task",
    )
    plot_per_dataset(
        rows_by_cond,
        output_path=output_dir / "verifier_cot_per_dataset.png",
    )

    verdicts = [r["final_verifier_verdict"] for r in judged]
    from collections import Counter
    print(f"\n[verifier verdict distribution] {dict(Counter(verdicts))}")

    # Score analysis (score_by_round.png, final_score_distribution.png,
    # accuracy_vs_score.png, confidence_analysis_summary.json) — same directory
    # as everything else here, run automatically rather than requiring a separate
    # manual invocation of analyze_confidence.py after the fact.
    print("\n[plot] score analysis...")
    from analyze_confidence import make_charts as make_score_charts
    score_summary = make_score_charts(judged, output_dir)
    score_summary_path = output_dir / "confidence_analysis_summary.json"
    score_summary_path.write_text(json.dumps(score_summary, indent=2))
    print(f"  Saved: {score_summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier-CoT — prompt-only verifier-in-the-loop experiment")
    parser.add_argument("--model",       required=True, help="HuggingFace model ID (general/diagnostic model)")
    parser.add_argument("--verifier_model", default=None,
                        help="If set, a SEPARATE model plays the verifier role instead of --model. "
                             "Swapped in/out at each round boundary (see ModelSwitcher) — omit to "
                             "keep the original single-model self-verification behavior.")
    parser.add_argument("--condition_label", default="Verifier-CoT",
                        help="Label for this run's condition in the CSV/charts, e.g. "
                             "'Verifier-CoT (Qwen-base)' — lets multiple --verifier_model runs "
                             "appear as distinct bars when combined (see CONDITION_STYLES).")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="vLLM max_model_len for every model load. Raised from cot_vision."
                             "load_llm's 8192 default — see run_verifier_ppt.py's --max_model_len "
                             "help for why (Qwen-family vision tokenizers are resolution-dependent).")
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
    parser.add_argument("--score_threshold", type=float, default=SCORE_THRESHOLD_DEFAULT,
                        help="Fine-grained verifier score [0,1] that ends the chain early. "
                             "Still capped at --max_rounds regardless of score.")
    parser.add_argument("--force_max_rounds", action="store_true",
                        help="Ignore the score threshold as a stopping condition — every image "
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

    if not _HAS_LMV:
        print("ERROR: llm-verifier package not installed. Run: uv add llm-verifier")
        return

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
    switcher = ModelSwitcher(args.model, args.verifier_model, args.max_model_len)
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
        verifier_rows = run_verifier_loop(
            all_items, switcher, max_rounds=args.max_rounds, score_threshold=args.score_threshold,
            batch_size=args.batch_size, ckpt_path=ckpt_verifier_cot,
            force_max_rounds=args.force_max_rounds, condition_label=args.condition_label,
        )
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
        judge_llm, judge_tokenizer = switcher.get("general")
        judged = apply_judge(
            verifier_rows, judge_llm, judge_tokenizer,
            system_prompt=_p["judge_system_vision"],
            judge_template=_p["judge_classification_template"],
            gt_key="ground_truth", pred_key="prediction",
        )
        _write_csv(judged, ckpt_judged, CLS_FIELDS)
        print(f"\nStage 2 done in {fmt_duration(time.perf_counter() - t_stage2)}")

    if need_model:
        elapsed = time.perf_counter() - t0
        print(f"\nInference done in {fmt_duration(elapsed)} ({switcher.n_reloads} model reloads)")
        switcher.destroy()

    image_keys = {(r["agml_dataset"], r["image_name"]) for r in judged}
    stage_banner(3, f"Plot ({len(image_keys)} unique images)", total_stages=3)

    existing = load_existing_rows(
        image_keys,
        vis_dir=Path(args.vis_dir),
        cot_dir=Path(args.cot_dir),
        aug_dir=Path(args.aug_dir),
        model_safe=model_safe,
    )

    plot_all(judged, existing, output_dir, condition_label=args.condition_label)
    print("\nDone.")


if __name__ == "__main__":
    main()
