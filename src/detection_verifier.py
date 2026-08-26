"""
Stage 5d — Detection Verifier: adapts verifier_cot.py's single-round
K-candidate + PPT-tournament verifier (Stage 4) to bounding-box
localization instead of free-text classification. See
docs/stage5_object_detection.md and src/config/verifier_prompts_detection.
yaml's header comment for the full design rationale — summarized here:

K candidates are generated from the SAME structured-template pattern
verifier_cot.py uses for classification: the model answers a fixed set of
localization criteria (object/class, instance count, location, occlusion,
size) BEFORE printing its boxes — the explicit design requirement that
separates this condition from detection_baseline.py (Stage 5b), which
prints boxes directly with no reasoning at all. That one pool of K is
ranked exactly once by the FULL, unmodified Probabilistic Pivot Tournament
(llm_verifier.pivot_tournament, same library verifier_cot.py/verifier_lmv.py
use) via src/common/ppt_scoring.py's two-pass prefill scoring over full
TEXT trajectories against a rubric — this needs NO changes for detection,
since a trajectory (criteria answers + box list) is still just text. No
incumbent, no persistent clusters, no stopping rule; there is only one
round (2026-08-23 simplification, mirroring verifier_cot.py's own — see
that module's docstring for the evidence behind dropping the multi-round
machinery).

Unlike classification (a single scalar label — argmax over the winner is
the whole answer), a spatial prediction benefits from POOLING evidence:
different candidates often localize different real instances the others
missed entirely, not just disagree about the SAME instance's box. So
instead of reporting only the single top-ranked candidate's own box list,
the top --top_k candidates by tournament win-rate (default 1, i.e. the
original single-winner behavior) have their boxes pooled together and
deduplicated via class-agnostic NMS (src/common/bbox_metrics.nms) before
scoring — see run_group's docstring. Since these box outputs have no
per-box confidence score (see detection_baseline.py's docstring), NMS
priority is each box's SOURCE CANDIDATE's own tournament win-rate reused
as a per-box proxy, not a genuine confidence value.

Ground-truth IoU/precision/recall (src/common/bbox_metrics.py) is computed
and logged for the final pooled+deduplicated prediction — the metric that
answers the research question, analogous to judge_f1_macro for
classification.

Usage:
    uv run python main.py detection-verifier --model google/gemma-4-E4B-it
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import re
import time

from PIL import Image as PILImage

from src.common.bbox_metrics import match_boxes, nms, precision_recall_f1
from src.common.bbox_parse import clamp_box, denormalize_box_1000, extract_boxes, needs_1000_scale_boxes
from src.common.chat_prompt import render_chat_template
from src.common.dataset_config import active_datasets, crop_display, load_hf_detection_samples, load_manifest
from src.common.io import append_jsonl, load_prompts
from src.common.ppt_scoring import (
    analysis_before_tags, build_score_prompt, extract_tag_score,
    pairwise_scoring_prompt, score_prefill_params,
)
from src.common.progress import ProgressBar

try:
    from llm_verifier.pivot_tournament import (
        DEFAULT_PIVOTS, bradley_terry, pivot_round_pairs, ring_cycle, select_pivots,
    )
except ImportError:
    DEFAULT_PIVOTS = 2
    bradley_terry = pivot_round_pairs = ring_cycle = select_pivots = None  # type: ignore[assignment]
from src.common.vision_eval import append_ckpt_row, archive_ckpt_if_rerun, load_ckpt, safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_VERIFIER_PROMPTS_DETECTION_PATH = Path(__file__).parent / "config" / "verifier_prompts_detection.yaml"
_VERIFIER_PROMPTS_PATH = Path(__file__).parent / "config" / "verifier_prompts.yaml"  # pairwise_prompt, reused verbatim

DET_VERIFIER_CKPT_FIELDS = [
    "agml_dataset", "class_label", "image_name",
    "pool_size", "top_k", "n_comparisons", "top1_win_rate", "verifier_feedback",
    "pred_boxes_json", "gt_boxes_json",
    "n_pred", "n_gt", "n_matched", "precision_at_iou50", "recall_at_iou50",
    "f1_at_iou50", "mean_iou_matched",
    "prompt_tokens", "completion_tokens", "elapsed_s", "avg_time_per_sample_s",
]


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    m = _REPEAT_RE.search(text)
    return text[:m.start()].strip() if m else text


def _sys_msg(prompts: dict) -> list[dict]:
    s = prompts.get("assessment_system", "")
    return [{"role": "system", "content": s}] if s else []


def _format_det_question(names: list[str], crop: str) -> str:
    names_str = ", ".join(names) if names else "object"
    return f"Identify and localize every {names_str} visible in this image of {crop}."


def det_question(ds) -> str:
    """`ds` is the dataset's DatasetConfig — needs both its class list (with
    any configured aliases folded in, see classes_with_aliases_display) and
    its crop, same convention as detection_baseline.py's det_question."""
    return _format_det_question(ds.classes_with_aliases_display(), crop_display(ds.crop))


def generate_class_name_variants(
    llm, tokenizer, class_names: list[str], n_variants: int, call_stats: dict | None = None,
) -> list[list[str]]:
    """One batched, text-only, no-image call asking for n_variants alternate
    name-sets for class_names — e.g. common names, synonyms, or plainer
    phrasings of a cryptic/coded dataset label (confirmed on real data this
    session: labels like "b_fully_ripened", "l_green" are dataset-internal
    codes, not always names the model's own vocabulary readily recognizes).
    Returns a list of n_variants lists, each the same length as
    class_names, so candidate k's own generation prompt can search using
    variant k's names instead of every candidate sharing one static name
    list — this DIVERSIFIES what each candidate searches for (more real
    instances found across the pooled top_k, see run_group's docstring),
    unlike classes_with_aliases_display's static "show every known alias to
    every candidate" approach, which doesn't vary anything across K.

    Falls back to n_variants copies of the original class_names on any
    parse failure — never blocks generation just because this optional
    step didn't parse."""
    names_str = ", ".join(class_names)
    prompt_text = (
        f"For each of these object/class names: {names_str}\n"
        f"Provide {n_variants} DIFFERENT alternate ways to refer to each one — "
        f"draw on whichever of these actually differ from the original and from "
        f"each other: common/everyday names, synonyms, the scientific or "
        f"botanical name, more specific terms (e.g. the specific part of the "
        f"plant, like \"pod\" or \"fruit\"), or — only if the given name looks "
        f"like a coded/abbreviated dataset label — a plainer description of it. "
        f"Even if the given name is already a common, everyday word, still "
        f"provide {n_variants} genuinely different alternates for it — do not "
        f"just repeat the original name across variants.\n"
        f"Respond with JSON only: a list of exactly {n_variants} lists, each containing "
        f"one alternate name per class, in the same order as given. Example for 2 "
        f'classes and 3 variants: [["name1a","name2a"],["name1b","name2b"],'
        f'["name1c","name2c"]]'
    )
    prompt = {"prompt": render_chat_template(tokenizer, [{"role": "user", "content": prompt_text}])}
    params = SamplingParams(temperature=0.7, max_tokens=300)
    outputs = llm.generate([prompt], sampling_params=params)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + 1

    fallback = [list(class_names) for _ in range(n_variants)]
    text = outputs[0].outputs[0].text
    try:
        start, end = text.find("["), text.rfind("]")
        parsed = json.loads(text[start:end + 1])
        variants = [
            [str(x) for x in v] for v in parsed
            if isinstance(v, list) and len(v) == len(class_names)
        ]
    except (ValueError, TypeError, json.JSONDecodeError):
        variants = []
    while len(variants) < n_variants:
        variants.append(fallback[len(variants)])
    return variants[:n_variants]


# ── Generation (K candidates) ────────────────────────────────────────────────

def run_batch_n(llm, tokenizer, images: list, histories: list[list[dict]], params,
                 call_stats: dict | None = None) -> list[list[dict]]:
    """n completions per prompt — verbatim port of verifier_cot.py's
    run_batch_n (fully domain-agnostic, no changes needed)."""
    prompts = [{"prompt": render_chat_template(tokenizer, h), "multi_modal_data": {"image": img}}
               for img, h in zip(images, histories)]
    outputs = llm.generate(prompts, sampling_params=params)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(prompts)
    results = []
    for o in outputs:
        p_tok = len(getattr(o, "prompt_token_ids", None) or [])
        results.append([
            {"text": _clean(c.text), "prompt_tokens": p_tok,
             "completion_tokens": len(getattr(c, "token_ids", None) or [])}
            for c in o.outputs
        ])
    return results


def _log_verifier_call(log_path: Path | None, **fields) -> None:
    if log_path is None:
        return
    append_jsonl(log_path, fields)


# ── PPT pairwise scoring (text trajectories vs. rubric — unchanged mechanism) ─

def _score_pairs_batch(
    llm, tokenizer, images, active, pools, pair_by_image, prompts, vprompts, cls_q, rubric,
    rng, analysis_params, score_params, call_stats, agml_dataset, class_label, image_names,
    log_path,
):
    """Verbatim port of verifier_cot.py's _score_pairs_batch — scores one
    pool-index pair per image via ppt_scoring.py's two-pass prefill
    mechanism. Domain-agnostic over trajectory text, so no changes needed
    for detection trajectories."""
    base_prompts, owners = [], []
    for i in active:
        a, b = pair_by_image[i]
        text_a, text_b = pools[i][a], pools[i][b]
        swapped = rng.random() < 0.5
        traj_a, traj_b = (text_b, text_a) if swapped else (text_a, text_b)
        pw_text = pairwise_scoring_prompt(rubric=rubric, question=cls_q, trajectory_a=traj_a, trajectory_b=traj_b,
                                           template=vprompts["pairwise_prompt"])
        msgs = _sys_msg(prompts) + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": pw_text},
        ]}]
        base_prompts.append({
            "prompt": render_chat_template(tokenizer, msgs),
            "multi_modal_data": {"image": images[i]},
        })
        owners.append((i, a, b, swapped, pw_text))

    analysis_outputs = llm.generate(base_prompts, sampling_params=analysis_params)
    if call_stats is not None:
        call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
        call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(base_prompts)
    analyses = [analysis_before_tags(out.outputs[0].text) for out in analysis_outputs]
    pass_tokens = [
        [len(getattr(out, "prompt_token_ids", None) or []),
         len(getattr(out.outputs[0], "token_ids", None) or [])]
        for out in analysis_outputs
    ]

    scores_a: list[float | None] = [None] * len(owners)
    scores_b: list[float | None] = [None] * len(owners)
    for tag, scores in (("<score_A>", scores_a), ("<score_B>", scores_b)):
        score_prompts = [build_score_prompt(bp, an, tag) for bp, an in zip(base_prompts, analyses)]
        score_outputs = llm.generate(score_prompts, sampling_params=score_params)
        if call_stats is not None:
            call_stats["n_calls"] = call_stats.get("n_calls", 0) + 1
            call_stats["n_prompts"] = call_stats.get("n_prompts", 0) + len(score_prompts)
        for idx, (analysis, out) in enumerate(zip(analyses, score_outputs)):
            comp = out.outputs[0]
            scores[idx] = extract_tag_score(analysis, tag, comp.text, comp.logprobs or [])
            pass_tokens[idx][0] += len(getattr(out, "prompt_token_ids", None) or [])
            pass_tokens[idx][1] += len(getattr(comp, "token_ids", None) or [])

    result: dict[int, tuple[float, float, str, list[int]]] = {}
    for idx, (i, a, b, swapped, pw_text) in enumerate(owners):
        sa, sb = scores_a[idx], scores_b[idx]
        r_a, r_b = (sb, sa) if swapped else (sa, sb)
        result[i] = (r_a, r_b, analyses[idx], pass_tokens[idx])
        _log_verifier_call(
            log_path, agml_dataset=agml_dataset, class_label=class_label,
            image_name=image_names[i], call_type="pairwise",
            pool_a=a, pool_b=b, swapped=swapped, prompt=pw_text, analysis=analyses[idx],
            score_a=r_a, score_b=r_b,
        )
    return result


# ── Single-round tournament ──────────────────────────────────────────────────

def run_group(
    llm, tokenizer, images: list, cls_q: str, gen_template: str, rubric: str, prompts: dict, vprompts: dict,
    gt_boxes_list: list[list[dict]], k_candidates: int, k_pivots: int, gen_temperature: float,
    rng: random.Random, agml_dataset: str, class_label: str, image_names: list[str],
    log_path: Path | None = None, call_stats: dict | None = None, seed: int | None = None,
    gen_max_tokens: int = 220, iou_threshold: float = 0.5, rescale_1000: bool = False,
    top_k: int = 1, nms_iou_threshold: float = 0.5,
    dynamic_synonyms: bool = False, class_names: list[str] | None = None, crop: str | None = None,
) -> list[dict]:
    """Detection analog of verifier_cot.py's run_group — generate K
    candidates, run the existing, unmodified PPT tournament exactly once
    over that pool. Unlike classification (where argmax over one winner
    makes sense — there's only one right label), a spatial prediction
    benefits from POOLING evidence across candidates: different
    candidates often catch different real instances the others missed,
    so instead of reporting only the single top-ranked candidate's own
    box list, the top `top_k` candidates by tournament win-rate have
    their boxes POOLED together, then deduplicated via class-agnostic NMS
    (src/common/bbox_metrics.nms) before being scored against ground
    truth. top_k=1 recovers the original single-winner behavior exactly
    (NMS over one candidate's own boxes is a no-op unless that candidate
    duplicated a box for the same instance). Returns one result dict per
    image.

    rescale_1000: True for model families confirmed to report boxes on
    Gemma's native 0-1000 normalized scale rather than real pixels (see
    bbox_parse.needs_1000_scale_boxes) — applied to every candidate's
    boxes before clamping, same fix as detection_baseline.py.

    nms_iou_threshold: overlap threshold NMS uses to collapse duplicate
    boxes from the pooled top_k candidates — a SEPARATE knob from
    iou_threshold (which only scores the final result against ground
    truth), since there's no reason the two need to match.

    dynamic_synonyms: when True (needs class_names/crop given), one extra
    text-only call (generate_class_name_variants) produces k_candidates
    alternate name-sets for class_names, and each candidate's OWN
    generation prompt searches using a different variant instead of every
    candidate sharing one static class-name list — diversifies what each
    candidate looks for (a cryptic dataset label like "b_fully_ripened"
    might only be recognized under one of its plainer phrasings), which
    the existing top_k pooling + NMS step can then combine. Because each
    candidate now has its own distinct prompt rather than one prompt
    sampled k_candidates times, generation runs as k_candidates separate
    n=1 calls (batched together in one llm.generate) instead of a single
    n=k_candidates call."""
    t_start = time.perf_counter()
    n = len(images)
    active = list(range(n))

    if dynamic_synonyms:
        variants = generate_class_name_variants(llm, tokenizer, class_names or [], k_candidates, call_stats)
        gen_qs = [gen_template.format(question=_format_det_question(v, crop or "plant")) for v in variants]

        flat_images, flat_hist = [], []
        for i in range(n):
            for k in range(k_candidates):
                flat_images.append(images[i])
                flat_hist.append(_sys_msg(prompts) + [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": gen_qs[k]},
                ]}])
        gen_params = SamplingParams(
            temperature=gen_temperature, max_tokens=gen_max_tokens,
            repetition_penalty=1.1, n=1, seed=seed,
        )
        flat_out = run_batch_n(llm, tokenizer, flat_images, flat_hist, gen_params, call_stats)
        gen_out = {i: [flat_out[i * k_candidates + k][0] for k in range(k_candidates)] for i in active}
    else:
        gen_q = gen_template.format(question=cls_q)
        hist = [_sys_msg(prompts) + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": gen_q},
        ]}] for _ in range(n)]
        gen_params = SamplingParams(
            temperature=gen_temperature, max_tokens=gen_max_tokens,
            repetition_penalty=1.1, n=k_candidates, seed=seed,
        )
        gen_out = {i: c for i, c in enumerate(run_batch_n(llm, tokenizer, images, hist, gen_params, call_stats))}

    pools: dict[int, list[str]] = {i: [c["text"] for c in gen_out[i]] for i in active}
    pool_boxes: dict[int, list[list[dict]]] = {}
    for i in active:
        img = images[i]
        cand_lists = []
        for c in gen_out[i]:
            cand_boxes = []
            for b in extract_boxes(c["text"]):
                bbox = denormalize_box_1000(b["bbox"], img.width, img.height) if rescale_1000 else b["bbox"]
                cand_boxes.append({**b, "bbox": clamp_box(bbox, img.width, img.height)})
            cand_lists.append(cand_boxes)
        pool_boxes[i] = cand_lists

    # Each candidate has its own prompt_tokens in dynamic_synonyms mode
    # (different name variants = different prompt lengths), so the total
    # per image is a genuine sum across candidates rather than one shared
    # prompt's token count reused k_candidates times.
    prompt_tokens = {i: sum(c["prompt_tokens"] for c in gen_out[i]) if dynamic_synonyms
                      else gen_out[i][0]["prompt_tokens"] for i in active}
    completion_tokens = {i: sum(c["completion_tokens"] for c in gen_out[i]) for i in active}

    for i in active:
        for cand_idx, c in enumerate(gen_out[i]):
            _log_verifier_call(
                log_path, agml_dataset=agml_dataset, class_label=class_label,
                image_name=image_names[i], call_type="candidate",
                candidate_index=cand_idx, raw_text=c["text"],
                extracted_boxes=pool_boxes[i][cand_idx],
                prompt_tokens=c["prompt_tokens"], completion_tokens=c["completion_tokens"],
            )

    pool_size = k_candidates
    w = {i: [0.0] * pool_size for i in active}
    c = {i: [0] * pool_size for i in active}
    n_comparisons = {i: 0 for i in active}
    first_analysis: dict[int, dict[int, str]] = {i: {} for i in active}

    analysis_params = SamplingParams(temperature=1.0, max_tokens=150, seed=seed)
    score_params = score_prefill_params()

    def _accumulate(pair_by_image: dict[int, tuple[int, int]]) -> None:
        result = _score_pairs_batch(
            llm, tokenizer, images, active, pools, pair_by_image, prompts, vprompts, cls_q, rubric, rng,
            analysis_params, score_params, call_stats, agml_dataset, class_label, image_names, log_path,
        )
        for i in active:
            r_a, r_b, analysis, tok = result[i]
            a, b = pair_by_image[i]
            p = bradley_terry(r_a, r_b)
            w[i][a] += p
            c[i][a] += 1
            w[i][b] += 1.0 - p
            c[i][b] += 1
            n_comparisons[i] += 1
            first_analysis[i].setdefault(a, analysis)
            first_analysis[i].setdefault(b, analysis)
            prompt_tokens[i] += tok[0]
            completion_tokens[i] += tok[1]

    if pool_size > 1:
        rings = {i: ring_cycle(pool_size, rng) for i in active}
        for t in range(pool_size):
            _accumulate({i: rings[i][t] for i in active})
        pivots = {i: select_pivots(w[i], c[i], k_pivots) for i in active}
        pivot_pairs = {i: pivot_round_pairs(pool_size, pivots[i]) for i in active}
        n_pivot_pairs = len(next(iter(pivot_pairs.values())))
        for t in range(n_pivot_pairs):
            _accumulate({i: pivot_pairs[i][t] for i in active})

    elapsed = time.perf_counter() - t_start
    avg_time_per_sample_s = elapsed / n if n else 0.0

    effective_top_k = min(top_k, pool_size)
    results = []
    for i in active:
        if pool_size == 1:
            win_rate, top1 = [0.5], 0
        else:
            win_rate = [w[i][k] / c[i][k] if c[i][k] else 0.0 for k in range(pool_size)]
            top1 = max(range(pool_size), key=lambda k: (win_rate[k], -k))
        ranked = sorted(range(pool_size), key=lambda k: (-win_rate[k], k))
        top_idx = ranked[:effective_top_k]

        pooled = [
            {**b, "priority": win_rate[k]}
            for k in top_idx for b in pool_boxes[i][k]
        ]
        deduped = nms(pooled, iou_threshold=nms_iou_threshold)
        report_boxes = [{"bbox": b["bbox"], "category_name": b.get("category_name", "")} for b in deduped]

        gt_boxes = gt_boxes_list[i]
        matches, unmatched_pred, unmatched_gt = match_boxes(report_boxes, gt_boxes, iou_threshold)
        iou_metrics = precision_recall_f1(matches, unmatched_pred, unmatched_gt)
        results.append({
            "pool_size": pool_size, "top_k": effective_top_k, "n_comparisons": n_comparisons[i],
            "top1_win_rate": round(win_rate[top1], 4),
            "verifier_feedback": first_analysis[i].get(top1, ""),
            "pred_boxes_json": json.dumps(report_boxes), "gt_boxes_json": json.dumps(gt_boxes),
            "n_pred": iou_metrics["n_pred"], "n_gt": iou_metrics["n_gt"], "n_matched": iou_metrics["n_matched"],
            "precision_at_iou50": iou_metrics["precision"], "recall_at_iou50": iou_metrics["recall"],
            "f1_at_iou50": iou_metrics["f1"], "mean_iou_matched": iou_metrics["mean_iou_matched"],
            "prompt_tokens": prompt_tokens[i], "completion_tokens": completion_tokens[i],
            "elapsed_s": round(elapsed, 3), "avg_time_per_sample_s": round(avg_time_per_sample_s, 3),
        })
    return results


def _result_row(entry: dict, agml_dataset: str, class_label: str, name: str) -> dict:
    return {"agml_dataset": agml_dataset, "class_label": class_label, "image_name": name, **entry}


# ── Main run ─────────────────────────────────────────────────────────────────

def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    vprompts = _load_yaml(_VERIFIER_PROMPTS_DETECTION_PATH)
    vprompts["pairwise_prompt"] = _load_yaml(_VERIFIER_PROMPTS_PATH)["pairwise_prompt"]  # reused verbatim
    manifest = load_manifest(Path(args.manifest))
    rescale_1000 = needs_1000_scale_boxes(model_id)
    if rescale_1000:
        print(f"  [bbox] {model_id}: rescaling predicted boxes from Gemma's native 0-1000 "
              f"normalized scale to real pixel coordinates (see bbox_parse.needs_1000_scale_boxes)")

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    log_path = model_dir / "verifier_log.jsonl"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=DET_VERIFIER_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} row(s) already scored in {ckpt_path.name}")
    seen_images = {(r["agml_dataset"], r["image_name"]) for r in rows}

    groups = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip:
            continue
        try:
            samples = load_hf_detection_samples(ds.agml_name, manifest, Path(args.images_root))
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest")
            continue
        pending = [s for s in samples if (ds.agml_name, Path(s["path"]).name) not in seen_images]
        if not pending:
            continue
        groups.append((ds, pending))

    if not groups:
        print("  Nothing to do — all samples already checkpointed.")
    else:
        n_pending = sum(len(g[1]) for g in groups)
        print(f"  {len(groups)} dataset group(s), {n_pending} pending sample(s)")

        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            caller_tag="detection_verifier",
        )

        rng = random.Random(args.seed)
        call_stats: dict = {}
        gen_template = (vprompts["direct_template"] if args.gen_mode == "direct"
                         else vprompts["structured_templates"]["object_detection"])
        rubric = vprompts["rubric"]

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Datasets")
        for ds, samples in groups:
            pbar.desc = f"Datasets ({ds.agml_name})"
            images = [PILImage.open(s["path"]).convert("RGB") for s in samples]
            names = [Path(s["path"]).name for s in samples]
            gt_boxes_list = [s["boxes"] for s in samples]
            primary_labels = [s["boxes"][0]["category_name"] if s["boxes"] else "" for s in samples]
            cls_q = det_question(ds)

            results = run_group(
                llm, tokenizer, images, cls_q, gen_template, rubric, prompts, vprompts, gt_boxes_list,
                k_candidates=args.k_candidates, k_pivots=args.k_pivots,
                gen_temperature=args.temperature, rng=rng,
                agml_dataset=ds.agml_name, class_label="", image_names=names,
                log_path=log_path, call_stats=call_stats, seed=args.seed,
                gen_max_tokens=args.gen_max_tokens, iou_threshold=args.iou_threshold,
                rescale_1000=rescale_1000, top_k=args.top_k, nms_iou_threshold=args.nms_iou_threshold,
                dynamic_synonyms=args.dynamic_synonyms, class_names=ds.clean_classes, crop=crop_display(ds.crop),
            )

            for name, label, entry in zip(names, primary_labels, results):
                row = _result_row(entry, ds.agml_name, label, name)
                append_ckpt_row(ckpt_path, row, fields=DET_VERIFIER_CKPT_FIELDS)
                rows.append(row)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s "
              f"({call_stats.get('n_calls', 0)} calls, {call_stats.get('n_prompts', 0)} prompts)")
        destroy_llm(llm)
        del tokenizer

    if rows:
        mean_p = sum(float(r["precision_at_iou50"]) for r in rows) / len(rows)
        mean_r = sum(float(r["recall_at_iou50"]) for r in rows) / len(rows)
        mean_f1 = sum(float(r["f1_at_iou50"]) for r in rows) / len(rows)
        mean_iou = sum(float(r["mean_iou_matched"]) for r in rows) / len(rows)
        metrics = {"n_total": len(rows), "mean_precision_at_iou50": round(mean_p, 4),
                   "mean_recall_at_iou50": round(mean_r, 4), "mean_f1_at_iou50": round(mean_f1, 4),
                   "mean_iou_matched": round(mean_iou, 4)}
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Overall: precision={mean_p:.3f} recall={mean_r:.3f} "
              f"f1={mean_f1:.3f} mean_iou={mean_iou:.3f} (n={len(rows)})")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    run_model(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Single model to run.")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts_detection.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images_detection.json")
    parser.add_argument("--images_root", default="dataset/images_detection")
    parser.add_argument("--output_dir", default="outputs_final/detection_verifier/")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--k_candidates", type=int, default=4,
                         help="Candidates generated per image (K) — the single tournament pool size.")
    parser.add_argument("--gen_mode", choices=["structured", "direct"], default="structured",
                         help="'structured' (default): each candidate answers the fill-in-the-blank "
                              "localization criteria (verifier_prompts_detection.yaml's "
                              "structured_templates) before printing boxes. 'direct': candidates print "
                              "boxes straight away, same instruction shape as detection_baseline.py's "
                              "own prompt, no criteria step at all. Either way the PPT tournament + "
                              "rubric-based pairwise scoring is unchanged — this only controls what a "
                              "candidate's own generation call looks like, isolating whether the "
                              "structured-criteria GENERATION step itself helps or hurts, independent "
                              "of whether K-candidate sampling + tournament selection helps.")
    parser.add_argument("--k_pivots", type=int, default=DEFAULT_PIVOTS,
                         help="k — pivot-set size for the full PPT tournament (arXiv:2607.05391 "
                              "Section 3.2, via llm_verifier.pivot_tournament) that ranks the pool.")
    parser.add_argument("--top_k", type=int, default=1,
                         help="How many of the K candidates (ranked by PPT tournament win-rate) to "
                              "POOL together before deduplicating with NMS, instead of reporting only "
                              "the single top-ranked candidate's own box list — different candidates "
                              "often catch different real instances the others missed, so combining "
                              "the top_k candidates' boxes can raise recall before NMS collapses "
                              "duplicates. top_k=1 (default) recovers the original single-winner "
                              "behavior exactly. Clamped to K if set higher than --k_candidates.")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5,
                         help="Overlap threshold NMS uses to collapse duplicate boxes from the pooled "
                              "top_k candidates (src/common/bbox_metrics.nms) — separate from "
                              "--iou_threshold, which only scores the final deduplicated result against "
                              "ground truth.")
    parser.add_argument("--dynamic_synonyms", action="store_true",
                         help="One extra text-only call per (dataset, class) group generates "
                              "--k_candidates alternate name-sets for the target classes (common names, "
                              "synonyms, or plainer phrasings of a cryptic/coded dataset label), and each "
                              "candidate's own generation prompt searches using a different variant "
                              "instead of every candidate sharing one static class-name list. Diversifies "
                              "what each candidate looks for, meant to be combined with --top_k > 1 so "
                              "the pooled+NMS step can recover instances only found under one name "
                              "variant. Off by default (every candidate uses the same class name list).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gen_max_tokens", type=int, default=220,
                         help="Larger than classification's 180 — detection trajectories carry a box "
                              "list on top of the same criteria fill-in-the-blank fields.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")

    main(parser.parse_args())
