"""
Stage 2d — CoT-YAML (structured multi-step reasoning): a fixed sequence of
task-specific analysis questions (src/config/cot_prompts.yaml), each its own
conversation turn building on the last, before a final classification turn
that has the whole accumulated history as context. Unlike self_cot.py (one
open-ended "describe this" turn), the model is walked through the same
structured checklist every time for a given task type.

Conversation per image (N = len(steps) for that task's cot group, from
cot_prompts.yaml's task_type_to_cot routing):
  Turn 1..N: [image] + step[i] -> observation (step 1 also carries the image;
             steps 2..N are text-only follow-ups in the same history)
  Turn N+1:  [image, full step history] + classification question -> answer

Classification-only scoring: no separate description-quality metric — the
concatenated Q&A step trail is stored in the `reasoning` column for
inspection, scored only via what it does to the final classification (F1,
ROUGE-SS vs ground truth class label, judge_accuracy).

Per-model output subfolder + a per-sample checkpoint CSV (flushed to disk
after each (dataset, class) group's full step sequence — same granularity as
vision_baseline.py / self_cot.py) so a run on a time-limited remote job can
resume exactly where it left off:
  <output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics.json

Run with --collate (no --model needed) once every model's run has finished,
to combine every <output_dir>/<model>/ folder into:
  <output_dir>/collated_summary.csv
  <output_dir>/collated_metrics.json

--per_step: also classify after EVERY intermediate step, not just the last,
so you can see how F1 evolves as reasoning accumulates. Adds a second
checkpoint (see STEP_CKPT_FIELDS below):
  <output_dir>/<safe_model_name>/ckpt_steps_<safe_model_name>.csv
View the resulting per-step F1 with `main.py cot-yaml-step-report`.

--swap_datasets: restricts to the given datasets and re-runs --per_step-style
generation once per rotation of the step order (so every question gets a
turn going first), to check the per-step F1 curve isn't just an artifact of
question order. Meant for a small, targeted set (e.g. the top gain/loss
datasets from `main.py cot-compare`) since it multiplies generation cost by
the number of steps. Writes a THIRD checkpoint instead of the normal run:
  <output_dir>/<safe_model_name>/ckpt_swap_<safe_model_name>.csv

All three checkpoints share the same judge implementation
(run_judge_phase / apply_judge in src/common/vision_eval.py and
src/common/metrics.py) as vision_baseline.py / desc_augmented_vision_v2.py /
self_cot.py — never a reimplementation per script.

Usage (one model at a time — defaults come from cot_config.yaml,
--model overrides it):
    uv run python main.py cot-yaml --model google/gemma-4-26B-A4B-it
    uv run python main.py cot-yaml --model Qwen/Qwen3.5-9B --per_step
    uv run python main.py cot-yaml --model Qwen/Qwen3.5-9B \\
        --swap_datasets some_dataset_classification another_dataset_classification

Collate after all models have run:
    uv run python main.py cot-yaml --collate
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
import time
from collections import defaultdict

from PIL import Image as PILImage

from src.common.chat_prompt import render_chat_template
from src.common.dataset_config import (
    DISEASE_TASK_TYPES, active_datasets, crop_display, load_hf_samples,
    load_manifest, parse_class_label,
)
from src.common.io import load_prompts
from src.common.metrics import contains_match, resolve_class, rouge_scores_per_sample, sample_class_images
from src.common.progress import ProgressBar
from src.common.vision_eval import (
    CKPT_FIELDS, append_ckpt_row, archive_ckpt_if_rerun, collate_models, compute_model_metrics,
    judge_gt_hint, load_ckpt, run_judge_phase, safe_model_name, write_collated, write_model_metrics,
)
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

# Per-step checkpoint schema (--per_step / --swap_datasets): one row per
# (image, chain position[, rotation]) instead of CKPT_FIELDS's one row per
# image, so the same image contributes multiple rows here — mode/step_idx/
# rotation/question_id (inserted right after image_name, the one point where
# this schema and CKPT_FIELDS actually diverge) disambiguate them. Reuses
# load_ckpt/append_ckpt_row/rewrite_ckpt/run_judge_phase's `fields=` param
# (see src/common/vision_eval.py) for checkpointing and judging — same
# resumable-schema-healing and same judge model/prompts as CKPT_FIELDS rows,
# just a different row shape.
STEP_CKPT_FIELDS = (
    CKPT_FIELDS[: CKPT_FIELDS.index("image_name") + 1]
    + ["mode", "step_idx", "rotation", "question_id"]
    + CKPT_FIELDS[CKPT_FIELDS.index("image_name") + 1 :]
)

try:
    from vllm import SamplingParams
    _COT_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    _COT_PARAMS = _CLS_PARAMS = None

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)

_COT_PROMPTS_PATH = Path(__file__).parent / "config" / "cot_prompts.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def cot_steps(task_type: str, cot_prompts: dict) -> list[str]:
    cot_key = cot_prompts["task_type_to_cot"].get(task_type, "disease_pest")
    return cot_prompts["thinking_prompts"][cot_key]["steps"]


# ── Prompt building (classification question — same routing as vision_baseline.py) ──

def cls_question(task_type: str, crop_val, prompts: dict) -> str:
    crop = crop_display(crop_val)
    if task_type in DISEASE_TASK_TYPES:
        inner = prompts["visual_classification_disease"].format(crop=crop)
    elif task_type == "quality_identification":
        inner = prompts["visual_classification_quality"].format(crop=crop)
    else:
        inner = prompts["visual_classification_weed"]
    return prompts["assessment_classification"].format(question=inner)


# ── vLLM helpers ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    m = _REPEAT_RE.search(text)
    return text[:m.start()].strip() if m else text



def _sys_msg(prompts: dict) -> list[dict]:
    s = prompts.get("assessment_system", "")
    return [{"role": "system", "content": s}] if s else []


def build_prompt(tokenizer, image: PILImage.Image, messages: list[dict]) -> dict:
    """messages already contains the full conversation history — the first
    user turn's content must be [{"type": "image"}, {"type": "text", ...}]."""
    return {"prompt": render_chat_template(tokenizer, messages), "multi_modal_data": {"image": image}}


def run_batch(llm, tokenizer, images: list, histories: list[list[dict]], params) -> list[str]:
    prompts = [build_prompt(tokenizer, img, hist) for img, hist in zip(images, histories)]
    outputs = llm.generate(prompts, sampling_params=params)
    return [_clean(o.outputs[0].text) for o in outputs]


# ── Group building (shared by the normal run and --swap_datasets) ───────────

def _build_groups(args: argparse.Namespace, manifest: dict, dataset_filter: set[str] | None = None):
    """[(ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name), ...]
    for every (dataset, class) — every sampled image name included, unfiltered
    by any checkpoint's resume state (callers decide what's pending). Restrict
    to `dataset_filter` (a set of agml_name) when given — used by
    --swap_datasets to scope the expensive rotation sweep to a handful of
    datasets instead of the whole config."""
    groups = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip or not ds.task_type:
            continue
        if dataset_filter is not None and ds.agml_name not in dataset_filter:
            continue
        try:
            samples = load_hf_samples(ds.agml_name, manifest, Path(args.images_root))
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest")
            continue
        by_label: dict[str, list[dict]] = {}
        for s in samples:
            by_label.setdefault(s["label"], []).append(s)
        for raw_label, items in by_label.items():
            path_by_name = {Path(it["path"]).name: it["path"] for it in items}
            names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            crop_val, clean_label = parse_class_label(raw_label, ds.crop)
            task_type = ds.effective_task_type(raw_label)
            aliases = ds.effective_search_terms(raw_label)[1:]
            gt_hint = judge_gt_hint(clean_label, aliases)
            groups.append((ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name))
    return groups


# ── Step-chain generation (shared by the normal run, --per_step, and the
#    --swap_datasets rotation sweep) ──────────────────────────────────────────

def run_group_steps(
    llm, tokenizer, images: list, steps: list[str], cls_q: str, prompts: dict,
    step_order: list[int] | None = None, capture_all_steps: bool = False,
) -> list[list[dict]]:
    """Run one (dataset, class) group's full CoT step chain, in `step_order`
    (a permutation of question indices — identity order if None), optionally
    classifying after EVERY intermediate step rather than only the last.

    Returns per_image_rows: one list per image, each entry
    {step_idx (1-based chain position), question_id (1-based ORIGINAL
    question number asked at that position), prediction, reasoning} for
    every step a classification was captured at. The final entry is always
    the last step regardless of capture_all_steps, so
    per_image_rows[i][-1] is always that image's final answer — callers that
    don't care about intermediate steps can ignore everything but [-1], which
    is why this one function serves both the plain run and the step-analysis
    modes without duplicating the generation loop."""
    order = step_order if step_order is not None else list(range(len(steps)))
    ordered_steps = [steps[i] for i in order]
    n = len(ordered_steps)

    histories = [
        _sys_msg(prompts) + [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": ordered_steps[0]},
        ]}]
        for _ in images
    ]
    step_responses: list[list[str]] = [[] for _ in images]
    per_image_rows: list[list[dict]] = [[] for _ in images]

    for step_idx in range(n):
        responses = run_batch(llm, tokenizer, images, histories, _COT_PARAMS)
        for i, resp in enumerate(responses):
            histories[i].append({"role": "assistant", "content": resp})
            step_responses[i].append(resp)

        is_last = step_idx == n - 1
        if capture_all_steps or is_last:
            cls_histories = [h + [{"role": "user", "content": cls_q}] for h in histories]
            preds = run_batch(llm, tokenizer, images, cls_histories, _CLS_PARAMS)
            for i, pred in enumerate(preds):
                reasoning = "\n\n".join(
                    f"Q: {ordered_steps[j]}\nA: {step_responses[i][j]}" for j in range(step_idx + 1)
                )
                per_image_rows[i].append({
                    "step_idx": step_idx + 1, "question_id": order[step_idx] + 1,
                    "prediction": pred, "reasoning": reasoning,
                })

        if step_idx < n - 1:
            for i in range(len(images)):
                histories[i].append({"role": "user", "content": ordered_steps[step_idx + 1]})

    return per_image_rows


def _step_row(
    entry: dict, ds, clean_label: str, crop_val, task_type: str, name: str, gt_hint: str,
    mode: str, rotation: str = "",
) -> dict:
    resolved = resolve_class(entry["prediction"], ds.clean_classes)
    correct = resolved is not None and resolved.lower() == clean_label.lower()
    c_match = contains_match(resolved or entry["prediction"], clean_label)
    scores = rouge_scores_per_sample(entry["prediction"], clean_label, include_rouge=False)
    return {
        "agml_dataset": ds.agml_name, "class_label": clean_label,
        "task_type": task_type, "crop": crop_display(crop_val),
        "image_name": name, "mode": mode, "step_idx": entry["step_idx"],
        "rotation": rotation, "question_id": entry["question_id"],
        "prediction": entry["prediction"], "resolved_class": resolved or "",
        "ground_truth": clean_label, "correct": int(correct), "contains_match": int(c_match),
        "resolvable": int(resolved is not None), "reasoning": entry["reasoning"],
        "judge_gt_hint": gt_hint, **scores,
    }


# ── Main run ─────────────────────────────────────────────────────────────────

def _step_seen_complete(step_rows: list[dict], cot_prompts: dict, mode: str = "cumulative") -> set:
    """{(dataset, class, image)} already fully stepped through (a row for
    every step_idx 1..N) for `mode` — used to skip re-generating an image's
    step chain on resume. A checkpoint with only SOME of an image's steps
    (e.g. a kill mid-chain) is correctly treated as not-yet-complete."""
    idxs_by_key: dict[tuple, set] = defaultdict(set)
    task_type_by_key: dict[tuple, str] = {}
    for r in step_rows:
        if r.get("mode") != mode:
            continue
        key = (r["agml_dataset"], r["class_label"], r["image_name"])
        idxs_by_key[key].add(int(r["step_idx"]))
        task_type_by_key[key] = r["task_type"]
    return {
        key for key, idxs in idxs_by_key.items()
        if len(idxs) >= len(cot_steps(task_type_by_key[key], cot_prompts))
    }


def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    cot_prompts = _load_yaml(_COT_PROMPTS_PATH)
    manifest = load_manifest(Path(args.manifest))

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path)
    if rows:
        print(f"  [resume] {len(rows)} samples already scored in {ckpt_path.name}")

    step_ckpt_path = model_dir / f"ckpt_steps_{safe_model_name(model_id)}.csv"
    step_rows: list[dict] = []
    step_seen: set = set()
    if args.per_step:
        archive_ckpt_if_rerun(step_ckpt_path, args.rerun)
        step_rows, _ = load_ckpt(step_ckpt_path, fields=STEP_CKPT_FIELDS)
        step_seen = _step_seen_complete(step_rows, cot_prompts)
        if step_rows:
            print(f"  [resume] {len(step_rows)} step row(s) already scored in {step_ckpt_path.name}")

    groups = []
    for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in _build_groups(args, manifest):
        key_of = lambda n: (ds.agml_name, clean_label, n)  # noqa: E731
        # A name is pending if EITHER checkpoint still needs it — with
        # --per_step off this reduces to exactly the old "not in seen"
        # check. Enabling --per_step on a resume of an already-finished
        # plain run re-generates that run's images (their step responses
        # were never saved the first time, so there's no way to backfill
        # step rows without redoing the chain) — an accepted cost of
        # turning the flag on after the fact rather than from the start.
        pending = [n for n in names if key_of(n) not in seen or (args.per_step and key_of(n) not in step_seen)]
        if not pending:
            continue
        groups.append((ds, clean_label, crop_val, task_type, gt_hint, pending, path_by_name))

    if not groups:
        print("  Nothing to do — all samples already checkpointed.")
    else:
        n_pending = sum(len(g[5]) for g in groups)
        print(f"  {len(groups)} (dataset, class) group(s), {n_pending} pending sample(s)")

        llm, tokenizer = load_llm(
            model_id, quantization=args.quantize,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            vision_token_budget=args.vision_token_budget,
            caller_tag="cot_yaml",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Classes")
        for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in groups:
            pbar.desc = f"Classes ({ds.agml_name} / {clean_label})"
            images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]
            steps = cot_steps(task_type, cot_prompts)
            cls_q = cls_question(task_type, crop_val, prompts)

            per_image_rows = run_group_steps(
                llm, tokenizer, images, steps, cls_q, prompts,
                capture_all_steps=args.per_step,
            )

            for name, entries in zip(names, per_image_rows):
                # _step_row's extra mode/step_idx/rotation/question_id keys
                # are harmless here — append_ckpt_row(fields=CKPT_FIELDS,
                # the default) only pulls the columns it knows about.
                row = _step_row(entries[-1], ds, clean_label, crop_val, task_type, name, gt_hint, mode="final")
                append_ckpt_row(ckpt_path, row)
                rows.append(row)

                if args.per_step:
                    for entry in entries:
                        srow = _step_row(entry, ds, clean_label, crop_val, task_type, name, gt_hint,
                                          mode="cumulative")
                        append_ckpt_row(step_ckpt_path, srow, fields=STEP_CKPT_FIELDS)
                        step_rows.append(srow)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer

    rows = run_judge_phase(rows, ckpt_path, args, prompts)

    metrics = compute_model_metrics(rows)
    write_model_metrics(model_dir, metrics)
    print(f"  Overall: f1_macro={metrics['overall'].get('f1_macro')} "
          f"rouge_ss={metrics['overall'].get('rouge_ss')} (n={metrics['overall'].get('n_total')})")

    if args.per_step:
        step_rows = run_judge_phase(step_rows, step_ckpt_path, args, prompts, fields=STEP_CKPT_FIELDS)
        print(f"  Per-step: {len(step_rows)} row(s) checkpointed to {step_ckpt_path.name} — "
              f"view with `main.py cot-yaml-step-report`")


# ── Swap-order (question-order robustness check for a small dataset set) ────

def run_swap_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    cot_prompts = _load_yaml(_COT_PROMPTS_PATH)
    manifest = load_manifest(Path(args.manifest))
    dataset_filter = set(args.swap_datasets)

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    swap_ckpt_path = model_dir / f"ckpt_swap_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(swap_ckpt_path, args.rerun)
    swap_rows, _ = load_ckpt(swap_ckpt_path, fields=STEP_CKPT_FIELDS)
    if swap_rows:
        print(f"  [resume] {len(swap_rows)} swap row(s) already scored in {swap_ckpt_path.name}")

    groups = _build_groups(args, manifest, dataset_filter=dataset_filter)
    missing = dataset_filter - {ds.agml_name for ds, *_ in groups}
    if missing:
        print(f"  [warn] --swap_datasets not found (skip/not in manifest?): {sorted(missing)}")
    if not groups:
        print("  Nothing to do — none of --swap_datasets matched an active, sampled dataset.")
        return

    llm, tokenizer = load_llm(
        model_id, quantization=args.quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        vision_token_budget=args.vision_token_budget,
        caller_tag="cot_yaml-swap",
    )

    t_start = time.perf_counter()
    for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in groups:
        steps = cot_steps(task_type, cot_prompts)
        cls_q = cls_question(task_type, crop_val, prompts)
        n_steps = len(steps)
        images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]

        for rot in range(n_steps):
            rot_complete = _step_seen_complete(
                [r for r in swap_rows if r.get("rotation") == str(rot)], cot_prompts, mode="swap",
            )
            pending_idx = [i for i, n in enumerate(names)
                           if (ds.agml_name, clean_label, n) not in rot_complete]
            if not pending_idx:
                continue
            print(f"  {ds.agml_name}/{clean_label} — rotation {rot + 1}/{n_steps} "
                  f"({len(pending_idx)}/{len(names)} pending)")
            step_order = list(range(rot, n_steps)) + list(range(rot))
            per_image_rows = run_group_steps(
                llm, tokenizer, [images[i] for i in pending_idx], steps, cls_q, prompts,
                step_order=step_order, capture_all_steps=True,
            )
            for i, entries in zip(pending_idx, per_image_rows):
                for entry in entries:
                    srow = _step_row(entry, ds, clean_label, crop_val, task_type, names[i], gt_hint,
                                      mode="swap", rotation=str(rot))
                    append_ckpt_row(swap_ckpt_path, srow, fields=STEP_CKPT_FIELDS)
                    swap_rows.append(srow)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Inference time: {elapsed:.1f}s")
    destroy_llm(llm)
    del tokenizer

    swap_rows = run_judge_phase(swap_rows, swap_ckpt_path, args, prompts, fields=STEP_CKPT_FIELDS)
    print(f"  Swap: {len(swap_rows)} row(s) checkpointed to {swap_ckpt_path.name} — "
          f"view with `main.py cot-yaml-step-report --swap`")


def main(args: argparse.Namespace) -> None:
    if args.collate:
        collated = collate_models(Path(args.output_dir))
        if not collated:
            print(f"No per-model output folders with checkpoints found under {args.output_dir}")
            return
        write_collated(Path(args.output_dir), collated)
        return

    if not args.model:
        print("ERROR: --model is required (or pass --collate to combine finished runs).")
        sys.exit(1)

    if args.swap_datasets:
        run_swap_model(args.model, args)
    else:
        run_model(args.model, args)


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--cot_config", default="src/config/cot_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cot_config", default=_pre_args.cot_config,
        help="YAML file of defaults, shared with self_cot.py — edit it directly for "
             "routine changes instead of passing flags every time. Any flag also "
             "passed on the command line overrides this file.",
    )
    parser.add_argument("--model", default=None, help="Single model to run. Overrides the config file.")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--output_dir", default="outputs_final/cot_yaml/")
    parser.add_argument("--n_samples", type=int, default=10, help="Images/class to evaluate.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint(s) (final and/or step/swap, whichever "
                              "this invocation would otherwise read from — and judge_log.jsonl) aside, "
                              "timestamped, and regenerate every row from scratch — unlike a plain "
                              "rerun, which would silently keep already-checkpointed rows even if "
                              "they're known-bad (e.g. corrupted by a since-fixed prompt-building "
                              "bug). Old files are archived, never deleted.")
    parser.add_argument("--per_step", action="store_true",
                         help="Also classify after every intermediate step (not just the last) "
                              "and checkpoint it separately to ckpt_steps_<model>.csv — view with "
                              "`main.py cot-yaml-step-report`. Roughly doubles generation cost "
                              "(one extra, cheap classification pass per step).")
    parser.add_argument("--swap_datasets", nargs="+", default=None,
                         help="Restrict to these dataset(s) and re-run --per_step-style generation "
                              "once per rotation of the step order (every question gets a turn "
                              "going first), checkpointed to ckpt_swap_<model>.csv instead of the "
                              "normal run — a robustness check for whether the per-step F1 curve "
                              "depends on question order. Meant for a small, deliberate set (e.g. "
                              "`main.py cot-compare`'s top gain/loss datasets), not the full config.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET,
                         help="Per-image vision token count to align every model family to (see "
                             "src/common/vlm_load.py) — Gemma defaults to 280 unaligned, Qwen scales "
                             "uncapped with image resolution unaligned; this pins both to the same "
                             "budget so an accuracy comparison isn't confounded by unequal visual "
                             "detail. Also bounds Qwen's prompt length, fixing the max_model_len "
                             "overflow some large images hit. Pass a lower value (e.g. 280) for a "
                             "faster/cheaper run, or 0 to opt out (each model's own default).")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it",
                         help="Fixed judge model used for every baseline/model being evaluated.")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200,
                         help="Rows per apply_judge() call — drives the judge progress bar "
                              "and how much work a kill mid-judging can lose.")
    parser.add_argument("--rejudge", action="store_true",
                         help="Clear judged_correct/judge_reasoning on every row and re-judge "
                              "from scratch — use after a judge bug may have corrupted results.")
    parser.add_argument("--skip_judge", action="store_true",
                         help="Skip the judge phase entirely (no judge model loaded) — for "
                              "preliminary/small-GPU runs of classification metrics only. "
                              "Rerun later without this flag to backfill judging.")
    parser.add_argument("--collate", action="store_true",
                         help="Skip inference — combine every <output_dir>/<model>/ folder "
                              "into collated_summary.csv / collated_metrics.json.")

    # Keys this script doesn't use but the shared config file also carries for self_cot.py.
    _FOREIGN_KEYS = {"self_cot_output_dir"}

    if _pre_args.cot_config and Path(_pre_args.cot_config).exists():
        import yaml as _yaml
        with open(_pre_args.cot_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        if "cot_yaml_output_dir" in _cfg:
            _cfg["output_dir"] = _cfg.pop("cot_yaml_output_dir")
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests - _FOREIGN_KEYS
        if _unknown:
            print(f"[warn] {_pre_args.cot_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
