"""
Stage 2c — Self-CoT (self-generated description): the model first describes
the image in its own words (no external reference), then classifies using
its own description as conversation-history context. A third point between
vision_baseline.py (no description at all) and desc_augmented_vision_v2.py
(a good pre-written description) — this measures what the model's own
unaided visual reasoning contributes when given room to "think out loud"
first, without being handed a ground-truth-adjacent description.

Two-turn conversation per image:
  Turn 1: [image] + description question  -> model's own free-text description
  Turn 2: [image, own description as history] + classification question -> answer

Classification-only scoring: no separate description-quality metric — the
self-generated description is stored in the `reasoning` column for
inspection, but scored only via what it does to the final classification
(F1 vs vision_baseline.py, ROUGE-SS vs ground truth class label, judge_accuracy).

Per-model output subfolder + a per-sample checkpoint CSV (flushed to disk
after each (dataset, class) group's two-turn batch — same granularity as
vision_baseline.py) so a run on a time-limited remote job can resume exactly
where it left off:
  <output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics.json

Run with --collate (no --model needed) once every model's run has finished,
to combine every <output_dir>/<model>/ folder into:
  <output_dir>/collated_summary.csv
  <output_dir>/collated_metrics.json

Usage (one model at a time — defaults come from cot_config.yaml,
--model overrides it):
    uv run python main.py self-cot --model google/gemma-4-26B-A4B-it
    uv run python main.py self-cot --model Qwen/Qwen3.5-9B

Collate after all models have run:
    uv run python main.py self-cot --collate
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
import time

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
    append_ckpt_row, archive_ckpt_if_rerun, collate_models, compute_model_metrics, judge_gt_hint,
    load_ckpt, run_judge_phase, safe_model_name, write_collated, write_model_metrics,
)
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

try:
    from vllm import SamplingParams
    _DESC_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, repetition_penalty=1.1)
    _CLS_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, repetition_penalty=1.1)
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    _DESC_PARAMS = _CLS_PARAMS = None

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)


# ── Prompt building ─────────────────────────────────────────────────────────

def desc_question(task_type: str, crop_val, prompts: dict) -> str:
    crop = crop_display(crop_val)
    if task_type in DISEASE_TASK_TYPES:
        return prompts["visual_description_disease"].format(crop=crop)
    if task_type == "quality_identification":
        return prompts["visual_description_quality"].format(crop=crop)
    return prompts["visual_description_weed"]


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


# ── Main run ─────────────────────────────────────────────────────────────────

def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    manifest = load_manifest(Path(args.manifest))

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path)
    if rows:
        print(f"  [resume] {len(rows)} samples already scored in {ckpt_path.name}")

    groups = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip or not ds.task_type:
            continue
        try:
            samples = load_hf_samples(
                ds.agml_name, manifest, Path(args.images_root), excluded_classes=ds.excluded_classes,
            )
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
            pending = [n for n in names if (ds.agml_name, clean_label, n) not in seen]
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
            caller_tag="self_cot",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Classes")
        for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in groups:
            pbar.desc = f"Classes ({ds.agml_name} / {clean_label})"
            images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]
            desc_q = desc_question(task_type, crop_val, prompts)
            cls_q = cls_question(task_type, crop_val, prompts)

            # Turn 1 — self-description
            turn1_histories = [
                _sys_msg(prompts) + [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": desc_q},
                ]}]
                for _ in images
            ]
            descriptions = run_batch(llm, tokenizer, images, turn1_histories, _DESC_PARAMS)

            # Turn 2 — classify, with turn 1's own description as history
            turn2_histories = [
                hist + [{"role": "assistant", "content": desc}, {"role": "user", "content": cls_q}]
                for hist, desc in zip(turn1_histories, descriptions)
            ]
            preds = run_batch(llm, tokenizer, images, turn2_histories, _CLS_PARAMS)

            for name, pred, desc in zip(names, preds, descriptions):
                resolved = resolve_class(pred, ds.clean_classes)
                correct = resolved is not None and resolved.lower() == clean_label.lower()
                c_match = contains_match(resolved or pred, clean_label)
                scores = rouge_scores_per_sample(pred, clean_label, include_rouge=False)
                row = {
                    "agml_dataset": ds.agml_name, "class_label": clean_label,
                    "task_type": task_type, "crop": crop_display(crop_val),
                    "image_name": name, "prediction": pred,
                    "resolved_class": resolved or "", "ground_truth": clean_label,
                    "correct": int(correct), "contains_match": int(c_match),
                    "resolvable": int(resolved is not None),
                    "reasoning": desc,
                    "judge_gt_hint": gt_hint,
                    **scores,
                }
                append_ckpt_row(ckpt_path, row)
                rows.append(row)
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

    run_model(args.model, args)


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--cot_config", default="src/config/cot_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cot_config", default=_pre_args.cot_config,
        help="YAML file of defaults, shared with cot_yaml.py — edit it directly for "
             "routine changes instead of passing flags every time. Any flag also "
             "passed on the command line overrides this file.",
    )
    parser.add_argument("--model", default=None, help="Single model to run. Overrides the config file.")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--output_dir", default="outputs_final/self_cot/")
    parser.add_argument("--n_samples", type=int, default=10, help="Images/class to evaluate.")
    parser.add_argument("--rerun", action="store_true",
                         help="Move any existing checkpoint (and judge_log.jsonl) aside, timestamped, "
                              "and regenerate every row from scratch — unlike a plain rerun, which "
                              "would silently keep already-checkpointed rows even if they're known-bad "
                              "(e.g. corrupted by a since-fixed prompt-building bug). Old files are "
                              "archived, never deleted.")
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

    # Keys this script doesn't use but the shared config file also carries for cot_yaml.py.
    _FOREIGN_KEYS = {"cot_yaml_output_dir"}

    if _pre_args.cot_config and Path(_pre_args.cot_config).exists():
        import yaml as _yaml
        with open(_pre_args.cot_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        if "self_cot_output_dir" in _cfg:
            _cfg["output_dir"] = _cfg.pop("self_cot_output_dir")
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests - _FOREIGN_KEYS
        if _unknown:
            print(f"[warn] {_pre_args.cot_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
