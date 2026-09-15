"""
Stage 2a — Vision Baseline (lower bound): image + classification question
only, no reference description. Companion to desc_augmented_vision_v2.py
(upper bound — same setup plus a reference description prepended) — together
they bracket how much of vision performance depends on a good text
description vs. the model's own raw visual reasoning.

Classification-only: no separate description-generation pass. Per sample,
scores resolve_class exact/contains match (-> F1) AND
rouge_scores_per_sample(prediction, ground_truth class label) for a softer
partial-credit signal alongside the strict F1.

Per-model output subfolder + a per-sample checkpoint CSV (flushed to disk
immediately after each sample is scored, not batched) so a run on a
time-limited remote job can resume exactly where it left off:
  <output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv
  <output_dir>/<safe_model_name>/metrics.json

Run with --collate (no --model needed) once every model's run has finished,
to combine every <output_dir>/<model>/ folder into:
  <output_dir>/collated_summary.csv
  <output_dir>/collated_metrics.json

Usage (one model at a time — defaults come from vision_baseline_config.yaml,
--model overrides it):
    uv run python main.py vision-baseline --model google/gemma-4-26B-A4B-it
    uv run python main.py vision-baseline --model Qwen/Qwen3.5-9B

Collate after all models have run:
    uv run python main.py vision-baseline --collate
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
import time

from PIL import Image as PILImage

from src.common.chat_prompt import build_multimodal_prompt
from src.common.dataset_config import (
    DISEASE_TASK_TYPES, active_datasets, crop_display, load_hf_samples,
    load_manifest, parse_class_label,
)
from src.common.io import load_prompts
from src.common.metrics import contains_match, extract_answer_tag, resolve_class, sample_class_images
from src.common.progress import ProgressBar
from src.common.vision_eval import (
    CKPT_FIELDS, append_ckpt_row, archive_ckpt_if_rerun, collate_models, compute_model_metrics,
    judge_gt_hint, load_ckpt, run_judge_phase, safe_model_name, write_collated, write_model_metrics,
)
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm

# No ROUGE/sem_sim here — this stage is classification-only (correct/
# contains_match/resolvable already the real signal), and rouge1/rouge2/
# rougeL/rouge_ss were already always blank via include_rouge=False;
# sem_sim was the only one actually computed (a real per-row cost, loading
# sentence-transformers) despite not being used anywhere downstream
# (vision_baseline_collate.py's stage2_summary.csv is F1-only). Dropping
# all five from this script's own checkpoint schema — desc_augmented_vision_v2.py
# / self_cot.py / cot_yaml.py are unaffected, they still use CKPT_FIELDS
# as-is. compute_model_metrics's aggregate_rouge() already skips missing
# keys gracefully, so metrics.json needs no changes either.
_DROPPED_FIELDS = {"rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim"}
_kept_fields = [f for f in CKPT_FIELDS if f not in _DROPPED_FIELDS]
# raw_response: the model's completely untouched generation output, before
# _clean() strips <think>/<thought> tags — added so a thinking-mode run
# (--enable_thinking) has somewhere to keep the reasoning text even when it
# isn't wrapped in a tag _clean's regex would catch (see run_batch's
# docstring — Gemma-4 has been observed emitting unwrapped reasoning prose
# directly before its <answer> tag). Inserted right after "prediction" so
# the two sit next to each other in the CSV. Blank for any pre-existing
# checkpoint row (backfilled as "" by load_ckpt's schema-drift healing, same
# as extracted_answer was before it).
_prediction_idx = _kept_fields.index("prediction")
VISION_BASELINE_CKPT_FIELDS = (
    _kept_fields[: _prediction_idx + 1] + ["raw_response"] + _kept_fields[_prediction_idx + 1 :]
)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)


# ── Prompt building ─────────────────────────────────────────────────────────

def cls_question(task_type: str, crop_val, prompts: dict) -> str:
    crop = crop_display(crop_val)
    if task_type in DISEASE_TASK_TYPES:
        inner = prompts["visual_classification_disease"].format(crop=crop)
    elif task_type == "quality_identification":
        inner = prompts["visual_classification_quality"].format(crop=crop)
    else:
        inner = prompts["visual_classification_weed"]
    return prompts["assessment_classification_tagged"].format(question=inner)


# ── vLLM helpers ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def run_batch(
    llm, tokenizer, images: list, question: str, system_prompt: str = "",
    enable_thinking: bool = False, max_tokens: int = 256,
) -> list[tuple[str, str]]:
    """Returns (raw, cleaned) pairs, one per image — raw is the model's
    completely untouched output text, cleaned is raw with any
    <think>/<thought>...</think|thought> block stripped (_clean), same as
    this function returned alone before raw was added.

    raw is kept around (see run_model's "raw_response" checkpoint column)
    because <think>/<thought> tags are not the only shape a model's
    reasoning can take — e.g. Gemma-4 has been observed emitting an
    unwrapped reasoning paragraph directly before its <answer> tag, with no
    tags _THINK_RE would ever match, so "cleaned" alone can silently still
    contain (or, for a differently-tagged future model, silently drop) the
    actual thinking text. Keeping raw preserves exactly what the model said
    regardless of what tagging convention (if any) it used, so this doesn't
    need to be re-litigated per model.

    enable_thinking (default False, unchanged behavior): forwarded to
    build_multimodal_prompt — set True to let the model's own extended
    thinking (<think>/<thought> tags) through instead of suppressing it.
    Raise max_tokens substantially (e.g. 1024+) when enabling it, or
    thinking traces will routinely get truncated before the model reaches
    its <answer> tag."""
    prompts = [
        build_multimodal_prompt(tokenizer, img, question, system_prompt=system_prompt,
                                 enable_thinking=enable_thinking)
        for img in images
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.1)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [(o.outputs[0].text, _clean(o.outputs[0].text)) for o in outputs]


# run_judge_phase() lives in src/common/vision_eval.py — shared verbatim with
# desc_augmented_vision_v2.py (and any future classification experiment) so
# judge behavior (model, chunking, checkpointing, resume, --rejudge,
# judge_gt_hint backfill) stays identical everywhere. Do not redefine it here.


# ── Main run ─────────────────────────────────────────────────────────────────

def run_model(model_id: str, args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")
    manifest = load_manifest(Path(args.manifest))

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    archive_ckpt_if_rerun(ckpt_path, args.rerun)
    rows, seen = load_ckpt(ckpt_path, fields=VISION_BASELINE_CKPT_FIELDS)
    if rows:
        print(f"  [resume] {len(rows)} samples already scored in {ckpt_path.name}")

    # Collect (dataset, raw_label, clean_label, crop, task_type, [(name, path)]) groups
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
            aliases = ds.effective_search_terms(raw_label)[1:]  # [1:] drops raw_label itself
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
            caller_tag="vision_baseline",
        )

        t_start = time.perf_counter()
        pbar = ProgressBar(total=len(groups), desc="Classes")
        for ds, clean_label, crop_val, task_type, gt_hint, names, path_by_name in groups:
            pbar.desc = f"Classes ({ds.agml_name} / {clean_label})"
            images = [PILImage.open(path_by_name[n]).convert("RGB") for n in names]
            question = cls_question(task_type, crop_val, prompts)

            preds = run_batch(llm, tokenizer, images, question, system_prompt=system_prompt,
                               enable_thinking=args.enable_thinking, max_tokens=args.gen_max_tokens)

            for name, (raw, pred) in zip(names, preds):
                answer, _ = extract_answer_tag(pred)
                resolved = resolve_class(answer, ds.clean_classes)
                correct = resolved is not None and resolved.lower() == clean_label.lower()
                c_match = contains_match(resolved or pred, clean_label)
                row = {
                    "agml_dataset": ds.agml_name, "class_label": clean_label,
                    "task_type": task_type, "crop": crop_display(crop_val),
                    "image_name": name, "prediction": pred, "raw_response": raw,
                    "resolved_class": resolved or "", "extracted_answer": answer,
                    "ground_truth": clean_label,
                    "correct": int(correct), "contains_match": int(c_match),
                    "resolvable": int(resolved is not None),
                    "judge_gt_hint": gt_hint,
                }
                append_ckpt_row(ckpt_path, row, fields=VISION_BASELINE_CKPT_FIELDS)
                rows.append(row)
            pbar.update()
        pbar.finish()

        elapsed = time.perf_counter() - t_start
        print(f"\n  Inference time: {elapsed:.1f}s")
        destroy_llm(llm)
        del tokenizer

    rows = run_judge_phase(rows, ckpt_path, args, prompts, fields=VISION_BASELINE_CKPT_FIELDS)

    metrics = compute_model_metrics(rows)
    write_model_metrics(model_dir, metrics)
    print(f"  Overall: f1_macro={metrics['overall'].get('f1_macro')} "
          f"judge_accuracy={metrics['overall'].get('judge_accuracy')} (n={metrics['overall'].get('n_total')})")


def main(args: argparse.Namespace) -> None:
    if args.collate:
        collated = collate_models(Path(args.output_dir), fields=VISION_BASELINE_CKPT_FIELDS)
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
    _pre.add_argument("--vision_baseline_config", default="src/config/vision_baseline_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vision_baseline_config", default=_pre_args.vision_baseline_config,
        help="YAML file of defaults, shared with desc_augmented_vision_v2.py — edit it "
             "directly for routine changes instead of passing flags every time. Any flag "
             "also passed on the command line overrides this file.",
    )
    parser.add_argument("--model", default=None, help="Single model to run. Overrides the config file.")
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--classification_jsonl", default="dataset/agml_classification.jsonl",
                         help="Unused here (only desc_augmented_vision_v2.py needs it) — accepted "
                              "so both scripts can share one config file without a key warning.")
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--output_dir", default="outputs_final/vision_baseline/")
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
                             "faster/cheaper run, or omit mm_processor_kwargs entirely by setting this "
                             "to 0 (falls back to each model's own default).")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true",
                         help="Let the model being evaluated use its own extended thinking "
                              "(<think>/<thought> tags) for the classification generation step, instead "
                              "of suppressing it (this script's default, matching every other generation "
                              "script in the repo — see src/common/chat_prompt.py's module docstring). "
                              "Does NOT affect the judge phase's own calls (see judge-leniency's "
                              "--judge_enable_thinking for that, a separate flag on a separate call). "
                              "Raise --gen_max_tokens when using this, or thinking traces will routinely "
                              "get truncated before the model reaches its <answer> tag.")
    parser.add_argument("--gen_max_tokens", type=int, default=256,
                         help="max_tokens for the classification generation step. 256 fits the tagged "
                              "prompt's direct answer with no thinking; raise substantially (e.g. "
                              "1024+) when --enable_thinking is set.")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it",
                         help="Fixed judge model used for every baseline/model being evaluated.")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200,
                         help="Rows per apply_judge() call — drives the judge progress bar "
                              "and how much work a kill mid-judging can lose.")
    parser.add_argument("--rejudge", action="store_true",
                         help="Clear judged_correct/judge_reasoning on every row and re-judge "
                              "from scratch — use after a judge bug (e.g. a blank judge_gt_hint "
                              "on old checkpoints) may have corrupted existing judge results.")
    parser.add_argument("--skip_judge", action="store_true",
                         help="Skip the judge phase entirely (no judge model loaded) — for "
                              "preliminary/small-GPU runs of classification metrics only. "
                              "Rerun later without this flag to backfill judging.")
    parser.add_argument("--collate", action="store_true",
                         help="Skip inference — combine every <output_dir>/<model>/ folder "
                              "into collated_summary.csv / collated_metrics.json.")

    # Keys this script doesn't use but the shared config file also carries
    # for desc_augmented_vision_v2.py / vision_baseline_collate.py.
    _FOREIGN_KEYS = {"aug_output_dir", "collated_output_dir", "n_boot", "seed"}

    if _pre_args.vision_baseline_config and Path(_pre_args.vision_baseline_config).exists():
        import yaml as _yaml
        with open(_pre_args.vision_baseline_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        if "baseline_output_dir" in _cfg:
            _cfg["output_dir"] = _cfg.pop("baseline_output_dir")
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests - _FOREIGN_KEYS
        if _unknown:
            print(f"[warn] {_pre_args.vision_baseline_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

    main(parser.parse_args())
