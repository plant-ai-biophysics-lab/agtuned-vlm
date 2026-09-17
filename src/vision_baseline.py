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
from src.common.metrics import (
    contains_match, extract_answer_tag, extract_channel_thought, resolve_class, sample_class_images,
)
from src.common.progress import ProgressBar
from src.common.vision_eval import (
    CKPT_FIELDS, append_ckpt_row, archive_ckpt_if_rerun, collate_models, compute_model_metrics,
    judge_gt_hint, load_ckpt, rewrite_ckpt, run_judge_phase, safe_model_name, write_collated, write_model_metrics,
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

# extract_channel_thought (Gemma-4's <|channel>thought...<channel|> format)
# now lives in src/common/metrics.py — shared with apply_judge, which needs
# the identical extraction for the judge's own thinking output.


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
    enable_thinking: bool = False, max_tokens: int = 256, chat_template: str | None = None,
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
    thinking (<think>/<thought> tags, OR Gemma-4's <|channel>thought
    channel if chat_template is also given) through instead of suppressing
    it. Raise max_tokens substantially (e.g. 1024+) when enabling it, or
    thinking traces will routinely get truncated before the model reaches
    its <answer> tag.

    chat_template (default None): forwarded to build_multimodal_prompt —
    Gemma-4's own bundled template has no thinking support at all, so
    enable_thinking alone is a no-op for it; pass the vendored
    src/config/tool_chat_template_gemma4.jinja's text here too. Also flips
    SamplingParams(skip_special_tokens=not enable_thinking) — Gemma-4's
    <|channel>/<channel|> markers are real special tokens (vLLM strips
    special tokens from decoded output by default), so without
    skip_special_tokens=False they'd vanish from the output text before
    extract_channel_thought ever saw them, even with the right template.
    Harmless when enable_thinking is False, or for models whose thinking
    tags (e.g. Qwen's <think>) aren't special tokens either way."""
    prompts = [
        build_multimodal_prompt(tokenizer, img, question, system_prompt=system_prompt,
                                 enable_thinking=enable_thinking, chat_template=chat_template)
        for img in images
    ]
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.1,
        skip_special_tokens=not enable_thinking,
    )
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
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    model_dir = Path(args.output_dir) / safe_model_name(model_id)
    # --num_shards > 1: each shard gets its own checkpoint filename (never
    # the canonical ckpt_<model>.csv) so N independent single-GPU processes
    # can run concurrently against the SAME model_dir with zero write
    # contention. The un-sharded case (num_shards=1, the default) is
    # byte-for-byte the original filename — zero behavior change for every
    # existing single-GPU run/script. Phase 1-4 (judge_leniency.py etc.)
    # already glob ckpt_*.csv inside a directory, so shard files work there
    # with NO further changes; --merge_shards (below) combines them back
    # into the canonical filename for --collate and anything else that
    # expects exactly one file.
    ckpt_stem = f"ckpt_{safe_model_name(model_id)}"
    if args.num_shards > 1:
        ckpt_stem += f"_shard{args.shard_index}of{args.num_shards}"
    ckpt_path = model_dir / f"{ckpt_stem}.csv"
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

    if args.num_shards > 1:
        n_before = len(groups)
        groups = groups[args.shard_index :: args.num_shards]
        print(f"  [shard {args.shard_index}/{args.num_shards}] {len(groups)}/{n_before} (dataset, class) "
              f"group(s) assigned to this shard")

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
                               enable_thinking=args.enable_thinking, max_tokens=args.gen_max_tokens,
                               chat_template=chat_template)

            for name, (raw, cleaned) in zip(names, preds):
                # Extract Gemma-4's <|channel>thought block (if any) BEFORE
                # answer-tag extraction, so a leftover channel marker never
                # confuses extract_answer_tag/resolve_class — a no-op
                # (thinking="", pred=cleaned) whenever no channel block is
                # present, which is every row unless --enable_thinking +
                # --chat_template_path were both used.
                thinking, pred = extract_channel_thought(cleaned)
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
                    "reasoning": thinking,
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

    if args.num_shards > 1:
        # Skip metrics.json here — computed from only THIS shard's rows, and
        # model_dir/metrics.json is a single fixed filename every shard would
        # otherwise overwrite with its own partial (and therefore misleading)
        # numbers. Run --merge_shards once all shards finish to get the real,
        # complete metrics.json from the merged full row set.
        print(f"  [shard {args.shard_index}/{args.num_shards}] done — {len(rows)} row(s) in {ckpt_path.name}. "
              f"Run --merge_shards once every shard has finished for a combined ckpt + metrics.json.")
    else:
        metrics = compute_model_metrics(rows)
        write_model_metrics(model_dir, metrics)
        print(f"  Overall: f1_macro={metrics['overall'].get('f1_macro')} "
              f"judge_accuracy={metrics['overall'].get('judge_accuracy')} (n={metrics['overall'].get('n_total')})")


def merge_shards(output_dir: Path, model_id: str, fields: list[str] = VISION_BASELINE_CKPT_FIELDS) -> None:
    """Combine every ckpt_<model>_shard<i>of<n>.csv in output_dir/<model>/
    into the canonical ckpt_<model>.csv (rewritten fresh, not appended —
    safe to re-run), so --collate/anything else expecting exactly one file
    per model works after a sharded run. Rows are joined on
    (agml_dataset, class_label, image_name); a duplicate key across shards
    (shouldn't happen — shards partition disjoint groups — but checked
    rather than assumed) keeps the first-seen row and prints a warning.

    IMPORTANT: leaves the shard files in place (never deletes them). Phase
    1-4 scripts (judge_leniency.py etc.) already glob ckpt_*.csv inside a
    directory and don't need this merge step at all — they work directly
    off shard files. If you run THIS merge in the same directory Phase 1-4
    already reads from, that glob will now match BOTH the shard files AND
    the new canonical ckpt_<model>.csv, double-counting every row. Either
    run this in a model_dir Phase 1-4 hasn't looked at yet, or delete/move
    the shard files after merging if Phase 1-4 needs to re-read that dir."""
    model_dir = output_dir / safe_model_name(model_id)
    shard_paths = sorted(model_dir.glob(f"ckpt_{safe_model_name(model_id)}_shard*of*.csv"))
    if not shard_paths:
        print(f"  No shard checkpoints found under {model_dir} (expected "
              f"ckpt_{safe_model_name(model_id)}_shard<i>of<n>.csv).")
        return

    merged: list[dict] = []
    seen_keys: set[tuple] = set()
    n_dupes = 0
    for shard_path in shard_paths:
        shard_rows, _ = load_ckpt(shard_path, fields=fields)
        for r in shard_rows:
            key = (r.get("agml_dataset", ""), r.get("class_label", ""), r.get("image_name", ""))
            if key in seen_keys:
                n_dupes += 1
                continue
            seen_keys.add(key)
            merged.append(r)
        print(f"  {shard_path.name}: {len(shard_rows)} row(s)")

    if n_dupes:
        print(f"  WARNING: {n_dupes} duplicate (agml_dataset, class_label, image_name) key(s) across "
              f"shards — kept the first-seen row for each. Shards should partition disjoint groups; "
              f"this suggests shards were run with different --config/--manifest/--n_samples, or "
              f"--num_shards/--shard_index don't match across the shard jobs.")

    ckpt_path = model_dir / f"ckpt_{safe_model_name(model_id)}.csv"
    rewrite_ckpt(ckpt_path, merged, fields=fields)
    metrics = compute_model_metrics(merged)
    write_model_metrics(model_dir, metrics)
    print(f"  Merged {len(merged)} row(s) from {len(shard_paths)} shard(s) into {ckpt_path.name}")
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

    if args.merge_shards:
        if not args.model:
            print("ERROR: --model is required with --merge_shards.")
            sys.exit(1)
        merge_shards(Path(args.output_dir), args.model, fields=VISION_BASELINE_CKPT_FIELDS)
        return

    if not args.model:
        print("ERROR: --model is required (or pass --collate to combine finished runs).")
        sys.exit(1)

    if args.num_shards > 1 and not (0 <= args.shard_index < args.num_shards):
        print(f"ERROR: --shard_index must be in [0, {args.num_shards}) — got {args.shard_index}.")
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
    parser.add_argument("--chat_template_path", default=None,
                         help="Override the tokenizer's own bundled chat template with a Jinja file at "
                              "this path — needed for --enable_thinking to actually do anything on some "
                              "models. Gemma-4's own bundled template has NO enable_thinking support at "
                              "all (confirmed empirically: 0/40978 real rows ever produced a thinking "
                              "block with the stock template) — pass "
                              "src/config/tool_chat_template_gemma4.jinja (vendored from vLLM) alongside "
                              "--enable_thinking to get real <|channel>thought...<channel|> output from "
                              "it, captured into the checkpoint's `reasoning` column. Not needed for "
                              "Qwen models, whose own bundled template already supports enable_thinking.")
    parser.add_argument("--judge_model", default="google/gemma-4-26B-A4B-it",
                         help="Fixed judge model used for every baseline/model being evaluated.")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200,
                         help="Rows per apply_judge() call — drives the judge progress bar "
                              "and how much work a kill mid-judging can lose.")
    parser.add_argument("--judge_enable_thinking", action="store_true",
                         help="Attempt tokenizer.apply_chat_template(..., enable_thinking=True) for "
                              "the judge's own prompt during this run's judge phase (ignored if "
                              "--skip_judge is set). Off by default, exact original prompt rendering. "
                              "A no-op if the judge tokenizer's chat template doesn't accept this "
                              "kwarg. Same flag/behavior as judge-leniency's --judge_enable_thinking, "
                              "now also available for this script's own built-in judge pass.")
    parser.add_argument("--judge_max_tokens", type=int, default=None,
                         help="Override the judge's default max_tokens=1024 for this run's judge "
                              "phase — a thinking trace (--judge_enable_thinking) can consume the "
                              "normal budget before reaching the final JSON verdict, truncating the "
                              "response.")
    parser.add_argument("--judge_chat_template_path", default=None,
                         help="Override the judge tokenizer's own bundled chat template with a Jinja "
                              "file at this path — needed for --judge_enable_thinking to do anything "
                              "on a Gemma-4 judge model, whose stock template has no enable_thinking "
                              "support at all. Pass src/config/tool_chat_template_gemma4.jinja "
                              "alongside --judge_enable_thinking for a Gemma-4 judge. Not needed for "
                              "a Qwen judge model.")
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
    parser.add_argument("--num_shards", type=int, default=1,
                         help="Split this model's (dataset, class) groups across N independent "
                              "processes (e.g. N separate single-GPU SLURM jobs/array tasks) for "
                              "N-way data-parallel throughput — each shard writes its own "
                              "ckpt_<model>_shard<i>of<N>.csv (never the canonical ckpt_<model>.csv, "
                              "so concurrent shards never write-contend), and Phase 1-4 scripts "
                              "(judge_leniency.py etc.) already glob ckpt_*.csv inside a directory, so "
                              "shard files work there with no further changes. Default 1 = today's "
                              "exact single-file behavior, zero change. Run --merge_shards once every "
                              "shard finishes to get a combined ckpt_<model>.csv + metrics.json for "
                              "--collate or anything else expecting exactly one file.")
    parser.add_argument("--shard_index", type=int, default=0,
                         help="Which shard this process is, in [0, --num_shards). Pass "
                              "$SLURM_ARRAY_TASK_ID for a SLURM job array.")
    parser.add_argument("--merge_shards", action="store_true",
                         help="Skip inference — combine ckpt_<model>_shard*of*.csv (written by "
                              "--num_shards > 1 runs) into the canonical ckpt_<model>.csv + "
                              "metrics.json for --model's output dir.")

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
