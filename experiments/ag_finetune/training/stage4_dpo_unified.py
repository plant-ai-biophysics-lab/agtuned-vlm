"""
Stage 4 (new) — unified DPO fine-tuning across all 4 ag tasks (disease_pest,
quality, species classification + object detection), as the preference-
learning comparison condition against the in-context skill-memory pipeline
AND against the GRPO alternative (experiments/ag_finetune/training/
stage4_grpo_unified.py). See the Obsidian note "Finetune DPO.md" for the
full write-up of why GRPO's reward collapses on open-set classification
(ground truth is a free-text label the policy's own prompt never narrows
to a candidate list, so K sampled rollouts per prompt very often land ALL
wrong — zero within-group reward variance — contributing nothing to the
gradient) and why DPO sidesteps that: it needs only ONE (chosen, rejected)
pair per prompt, built OFFLINE from Stage 3/4's own explain-traces
checkpoints, never from live rollout sampling.

Reads experiments/ag_finetune/data_prep/build_dpo_unified_data.py's own
output JSONL — one row per (image, question, chosen, rejected) pair,
already pooled across all 4 tasks with a "task" tag (unused by this script;
DPO's loss doesn't need a per-task dispatch the way GRPO's reward function
did, since chosen/rejected are already decided at data-prep time).

Confirmed against the REAL installed trl==0.24.0 source (not assumed — see
Finetune DPO.md's own "Method" section):
  - DPOTrainer has a real, working vision-model branch
    (self.is_vision_model = model.config.model_type in
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES; "gemma4" IS registered
    there, confirmed directly against the installed transformers mapping)
    that calls processor(images=..., text=prompt) for pixel_values and
    tokenizes chosen/rejected separately (process_row, dpo_trainer.py).
  - concatenated_inputs duplicates the SAME pixel_values tensor across both
    the chosen and rejected forward passes — one real image per row, never
    an annotated/boxed version, never two different images.
  - For a PEFT/LoRA model (confirmed: self.is_peft_model or
    args.precompute_ref_log_probs -> self.ref_model = None), DPOTrainer
    does NOT need a second loaded reference model at all — it computes the
    reference log-probs by disabling the LoRA adapter on the SAME model
    instead. This is why this script needs only ONE GPU, unlike GRPO's
    judge-server + rollout-server layout: no judge, no second model, no
    live sampling — every (chosen, rejected) pair was already decided
    offline by build_dpo_unified_data.py.

GPU layout: single GPU, no standalone servers needed at all (contrast with
stage4_grpo_unified.py's 3-process layout). The usual QLoRA 4-bit default
(--load_in_4bit) keeps a 31B model's weights around ~16GB, same reasoning
as the GRPO script's own --load_in_4bit docstring.

Usage:
  uv run python experiments/ag_finetune/data_prep/build_dpo_unified_data.py \
      --output outputs/dpo_unified_pairs.jsonl
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/ag_finetune/training/stage4_dpo_unified.py \
      --model unsloth/gemma-4-31B-it \
      --data outputs/dpo_unified_pairs.jsonl \
      --output_dir outputs/checkpoints/stage4_dpo_unified
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def build_lightning_logger_callback(experiment):
    """Returns a transformers.TrainerCallback instance that forwards every
    scalar in HF Trainer's `logs` dict (loss, rewards/margins, learning
    rate, etc. — whatever DPOTrainer passes at each --logging_steps
    interval) to a Lightning AI `litlogger` experiment, so training is
    visible live in Lightning's Experiments viewer. litlogger is NOT a
    built-in HF Trainer integration (unlike wandb/tensorboard), so this is
    a small custom bridge rather than a report_to="..." string — one
    experiment[key] time-series per metric name, appended on every log
    event. Imports transformers.TrainerCallback lazily (only when this
    function is actually called, i.e. --lightning_experiment_name was
    passed) so a run on non-Lightning infra pays zero cost for this."""
    from transformers import TrainerCallback

    class LightningLoggerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    experiment[key].append(value)

    return LightningLoggerCallback()


def build_prompt(question: str, system_prompt: str) -> list[dict]:
    """Same shape (and same system_prompt — assessment_system, applied
    uniformly to every task including detection) as
    stage4_grpo_unified.py's own build_prompt(), kept consistent across
    both training scripts for a clean comparison."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/gemma-4-31B-it",
                     help="Base VLM checkpoint FastVisionModel can load for training — NOT the FP8 "
                          "vLLM-serving export (inference-only), same caveat as the GRPO script.")
    ap.add_argument("--data", default="outputs/dpo_unified_pairs.jsonl",
                     help="Output of build_dpo_unified_data.py.")
    ap.add_argument("--prompts", default="src/config/prompts.yaml")
    ap.add_argument("--output_dir", default="outputs/checkpoints/stage4_dpo_unified")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--save_steps", type=int, default=200,
                     help="Checkpoint frequency — lower this (e.g. 50-100) on interruptible/spot "
                          "compute to bound how much work a mid-run interruption can lose.")
    ap.add_argument("--save_total_limit", type=int, default=3,
                     help="Keep only the N most recent checkpoints (HF Trainer deletes older ones "
                          "automatically) — without a cap, frequent --save_steps on a long run can "
                          "fill disk with checkpoint directories that are no longer needed once a "
                          "newer one exists.")
    ap.add_argument("--resume_from_checkpoint", default=None,
                     help="Explicit checkpoint directory to resume from (e.g. "
                          "outputs/checkpoints/stage4_dpo_unified/checkpoint-400). If omitted, the "
                          "script auto-detects the latest checkpoint already in --output_dir and "
                          "resumes from it if one exists — the normal case after an interruption on "
                          "spot/preemptible compute, no flag needed to just continue where it left off.")
    ap.add_argument("--lr", type=float, default=5e-6,
                     help="DPO typically wants a lower LR than GRPO/SFT — 5e-6 is a common default "
                          "for preference tuning (vs. GRPO's 1e-5 in the sibling script); raise toward "
                          "1e-5 only if early loss curves show it undertraining.")
    ap.add_argument("--beta", type=float, default=0.1,
                     help="DPOConfig's own KL-deviation-from-reference strength — higher keeps the "
                          "policy closer to the base model. 0.1 is TRL's own default; left at default "
                          "value since there's no real-run evidence yet to justify deviating from it.")
    ap.add_argument("--loss_type", default="sigmoid",
                     help="Standard DPO loss (TRL's own default) — see DPOConfig's docstring for the "
                          "full list of alternatives (ipo/hinge/etc.) if sigmoid underperforms.")
    ap.add_argument("--max_prompt_length", type=int, default=1536,
                     help="Raised from DPOConfig's own default (512) — detection's question text lists "
                          "every class name for the dataset, can run long for datasets with many "
                          "classes (e.g. MH_Weed16's 16-way weed taxonomy).")
    ap.add_argument("--max_completion_length", type=int, default=2048,
                     help="Matches stage4_grpo_unified.py's own reasoning: detection's `chosen` field "
                          "carries a collated per-box explanation PLUS a full box_2d JSON array for "
                          "every ground-truth instance — dense scenes (~50 GT instances/image on some "
                          "held-out datasets) need real headroom here, not DPOConfig's modest default.")
    ap.add_argument("--max_length", type=int, default=3584,
                     help="Overall prompt+completion cap (DPOConfig's own `max_length`) — "
                          "max_prompt_length + max_completion_length plus slack for special/role tokens.")
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                     help="QLoRA 4-bit — same reasoning as stage4_grpo_unified.py's own flag: bf16 "
                          "needs ~62GB for a 31B model's weights alone, out of reach on a single 48GB "
                          "card; 4-bit brings that to ~16GB.")
    ap.add_argument("--dataset_num_proc", type=int, default=4,
                     help="Worker processes for DPOTrainer's own dataset .map() calls (prompt "
                          "extraction, chat-template rendering, tokenization). Left unset before, "
                          "TRL/HF datasets auto-detects this from CPU core count — on a machine with "
                          "many cores this can spawn a high process count that each decode/hold images "
                          "in memory simultaneously, causing a real CPU-RAM OOM kill (confirmed: a run "
                          "died silently at num_proc=20, 99% through tokenization, before any GPU "
                          "training step even started). 4 is a conservative default; raise it if your "
                          "machine has RAM to spare and dataset-prep speed matters.")
    ap.add_argument("--lightning_experiment_name", default=None,
                     help="If set, logs every training metric (loss, rewards/margins, lr, ...) live to "
                          "a Lightning AI `litlogger` experiment of this name, viewable in Lightning's "
                          "Experiments viewer while training runs. Off by default — no litlogger import "
                          "or API call happens unless this is passed.")
    ap.add_argument("--lightning_teamspace", default="default-teamspace",
                     help="Only used when --lightning_experiment_name is set.")
    args = ap.parse_args()

    from datasets import Dataset, Image as HFImage
    from unsloth import FastVisionModel
    from trl import DPOTrainer, DPOConfig

    from src.common.io import load_prompts

    prompts = load_prompts(Path(args.prompts))
    system_prompt = prompts.get("assessment_system", "")

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} DPO pair(s) across tasks: {sorted({r['task'] for r in rows})}")

    cols: dict[str, list] = {
        "images": [row["image_path"] for row in rows],  # cast to HFImage() below for lazy decoding
        "prompt": [build_prompt(row["question"], system_prompt) for row in rows],
        "chosen": [[{"role": "assistant", "content": row["chosen"]}] for row in rows],
        "rejected": [[{"role": "assistant", "content": row["rejected"]}] for row in rows],
    }
    dataset = Dataset.from_dict(cols).cast_column("images", HFImage())

    print(f"  Loading policy {args.model!r} with load_in_4bit={args.load_in_4bit}")
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=False,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        random_state=3407, use_rslora=False, loftq_config=None,
    )
    FastVisionModel.for_training(model)

    # Lightning AI live experiment tracking (opt-in) — a litlogger
    # experiment + a TrainerCallback bridge (build_lightning_logger_callback,
    # above) that forwards every HF Trainer log event to it, since
    # litlogger is not a built-in report_to="..." integration.
    lightning_experiment = None
    callbacks = []
    if args.lightning_experiment_name:
        # Defensive — this is a purely optional observability feature; a
        # missing/broken litlogger (e.g. pip-installed into a different
        # Python environment than `uv run` actually uses, or a Lightning
        # env-snapshot that didn't capture it) must never be able to crash
        # an otherwise-healthy multi-hour training job. Warn and continue
        # without Lightning logging rather than letting the exception
        # propagate out of main().
        try:
            import litlogger
            print(f"  Logging to Lightning AI experiment {args.lightning_experiment_name!r} "
                  f"(teamspace={args.lightning_teamspace!r})")
            lightning_experiment = litlogger.init(
                name=args.lightning_experiment_name, teamspace=args.lightning_teamspace,
            )
            # litlogger's metadata assignment only accepts str or File
            # (confirmed: TypeError: Can only assign str or File, got float,
            # on a real run) -- cast every value explicitly rather than
            # relying on litlogger to coerce native Python types itself.
            lightning_experiment["model"] = str(args.model)
            lightning_experiment["data"] = str(args.data)
            lightning_experiment["lr"] = str(args.lr)
            lightning_experiment["beta"] = str(args.beta)
            lightning_experiment["batch_size"] = str(args.batch_size)
            callbacks.append(build_lightning_logger_callback(lightning_experiment))
        except Exception as e:
            print(f"  [warn] --lightning_experiment_name was set but litlogger setup failed "
                  f"({type(e).__name__}: {e}) — continuing WITHOUT Lightning experiment tracking. "
                  f"If this is an import error, check `uv run python -c 'import litlogger'` works "
                  f"(uv run uses its own managed venv, separate from any plain `pip install` you "
                  f"may have run interactively — try `uv add litlogger` or `uv pip install litlogger` "
                  f"instead).")
            lightning_experiment = None

    trainer = DPOTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        callbacks=callbacks or None,
        args=DPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            beta=args.beta,
            loss_type=[args.loss_type],
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            max_length=args.max_length,
            bf16=True,
            logging_steps=20,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            dataset_num_proc=args.dataset_num_proc,
            # No ref_model passed to DPOTrainer below — a PEFT model gets
            # its reference log-probs by disabling the LoRA adapter on
            # this SAME model (confirmed in the real trl source, see this
            # module's own docstring), not a second loaded copy.
        ),
    )

    # Resume support for interruptible/spot compute — explicit
    # --resume_from_checkpoint wins; otherwise auto-detect the latest
    # checkpoint already in --output_dir (the normal case right after a
    # preemption) via HF's own get_last_checkpoint, same mechanism
    # Trainer's own resume_from_checkpoint=True would use internally, done
    # explicitly here so a genuinely fresh --output_dir (no checkpoints
    # yet) doesn't hit the ValueError Trainer raises when
    # resume_from_checkpoint=True finds nothing to resume from.
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None:
        from transformers.trainer_utils import get_last_checkpoint
        resume_checkpoint = get_last_checkpoint(args.output_dir)
    if resume_checkpoint:
        print(f"  Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("  No checkpoint found — starting fresh.")

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        # finalize() even on a crash/interrupt, so whatever metrics did
        # get logged are still visible in the Experiments viewer rather
        # than left in a dangling, never-finalized state.
        if lightning_experiment is not None:
            lightning_experiment.finalize()

    model.save_pretrained(f"{args.output_dir}/lora_adapters")
    processor.save_pretrained(f"{args.output_dir}/lora_adapters")
    model.save_pretrained_merged(f"{args.output_dir}/vllm_model", processor, save_method="merged_16bit")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
