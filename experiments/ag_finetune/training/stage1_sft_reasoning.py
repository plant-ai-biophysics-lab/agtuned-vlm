"""
SFT on the consolidated Q&A data (build_sft_data.py) with LoRA (Unsloth).

Plain open-ended {question, answer} — no <think> chain wrapper (that was
the old stage4-based format; this simplified path drops it, see
build_sft_data.py's docstring).

Settled verifier model: Qwen/Qwen3.5-9B (default for --model). Chosen over
Gemma-4 after a real comparison run (see outputs/verifier_ppt_*/score_summary.json):
Gemma-4-26B-A4B verifying itself collapses hard (fine_grained_std=0.033,
essentially all mass at the ceiling); every Qwen pairing — self or
cross-model — showed 5-8x more spread. On top of that, Gemma-4 is MoE
(26B total / ~4B active), which brings real fine-tuning-specific risk Qwen's
dense architecture doesn't have: bitsandbytes can't quantize its fused expert
tensors (4-bit QLoRA is out, >40GB needed for bf16 LoRA), MoE-LoRA adapters
generally don't merge cleanly back into the base model (threatens the
save_pretrained_merged step below, which is exactly what makes the checkpoint
droppable into run_verifier_ppt.py's --verifier_model), and expert routing
can shift unpredictably during fine-tuning in a way dense models don't
exhibit. None of that is worth taking on for what's meant to be the simple
path.

Still trains via unsloth's FastVisionModel (not FastLanguageModel) — Qwen3.5
is vision-capable, and this repo's design keeps the verifier able to see the
image (run_verifier_ppt.py's --verifier_model flag), even though this
particular data (build_sft_data.py's Q&A) is text-only. Only the language
layers are unfrozen (finetune_vision_layers=False) accordingly — the
opposite split from experiments/lora_setup/lora_gemma4.py, which tunes
vision layers for image diagnosis and freezes language.

Saves both outputs, matching lora_gemma4.py's convention:
  <output_dir>/lora_adapters/  — raw PEFT adapters
  <output_dir>/vllm_model/     — merged 16-bit weights, directly loadable by
                                  vLLM: run_verifier_ppt.py --verifier_model <this path>

Loss/metrics tracking: report_to="trackio" (matches lora_gemma4.py) — a
local-first Gradio dashboard, no account needed; run `trackio show` after
training to view it, or pass --output_dir's path if it doesn't auto-detect.
Also writes <output_dir>/training_stats.json with the full per-step loss
history and peak GPU memory, so the numbers are on disk even if nobody had
the dashboard open during the run (same reasoning as lora_gemma4.py's
gpu_stats.json, extended with the loss curve it didn't capture).

Usage:
  uv run python experiments/ag_finetune/training/stage1_sft_reasoning.py \
    --data  outputs/sft_data.jsonl \
    --output_dir outputs/checkpoints/stage1_sft
  # (--model defaults to unsloth/Qwen3.5-9B; pass --model to override)
"""

import argparse
import json
from pathlib import Path

# Robust to CWD (matches data_prep/*.py's OUT_DIR pattern) — always resolves
# to experiments/ag_finetune/outputs/ regardless of where this script is
# invoked from, unlike a bare "outputs/..." default which only works if CWD
# happens to already be experiments/ag_finetune/.
_OUT_DIR = Path(__file__).parent.parent / "outputs"


def plot_loss_curve(training_stats: dict, output_dir: str, title_prefix: str) -> None:
    """Loss + LR curves from trainer.state.log_history — a local, always-
    available alternative to trackio's live dashboard (which needs HF PRO
    for hosted Space compute and may not always be reachable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = [e for e in training_stats["log_history"] if "loss" in e and "grad_norm" in e]
    if not history:
        print("  [skip] plot_loss_curve — no per-step loss entries in log_history")
        return

    steps = [e["step"] for e in history]
    losses = [e["loss"] for e in history]
    lrs = [e["learning_rate"] for e in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(steps, losses, color="#4C72B0", linewidth=1.2)
    ax1.set_ylabel("Training loss")
    ax1.set_title(
        f"{title_prefix} — final_train_loss={training_stats['final_train_loss']:.3f}, "
        f"runtime={training_stats['train_runtime_s'] / 3600:.1f}h, "
        f"peak_mem={training_stats['peak_reserved_memory_gb']:.1f}GB"
    )
    ax1.yaxis.grid(True, alpha=0.3)

    ax2.plot(steps, lrs, color="#C44E52", linewidth=1.2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning rate")
    ax2.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    p = Path(output_dir) / "loss_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def make_sample(question: str, answer: str) -> dict:
    """Text-only turn in the vision-model content-list convention (matches
    lora_gemma4.py's make_sample / this repo's documented multimodal sample
    format) — no image entries, since this data has none."""
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="unsloth/Qwen3.5-9B",
                        help="Vision-capable base model for the dedicated verifier — "
                             "see module docstring for why Qwen3.5-9B over Gemma-4.")
    parser.add_argument("--data",          default=str(_OUT_DIR / "sft_data.jsonl"))
    parser.add_argument("--output_dir",    default=str(_OUT_DIR / "checkpoints" / "stage1_sft"))
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--max_seq_len",   type=int,   default=2048)
    parser.add_argument("--trackio_space_id", default=None,
                        help="HF Space id, e.g. 'yourusername/agtuned-verifier-training'. "
                             "If set, trackio syncs to that Space instead of a local-only "
                             "dashboard — viewable live from any browser (requires being "
                             "logged into HF: huggingface-cli login or HF_TOKEN). Equivalent "
                             "to exporting TRACKIO_SPACE_ID yourself; this just saves typing "
                             "it every run.")
    args = parser.parse_args()

    if args.trackio_space_id:
        import os
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space_id
        print(f"trackio syncing to HF Space: {args.trackio_space_id}")

    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run build_sft_data.py first. Missing: {data_path}")
        return

    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} training examples")

    dataset = [make_sample(r["question"], r["answer"]) for r in rows]

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        # Qwen3.5's whole family (Qwen/unsloth org, all sizes) shards weights
        # as "model.safetensors-NNNNN-of-MMMMM.safetensors" instead of the
        # standard "model-NNNNN-of-MMMMM.safetensors" — a real upstream
        # naming quirk, not a broken repo (vLLM loads the same checkpoint
        # fine elsewhere in this repo). transformers' default file-discovery
        # (use_safetensors=None -> autodetect) misses this pattern and
        # raises "does not appear to have a file named ... model.safetensors".
        # Forcing True routes through a different discovery branch that
        # (per the underlying transformers bug report) actually looks for
        # this shard pattern.
        use_safetensors=True,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # text-only data — leave the vision tower alone
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_8bit",
            seed=3407,
            report_to="trackio",
            # Required for vision finetuning (see lora_gemma4.py):
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_len,
        ),
    )
    trainer_stats = trainer.train()

    # SFTConfig's max_length=args.max_seq_len (a training-time truncation
    # setting) ends up in the tokenizer's init_kwargs["max_length"] — a
    # SEPARATE field from init_kwargs["model_max_length"], which stays
    # correct (262144). Qwen3VLProcessor.__call__ merges the tokenizer's
    # entire init_kwargs into its own internal tokenizer call, so this
    # stray max_length silently truncates every future inference call
    # regardless of model_max_length or vLLM's --max_model_len. Confirmed
    # by directly reproducing the failure this caused (image-token mismatch
    # on a 3000x3000 image, input_ids truncated to exactly 2048) and
    # verifying this removal fixes it before adding this fix.
    processor.tokenizer.init_kwargs.pop("max_length", None)

    model.save_pretrained(f"{args.output_dir}/lora_adapters")
    processor.save_pretrained(f"{args.output_dir}/lora_adapters")
    model.save_pretrained_merged(f"{args.output_dir}/vllm_model", processor, save_method="merged_16bit")

    import torch
    training_stats = {
        "train_runtime_s": trainer_stats.metrics.get("train_runtime"),
        "final_train_loss": trainer_stats.metrics.get("train_loss"),
        "peak_reserved_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "log_history": trainer.state.log_history,  # per-step loss/lr, for plotting after the fact
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "training_stats.json").write_text(json.dumps(training_stats, indent=2))
    print(f"Saved training_stats.json (final_train_loss={training_stats['final_train_loss']})")
    plot_loss_curve(training_stats, args.output_dir, title_prefix=f"Stage 1 SFT ({args.model})")

    print(f"\nSaved adapters to {args.output_dir}/lora_adapters")
    print(f"Saved merged 16-bit weights to {args.output_dir}/vllm_model "
          f"(pass this path to run_verifier_ppt.py --verifier_model)")


if __name__ == "__main__":
    main()
