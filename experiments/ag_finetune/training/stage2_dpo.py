"""
DPO alignment on the consolidated preference pairs (build_dpo_data.py), with
LoRA (Unsloth). Warm-starts from a Stage 1 SFT checkpoint by default.

Plain {prompt, chosen, rejected} — no verifier JSON schema, no confidence
field (that's stage7_verifier_dpo_pairs.jsonl's job, left untouched).

Settled verifier model: Qwen/Qwen3.5-9B, same as stage1_sft_reasoning.py —
see that script's docstring for why (real comparison run showed Gemma-4
self-verification collapsing hard, plus MoE-specific fine-tuning risk Qwen's
dense architecture avoids). Default --model here is stage1's own output, so
this warm-starts from the SFT'd Qwen3.5-9B rather than the raw base model.

Uses unsloth's FastVisionModel (not FastLanguageModel), text-only data so
only language layers are unfrozen — same reasoning as stage1_sft_reasoning.py.

Saves both outputs, matching stage1_sft_reasoning.py's convention:
  <output_dir>/lora_adapters/  — raw PEFT adapters
  <output_dir>/vllm_model/     — merged 16-bit weights, directly loadable by
                                  vLLM: run_verifier_ppt.py --verifier_model <this path>

Loss/metrics tracking: same as stage1_sft_reasoning.py — report_to="trackio"
(local-first Gradio dashboard) plus <output_dir>/training_stats.json with the
full per-step loss history and peak GPU memory.

Usage:
  uv run python experiments/ag_finetune/training/stage2_dpo.py \
    --data  outputs/dpo_pairs.jsonl \
    --output_dir outputs/checkpoints/stage2_dpo
  # (--model defaults to outputs/checkpoints/stage1_sft/vllm_model)
"""

import argparse
import json
from pathlib import Path

# Robust to CWD — see stage1_sft_reasoning.py's _OUT_DIR comment for why.
_OUT_DIR = Path(__file__).parent.parent / "outputs"


def plot_loss_curve(training_stats: dict, output_dir: str, title_prefix: str) -> None:
    """Duplicated from stage1_sft_reasoning.py, not imported, per this repo's
    convention of keeping experiment/training scripts self-contained."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=str(_OUT_DIR / "checkpoints" / "stage1_sft" / "vllm_model"),
                        help="Vision-capable base — a Stage 1 SFT merged checkpoint by "
                             "default, or pass a fresh base model to skip SFT warm-start.")
    parser.add_argument("--data",        default=str(_OUT_DIR / "dpo_pairs.jsonl"))
    parser.add_argument("--output_dir",  default=str(_OUT_DIR / "checkpoints" / "stage2_dpo"))
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=16)
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--max_seq_len", type=int,   default=1024)
    parser.add_argument("--trackio_space_id", default=None,
                        help="HF Space id — see stage1_sft_reasoning.py's --trackio_space_id "
                             "for details. Use the SAME Space id across stage1/stage2 to see "
                             "both runs in one dashboard, grouped by run name.")
    args = parser.parse_args()

    if args.trackio_space_id:
        import os
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space_id
        print(f"trackio syncing to HF Space: {args.trackio_space_id}")

    from unsloth import FastVisionModel, is_bf16_supported
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Run build_dpo_data.py first. Missing: {data_path}")
        return

    pairs = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs):,} DPO pairs")

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        # See stage1_sft_reasoning.py's from_pretrained comment — Qwen3.5's
        # non-standard sharded-safetensors naming trips up transformers'
        # default file-discovery; forcing use_safetensors=True works around it.
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

    dataset = Dataset.from_dict({
        "prompt":   [p["prompt"]   for p in pairs],
        "chosen":   [p["chosen"]   for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=processor,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=20,
            save_steps=200,
            max_length=args.max_seq_len,
            report_to="trackio",
        ),
    )
    trainer_stats = trainer.train()

    # See stage1_sft_reasoning.py's comment above the same line — the
    # trainer sets processor.tokenizer.model_max_length to --max_seq_len,
    # which then silently truncates every future inference call unless reset.
    processor.tokenizer.model_max_length = 131072

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
    plot_loss_curve(training_stats, args.output_dir, title_prefix=f"Stage 2 DPO ({args.model})")

    print(f"\nSaved adapters to {args.output_dir}/lora_adapters")
    print(f"Saved merged 16-bit weights to {args.output_dir}/vllm_model "
          f"(pass this path to run_verifier_ppt.py --verifier_model)")


if __name__ == "__main__":
    main()
