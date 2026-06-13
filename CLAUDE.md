# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`agtuned-vlm` is a research framework for fine-tuning and evaluating Gemma-4 vision-language models on agricultural image understanding tasks. It systematically tests three hypotheses about why VLMs fail at agricultural tasks through modular, standalone experiments.

## Setup and Running Experiments

This project uses `uv` for dependency management with Python 3.12.

```bash
# Install dependencies
uv sync

# Run any experiment
uv run python experiments/<module>/<script>.py --<args>
```

Each experiment is a self-contained CLI script. There is no top-level orchestrator — `main.py` is a placeholder.

## Experiment Modules

| Module | Script | Purpose |
|---|---|---|
| `lora_setup/` | `lora_gemma4.py` | Fine-tune Gemma-4 with LoRA on PlantVillage |
| `linear_probe/` | `linear_probe.py` | H1: Test vision encoder separability across backbones |
| `text_qa/` | `text_qa.py` | H2: Agricultural language knowledge (text-only) |
| `vllm_inference/` | `gemma4_inference.py` | Benchmark inference speed + accuracy via vLLM |
| `vision_feats/` | `01–04_*.py` | Analyze vision representations at different model levels |
| `alignment_test/` | `check_alignment.py` | H3: Vision-text alignment via attention-bbox overlap |
| `vocab_grounding/` | `vocab_grounding.py` | Test whether injecting class-discriminative vocab tokens into prompts improves F1 |

### Common CLI arguments
- `--model`: HuggingFace model ID or local path (Gemma-4 variants: E2B, E4B, 26B, 31B)
- `--output_dir`: where to write results (JSON/CSV metrics + visualizations)
- `--agml_dataset`: AgML dataset name for linear probe / LoRA training
- `--n_samples`: limit evaluation samples

## Architecture

**Three-hypothesis structure**: Each major experiment maps to a hypothesis:
- **H1 (linear_probe)**: Did fine-tuning improve the vision encoder's feature separability?
- **H2 (text_qa)**: Does Gemma-4 have sufficient agricultural language knowledge?
- **H3 (alignment_test)**: Are vision-token attention maps aligned with ground-truth bounding boxes?

**LoRA fine-tuning** (`lora_setup/`): Uses `unsloth` + `trl.SFTTrainer`. Only vision layers and attention are tuned (language layers frozen). Saves both merged 16-bit weights (`vllm_model/`) and raw adapters (`lora_adapters/`). Quantization: 4-bit loading, bf16 training. Training metrics written to `gpu_stats.json`.

**Linear probe** (`linear_probe/`): Extracts frozen embeddings from four backbones (fine-tuned Gemma-4, base Gemma-4, DINOv2, YOLOv8), trains logistic regression, compares accuracy/F1 + t-SNE + confusion matrices.

**Vision feature analysis** (`vision_feats/`): Four scripts analyze different representation stages — DINOv2 patch attention, iNatAg Swin GradCAM, and Gemma-4 at three levels (pre-projector SigLIP patches, post-projector, cross-attention).

**Multimodal sample format** used across experiments:
```python
{"messages": [
    {"role": "user", "content": [<PIL image>, <text prompt>]},
    {"role": "assistant", "content": [<text label>]}
]}
```

**Attention extraction requires eager mode**: Scripts that visualize attention (vision_feats, alignment_test) must load the model with `attn_implementation="eager"` — flash-attention does not expose attention weights.

## Key Dependencies

- `unsloth`: Gemma-4 with optimized LoRA
- `trl` + `peft`: SFTTrainer and LoRA adapters
- `agml`: Agricultural ML datasets (PlantVillage, Digital Green)
- `vllm`: Optimized inference engine for benchmarking
- `timm`: DINOv2, Swin Transformer backbones
- `datasets`: KisanVaani agriculture-qa dataset for H2
