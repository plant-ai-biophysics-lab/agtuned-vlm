# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding.

## Prerequisites

```bash
uv sync
```

## Experiments Pipeline

| Stage | Doc | What it does |
|---|---|---|
| 0a | [docs/stage0a_datasets.md](docs/stage0a_datasets.md) | Download & sample datasets from Project-AgML (HuggingFace) |
| 0b | [docs/stage0_data_generation.md](docs/stage0_data_generation.md) | (OPTIONAL) Fetch grounding sources (Wikipedia + web search) then synthesize classification JSONL from them |
| 1 | [docs/stage1_linear_probe.md](docs/stage1_linear_probe.md) | H1: test vision encoder feature separability across backbones |
| 2 | [docs/stage2_lower_upper_bounds.md](docs/stage2_lower_upper_bounds.md) | Lower (vision-only) vs. upper (description-augmented) classification bounds |
| 3 | [docs/stage3_cot.md](docs/stage3_cot.md) | Chain-of-thought: self-generated description vs. structured multi-step reasoning |

## Viewers

```bash
streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
```

Browse Stage 0b's output: dataset/crop/class per record, the grounding sources fetched for it, and the synthesized description. Works right after Step 1 alone (before descriptions exist) too — see [app/README.md](app/README.md).
