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
| 0 | [docs/stage0_data_generation.md](docs/stage0_data_generation.md) | Fetch grounding sources (Wikipedia + web search) then synthesize classification JSONL from them (GPU) |
