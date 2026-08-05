# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding.

## Prerequisites

```bash
uv sync
```

## Pipeline

Run stages in this order. Each links to its own doc under `docs/` — this file stays a table of contents, not a growing wall of instructions.

| Stage | Doc | What it does |
|---|---|---|
| 0a | [docs/stage0a_datasets.md](docs/stage0a_datasets.md) | Download & sample datasets from Project-AgML (HuggingFace) |
| 0 | [docs/stage0_data_generation.md](docs/stage0_data_generation.md) | Generate description/classification JSONL for language & vision assessment |
