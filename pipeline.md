# Pipeline: AgML Agricultural VLM Benchmark

## Goal

Combine four experiments into a cohesive, reproducible pipeline that evaluates VLMs on
agricultural image understanding. Results aggregate into a single comparison table updated
after each stage completes.

---

## Directory Structure

```
src/
├── config/
│   ├── datasets.yaml               # master dataset registry — shared by all stages
│   └── prompts.yaml                # all prompt templates — edit here, not in code
├── data_generation/
│   ├── generate.py                 # Stage 1a: LLM-generated descriptions & classification pairs
│   ├── build_reference_db.py       # Stage 1b: validate with Wikipedia / IDtools / CGIAR
│   ├── app.py                      # UI to browse generated samples
│   └── README.md                   # scraping & generation methodology
├── linear_probe/
│   └── probe.py                    # Stage 2: logistic regression on frozen embeddings (all datasets)
├── language_assessment/
│   └── assess.py                   # Stage 3: text-only QA (language models)
├── vision_assessment/
│   └── assess.py                   # Stage 4: multimodal QA (VLMs)
└── common/
    ├── dataset_config.py           # load / parse datasets.yaml
    ├── metrics.py                  # ROUGE, F1, accuracy, ROUGE-SS, LLM judge
    └── io.py                       # JSONL / CSV read-write helpers

outputs/
├── data_generation/
├── linear_probe/
├── language_assessment/
├── vision_assessment/
└── pipeline_results.csv            # incremental — updated after each stage
```

---

## Stages

### Stage 1a — Generate Descriptions (`src/data_generation/generate.py`)

- Loads all non-skipped datasets from `src/config/datasets.yaml`
- For each dataset × class: vision pass (N images → raw captions) + synthesis pass → description
- Prompts loaded from `src/config/prompts.yaml`
- Outputs: `agml_descriptions.jsonl`, `agml_classification.jsonl`, `descriptions_export.csv`
- Checkpointed per dataset/class — safe to resume

```bash
uv run python src/data_generation/generate.py \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --output_dir outputs/data_generation/
```

### Stage 1b — Build Reference DB (`src/data_generation/build_reference_db.py`)

- Fetches Wikipedia, USDA APHIS IDtools, CGIAR GARDIAN for each class label in descriptions_export.csv
- Outputs: `reference_db.json`

```bash
uv run python src/data_generation/build_reference_db.py \
  --descriptions_csv outputs/data_generation/descriptions_export.csv \
  --output_db        outputs/data_generation/reference_db.json
```

### Stage 2 — Linear Probe (`src/linear_probe/probe.py`)

- Runs all datasets from `datasets.yaml` in a single pass
- Backbones: `dinov3`, `yolo11x-cls`, `gemma-4-base`, `gemma-4-finetuned`
- Aggregates F1/accuracy per dataset → per task type → overall
- t-SNE plots (test set only), confusion matrices — paper-ready
- Updates `pipeline_results.csv` on completion

```bash
uv run python src/linear_probe/probe.py \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/
```

### Stage 3 — Language Assessment (`src/language_assessment/assess.py`)

- Models: `openai/gpt-oss-20b`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Description task: ROUGE-SS primary metric
- Classification task: F1-macro primary metric; LLM judge for classification only
- Prompts from `src/config/prompts.yaml`
- Metrics saved per dataset → per task type → overall
- Updates `pipeline_results.csv` on completion

```bash
uv run python src/language_assessment/assess.py \
  --descriptions_jsonl     outputs/data_generation/agml_descriptions.jsonl \
  --classification_jsonl   outputs/data_generation/agml_classification.jsonl \
  --config                 src/config/datasets.yaml \
  --prompts                src/config/prompts.yaml \
  --models openai/gpt-oss-20b meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --output_dir             outputs/language_assessment/
```

### Stage 4 — Vision Assessment (`src/vision_assessment/assess.py`)

- Models: `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.6-27B`
- Same tasks and metrics as Stage 3 but with image input
- Updates `pipeline_results.csv` on completion

```bash
uv run python src/vision_assessment/assess.py \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --output_dir outputs/vision_assessment/
```

---

## Results Table (`outputs/pipeline_results.csv`)

Updated incrementally — each stage appends its columns. Final table:

| Model Types | Analysis Type | Model | Linear Probe F1 | Lang. Description ROUGE-SS | Lang. Classification F1 | Vision Description ROUGE-SS | Vision Classification F1 |
|---|---|---|---|---|---|---|---|
| Baseline | naïve | — | — | — | — | — | — |
| Vision Models | — | yolo11x-cls | 0.47 | — | — | — | — |
| Vision Models | — | dinov3 | 0.86 | — | — | — | — |
| Language Models | — | gpt-oss-20b | — | — | — | — | — |
| Language Models | — | llama-4-scout-17b | — | — | — | — | — |
| VLMs | — | gemma-4-26b-a4b-it | 0.86 | — | — | — | — |
| VLMs | — | qwen3.6-27b | — | — | — | — | — |

---

## Metrics Summary

| Stage | Task | Primary | Also Reported |
|---|---|---|---|
| Linear Probe | classification | F1-macro, accuracy | per dataset, per task type |
| Language Assessment | description | ROUGE-SS | ROUGE-1, ROUGE-2, ROUGE-L |
| Language Assessment | classification | F1-macro | F1-weighted, accuracy, contains-accuracy, judge-accuracy |
| Vision Assessment | description | ROUGE-SS | ROUGE-1, ROUGE-2, ROUGE-L |
| Vision Assessment | classification | F1-macro | F1-weighted, accuracy, contains-accuracy, judge-accuracy |

All metrics saved at three granularities: per dataset, per task type, and overall summary.

---

## Prompts (`src/config/prompts.yaml`)

All question templates live in `prompts.yaml`. Edit there — no code changes needed.
Template keys used across stages:

- `visual_description_disease` / `visual_description_weed`
- `text_description_disease` / `text_description_weed`
- `visual_classification_disease` / `visual_classification_weed`
- `text_classification_disease` / `text_classification_weed`
- `synthesis`

---

## Code Standards

- `src/common/` provides shared utilities; **never duplicate logic across stages**
- Each stage script is a standalone CLI (`argparse`) — runnable independently
- Paper-ready plots: no label overlap, consistent color scheme, task-level and summary-level
- Professional naming, no redundant comments, no dead code
