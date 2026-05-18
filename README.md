# agtuned-vlm

Fine-tuning and evaluation framework for vision-language models on agricultural image understanding. The pipeline systematically benchmarks VLMs across four stages, aggregating results into a single comparison table.

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a — Generate | `src/data_generation/generate.py` | 3-pass VLM pipeline: vision captions → synthesis → compression → JSONL datasets |
| 1b — Reference DB | `src/data_generation/build_reference_db.py` | Export CSV + fetch reference excerpts from Wikipedia / USDA APHIS / CGIAR |
| 1c — Verify | `src/data_generation/verify.py` | Rewrite descriptions grounded in reference sources |
| 2 — Linear Probe | `src/linear_probe/probe.py` | Logistic regression on frozen embeddings across all datasets |
| 3 — Language Assessment | `src/language_assessment/assess.py` | Text-only agricultural VQA (ROUGE-SS + F1) |
| 4 — Vision Assessment | `src/vision_assessment/assess.py` | Multimodal VQA with image input (ROUGE-SS + F1) |

## Prerequisites

```bash
uv sync
```

## Running the Pipeline

All stages are accessible from the repo root via `main.py`:

```bash
python main.py <stage> [args...]
python main.py --help          # show all stages and examples
python main.py probe --help    # show options for a specific stage
```

### Stage 1a — Generate descriptions
Default model: `Qwen/Qwen3.6-35B-A3B`. Override with `--model`.
```bash
python main.py generate \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --output_dir outputs/data_generation/
```

### Stage 1b — Build reference database
Exports `descriptions_export.csv` as a side effect, then fetches reference sources.
```bash
python main.py build-db \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --output_db          outputs/data_generation/reference_db.json
```

### Stage 1c — Verify and rewrite descriptions
Default model: `Qwen/Qwen3.6-27B`. Override with `--model`.
```bash
python main.py verify \
  --descriptions_csv outputs/data_generation/descriptions_export.csv \
  --reference_db     outputs/data_generation/reference_db.json \
  --output_csv       outputs/data_generation/descriptions_corrected.csv
```

### Streamlit UI (browse generated samples and corrections)
```bash
streamlit run src/data_generation/app.py -- --output_dir outputs/data_generation/
```

### Stage 2 — Linear probe
```bash
python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/
```

### Stage 3 — Language assessment
```bash
python main.py language \
  --descriptions_jsonl   outputs/data_generation/agml_descriptions.jsonl \
  --classification_jsonl outputs/data_generation/agml_classification.jsonl \
  --config               src/config/datasets.yaml \
  --prompts              src/config/prompts.yaml \
  --models openai/gpt-oss-20b meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --output_dir           outputs/language_assessment/
```

### Stage 4 — Vision assessment
```bash
python main.py vision \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --config             src/config/datasets.yaml \
  --prompts            src/config/prompts.yaml \
  --models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --output_dir         outputs/vision_assessment/
```

## Outputs

```
outputs/
├── data_generation/       agml_descriptions.jsonl, agml_classification.jsonl, reference_db.json
├── linear_probe/          linear_probe_results.json, t-SNE plots, confusion matrices
├── language_assessment/   per-model CSVs, metrics JSON, summary.txt, plots
├── vision_assessment/     per-model CSVs, metrics JSON, summary.txt, plots
└── pipeline_results.csv   incremental summary table — updated after each stage
```

All prompt templates live in `src/config/prompts.yaml`. Edit there — no code changes needed.
