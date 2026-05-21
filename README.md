# agtuned-vlm

Fine-tuning and evaluation framework for vision-language models on agricultural image understanding. The pipeline systematically benchmarks VLMs across four stages, aggregating results into a single comparison table.

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a — Generate | `src/data_generation/generate.py` | 3-pass VLM pipeline: vision captions → synthesis → compression → JSONL datasets |
| 1b — Reference DB | `src/data_generation/build_reference_db.py` | Export CSV + fetch reference excerpts from Wikipedia / USDA APHIS / CGIAR |
| 1c — Verify | `src/data_generation/verify.py` | Rewrite descriptions grounded in reference sources; healthy classes are overridden automatically |
| patch-healthy | `src/data_generation/patch_healthy.py` | Patch healthy-class descriptions in an already-generated CSV without re-running verify |
| 2 — Linear Probe | `src/linear_probe/probe.py` | Logistic regression on frozen embeddings across all datasets and backbones |
| 3 — Language Assessment | `src/language_assessment/assess.py` | Text-only agricultural VQA (ROUGE-SS + F1) — also supports VLMs |
| language-plot | `src/language_assessment/plot.py` | Regenerate language assessment charts from saved metrics JSON (no inference) |
| 4 — Vision Assessment | `src/vision_assessment/assess.py` | Multimodal VQA with image input (ROUGE-SS + F1) |
| vision-plot | `src/vision_assessment/plot.py` | Regenerate vision assessment charts from saved metrics JSON (no inference) |
| 5 — Collate | `src/collate_results.py` | Merge all stage metrics into a single CSV + LaTeX results table |
| 6 — Error Overlap | `src/error_overlap/analyze.py` | Per-class error overlap between language and vision assessments (quadrant analysis) |

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

---

### Stage 1a — Generate descriptions

Default model: `Qwen/Qwen3.6-35B-A3B`. Override with `--model`.

```bash
python main.py generate \
  --model      Qwen/Qwen3.6-35B-A3B \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --output_dir outputs/data_generation/
```

One sample image per class is saved to `outputs/data_generation/sample_images/` for use in the Streamlit UI. If you already have a checkpoint and just want to backfill the images without re-running inference:

```bash
python main.py generate --output_dir outputs/data_generation/ --backfill_images
```

---

### Stage 1b — Build reference database

Exports `descriptions_export.csv` as a side effect, then fetches reference excerpts (up to 1 000 chars × top 3 sources, separated by `---`).

```bash
python main.py build-db \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --output_db          outputs/data_generation/reference_db.json
```

Stage 1b and 1c can be re-run without deleting existing files — they resume from checkpoints.

---

### Stage 1c — Verify and rewrite descriptions

Default model: `Qwen/Qwen3.6-35B-A3B`. Override with `--model`.
Classes whose label contains `"healthy"` are automatically overridden to `"This leaf/plant is healthy."` without calling the LLM.

```bash
python main.py verify \
  --descriptions_csv outputs/data_generation/descriptions_export.csv \
  --reference_db     outputs/data_generation/reference_db.json \
  --output_csv       outputs/data_generation/descriptions_corrected.csv
```

#### Patch healthy classes in an existing CSV

```bash
python main.py patch-healthy \
  --csv outputs/data_generation/descriptions_corrected.csv
```

---

### Streamlit UI (browse generated samples and corrections)

```bash
streamlit run src/data_generation/app.py -- --output_dir outputs/data_generation/
```

Click any row in the **Reference Database** or **Corrected Descriptions** tab to see the sample image alongside the description and reference excerpts.

---

### Stage 2 — Linear probe

Evaluates frozen-embedding separability across one or more backbones. Models are loaded and unloaded sequentially to stay within GPU memory.

```bash
python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/ \
  --models     google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --n_samples  300
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | One or more HuggingFace model IDs |
| `--n_samples` | all | Images per class to embed |
| `--quantize` | none | `4bit` or `8bit` (BitsAndBytes) for large models |

---

### Stage 3 — Language assessment

Evaluates models on text-only agricultural VQA. VLMs (multimodal models) can also be evaluated here using text-only inputs. Each model runs in an isolated subprocess — GPU memory is fully released between models. Mamba-architecture models have their `max_num_seqs` lowered automatically.

```bash
python main.py language \
  --descriptions_jsonl   outputs/data_generation/agml_descriptions.jsonl \
  --classification_jsonl outputs/data_generation/agml_classification.jsonl \
  --config               src/config/datasets.yaml \
  --prompts              src/config/prompts.yaml \
  --models               openai/gpt-oss-20b google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --multimodal_models    google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --output_dir           outputs/language_assessment/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of model IDs |
| `--multimodal_models` | none | Subset of `--models` that are VLMs — shown with diagonal hatching in charts |
| `--quantization` | none | `awq`, `gptq`, `fp8`, or `bitsandbytes` |
| `--max_num_seqs` | 128 | Max concurrent sequences; auto-lowered for Mamba models |
| `--n_samples` | all | Limit records per JSONL |
| `--judge_model` | none | LLM-as-a-judge model for classification (runs after all eval models) |

Charts produced: classification per dataset, classification per task type, description ROUGE per task type, overall description ROUGE, overall classification.

#### Regenerate charts without re-running inference

```bash
python main.py language-plot \
  --output_dir        outputs/language_assessment/ \
  --config            src/config/datasets.yaml \
  --multimodal_models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B
```

---

### Stage 4 — Vision assessment

```bash
python main.py vision \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --config             src/config/datasets.yaml \
  --prompts            src/config/prompts.yaml \
  --models             google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --output_dir         outputs/vision_assessment/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of multimodal model IDs |
| `--quantization` | none | `awq`, `gptq`, `fp8`, or `bitsandbytes` |
| `--max_num_seqs` | 64 | Max concurrent sequences |
| `--n_samples` | all | Limit images per class |

---

### Stage 5 — Collate results

Merges outputs from all stages into `results_table.csv` and `results_table.tex`. Uses judged F1 where available, falls back to standard F1. Naive baseline is included as the first row.

```bash
python main.py collate \
  --linear_probe_dir outputs/linear_probe/ \
  --language_dir     outputs/language_assessment/ \
  --vision_dir       outputs/vision_assessment/ \
  --output_dir       outputs/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--linear_probe_dir` | `outputs/linear_probe/` | Directory containing `linear_probe_results.json` |
| `--language_dir` | `outputs/language_assessment/` | Directory containing `language_assessment_metrics.json` |
| `--vision_dir` | `outputs/vision_assessment/` | Directory containing `vision_assessment_metrics.json` |
| `--output_dir` | `outputs/` | Where to write `results_table.csv` and `results_table.tex` |

The LaTeX table uses `\resizebox{\textwidth}{!}{...}` and requires `booktabs` and `multirow` packages.

---

### Stage 6 — Error overlap analysis

Classifies every (dataset, class) pair into one of four quadrants based on per-class accuracy in language vs. vision assessment:

| Quadrant | Interpretation |
|----------|---------------|
| **Both pass** | Model is aligned — sees and knows |
| **Lang only** | Language correct, vision wrong — alignment broken |
| **Vision only** | Vision correct, language wrong — visual knowledge compensates |
| **Both fail** | Both wrong — knowledge bottleneck |

Requires models evaluated in **both** language and vision assessment (only VLMs appear here).

```bash
python main.py error-overlap \
  --language_dir outputs/language_assessment/ \
  --vision_dir   outputs/vision_assessment/ \
  --output_dir   outputs/error_overlap/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--language_dir` | `outputs/language_assessment/` | Directory with `cls_*.csv` from language assessment |
| `--vision_dir` | `outputs/vision_assessment/` | Directory with `cls_*.csv` from vision assessment |
| `--output_dir` | `outputs/error_overlap/` | Where to write per-model CSVs, scatter plots, and summary |
| `--threshold` | `0.5` | Per-class accuracy threshold for pass/fail |

Outputs per model: a scatter plot (language accuracy vs. vision accuracy, colour-coded by quadrant) and a per-class CSV. A stacked bar `error_overlap_heatmap.png` compares quadrant fractions across all models.

---

## Outputs

```
outputs/
├── data_generation/
│   ├── agml_descriptions.jsonl       open-ended description Q&A pairs
│   ├── agml_classification.jsonl     short-answer classification Q&A pairs
│   ├── descriptions_export.csv       flat export of all generated descriptions
│   ├── descriptions_corrected.csv    after Stage 1c verify + rewrite
│   ├── reference_db.json             Wikipedia / USDA APHIS / CGIAR excerpts
│   └── sample_images/                one image per class per dataset (for UI)
├── linear_probe/
│   ├── linear_probe_results.json
│   ├── t-SNE plots
│   └── confusion matrices
├── language_assessment/
│   ├── desc_<model>.csv / cls_<model>.csv   per-model prediction rows
│   ├── language_assessment_metrics.json     all metrics (input to language-plot)
│   ├── summary.txt
│   └── *.png                                five chart files
├── vision_assessment/
│   ├── desc_<model>.csv / cls_<model>.csv
│   ├── vision_assessment_metrics.json
│   ├── summary.txt
│   └── *.png
├── results_table.csv                 collated metrics across all stages
├── results_table.tex                 LaTeX booktabs table (ready to \input into a paper)
└── error_overlap/
    ├── error_overlap_<model>.csv     per-class quadrant table
    ├── error_overlap_<model>.png     scatter plot (lang acc vs vision acc)
    ├── error_overlap_heatmap.png     quadrant fraction comparison across models
    └── error_overlap_summary.json    aggregate quadrant counts per model
```

All prompt templates live in `src/config/prompts.yaml`. Edit there — no code changes needed.
