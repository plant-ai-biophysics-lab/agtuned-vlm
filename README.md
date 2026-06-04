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
| 7 — Augmented Vision | `src/vision_assessment/assess_augmented.py` | Re-run vision assessment with hint injection: description hint for classification, class-name hint for description |
| vision-augmented-plot | `src/vision_assessment/plot_augmented.py` | Regenerate augmented vision assessment charts (including before/after) from saved metrics JSON (no inference) |
| 8 — Fine-tune | `src/finetune/train.py` | LoRA fine-tuning on Digital Green dataset with configurable component targeting |

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

### Stage 1d — Build Digital Green JSONL

Direct adapter — no inference required. Converts existing `diagnosis` + `details` columns from the Digital Green annotations CSV into the same JSONL format produced by Stages 1a–1c, so all downstream stages work unchanged.

```bash
python main.py build-dg-jsonl \
  --annotations  data/digital_green/annotations.csv \
  --images_dir   data/digital_green/images/ \
  --output_dir   outputs/data_generation/ \
  --prompts      src/config/prompts.yaml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--annotations` | `data/digital_green/annotations.csv` | Path to Digital Green annotations CSV |
| `--images_dir` | `data/digital_green/images/` | Local cache for downloaded images |
| `--output_dir` | `outputs/data_generation/` | Writes `dg_descriptions.jsonl`, `dg_classification.jsonl`, `digital_green_datasets.yaml` |
| `--prompts` | `src/config/prompts.yaml` | Prompts config (same file used by all stages) |

Outputs:
- `dg_descriptions.jsonl` — one record per (crop, diagnosis) group, canonical description = most common `details` text
- `dg_classification.jsonl` — matching classification records
- `digital_green_datasets.yaml` — dataset config with discovered classes (passed to `--dg_config` in assess stages)

---

### Streamlit UI (browse generated samples and corrections)

```bash
streamlit run src/data_generation/app.py -- --output_dir outputs/data_generation/
```

Click any row in the **Reference Database** or **Corrected Descriptions** tab to see the sample image alongside the description and reference excerpts.

---

### Stage 2 — Linear probe

Evaluates frozen-embedding separability across one or more backbones. Models are loaded and unloaded sequentially to stay within GPU memory.

Always runs all four probe experiments in a single call:

| Experiment | Probe | Scope |
|------------|-------|-------|
| LR per-dataset | Logistic regression | One probe per dataset |
| LR combined | Logistic regression | All datasets merged into one label space |
| FC per-dataset | Single `nn.Linear` + Adam | One probe per dataset |
| FC combined | Single `nn.Linear` + Adam | All datasets merged into one label space |

```bash
python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/ \
  --models     google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --n_samples  300
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | none | One or more HuggingFace model IDs (VLMs) |
| `--dino_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | DINOv3 backbone. Pass `none` to skip |
| `--yolo_model` | `yolo11x-cls.pt` | YOLO classification backbone. Pass `none` to skip |
| `--n_samples` | all | Images per class to embed (stratified) |
| `--quantize` | none | `4bit` or `8bit` (BitsAndBytes) for large VLMs |
| `--fc_epochs` | `100` | Training epochs for FC probe |
| `--fc_lr` | `1e-3` | Adam learning rate for FC probe |
| `--fc_batch_size` | `256` | Batch size for FC probe training |
| `--dataset_source` | `agml` | `agml` or `digital_green` |
| `--images_dir` | none | Local image directory (required for `digital_green`) |

**Output structure:**

```
outputs/linear_probe/
├── linear_probe_results.json       all four probe results (collate reads from "overall")
├── probe_comparison_all.png        master chart — all 4 experiments × all backbones
├── summary_overall.png             LR per-dataset F1 + accuracy across backbones
├── summary_by_task.png             LR per-dataset F1 by task type
├── tsne/
│   └── tsne_{dataset}_{backbone}.png
├── confusion_matrices/
│   └── confusion_matrix_{dataset}_{backbone}.png
├── per_class_f1/
│   └── per_class_f1_{dataset}.png
└── loss_curves/
    ├── fc_loss_{dataset}_{backbone}_fc.png   FC per-dataset loss curves
    └── fc_loss_combined_{backbone}_fc.png    FC combined loss curves
```

Loss curves show **train loss** (solid) and **val loss** (dashed) vs. epoch. If val loss diverges early, lower `--fc_lr`. If both are still dropping at the last epoch, increase `--fc_epochs`.

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
| `--enforce_eager` | off | Disable CUDA graph capture — use if Triton kernel crashes with AWQ models |
| `--dataset_source` | `agml` | `agml` or `digital_green` — selects input JSONL and dataset config |
| `--dg_descriptions_jsonl` | `outputs/data_generation/dg_descriptions.jsonl` | Used when `--dataset_source digital_green` |
| `--dg_classification_jsonl` | `outputs/data_generation/dg_classification.jsonl` | Used when `--dataset_source digital_green` |
| `--dg_config` | `outputs/data_generation/digital_green_datasets.yaml` | Used when `--dataset_source digital_green` |

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
| `--enforce_eager` | off | Disable CUDA graph capture — use if Triton kernel crashes with AWQ models |
| `--dataset_source` | `agml` | `agml` or `digital_green` — selects input JSONL, dataset config, and image directory |
| `--dg_descriptions_jsonl` | `outputs/data_generation/dg_descriptions.jsonl` | Used when `--dataset_source digital_green` |
| `--dg_config` | `outputs/data_generation/digital_green_datasets.yaml` | Used when `--dataset_source digital_green` |
| `--images_dir` | none | Local directory of Digital Green images (used when `--dataset_source digital_green`) |

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
| **Both pass** | Model is fully aligned |
| **Lang only** | Language correct, vision wrong — textual knowledge compensates, visual alignment broken |
| **Vision only** | Vision correct, language wrong — visual knowledge compensates |
| **Both fail** | Both wrong — alignment broken (visual features are separable per LP, but the visual-to-output pathway fails) |

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

### Stage 7 — Augmented Vision Assessment

Re-runs the vision assessment with hint injection to test the alignment bottleneck (H3). Two augmentation modes are available:

| Mode | What changes |
|------|-------------|
| `cls_with_desc` | The model's own predicted visual description is prepended to the classification prompt |
| `desc_with_class` | The ground-truth class label is prepended to the description prompt |

By default `cls_with_desc` uses the model's own descriptions from the prior Stage 4 run. Pass `--use_groundtruth_desc` to use the ground-truth descriptions from `agml_descriptions.jsonl` instead.

```bash
python main.py vision-augmented \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --config             src/config/datasets.yaml \
  --prompts            src/config/prompts.yaml \
  --models             google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --augment            both \
  --baseline_dir       outputs/vision_assessment/ \
  --output_dir         outputs/vision_assessment_augmented/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of multimodal model IDs |
| `--augment` | `both` | `cls_with_desc`, `desc_with_class`, or `both` |
| `--baseline_dir` | `../vision_assessment/` | Directory containing Stage 4 `desc_<model>.csv` files (used as hint source) |
| `--use_groundtruth_desc` | off | Use ground-truth descriptions instead of model-predicted ones for `cls_with_desc` |
| `--quantization` | none | `awq`, `gptq`, `fp8`, or `bitsandbytes` |
| `--max_num_seqs` | 64 | Max concurrent sequences |
| `--n_samples` | all | Limit images per class |
| `--enforce_eager` | off | Disable CUDA graph capture |

#### Regenerate charts without re-running inference

```bash
python main.py vision-augmented-plot \
  --output_dir   outputs/vision_assessment_augmented/ \
  --baseline_dir outputs/vision_assessment/ \
  --config       src/config/datasets.yaml \
  --multimodal_models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B
```

---

### Stage 8 — Fine-tune

> **Separate environment required.** unsloth and vLLM conflict on torch/triton versions and cannot share the root environment. The fine-tuning script has its own `pyproject.toml` in `src/finetune/`.
>
> ```bash
> cd src/finetune && uv sync   # one-time setup
> ```
>
> Run all fine-tuning commands from `src/finetune/` using `uv run`. All other pipeline stages use the root environment as normal.

LoRA fine-tuning on the Digital Green dataset using unsloth. `--train_layers` and `--tasks` are independent flags so you can mix any combination.

Images are downloaded from S3 on first run and cached locally. Each row produces one sample per task per epoch.

#### Running the three experiments

```bash
# From src/finetune/ with its own uv environment:
cd src/finetune && uv sync   # first time only

# Experiment 1 — vision layers only, image classification only
uv run python ../../main.py finetune \
  --model        unsloth/gemma-4-E2B-it \
  --annotations  /path/to/annotations.csv \
  --train_layers vision \
  --tasks        vis_cls \
  --output_dir   ../../outputs/finetune/

# Experiment 2 — language layers only, text tasks only
uv run python ../../main.py finetune \
  --model        unsloth/gemma-4-E2B-it \
  --annotations  /path/to/annotations.csv \
  --train_layers language \
  --tasks        lang_cls lang_desc \
  --output_dir   ../../outputs/finetune/

# Experiment 3 — both layers, all 4 tasks
uv run python ../../main.py finetune \
  --model        unsloth/gemma-4-E2B-it \
  --annotations  /path/to/annotations.csv \
  --train_layers vision_language \
  --tasks        vis_cls vis_desc lang_cls lang_desc \
  --output_dir   ../../outputs/finetune/

# Output structure (auto-generated from --train_layers and --model):
#   outputs/finetune/vision/gemma-4-E2B-it/
#   outputs/finetune/language/gemma-4-E2B-it/
#   outputs/finetune/vision_language/gemma-4-E2B-it/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/gemma-4-E2B-it` | Any unsloth-compatible model ID |
| `--annotations` | `data/digital_green/annotations.csv` | Path to Digital Green annotations CSV |
| `--images_dir` | `data/digital_green/images/` | Local cache for downloaded images |
| `--train_layers` | `vision_language` | Which weights get gradient updates: `vision`, `language`, or `vision_language` |
| `--tasks` | all four | Prompt configs to use: any subset of `vis_cls vis_desc lang_cls lang_desc` |
| `--epochs` | `1` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `4` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--lora_r` | `16` | LoRA rank (lora_alpha = lora_r) |
| `--load_in_4bit` | off | Load model in 4-bit BnB quantization |

Outputs per run: `lora_adapters/` (raw LoRA weights), `vllm_model/` (merged 16-bit), `train_stats.json`. Training is logged to trackio.

#### Evaluating fine-tuned models

Point `--models` at the local `vllm_model/` directory — vLLM accepts local paths the same as HuggingFace IDs:

```bash
python main.py language \
  --models outputs/finetune/vision_language/vllm_model/ \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --classification_jsonl outputs/data_generation/agml_classification.jsonl \
  --config src/config/datasets.yaml \
  --prompts src/config/prompts.yaml \
  --output_dir outputs/language_assessment_finetuned/

python main.py vision \
  --models outputs/finetune/vision_language/vllm_model/ \
  --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \
  --config src/config/datasets.yaml \
  --prompts src/config/prompts.yaml \
  --output_dir outputs/vision_assessment_finetuned/

python main.py error-overlap \
  --language_dir outputs/language_assessment_finetuned/ \
  --vision_dir   outputs/vision_assessment_finetuned/ \
  --output_dir   outputs/error_overlap_finetuned/
```

Compare the quadrant distributions in `error_overlap_finetuned/` against the baseline `error_overlap/` to quantify alignment improvement.

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
├── vision_assessment_augmented/
│   ├── desc_<model>_desc_with_class.csv / cls_<model>_cls_with_desc.csv
│   ├── vision_assessment_augmented_metrics.json
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
