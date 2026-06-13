# agtuned-vlm

Fine-tuning and evaluation framework for vision-language models on agricultural image understanding. The pipeline systematically benchmarks VLMs across four stages, aggregating results into a single comparison table.

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a — Generate | `src/data_generation/generate.py` | 3-pass VLM pipeline: vision captions → synthesis → compression → JSONL datasets |
| 1b — Reference DB | `src/data_generation/build_reference_db.py` | Export CSV + fetch reference excerpts from Wikipedia / USDA APHIS / CGIAR |
| 1c — Verify | `src/data_generation/verify.py` | Rewrite descriptions grounded in reference sources; healthy classes are overridden automatically |
| patch-healthy | `src/data_generation/patch_healthy.py` | Patch healthy-class descriptions in an already-generated CSV without re-running verify |
| Layer Explorer | `app/layer_explorer.py` | Streamlit GUI: browse any HF model's module tree, select pre/post-projector layers, save to `src/config/layer_selections.yaml` |
| 2 — Linear Probe | `src/linear_probe/probe.py` | Frozen-embedding separability across backbones (LR + optional FC, per-dataset + combined) |
| 3 — Language Assessment | `src/language_assessment/assess.py` | Text-only agricultural VQA (ROUGE-SS + F1) — also supports VLMs |
| language-plot | `src/language_assessment/plot.py` | Regenerate language assessment charts from saved metrics JSON (no inference) |
| 4 — Vision Assessment | `src/vision_assessment/assess.py` | Multimodal VQA with image input (ROUGE-SS + F1) |
| vision-plot | `src/vision_assessment/plot.py` | Regenerate vision assessment charts from saved metrics JSON (no inference) |
| 5 — Collate | `src/collate_results.py` | Merge all stage metrics into a single CSV + LaTeX results table |
| 6 — Error Overlap | `src/error_overlap/analyze.py` | Per-class error overlap between language and vision assessments (quadrant analysis) |
| 7 — Augmented Vision | `src/vision_assessment/assess_augmented.py` | Re-run vision assessment with hint injection: description hint for classification, class-name hint for description |
| vision-augmented-plot | `src/vision_assessment/plot_augmented.py` | Regenerate augmented vision assessment charts (including before/after) from saved metrics JSON (no inference) |
| 8 — Fine-tune | `src/finetune/train.py` | LoRA fine-tuning on Digital Green dataset with configurable component targeting |
| 9 — Prompt Optimisation | `src/prompt_optim/optimise.py` | Meta-prompting pipeline: iteratively optimise per-class VLM prompts via Llama-1B, accumulate a hierarchical knowledge base, evaluate generalisation on held-out datasets |

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

### Layer Explorer (VLM module browser)

A Streamlit app for inspecting any HuggingFace model's layer tree and saving pre/post-projector
layer paths for use by the linear probe. Loads the model on the `"meta"` device — weights are
shape-only tensors, so only ~200 MB RAM is needed and no GPU is required.

```bash
streamlit run app/layer_explorer.py
```

**Workflow:**
1. Enter a HF model ID (e.g. `google/gemma-4-26B-A4B-it`) and click **Load**.
2. Filter the module table by name or type; toggle between leaf-only and full tree view.
3. Select a **Pre-projector** layer (vision encoder output) and **Post-projector** layer
   (after the vision→LM projection) from the dropdowns. Each selection shows module type,
   input shape, output shape, and parameter count derived from static weight attributes.
4. Click **Save** — selections are written to `src/config/layer_selections.yaml` and
   automatically used the next time `probe.py` runs for that model ID.

**Shape info** is read from static weight attributes (`nn.Linear` → `in_features/out_features`,
`nn.Conv2d` → `in/out channels`, `nn.Embedding` → vocab/dim, etc.) without any forward pass.
For composite blocks (transformer layers, attention) that have no weights of their own, only
parameter counts are shown.

**Config format** (`src/config/layer_selections.yaml`):

```yaml
google/gemma-4-26B-A4B-it:
  pre: model.model.vision_tower
  post: model.model.embed_vision
Qwen/Qwen3.6-27B:
  pre: model.model.visual.patch_embed
  post: model.model.visual.merger
```

If no entry exists for a given model, `probe.py` falls back to its built-in heuristics
(`vision_tower` attribute for pre, and a list of known projector module names for post)
and prints a tip to run the layer explorer.

---

### Stage 2 — Linear probe

Evaluates frozen-embedding separability across one or more backbones. Models are loaded
and unloaded sequentially to stay within GPU memory.

**Default mode** runs LR per-dataset + LR combined for every backbone. Add `--all_experiments`
to also run FC probes.

| Experiment | Probe | Scope | When |
|------------|-------|-------|------|
| LR per-dataset | Logistic regression | One probe per dataset | Always |
| LR combined | Logistic regression | All datasets merged into one label space | Always |
| FC per-dataset | Single `nn.Linear` + Adam | One probe per dataset | `--all_experiments` |
| FC combined | Single `nn.Linear` + Adam | All datasets merged into one label space | `--all_experiments` |
| Post-proj LR combined | LR on post-projector embeddings | All datasets merged | VLMs only, always |

```bash
python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/ \
  --models     google/gemma-4-26B-A4B-it \
  --n_samples  300
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | none | One or more HuggingFace VLM model IDs |
| `--dino_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | DINOv3 backbone. Pass `none` to skip |
| `--yolo_model` | `yolo11x-cls.pt` | YOLO classification backbone. Pass `none` to skip |
| `--bioclip_model` | `imageomics/bioclip` | BioCLIP backbone (ViT-B/16 trained on biological images). Pass `none` to skip |
| `--n_samples` | all | Max images per dataset (stratified subsample) |
| `--quantize` | none | `4bit` or `8bit` (BitsAndBytes) for large VLMs |
| `--all_experiments` | off | Also run FC per-dataset and FC combined probes |
| `--postproj_only` | off | For VLMs: extract only post-projector embeddings, skip pre-projector pass |
| `--replot` | off | Reload `linear_probe_results.json` and regenerate all plots without re-running inference |
| `--fc_epochs` | `100` | Training epochs for FC probe |
| `--fc_lr` | `1e-3` | Adam learning rate for FC probe |
| `--fc_batch_size` | `256` | Batch size for FC probe training |
| `--dataset_source` | `agml` | `agml` or `digital_green` |
| `--images_dir` | none | Local image directory (required for `digital_green`) |

**Layer selection for VLMs** — if `src/config/layer_selections.yaml` has an entry for the model
ID (written by the Layer Explorer), the probe hooks exactly those modules. Otherwise it falls
back to built-in heuristics (`vision_tower` for pre, known projector names for post).

**Intermediate saves** — after each backbone completes the LR combined phase, partial metrics are
written to `linear_probe_results_intermediate.json`. This allows `--replot` to work even if the
full run is interrupted later.

**Output structure:**

```
outputs/linear_probe/
├── linear_probe_results.json             all probe results (full run)
├── linear_probe_results_intermediate.json  partial results saved after each backbone
├── feature_dimensions.txt               pre/post-projector dim per VLM backbone
├── summary_overall.png                  LR per-dataset F1 + accuracy across backbones
├── summary_by_task.png                  LR per-dataset F1 by task type across backbones
├── projector_before_after.png           pre vs post-projector LR F1 per VLM (with ratio)
├── class_distribution_long_tail.png     sample counts per class across all datasets
├── class_accuracy_vs_frequency/
│   ├── class_acc_vs_freq_{backbone}.png   per-backbone bar chart: per-class F1 vs count rank
│   └── class_acc_vs_freq_summary.png      averaged over all backbones
├── tsne/
│   └── tsne_combined_{backbone}_lr.png   t-SNE of combined LR embeddings (no legend — see below)
├── tsne_legends/
│   └── legend_combined_{backbone}_lr.png  colour-class legend for the corresponding t-SNE
├── confusion_matrices/
│   └── confusion_matrix_combined_{backbone}_lr.png
├── per_class_f1/
│   └── per_class_f1_combined_lr.png
└── loss_curves/                           FC probe loss curves (only with --all_experiments)
```

`--all_experiments` additionally produces per-dataset confusion matrices, t-SNE plots,
per-class F1 charts, FC loss curves, and a `probe_comparison_all.png` master chart.

Loss curves: **train loss** solid, **val loss** dashed vs. epoch. Val loss diverging early → lower
`--fc_lr`. Both still dropping at the last epoch → increase `--fc_epochs`.

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

### Stage 9 — Prompt Optimisation

Meta-prompting pipeline that iteratively improves per-class classification prompts for a VLM.
Uses `meta-llama/Llama-3.2-1B-Instruct` as the optimizer and any VLM (Gemma-4, Qwen, etc.)
as the classifier. Both models are loaded simultaneously.

**Train loop** (14 datasets, all classes):
- Llama generates 5 candidate prompts per iteration (10 iterations per class) using: crop/class info + consolidated knowledge base + best prompt so far
- VLM scores each candidate via per-class F1 (substring matching)
- After each class: Llama writes a per-class summary, then consolidates all summaries into an updated global knowledge base
- Closed-form prompts (with `{class_list}`) used throughout train for a clean F1 signal

**Test evaluation** (14 held-out datasets, all classes — no loop):
- Llama reads the final consolidated knowledge base + class info → generates one prompt per class
- VLM compares that prompt against the open-ended base prompt (no class list)
- Measures how well the accumulated knowledge generalises to unseen datasets and classes

```bash
uv run python main.py prompt-optim \
  --model      google/gemma-4-26B-A4B-it \
  --config     src/config/datasets.yaml \
  --output_dir outputs/prompt_optim/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3.6-27B` | VLM used for classification evaluation |
| `--config` | `src/config/datasets.yaml` | AgML datasets config |
| `--optim_config` | `src/config/prompt_optim.yaml` | Prompt optimisation hyperparameters and prompts |
| `--n_samples_per_class` | `30` | Images sampled per class for optimisation |
| `--n_eval_per_candidate` | `15` | Images used to score each candidate prompt |
| `--output_dir` | `outputs/prompt_optim/` | Output directory |
| `--probe_dir` | `outputs/linear_probe/` | Linear probe output dir — used to find t-SNE coords for overlay plot |
| `--split` | `both` | `train`, `test`, or `both` |
| `--force` | off | Re-run even if checkpoint shows a class is done |
| `--skip_plots` | off | Skip plot generation |
| `--seed` | `42` | Random seed for image sampling |

**Train/test split** is defined in `src/config/dataset_split.yaml` (14 train / 14 test datasets,
stratified by task type: disease, pest_damage, weed_identification).

**Output structure:**

```
outputs/prompt_optim/
├── run_log.jsonl                      one record per candidate evaluated (prompt, F1, VLM responses, Llama raw output)
├── test_log.jsonl                     one record per test class (base prompt vs summary-informed prompt)
├── best_prompts.json                  best prompt per train class at end of run
├── per_class_summaries.jsonl          Llama-written summary after each train class completes
├── consolidated_summary_history.jsonl consolidated knowledge base snapshot after each class
├── checkpoint.json                    resume state (re-run resumes from last completed class)
├── results.json                       final F1: base vs optimised, per class / dataset / task / overall
└── plots/
    ├── f1_overall.png                 overall / train / test grouped bar chart
    ├── f1_by_task.png                 F1 by task type (disease, pest_damage, weed_identification)
    ├── f1_by_dataset.png              F1 by dataset (mean across classes)
    ├── f1_per_class.png               F1 per class (most granular)
    └── tsne_overlay.png               linear probe t-SNE with ■ base and ★ optimised centroids
```

The t-SNE overlay reuses coordinates saved by `probe.py` — run `probe` at least once before
`prompt-optim` to generate the `tsne/*_coords.npz` files. The overlay is skipped with a
clear message if the file is not found.

---

### Stage 10 — Vocab Grounding

Tests whether injecting class-discriminative tokens from the VLM's own embedding space into
classification prompts improves F1. Hypothesis: diagnostic vocabulary (`disease`, `pest`,
`stress`) is weakly grounded in visual space; tokens whose embeddings are geometrically close
to a disease class's image representations should anchor attention to the right visual features.

**Phase 1 — Grounding analysis** (HuggingFace transformers, `output_hidden_states=True`):
- For each class: forward N images, compute mean image-token hidden-state centroid at final layer
- Search full vocab embedding table for tokens with highest `sim_to_target − mean_sim_to_others`
- Saves `discriminative_vocab.json`

**Phase 2 — Evaluation** (vLLM, fast batch):
- Base prompt: `visual_classification_disease` from `src/config/prompts.yaml` (open-ended, no class list)
- Contrastive prompt: discriminative tokens prepended bare → `"pustule, lesion. What disease... in this image of a {crop}?"`
- Measures F1 gain per class across all evaluated datasets

```bash
# Full run (test split)
uv run python main.py vocab-grounding \
  --model google/gemma-4-26B-A4B-it \
  --split test \
  --output_dir outputs/vocab_grounding

# Re-run evaluation only (Phase 1 already done)
uv run python main.py vocab-grounding \
  --skip_grounding \
  --output_dir outputs/vocab_grounding

# Replot from existing results (no inference)
uv run python main.py vocab-grounding \
  --skip_grounding --skip_eval \
  --output_dir outputs/vocab_grounding
```

**Output structure:**

```
outputs/vocab_grounding/
├── discriminative_vocab.json          top discriminative tokens per class + scores
├── results.json                       F1 base vs contrastive, per class / dataset / task / overall
├── class_log.jsonl                    per-class evaluation records with raw responses
├── checkpoint.json                    resume state
└── plots/
    ├── f1_overall.png                 overall F1 grouped bar chart
    ├── f1_by_task.png                 F1 by task type
    ├── f1_by_dataset.png              F1 by dataset
    ├── f1_per_class.png               F1 per class
    ├── token_grounding_{dataset}.png  discriminability heatmap: tokens × classes
    ├── discriminability_vs_gain.png   scatter: does disc score predict F1 gain?
    ├── gain_distribution.png          histogram of F1 gains across classes
    └── threshold_sensitivity.png      coverage vs min_gap threshold
```

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
│   ├── linear_probe_results_intermediate.json
│   ├── feature_dimensions.txt
│   ├── summary_overall.png / summary_by_task.png / projector_before_after.png
│   ├── class_distribution_long_tail.png
│   ├── class_accuracy_vs_frequency/
│   ├── tsne/ + tsne_legends/
│   └── confusion_matrices/ + per_class_f1/ + loss_curves/
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
├── prompt_optim/
│   ├── run_log.jsonl / test_log.jsonl
│   ├── best_prompts.json
│   ├── per_class_summaries.jsonl / consolidated_summary_history.jsonl
│   ├── results.json
│   └── plots/
├── results_table.csv                 collated metrics across all stages
├── results_table.tex                 LaTeX booktabs table (ready to \input into a paper)
└── error_overlap/
    ├── error_overlap_<model>.csv     per-class quadrant table
    ├── error_overlap_<model>.png     scatter plot (lang acc vs vision acc)
    ├── error_overlap_heatmap.png     quadrant fraction comparison across models
    └── error_overlap_summary.json    aggregate quadrant counts per model
```

All prompt templates live in `src/config/prompts.yaml`. Edit there — no code changes needed.
