# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding. Three controlled experiments benchmark VLM performance across visual, linguistic, and representational dimensions.

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 — Linear Probe | `src/linear_probe/probe.py` | Frozen-embedding separability across backbones (LR + optional FC, per-dataset + combined) |
| 2 — Language Assessment | `src/language_assessment/assess.py` | Text-only agricultural VQA (ROUGE-SS + F1-macro) — also accepts VLMs in text-only mode |
| language-plot | `src/language_assessment/plot.py` | Regenerate language assessment charts without re-running inference |
| 3 — Vision Assessment | `src/vision_assessment/assess.py` | Multimodal VQA with image input (ROUGE-SS + F1-macro) |
| vision-plot | `src/vision_assessment/plot.py` | Regenerate vision assessment charts without re-running inference |
| 4 — Collate | `src/collate_results.py` | Merge all stage metrics into `results_table.csv` + `results_table.tex` |

All stages share a single metrics module (`src/common/metrics.py`): same `resolve_class`, `compute_classification_metrics`, `apply_judge`, and image sampling (`sample_class_images`, seed 42) across every experiment.

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

### Stage 1 — Linear Probe

Evaluates frozen-embedding separability across one or more backbones. Models are loaded and unloaded sequentially.

**Default mode** runs LR per-dataset + LR combined. Add `--all_experiments` to also run FC probes.

| Experiment | Probe | Scope | When |
|------------|-------|-------|------|
| LR per-dataset | Logistic regression | One probe per dataset | Always |
| LR combined | Logistic regression | All datasets merged | Always |
| FC per-dataset | Single `nn.Linear` + Adam | One probe per dataset | `--all_experiments` |
| FC combined | Single `nn.Linear` + Adam | All datasets merged | `--all_experiments` |
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
| `--bioclip_model` | `imageomics/bioclip` | BioCLIP backbone. Pass `none` to skip |
| `--n_samples` | all | Max images **per class** (balanced, seed 42) |
| `--quantize` | none | `4bit` or `8bit` (BitsAndBytes) for large VLMs |
| `--all_experiments` | off | Also run FC per-dataset and FC combined probes |
| `--postproj_only` | off | For VLMs: extract only post-projector embeddings |
| `--replot` | off | Reload saved results and regenerate all plots without re-running inference |

**Layer selection for VLMs** — if `src/config/layer_selections.yaml` has an entry for the model ID (written by the Layer Explorer), the probe hooks exactly those modules. Otherwise it falls back to built-in heuristics.

**Output structure:**

```
outputs/linear_probe/
├── linear_probe_results.json
├── linear_probe_results_intermediate.json
├── feature_dimensions.txt
├── summary_overall.png / summary_by_task.png / projector_before_after.png
├── class_distribution_long_tail.png
├── class_accuracy_vs_frequency/
├── tsne/ + tsne_legends/
└── confusion_matrices/ + per_class_f1/ + loss_curves/
```

---

### Layer Explorer (VLM module browser)

Streamlit app for inspecting any HuggingFace model's layer tree and saving pre/post-projector layer paths. Loads on the `"meta"` device — no GPU required.

```bash
streamlit run app/layer_explorer.py
```

Selections are written to `src/config/layer_selections.yaml` and automatically picked up by `probe.py`.

---

### Stage 2 — Language Assessment

Evaluates models on text-only agricultural VQA. VLMs can also be evaluated here in text-only mode. Each model runs in an isolated subprocess — GPU memory is fully released between models.

```bash
python main.py language \
  --descriptions_jsonl   outputs/data_generation/agml_descriptions.jsonl \
  --classification_jsonl outputs/data_generation/agml_classification.jsonl \
  --config               src/config/datasets.yaml \
  --prompts              src/config/prompts.yaml \
  --models               google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --multimodal_models    google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \
  --output_dir           outputs/language_assessment/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of model IDs |
| `--multimodal_models` | none | Subset of `--models` that are VLMs (shown with hatching in charts) |
| `--n_samples` | all | Limit total records per JSONL (caps number of classes evaluated) |
| `--judge_model` | none | LLM-as-a-judge model for classification |
| `--enforce_eager` | off | Disable CUDA graph capture |
| `--dataset_source` | `agml` | `agml` or `digital_green` |

#### Regenerate charts without re-running inference

```bash
python main.py language-plot \
  --output_dir        outputs/language_assessment/ \
  --config            src/config/datasets.yaml \
  --multimodal_models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B
```

**Output structure:**

```
outputs/language_assessment/
├── desc_<model>.csv / cls_<model>.csv
├── language_assessment_metrics.json
├── summary.txt
└── *.png  (5 chart files)
```

---

### Stage 3 — Vision Assessment

Evaluates multimodal models on image + text VQA. Images are sampled with a fixed seed (42) per class so results are reproducible and directly comparable across models and runs.

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
| `--n_samples` | all | Max images **per class** (seeded sample, consistent across runs) |
| `--judge_model` | none | LLM-as-a-judge model for classification |
| `--enforce_eager` | off | Disable CUDA graph capture |
| `--dataset_source` | `agml` | `agml` or `digital_green` |

#### Regenerate charts without re-running inference

```bash
python main.py vision-plot \
  --output_dir outputs/vision_assessment/ \
  --config     src/config/datasets.yaml
```

**Output structure:**

```
outputs/vision_assessment/
├── desc_<model>.csv / cls_<model>.csv
├── vision_assessment_metrics.json
├── summary.txt
└── *.png  (5 chart files)
```

---

---

### Stage 4 — Collate results

Merges outputs from all stages into `results_table.csv` and `results_table.tex`.

```bash
python main.py collate \
  --linear_probe_dir outputs/linear_probe/ \
  --language_dir     outputs/language_assessment/ \
  --vision_dir       outputs/vision_assessment/ \
  --output_dir       outputs/
```

---

### Results Viewer (Streamlit)

Browse per-class outputs interactively. Two tabs:

- **Summary** — collated `results_table.csv` with gradient colouring
- **Class Explorer** — per (class, model): dataset-level F1 (non-judged + judged), per-class accuracy, ROUGE-SS, and per-image classification + description responses with correct/incorrect flags for both language and vision assessment

```bash
streamlit run app/results_viewer.py -- \
  --language_dir outputs/language_assessment/ \
  --vision_dir   outputs/vision_assessment/ \
  --results_csv  outputs/results_table.csv
```

---

## Metric consistency

All three experiments share the same evaluation code in `src/common/metrics.py`:

| Concern | Implementation |
|---------|---------------|
| Class resolution | `resolve_class` — exact match then single-match substring fallback |
| Classification metrics | `compute_classification_metrics` — accuracy, contains_accuracy, F1-macro, F1-weighted |
| ROUGE | `rouge_scores_per_sample` — rouge1, rouge2, rougeL, rouge_ss (WordNet synonym expansion) |
| LLM-as-judge | `apply_judge` — JSON `{is_match, confidence}` parsed from model output |
| Think-block stripping | `THINK_RE` — covers both `<think>` and `<thought>` (Gemma-4) |
| Image sampling | `sample_class_images(names, n, seed=42)` — seeded shuffle, consistent across runs |

All prompt templates live in `src/config/prompts.yaml`. Edit there — no code changes needed.
