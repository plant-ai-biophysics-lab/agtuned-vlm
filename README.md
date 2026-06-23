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
| 4 — Aug Vision | `src/desc_augmented_vision.py` | Test whether language descriptions of judged-correct classes improve vision classification |
| aug-vision-plot | `src/desc_augmented_vision_plot.py` | Regenerate aug-vision charts from saved CSVs |
| 5 — Collate | `src/collate_results.py` | Merge all stage metrics into `results_table.csv` + `results_table.tex` |

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

### Stage 0 — Data Generation (pre-built, committed to repo)

The JSONL files required by language and vision assessment are committed to `dataset/` and ready to use:

| File | Used by |
|------|---------|
| `dataset/agml_descriptions.jsonl` | `language` and `vision` stages (`--descriptions_jsonl`) |
| `dataset/agml_classification.jsonl` | `language` stage (`--classification_jsonl`) |

If you need to regenerate them (e.g. for a new dataset or updated prompts), use `src/legacy/data_generation/generate.py`:

```bash
uv run python src/legacy/data_generation/generate.py \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --model      google/gemma-4-26B-A4B-it \
  --output_dir outputs/data_generation/
```

Add `--force` to regenerate all classes from scratch. Progress is checkpointed so re-runs skip completed classes.

---

### Stage 1 — Linear Probe

Evaluates frozen-embedding separability across one or more backbones. Models are loaded and unloaded sequentially. Writes `test_split.json` immediately after dataset loading — this file is used by Stages 3 and 4 to evaluate the exact same images.

**Default mode** runs FC combined only (GPU-accelerated, 10 train images per class). Add flags to enable additional experiments.

| Experiment | Probe | Scope | When |
|------------|-------|-------|------|
| FC combined | `nn.Linear` + Adam (GPU) | All datasets merged, 10/class train | Always (default) |
| LR combined | Logistic regression (CPU) | All datasets merged, 10/class train | `--lr_combined` |
| LR per-dataset | Logistic regression | One probe per dataset, 10/class train | `--per_dataset` |
| FC per-dataset | `nn.Linear` + Adam (GPU) | One probe per dataset | `--all_experiments` |
| Post-proj FC/LR combined | Post-projector embeddings | All datasets merged | VLMs only, always |

```bash
# Default: FC combined, 10 train images per class
python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/ \
  --models     google/gemma-4-26B-A4B-it \
  --cache_embeddings
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | none | One or more HuggingFace VLM model IDs |
| `--cache_embeddings` | off | Save extracted embeddings to `.npz` files (required for `cluster` stage) |
| `--dino_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | DINOv3 backbone. Pass `none` to skip |
| `--yolo_model` | `yolo11x-cls.pt` | YOLO classification backbone. Pass `none` to skip |
| `--bioclip_model` | `imageomics/bioclip` | BioCLIP backbone. Pass `none` to skip |
| `--n_samples` | 10 | Max images **per class for training** in per-dataset split (seed 42); rest goes to test. Ignored by combined probe which always uses 80/20 stratified split. |
| `--quantize` | none | `4bit` or `8bit` (BitsAndBytes) for large VLMs |
| `--per_dataset` | off | Also run LR probe per-dataset (one probe per dataset) |
| `--lr_combined` | off | Also run LR probe on combined embeddings (CPU-bound, slower than FC) |
| `--all_experiments` | off | Also run FC per-dataset probes |
| `--fc_epochs` | 100 | Training epochs for FC probe |
| `--fc_lr` | 1e-3 | Adam learning rate for FC probe |
| `--fc_batch_size` | 256 | Batch size for FC probe |
| `--postproj_only` | off | For VLMs: extract only post-projector embeddings |
| `--replot` | off | Reload saved results and regenerate all plots without re-running inference |

**Layer selection for VLMs** — if `src/config/layer_selections.yaml` has an entry for the model ID (written by the Layer Explorer), the probe hooks exactly those modules. Otherwise it falls back to built-in heuristics.

**Output structure:**

```
outputs/linear_probe/
├── test_split.json                        ← used by vision & aug-vision stages
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

### Stage 1b — Clustering Evaluation

Measures whether frozen visual features naturally cluster by class **without using labels for training** — a stricter, label-free complement to the linear probe. The linear probe can compensate for weak features by learning a good hyperplane; clustering cannot. Labels are only used for scoring after clustering is complete.

Requires `--cache_embeddings` to have been set on the probe run. Reads existing `.npz` files — no GPU needed, runs on CPU.

```bash
python main.py cluster \
  --cache_dir  outputs/linear_probe/embedding_cache/ \
  --output_dir outputs/clustering_eval/
```

Add `--max_samples 1000` for a quick local sanity check (subsamples each dataset before clustering). Run without it on the cluster for full results.

| Metric | Measures | Range | Interpretation |
|--------|----------|-------|----------------|
| **NMI** | How much cluster assignments share information with true classes | 0 → 1 | > 0.5 is strong |
| **ARI** | Cluster-to-class agreement, chance-corrected | −1 → 1 | 0 = random; > 0.5 is strong; negative = worse than random |
| **Silhouette** | Intra-cluster compactness vs inter-cluster separation (no labels used) | −1 → 1 | > 0.3 is strong; **negative means images are closer to a different cluster than their own** — the encoder has no meaningful class grouping |

All three metrics are computed on PCA-50 reduced features. Bigger is better for all three.

**Output structure:**

```
outputs/clustering_eval/
├── clustering_metrics.json
├── clustering_overall.png               ← mean NMI/ARI/Sil per backbone
├── clustering_per_task.png              ← per task type
├── clustering_heatmap_nmi.png           ← backbone × dataset heatmap
├── clustering_heatmap_ari.png
├── clustering_heatmap_silhouette.png
└── clustering_scatter_nmi_vs_sil.png    ← NMI vs Silhouette per (backbone, dataset)
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
  --models            google/gemma-4-26B-A4B-it \
  --multimodal_models google/gemma-4-26B-A4B-it \
  --output_dir        outputs/language_assessment/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of model IDs |
| `--multimodal_models` | none | Subset of `--models` that are VLMs (shown with hatching in charts) |
| `--descriptions_jsonl` | `dataset/agml_descriptions.jsonl` | Pre-built descriptions JSONL |
| `--classification_jsonl` | `dataset/agml_classification.jsonl` | Pre-built classification JSONL |
| `--n_samples` | all | Limit total records per JSONL (caps number of classes evaluated) |
| `--judge_model` | first model in `--models` | LLM-as-a-judge model for classification |
| `--skip_existing` | off | Skip inference + judging for models whose CSVs already exist; load from disk instead (use when adding a new model to an existing run) |
| `--enforce_eager` | off | Disable CUDA graph capture |

#### Regenerate charts without re-running inference

```bash
python main.py language-plot \
  --output_dir        outputs/language_assessment/ \
  --multimodal_models google/gemma-4-26B-A4B-it
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

Evaluates multimodal models on image + text VQA. Uses the same test images as the linear probe (`test_split.json`) so results are directly comparable. Images are sampled with a fixed seed (42) per class when no test split is provided.

```bash
python main.py vision \
  --models     google/gemma-4-26B-A4B-it \
  --output_dir outputs/vision_assessment/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of multimodal model IDs |
| `--descriptions_jsonl` | `dataset/agml_descriptions.jsonl` | Pre-built descriptions JSONL |
| `--test_split` | `outputs/linear_probe/test_split.json` | Image split from linear probe (ensures same test images) |
| `--n_samples` | all | Max images **per class** (used only if no test split) |
| `--judge_model` | first model in `--models` | LLM-as-a-judge model for classification |
| `--skip_existing` | off | Skip inference + judging for models whose CSVs already exist; load from disk instead |
| `--enforce_eager` | off | Disable CUDA graph capture |

#### Regenerate charts without re-running inference

```bash
python main.py vision-plot \
  --output_dir outputs/vision_assessment/
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

### Stage 4 — Description-Augmented Vision

Tests whether prepending a language description of a class helps the model classify that class from an image. Supports two modes run separately:

- **Filtered** (default): only classes where the language assessment produced a judged-correct answer
- **Unfiltered** (`--all_classes`): all classes that have a description response

Flow: language cls/desc CSVs → select classes → load description responses → re-run vision inference with description prepended → compare baseline vs augmented accuracy.

All output files include `filtered` or `all` in their name so both runs coexist in the same `--output_dir`.

```bash
# Filtered run (judged-correct classes only)
python main.py aug-vision \
  --models     google/gemma-4-26B-A4B-it \
  --lang_dir   outputs/language_assessment/ \
  --vis_dir    outputs/vision_assessment/ \
  --output_dir outputs/desc_augmented_vision/

# Unfiltered run (all classes with descriptions)
python main.py aug-vision \
  --models      google/gemma-4-26B-A4B-it \
  --lang_dir    outputs/language_assessment/ \
  --vis_dir     outputs/vision_assessment/ \
  --output_dir  outputs/desc_augmented_vision/ \
  --all_classes

# Sub-step: use vision-generated descriptions (per-image alignment) + correlation plot
# Run AFTER the filtered/all aug-vision runs above are complete.
# --resume skips re-running the ground-truth aug-vision inference (loads existing CSV).
# Only the vision-description sub-step runs fresh and outputs the correlation plot.
python main.py aug-vision \
  --models                  google/gemma-4-26B-A4B-it \
  --lang_dir                outputs/language_assessment/ \
  --vis_dir                 outputs/vision_assessment/ \
  --output_dir              outputs/desc_augmented_vision/ \
  --use_vision_descriptions \
  --resume
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | Space-separated list of multimodal model IDs |
| `--lang_dir` | `outputs/language_assessment/` | Language assessment output directory |
| `--vis_dir` | `outputs/vision_assessment/` | Vision assessment output directory (for baseline rows) |
| `--test_split` | `outputs/linear_probe/test_split.json` | Same test split used in probe and vision assessment |
| `--all_classes` | off | Use all classes with descriptions instead of only judged-correct ones |
| `--use_vision_descriptions` | off | Sub-step: inject per-image vision-generated descriptions (from Stage 3 desc CSV) instead of reference descriptions; produces `vis_desc_aug_cls_<model>.csv` and a correlation scatter plot (F1 vs rouge_ss / sem_sim) |
| `--judge_model` | first model in `--models` | LLM-as-a-judge for augmented predictions |
| `--resume` | off | Skip inference/judging if output CSVs already exist |
| `--enforce_eager` | off | Disable CUDA graph capture |

#### Regenerate charts without re-running inference

```bash
python main.py aug-vision-plot \
  --models     google/gemma-4-26B-A4B-it \
  --vis_dir    outputs/vision_assessment/ \
  --output_dir outputs/desc_augmented_vision/
```

**Output structure:**

```
outputs/desc_augmented_vision/
├── aug_cls_filtered_<model>.csv / baseline_cls_filtered_<model>.csv
├── aug_cls_all_<model>.csv      / baseline_cls_all_<model>.csv
├── desc_augmented_metrics_filtered.json
├── desc_augmented_metrics_all.json
├── aug_cls_filtered_<model>.csv / baseline_cls_filtered_<model>.csv
├── aug_cls_all_<model>.csv      / baseline_cls_all_<model>.csv
├── vis_desc_aug_cls_<model>.csv             ← per-image vision-desc sub-step output
├── desc_augmented_metrics_filtered.json
├── desc_augmented_metrics_all.json
├── cls_comparison_per_dataset_filtered.png / cls_comparison_per_dataset_all.png
├── cls_comparison_per_task_filtered.png    / cls_comparison_per_task_all.png
├── cls_overall_comparison_filtered.png     / cls_overall_comparison_all.png
├── transitions_filtered_<model>.png        / transitions_all_<model>.png
└── desc_quality_correlation_vis_desc_<model>.png  ← F1 vs rouge_ss / sem_sim scatter
```

---

### Stage 5 — Collate results

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

All experiments share the same evaluation code in `src/common/metrics.py`:

| Concern | Implementation |
|---------|---------------|
| Class resolution | `resolve_class` — exact match then single-match substring fallback |
| Classification metrics | `compute_classification_metrics` — accuracy, contains_accuracy, F1-macro, F1-weighted |
| ROUGE + sem_sim | `rouge_scores_per_sample` — rouge1, rouge2, rougeL, rouge_ss (WordNet synonym expansion), sem_sim (cosine similarity via `sentence-transformers/all-MiniLM-L6-v2`, loaded lazily) |
| LLM-as-judge | `apply_judge` — JSON `{is_match, confidence}` parsed from model output |
| Think-block stripping | `THINK_RE` — covers both `<think>` and `<thought>` (Gemma-4) |
| Image sampling | `sample_class_images(names, n, seed=42)` — seeded shuffle, consistent across runs |

All prompt templates live in `src/config/prompts.yaml`. Edit there — no code changes needed.
