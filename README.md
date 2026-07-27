# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding.

## Prerequisites

```bash
uv sync
```

## Pipeline

Run stages in this order.

---

### Stage 0a — Download & sample datasets (Project-AgML on HuggingFace)

Image datasets are sourced from [huggingface.co/Project-AgML](https://huggingface.co/Project-AgML) (259 datasets, replacing the old `agml` pip package's implicit `~/.agml/datasets` auto-download). Only classification datasets are used — detection/segmentation datasets in the org are excluded.

To (re)build the dataset registry from the current HF catalog:

```bash
# One-time / when the Project-AgML org's dataset list changes:
uv run python scripts/agml_hf/catalog_datasets.py \
  --org Project-AgML \
  --output src/config/hf_catalog.json

# Regenerate src/config/datasets.yaml (writes datasets.yaml.new — review the
# diff and outputs/migration/datasets_migration_review.md before overwriting):
uv run python scripts/agml_hf/migrate_datasets_yaml.py \
  --catalog src/config/hf_catalog.json \
  --current src/config/datasets.yaml \
  --output  src/config/datasets.yaml.new
```

To download all active datasets and take a class-balanced subsample (fixed cap per class, default 50, adjustable via `--n_per_class`), recording exactly which images were chosen for reproducibility:

```bash
uv run python scripts/agml_hf/download_and_sample.py \
  --config src/config/datasets.yaml \
  --n_per_class 50 \
  --output dataset/selected_images.json
```

`dataset/selected_images.json` records the HF row index of every selected image per `(dataset, class)`. Re-running with the same `--seed`/`--n_per_class` reproduces the identical selection; pass `--from_manifest dataset/selected_images.json` to reuse an existing selection exactly (e.g. after re-cataloging) without re-sampling.

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

```bash
uv run python main.py probe \
  --config     src/config/datasets.yaml \
  --output_dir outputs/linear_probe/ \
  --models     google/gemma-4-26B-A4B-it \
  --cache_embeddings
```

---

### Stage 2 — Language Assessment

```bash
uv run python main.py language \
  --models            google/gemma-4-26B-A4B-it \
  --multimodal_models google/gemma-4-26B-A4B-it \
  --output_dir        outputs/language_assessment/
```

---

### Stage 3 — Vision Assessment

```bash
uv run python main.py vision \
  --models     google/gemma-4-26B-A4B-it \
  --output_dir outputs/vision_assessment/ \
  --skip_existing
```

---

### Stage 4 — Description-Augmented Vision

```bash
uv run python src/desc_augmented_vision.py \
  --models      google/gemma-4-26B-A4B-it \
  --vis_dir     outputs/vision_assessment/ \
  --output_dir  outputs/desc_augmented_vision/ \
  --descriptions_jsonl dataset/agml_descriptions.jsonl \
  --all_classes
```

#### Regenerate charts without re-running inference

```bash
uv run python main.py aug-vision-plot \
  --models     google/gemma-4-26B-A4B-it \
  --vis_dir    outputs/vision_assessment/ \
  --output_dir outputs/desc_augmented_vision/ \
  --all_classes
```

---

### Stage 4b — CoT Vision Assessment

```bash
uv run python main.py cot
```

---

### CoT Step Analysis

```bash
uv run python src/cot_step_analysis.py
```
