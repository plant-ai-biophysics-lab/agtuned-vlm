# AgML Scraper: Synthetic VQA Dataset Generation

## Overview

This experiment generates two complementary synthetic datasets from AgML image classification
benchmarks (see [arxiv:2512.15977](https://arxiv.org/pdf/2512.15977)):

- **Description dataset** (`agml_descriptions.jsonl`) — asks the model to visually describe a
  class; ground truth is a synthesized multi-image description.
- **Classification dataset** (`agml_classification.jsonl`) — asks the model to identify the
  class; ground truth is the class label. Derived from the same inference run as the description
  dataset at no extra cost.

Both JSONL files are directly loadable by `text_qa.py --idtools_jsonl` for H2 evaluation.

---

## Dataset A: Visual Description

### Schema

**Disease / pest / stress:**
```
Visual (stored in HF dataset, used during generation):
  image: <image>
  question: "What disease, pest, damage type, or other stress, if any, is exhibited
             in this image of a {crop}?"  [used to generate per-image captions]
  description question: "Can you describe visually the {class} that is exhibited
                         in this image of {crop}?"
  answer: <synthesized visual description>

Text-only (stored in JSONL, used in text_qa.py):
  question: "Can you describe visually the {class} of a {crop}?"
  answer: <synthesized visual description>
```

**Weed / crop identification:**
```
Visual:
  image: <image>
  description question: "Can you describe visually the {species} exhibited in this image?"
  answer: <synthesized visual description>

Text-only:
  question: "Can you describe visually the {species}?"
  answer: <synthesized visual description>
```

### Caption Aggregation (two-pass)

**Pass 1 — Per-image captions (vision pass)**
The VLM receives each image individually with the description question. This produces
`n_samples_per_class` independent raw captions (default: 10, set via `--n_samples_per_class`).
Each caption is saved to the checkpoint immediately so this pass is resumable.

**Pass 2 — Meta-synthesis (text-only pass)**
All N raw captions are fed back to the model in one text-only prompt:

```
You are given {N} visual descriptions of {class_label} on {crop}, each written from a
separate photograph. Synthesize these into one clear, representative 2–3 sentence description
that captures the consensus visual features. Ignore noise or one-off observations.

Descriptions:
1. {caption_1}
2. {caption_2}
...
```

The synthesized output becomes the `answer` for this class entry.

**Rationale**: Drawn from ShareGPT4V and LLaVA-style dataset construction — dense per-image
captions are refined into one coherent ground truth. Synthesizing across N images reduces
hallucination from any single frame, captures intra-class visual variance, and keeps outputs
interpretable (individual captions are preserved in metadata).

---

## Dataset B: Visual Classification

Derived entirely from Dataset A inference results — no additional model calls required.

### Schema

**Disease / pest / stress:**
```
Visual (stored in HF dataset):
  image: <image>  [each of the N sampled images becomes its own record]
  question: "What disease, pest, damage type, or other stress, if any, is exhibited
             in this image of a {crop}?"
  answer: {class_label}

Text QA (stored in JSONL, used in text_qa.py):
  question: "What disease, pest, damage type, or other stress, if any, is exhibited
             in this description of a {crop}?

             Description: {synthesized_description}"
  answer: {class_label}
```

**Weed / crop identification:**
```
Visual (stored in HF dataset):
  image: <image>
  question: "What plant or weed is exhibited in this image?"
  answer: {species}

Text QA (stored in JSONL, used in text_qa.py):
  question: "What plant or weed is exhibited in this description?

             Description: {synthesized_description}"
  answer: {species}
```

---

## Question Templates by `task_type`

| task_type | Description visual Q | Description text Q | Classification visual Q | Classification text Q |
|-----------|---------------------|--------------------|------------------------|-----------------------|
| `disease` | "...describe visually the `{class}` that is exhibited in this image of `{crop}`?" | "...describe visually the `{class}` of a `{crop}`?" | "What disease... exhibited in this image of a `{crop}`?" | "...exhibited in this description of a `{crop}`?" |
| `pest` | same as disease | same as disease | same as disease | same as disease |
| `stress` | same as disease | same as disease | same as disease | same as disease |
| `weed_identification` | "...describe visually the `{species}` exhibited in this image?" | "...describe visually the `{species}`?" | "What plant or weed is exhibited in this image?" | "...exhibited in this description?" |
| `crop_identification` | same as weed | same as weed | same as weed | same as weed |

---

## Dataset Configuration (`datasets.yaml`)

All datasets and their metadata are declared in `datasets.yaml`. This is the single file to
edit when adding datasets, changing task types, or skipping entries.

```yaml
datasets:
  - agml_name: plant_village_classification
    crop: multiple
    task_type: disease
    n_images: 55448
    classes: [Apple___Apple_scab, Apple___Black_rot, ...]
    skip: false
    notes: "Multi-crop; class label encodes crop and condition separated by ___"

  - agml_name: soybean_insect_classification
    crop: soybean
    task_type: pest
    n_images: 6410
    classes: [Caterpillar, Diabrotica_speciosa, Healthy]
    skip: false

  - agml_name: rangeland_weeds_australia
    crop: null
    task_type: weed_identification
    n_images: 17509
    classes: [chinee_apple, lantana, parkinsonia, ...]
    skip: false

  - agml_name: leaf_counting_denmark
    crop: null
    task_type: null
    skip: true
    notes: "Leaf count regression task — not applicable"
```

The full registry is in `datasets.yaml` (same directory). All 29 AgML image classification
datasets are listed with inferred `task_type`, `crop`, classes, and `skip` flags. Two are
skipped by default:
- `leaf_counting_denmark` — leaf count regression, not applicable
- `riseholme_strawberry_classification_2021` — fruit ripeness, not applicable

---

## Output Format

### JSONL files (`text_qa.py` compatible)

Both JSONL files use the same messages schema. The description stored in
`messages[0].content` is the text-only question (no image reference), so both files load
directly via `text_qa.py --idtools_jsonl`.

**`agml_descriptions.jsonl`** — one record per class:
```json
{
  "messages": [
    {"role": "user",      "content": "Can you describe visually the Early Blight of a Tomato?"},
    {"role": "assistant", "content": "Early Blight on tomato presents as dark brown circular lesions..."}
  ],
  "metadata": {
    "record_type": "description",
    "agml_dataset": "plant_village_classification",
    "crop": "Tomato",
    "class_label": "Early Blight",
    "task_type": "disease",
    "visual_question": "Can you describe visually the Early Blight that is exhibited in this image of Tomato?",
    "raw_captions": ["...", "..."],
    "n_images_used": 10,
    "generator_model": "Qwen/Qwen3.6-35B-A3B"
  }
}
```

**`agml_classification.jsonl`** — one record per class (description embedded in question):
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What disease, pest, damage type, or other stress, if any, is exhibited in this description of a Tomato?\n\nDescription: Early Blight on tomato presents as dark brown circular lesions..."
    },
    {"role": "assistant", "content": "Early Blight"}
  ],
  "metadata": {
    "record_type": "classification",
    "agml_dataset": "plant_village_classification",
    "crop": "Tomato",
    "class_label": "Early Blight",
    "task_type": "disease",
    "generator_model": "Qwen/Qwen3.6-35B-A3B"
  }
}
```

### HuggingFace Datasets

Two separate HF datasets, one per task type. Shared columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_paths` | list[string] | Paths to the N sampled images |
| `agml_dataset` | string | Source AgML dataset name |
| `crop` | string | Crop name (null for weed/crop ID tasks) |
| `class_label` | string | Cleaned class label |
| `task_type` | string | `disease` / `pest` / `stress` / `weed_identification` / `crop_identification` |
| `generator_model` | string | Model used for generation |
| `n_images_used` | int | Actual number of images sampled |

Description dataset adds: `visual_description_question`, `text_description_question`, `answer` (description)

Classification dataset adds: `visual_classification_question`, `text_classification_question`, `answer` (class label)

---

## Checkpointing

Generation state is written to `checkpoint.json` after every class. Only the description
pipeline runs inference; classification records are derived from the same data.

```json
{
  "plant_village_classification": {
    "Early Blight (Tomato)": {
      "status": "done",
      "synthesized_description": "Early Blight on tomato presents as...",
      "raw_captions": ["...", "..."],
      "image_paths": ["path/to/img1.jpg", "..."]
    },
    "Late Blight (Tomato)": {
      "status": "pending"
    }
  }
}
```

On re-run, completed entries are skipped. Pass `--force` to regenerate from scratch. Changing
`--model` and re-running will only regenerate pending entries unless `--force` is set.

---

## Model

Default: `Qwen/Qwen3.6-35B-A3B` via vLLM.

Loaded once with `LLM(model=..., dtype="bfloat16")`. Both vision pass (Pass 1) and synthesis
pass (Pass 2) use the same instance. The model ID is a CLI argument — any vLLM-compatible VLM
can be substituted without code changes.

---

## CLI

```bash
uv run python agml_scraper.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --datasets_config datasets.yaml \
  --n_samples_per_class 10 \
  --output_dir outputs/ \
  --hub_repo_descriptions your-hf-username/agml-descriptions \
  --hub_repo_classification your-hf-username/agml-classification \
  [--force]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3.6-35B-A3B` | vLLM-compatible VLM |
| `--datasets_config` | `datasets.yaml` | Path to dataset registry |
| `--n_samples_per_class` | `10` | Images sampled per class |
| `--output_dir` | `outputs/` | Directory for JSONL files, checkpoint, and logs |
| `--hub_repo_descriptions` | `None` | HF repo for description dataset (skipped if not set) |
| `--hub_repo_classification` | `None` | HF repo for classification dataset (skipped if not set) |
| `--force` | `False` | Ignore checkpoint and regenerate all entries |

---

## Pipeline Flow

```
datasets.yaml
     │
     ▼
For each dataset (skip: false):
  Load AgML dataset → get class list
  For each class:
    Sample n_samples_per_class images (random, seeded)
    ┌─ Pass 1: VLM describes each image individually ─────── vision pass
    │           (N raw captions saved to checkpoint)
    └─ Pass 2: LLM synthesizes N captions → one description ─ text-only pass

    → Description record  → agml_descriptions.jsonl
    → Classification record (reuses description + class label, no inference)
                          → agml_classification.jsonl
     │
     ▼
Push both JSONL files → two HuggingFace datasets (optional)
```

---

## Connecting to `text_qa.py`

Both output files are drop-in replacements for KisanVaani in the text QA evaluation pipeline.

```bash
# Generate both datasets
uv run python agml_scraper.py --output_dir outputs/

# Evaluate description task (open-ended generation, scored with ROUGE)
uv run python ../text_qa/text_qa.py \
  --idtools_jsonl outputs/agml_descriptions.jsonl \
  --model google/gemma-4-E2B-it \
  --output_dir results/agml_descriptions/

# Evaluate classification task (short answer, scored with exact/contains match)
uv run python ../text_qa/text_qa.py \
  --idtools_jsonl outputs/agml_classification.jsonl \
  --model google/gemma-4-E2B-it \
  --output_dir results/agml_classification/
```

**Description task** tests whether the model has learned what diseases/weeds look like
(visual knowledge in language weights) — same H2 hypothesis as KisanVaani but grounded in
AgML visual classes.

**Classification task** tests whether the model can correctly identify a class from a
description alone — a harder, more diagnostic probe of agricultural concept grounding.
