# AgML Scraper: Synthetic VQA Dataset Generation

## Overview

This experiment generates two complementary synthetic datasets from AgML image classification
benchmarks (see [arxiv:2512.15977](https://arxiv.org/pdf/2512.15977)):

- **Description dataset** (`agml_descriptions.jsonl`) — asks the model to visually describe a
  class; ground truth is a synthesized multi-image description.
- **Classification dataset** (`agml_classification.jsonl`) — asks the model to identify the
  class from a description; ground truth is the class label. Derived from the same inference
  run as the description dataset at no extra cost.

Both JSONL files are directly loadable by `text_qa.py --idtools_jsonl` for H2 evaluation.
For visual inference (VLM with images), a separate inference script parses the JSONL metadata
and loads images via the AgML loader — no images are embedded in the dataset files.

---

## Dataset A: Visual Description

### Schema

**Disease / pest / stress:**
```
Visual:
  image:    <loaded from AgML via image_paths in metadata>
  question: "Can you describe visually the {class} that is exhibited in this image of {crop}?"
  answer:   <synthesized visual description>

Text-only (stored in JSONL, used in text_qa.py):
  question: "Can you describe visually the {class} of a {crop}?"
  answer:   <synthesized visual description>
```

**Weed / crop identification:**
```
Visual:
  image:    <loaded from AgML via image_paths in metadata>
  question: "Can you describe visually the {species} exhibited in this image?"
  answer:   <synthesized visual description>

Text-only (stored in JSONL, used in text_qa.py):
  question: "Can you describe visually the {species}?"
  answer:   <synthesized visual description>
```

Both question forms instruct the model to describe only observable visual characteristics
(color, texture, shape, pattern, size, location on plant) and **not name the class label**
in the answer — keeping the classification task non-trivial.

### Caption Aggregation (two-pass)

**Pass 1 — Per-image captions (vision pass)**
The VLM receives each image individually with the description question plus a no-label suffix.
This produces `n_samples_per_class` independent raw captions (default: 10, set via
`--n_samples_per_class`). Each caption is saved to the checkpoint immediately.

**Pass 2 — Meta-synthesis (text-only pass)**
All N raw captions are fed back to the model in one text-only prompt:

```
You are given {N} visual descriptions of {class} on {crop}, each written from a separate
photograph. Synthesize these into one clear, representative 2-3 sentence description that
captures the consensus visual features (color, texture, shape, pattern, size, location on
plant). Ignore noise or one-off observations. Do not name or repeat the disease, pest,
species, or class label — describe only what can be visually observed.

Descriptions:
1. {caption_1}
2. {caption_2}
...
```

The synthesized output (2-3 sentences) is stored in `metadata.synthesized_description`.

**Pass 3 — Compression (text-only pass)**
The synthesized description is compressed into a single sentence:

```
Compress the following visual description into exactly one concise sentence.
Preserve the most distinctive visual features.
Do not name or repeat any disease, pest, species, or class label.

Description: {synthesized_description}
```

The compressed output becomes the `answer` in `messages[1].content` — the ground truth
for ROUGE scoring in Dataset A — and is also stored as `metadata.compressed_description`
for convenience. The full 2-3 sentence synthesis remains in `metadata.synthesized_description`.

**Rationale**: Drawn from ShareGPT4V and LLaVA-style dataset construction — dense per-image
captions are refined into one coherent ground truth. Synthesizing across N images reduces
hallucination from any single frame, captures intra-class visual variance, and keeps outputs
interpretable (individual raw captions are preserved in JSONL metadata). The one-sentence
compression provides a more compact ground truth for probing whether a model can identify
a class from minimal visual context.

---

## Dataset B: Visual Classification

Derived entirely from Dataset A inference results — no additional model calls required.

### Schema

**Disease / pest / stress:**
```
Visual:
  image:    <loaded from AgML via metadata.image_paths>
  question: "What disease, pest, damage type, or other stress, if any, is exhibited
             in this image of a {crop}?"
  answer:   {class_label}

Text QA (stored in JSONL, used in text_qa.py):
  question: "What disease, pest, damage type, or other stress, if any, is exhibited
             in this description of a {crop}?

             Description: {synthesized_description}"
  answer:   {class_label}
```

**Weed / crop identification:**
```
Visual:
  image:    <loaded from AgML via metadata.image_paths>
  question: "What plant or weed is exhibited in this image?"
  answer:   {species}

Text QA (stored in JSONL, used in text_qa.py):
  question: "What plant or weed is exhibited in this description?

             Description: {synthesized_description}"
  answer:   {species}
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

```

The full registry is in `datasets.yaml` (same directory). All 29 AgML image classification
datasets are listed with inferred `task_type`, `crop`, classes, and `skip` flags. Two are
skipped by default:
- `leaf_counting_denmark` — leaf count regression, not applicable
- `riseholme_strawberry_classification_2021` — fruit ripeness, not applicable

---

## Output Format

### JSONL files (`text_qa.py` compatible)

Both files use the messages schema below. `messages[0].content` is always the text-only
question so both load directly via `text_qa.py --idtools_jsonl`.

**`agml_descriptions.jsonl`** — one record per class:
```json
{
  "messages": [
    {"role": "user",      "content": "Can you describe visually the Early Blight of a Tomato?"},
    {"role": "assistant", "content": "The affected leaves show dark brown, target-like lesions..."}
  ],
  "metadata": {
    "record_type":            "description",
    "agml_dataset":           "plant_village_classification",
    "crop":                   "Tomato",
    "class_label":            "Early Blight",
    "task_type":              "disease",
    "visual_question":        "Can you describe visually the Early Blight that is exhibited in this image of Tomato?",
    "visual_cls_question":    "What disease, pest, damage type, or other stress, if any, is exhibited in this image of a Tomato?",
    "synthesized_description": "The affected leaves show dark brown, target-like lesions...",
    "compressed_description": "The leaves show dark, target-like lesions with concentric rings and yellowing margins.",
    "raw_captions":           ["...", "..."],
    "image_names":            ["img_001.jpg", "img_042.jpg", "..."],
    "n_images_used":          10,
    "generator_model":        "Qwen/Qwen3.6-35B-A3B"
  }
}
```

**`agml_classification.jsonl`** — one record per class (description embedded in question):
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What disease, pest, damage type, or other stress, if any, is exhibited in this description of a Tomato?\n\nDescription: The affected leaves show dark brown, target-like lesions..."
    },
    {"role": "assistant", "content": "Early Blight"}
  ],
  "metadata": {
    "record_type":              "classification",
    "agml_dataset":             "plant_village_classification",
    "crop":                     "Tomato",
    "class_label":              "Early Blight",
    "task_type":                "disease",
    "synthesized_description":  "The affected leaves show dark brown, target-like lesions...",
    "compressed_description":   "The leaves show dark, target-like lesions with concentric rings and yellowing margins.",
    "generator_model":          "Qwen/Qwen3.6-35B-A3B"
  }
}
```

### Visual inference (future VLM script)

Images are not stored in the dataset files. A VLM inference script loads them from
`metadata.image_paths` in the description JSONL:

```python
for line in open("outputs/agml_descriptions.jsonl"):
    r    = json.loads(line)
    meta = r["metadata"]
    # reload images by name from the AgML dataset
    loader     = agml.data.AgMLDataLoader(meta["agml_dataset"])
    contents   = {Path(p).name: p for p in loader.export_contents()}
    images     = [PILImage.open(contents[n]).convert("RGB") for n in meta["image_names"]]
    visual_question = meta["visual_question"]
    # run VLM inference: image + visual_question → predicted description
```

For the classification task, the visual question and `synthesized_description` are both
available in `agml_classification.jsonl` metadata for constructing image+description prompts.

---

## Checkpointing

Generation state is written to `checkpoint.json` after every class.

```json
{
  "plant_village_classification": {
    "Early Blight (Tomato)": {
      "status":                  "done",
      "crop":                    "Tomato",
      "clean_cls":               "Early Blight",
      "task_type":               "disease",
      "synthesized_description": "The affected leaves show dark brown...",
      "compressed_description":  "The leaves show dark, target-like lesions with concentric rings.",
      "raw_captions":            ["...", "..."],
      "image_paths":             ["path/to/img1.jpg", "..."],
      "visual_desc_question":    "Can you describe visually...",
      "visual_cls_question":     "What disease...",
      "text_desc_question":      "Can you describe visually the Early Blight of a Tomato?",
      "text_cls_question":       "What disease... Description: ..."
    }
  }
}
```

On re-run, completed entries are skipped. Pass `--force` to regenerate from scratch.

---

## Model

Default: `Qwen/Qwen3.6-35B-A3B` via vLLM.

Loaded once with `LLM(model=..., dtype="bfloat16", max_model_len=16384, max_num_seqs=64)`.
Both vision pass (Pass 1) and synthesis pass (Pass 2) use the same instance. The model ID
is a CLI argument — any vLLM-compatible VLM can be substituted without code changes.

`max_num_seqs=64` and `max_model_len=16384` are required for Mamba-based MoE models and
high-resolution image inputs respectively.

---

## CLI

```bash
uv run python agml_scraper.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --datasets_config datasets.yaml \
  --n_samples_per_class 10 \
  --output_dir outputs/ \
  [--hub_repo_descriptions your-hf-username/agml-descriptions] \
  [--hub_repo_classification your-hf-username/agml-classification] \
  [--force]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3.6-35B-A3B` | vLLM-compatible VLM |
| `--datasets_config` | `datasets.yaml` | Path to dataset registry |
| `--n_samples_per_class` | `10` | Images sampled per class |
| `--output_dir` | `outputs/` | Directory for JSONL files and checkpoint |
| `--hub_repo_descriptions` | `None` | HF repo to push text description JSONL (optional) |
| `--hub_repo_classification` | `None` | HF repo to push text classification JSONL (optional) |
| `--force` | `False` | Ignore checkpoint and regenerate all entries |
| `--seed` | `42` | Random seed for image sampling |

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
    ┌─ Pass 1: VLM describes each image  ──────────────── vision pass
    │           (N raw captions + no-label instruction)
    ├─ Pass 2: LLM synthesizes N captions → 2-3 sentences  text-only pass
    └─ Pass 3: LLM compresses synthesis → 1 sentence ───── text-only pass

    → Description record  → agml_descriptions.jsonl
         answer = compressed (1 sentence)
         metadata.synthesized_description = synthesized (2-3 sentences)
    → Classification record (reuses both, no extra inference)
                          → agml_classification.jsonl
     │
     ▼
Optionally push text JSONL files → HuggingFace (--hub_repo_*)
```

---

## Visual Inference (`agml_inference.py`)

Evaluates a VLM directly on images using the JSONL files produced by `agml_scraper.py`.
Images are loaded from AgML using `agml_dataset` + `image_names` stored in the description
JSONL metadata — no images are embedded anywhere.

### Dataset A — Visual Description

For each record in `agml_descriptions.jsonl`:
1. Load images by matching `image_names` against the AgML filename index (`build_name_map`).
2. Send each image + `visual_question` (from `metadata.visual_question`) to the VLM.
3. Score each predicted description against the `synthesized_description` (ground truth) using
   **ROUGE-1, ROUGE-2, ROUGE-L** (F-measure, stemming enabled).

Ground truth = the synthesized description stored in `messages[1].content`.

### Dataset B — Visual Classification

For each record in `agml_descriptions.jsonl`:
1. Load the same images.
2. Append a classification constraint suffix to the visual classification question
   (`metadata.visual_cls_question`):
   ```
   " Respond with only the class name, nothing else."
   ```
3. Run inference with short `max_tokens=32` and stop tokens `["\n", ".", ",", ";"]` to
   suppress prose output.
4. Resolve the raw response to a known class via case-insensitive exact match, then
   single-match substring fallback against the cleaned class list from `datasets.yaml`.
5. Score resolved predictions against `class_label` using **Accuracy, F1-macro, F1-weighted**
   (sklearn). Unresolvable responses are excluded from metric computation but counted.

The `visual_cls_question` is stored in `metadata.visual_cls_question` of each description
record — no separate classification JSONL is needed for visual inference.

### Output Files

| File | Contents |
|------|----------|
| `agml_inference_descriptions.csv` | Per-image ROUGE scores (prediction, ground_truth, rouge1/2/L, dataset, class) |
| `agml_inference_classification.csv` | Per-image classification results (raw_prediction, resolved_class, correct, resolvable) |
| `agml_inference_metrics.json` | Aggregated ROUGE and classification metrics per dataset + overall |
| `rouge_by_dataset.png` | Bar chart: ROUGE-1/2/L per AgML dataset |
| `classification_by_dataset.png` | Bar chart: Accuracy/F1-macro/F1-weighted per AgML dataset |

### Sampling Parameters

| Pass | `temperature` | `max_tokens` | `repetition_penalty` | Stop tokens |
|------|--------------|-------------|---------------------|-------------|
| Description | 0.0 | 256 | 1.1 | — |
| Classification | 0.0 | 32 | 1.1 | `\n . , ;` |

### CLI

```bash
uv run python agml_inference.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --datasets_config datasets.yaml \
  --output_dir inference_results/ \
  [--n_samples 3]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3.6-35B-A3B` | vLLM-compatible VLM |
| `--descriptions_jsonl` | `outputs/agml_descriptions.jsonl` | JSONL produced by `agml_scraper.py` |
| `--datasets_config` | `datasets.yaml` | Controls which datasets are evaluated (skip flags respected) |
| `--output_dir` | `inference_results/` | Directory for CSV, JSON, and plots |
| `--n_samples` | `None` (all) | Max images per class to evaluate |

---

---

## Reference Verification Pipeline

The compressed descriptions generated by `agml_scraper.py` are cross-checked against free,
authoritative online sources and optionally refined by a local LLM. The pipeline is fully
open-source and incurs no API costs.

### Scripts

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `export_descriptions_csv.py` | `agml_descriptions.jsonl` | `descriptions_export.csv` | Flatten JSONL to reviewable CSV |
| `build_reference_db.py` | `descriptions_export.csv` | `reference_db.json` | Query Wikipedia, USDA APHIS IDtools, CGIAR HuggingFace per class |
| `verify_descriptions.py` | CSV + reference DB | `descriptions_corrected.csv` | LLM-refine descriptions using references |
| `update_jsonl.py` | corrected CSV + JSONL files | `*_verified.jsonl` | Apply corrections back to JSONL |
| `app.py` | reference DB + corrected CSV | — | Streamlit review UI |

### Step 1 — Export CSV

```bash
uv run python export_descriptions_csv.py \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --output_csv outputs/descriptions_export.csv
```

Extracts `agml_dataset, crop, class_label, task_type, compressed_description` — one row per class.

### Step 2 — Build Reference Database

```bash
uv run python build_reference_db.py \
  --descriptions_csv outputs/descriptions_export.csv \
  --output_db outputs/reference_db.json

# Resume an interrupted run without re-fetching completed entries
uv run python build_reference_db.py \
  --descriptions_csv outputs/descriptions_export.csv \
  --output_db outputs/reference_db.json \
  --skip_existing
```

For each unique `(class_label, crop)`, queries **three free sources** (all open, no API keys required):

| # | Source | API / Method | What is extracted | Limitation |
|---|--------|-------------|-------------------|------------|
| 1 | **Wikipedia** | [Wikipedia REST API](https://en.wikipedia.org/w/api.php) — search + extract endpoints | Introductory plain-text extract of the best-matching article (up to 800 chars) | General encyclopaedia; may lack visual symptom specificity |
| 2 | **USDA APHIS IDtools** | Crawl [idtools.org](https://idtools.org) package index pages → subject→URL map; scrape visual sections of matching fact sheets | Visual identification / symptom / description sections from USDA APHIS Identification Technology Program fact sheets | Coverage is strongest for citrus pests and diseases, phytophthora, aphids, and thrips; limited for other crops |
| 3 | **CGIAR GARDIAN** | HuggingFace `datasets` library — loads `CGIAR/ragas_gardian_evaluation_non_overlapping` and `CGIAR/ragas_gardian_evaluation_overlapping` at startup (~44K × 288-word chunks from CGIAR research publications); keyword search over `context` column | Actual document text from CGIAR agricultural research papers, returned as up to 2 matching chunks per class | ~44K chunks cached in memory at startup (~1–2 min first run, then HF cache); gated datasets (gardian-ai-ready-docs etc.) excluded |

IDtools content is produced by USDA APHIS PPQ in collaboration with university partners and is in the public domain under 17 U.S.C. § 105 (US government works). The index is built once at startup by crawling package index pages (e.g. `?packageID=63`) which list all fact sheet links; individual fact sheets are fetched only for classes that match a subject name.

Entries are written to disk after every class so partial runs are safe. Resume with
`--skip_existing`.

**Status flags:**
- `"found"` — at least one source returned a non-empty excerpt
- `"no_source_found"` — all sources returned nothing; pipeline continues gracefully

**Reference DB schema** (`schema_version: "1.2"`):
```json
{
  "schema_version": "1.2",
  "built_date": "2026-05-14",
  "entries": {
    "plant_village_classification__early_blight__tomato": {
      "agml_dataset": "plant_village_classification",
      "crop": "Tomato",
      "class_label": "Early Blight",
      "task_type": "disease",
      "status": "found",
      "sources": [
        {
          "organization": "Wikipedia",
          "title": "Early blight",
          "url": "https://en.wikipedia.org/wiki/Early_blight",
          "type": "encyclopedia",
          "accessed_date": "2026-05-14",
          "excerpt": "Early blight is caused by Alternaria solani...",
          "search_query": "Early Blight Tomato plant disease symptoms"
        },
        {
          "organization": "USDA APHIS IDtools",
          "title": "Early Blight",
          "url": "https://idtools.org/citrus_diseases/index.cfm?packageID=14&entityID=123",
          "type": "fact_sheet",
          "accessed_date": "2026-05-14",
          "excerpt": "Symptoms: Dark brown to black lesions with concentric rings...",
          "search_query": "Early Blight Tomato"
        },
        {
          "organization": "CGIAR GARDIAN",
          "title": "CGIAR GARDIAN Research Corpus",
          "url": "https://huggingface.co/datasets/CGIAR/ragas_gardian_evaluation_non_overlapping",
          "type": "research_document",
          "accessed_date": "2026-05-14",
          "excerpt": "...early blight caused by Alternaria solani appears as dark, target-like lesions...",
          "search_query": "Early Blight Tomato"
        }
      ]
    }
  }
}
```

### Step 3 — Verify and Refine Descriptions

```bash
uv run python verify_descriptions.py \
  --model Qwen/Qwen3-8B-Instruct \
  --descriptions_csv outputs/descriptions_export.csv \
  --reference_db outputs/reference_db.json \
  --output_csv outputs/descriptions_corrected.csv
```

For each row:
- **Source found** → uses vLLM to rewrite the description as a single sentence grounded in
  the reference excerpts. The LLM is instructed to describe only visual symptoms and to
  **not name the disease, pest, crop, or any direct class identifier**.
- **No source found** → keeps the original compressed description unchanged (`status: no_source`).

Output CSV columns: `agml_dataset, crop, class_label, task_type, compressed_description,
corrected_description, status, source_orgs, source_urls, excerpt_used`

`status` values: `corrected` | `no_source` | `error`

### Step 4 — Apply Corrections to JSONL

```bash
uv run python update_jsonl.py \
  --corrected_csv outputs/descriptions_corrected.csv \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --classification_jsonl outputs/agml_classification.jsonl
```

Writes `agml_descriptions_verified.jsonl` and `agml_classification_verified.jsonl`. For records
with `status: corrected`:
- Descriptions JSONL: `messages[1].content` and `metadata.compressed_description` → corrected text;
  `metadata.verified = true`, `metadata.verification_sources` = list of source URLs
- Classification JSONL: text after `"Description: "` in `messages[0].content` → corrected text

Original files are never modified.

### Review Frontend

```bash
uv run streamlit run app.py
```

Three-tab Streamlit app:

| Tab | What it shows |
|-----|---------------|
| **Reference Database** | Browse all reference DB entries; expand each source to see title, URL, and excerpt |
| **Corrected Descriptions** | Side-by-side original vs. corrected; words shared between the reference excerpt and the corrected description are **highlighted** (yellow `<mark>` in corrected text, bold in excerpt) |
| **Stats** | Count tiles + bar charts by task_type and dataset |

### Full Pipeline — Run Order

```bash
cd experiments/agml_scraper

# 0. Generate JSONL files first (if not already done)
uv run python agml_scraper.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --datasets_config datasets.yaml \
  --n_samples_per_class 10 \
  --output_dir outputs/

# 1. Export JSONL to CSV
uv run python export_descriptions_csv.py \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --output_csv outputs/descriptions_export.csv

# 2. Build reference database (Wikipedia + USDA APHIS IDtools + CGIAR HuggingFace)
#    First run: ~5–15 min depending on number of classes and network latency
uv run python build_reference_db.py \
  --descriptions_csv outputs/descriptions_export.csv \
  --output_db outputs/reference_db.json

# 2b. Resume an interrupted run
uv run python build_reference_db.py \
  --descriptions_csv outputs/descriptions_export.csv \
  --output_db outputs/reference_db.json \
  --skip_existing

# 3. Verify and refine descriptions (requires GPU for vLLM)
#    Only rows with status: "found" are rewritten; no_source rows keep original
uv run python verify_descriptions.py \
  --model Qwen/Qwen3-8B-Instruct \
  --descriptions_csv outputs/descriptions_export.csv \
  --reference_db outputs/reference_db.json \
  --output_csv outputs/descriptions_corrected.csv

# 4. Review in browser before applying changes
#    Pass --output_dir to point at your outputs folder (contains reference_db.json
#    and descriptions_corrected.csv). The sidebar also has a text input to change it at runtime.
uv run streamlit run app.py -- --output_dir outputs/

# Custom path example
uv run streamlit run app.py -- --output_dir /data/agtuned/my_outputs/

# 5. Apply corrections to JSONL files (originals never modified)
uv run python update_jsonl.py \
  --corrected_csv outputs/descriptions_corrected.csv \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --classification_jsonl outputs/agml_classification.jsonl \
  --output_descriptions_jsonl outputs/agml_descriptions_verified.jsonl \
  --output_classification_jsonl outputs/agml_classification_verified.jsonl

# 6. Evaluate verified descriptions with text_qa.py (text-only, no images)
uv run python ../text_qa/text_qa.py \
  --idtools_jsonl outputs/agml_descriptions_verified.jsonl \
  --datasets_config datasets.yaml \
  --model google/gemma-4-E2B-it \
  --output_dir results/agml_descriptions_verified/

# 7. Run agml_inference.py on verified description JSONL (VLM + images, Dataset A + B)
uv run python agml_inference.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --descriptions_jsonl outputs/agml_descriptions_verified.jsonl \
  --datasets_config datasets.yaml \
  --output_dir inference_results/verified/

# 7b. Compare against unverified baseline
uv run python agml_inference.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --descriptions_jsonl outputs/agml_descriptions.jsonl \
  --datasets_config datasets.yaml \
  --output_dir inference_results/baseline/

# 8. Run agml_inference.py on verified classification JSONL (Dataset B only)
#    The classification JSONL embeds the description in the question, so Dataset B
#    scores directly reflect whether the corrected descriptions improve class prediction.
uv run python agml_inference.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --descriptions_jsonl outputs/agml_descriptions_verified.jsonl \
  --datasets_config datasets.yaml \
  --output_dir inference_results/verified_cls/

# 9. [Optional] LLM-as-a-judge — re-run steps 7/8 with --judge_model
#    The judge loads after the main VLM is freed (no extra GPU overhead at inference time).
#    Adds judged_correct column to CSVs and a [judged] J-Acc section to summary.txt.
uv run python agml_inference.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --descriptions_jsonl outputs/agml_descriptions_verified.jsonl \
  --datasets_config datasets.yaml \
  --output_dir inference_results/verified_judged/ \
  --judge_model Qwen/Qwen2.5-20B-Instruct

# 9b. text_qa.py also supports --judge_model for text-only QA
uv run python ../text_qa/text_qa.py \
  --idtools_jsonl outputs/agml_descriptions_verified.jsonl \
  --datasets_config datasets.yaml \
  --model google/gemma-4-E2B-it \
  --output_dir results/agml_descriptions_judged/ \
  --judge_model Qwen/Qwen2.5-20B-Instruct
```

> **Note:** `agml_inference.py` always reads `agml_descriptions_verified.jsonl` (or the
> unverified equivalent) for both Dataset A and Dataset B — the classification questions and
> ground-truth labels are derived from the description JSONL metadata at inference time.
> There is no separate `--classification_jsonl` flag; pass `--descriptions_jsonl` only.

### Description Constraint

All descriptions — original and corrected — must satisfy:
- No disease/pest/condition name or synonym
- No crop or plant species name
- Visual symptoms only: color, texture, shape, pattern, lesion morphology, size, location on plant

This constraint is enforced by the LLM prompt in `verify_descriptions.py` and checked visually
in the Streamlit review UI before `update_jsonl.py` is run.

---

## LLM-as-a-Judge Evaluation

Both `agml_inference.py` and `text_qa.py` support an optional `--judge_model` argument that
loads a second LLM after the main inference model is freed and uses it as an automatic judge.

### How it works

The judge is given a (ground truth, prediction) pair and asked to respond **YES** or **NO**:

| Mode | Prompt | Ground truth | Prediction |
|------|--------|-------------|------------|
| Description (A) | Do these descriptions capture the same visual features? | synthesized description | model's visual description |
| Classification (B) | Is this prediction the correct disease/pest/class? | class label | model's resolved class |
| Text QA | Is this answer semantically correct given the reference? | KisanVaani answer | model's QA answer |

### Execution order

The judge runs **after** the main model is deleted from GPU memory — it never runs in parallel
with the main model, so there is no additional peak GPU requirement. Timing for the judge is
separate from `total_time_s`.

### Output

| Item | Where |
|------|-------|
| `judged_correct` (0/1) | appended column in `agml_inference_descriptions.csv`, `agml_inference_classification.csv`, and `text_qa_*.csv` |
| `judge_accuracy` per dataset/task | `dataset_a_judged`, `dataset_b_judged` keys in `agml_inference_metrics.json`; `judge_accuracy` key per model in `text_qa_metrics.json` |
| `[judged]` section in `summary.txt` | clearly separated from automatic ROUGE/accuracy metrics |

Automatic metrics (ROUGE, accuracy, BERTScore) and judged metrics are always reported separately
so results remain comparable to runs without a judge.

### Default judge model

`Qwen/Qwen2.5-20B-Instruct` — 20B open-source model, strong instruction following, runs on the
same GPU as the main models. Any vLLM-compatible text model can be substituted.

---

## Connecting to `text_qa.py`

Both JSONL files are drop-in replacements for KisanVaani in the text QA evaluation pipeline.

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

**Description task** — tests whether the model has learned what diseases/weeds look like from
language weights alone (H2 hypothesis, grounded in AgML visual classes).

**Classification task** — tests whether the model can identify a class from a visual description
alone (harder, more diagnostic probe of agricultural concept grounding).
