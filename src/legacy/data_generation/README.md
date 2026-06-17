# Data Generation

Generates two complementary VQA datasets from AgML image classification benchmarks, then validates each class label against authoritative agricultural references.

## What this stage produces

| File | Description |
|------|-------------|
| `agml_descriptions.jsonl` | Open-ended visual description Q&A — scored with ROUGE-SS |
| `agml_classification.jsonl` | Short-answer classification Q&A — scored with F1-macro |
| `checkpoint.json` | Per-class progress state — enables safe resume |
| `reference_db.json` | Wikipedia / IDtools / CGIAR excerpts per class label |

---

## Stage 1a — Generate (`generate.py`)

A 3-pass VLM pipeline applied per dataset × class:

1. **Vision pass** — N images → N raw per-image captions (vision model, temperature=0.2)
2. **Synthesis pass** — N captions → 2-3 sentence consensus description (text-only, temperature=0)
3. **Compression pass** — synthesized description → 1 concise sentence (text-only, temperature=0)

All prompt templates are loaded from `src/config/prompts.yaml` — no strings are hardcoded.

```bash
uv run python src/data_generation/generate.py \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --model      Qwen/Qwen3.6-35B-A3B \
  --output_dir outputs/data_generation/
```

### Checkpointing

Progress is saved to `checkpoint.json` after each class. Re-running the command skips already-completed classes. Use `--force` to regenerate all.

---

## Stage 1b — Build Reference DB (`build_reference_db.py`)

Fetches authoritative descriptions for each class from three sources:

1. **Wikipedia** — introductory encyclopedia excerpts via the MediaWiki API
2. **USDA APHIS IDtools** — pest and disease fact sheets (requires `scrape_idtools.py` discovery pass)
3. **CGIAR GARDIAN** — ~219K research publication chunks via cosine similarity retrieval (cached after first run)

```bash
uv run python src/data_generation/build_reference_db.py \
  --descriptions_csv outputs/data_generation/descriptions_export.csv \
  --output_db        outputs/data_generation/reference_db.json \
  --skip_existing
```

Use `--skip_existing` to avoid re-fetching already-processed entries.

---

## Streamlit UI (`app.py`)

Browse generated descriptions and reference sources side by side.

```bash
streamlit run src/data_generation/app.py -- --output_dir outputs/data_generation/
```

Tabs:
- **Reference Database** — filtered view of all class entries and their sources
- **Corrected Descriptions** — compare original vs. corrected descriptions with source highlights
- **Stats** — counts by task type, dataset, and correction status

---

## Output file formats

**`agml_descriptions.jsonl`** — one record per class:
```json
{
  "messages": [
    {"role": "user", "content": "<text description question>"},
    {"role": "assistant", "content": "<compressed 1-sentence description>"}
  ],
  "metadata": {
    "agml_dataset": "bean_disease_uganda",
    "crop": "bean",
    "class_label": "angular leaf spot",
    "task_type": "disease",
    "visual_question": "...",
    "synthesized_description": "...",
    "image_names": ["img001.jpg", "..."],
    "generator_model": "Qwen/Qwen3.6-35B-A3B"
  }
}
```

**`agml_classification.jsonl`** — one record per class, derived at no extra inference cost:
```json
{
  "messages": [
    {"role": "user", "content": "<text classification question with description>"},
    {"role": "assistant", "content": "angular leaf spot"}
  ],
  "metadata": { ... }
}
```

**`reference_db.json`**:
```json
{
  "schema_version": "1.2",
  "entries": {
    "bean_disease_uganda__angular_leaf_spot__bean": {
      "status": "found",
      "sources": [{"organization": "Wikipedia", "excerpt": "..."}]
    }
  }
}
```
