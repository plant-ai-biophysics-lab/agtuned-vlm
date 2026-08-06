# Data Generation (legacy)

`generate.py` and `build_reference_db.py` moved to `src/data_generation/` and were rewritten — see [docs/stage0_data_generation.md](../../../docs/stage0_data_generation.md) for the current pipeline. They no longer produce `agml_descriptions.jsonl`.

The tools below are specific to that older, description-focused pipeline (`agml_descriptions.jsonl`, CGIAR/IDtools sourcing, description verification/correction) and stay here.

---

## Streamlit UI (`app.py`)

Browse generated descriptions and reference sources side by side.

```bash
streamlit run src/legacy/data_generation/app.py -- --output_dir outputs/data_generation/
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
