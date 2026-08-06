# Streamlit viewers

Each is standalone — run with `streamlit run app/<file>.py -- <args>`.

## classification_viewer.py

```bash
streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
```

Reads `agml_classification.jsonl` (dataset, crop, class, task type, grounding status, synthesized description) and joins each row against `reference_db.json` for the actual sources (organization, title, URL, excerpt) that produced it. Sidebar filters by dataset / task type / grounding status; click a row to see the sample image, description, and sources for that class.
