# Stage 0 — Data Generation

[← Back to README](../README.md)

Produces two files:

| File | What it is |
|------|------------|
| `outputs/data_generation/reference_db.json` | Per-class grounding sources fetched from Wikipedia + live web search |
| `outputs/data_generation/agml_classification.jsonl` | Classification Q&A used by `language` assessment (`--classification_jsonl`) |

Two steps, in order.

## Step 1 — fetch grounding sources per class

```bash
uv run python main.py build-reference-db \
  --config     src/config/datasets.yaml \
  --output_db  outputs/data_generation/reference_db.json
```

No GPU or API key needed. Expect multiple hours for the full ~997-class pool. Writes incrementally, so it's safe to interrupt and resume.

To run it dataset-by-dataset instead of one long call:

```bash
DATASET_NAMES=$(uv run python -c "from src.common.dataset_config import active_datasets as a; [print(d.agml_name) for d in a('src/config/datasets.yaml')]")
TOTAL=$(echo "$DATASET_NAMES" | wc -l | tr -d ' ')
n=0
for ds in $DATASET_NAMES; do
    n=$((n + 1))
    echo "[$n/$TOTAL] --- $ds ---"
    uv run python main.py build-reference-db \
        --config src/config/datasets.yaml \
        --output_db outputs/data_generation/reference_db.json \
        --skip_existing \
        --only "^${ds}\$"
done
```

## Step 2 — synthesize classification data from those sources

Needs a GPU (vLLM).

```bash
uv run python main.py data-gen \
  --reference_db outputs/data_generation/reference_db.json \
  --prompts      src/config/prompts.yaml \
  --model        google/gemma-4-26B-A4B-it \
  --output_dir   outputs/data_generation/
```

Descriptions are grounded in up to two sources per class, combined: the web sources from Step 1, and a vision pass that samples `--n_vision_samples` images (default 5) per class and captions them with the same model. Each record's `metadata.grounding_status` says which: `grounded_web_and_vision` (both, merged) / `grounded_vision_only` / `grounded_web_only` / `healthy_skip` / `no_source_found` (neither — empty description, needs a manual fill). The record also keeps `web_description` and `vision_description` separately, not just the final merged one.

Add `--force` to regenerate all classes from scratch. Progress is checkpointed (`checkpoint.json` in `--output_dir`) so re-runs skip completed classes.

## Committing the result

`--output_dir` defaults to `outputs/data_generation/`, which is gitignored (`/outputs`). `dataset/agml_classification.jsonl` is the committed copy everyone else actually uses — after a real run, copy the fresh file over and commit it:

```bash
cp outputs/data_generation/agml_classification.jsonl dataset/agml_classification.jsonl
git add dataset/agml_classification.jsonl
git commit -m "Regenerate agml_classification.jsonl"
```

## Browsing the output

```bash
streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
```

Shows each class's dataset/crop/label, the sources that were fetched for it, and the final synthesized description — see [app/README.md](../app/README.md).
