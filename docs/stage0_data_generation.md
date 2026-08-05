# Stage 0 — Data Generation (pre-built, committed to repo)

[← Back to README](../README.md)

The JSONL files required by language and vision assessment are committed to `dataset/` and ready to use:

| File | Used by |
|------|---------|
| `dataset/agml_descriptions.jsonl` | `language` and `vision` stages (`--descriptions_jsonl`) |
| `dataset/agml_classification.jsonl` | `language` stage (`--classification_jsonl`) |

If you need to regenerate them (e.g. for a new dataset or updated prompts):

```bash
uv run python main.py data-gen \
  --config     src/config/datasets.yaml \
  --prompts    src/config/prompts.yaml \
  --model      google/gemma-4-26B-A4B-it \
  --output_dir outputs/data_generation/
```

Add `--force` to regenerate all classes from scratch. Progress is checkpointed so re-runs skip completed classes.
