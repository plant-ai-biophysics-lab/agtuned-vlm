# Stage 0 — Data Generation

[← Back to README](../README.md)

Produces the JSONL files used by language and vision assessment:

| File | Used by |
|------|---------|
| `dataset/agml_descriptions.jsonl` | `language` and `vision` stages (`--descriptions_jsonl`) |
| `dataset/agml_classification.jsonl` | `language` stage (`--classification_jsonl`) |

**The committed copies of these two files are stale** — generated under an older, pre-HF-migration pipeline that captioned raw images with a VLM instead of grounding in external sources, and covering only 215 classes (mostly `disease`) out of the current 121-dataset/997-class active pool. Regenerating them for the full pool with the current pipeline hasn't been done yet.

**Two steps, in order, split across two machines.** Descriptions are grounded in externally fetched sources, not model-generated captions from images — `data-gen` has no vision pass at all; it only synthesizes from whatever `build-reference-db` already fetched. Step 1 is pure network I/O (no GPU needed) — run it locally. Step 2 needs a GPU for vLLM — run it on the farm, using the `reference_db.json` Step 1 produced.

## Step 1 (local) — fetch grounding sources per class

Wikipedia + a live web search (`ddgs`, no API key needed) per class; each candidate web page is fetched fresh and cleaned to plain text with `trafilatura`, kept only if it actually mentions the class label. No CGIAR/USDA pre-built corpus — those didn't guarantee coverage of any *specific* class, a fresh per-class search gets closer to that.

```bash
uv run python main.py build-reference-db \
  --config     src/config/datasets.yaml \
  --output_db  outputs/data_generation/reference_db.json
```

**This is slow — expect multiple hours for the full ~997-class pool** (Wikipedia's API is rate-limited with polite retry/backoff; each class also fetches up to 5 web pages with a 1s sleep between each). Writes incrementally to `--output_db` after every class, so it's safe to interrupt (Ctrl-C) and resume later.

**Recommended: run it dataset-by-dataset, not as one long call**, same pattern as `download-datasets` (see [stage0a_datasets.md](stage0a_datasets.md)) — a single multi-hour foreground process is fragile (one flaky network blip and you lose your terminal session's progress display, though not the actual data). Wrap it with `--only` per dataset:

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

`--skip_existing` means a killed/interrupted run just picks up where it left off on the next invocation — re-run the whole loop as many times as needed. Or run it in the background (`nohup ... &`, `tmux`, etc.) if you'd rather not babysit it in the foreground at all — there's nothing interactive about it.

Once done, copy `outputs/data_generation/reference_db.json` to the farm (e.g. `scp`) for Step 2.

## Step 2 (farm, GPU) — synthesize descriptions from those sources

```bash
uv run python main.py data-gen \
  --reference_db outputs/data_generation/reference_db.json \
  --prompts      src/config/prompts.yaml \
  --model        google/gemma-4-26B-A4B-it \
  --output_dir   outputs/data_generation/
```

A class with no usable source (nothing found, or a "healthy" class that's intentionally skipped at fetch time, or the synthesis pass judging the fetched excerpts to have too little visual detail) still gets a record, but with an empty description — check `metadata.grounding_status` in the output JSONL (`grounded` / `no_source_found` / `healthy_skip` / `insufficient_visual_detail`) to find and manually fill these in rather than assuming every row has a usable description.

Add `--force` to regenerate all classes from scratch. Progress is checkpointed (`checkpoint.json` in `--output_dir`) so re-runs skip completed classes.
