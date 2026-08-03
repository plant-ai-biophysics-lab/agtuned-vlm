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

Image datasets are sourced from [huggingface.co/Project-AgML](https://huggingface.co/Project-AgML) Only classification datasets are used — detection/segmentation datasets in the org are excluded in this step.

**`src/config/datasets.yaml` is committed pre-curated** — you don't need to run the two commands below to get started; `download-datasets` (next section) already works against it as-is. Only re-run them if the Project-AgML org's dataset list has changed since, or you want to bring a currently-inactive dataset online.

```bash
# One-time / when the Project-AgML org's dataset list changes:
uv run python main.py catalog-datasets \
  --org Project-AgML \
  --output src/config/hf_catalog.json

# Regenerate src/config/datasets.yaml (writes datasets.yaml.new — review the
# diff and outputs/migration/datasets_migration_review.md before overwriting):
uv run python main.py migrate-datasets \
  --catalog src/config/hf_catalog.json \
  --current src/config/datasets.yaml \
  --output  src/config/datasets.yaml.new
```

**Manual review is required for any dataset new to this migration.** `migrate-datasets` can only auto-fill `crop`/`task_type`/`class_overrides` for datasets that already existed in the previous `datasets.yaml` (carried forward by name match); every genuinely new dataset gets written with `crop: null`, `task_type: null`, and `skip: true` — inert until a human fills those in and flips `skip: false`. This is why the committed file only has 26 of its 178 datasets active. When adding a new dataset by hand:
- `task_type` must be one of `disease` / `pest_damage` / `weed_identification` / `crop_identification` (see the options block at the top of the YAML).
- `class_overrides` reassigns individual classes to a different `task_type` than the dataset's default — e.g. a `disease` dataset with an incidental pest class. **The override key must match the class string exactly as it appears in that dataset's `classes:` list** (same underscores/casing) — a cosmetic-looking mismatch (e.g. `mites and trips` vs. the real `mites_and_trips`) silently no-ops instead of erroring, so double check with `DatasetConfig.effective_task_type()` after editing, not just by eye.

To download all active datasets and take a class-balanced subsample (fixed cap per class, default 50, adjustable via `--n_per_class`), recording exactly which images were chosen for reproducibility:

```bash
uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --n_per_class 50 \
  --output dataset/selected_images.json
```

`dataset/selected_images.json` records the HF row index of every selected image per `(dataset, class)`. Re-running with the same `--seed`/`--n_per_class` reproduces the identical selection; pass `--from_manifest dataset/selected_images.json` to reuse an existing selection exactly (e.g. after re-cataloging) without re-sampling.

**If HuggingFace rate-limits you partway through:** the script retries each dataset automatically on a 429/503 (exponential backoff, 5 attempts) before giving up and recording it under `"failed"` in the manifest. If it still doesn't get through everything in one run, resume with `--skip_existing` — this reuses every dataset already recorded in `--output` and only (re)processes what's missing or failed, instead of starting over:

```bash
uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --n_per_class 50 \
  --output dataset/selected_images.json \
  --skip_existing
```

**Changing `--n_per_class`:** `--skip_existing` only reuses prior results when the requested `--n_per_class` matches what's already in the manifest — a different cap invalidates every dataset's sample (whichever classes exceed the cap get freshly, independently resampled; it is not an incremental "add more images" operation), so with a new value the script warns and reprocesses everything from scratch.

**Summary figures:** every run also writes a set of charts to `--chart_dir` (default `outputs/agml_hf/`):

| File | Shows |
|------|-------|
| `datasets_per_task_type.png` | Number of active datasets per `task_type` (disease / pest_damage / weed_identification / crop_identification) |
| `classes_per_dataset.png` | Number of classes in each dataset, sorted, color-coded by task type |
| `images_selected_per_dataset.png` | Total images selected per dataset after capping |
| `images_per_class_distribution.png` | Histogram of images-per-class, with a line at the `--n_per_class` cap so you can see how many classes actually hit it |
| `summary.txt` | Total dataset/class/image counts + per-task-type breakdown |

Regenerate just the charts from an existing manifest without re-downloading:

```bash
uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --output dataset/selected_images.json \
  --plot_only
```

---

### Stage 0 — Data Generation (pre-built, committed to repo)

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
