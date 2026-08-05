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

Image datasets are sourced from [huggingface.co/Project-AgML](https://huggingface.co/Project-AgML). Only classification datasets are used — detection/segmentation/multimodal datasets in the org are excluded in this step.

#### Quick start — just want the images? Start here.

**`src/config/datasets.yaml`, `src/config/hf_catalog.json`, `src/config/hf_catalog_excluded.json`, and `dataset/selected_images.json` are all committed** — every fragile, slow, network-dependent step (org-wide cataloging, dataset-registry migration, and picking exactly which images to sample) is already done. **Do not run `catalog-datasets` or `migrate-datasets` to get started** — they exist for maintaining this pipeline, not for using it, and re-running them means re-fighting real issues we already hit and fixed (HF's `trust_remote_code` API break, multi-config datasets with no default, farm DNS/Xet hangs — see git history on these three scripts if curious).

Your one command:

```bash
uv sync
uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --catalog src/config/hf_catalog.json \
  --from_manifest dataset/selected_images.json \
  --output dataset/selected_images.json
```

`--from_manifest` reproduces the *exact* image selection already committed (same HF row index per `(dataset, class)`) — not just a similarly-sized independent sample — so this pulls the same images already in use, byte-for-byte reproducible. This only downloads the actual image data; it doesn't re-decide what to download.

**If a dataset hangs** (a real, still-open HuggingFace issue with their newer Xet transfer backend — see [huggingface/datasets#8129](https://github.com/huggingface/datasets/issues/8129)): the script has no internal timeout, so a stalled connection blocks forever with no error. Don't run the command above as one long call on a flaky network — wrap it per-dataset instead:

```bash
DATASET_NAMES=$(uv run python -c "from src.common.dataset_config import active_datasets as a; [print(d.agml_name) for d in a('src/config/datasets.yaml')]")
TOTAL=$(echo "$DATASET_NAMES" | wc -l | tr -d ' ')
n=0
for ds in $DATASET_NAMES; do
    n=$((n + 1))
    echo "[$n/$TOTAL] --- $ds ---"
    timeout 300 uv run python main.py download-datasets \
        --config src/config/datasets.yaml \
        --catalog src/config/hf_catalog.json \
        --from_manifest dataset/selected_images.json \
        --output dataset/selected_images.json \
        --skip_existing \
        --no_charts \
        --only "^${ds}\$"
done

uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --output dataset/selected_images.json \
  --plot_only
```

`timeout 300` (5 min) kills a stuck dataset instead of blocking the whole batch; `--skip_existing` means anything already downloaded is skipped, so a killed/failed dataset just gets retried the next time you run this — including the final `--plot_only` invocation, which regenerates charts once at the end against everything actually downloaded so far. Re-run the whole loop as many times as needed; it only ever fills in gaps, never redoes completed work.

**Checking what's actually done vs. missing:** `outputs/agml_hf/summary.txt` (written by the `--plot_only` step above) reports `"X succeeded, Y failed"` up front, with every failed/missing dataset listed by name and reason underneath. This is computed by diffing the full active dataset list in `datasets.yaml` against what's actually present in the manifest — so it correctly counts a dataset that silently hung and got `timeout`-killed as missing too, not just ones that raised a catchable error.

**Summary figures**, written to `--chart_dir` (default `outputs/agml_hf/`):

| File | Shows |
|------|-------|
| `datasets_per_task_type.png` | Number of active datasets per `task_type` (disease / pest_damage / weed_identification / crop_identification) |
| `classes_per_task_type.png` | Same, but counting individual classes at their **effective** task type — i.e. `class_overrides` applied, so a pest class nested in a `disease` dataset counts under `pest_damage` here |
| `task_type_split_pie.png` | Both of the above as a side-by-side pie comparison |
| `classes_per_dataset.png` | Number of classes in each dataset, sorted, color-coded by task type |
| `images_selected_per_dataset.png` | Total images selected per dataset after capping |
| `images_per_class_distribution.png` | Histogram of images-per-class, with a line at the `--n_per_class` cap so you can see how many classes actually hit it |
| `summary.txt` | Total dataset/class/image counts, per-task-type breakdown (dataset- and class-level), and the failed/missing dataset list described above |

All charts only ever count successfully-downloaded data — failed/missing datasets are excluded from every number and every plot, not just flagged.

---

#### Maintenance — only if you're changing what gets downloaded, not just downloading it

Re-run these only if the Project-AgML org's dataset list has changed, or you want to bring a currently-inactive dataset online — not as part of normal setup.

```bash
uv run python main.py catalog-datasets \
  --org Project-AgML \
  --output src/config/hf_catalog.json \
  --excluded_output src/config/hf_catalog_excluded.json

# Regenerate src/config/datasets.yaml (writes datasets.yaml.new — review the
# diff and outputs/migration/datasets_migration_review.md before overwriting):
uv run python main.py migrate-datasets \
  --catalog src/config/hf_catalog.json \
  --current src/config/datasets.yaml \
  --output  src/config/datasets.yaml.new
```

**Manual review is required for any dataset new to this migration.** `migrate-datasets` can only auto-fill `crop`/`task_type`/`class_overrides` for datasets that already existed in the previous `datasets.yaml` (carried forward by name match); every genuinely new dataset gets written with `crop: null`, `task_type: null`, and `skip: true` — inert until a human fills those in and flips `skip: false`. When adding a new dataset by hand:
- `task_type` must be one of `disease` / `pest_damage` / `weed_identification` / `crop_identification` (see the options block at the top of the YAML).
- `class_overrides` reassigns individual classes to a different `task_type` than the dataset's default — e.g. a `disease` dataset with an incidental pest class. **The override key must match the class string exactly as it appears in that dataset's `classes:` list** (same underscores/casing) — a cosmetic-looking mismatch (e.g. `mites and trips` vs. the real `mites_and_trips`) silently no-ops instead of erroring, so double check with `DatasetConfig.effective_task_type()` after editing, not just by eye.

Changing the sample itself — a fresh (not `--from_manifest`) class-balanced subsample, fixed cap per class (default 50, adjustable via `--n_per_class`):

```bash
uv run python main.py download-datasets \
  --config src/config/datasets.yaml \
  --catalog src/config/hf_catalog.json \
  --n_per_class 50 \
  --output dataset/selected_images.json
```

**Changing `--n_per_class`:** `--skip_existing` only reuses prior results when the requested `--n_per_class` matches what's already in the manifest — a different cap invalidates every dataset's sample (whichever classes exceed the cap get freshly, independently resampled; it is not an incremental "add more images" operation), so with a new value the script warns and reprocesses everything from scratch.

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
