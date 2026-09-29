# Paper 2, Stage 2 — Baseline Object Detection Runs

[← Back to README](../README.md)

A leftover from Paper 1: every prior stage in this repo classifies a whole image into a single label. Project-AgML also has genuine object-detection datasets (bounding boxes, not a single class per image). This stage is the spatial-task analog of [Stage 1](stage1_baseline_classification.md) — **Detection-Baseline**: direct box prediction, no reasoning, no rubric, no tournament/verifier machinery.

Rubric and verifier conditions for detection will likely follow the same shape as classification's (Stage 3) once these baseline runs are done — deferred for now, not part of this stage yet. `detection-baseline --rubric` and `detection-rubric-summary` still exist in the code if you want to try them early, just not covered here.

## Step 0 — build the detection dataset manifest

Same class-balanced sampling as the classification pipeline, applied to detection-type Project-AgML repos — balances each dataset on every image's primary annotated category, but caches the full per-image annotation list.

```bash
uv run python main.py catalog-detection-datasets
uv run python main.py migrate-detection-datasets
uv run python main.py download-detection-datasets \
    --output dataset/selected_images_detection_10.json
```

Writes the manifest plus per-image annotation sidecars under `dataset/images_detection/`. **Run this remotely** — real HF downloads, some detection repos are several GB and occasionally stall. `download-detection-datasets` supports `--only`/`--skip_existing` for per-dataset retry loops (see Stage 5's own doc for the pattern).

### Smoke test (recommended, before Step 1)

Validate the whole pipeline — manifest → generation → box parsing → scoring — on one small, known-good dataset (`GEMINI_cowpea_flower_detection`, 990 real images) before committing GPU time to all ~50 datasets.

`GEMINI_cowpea_flower_detection` isn't in `dataset/selected_images_detection_10.json` yet — add it (merges into the existing manifest, doesn't touch the other datasets already in it):

```bash
uv run python main.py catalog-detection-datasets \
    --only "^GEMINI_cowpea_flower_detection$"
uv run python main.py migrate-detection-datasets
uv run python main.py download-detection-datasets \
    --only "^GEMINI_cowpea_flower_detection$" \
    --skip_existing \
    --output dataset/selected_images_detection_10.json
```

Check `src/config/detection_datasets.yaml`'s entry for it afterward — if `class_names` didn't auto-resolve, it'll be marked `skip: true` and needs a manual fix (same "nothing silently dropped, inspectable in the yaml" convention as Stage 5's own catalog step) before the download step above will pick it up.

Then run the actual smoke test:

```bash
uv run python main.py detection-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --manifest dataset/selected_images_detection_10.json \
    --smoketest
```

`--smoketest` restricts the run to just this one dataset (10 images) regardless of what else is in the manifest/config. Confirm the checkpoint looks sane (non-empty `pred_boxes_json`, plausible precision/recall) and run Step 2's QC plot on it before moving on to the full Step 1 run below.

## Step 1 — baseline

```bash
uv run python main.py detection-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --manifest dataset/selected_images_detection_10.json
```

## Step 2 — QC: plot predicted vs. ground-truth boxes

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

The only reliable way to catch a bbox-coordinate-alignment bug — run on a handful of images before trusting the checkpoint at scale.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--n_per_class` (`download-detection-datasets`) | `10` | Images per class |
| `--output` (`download-detection-datasets`) | `dataset/selected_images_detection.json` | Manifest path |
| `--model` (`detection-baseline`) | *(required)* | Model to run |
| `--manifest` (`detection-baseline`) | `dataset/selected_images_detection.json` | Which manifest to use |
| `--output_dir` (`detection-baseline`) | `outputs_final/detection_baseline/` | Checkpoint output dir |
| `--smoketest` (`detection-baseline`) | off | Restrict to `GEMINI_cowpea_flower_detection` only |
| `--iou_threshold` (`detection-baseline`) | `0.5` | IoU cutoff for precision/recall/F1 |
| `--n_per_dataset` (`detection-qc-plot`) | `3` | Sampled images per dataset |
