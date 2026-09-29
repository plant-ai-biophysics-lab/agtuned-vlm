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

## Step 1 — baseline

```bash
uv run python main.py detection-baseline \
    --model google/gemma-4-E4B-it \
    --manifest dataset/selected_images_detection_10.json
```

## Step 2 — QC: plot predicted vs. ground-truth boxes

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv
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
| `--iou_threshold` (`detection-baseline`) | `0.5` | IoU cutoff for precision/recall/F1 |
| `--n_per_dataset` (`detection-qc-plot`) | `3` | Sampled images per dataset |
