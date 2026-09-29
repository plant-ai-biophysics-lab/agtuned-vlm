# Paper 2, Stage 2 — Object Detection (Baseline vs. Rubric)

[← Back to README](../README.md)

A leftover from Paper 1: every prior stage in this repo classifies a whole image into a single label. Project-AgML also has genuine object-detection datasets (bounding boxes, not a single class per image). This stage brings those in as their own condition — **Detection-Baseline** (direct box prediction, no reasoning) vs. **Detection-Baseline-with-Rubric** (a fill-in-the-blank localization checklist before printing boxes) — the same rubric idea as classification's `verifier-cot` generation step (Stage 3). Same single-shot generation both times, **no tournament/verifier machinery** (see Stage 5 for that).

## Step 0 — build the detection dataset manifest

Same class-balanced sampling as the classification pipeline, applied to detection-type Project-AgML repos — balances each dataset on every image's primary annotated category, but caches the full per-image annotation list.

```bash
uv run python main.py catalog-detection-datasets
uv run python main.py migrate-detection-datasets
uv run python main.py download-detection-datasets \
    --output dataset/selected_images_detection_10.json
```

Writes the manifest plus per-image annotation sidecars under `dataset/images_detection/`. **Run this remotely** — real HF downloads, some detection repos are several GB and occasionally stall. `download-detection-datasets` supports `--only`/`--skip_existing` for per-dataset retry loops (see Stage 5's own doc for the pattern).

## Step 1 — baseline vs. baseline with rubric

```bash
uv run python main.py detection-baseline \
    --model google/gemma-4-E4B-it \
    --manifest dataset/selected_images_detection_10.json

uv run python main.py detection-baseline \
    --model google/gemma-4-E4B-it \
    --manifest dataset/selected_images_detection_10.json \
    --rubric --output_dir outputs_final/detection_baseline_rubric/
```

`--rubric` wraps the question through the same fill-in-the-blank localization checklist `detection-verifier`'s own structured generation uses — object/class, location, occlusion, size, before printing boxes. A different `--output_dir` is required so the two runs don't collide.

## Step 2 — QC: plot predicted vs. ground-truth boxes

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv
```

The only reliable way to catch a bbox-coordinate-alignment bug — run on a handful of images from each condition before trusting either at scale. To see both conditions on the same sampled images side by side:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --compare_ckpt outputs_final/detection_baseline_rubric/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --compare_labels Baseline,Rubric
```

## Step 3 — comparison: bar charts

```bash
uv run python main.py detection-rubric-summary
```

No GPU needed — filters both checkpoints to the exact same overlapping `(agml_dataset, image_name)` pairs. Writes (under `outputs_final/detection_rubric_summary/<safe_model_name>/`): `detection_rubric_summary.png` (mean precision/recall/F1@IoU50 + mean matched IoU, 95% CI), `detection_rubric_summary_by_dataset.png`, `detection_rubric_summary_iou_thresholds.png` (F1 vs. IoU threshold, since a fixed 0.5 cutoff can zero out real "close but not tight enough" localizations), plus CSV equivalents. Shares `src/common/bbox_summary.py` with Stage 5's own Baseline-vs-Verifier report.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--n_per_class` (`download-detection-datasets`) | `10` | Images per class |
| `--output` (`download-detection-datasets`) | `dataset/selected_images_detection.json` | Manifest path |
| `--model` (`detection-baseline`) | *(required)* | Model to run |
| `--manifest` (`detection-baseline`) | `dataset/selected_images_detection.json` | Which manifest to use |
| `--output_dir` (`detection-baseline`) | `outputs_final/detection_baseline/` | Checkpoint output dir |
| `--rubric` (`detection-baseline`) | off | Fill-in-the-blank localization checklist before boxes |
| `--iou_threshold` (`detection-baseline`) | `0.5` | IoU cutoff for precision/recall/F1 |
| `--n_per_dataset` (`detection-qc-plot`) | `3` | Sampled images per dataset |
| `--compare_ckpt` (`detection-qc-plot`) | none | Second checkpoint to plot side by side |
| `--baseline_dir` / `--rubric_dir` (`detection-rubric-summary`) | `outputs_final/detection_baseline/` / `outputs_final/detection_baseline_rubric/` | Checkpoint dirs to compare |
| `--model` (`detection-rubric-summary`) | `google/gemma-4-E4B-it` | Model whose checkpoints to compare |
