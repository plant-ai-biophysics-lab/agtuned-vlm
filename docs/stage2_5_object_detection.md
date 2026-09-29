# Paper 1, Stage 2.5 — Object Detection (Baseline vs. Rubric)

[← Back to README](../README.md)

A leftover from Paper 1: every prior stage in this repo classifies a whole
image into a single label. Project-AgML also has genuine object-detection
datasets (bounding boxes, not a single class per image), never pulled or
used. This stage brings those in as their own Paper 1 condition —
**Detection-Baseline** (direct box prediction, no reasoning) vs.
**Detection-Baseline-with-Rubric** (a fill-in-the-blank localization
checklist before printing boxes) — mirroring Stage 2's classification
"Baseline vs. Rubric" comparison (`docs/stage2_baseline_rubric.md`), just for
the spatial task. Same single-shot generation both times, **no
tournament/verifier machinery** — that's a separate, later idea
(`docs/stage5_object_detection.md`'s Detection-Verifier), not part of this
stage.

All of the scripts below already existed as Stage 5's data-prep pipeline
(built earlier, never actually run) — this stage is mostly about actually
running them and adding the rubric condition + comparison charts on top.

## Step 0 — build the detection dataset manifest (10 images/class)

Same class-balanced sampling as the classification pipeline's
`dataset/selected_images_10.json`, applied to detection-type Project-AgML
repos — balances each dataset on every image's PRIMARY annotated category
(a detection image can have multiple instances/categories at once; see
`download_and_sample_detection.py`'s module docstring), but still caches the
FULL per-image annotation list, not just the primary one.

```bash
uv run python main.py catalog-detection-datasets \
    --excluded_catalog src/config/hf_catalog_excluded.json \
    --output src/config/hf_detection_catalog.json

uv run python main.py migrate-detection-datasets \
    --catalog src/config/hf_detection_catalog.json \
    --output src/config/detection_datasets.yaml

uv run python main.py download-detection-datasets \
    --config src/config/detection_datasets.yaml \
    --catalog src/config/hf_detection_catalog.json \
    --n_per_class 10 \
    --output dataset/selected_images_detection_10.json
```

This writes `dataset/selected_images_detection_10.json` — same top-level
JSON shape as `dataset/selected_images_10.json` (`n_per_class`,
`datasets: {name: {classes: {label: [row_indices]}, hf_repo_id,
hf_config}}`), plus `class_names`/`bbox_convention` per dataset. Per-image
annotations (boxes + category ids/names) are cached as a JSON sidecar next
to each downloaded image under `dataset/images_detection/`, so the manifest
+ image cache together are a complete, inspectable local copy of everything
downstream reads.

**Run this remotely** — real HF downloads, some detection repos are several
GB (e.g. `CocoaMFDB_detection` ~5.5GB) and occasionally stall with no
exception raised. `download-detection-datasets` supports `--only`/
`--skip_existing` for exactly this — see `docs/stage5_object_detection.md`'s
Step 2 for the per-dataset `timeout`-guarded retry loop, same flags apply
here unchanged.

## Step 1 — baseline (direct box prediction, no reasoning)

```bash
uv run python main.py detection-baseline \
    --model google/gemma-4-E4B-it \
    --config src/config/detection_datasets.yaml \
    --manifest dataset/selected_images_detection_10.json \
    --output_dir outputs_final/detection_baseline/
```

## Step 2 — baseline with rubric

Same script, `--rubric` wraps the question through
`src/config/verifier_prompts_detection.yaml`'s `structured_templates`
(`object_detection` key) — the model fills in object/class, general
location, overlap/occlusion, and approximate size for what it observes
*before* printing boxes. This is the same criteria chain
`detection-verifier`'s own structured candidate generation already uses,
just with no K-candidate sampling or PPT tournament on top — a plain
single-shot prompt, like Stage 2's classification rubric.

```bash
uv run python main.py detection-baseline \
    --model google/gemma-4-E4B-it \
    --config src/config/detection_datasets.yaml \
    --manifest dataset/selected_images_detection_10.json \
    --rubric \
    --output_dir outputs_final/detection_baseline_rubric/
```

(A different `--output_dir` than Step 1 is required — both write to
`<output_dir>/<safe_model_name>/ckpt_<safe_model_name>.csv`.)

## Step 3 — QC: plot predicted vs. ground-truth boxes

Run this on a handful of images from EACH condition before trusting
either at scale — the only reliable way to catch a bbox-coordinate-
alignment bug (see `detection_qc_plot.py`'s module docstring).

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --n_per_dataset 3

uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline_rubric/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --n_per_dataset 3
```

To see both conditions' predictions on the SAME sampled images, side by
side against the same ground truth:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --compare_ckpt outputs_final/detection_baseline_rubric/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
    --compare_labels Baseline,Rubric \
    --n_per_dataset 3
```

## Step 4 — comparison: bar charts

```bash
uv run python main.py detection-rubric-summary \
    --baseline_dir outputs_final/detection_baseline/ \
    --rubric_dir outputs_final/detection_baseline_rubric/ \
    --model google/gemma-4-E4B-it
```

No GPU needed — filters both checkpoints to the exact same overlapping
`(agml_dataset, image_name)` pairs, then writes (under
`outputs_final/detection_rubric_summary/<safe_model_name>/`):

- `detection_rubric_summary.png` — mean precision/recall/F1@IoU50 + mean
  matched IoU, 95% bootstrap CI, one bar per condition.
- `detection_rubric_summary_by_dataset.png` — mean F1@IoU50 per dataset.
- `detection_rubric_summary_iou_thresholds.png` — F1 vs. IoU threshold
  (0.3/0.5/0.75), one line per condition — a single fixed 0.5 cutoff can
  zero out real "close but not tight enough" localizations.
- `detection_rubric_summary.csv` / `detection_rubric_summary_iou_thresholds.csv`
  — the same numbers as CSV.

This reuses `src/common/bbox_summary.py`, the same condition-generic
comparison core `docs/stage5_object_detection.md`'s Baseline-vs-Verifier
`detection-summary` report uses — adding a third condition later (e.g. a
detection verifier) means adding one more `(label, color, rows)` tuple, not
a new plotting module.
