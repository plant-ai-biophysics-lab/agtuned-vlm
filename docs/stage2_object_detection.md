# Paper 2, Stage 2 — Baseline Object Detection Runs

[← Back to README](../README.md)

A leftover from Paper 1: every prior stage in this repo classifies a whole image into a single label. Project-AgML also has genuine object-detection datasets (bounding boxes, not a single class per image). This stage is the spatial-task analog of [Stage 1](stage1_baseline_classification.md) — **Detection-Baseline**: direct box prediction, no reasoning, no rubric, no tournament/verifier machinery.

Rubric and verifier conditions for detection are [Stage 4](stage4_detection_verifier.md) — design not started yet, deferred until these baseline runs are done. `detection-baseline --rubric` and `detection-rubric-summary` still exist in the code if you want to try them early, just not covered here.

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

Validate the whole pipeline — manifest → generation → box parsing → scoring — on one small, known-good dataset (`GEMINI_cowpea_flower_detection`, 990 real images) before committing GPU time to all ~50 datasets:

```bash
uv run python main.py detection-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --manifest dataset/selected_images_detection_10.json \
    --output_dir outputs_final/detection_baseline_thinking_smoketest/ \
    --smoketest
```

`--smoketest` restricts the run to just this one dataset (10 images) regardless of what else is in the manifest/config. Confirm the checkpoint looks sane (non-empty `pred_boxes_json`, plausible precision/recall) and run Step 2's QC plot on it before moving on to the full Step 1 run below.

## Step 1 — baseline (thinking enabled)

```bash
uv run python main.py detection-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --manifest dataset/selected_images_detection_10.json \
    --output_dir outputs_final/detection_baseline_thinking/
```

Thinking is **on by default** — same standard workflow as [Stage 1](stage1_baseline_classification.md)'s classification baseline. `--chat_template_path` defaults to the vendored Gemma-4 template alongside it (its stock template has no thinking support at all); `--gen_max_tokens` defaults to `4096`, enough headroom that a real thinking trace won't truncate before reaching box output (confirmed empirically: at `1024`, 4/10 rows on a real run never even finished the thinking block).

## Without thinking (optional, for a thinking-vs-not comparison)

Add `--no-enable_thinking` — same command otherwise, a different `--output_dir` (e.g. `outputs_final/detection_baseline/`) so the two checkpoints don't collide.

## Step 2 — QC: plot predicted vs. ground-truth boxes

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

The only reliable way to catch a bbox-coordinate-alignment bug — run on a handful of images before trusting the checkpoint at scale.

## For a non-Gemma model

`--chat_template_path`'s default assumes Gemma-4. If you override `--model` to a Qwen checkpoint (whose own bundled template already supports thinking directly), also pass `--chat_template_path ""` to unset the Gemma-4-specific template — otherwise it renders the wrong prompt format entirely, not just fail to enable thinking.

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
| `--enable_thinking` / `--no-enable_thinking` (`detection-baseline`) | **on** | Extended thinking before box output |
| `--chat_template_path` (`detection-baseline`) | vendored Gemma-4 template | Custom chat template — pass `""` to unset for a non-Gemma model |
| `--gen_max_tokens` (`detection-baseline`) | `4096` | Generation budget |
| `--n_per_dataset` (`detection-qc-plot`) | `3` | Sampled images per dataset |
