# Stage 5 — Object detection

[← Back to README](../README.md)

Tests whether the same round-based net-score verifier mechanism validated on
classification ([stage4_verifier.md](stage4_verifier.md)) generalizes to a
spatial task — localizing objects with bounding boxes — instead of free-text
classification. Two conditions, directly comparable at an equal image/round
budget:

- **Detection-Baseline** (`src/detection_baseline.py`) — single turn, prints
  bounding boxes directly with no intermediate reasoning.
- **Detection-Verifier** (`src/detection_verifier.py`) — adapts
  `verifier_cot.py`'s round loop: each round the model first answers a
  structured localization rubric (object/class, instance count, location,
  occlusion, size — mirroring Stage 4's disease/pest rubric shape) *before*
  printing boxes, a PPT tournament ranks K candidates each round (same
  `src/common/ppt_scoring.py` mechanism, unmodified), and a persistent
  per-cluster net win/loss score decides the reported answer each round.
  Candidate/cluster identity is resolved geometrically (IoU overlap) instead
  of via an LLM "same answer?" call, since box-set identity is arithmetic,
  not a language judgment.

Status: **in progress** — this doc is updated as each sub-stage lands. See
the implementation plan for full design rationale and the exact file list.

## Sub-stages

| Sub-stage | Script | Status | Output |
|---|---|---|---|
| 5a (prep) | `scripts/agml_hf/catalog_detection_datasets.py` | built, validated on a 3-dataset sample (confirmed real per-dataset variation: pixel vs. normalized coords, `category` vs. `categories` schema key) — **full 64-dataset run not yet done, run it remotely** | `src/config/hf_detection_catalog.json` |
| 5a (prep) | `scripts/agml_hf/migrate_detection_datasets_yaml.py` | built, not yet run against full catalog | `src/config/detection_datasets.yaml` |
| 5a | `scripts/agml_hf/download_and_sample_detection.py` | built, not yet run | `dataset/selected_images_detection.json`, cached images + per-image annotation sidecars under `dataset/images_detection/` |
| 5b | `src/detection_baseline.py` | built, syntax/import-verified — **no GPU run yet** | `outputs_final/detection_baseline/<model>/ckpt_<model>.csv` |
| 5c | `src/detection_qc_plot.py` | built and verified end-to-end on synthetic fixtures | `outputs_final/detection_baseline/<model>/qc_plots/<dataset>/<image>.png` |
| 5d | `src/detection_verifier.py` | built and verified end-to-end on a mocked round loop (geometric cluster resolution, leader_margin, lock_after_n_defends, per-round IoU all confirmed against synthetic candidates) — **no GPU run yet** | `outputs_final/detection_verifier/<model>/ckpt_<model>.csv` (+ `verifier_log.jsonl` per-round call transparency, same convention as `verifier_cot.py`) |

Shared modules (no GPU, unit-tested against synthetic data): `src/common/bbox_parse.py` (`extract_boxes`/`clamp_box`), `src/common/bbox_metrics.py` (`iou`/`match_boxes`/`precision_recall_f1`), `src/common/bbox_viz.py` (`draw_boxes`/`plot_predictions`).

**Note on local development**: all of the above was validated with synthetic fixtures and mocked generation (no real HF downloads, no real model inference) — this local environment is for code editing only. Every stage that touches real data or a GPU (the full 5a catalog run, 5a downloads, 5b/5d inference) needs to be run remotely.

## Step 1 — catalog + configure detection datasets (5a prep)

```bash
uv run python main.py catalog-detection-datasets \
  --excluded_catalog src/config/hf_catalog_excluded.json \
  --output src/config/hf_detection_catalog.json

uv run python main.py migrate-detection-datasets \
  --catalog src/config/hf_detection_catalog.json \
  --output src/config/detection_datasets.yaml
```

`catalog-detection-datasets` resolves `category_id → class_name` per dataset
(tries the HF `Features` schema first, then the dataset card, then a sample
row) and empirically measures each dataset's bounding-box coordinate
convention (pixel `[x,y,w,h]`, normalized, xyxy — printed per dataset, not
assumed). Datasets where names can't be resolved are marked `skip: true` in
`detection_datasets.yaml`, same convention Stage 0a uses for datasets needing
manual review — nothing is silently dropped, it's inspectable in the yaml.

## Step 2 — download + sample (5a)

```bash
uv run python main.py download-detection-datasets \
  --config src/config/detection_datasets.yaml \
  --output dataset/selected_images_detection.json
```

Per-image annotations (boxes + category ids/names) are cached as a JSON
sidecar next to each downloaded image, so the manifest + cache together are a
complete, inspectable local copy of everything downstream reads.

## Step 3 — baseline inference

```bash
uv run python main.py detection-baseline --model google/gemma-4-E4B-it
```

## Step 4 — QC: plot predicted vs. ground-truth boxes

```bash
uv run python main.py detection-qc-plot \
  --ckpt outputs_final/detection_baseline/google_gemma-4-E4B-it/ckpt_google_gemma-4-E4B-it.csv \
  --n_per_dataset 3
```

Run this immediately after the first baseline checkpoint rows exist, on a
handful of images, before trusting anything at scale — it's the only way to
visually catch a bbox-coordinate-alignment bug (predicted boxes plotted
against the correct-resolution image; if they're systematically offset, the
model's coordinates are relative to an internally-rescaled view and need a
rescale-back fix in `src/common/bbox_parse.py`).

## Step 5 — verifier

```bash
uv run python main.py detection-verifier --model google/gemma-4-E4B-it
```

Emits the same round-level diagnostic shape as `verifier_cot.py`
(`round`, `n_clusters`, `pairwise_winner`, mean IoU vs. ground truth per
round) so a `detection_round1_vs_last.png`-style comparison chart can reuse
`verifier_summary.py`'s existing plotting pattern once real data exists.
