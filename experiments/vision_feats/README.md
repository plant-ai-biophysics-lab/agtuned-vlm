# O1 Vision Feature Analysis Pipeline

A diagnostic suite for comparing vision feature representations across three model families on agricultural images. Built to support the O1 objective: *Extract and analyze vision features to assess whether agriculturally relevant concepts are accurately represented within the model's visual encoder.*

## Overview

Three separate scripts extract vision features from three model types, then a fourth script merges results into a unified comparison.

| Script | Model | What it extracts |
|---|---|---|
| `01_dinov2_features.py` | DINOv2-large (self-supervised) | CLS self-attention + query-patch cosine similarity |
| `02_inatag_features.py` | iNatAg Swin (AgML fine-tuned) | Penultimate features + GradCAM heatmap |
| `03_gemma4_features.py` | Gemma 4 vision encoder (VLM) | Raw SigLIP patch features (O1) + V↔T projected alignment (O2 preview) |
| `04_compare_models.py` | — | Cross-model unified comparison figure |

Each script runs independently and writes:
- A PNG figure with three panels (input image + two visualization variants)
- A `*_metrics.txt` file with quantitative measurements

---

## Setup (one-time)

```
### 1. HuggingFace authentication (required for Gemma 4 only)

Gemma 4 is a gated model. Authenticate once:

```bash
huggingface-cli login
```

Paste your HF access token when prompted.

### 2. Download the iNatAg checkpoint (required for script 02 only)

Pick one from [Project-AgML/iNatAg-models](https://huggingface.co/Project-AgML/iNatAg-models/tree/main). Recommended for fair feature analysis:

- `swin_large_without_latlon_with_lora.pth` — Swin-Large + LoRA, no lat/lon fusion (cleanest pure-vision representation)
- `swin_large_with_lora.pth` — full model with lat/lon head (compare against the above as an ablation)

```bash
# Example
wget https://huggingface.co/Project-AgML/iNatAg-models/resolve/main/swin_large_without_latlon_with_lora.pth
```

---

## Script 1 — DINOv2 features

**What it does.** Loads DINOv2-large (`facebook/dinov2-large`) and produces:
- CLS self-attention map (where the [CLS] token attends across patches in the last layer)
- Query-patch cosine similarity map (similarity from a user-specified point to all other patches)
- Quantitative metrics: attention entropy, mean pairwise patch cosine similarity, std of patch L2 norms

**Why DINOv2.** Self-supervised pretraining on natural images. Tests H1 — does the encoder represent agricultural features without any task-specific finetuning?

### Basic usage

```bash
python 01_dinov2_features.py \
    --image_path /path/to/leaf.jpg \
    --label "leaf rust" \
    --query_point 0.5 0.4 \
    --output_dir ./outputs/dinov2
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image_path` | yes | — | Path to input image (any common format: jpg, png, etc.) |
| `--label` | yes | — | Target concept label. Used only for figure titles and filenames. DINOv2 doesn't use text. |
| `--query_point` | no | `0.5 0.5` | Two floats `X Y` in [0, 1] — normalized image coordinates. The patch nearest to this point becomes the query for the cosine-similarity map. Example: `0.3 0.7` is left-of-center, near the bottom. |
| `--output_dir` | no | `./outputs/dinov2` | Where to write the figure and metrics |
| `--model_name` | no | `facebook/dinov2-large` | HuggingFace model ID. Can swap to `facebook/dinov2-base`, `facebook/dinov2-giant`, etc. |

### Outputs

```
./outputs/dinov2/
├── leaf_dinov2.png          # 3-panel figure: input+query | CLS attention | query similarity
└── leaf_dinov2_metrics.txt  # numerical metrics
```

### How to choose a query point

The query point should be near the visual feature you care about. Examples:
- For a leaf with a disease spot in the center: `--query_point 0.5 0.5`
- For an insect on the lower-left of a leaf: `--query_point 0.3 0.7`
- For a fruit at the top-right: `--query_point 0.75 0.25`

The cosine similarity map will light up regions whose patch features are similar to the query patch — telling you whether DINOv2 perceives that region as visually distinctive.

---

## Script 2 — iNatAg Swin features

**What it does.** Loads the iNatAg Swin-Large checkpoint and produces:
- GradCAM heatmap targeting either the predicted class or a user-specified class
- Penultimate-layer feature vector statistics
- Quantitative metrics: feature L2 norm, sparsity, GradCAM localization concentration, prediction confidence

**Why iNatAg.** Fully supervised on iNaturalist agricultural species. Represents the "best case" — what does a vision encoder look like when it's been trained directly on the target domain? This is the upper bound for vision feature quality.

### Basic usage

```bash
python 02_inatag_features.py \
    --image_path /path/to/leaf.jpg \
    --label "corn smut" \
    --checkpoint /path/to/swin_large_without_latlon_with_lora.pth \
    --no_latlon \
    --output_dir ./outputs/inatag
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image_path` | yes | — | Path to input image |
| `--label` | yes | — | Target concept label (for figure titles) |
| `--checkpoint` | yes | — | Path to downloaded `.pth` file |
| `--num_classes` | no | auto-inferred | Number of species classes the model was trained on. Auto-detected from the checkpoint's `fc_species.weight` shape. |
| `--class_idx` | no | `0` (uses argmax) | Class index for GradCAM target. If left as `0`, the script uses the model's predicted class. Set explicitly to compute CAM for a non-predicted class. |
| `--no_lora` | no | off | Disable LoRA wrapping. Use for `*_no_lora.pth` checkpoints. |
| `--no_latlon` | no | off | Use the without-latlon model variant. **Required for `*_without_latlon_*` checkpoints.** |
| `--output_dir` | no | `./outputs/inatag` | Output directory |

### Choosing the right flag combination

The iNatAg repo provides four checkpoint variants. Match the flags accordingly:

| Checkpoint name | `--no_latlon`? | `--no_lora`? |
|---|---|---|
| `swin_large_with_lora.pth` | no | no |
| `swin_large_without_lora.pth` | no | yes |
| `swin_large_without_latlon_with_lora.pth` | yes | no |
| `swin_large_without_latlon_without_lora.pth` | yes | yes |

For pure vision feature analysis, **prefer the `without_latlon` variants** — they remove the lat/lon fusion head, so the feature vector is purely from Swin without any geographic conditioning.

### Outputs

```
./outputs/inatag/
├── leaf_inatag.png          # 3-panel figure: input | GradCAM raw | GradCAM overlay
└── leaf_inatag_metrics.txt
```

The script will print the predicted class index and confidence — useful for sanity-checking whether the model recognized the input.

---

## Script 3 — Gemma 4 vision encoder features

**What it does.** Loads Gemma 4 (`AutoModelForMultimodalLM`) and produces **two figures**:

1. **O1 — Pre-projector figure** (`*_o1_preprojector.png`): Raw SigLIP patch activations. Comparable to DINOv2 and iNatAg outputs.
2. **O2 — Post-projector figure** (`*_o2_postprojector.png`): Cosine similarity between projected patch embeddings and the label's text embedding. This measures vision-language alignment — a preview of O2.

**Why Gemma 4.** A modern open-source VLM. Tests whether the vision encoder of a multimodal model captures agricultural features (H1) and whether the projector preserves the alignment to text (H3).

### Basic usage (A100 80GB — your setup)

```bash
python 03_gemma4_features.py \
    --image_path /path/to/leaf.jpg \
    --label "leaf rust" \
    --output_dir ./outputs/gemma4
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image_path` | yes | — | Path to input image |
| `--label` | yes | — | Target concept label. **Used as the text token for V↔T similarity computation.** Choose carefully — this affects the O2 alignment map. |
| `--output_dir` | no | `./outputs/gemma4` | Output directory |
| `--model_name` | no | `google/gemma-4-E4B-it` | HF model ID. One arg switches all Gemma 4 variants (see table below). |
| `--use_4bit` | no | off | Enable 4-bit NF4 quantization. Use on A100 40GB. |
| `--offload_lm` | no | off | 4-bit NF4 + offload LLM decoder to CPU. Use on RTX 3060 / 12 GB VRAM. |

`--use_4bit` and `--offload_lm` are mutually exclusive. The default (no flag) loads the model in bfloat16.

### Hardware-specific commands

| Hardware | Command |
|---|---|
| A100 80GB | `python 03_gemma4_features.py --image_path img.jpg --label "leaf rust"` |
| A100 40GB | `python 03_gemma4_features.py --image_path img.jpg --label "leaf rust" --use_4bit` |
| RTX 3060 12GB | `python 03_gemma4_features.py --image_path img.jpg --label "leaf rust" --use_4bit --offload_lm` |

### Switching Gemma 4 variants

| Model | Approx VRAM (bf16) | Recommended on A100 80GB |
|---|---|---|
| `google/gemma-4-E2B-it` | ~5 GB | ✅ Default flags |
| `google/gemma-4-E4B-it` | ~10 GB | ✅ Default flags (current default) |
| `google/gemma-4-26B-A4B-it` | ~52 GB | Add `--use_4bit` |
| `google/gemma-4-31B-it` | ~62 GB | Add `--use_4bit` |

To switch, just change the `--model_name` argument:

```bash
python 03_gemma4_features.py \
    --image_path img.jpg --label "leaf rust" \
    --model_name google/gemma-4-31B-it \
    --use_4bit
```

### Outputs

```
./outputs/gemma4/
├── leaf_gemma4_o1_preprojector.png   # raw SigLIP patch features
├── leaf_gemma4_o2_postprojector.png  # V↔T cosine similarity map
└── leaf_gemma4_metrics.txt           # both O1 and O2 metrics, labeled with comments
```

### Choosing a label

The label is more meaningful here than for DINOv2 because it's actually fed into the model. Tips:
- Use specific terms ("powdery mildew on cucumber leaves") rather than vague ones ("plant disease") — the V↔T similarity map will be more focused.
- Try multiple labels for the same image to see which the encoder aligns to best.

---

## Script 4 — Cross-model comparison

**What it does.** Reads the metrics and figures produced by scripts 01–03 and assembles a unified comparison view: per-model figures stacked vertically, plus harmonized bar charts of *spatial focus* and *feature diversity* across all three models, plus a raw metrics table.

**Run this AFTER scripts 01, 02, and 03 have all completed for the same image.**

### Basic usage

```bash
python 04_compare_models.py \
    --image_path /path/to/leaf.jpg \
    --label "leaf rust" \
    --dinov2_dir ./outputs/dinov2 \
    --inatag_dir ./outputs/inatag \
    --gemma4_dir ./outputs/gemma4 \
    --output_dir ./outputs/comparison
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image_path` | yes | — | Original input image (for reference in the figure title) |
| `--label` | yes | — | Target concept label. **Must match what was passed to scripts 01–03.** Used to find/name files. |
| `--dinov2_dir` | yes | — | Directory containing script 01's output |
| `--inatag_dir` | yes | — | Directory containing script 02's output |
| `--gemma4_dir` | yes | — | Directory containing script 03's output |
| `--output_dir` | no | `./outputs/comparison` | Where to write the combined figure |

### Outputs

```
./outputs/comparison/
├── comparison_leaf_rust.png    # unified 3-row comparison figure
└── summary_leaf_rust.csv       # CSV of harmonized metrics
```

### How metrics are harmonized

Models use very different feature types, so direct numerical comparison is tricky. The comparison script normalizes everything to two interpretable scores in `[0, 1]`:

- **Spatial focus** — how concentrated/peaked the spatial activation is.
  - DINOv2: `1 - normalized(attention entropy)`
  - iNatAg: GradCAM localization concentration (top-20% energy fraction)
  - Gemma 4: `1 - normalized(patch embedding entropy)` — uses O1 metric for fairness, *not* the V↔T similarity
- **Feature diversity** — how distinct patch features are across the image.
  - DINOv2 / Gemma 4: `1 - mean pairwise cosine similarity`
  - iNatAg: `1 - feature sparsity`

The Gemma 4 vision↔text similarity (O2) is reported separately in the metrics table since it has no analog in DINOv2 or iNatAg.

---

## End-to-end example

```bash
# 1. Activate environment
conda activate feat_analysis

# 2. Run all three feature extractors on the same image
IMG="./images/corn_leaf_disease.jpg"
LABEL="northern corn leaf blight"

python 01_dinov2_features.py \
    --image_path "$IMG" --label "$LABEL" \
    --query_point 0.5 0.5 \
    --output_dir ./outputs/dinov2

python 02_inatag_features.py \
    --image_path "$IMG" --label "$LABEL" \
    --checkpoint ./checkpoints/swin_large_without_latlon_with_lora.pth \
    --no_latlon \
    --output_dir ./outputs/inatag

python 03_gemma4_features.py \
    --image_path "$IMG" --label "$LABEL" \
    --output_dir ./outputs/gemma4

# 3. Build the unified comparison
python 04_compare_models.py \
    --image_path "$IMG" --label "$LABEL" \
    --dinov2_dir ./outputs/dinov2 \
    --inatag_dir ./outputs/inatag \
    --gemma4_dir ./outputs/gemma4 \
    --output_dir ./outputs/comparison
```

For SLURM submission on UC Davis FARM, see `submit_jobs.sh`.

---

## Troubleshooting

**`Model returned no attention tensors` (script 01)**
The DINOv2 model loaded with SDPA attention instead of eager. The script forces `attn_implementation="eager"` — if you see this error, your transformers version may need upgrading: `pip install --upgrade transformers`.

**`fc_species.weight not found in checkpoint` (script 02)**
The checkpoint may be a raw Swin checkpoint without the iNatAg fine-tuning heads. Make sure you downloaded one of the four files from the iNatAg repo (not the base Swin model).

**`Expected submodule 'vision_tower' not found` (script 03)**
The Gemma 4 model architecture deviates from the Gemma 3 convention this script assumes. Print the available submodules from the error message and update the attribute names in the script (`vision_tower`, `multi_modal_projector`, `language_model`).

**Out of memory on Gemma 4**
Add `--use_4bit` (saves ~50% VRAM). If still OOM, add `--offload_lm` (puts LLM decoder on CPU — only vision tower and projector use GPU). The LLM is never executed in this script, so CPU offload has zero runtime cost.

**Comparison script can't find figures**
The image stem in the figure filename comes from the input image's basename. If you ran scripts 01–03 with different `--image_path` values for the same logical image, the comparison script will pick the most recent figure of each type. Re-run with consistent paths if needed.
