"""
H3: Vision-Text Alignment — Attention Map vs Bounding Box Analysis
==================================================================
Tests whether the model attends to the correct image regions when processing
agricultural content, using ground-truth bounding boxes as spatial reference.

Method:
  1. Feed each image through the full Gemma4 model.
  2. Extract attention weights from the last N language model layers —
     specifically, how much the last prompt token attends to each image token.
  3. Reshape the flat image-token attention vector into a spatial grid and
     upsample to the original image size.
  4. Compare the resulting heatmap against ground-truth bounding boxes.

Metrics:
  - Pointing game accuracy : is the argmax of the heatmap inside a bbox?
  - Energy inside box      : fraction of total attention inside bboxes

If H1 and H2 pass but pointing game accuracy is low → H3 confirmed:
the encoder represents the image fine, the language model knows the concepts,
but they don't share the same spatial embedding space.

Dataset: wheat_head_counting (AgML, COCO format bboxes [x, y, w, h])

Usage:
    python h3_alignment.py --model google/gemma-4-E2B-it
    python h3_alignment.py --model /path/to/finetuned --n_samples 50
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, cast

import agml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.ndimage import zoom as sp_zoom
from transformers import AutoProcessor, AutoModelForImageTextToText

PROMPT = "Identify all wheat heads visible in this image."


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_od_samples(dataset_name: str, n_samples: int | None = None) -> list[dict]:
    loader = agml.data.AgMLDataLoader(dataset_name)
    loader.split(train=0.8, val=0.1, test=0.1)
    contents = cast(Dict[str, Any], loader.export_contents())

    samples = []
    for path, annots in contents.items():
        # COCO format: list of {'bbox': [x,y,w,h], 'category_id': ..., ...}
        if isinstance(annots, list):
            bboxes = [a["bbox"] for a in annots if "bbox" in a]
        elif isinstance(annots, dict):
            bboxes = annots.get("bboxes", annots.get("bbox", []))
        else:
            bboxes = []

        if not bboxes:
            continue

        samples.append({"path": str(path), "bboxes": bboxes})
        if n_samples and len(samples) >= n_samples:
            break

    return samples


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    local = os.path.abspath(model_id)
    if os.path.exists(local):
        model_id = local
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n_img_tokens = getattr(model.config, "vision_soft_tokens_per_image", None)
    img_token_id = getattr(model.config, "image_token_id", None)
    print(f"  image_token_id          : {img_token_id}")
    print(f"  vision_soft_tokens/image: {n_img_tokens}")
    return processor, model, img_token_id, n_img_tokens


def build_prompt_text(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Attention extraction ──────────────────────────────────────────────────────

def find_image_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return positions


def extract_attention_heatmap(
    model,
    inputs: dict,
    img_positions: torch.Tensor,
    n_layers: int = 4,
) -> np.ndarray:
    """
    Run a forward pass and return mean attention from the last text token
    to each image token position, averaged over the last n_layers and all heads.
    Returns a 1D float32 numpy array of length len(img_positions), normalized [0,1].
    """
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise RuntimeError(
            "output_attentions=True returned no attention weights. "
            "The model may not support this kwarg at the top level."
        )

    # Each element of outputs.attentions: (1, n_heads, seq_len, seq_len)
    maps = []
    for layer_attn in outputs.attentions[-n_layers:]:
        # Attention from the last token (end of prompt) to all positions
        last_tok = layer_attn[0, :, -1, :]           # (n_heads, seq_len)
        img_attn = last_tok[:, img_positions]         # (n_heads, n_img_tokens)
        maps.append(img_attn.float().mean(dim=0))     # (n_img_tokens,)

    flat = torch.stack(maps).mean(dim=0).cpu().numpy()
    flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
    return flat


# ── Spatial upsampling ────────────────────────────────────────────────────────

def flat_to_image(flat: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Reshape flat patch attention to a spatial grid that matches the image
    aspect ratio, then bilinearly upsample to (img_h, img_w).
    """
    n = len(flat)

    # Find the rectangular grid whose aspect ratio best matches the image
    best_rows, best_cols, best_diff = 1, n, float("inf")
    for r in range(1, int(np.sqrt(n)) + 2):
        if n % r == 0:
            c = n // r
            diff = abs(r / c - img_h / img_w)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = r, c
        # Also try ceiling division
        c = int(np.ceil(n / r))
        if c * r >= n:
            diff = abs(r / c - img_h / img_w)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = r, c

    padded = np.zeros(best_rows * best_cols)
    padded[:n] = flat
    grid = padded.reshape(best_rows, best_cols)

    zoomed = sp_zoom(grid, (img_h / best_rows, img_w / best_cols), order=1)
    return zoomed.astype(np.float32)


# ── Bounding box helpers ──────────────────────────────────────────────────────

def bboxes_to_mask(bboxes: list, img_h: int, img_w: int) -> np.ndarray:
    """Union binary mask of all bboxes. Bboxes assumed [x, y, w, h] pixels."""
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x, y, w, h = (float(v) for v in bbox[:4])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
        mask[y1:y2, x1:x2] = 1.0
    return mask


def pointing_game(heatmap: np.ndarray, bbox_mask: np.ndarray) -> bool:
    peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return bool(bbox_mask[peak] > 0)


def energy_inside_box(heatmap: np.ndarray, bbox_mask: np.ndarray) -> float:
    total = heatmap.sum()
    if total < 1e-8:
        return 0.0
    return float((heatmap * bbox_mask).sum() / total)


# ── Visualization ─────────────────────────────────────────────────────────────

def save_overlay(
    image: PILImage.Image,
    heatmap: np.ndarray,
    bboxes: list,
    path: Path,
    title: str = "",
):
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def draw_boxes(ax):
        for bbox in bboxes:
            if len(bbox) < 4:
                continue
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            rect = mpatches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    axes[0].imshow(img_np)
    draw_boxes(axes[0])
    axes[0].set_title(f"Ground Truth ({len(bboxes)} boxes)")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    draw_boxes(axes[2])
    axes[2].set_title(f"Overlay — {title}")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    pg_scores: list[int],
    energy_scores: list[float],
    model_label: str,
    output_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_total = len(pg_scores)
    n_correct = sum(pg_scores)
    axes[0].bar(
        ["Correct", "Incorrect"],
        [n_correct, n_total - n_correct],
        color=["#2ecc71", "#e74c3c"],
    )
    for i, v in enumerate([n_correct, n_total - n_correct]):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=10)
    axes[0].set_title(f"Pointing Game Accuracy = {n_correct / n_total:.3f}")
    axes[0].set_ylabel("Count")

    axes[1].hist(energy_scores, bins=20, color="steelblue", edgecolor="white")
    axes[1].axvline(
        np.mean(energy_scores),
        color="red",
        linestyle="--",
        label=f"mean = {np.mean(energy_scores):.3f}",
    )
    axes[1].set_title("Energy Inside Bounding Boxes")
    axes[1].set_xlabel("Fraction of attention inside boxes")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle(f"H3 Alignment — {Path(model_label).name}", fontsize=11)
    plt.tight_layout()
    path = output_dir / "h3_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading AgML dataset: {args.agml_dataset}")
    samples = load_od_samples(args.agml_dataset, args.n_samples)
    print(f"  {len(samples)} samples with bounding boxes")

    processor, model, img_token_id, n_img_tokens = load_model(args.model)

    if img_token_id is None:
        raise ValueError(
            "model.config.image_token_id not found. "
            "Check model config for the correct attribute name."
        )

    prompt_text = build_prompt_text(processor)
    lm_device = next(model.parameters()).device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    pg_scores: list[int] = []
    energy_scores: list[float] = []
    per_sample: list[dict] = []

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {Path(sample['path']).name}")

        image = PILImage.open(sample["path"]).convert("RGB")
        img_w, img_h = image.size
        bboxes = sample["bboxes"]

        # Prepare inputs
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to(lm_device) for k, v in inputs.items()}

        # Find image token positions
        img_positions = find_image_positions(inputs["input_ids"], img_token_id)
        if len(img_positions) == 0:
            print(f"  WARNING: No image tokens found in input_ids. Skipping.")
            print(f"    input_ids unique: {inputs['input_ids'].unique().tolist()[:10]}")
            continue

        print(f"  seq_len={inputs['input_ids'].shape[1]} | img_tokens={len(img_positions)}")

        # Extract attention heatmap
        try:
            flat = extract_attention_heatmap(
                model, inputs, img_positions, n_layers=args.n_layers
            )
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            continue

        # Upsample to image size
        heatmap = flat_to_image(flat, img_h, img_w)

        # Metrics
        bbox_mask = bboxes_to_mask(bboxes, img_h, img_w)
        pg = pointing_game(heatmap, bbox_mask)
        energy = energy_inside_box(heatmap, bbox_mask)

        pg_scores.append(int(pg))
        energy_scores.append(energy)

        print(f"  Pointing game : {'✓' if pg else '✗'}  |  Energy inside: {energy:.4f}")

        per_sample.append({
            "image": sample["path"],
            "n_bboxes": len(bboxes),
            "n_img_tokens": len(img_positions),
            "pointing_game": int(pg),
            "energy_inside_box": round(energy, 4),
        })

        # Save overlays for first n_overlays samples
        if i < args.n_overlays:
            save_overlay(
                image,
                heatmap,
                bboxes,
                path=overlay_dir / f"overlay_{i:03d}.png",
                title=f"PG={'✓' if pg else '✗'}  E={energy:.2f}",
            )

        # Free memory between samples
        del inputs, flat, heatmap, bbox_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_sample:
        print("\nNo valid samples processed.")
        return

    pg_acc = float(np.mean(pg_scores))
    mean_energy = float(np.mean(energy_scores))

    print(f"\n=== H3 Results ({args.agml_dataset}) ===")
    print(f"  Samples evaluated    : {len(per_sample)}")
    print(f"  Pointing game acc    : {pg_acc:.4f}")
    print(f"  Mean energy inside   : {mean_energy:.4f}")

    summary = {
        "model": args.model,
        "dataset": args.agml_dataset,
        "n_samples": len(per_sample),
        "n_layers_averaged": args.n_layers,
        "pointing_game_accuracy": round(pg_acc, 4),
        "mean_energy_inside_box": round(mean_energy, 4),
        "per_sample": per_sample,
    }

    result_path = output_dir / "h3_results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {result_path}")

    print("\n=== Generating plots ===")
    plot_summary(pg_scores, energy_scores, args.model, output_dir)
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="Gemma4 model (HF ID or local path to fine-tuned)")
    parser.add_argument("--agml_dataset", type=str, default="wheat_head_counting",
                        help="AgML object detection dataset name")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of last LM attention layers to average (default: 4)")
    parser.add_argument("--n_overlays", type=int, default=20,
                        help="Number of overlay images to save (default: 20)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
