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

# ── Dataset ───────────────────────────────────────────────────────────────────

def _clean_label(name: str) -> str:
    return name.replace("___", " ").replace("_", " ").strip()


def _resolve_class(cat_id: int, class_names: list[str]) -> str | None:
    """Handle both 0-indexed and 1-indexed COCO category_ids."""
    if 0 <= cat_id < len(class_names):
        return class_names[cat_id]
    if 0 <= cat_id - 1 < len(class_names):
        return class_names[cat_id - 1]
    return None


def load_od_samples(dataset_name: str, n_samples: int | None = None) -> list[dict]:
    loader = agml.data.AgMLDataLoader(dataset_name)
    loader.split(train=0.8, val=0.1, test=0.1)
    contents = cast(Dict[str, Any], loader.export_contents())
    class_names = cast(list[str], loader.classes)

    samples = []
    for path, annots in contents.items():
        # COCO format: list of {'bbox': [x,y,w,h], 'category_id': ..., ...}
        if isinstance(annots, list):
            bboxes = [a["bbox"] for a in annots if "bbox" in a]
            cat_ids = [a["category_id"] for a in annots if "category_id" in a]
        elif isinstance(annots, dict):
            bboxes = annots.get("bboxes", annots.get("bbox", []))
            cat_ids = annots.get("category_ids", [])
        else:
            bboxes = []
            cat_ids = []

        if not bboxes:
            continue

        unique_cats = sorted(set(cat_ids))
        img_classes = [
            _clean_label(n)
            for c in unique_cats
            if (n := _resolve_class(c, class_names)) is not None
        ]

        samples.append({"path": str(path), "bboxes": bboxes, "classes": img_classes})
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
        model_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()
    n_img_tokens = getattr(model.config, "vision_soft_tokens_per_image", None)
    img_token_id = getattr(model.config, "image_token_id", None)
    print(f"  image_token_id          : {img_token_id}")
    print(f"  vision_soft_tokens/image: {n_img_tokens}")
    return processor, model, img_token_id, n_img_tokens


def build_prompt_text(processor, class_label: str) -> str:
    prompt = f"Identify all {class_label} visible in this image."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Attention extraction ──────────────────────────────────────────────────────

def find_image_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return positions


def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _layer_attn_to_flat(attentions, positions, token_positions, n_layers) -> np.ndarray:
    """Average attention from token_positions → image positions over last n_layers."""
    maps = []
    for layer_attn in attentions[-n_layers:]:
        # layer_attn: (1, n_heads, seq_len, seq_len)
        tok_attn = layer_attn[0, :, token_positions, :]   # (n_heads, n_toks, seq_len)
        img_attn = tok_attn[:, :, positions]              # (n_heads, n_toks, n_img)
        maps.append(img_attn.float().mean(dim=(0, 1)))    # (n_img,)
    flat = torch.stack(maps).mean(dim=0).cpu().numpy()
    return _normalize(flat)


def all_prompt_tokens_to_flat(
    attentions, img_positions: torch.Tensor, input_len: int, n_layers: int
) -> np.ndarray:
    """Average attention from all text prompt tokens → image positions.
    Only uses tokens that come after the last image token (causal: these are
    the only tokens that can actually attend to all image positions)."""
    first_text_pos = int(img_positions.max().item()) + 1
    device = img_positions.device
    text_positions = torch.arange(first_text_pos, input_len, device=device)
    if len(text_positions) == 0:
        text_positions = torch.tensor([input_len - 1], device=device)
    return _layer_attn_to_flat(attentions, img_positions, text_positions, n_layers)


def attention_rollout_to_flat(
    attentions, img_positions: torch.Tensor, query_pos: int
) -> np.ndarray:
    """Propagate attention signal across all layers via matrix multiplication.
    Each layer's attention (+ residual) is multiplied into a running rollout
    matrix so attention from early layers reaches the final query position."""
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    rollout = torch.eye(seq_len, dtype=torch.float32, device=device)

    for layer_attn in attentions:
        attn = layer_attn[0].float().mean(dim=0)                    # (seq_len, seq_len)
        attn = attn + torch.eye(seq_len, device=device)             # residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)                # re-normalize rows
        rollout = attn @ rollout

    img_attn = rollout[query_pos, img_positions].cpu().numpy()
    return _normalize(img_attn)


def compute_gradcam(
    model,
    mm_root,
    full_inputs: dict,
    class_positions: list[int],
    input_len: int,
) -> np.ndarray | None:
    """GradCAM: gradient of class token logits w.r.t. last SigLIP encoder layer.
    Returns a flat array with one value per vision encoder patch (pre-projector)."""
    vt = mm_root.vision_tower
    if hasattr(vt, "encoder") and hasattr(vt.encoder, "layers"):
        target_layer = vt.encoder.layers[-1]
    elif hasattr(vt, "vision_model") and hasattr(vt.vision_model, "encoder"):
        target_layer = vt.vision_model.encoder.layers[-1]
    else:
        print("  GradCAM: could not locate last vision encoder layer — skipping")
        return None

    saved_act: list[torch.Tensor] = []

    def fwd_hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.retain_grad()
        saved_act.append(t)

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        # Set requires_grad on pixel_values to anchor the computation graph
        # so intermediate activations accumulate gradients on a frozen model.
        mod_inputs = {
            k: (v.detach().requires_grad_(True) if k == "pixel_values" else v.detach())
            for k, v in full_inputs.items()
        }
        with torch.enable_grad():
            outputs = model(**mod_inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            if class_positions:
                target = torch.stack([
                    logits[0, pos - 1].max()
                    for pos in class_positions if pos > 0
                ]).sum()
            else:
                target = logits[0, input_len - 1].max()
            target.backward()
    except Exception as e:
        print(f"  GradCAM: failed ({e}) — skipping")
        return None
    finally:
        handle.remove()

    if not saved_act or saved_act[0].grad is None:
        print("  GradCAM: no gradient captured — skipping")
        return None

    act = saved_act[0].float()         # (1, n_tokens, hidden_dim)
    grad = saved_act[0].grad.float()

    if act.dim() == 3:
        act, grad = act[0], grad[0]    # (n_tokens, hidden_dim)

    # Drop CLS token (index 0) — keep patch tokens only
    act, grad = act[1:], grad[1:]      # (n_patches, hidden_dim)

    cam = torch.relu((grad * act).sum(dim=-1)).cpu().numpy()  # (n_patches,)
    return _normalize(cam)


def find_class_token_positions(
    generated_ids: torch.Tensor, input_len: int, tokenizer, class_words: list[str]
) -> list[int]:
    """Return sequence positions of class word tokens in the generated portion."""
    positions = []
    gen_part = generated_ids[0, input_len:]
    for word in class_words:
        for variant in (word, f" {word}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tok_id in ids:
                hits = (gen_part == tok_id).nonzero(as_tuple=True)[0]
                positions += [input_len + h.item() for h in hits]
    return sorted(set(positions))


def extract_combined_heatmaps(
    model,
    processor,
    inputs: dict,
    img_positions: torch.Tensor,
    mm_root,
    class_words: list[str],
    n_layers: int = 4,
    max_new_tokens: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """
    1. Generate a response.
    2. Single forward pass on (input + response) with output_attentions=True.
    3. Separate GradCAM pass with gradients enabled.
    4. Return:
       - prompt_flat    : attention from last prompt token → image positions
       - allprompt_flat : attention averaged over all text prompt tokens → image positions
       - rollout_flat   : attention rollout across all layers → image positions
       - gradcam_flat   : GradCAM from last SigLIP encoder layer (None if failed)
       - semantic_flat  : attention from class word tokens → image positions (None if not found)
       - response_text  : decoded generated response
    """
    input_len = inputs["input_ids"].shape[1]

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    tokenizer = processor.tokenizer
    response_text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    # Build full-sequence inputs (input + generated tokens)
    full_inputs = dict(inputs)
    full_inputs["input_ids"] = generated_ids
    if "attention_mask" in full_inputs:
        extra = torch.ones(
            1, generated_ids.shape[1] - input_len,
            device=full_inputs["attention_mask"].device,
            dtype=full_inputs["attention_mask"].dtype,
        )
        full_inputs["attention_mask"] = torch.cat(
            [full_inputs["attention_mask"], extra], dim=1
        )

    # Single forward pass for both heatmaps
    with torch.no_grad():
        outputs = model(**full_inputs, output_attentions=True)

    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise RuntimeError(
            "output_attentions=True returned no attention weights. "
            "Load the model with attn_implementation='eager'."
        )

    # Prompt-token heatmap: attention from last prompt token
    prompt_flat = _layer_attn_to_flat(
        outputs.attentions, img_positions,
        token_positions=torch.tensor([input_len - 1], device=img_positions.device),
        n_layers=n_layers,
    )

    # Semantic heatmap: attention from class word tokens in generated response
    class_positions = find_class_token_positions(generated_ids, input_len, tokenizer, class_words)
    if class_positions:
        class_pos_tensor = torch.tensor(class_positions, device=img_positions.device)
        semantic_flat = _layer_attn_to_flat(
            outputs.attentions, img_positions,
            token_positions=class_pos_tensor,
            n_layers=n_layers,
        )
    else:
        semantic_flat = None

    # All-prompt-tokens heatmap and attention rollout
    allprompt_flat = all_prompt_tokens_to_flat(
        outputs.attentions, img_positions, input_len, n_layers
    )
    rollout_flat = attention_rollout_to_flat(
        outputs.attentions, img_positions, query_pos=input_len - 1
    )

    # GradCAM — separate forward pass with gradients enabled
    gradcam_flat = compute_gradcam(
        model, mm_root, full_inputs, class_positions, input_len
    )

    return prompt_flat, allprompt_flat, rollout_flat, gradcam_flat, semantic_flat, response_text


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
    prompt_heatmap: np.ndarray,
    semantic_heatmap: np.ndarray | None,
    bboxes: list,
    path: Path,
    title: str = "",
):
    img_np = np.array(image)
    n_cols = 4 if semantic_heatmap is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

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

    axes[1].imshow(img_np)
    axes[1].imshow(prompt_heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    draw_boxes(axes[1])
    axes[1].set_title(f"Prompt attention overlay")
    axes[1].axis("off")

    if semantic_heatmap is not None:
        axes[2].imshow(img_np)
        axes[2].imshow(semantic_heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        draw_boxes(axes[2])
        axes[2].set_title(f"Semantic correspondence")
        axes[2].axis("off")

        axes[3].imshow(img_np)
        diff = semantic_heatmap - prompt_heatmap
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[3].imshow(diff, cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
        draw_boxes(axes[3])
        axes[3].set_title("Semantic − Prompt\n(red = semantic gains)")
        axes[3].axis("off")

    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_overlay_v2(
    image: PILImage.Image,
    allprompt_heatmap: np.ndarray,
    rollout_heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray | None,
    semantic_heatmap: np.ndarray | None,
    bboxes: list,
    path: Path,
    title: str = "",
):
    """Improved overlays: all-prompt-tokens average + attention rollout + GradCAM."""
    img_np = np.array(image)
    panels = [
        ("Ground Truth", None),
        ("All prompt tokens\n(avg over text positions)", allprompt_heatmap),
        ("Attention rollout\n(propagated across all layers)", rollout_heatmap),
    ]
    if gradcam_heatmap is not None:
        panels.append(("GradCAM\n(last SigLIP encoder layer)", gradcam_heatmap))
    if semantic_heatmap is not None:
        panels.append(("Semantic correspondence", semantic_heatmap))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    def draw_boxes(ax):
        for bbox in bboxes:
            if len(bbox) < 4:
                continue
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            rect = mpatches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    for ax, (panel_title, heatmap) in zip(axes, panels):
        ax.imshow(img_np)
        if heatmap is not None:
            ax.imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        else:
            draw_boxes(ax)
            ax.set_title(f"{panel_title} ({len(bboxes)} boxes)")
            ax.axis("off")
            continue
        draw_boxes(ax)
        ax.set_title(panel_title)
        ax.axis("off")

    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    pg_scores: list[int],
    energy_scores: list[float],
    sem_pg_scores: list[int],
    sem_energy_scores: list[float],
    model_label: str,
    output_dir: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def pointing_bar(ax, scores, title):
        n_total = len(scores)
        n_correct = sum(scores)
        bars = ax.bar(
            ["Correct", "Incorrect"],
            [n_correct, n_total - n_correct],
            color=["#2ecc71", "#e74c3c"],
        )
        for bar, v in zip(bars, [n_correct, n_total - n_correct]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=10)
        ax.set_title(f"{title}\nacc = {n_correct / n_total:.3f}")
        ax.set_ylabel("Count")

    def energy_hist(ax, scores, title):
        ax.hist(scores, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(
            np.mean(scores), color="red", linestyle="--",
            label=f"mean = {np.mean(scores):.3f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Fraction of attention inside boxes")
        ax.set_ylabel("Count")
        ax.legend()

    pointing_bar(axes[0, 0], pg_scores, "Pointing Game — Prompt Attention")
    energy_hist(axes[0, 1], energy_scores, "Energy Inside Box — Prompt Attention")

    if sem_pg_scores:
        pointing_bar(axes[1, 0], sem_pg_scores, "Pointing Game — Semantic Correspondence")
        energy_hist(axes[1, 1], sem_energy_scores, "Energy Inside Box — Semantic Correspondence")
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].text(0.5, 0.5, "No class tokens found in any response",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].axis("off")

    plt.suptitle(f"H3 Alignment — {Path(model_label).name}", fontsize=12)
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

    lm_device = next(model.parameters()).device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    overlay_v2_dir = output_dir / "overlay_v2"
    overlay_v2_dir.mkdir(exist_ok=True)

    pg_scores: list[int] = []
    energy_scores: list[float] = []
    sem_pg_scores: list[int] = []
    sem_energy_scores: list[float] = []
    per_sample: list[dict] = []

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {Path(sample['path']).name}")

        image = PILImage.open(sample["path"]).convert("RGB")
        img_w, img_h = image.size
        bboxes = sample["bboxes"]
        img_classes = sample["classes"]

        # Build per-sample prompt and class word list from dataset annotations
        class_label = " and ".join(img_classes) if img_classes else "objects"
        class_words = list({w for c in img_classes for w in c.split()})
        prompt_text = build_prompt_text(processor, class_label)
        print(f"  Prompt   : Identify all {class_label} visible in this image.")

        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to(lm_device) for k, v in inputs.items()}

        img_positions = find_image_positions(inputs["input_ids"], img_token_id)
        if len(img_positions) == 0:
            print(f"  WARNING: No image tokens found in input_ids. Skipping.")
            print(f"    input_ids unique: {inputs['input_ids'].unique().tolist()[:10]}")
            continue

        print(f"  seq_len={inputs['input_ids'].shape[1]} | img_tokens={len(img_positions)}")

        try:
            prompt_flat, allprompt_flat, rollout_flat, gradcam_flat, semantic_flat, response_text = extract_combined_heatmaps(
                model, processor, inputs, img_positions, mm_root,
                class_words=class_words, n_layers=args.n_layers,
            )
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            continue

        print(f"  Response: {response_text[:80]}")
        if semantic_flat is not None:
            found = [w for w in class_words if w in response_text.lower()]
            print(f"  Class tokens found: {found}")
        else:
            print(f"  Class tokens found: none — semantic heatmap skipped")

        bbox_mask = bboxes_to_mask(bboxes, img_h, img_w)

        prompt_heatmap = flat_to_image(prompt_flat, img_h, img_w)
        pg = pointing_game(prompt_heatmap, bbox_mask)
        energy = energy_inside_box(prompt_heatmap, bbox_mask)
        pg_scores.append(int(pg))
        energy_scores.append(energy)
        print(f"  Prompt   → PG: {'✓' if pg else '✗'}  Energy: {energy:.4f}")

        sem_pg = sem_energy = None
        if semantic_flat is not None:
            sem_heatmap = flat_to_image(semantic_flat, img_h, img_w)
            sem_pg = pointing_game(sem_heatmap, bbox_mask)
            sem_energy = energy_inside_box(sem_heatmap, bbox_mask)
            sem_pg_scores.append(int(sem_pg))
            sem_energy_scores.append(sem_energy)
            print(f"  Semantic → PG: {'✓' if sem_pg else '✗'}  Energy: {sem_energy:.4f}")
        else:
            sem_heatmap = None

        per_sample.append({
            "image": sample["path"],
            "response": response_text,
            "n_bboxes": len(bboxes),
            "n_img_tokens": len(img_positions),
            "prompt_pointing_game": int(pg),
            "prompt_energy_inside_box": round(energy, 4),
            "semantic_pointing_game": int(sem_pg) if sem_pg is not None else None,
            "semantic_energy_inside_box": round(sem_energy, 4) if sem_energy is not None else None,
        })

        allprompt_heatmap = flat_to_image(allprompt_flat, img_h, img_w)
        rollout_heatmap = flat_to_image(rollout_flat, img_h, img_w)
        gradcam_heatmap = flat_to_image(gradcam_flat, img_h, img_w) if gradcam_flat is not None else None

        if i < args.n_overlays:
            save_overlay(
                image, prompt_heatmap, sem_heatmap, bboxes,
                path=overlay_dir / f"overlay_{i:03d}.png",
                title=(
                    f"Prompt PG={'✓' if pg else '✗'} E={energy:.2f}  |  "
                    f"Semantic PG={'✓' if sem_pg else ('✗' if sem_pg is not None else 'N/A')} "
                    f"E={f'{sem_energy:.2f}' if sem_energy is not None else 'N/A'}"
                ),
            )
            save_overlay_v2(
                image, allprompt_heatmap, rollout_heatmap, gradcam_heatmap, sem_heatmap, bboxes,
                path=overlay_v2_dir / f"overlay_{i:03d}.png",
                title=(
                    f"All-prompt PG={'✓' if pg else '✗'} E={energy:.2f}  |  "
                    f"Rollout  |  GradCAM: {'ok' if gradcam_heatmap is not None else 'skip'}  |  "
                    f"Semantic PG={'✓' if sem_pg else ('✗' if sem_pg is not None else 'N/A')} "
                    f"E={f'{sem_energy:.2f}' if sem_energy is not None else 'N/A'}"
                ),
            )

        del inputs, prompt_flat, allprompt_flat, rollout_flat, gradcam_flat, semantic_flat
        del prompt_heatmap, allprompt_heatmap, rollout_heatmap, gradcam_heatmap, sem_heatmap, bbox_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_sample:
        print("\nNo valid samples processed.")
        return

    pg_acc = float(np.mean(pg_scores))
    mean_energy = float(np.mean(energy_scores))

    print(f"\n=== H3 Results ({args.agml_dataset}) ===")
    print(f"  Samples evaluated         : {len(per_sample)}")
    print(f"  Prompt   — PG acc         : {pg_acc:.4f}")
    print(f"  Prompt   — Mean energy    : {mean_energy:.4f}")
    if sem_pg_scores:
        sem_pg_acc = float(np.mean(sem_pg_scores))
        sem_mean_energy = float(np.mean(sem_energy_scores))
        print(f"  Semantic — PG acc         : {sem_pg_acc:.4f}  ({len(sem_pg_scores)} samples with class tokens)")
        print(f"  Semantic — Mean energy    : {sem_mean_energy:.4f}")
    else:
        sem_pg_acc = sem_mean_energy = None
        print(f"  Semantic — no class tokens found in any response")

    summary = {
        "model": args.model,
        "dataset": args.agml_dataset,
        "class_words": sorted({w for s in samples for w in s["classes"]}),
        "n_samples": len(per_sample),
        "n_layers_averaged": args.n_layers,
        "prompt_pointing_game_accuracy": round(pg_acc, 4),
        "prompt_mean_energy_inside_box": round(mean_energy, 4),
        "semantic_pointing_game_accuracy": round(sem_pg_acc, 4) if sem_pg_acc is not None else None,
        "semantic_mean_energy_inside_box": round(sem_mean_energy, 4) if sem_mean_energy is not None else None,
        "semantic_samples_with_class_tokens": len(sem_pg_scores),
        "per_sample": per_sample,
    }

    result_path = output_dir / "h3_results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {result_path}")

    print("\n=== Generating plots ===")
    plot_summary(pg_scores, energy_scores, sem_pg_scores, sem_energy_scores, args.model, output_dir)
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
