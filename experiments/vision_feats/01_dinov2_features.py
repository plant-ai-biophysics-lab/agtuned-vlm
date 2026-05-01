"""
O1 Feature Analysis — Script 1: DINOv2
=======================================
Extracts patch-level self-attention maps and CLS-token feature similarities
from DINOv2 (facebook/dinov2-large) for a given agricultural image.

Usage:
    python 01_dinov2_features.py \
        --image_path /path/to/image.jpg \
        --label "leaf rust" \
        --query_point 0.5 0.4 \
        --output_dir ./outputs/dinov2

Arguments:
    --image_path    Path to input agricultural image
    --label         Target concept label (used only for plot title/naming)
    --query_point   Normalized (x, y) coordinates [0,1] of the point of interest
                    e.g. 0.5 0.4 means center-left of image. DINOv2 uses this
                    to select the patch token whose similarity is visualized
                    against all other patches.
    --output_dir    Directory to save outputs (created if not exists)
    --model_name    HuggingFace model ID (default: facebook/dinov2-large) """

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def point_to_patch_idx(query_xy: tuple, grid_h: int, grid_w: int) -> int:
    """Convert normalized (x, y) coordinate to flat patch index."""
    col = min(int(query_xy[0] * grid_w), grid_w - 1)
    row = min(int(query_xy[1] * grid_h), grid_h - 1)
    return row * grid_w + col


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.55, cmap: str = "inferno") -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB image."""
    img_np = np.array(image.resize((heatmap.shape[1], heatmap.shape[0]))) / 255.0
    colormap = plt.get_cmap(cmap)
    heat_rgb = colormap(heatmap)[..., :3]
    blended = (1 - alpha) * img_np + alpha * heat_rgb
    return np.clip(blended, 0, 1)


# ── Core extraction ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_dinov2_features(image: Image.Image,
                             processor,
                             model,
                             query_xy: tuple,
                             device: str = "cuda"):
    """
    Returns:
        attn_map      : (grid_h, grid_w) — CLS self-attention (avg over heads, last layer)
        sim_map       : (grid_h, grid_w) — cosine similarity from query patch to all patches
        patch_features: (N_patches, embed_dim) — raw patch token embeddings
        grid_h, grid_w: spatial grid dimensions
        entropy       : scalar attention entropy (measure of focus)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)

    # ── Patch token features ──────────────────────────────────────────────────
    # last_hidden_state: (1, 1+N_patches, embed_dim) for plain DINOv2
    # For DINOv2-with-registers it's (1, 1+R+N_patches, embed_dim) where R=4
    hidden = outputs.last_hidden_state[0]           # (seq_len, embed_dim)

    # Determine number of "extra" tokens (CLS + registers) by trying to find a
    # square N. We trust that the patch token count must be a perfect square.
    seq_len = hidden.shape[0]
    n_extra = None
    for guess in (1, 5):  # 1 = just CLS, 5 = CLS + 4 register tokens
        n_patches = seq_len - guess
        side      = int(n_patches ** 0.5)
        if side * side == n_patches and n_patches > 0:
            n_extra = guess
            break
    if n_extra is None:
        raise RuntimeError(f"Cannot determine patch grid from sequence length {seq_len}")

    patch_features = hidden[n_extra:, :]            # drop CLS (+ registers)

    # ── CLS attention map ─────────────────────────────────────────────────────
    # attentions[-1]: (1, num_heads, seq_len, seq_len)
    if not outputs.attentions or len(outputs.attentions) == 0:
        raise RuntimeError(
            "Model returned no attention tensors. Make sure the model was loaded "
            "with attn_implementation='eager' (not SDPA or flash_attention)."
        )
    last_attn     = outputs.attentions[-1][0]            # (heads, seq, seq)
    cls_attn      = last_attn[:, 0, n_extra:]            # (heads, N_patches) — CLS→patches
    cls_attn_mean = cls_attn.mean(dim=0).cpu().float().numpy()  # average over heads

    # ── Infer grid size ───────────────────────────────────────────────────────
    N = patch_features.shape[0]
    grid_h = grid_w = int(N ** 0.5)
    assert grid_h * grid_w == N, f"Unexpected patch count {N}"

    attn_map = cls_attn_mean.reshape(grid_h, grid_w)
    attn_map = normalize_map(attn_map)

    # ── Attention entropy (lower = more focused) ──────────────────────────────
    p = cls_attn_mean / (cls_attn_mean.sum() + 1e-9)
    entropy = float(-(p * np.log(p + 1e-9)).sum())

    # ── Query-patch cosine similarity map ─────────────────────────────────────
    q_idx = point_to_patch_idx(query_xy, grid_h, grid_w)
    q_feat = patch_features[q_idx]                          # (embed_dim,)
    sims   = torch.nn.functional.cosine_similarity(
        patch_features, q_feat.unsqueeze(0), dim=-1
    ).cpu().numpy()                                          # (N_patches,)
    sim_map = normalize_map(sims.reshape(grid_h, grid_w))

    return attn_map, sim_map, patch_features.cpu().numpy(), grid_h, grid_w, entropy


# ── Quantitative metrics ──────────────────────────────────────────────────────

def compute_feature_stats(patch_features: np.ndarray) -> dict:
    """
    Compute patch-level feature statistics.
      - mean_pairwise_sim : average cosine similarity across all patch pairs
                            (low = diverse/distributed features; high = redundant)
      - std_activation    : std of L2 norms across patches
    """
    feats = torch.tensor(patch_features)
    normed = torch.nn.functional.normalize(feats, dim=-1)
    sim_matrix = (normed @ normed.T).numpy()
    np.fill_diagonal(sim_matrix, np.nan)
    mean_sim = float(np.nanmean(sim_matrix))

    norms = np.linalg.norm(patch_features, axis=-1)
    std_act = float(norms.std())

    return {"mean_pairwise_cosine_sim": mean_sim,
            "std_patch_l2_norm": std_act}


# ── Visualization ──────────────────────────────────────────────────────────────

def make_figure(image: Image.Image,
                attn_map: np.ndarray,
                sim_map: np.ndarray,
                query_xy: tuple,
                grid_h: int,
                grid_w: int,
                label: str,
                entropy: float,
                stats: dict,
                output_dir: str,
                image_stem: str):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"DINOv2 Feature Analysis  |  Label: \"{label}\"", fontsize=13, fontweight="bold")

    # Panel 1 — original image with query point
    ax = axes[0]
    ax.imshow(image)
    W, H = image.size
    px, py = query_xy[0] * W, query_xy[1] * H
    ax.scatter([px], [py], c="red", s=120, zorder=5, label="Query point")
    ax.set_title("Input Image + Query Point")
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2 — CLS attention heatmap
    h_up = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)) / 255.0
    ax = axes[1]
    ax.imshow(overlay_heatmap(image, h_up))
    ax.set_title(f"CLS Self-Attention\nEntropy: {entropy:.3f}")
    ax.axis("off")

    # Panel 3 — query-patch cosine similarity
    s_up = np.array(Image.fromarray((sim_map * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)) / 255.0
    ax = axes[2]
    ax.imshow(overlay_heatmap(image, s_up, cmap="viridis"))
    ax.scatter([px], [py], c="red", s=80, zorder=5)
    ax.set_title(f"Query-Patch Cosine Sim\nMean pairwise sim: {stats['mean_pairwise_cosine_sim']:.3f}")
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{image_stem}_dinov3.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DINOv2] Figure saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Vision Feature Analysis")
    parser.add_argument("--image_path",   required=True,  help="Path to input image")
    parser.add_argument("--label",        required=True,  help="Target concept label (for titling)")
    parser.add_argument("--query_point",  nargs=2, type=float, default=[0.5, 0.5],
                        metavar=("X", "Y"),
                        help="Normalized (x,y) query point in [0,1]. Default: 0.5 0.5")
    parser.add_argument("--output_dir",   default="./outputs/dinov2")
    parser.add_argument("--model_name",   default="facebook/dinov2-large")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DINOv2] Using device: {device}")
    print(f"[DINOv2] Loading model: {args.model_name}")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    # IMPORTANT: attn_implementation="eager" is required for output_attentions=True
    # to return non-empty attention tensors. Newer transformers versions default
    # to SDPA which doesn't expose attention weights.
    model     = AutoModel.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    ).to(device).eval()

    image      = load_image(args.image_path)
    query_xy   = tuple(args.query_point)
    image_stem = os.path.splitext(os.path.basename(args.image_path))[0]

    print(f"[DINOv2] Extracting features for '{args.label}' at query point {query_xy}...")
    attn_map, sim_map, patch_features, grid_h, grid_w, entropy = extract_dinov2_features(
        image, processor, model, query_xy, device=device
    )

    stats = compute_feature_stats(patch_features)

    print(f"\n── DINOv2 Quantitative Results ──────────────────────")
    print(f"  Attention entropy         : {entropy:.4f}  (↓ = more focused)")
    print(f"  Mean pairwise cosine sim  : {stats['mean_pairwise_cosine_sim']:.4f}  (↑ = redundant features)")
    print(f"  Std of patch L2 norms     : {stats['std_patch_l2_norm']:.4f}  (↑ = diverse activations)")
    print(f"─────────────────────────────────────────────────────\n")

    make_figure(image, attn_map, sim_map, query_xy, grid_h, grid_w,
                args.label, entropy, stats, args.output_dir, image_stem)

    # Save metrics to text file
    metrics_path = os.path.join(args.output_dir, f"{image_stem}_dinov3_metrics.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"model: {args.model_name}\n")
        f.write(f"image: {args.image_path}\n")
        f.write(f"label: {args.label}\n")
        f.write(f"query_point: {query_xy}\n")
        f.write(f"attention_entropy: {entropy:.6f}\n")
        f.write(f"mean_pairwise_cosine_sim: {stats['mean_pairwise_cosine_sim']:.6f}\n")
        f.write(f"std_patch_l2_norm: {stats['std_patch_l2_norm']:.6f}\n")
    print(f"[DINOv2] Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
