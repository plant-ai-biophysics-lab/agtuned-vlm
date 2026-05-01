"""
O1 Feature Analysis — Script 4: Cross-Model Comparison
=======================================================
Reads per-model metric files from scripts 01–03 and produces a unified
side-by-side comparison figure and summary table.

Run this AFTER all three model scripts have been run on the same image.

Usage:
    python 04_compare_models.py \
        --image_path /path/to/image.jpg \
        --label "leaf rust" \
        --dinov2_dir   ./outputs/dinov2 \
        --inatag_dir   ./outputs/inatag \
        --gemma4_dir   ./outputs/gemma4 \
        --output_dir   ./outputs/comparison

It reads the *_metrics.txt files and the *_model.png visualizations
produced by each script and assembles them into one comparison figure.
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage


# ── Metric file parser ────────────────────────────────────────────────────────

def parse_metrics(txt_path: str) -> dict:
    metrics = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                metrics[key.strip()] = val.strip()
    return metrics


def find_metrics_file(directory: str, suffix: str) -> str:
    """Find *_{suffix}_metrics.txt in directory."""
    matches = glob.glob(os.path.join(directory, f"*_{suffix}_metrics.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No *_{suffix}_metrics.txt found in {directory}. "
            f"Run script 0{'1' if suffix=='dinov2' else '2' if suffix=='inatag' else '3'} first."
        )
    return matches[0]


def find_figure(directory: str, suffix: str) -> str:
    """
    Find the per-model figure for cross-model comparison.
    For Gemma4 we specifically want the O1 (pre-projector) figure since
    that's the one comparable to DINOv2 and iNatAg (pure vision features).
    """
    if suffix == "gemma4":
        matches = glob.glob(os.path.join(directory, "*_gemma4_o1_preprojector.png"))
        if not matches:
            # Fallback: any gemma4 figure
            matches = glob.glob(os.path.join(directory, "*_gemma4*.png"))
    else:
        # DINOv2 / iNatAg use simple suffix; exclude any subscripted variants
        all_matches = glob.glob(os.path.join(directory, f"*_{suffix}.png"))
        matches = [m for m in all_matches if not any(
            x in os.path.basename(m) for x in ["_o1_", "_o2_"]
        )]
    if not matches:
        raise FileNotFoundError(f"No matching figure found for '{suffix}' in {directory}.")
    return matches[0]


# ── Metrics normalization for radar / bar chart ────────────────────────────────

def _safe_get(d, key, default=float("nan")):
    val = d.get(key, default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_comparison_table(m_dino: dict, m_inat: dict, m_gem: dict) -> dict:
    """
    Harmonize metrics across models into a common comparison dict.

    Note: For Gemma 4 we use the O1 pre-projector metrics for fair comparison
    with DINOv2 and iNatAg (all measure pure vision features). The O2 V↔T
    similarity is reported separately in the metrics table.

    Shared comparison metrics (clamped to [0, 1] for visualization):
      - spatial_focus    : how concentrated/peaked the spatial map is
                           DINOv2  → 1 - clamp(attention_entropy / log(N_patches))
                           iNatAg  → cam_localization_score (already [0,1])
                           Gemma4  → no direct O1 analog → use 1 - clamp(patch_entropy / log(D))
      - feature_diversity: 1 - mean_pairwise_cosine_sim (clamped)
                           For iNatAg this isn't directly available so we use
                           1 - feature_sparsity as a proxy.
    """
    import math

    # Approximate normalizers for entropy
    # DINOv2-large @ 224×224 with patch=14: 16×16 = 256 patches → log(256) ≈ 5.55
    # SigLIP/Gemma4 patch dim ≈ 1152 (for E4B) → log(1152) ≈ 7.05
    # We use 5.55 as a reasonable normalizer that won't massively blow up
    DINO_ENTROPY_NORM = 5.55
    GEM_ENTROPY_NORM  = 7.05

    def clamp01(x):
        try:
            x = float(x)
            if math.isnan(x):
                return 0.0
            return max(0.0, min(1.0, x))
        except (ValueError, TypeError):
            return 0.0

    dino_attn_entropy = _safe_get(m_dino, "attention_entropy", DINO_ENTROPY_NORM)
    dino_mean_sim     = _safe_get(m_dino, "mean_pairwise_cosine_sim", 0.0)
    inat_cam          = _safe_get(m_inat, "cam_localization_score", 0.0)
    inat_sparsity     = _safe_get(m_inat, "feature_sparsity", 0.0)
    gem_patch_entropy = _safe_get(m_gem,  "patch_embedding_entropy", GEM_ENTROPY_NORM)
    gem_mean_sim      = _safe_get(m_gem,  "mean_pairwise_cosine_sim", 0.0)
    gem_vt_sim        = _safe_get(m_gem,  "mean_vision_text_sim", float("nan"))

    table = {
        "DINOv2": {
            "attention_entropy":        dino_attn_entropy,
            "mean_pairwise_cosine_sim": dino_mean_sim,
            "spatial_focus":            clamp01(1.0 - dino_attn_entropy / DINO_ENTROPY_NORM),
            "feature_diversity":        clamp01(1.0 - dino_mean_sim),
        },
        "iNatAg Swin": {
            "cam_localization_score":   inat_cam,
            "feature_sparsity":         inat_sparsity,
            "pred_confidence":          _safe_get(m_inat, "pred_confidence"),
            "spatial_focus":            clamp01(inat_cam),
            "feature_diversity":        clamp01(1.0 - inat_sparsity),
        },
        "Gemma 4 Vision": {
            "patch_embedding_entropy":  gem_patch_entropy,
            "mean_pairwise_cosine_sim": gem_mean_sim,
            "mean_vision_text_sim":     gem_vt_sim,  # O2-only, not used in spatial_focus
            "spatial_focus":            clamp01(1.0 - gem_patch_entropy / GEM_ENTROPY_NORM),
            "feature_diversity":        clamp01(1.0 - gem_mean_sim),
        },
    }
    return table


# ── Comparison visualization ───────────────────────────────────────────────────

def make_comparison_figure(image_path: str,
                           fig_dinov2: str,
                           fig_inatag: str,
                           fig_gemma4: str,
                           table: dict,
                           label: str,
                           output_dir: str):

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"O1 Vision Feature Analysis — Cross-Model Comparison\nLabel: \"{label}\"",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.03, right=0.97,
                           top=0.93, bottom=0.05,
                           hspace=0.35, wspace=0.12)

    # Row 0-2, col 0: per-model figures stacked
    for row, (fig_path, model_name) in enumerate([
        (fig_dinov2, "DINOv2 (Self-supervised)"),
        (fig_inatag, "iNatAg Swin (AgML fine-tuned)"),
        (fig_gemma4, "Gemma 4 Vision Encoder (VLM)"),
    ]):
        ax = fig.add_subplot(gs[row, 0])
        img = PILImage.open(fig_path)
        ax.imshow(img)
        ax.set_title(model_name, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")

    # Col 1, row 0: Spatial Focus bar chart
    ax_focus = fig.add_subplot(gs[0, 1])
    models   = list(table.keys())
    colors   = ["#4C8EF7", "#F76C4C", "#4CF7A8"]
    focus_vals = [table[m]["spatial_focus"] for m in models]
    bars = ax_focus.barh(models, focus_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax_focus.set_xlim(0, 1)
    ax_focus.set_xlabel("Score [0–1]  (↑ = more spatially focused)")
    ax_focus.set_title("Spatial Focus\n(DINOv2: 1-norm_entropy | iNatAg: CAM score | Gemma4: V↔T sim)",
                       fontsize=9)
    for bar, val in zip(bars, focus_vals):
        ax_focus.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{val:.3f}", va="center", fontsize=9)

    # Col 1, row 1: Feature Diversity bar chart
    ax_div = fig.add_subplot(gs[1, 1])
    div_vals = [table[m]["feature_diversity"] for m in models]
    bars2 = ax_div.barh(models, div_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax_div.set_xlim(0, 1)
    ax_div.set_xlabel("Score [0–1]  (↑ = more diverse patch features)")
    ax_div.set_title("Feature Diversity\n(1 − mean pairwise cosine similarity)", fontsize=9)
    for bar, val in zip(bars2, div_vals):
        ax_div.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    # Col 1, row 2: Raw metrics table
    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.axis("off")

    row_labels = []
    row_data   = []
    row_labels.append("DINOv2 — Attn Entropy")
    row_data.append(f"{table['DINOv2']['attention_entropy']:.4f}")
    row_labels.append("DINOv2 — Mean Patch Sim")
    row_data.append(f"{table['DINOv2']['mean_pairwise_cosine_sim']:.4f}")
    row_labels.append("iNatAg — CAM Localization")
    row_data.append(f"{table['iNatAg Swin']['cam_localization_score']:.4f}")
    row_labels.append("iNatAg — Feature Sparsity")
    row_data.append(f"{table['iNatAg Swin']['feature_sparsity']:.4f}")
    row_labels.append("iNatAg — Pred Confidence")
    row_data.append(f"{table['iNatAg Swin']['pred_confidence']:.4f}")
    row_labels.append("Gemma4 — Patch Entropy")
    row_data.append(f"{table['Gemma 4 Vision']['patch_embedding_entropy']:.4f}")
    row_labels.append("Gemma4 — Mean Patch Sim")
    row_data.append(f"{table['Gemma 4 Vision']['mean_pairwise_cosine_sim']:.4f}")
    row_labels.append("Gemma4 — V↔T Sim (label)")
    row_data.append(f"{table['Gemma 4 Vision']['mean_vision_text_sim']:.4f}")

    tbl = ax_tbl.table(
        cellText  = [[v] for v in row_data],
        rowLabels = row_labels,
        colLabels = ["Value"],
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.4, 1.5)
    ax_tbl.set_title("Raw Metrics Summary", fontsize=10, fontweight="bold", pad=8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"comparison_{label.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Compare] Comparison figure saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model feature comparison")
    parser.add_argument("--label",       required=True,
                        help="Target concept label (must match what was used in scripts 01-03)")
    parser.add_argument("--image_path",  required=True,
                        help="Original input image path (for reference)")
    parser.add_argument("--dinov2_dir",  required=True)
    parser.add_argument("--inatag_dir",  required=True)
    parser.add_argument("--gemma4_dir",  required=True)
    parser.add_argument("--output_dir",  default="./outputs/comparison")
    args = parser.parse_args()

    print("[Compare] Loading metric files...")
    m_dino = parse_metrics(find_metrics_file(args.dinov2_dir, "dinov2"))
    m_inat = parse_metrics(find_metrics_file(args.inatag_dir, "inatag"))
    m_gem  = parse_metrics(find_metrics_file(args.gemma4_dir, "gemma4"))

    print("[Compare] Loading per-model figures...")
    fig_dino = find_figure(args.dinov2_dir, "dinov2")
    fig_inat = find_figure(args.inatag_dir, "inatag")
    fig_gem  = find_figure(args.gemma4_dir, "gemma4")

    table = build_comparison_table(m_dino, m_inat, m_gem)

    print("\n── Cross-Model Summary ──────────────────────────────")
    for model, metrics in table.items():
        print(f"  {model}:")
        print(f"    Spatial focus    : {metrics['spatial_focus']:.4f}")
        print(f"    Feature diversity: {metrics['feature_diversity']:.4f}")
    print("─────────────────────────────────────────────────────\n")

    make_comparison_figure(
        args.image_path, fig_dino, fig_inat, fig_gem,
        table, args.label, args.output_dir
    )

    # Save summary CSV
    csv_path = os.path.join(args.output_dir, f"summary_{args.label.replace(' ', '_')}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("model,spatial_focus,feature_diversity\n")
        for model, metrics in table.items():
            f.write(f"{model},{metrics['spatial_focus']:.6f},{metrics['feature_diversity']:.6f}\n")
    print(f"[Compare] CSV summary saved → {csv_path}")


if __name__ == "__main__":
    main()
