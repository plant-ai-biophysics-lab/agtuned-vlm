"""
O1 Feature Analysis — Script 2: iNatAg Swin Transformer
=========================================================
Extracts penultimate-layer features and GradCAM heatmaps from the iNatAg
Swin Transformer (SwinWithLatLon architecture from AgML-iNat-Species-Classifier).

Supports both `with_lora` and `without_lora` checkpoints.
Uses `swin_large_without_latlon_with_lora.pth` by default (purest vision signal).
If the checkpoint has lat/lon fusion weights but you want to skip lat/lon input,
pass --dummy_latlon (feeds zeros — valid for feature visualization).

Usage:
    python 02_inatag_features.py \
        --image_path /path/to/image.jpg \
        --label "corn smut" \
        --checkpoint /path/to/swin_large_with_lora.pth \
        --num_classes 2587 \
        --output_dir ./outputs/inatag

Arguments:
    --image_path    Path to input agricultural image
    --label         Target concept / species name (used for GradCAM target)
    --checkpoint    Path to downloaded .pth checkpoint from HuggingFace
    --num_classes   Number of species classes the model was trained on.
                    Inferred automatically from checkpoint if not provided.
    --use_lora      Flag: whether checkpoint contains LoRA weights (default: True)
    --dummy_latlon  Flag: feed zero lat/lon (for checkpoints with lat/lon head)
    --no_latlon     Flag: model is the without_latlon variant (no lat/lon head)
    --output_dir    Directory to save outputs
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import SwinConfig, SwinForImageClassification
from torchvision import transforms


# ── Model Definition (mirrors train.py) ──────────────────────────────────────

class SwinWithLatLon(nn.Module):
    """Full model with lat/lon fusion head — mirrors train.py exactly."""
    def __init__(self, num_species: int, num_crop_weed: int = 2,
                 use_lora: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        base_id = "microsoft/swin-large-patch4-window12-384-in22k"
        config   = SwinConfig.from_pretrained(base_id)
        self.swin = SwinForImageClassification.from_pretrained(
            base_id, config=config, ignore_mismatched_sizes=True
        )
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=32, lora_alpha=16, lora_dropout=0.1,
                target_modules=["query", "key", "value"], use_rslora=True
            )
            self.swin = get_peft_model(self.swin, lora_config)

        in_features = self.swin.classifier.in_features
        self.swin.classifier = nn.Identity()

        self.fc_latlon  = nn.Linear(2, 32)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc_species = nn.Linear(in_features + 32, num_species)
        self.fc_crop_weed = nn.Linear(in_features + 32, num_crop_weed)

    def forward_features(self, pixel_values, lat_lon=None):
        """Return penultimate feature vector (before classifier)."""
        out = self.swin(pixel_values, output_hidden_states=True)
        swin_feat = out.hidden_states[-1].mean(dim=1)   # (B, embed_dim)
        if lat_lon is not None:
            latlon_feat  = self.fc_latlon(lat_lon)
            combined     = torch.cat((swin_feat, latlon_feat), dim=1)
            combined     = self.dropout(combined)
        else:
            combined = swin_feat
        return combined, swin_feat

    def forward(self, pixel_values, lat_lon=None):
        combined, _ = self.forward_features(pixel_values, lat_lon)
        return self.fc_species(combined), self.fc_crop_weed(combined)


class SwinNoLatLon(nn.Module):
    """Model variant without lat/lon head (without_latlon checkpoints)."""
    def __init__(self, num_species: int, num_crop_weed: int = 2,
                 use_lora: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        base_id = "microsoft/swin-large-patch4-window12-384-in22k"
        config   = SwinConfig.from_pretrained(base_id)
        self.swin = SwinForImageClassification.from_pretrained(
            base_id, config=config, ignore_mismatched_sizes=True
        )
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=32, lora_alpha=16, lora_dropout=0.1,
                target_modules=["query", "key", "value"], use_rslora=True
            )
            self.swin = get_peft_model(self.swin, lora_config)

        in_features = self.swin.classifier.in_features
        self.swin.classifier = nn.Identity()
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc_species = nn.Linear(in_features, num_species)
        self.fc_crop_weed = nn.Linear(in_features, num_crop_weed)

    def forward_features(self, pixel_values):
        out = self.swin(pixel_values, output_hidden_states=True)
        swin_feat = out.hidden_states[-1].mean(dim=1)
        return self.dropout(swin_feat), swin_feat

    def forward(self, pixel_values):
        combined, _ = self.forward_features(pixel_values)
        return self.fc_species(combined), self.fc_crop_weed(combined)


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def infer_num_classes_from_checkpoint(ckpt_path: str, no_latlon: bool) -> int:
    """Read fc_species weight shape from checkpoint to get num_classes."""
    state = torch.load(ckpt_path, map_location="cpu")
    # Handle DataParallel-wrapped keys
    state = {k.replace("module.", ""): v for k, v in state.items()}
    key = "fc_species.weight"
    if key not in state:
        raise KeyError(f"Cannot find '{key}' in checkpoint. Available keys (first 10): "
                       f"{list(state.keys())[:10]}")
    return state[key].shape[0]


def load_model(ckpt_path: str, num_classes: int, use_lora: bool,
               no_latlon: bool, device: str):
    if no_latlon:
        model = SwinNoLatLon(num_species=num_classes, use_lora=use_lora)
    else:
        model = SwinWithLatLon(num_species=num_classes, use_lora=use_lora)

    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ── Image preprocessing ───────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── GradCAM for Swin ──────────────────────────────────────────────────────────

class SwinGradCAM:
    """
    GradCAM targeting the last stage's spatial feature map.
    Hooks into the output of the last Swin stage (before global pooling).
    """
    def __init__(self, model, no_latlon: bool):
        self.model    = model
        self.no_latlon = no_latlon
        self.features = None
        self.grads    = None

        # Locate the underlying SwinModel.encoder.layers regardless of wrapping.
        # Possible cases:
        #   With LoRA :  self.swin = PeftModel
        #                self.swin.base_model.model.swin.encoder.layers
        #   No LoRA  :   self.swin = SwinForImageClassification
        #                self.swin.swin.encoder.layers
        swin_module = model.swin
        if hasattr(swin_module, "base_model"):
            # PEFT-wrapped: base_model is LoraModel, .model is the SwinForImageClassification
            inner_classifier_model = swin_module.base_model.model
        else:
            inner_classifier_model = swin_module
        # inner_classifier_model is SwinForImageClassification
        # Its inner SwinModel is .swin
        last_stage = inner_classifier_model.swin.encoder.layers[-1]

        self._fwd_hook = last_stage.register_forward_hook(self._save_features)
        self._bwd_hook = last_stage.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, input, output):
        # HF Swin stage returns tuple (hidden_states, ...)
        feat = output[0] if isinstance(output, tuple) else output
        self.features = feat.detach()

    def _save_grads(self, module, grad_input, grad_output):
        # grad_output is always a tuple (one entry per output)
        self.grads = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self, pixel_values, lat_lon=None, class_idx: int = 0) -> np.ndarray:
        self.model.zero_grad()
        if self.no_latlon:
            species_out, _ = self.model(pixel_values)
        else:
            species_out, _ = self.model(pixel_values, lat_lon)

        score = species_out[0, class_idx]
        score.backward()

        # grads/features: (B, H*W, C)
        grads    = self.grads[0]        # (H*W, C)
        features = self.features[0]     # (H*W, C)

        weights  = grads.mean(dim=0)    # (C,) — global avg pool over spatial
        cam      = (features * weights.unsqueeze(0)).sum(dim=-1)  # (H*W,)
        cam      = torch.relu(cam).cpu().float().numpy()

        # Reshape to spatial grid
        N = cam.shape[0]
        h = w = int(N ** 0.5)
        cam = cam.reshape(h, w)
        cam = (cam - cam.min()) / (cam.max() + 1e-9)
        return cam


# ── Quantitative metrics ──────────────────────────────────────────────────────

def compute_feature_stats(feat_vec: np.ndarray) -> dict:
    """
    For a single feature vector (embed_dim,):
      - l2_norm      : overall activation magnitude
      - sparsity     : fraction of near-zero activations (< 0.01 * max)
    """
    l2_norm  = float(np.linalg.norm(feat_vec))
    thresh   = 0.01 * np.abs(feat_vec).max()
    sparsity = float((np.abs(feat_vec) < thresh).mean())
    return {"l2_norm": l2_norm, "sparsity": sparsity}


def cam_localization_score(cam: np.ndarray, top_k_frac: float = 0.2) -> float:
    """
    Localization concentration: fraction of total CAM energy in the top-K% patches.
    Higher = more spatially concentrated (better localization).
    """
    flat  = cam.flatten()
    k     = max(1, int(len(flat) * top_k_frac))
    top_k = np.sort(flat)[::-1][:k]
    score = float(top_k.sum() / (flat.sum() + 1e-9))
    return score


# ── Visualization ──────────────────────────────────────────────────────────────

def overlay_heatmap(image: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.55, cmap: str = "inferno") -> np.ndarray:
    img_np   = np.array(image) / 255.0
    colormap = plt.get_cmap(cmap)
    heat_rgb = colormap(heatmap)[..., :3]
    return np.clip((1 - alpha) * img_np + alpha * heat_rgb, 0, 1)


def make_figure(image: Image.Image,
                cam: np.ndarray,
                label: str,
                feat_stats: dict,
                loc_score: float,
                checkpoint_name: str,
                output_dir: str,
                image_stem: str):

    W, H = image.size
    cam_up = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ckpt_short = os.path.basename(checkpoint_name).replace(".pth", "")
    fig.suptitle(f"iNatAg Swin Feature Analysis  |  Label: \"{label}\"\n{ckpt_short}",
                 fontsize=12, fontweight="bold")

    # Panel 1 — original
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 2 — raw GradCAM
    axes[1].imshow(cam, cmap="inferno", interpolation="bilinear")
    axes[1].set_title(f"GradCAM (raw grid)\nLocalization score: {loc_score:.3f}")
    axes[1].axis("off")

    # Panel 3 — overlay
    axes[2].imshow(overlay_heatmap(image.resize((W, H)), cam_up))
    axes[2].set_title(f"GradCAM Overlay\nL2 norm: {feat_stats['l2_norm']:.2f}  "
                      f"Sparsity: {feat_stats['sparsity']:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{image_stem}_inatag.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[iNatAg] Figure saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="iNatAg Swin Feature Analysis")
    parser.add_argument("--image_path",   required=True)
    parser.add_argument("--label",        required=True,
                        help="Target species/concept label (used for plot title)")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--num_classes",  type=int, default=None,
                        help="Number of species classes. Auto-inferred if not given.")
    parser.add_argument("--class_idx",    type=int, default=0,
                        help="Class index to compute GradCAM for. Default 0 (use "
                             "argmax prediction if unsure). Use --list_topk to explore.")
    parser.add_argument("--use_lora",     action="store_true", default=True,
                        help="Checkpoint contains LoRA weights (default: True)")
    parser.add_argument("--no_lora",      action="store_true",
                        help="Override: checkpoint has NO LoRA weights")
    parser.add_argument("--no_latlon",    action="store_true",
                        help="Use the without_latlon model variant")
    parser.add_argument("--output_dir",   default="./outputs/inatag")
    args = parser.parse_args()

    use_lora = not args.no_lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[iNatAg] Using device: {device}")

    # Auto-infer num_classes
    if args.num_classes is None:
        print("[iNatAg] Inferring num_classes from checkpoint...")
        args.num_classes = infer_num_classes_from_checkpoint(args.checkpoint, args.no_latlon)
    print(f"[iNatAg] num_classes = {args.num_classes}")

    print("[iNatAg] Loading model (downloads Swin-Large base on first run)...")
    model = load_model(args.checkpoint, args.num_classes, use_lora,
                       args.no_latlon, device)

    transform  = get_transform()
    image      = Image.open(args.image_path).convert("RGB")
    tensor     = transform(image).unsqueeze(0).to(device)
    lat_lon    = torch.zeros(1, 2).to(device)  # dummy if needed
    image_stem = os.path.splitext(os.path.basename(args.image_path))[0]

    # ── Run forward to get top prediction ─────────────────────────────────────
    with torch.no_grad():
        if args.no_latlon:
            species_out, cw_out = model(tensor)
        else:
            species_out, cw_out = model(tensor, lat_lon)
        pred_class = int(species_out.argmax(dim=1).item())
        pred_conf  = float(species_out.softmax(dim=1).max().item())

    class_idx = args.class_idx if args.class_idx != 0 else pred_class
    print(f"[iNatAg] Predicted class index: {pred_class}  (confidence: {pred_conf:.4f})")
    print(f"[iNatAg] Computing GradCAM for class index: {class_idx}")

    # ── Extract penultimate features ───────────────────────────────────────────
    with torch.no_grad():
        if args.no_latlon:
            combined_feat, swin_feat = model.forward_features(tensor)
        else:
            combined_feat, swin_feat = model.forward_features(tensor, lat_lon)
    feat_vec   = swin_feat[0].cpu().numpy()
    feat_stats = compute_feature_stats(feat_vec)

    # ── GradCAM ───────────────────────────────────────────────────────────────
    grad_cam = SwinGradCAM(model, args.no_latlon)
    cam = grad_cam.compute(tensor,
                            lat_lon=None if args.no_latlon else lat_lon,
                            class_idx=class_idx)
    grad_cam.remove_hooks()

    loc_score = cam_localization_score(cam)

    print(f"\n── iNatAg Quantitative Results ──────────────────────")
    print(f"  Predicted class index     : {pred_class}")
    print(f"  Prediction confidence     : {pred_conf:.4f}")
    print(f"  Feature L2 norm           : {feat_stats['l2_norm']:.4f}  (↑ = stronger activations)")
    print(f"  Feature sparsity          : {feat_stats['sparsity']:.4f}  (↑ = more sparse)")
    print(f"  CAM localization score    : {loc_score:.4f}  (↑ = more spatially focused)")
    print(f"─────────────────────────────────────────────────────\n")

    make_figure(image, cam, args.label, feat_stats, loc_score,
                args.checkpoint, args.output_dir, image_stem)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{image_stem}_inatag_metrics.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"image: {args.image_path}\n")
        f.write(f"label: {args.label}\n")
        f.write(f"pred_class_idx: {pred_class}\n")
        f.write(f"pred_confidence: {pred_conf:.6f}\n")
        f.write(f"gradcam_class_idx: {class_idx}\n")
        f.write(f"feature_l2_norm: {feat_stats['l2_norm']:.6f}\n")
        f.write(f"feature_sparsity: {feat_stats['sparsity']:.6f}\n")
        f.write(f"cam_localization_score: {loc_score:.6f}\n")
    print(f"[iNatAg] Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
