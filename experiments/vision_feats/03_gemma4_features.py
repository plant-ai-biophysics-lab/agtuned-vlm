"""
O1/O2 Feature Analysis — Script 3: Gemma 4 Vision Encoder
==========================================================
Extracts THREE levels of features from Gemma 4 (SigLIP-based):

  [O1] Pre-projector  — raw SigLIP patch embeddings (vision_hidden_dim)
       Pure vision representation, directly comparable to DINOv2 and iNatAg.
       Visualized as: patch activation heatmap (L2 norm per patch).

  [O2] Post-projector — SigLIP embeddings projected into LM embedding space
       via Gemma 4's embed_vision module.
       Cosine similarity with label text embedding = geometric alignment proxy.
       Visualized as: V↔T cosine similarity heatmap overlaid on image.

  [O2b] Cross-attention (--cross_attention) — true LLM attention from label
        text token positions back to image token positions, extracted from
        the last N decoder layers during a full (no-generation) forward pass.
        This is the genuine multimodal reasoning signal: which image patches
        does the model attend to when processing the label text?
        Requires full model on GPU — not compatible with --offload_lm.
        Visualized as: per-patch attention heatmap from label tokens.

Model: google/gemma-4-E4B-it  (default, ~4B params)
Switching variants: change --model_name only.

VRAM requirements:
  - A100 80GB : default (bfloat16, device_map=auto)  (~10-12 GB)
  - A100 40GB : --use_4bit                            (~5 GB)
  - RTX 3060  : --offload_lm  (O1 only, O2/O2b unavailable)

Usage:
    # A100 80GB — O1 + O2 only
    python 03_gemma4_features.py --image_path img.jpg --label "leaf rust"

    # A100 80GB — O1 + O2 + O2b cross-attention
    python 03_gemma4_features.py --image_path img.jpg --label "leaf rust" --cross_attention

    # RTX 3060 — O1 only
    python 03_gemma4_features.py --image_path img.jpg --label "leaf rust" --offload_lm
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _import_transformers():
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        return AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        try:
            # Fallback for older transformers versions
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            return AutoProcessor, AutoModelForMultimodalLM
        except ImportError:
            raise ImportError(
                "transformers >= 4.51 required for Gemma 4. "
                "Run: pip install --upgrade transformers"
            )

def _import_bnb():
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig
    except ImportError:
        raise ImportError("bitsandbytes required. Run: pip install bitsandbytes")


# ── Model loading ─────────────────────────────────────────────────────────────

def build_load_kwargs(args) -> dict:
    """
    Build base loading kwargs (dtype, quantization). The device_map for
    --offload_lm is constructed AFTER initial model inspection (see main),
    because Gemma 4's submodule naming varies and accelerate needs every
    parameter to be assigned.
    """
    load_kwargs = {"dtype": torch.bfloat16}

    if args.offload_lm:
        if args.use_4bit:
            print("[Gemma4] Note: --use_4bit ignored when --offload_lm is set "
                  "(bitsandbytes incompatible with CPU offload).")
        print("[Gemma4] Mode: bf16 + LLM offloaded to CPU  (RTX 3060 / 12 GB VRAM)")
        # device_map is built dynamically after loading config — see main()

    elif args.use_4bit:
        load_kwargs["device_map"] = "auto"
        print("[Gemma4] Mode: 4-bit NF4, device_map=auto  (A100 40GB)")
        BitsAndBytesConfig = _import_bnb()
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    else:
        load_kwargs["device_map"] = "auto"
        print("[Gemma4] Mode: bfloat16, device_map=auto  (default — A100 80GB)")

    return load_kwargs


def build_offload_device_map(model_name: str, processor_dummy) -> dict:
    """
    Inspect the Gemma 4 model on the meta device to discover submodule names,
    then build a device_map that puts vision-related modules on cuda and
    everything else (LLM, audio tower, etc.) on cpu.

    We don't load real weights here — meta loading is fast and uses no memory.
    """
    from transformers import AutoConfig
    try:
        from transformers import AutoModelForImageTextToText as _ModelClass
    except ImportError:
        from transformers import AutoModelForMultimodalLM as _ModelClass
    import accelerate

    config = AutoConfig.from_pretrained(model_name)
    with accelerate.init_empty_weights():
        meta_model = _ModelClass.from_config(config)

    # Walk top-level submodules under .model (or root) and classify them.
    if hasattr(meta_model, "model"):
        trunk = meta_model.model
        prefix = "model"
    else:
        trunk = meta_model
        prefix = ""

    children = [n for n, _ in trunk.named_children()]
    print(f"[Gemma4] Discovered submodules under '{prefix or '<root>'}': {children}")

    # Vision-related modules go to GPU; everything else (LLM, audio, etc.) to CPU.
    # We look for submodules whose name suggests they're vision-related.
    VISION_KEYWORDS = ("vision", "visual", "image", "projector", "multi_modal", "mm_")
    device_map = {}
    for name in children:
        full = f"{prefix}.{name}" if prefix else name
        if any(kw in name for kw in VISION_KEYWORDS):
            device_map[full] = "cuda"
        else:
            device_map[full] = "cpu"

    # Catch the lm_head at top level if present
    if hasattr(meta_model, "lm_head"):
        device_map["lm_head"] = "cpu"

    print(f"[Gemma4] Built device_map:")
    for k, v in device_map.items():
        print(f"           {k:50s} → {v}")
    return device_map


# ── Input preparation ─────────────────────────────────────────────────────────

def prepare_inputs(image: Image.Image, label: str, processor, device: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"Is this image showing {label}?"}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


# ── [O1] Pre-projector: raw SigLIP patch embeddings ──────────────────────────

@torch.no_grad()
def extract_raw_patch_embeddings(inputs: dict, mm_root) -> tuple:
    """
    Run only the vision tower (SigLIP for Gemma 3 / Gemma4VisionModel for Gemma 4).
    Gemma 4 requires pixel_position_ids in addition to pixel_values.
    Returns:
        patch_embeds : cpu tensor (N_patches, vision_dim)
        grid_h, grid_w : int
    """
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError("No pixel_values found in inputs.")

    # Gemma 4 may pack pixel_values as (B, num_images, C, H, W) or (B, C, H, W).
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.squeeze(0)
        if pixel_values.dim() == 4 and pixel_values.shape[0] > 1:
            print(f"[Gemma4] Multi-image input detected ({pixel_values.shape[0]}); using image 0 only.")
            pixel_values = pixel_values[0:1]

    # Gemma 4's vision tower forward signature includes pixel_position_ids.
    # The processor populates this for vision inputs. We forward whichever
    # vision-related kwargs the processor produced.
    import inspect
    forward_params = set(inspect.signature(mm_root.vision_tower.forward).parameters.keys())
    print(f"[Gemma4] vision_tower.forward accepts: {forward_params}")

    # Only pass vision-relevant keys. The processor also outputs text keys
    # (input_ids, attention_mask, mm_token_type_ids) which must NOT be forwarded
    # to the vision tower — it has its own internal attention_mask and passing
    # the text one causes a "multiple values" error.
    # Known processor→forward name remaps (Gemma 4):
    #   image_position_ids  →  pixel_position_ids
    REMAP = {"image_position_ids": "pixel_position_ids"}
    VISION_KEYS = {"pixel_values", "pixel_position_ids", "pixel_attention_mask",
                   "image_position_ids", "spatial_shapes", "image_sizes",
                   "vision_position_ids"}

    vision_kwargs = {"pixel_values": pixel_values}
    for k, v in inputs.items():
        if k == "pixel_values":
            continue
        if k not in VISION_KEYS:
            continue
        forward_key = REMAP.get(k, k)
        vision_kwargs[forward_key] = v

    if "output_hidden_states" in forward_params:
        vision_kwargs["output_hidden_states"] = True
    if "return_dict" in forward_params:
        vision_kwargs["return_dict"] = True

    print(f"[Gemma4] Calling vision_tower with kwargs: {list(vision_kwargs.keys())}")

    vision_out = mm_root.vision_tower(**vision_kwargs)

    # Output may be a ModelOutput or a tuple
    if hasattr(vision_out, "last_hidden_state"):
        hidden = vision_out.last_hidden_state
    elif isinstance(vision_out, tuple):
        hidden = vision_out[0]
    else:
        hidden = vision_out

    # Drop batch dim
    if hidden.dim() == 3:
        hidden = hidden[0]    # (N, D)
    elif hidden.dim() == 2:
        pass                  # already (N, D)
    else:
        raise ValueError(f"Unexpected vision_tower output shape: {hidden.shape}")

    # Determine grid shape. SigLIP-style models give a square grid; Gemma 4
    # supports variable aspect ratios so the grid may not be square. We try
    # to recover (h, w) by reading from inputs (image_sizes / spatial_shapes)
    # if present, otherwise we fall back to assuming square.
    N = hidden.shape[0]

    # Try to get explicit grid shape from inputs
    grid_h, grid_w = None, None
    for key in ("spatial_shapes", "image_sizes", "image_grid_thw"):
        if key in inputs:
            v = inputs[key]
            try:
                v = v.cpu().numpy() if hasattr(v, "cpu") else v
                # spatial_shapes is typically [[h, w]] for one image
                arr = v.reshape(-1, v.shape[-1]) if hasattr(v, "reshape") else v
                if len(arr) >= 1 and len(arr[0]) >= 2:
                    grid_h, grid_w = int(arr[0][-2]), int(arr[0][-1])
                    print(f"[Gemma4] Read grid shape from '{key}': {grid_h}×{grid_w}")
                    break
            except Exception:
                pass

    if grid_h is None or grid_w is None:
        # Fall back: assume square. If there's a CLS token, strip it.
        side = int(N ** 0.5)
        if side * side != N:
            # Try stripping a leading CLS-like token
            hidden = hidden[1:]
            N      = hidden.shape[0]
            side   = int(N ** 0.5)
        if side * side == N:
            grid_h, grid_w = side, side
        else:
            # Last resort: assume rectangular based on common aspect ratios
            raise ValueError(
                f"Cannot determine grid shape from {N} patches. "
                f"No spatial_shapes/image_sizes in inputs. Available input keys: {list(inputs.keys())}"
            )

    if grid_h * grid_w != N:
        # Mismatch — strip a CLS token if there's exactly one extra
        if grid_h * grid_w == N - 1:
            hidden = hidden[1:]
        else:
            raise ValueError(
                f"Patch count mismatch: hidden={N}, grid={grid_h}×{grid_w}={grid_h*grid_w}"
            )

    return hidden.cpu(), grid_h, grid_w


# ── [O2] Post-projector: projected embeddings + V↔T similarity ───────────────

@torch.no_grad()
def get_label_text_embedding(label: str, mm_root, processor) -> torch.Tensor:
    """
    Get a mean-pooled text embedding for the label from the LM embedding table.
    Returns float32 tensor on cpu, or None if the embedding table is on the
    meta device (which happens with --offload_lm in bf16 mode).
    O2 is skipped gracefully when this returns None.
    """
    lm_embed = mm_root.language_model.get_input_embeddings()
    weight   = next(lm_embed.parameters())

    if weight.device.type == "meta":
        # Check accelerate hook for the real execution device
        if hasattr(lm_embed, "_hf_hook") and hasattr(lm_embed._hf_hook, "execution_device"):
            embed_device = lm_embed._hf_hook.execution_device
            print(f"[Gemma4][O2] Embedding execution device (from hook): {embed_device}")
        else:
            print("[Gemma4][O2] LM embedding table is on meta device with no hook. "
                  "O2 skipped.")
            return None
    else:
        embed_device = weight.device

    tokens = processor.tokenizer(label, return_tensors="pt")
    try:
        token_embeds = lm_embed(tokens["input_ids"].to(embed_device))
        return token_embeds[0].mean(dim=0).cpu().float()
    except (NotImplementedError, RuntimeError) as e:
        print(f"[Gemma4][O2] Embedding lookup failed: {e}. Skipping O2.")
        return None


@torch.no_grad()
def project_and_align(patch_embeds: torch.Tensor,
                      label_embed: torch.Tensor,
                      grid_h: int, grid_w: int,
                      mm_root) -> tuple:
    """
    Project raw patch embeddings through multi_modal_projector,
    compute cosine similarity with label_embed.

    Returns:
        sim_map_norm     : np.ndarray (grid_h, grid_w)  normalized [0,1]
        sim_map_raw      : np.ndarray (grid_h, grid_w)  raw cosine [-1,1]
        mean_vt_sim      : float
        projected_embeds : cpu tensor (N, lm_dim)
    """
    proj = mm_root._resolved_projector

    # When accelerate has dispatched the model, parameters may be on the meta
    # device even though the module will execute on CPU or GPU. The actual
    # execution device is stored in the accelerate hook.
    if hasattr(proj, "_hf_hook") and hasattr(proj._hf_hook, "execution_device"):
        proj_device = proj._hf_hook.execution_device
        print(f"[Gemma4][O2] Projector execution device (from hook): {proj_device}")
    else:
        proj_device = next(
            (p.device for p in proj.parameters() if p.device.type != "meta"),
            torch.device("cpu")
        )
        print(f"[Gemma4][O2] Projector execution device (from params): {proj_device}")

    # Use bf16 to match typical model dtype; projector handles its own casting
    proj_input = patch_embeds.to(device=proj_device, dtype=torch.bfloat16).unsqueeze(0)
    projected  = proj(proj_input)[0].cpu().float()  # (N, lm_dim)

    proj_norm  = F.normalize(projected, dim=-1)
    label_norm = F.normalize(label_embed.unsqueeze(0), dim=-1)

    sims_raw    = (proj_norm @ label_norm.T).squeeze(-1).numpy()  # (N,)
    mean_vt_sim = float(sims_raw.mean())

    sims_norm = sims_raw - sims_raw.min()
    if sims_norm.max() > 0:
        sims_norm /= sims_norm.max()

    return (sims_norm.reshape(grid_h, grid_w),
            sims_raw.reshape(grid_h, grid_w),
            mean_vt_sim,
            projected)


# ── Shared metrics ────────────────────────────────────────────────────────────

def mean_patch_cosine_sim(patch_embeds: torch.Tensor) -> float:
    normed  = F.normalize(patch_embeds.float(), dim=-1)
    sim_mat = (normed @ normed.T).numpy()
    np.fill_diagonal(sim_mat, np.nan)
    return float(np.nanmean(sim_mat))


def patch_l2_map(patch_embeds: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    """Normalized per-patch L2 norm — used as O1 activation heatmap."""
    norms = torch.norm(patch_embeds.float(), dim=-1).numpy()
    norms = norms - norms.min()
    if norms.max() > 0:
        norms /= norms.max()
    return norms.reshape(grid_h, grid_w)


def patch_embedding_entropy(patch_embeds: np.ndarray) -> float:
    feats = torch.tensor(patch_embeds)
    probs = F.softmax(feats, dim=-1).numpy()
    return float(-(probs * np.log(probs + 1e-9)).sum(axis=-1).mean())


# ── Visualization helpers ─────────────────────────────────────────────────────

def _overlay(image: Image.Image, heatmap: np.ndarray,
             alpha: float = 0.55, cmap: str = "plasma") -> np.ndarray:
    W, H     = image.size
    img_np   = np.array(image) / 255.0
    colormap = plt.get_cmap(cmap)
    heat_up  = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ) / 255.0
    return np.clip((1 - alpha) * img_np + alpha * colormap(heat_up)[..., :3], 0, 1)


# ── Figure 1: O1 — Pre-projector (pure vision, comparable to DINOv2/iNatAg) ──

def make_figure_o1(image, l2_map, grid_h, grid_w,
                   label, model_name, entropy, mean_pw_sim,
                   output_dir, image_stem) -> str:

    short = model_name.split("/")[-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"[O1 — Pre-Projector] Gemma 4 Raw SigLIP Features  |  Label: \"{label}\"\n"
        f"{short}  |  Comparable to DINOv2 & iNatAg",
        fontsize=11, fontweight="bold"
    )

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(l2_map, cmap="inferno", interpolation="bilinear", vmin=0, vmax=1)
    axes[1].set_title(f"Patch Activation (L2 norm)\nGrid: {grid_h}×{grid_w}  |  "
                      f"Mean pairwise sim: {mean_pw_sim:.3f}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(_overlay(image, l2_map, cmap="inferno"))
    axes[2].set_title(f"Activation Overlay\nEntropy: {entropy:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{image_stem}_gemma4_o1_preprojector.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gemma4][O1] Saved → {out}")
    return out


# ── Figure 2: O2 — Post-projector (vision-language alignment) ────────────────

def make_figure_o2(image, sim_map_norm, sim_map_raw, grid_h, grid_w,
                   label, model_name, mean_vt_sim, projected_pw_sim,
                   output_dir, image_stem) -> str:

    short = model_name.split("/")[-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"[O2 — Post-Projector] Gemma 4 Vision↔Text Alignment  |  Label: \"{label}\"\n"
        f"{short}  |  After multi_modal_projector",
        fontsize=11, fontweight="bold"
    )

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    vmin, vmax = sim_map_raw.min(), sim_map_raw.max()
    im = axes[1].imshow(sim_map_raw, cmap="RdBu_r", interpolation="bilinear",
                        vmin=vmin, vmax=vmax)
    axes[1].set_title(f"V↔T Cosine Sim (raw)\nrange [{vmin:.2f}, {vmax:.2f}]  |  "
                      f"grid {grid_h}×{grid_w}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(_overlay(image, sim_map_norm, cmap="plasma"))
    axes[2].set_title(f"Alignment Overlay (normalized)\n"
                      f"Mean V↔T sim: {mean_vt_sim:.3f}  |  "
                      f"Projected patch sim: {projected_pw_sim:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{image_stem}_gemma4_o2_postprojector.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gemma4][O2] Saved → {out}")
    return out


# ── [O2b] Cross-attention: LLM attention from label tokens → image tokens ─────

def find_image_and_label_positions(inputs: dict, processor, label: str) -> tuple:
    """
    Identify which token positions in the full input sequence are:
      - image tokens  (where image patches were inserted)
      - label tokens  (where the label text appears in the prompt)

    Uses mm_token_type_ids if available (Gemma 4 provides this):
      0 = text token
      1 = image token
      2 = audio token (ignored)

    Returns:
        image_positions : list[int]  — token indices of image patches
        label_positions : list[int]  — token indices of label text tokens
    """
    input_ids = inputs["input_ids"][0]  # (seq_len,)

    # Image token positions from mm_token_type_ids
    if "mm_token_type_ids" in inputs:
        mm_types = inputs["mm_token_type_ids"][0]
        image_positions = (mm_types == 1).nonzero(as_tuple=True)[0].tolist()
    else:
        # Fallback: find image token ID from processor
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None and hasattr(processor, "tokenizer"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is not None:
            image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
        else:
            raise ValueError("Cannot determine image token positions. "
                             "mm_token_type_ids not found and image_token_id unknown.")

    # Label token positions: tokenize the label and find its span in input_ids
    label_token_ids = processor.tokenizer(
        label, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]

    # Slide over input_ids to find the label span
    label_positions = []
    seq = input_ids.tolist()
    tgt = label_token_ids.tolist()
    for i in range(len(seq) - len(tgt) + 1):
        if seq[i:i + len(tgt)] == tgt:
            label_positions = list(range(i, i + len(tgt)))
            break

    if not label_positions:
        # Fallback: use last N non-image, non-special tokens as proxy for label region
        special_ids = {processor.tokenizer.pad_token_id,
                       processor.tokenizer.eos_token_id,
                       processor.tokenizer.bos_token_id}
        special_ids.discard(None)
        image_set = set(image_positions)
        text_pos  = [i for i, t in enumerate(seq)
                     if i not in image_set and t not in special_ids]
        label_positions = text_pos[-len(tgt):] if text_pos else text_pos
        print(f"[Gemma4][O2b] Warning: label span not found exactly; "
              f"using last {len(label_positions)} text tokens as proxy.")

    print(f"[Gemma4][O2b] Sequence length: {len(seq)}")
    print(f"[Gemma4][O2b] Image token positions: {len(image_positions)} tokens "
          f"(indices {image_positions[0]}–{image_positions[-1]})")
    print(f"[Gemma4][O2b] Label token positions: {len(label_positions)} tokens "
          f"(indices {label_positions})")

    return image_positions, label_positions


def extract_cross_attention(model, inputs: dict, image_positions: list,
                            label_positions: list, grid_h: int, grid_w: int,
                            n_last_layers: int = 4, device: str = "cuda") -> tuple:
    """
    Run a full forward pass through the LLM with output_attentions=True,
    then extract attention from label token positions → image token positions.

    Aggregation:
      1. Average over all attention heads
      2. Average over all label token positions (rows)
      3. Select only the image token columns
      4. Average over the last n_last_layers decoder layers
      5. Reshape to (grid_h, grid_w) and normalize

    Returns:
        attn_map  : np.ndarray (grid_h, grid_w) normalized [0,1]
        mean_attn : float  mean attention weight on image tokens from label tokens
        layer_maps: list[np.ndarray]  per-layer maps (for diagnostic use)
    """
    # Convert positions to tensors for indexing
    img_idx   = torch.tensor(image_positions, dtype=torch.long)
    label_idx = torch.tensor(label_positions, dtype=torch.long)

    # Ensure inputs are on the right device
    fwd_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
    # Remove vision-side keys that the full LLM forward doesn't expect directly
    # (pixel_values etc. are handled internally via the multimodal token merging)
    FULL_FWD_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids",
                     "pixel_values", "image_position_ids"}
    fwd_inputs = {k: v for k, v in fwd_inputs.items() if k in FULL_FWD_KEYS}

    print(f"[Gemma4][O2b] Running full forward pass (output_attentions=True)...")
    print(f"[Gemma4][O2b] Using last {n_last_layers} decoder layers.")

    with torch.no_grad():
        outputs = model(
            **fwd_inputs,
            output_attentions=True,
            return_dict=True,
        )

    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise RuntimeError(
            "Model returned no attentions. The model may have been loaded with "
            "SDPA or flash attention which doesn't expose weights. Try loading "
            "with attn_implementation='eager' (adds ~30% memory overhead)."
        )

    total_layers = len(outputs.attentions)
    selected     = outputs.attentions[-n_last_layers:]
    print(f"[Gemma4][O2b] Total decoder layers: {total_layers}, "
          f"averaging last {len(selected)}.")

    layer_maps = []
    for layer_attn in selected:
        # layer_attn: (batch, heads, seq_len, seq_len)
        attn = layer_attn[0].float().cpu()      # (heads, seq_len, seq_len)
        attn = attn.mean(dim=0)                  # (seq_len, seq_len) avg over heads

        # Rows = query (label tokens), cols = key (image tokens)
        label_to_img = attn[label_idx][:, img_idx]  # (n_label, n_image)
        patch_attn   = label_to_img.mean(dim=0)      # (n_image,) avg over label tokens

        patch_attn_np = patch_attn.numpy()
        layer_maps.append(patch_attn_np)

    # Average across selected layers
    mean_map = np.stack(layer_maps, axis=0).mean(axis=0)  # (n_image,)

    # Sanity check: n_image should match grid_h * grid_w
    n_img = len(image_positions)
    if n_img != grid_h * grid_w:
        print(f"[Gemma4][O2b] Warning: image token count ({n_img}) != "
              f"patch grid ({grid_h}×{grid_w}={grid_h*grid_w}). "
              f"Using available tokens; map may be approximate.")
        # Pad or trim to fit grid
        target = grid_h * grid_w
        if n_img < target:
            mean_map = np.pad(mean_map, (0, target - n_img))
        else:
            mean_map = mean_map[:target]

    mean_attn = float(mean_map.mean())

    # Normalize to [0, 1]
    attn_map = mean_map.reshape(grid_h, grid_w)
    attn_map = attn_map - attn_map.min()
    if attn_map.max() > 0:
        attn_map /= attn_map.max()

    return attn_map, mean_attn, [m.reshape(grid_h, grid_w) for m in layer_maps]


def make_figure_o2b(image: Image.Image,
                    attn_map: np.ndarray,
                    layer_maps: list,
                    grid_h: int, grid_w: int,
                    label: str, model_name: str,
                    mean_attn: float, n_last_layers: int,
                    output_dir: str, image_stem: str) -> str:

    short  = model_name.split("/")[-1]
    n_show = min(len(layer_maps), 3)  # show up to 3 individual layers

    fig, axes = plt.subplots(1, 3 + n_show, figsize=(5 * (3 + n_show), 5))
    fig.suptitle(
        f"[O2b — Cross-Attention] LLM Attention: label tokens → image patches\n"
        f"Label: \"{label}\"  |  {short}  |  Avg last {n_last_layers} layers",
        fontsize=11, fontweight="bold"
    )

    # Panel 0 — original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 1 — averaged attention heatmap (raw grid)
    im = axes[1].imshow(attn_map, cmap="hot", interpolation="bilinear", vmin=0, vmax=1)
    axes[1].set_title(f"Cross-Attn Map (avg {n_last_layers} layers)\n"
                      f"Mean attn on image: {mean_attn:.4f}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 2 — overlay on image
    axes[2].imshow(_overlay(image, attn_map, cmap="hot"))
    axes[2].set_title("Cross-Attn Overlay")
    axes[2].axis("off")

    # Panels 3+ — individual layer maps (last n_show layers)
    for i, lmap in enumerate(layer_maps[-n_show:]):
        lmap_norm = lmap - lmap.min()
        if lmap_norm.max() > 0:
            lmap_norm /= lmap_norm.max()
        ax = axes[3 + i]
        ax.imshow(lmap_norm, cmap="hot", interpolation="bilinear", vmin=0, vmax=1)
        layer_idx = len(layer_maps) - n_show + i
        ax.set_title(f"Layer -{len(layer_maps) - layer_idx}")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{image_stem}_gemma4_o2b_crossattn.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gemma4][O2b] Saved → {out}")
    return out

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Vision Feature Analysis (O1 + O2)")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--label",      required=True,
                        help="Target concept label (e.g. 'leaf rust', 'aphid damage')")
    parser.add_argument("--output_dir", default="./outputs/gemma4")
    parser.add_argument("--model_name", default="google/gemma-4-E4B-it",
                        help="HuggingFace model ID — one arg to switch any Gemma 4 variant.")

    parser.add_argument("--use_4bit",   action="store_true",
                        help="Enable 4-bit NF4 quantization. Use for A100 40GB.")
    parser.add_argument("--offload_lm", action="store_true",
                        help="Offload LLM decoder to CPU. O1 only. Use for RTX 3060 (12 GB).")
    parser.add_argument("--cross_attention", action="store_true",
                        help="[O2b] Extract true LLM cross-attention from label tokens → "
                             "image patches. Requires full model on GPU (not compatible "
                             "with --offload_lm). Runs a full LLM forward pass.")
    parser.add_argument("--n_attn_layers", type=int, default=4,
                        help="Number of last LLM decoder layers to average for O2b "
                             "cross-attention map. Default: 4.")
    parser.add_argument("--eager_attn", action="store_true",
                        help="Load model with attn_implementation='eager'. Required for "
                             "--cross_attention. Adds ~30%% memory overhead but exposes "
                             "attention weights that SDPA does not.")
    args = parser.parse_args()

    AutoProcessor, AutoModelForImageTextToText = _import_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[Gemma4] GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    print(f"[Gemma4] Loading: {args.model_name}")
    load_kwargs = build_load_kwargs(args)
    processor   = AutoProcessor.from_pretrained(args.model_name)

    # For offload_lm we need to inspect the architecture first to build a
    # complete device_map (Gemma 4 has audio_tower, varying projector names, etc.)
    if args.offload_lm:
        load_kwargs["device_map"] = build_offload_device_map(args.model_name, processor)

    if args.eager_attn:
        load_kwargs["attn_implementation"] = "eager"
        print("[Gemma4] Using eager attention (required for --cross_attention)")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, **load_kwargs
    ).eval()

    # For Gemma 4's hybrid attention architecture, the top-level
    # attn_implementation kwarg may not propagate to all decoder layers.
    # Force eager on every attention submodule that has _attn_implementation.
    if args.eager_attn:
        n_patched = 0
        for module in model.modules():
            if hasattr(module, "_attn_implementation"):
                module._attn_implementation = "eager"
                n_patched += 1
        print(f"[Gemma4] Patched eager attention on {n_patched} submodules.")

    # Resolve the multimodal trunk. Gemma 4 nests submodules under .model
    # one level deeper than Gemma 3.
    if hasattr(model, "vision_tower"):
        mm_root = model
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        mm_root = model.model
    else:
        available = [n for n, _ in model.named_children()]
        nested    = ([n for n, _ in model.model.named_children()]
                     if hasattr(model, "model") else [])
        raise AttributeError(
            f"Could not locate vision_tower on the model.\n"
            f"  Top-level submodules: {available}\n"
            f"  model.model submodules: {nested}"
        )

    # Find projector and language_model dynamically. The projector name varies
    # across Gemma versions:
    #   Gemma 3: multi_modal_projector
    #   Gemma 4: embed_vision  (also has embed_audio for audio modality)
    PROJECTOR_NAMES = ("embed_vision", "multi_modal_projector", "mm_projector",
                       "vision_projector", "image_projector", "projector")
    projector = None
    proj_name = None
    for name in PROJECTOR_NAMES:
        if hasattr(mm_root, name):
            projector = getattr(mm_root, name)
            proj_name = name
            break
    if projector is None:
        raise AttributeError(
            f"Could not find a projector submodule on multimodal root. "
            f"Tried: {PROJECTOR_NAMES}. "
            f"Available: {[n for n, _ in mm_root.named_children()]}"
        )

    if not hasattr(mm_root, "language_model"):
        raise AttributeError(
            f"language_model submodule not found. "
            f"Available: {[n for n, _ in mm_root.named_children()]}"
        )

    print(f"[Gemma4] Resolved: vision_tower=mm_root.vision_tower, "
          f"projector=mm_root.{proj_name}, language_model=mm_root.language_model")

    # Bind the projector under a normalized name so downstream code can use it
    # without caring about the actual attribute name.
    mm_root._resolved_projector = projector
    print(f"[Gemma4] Projector ({proj_name}) parameter device: "
          f"{next(projector.parameters()).device}")

    image      = Image.open(args.image_path).convert("RGB")
    image_stem = os.path.splitext(os.path.basename(args.image_path))[0]

    print(f"[Gemma4] Preparing inputs for label: '{args.label}'")
    inputs = prepare_inputs(image, args.label, processor, device)
    print(f"[Gemma4] Processor produced keys: {list(inputs.keys())}")

    # ── [O1] Raw SigLIP patch embeddings ─────────────────────────────────────
    print("[Gemma4][O1] Extracting raw SigLIP patch embeddings (pre-projector)...")
    patch_embeds, grid_h, grid_w = extract_raw_patch_embeddings(inputs, mm_root)
    # Cast bf16 → float32 immediately (numpy doesn't support bf16)
    patch_embeds = patch_embeds.float()
    print(f"[Gemma4][O1] Grid: {grid_h}×{grid_w}  dim={patch_embeds.shape[-1]}")

    entropy     = patch_embedding_entropy(patch_embeds.numpy())
    mean_pw_sim = mean_patch_cosine_sim(patch_embeds)
    l2_map      = patch_l2_map(patch_embeds, grid_h, grid_w)

    make_figure_o1(image, l2_map, grid_h, grid_w, args.label, args.model_name,
                   entropy, mean_pw_sim, args.output_dir, image_stem)

    # ── [O2] Projected embeddings + V↔T alignment ────────────────────────────
    mean_vt_sim      = float("nan")
    projected_pw_sim = float("nan")
    label_embed      = None

    print("[Gemma4][O2] Getting label text embedding...")
    label_embed = get_label_text_embedding(args.label, mm_root, processor)

    if label_embed is not None:
        print("[Gemma4][O2] Projecting and computing V↔T similarity map...")
        sim_map_norm, sim_map_raw, mean_vt_sim, projected = project_and_align(
            patch_embeds, label_embed, grid_h, grid_w, mm_root
        )
        projected_pw_sim = mean_patch_cosine_sim(projected)
        make_figure_o2(image, sim_map_norm, sim_map_raw, grid_h, grid_w,
                       args.label, args.model_name, mean_vt_sim, projected_pw_sim,
                       args.output_dir, image_stem)

    # ── [O2b] True LLM cross-attention ────────────────────────────────────────
    mean_cross_attn = float("nan")
    if args.cross_attention:
        print("[Gemma4][O2b] Extracting true LLM cross-attention...")
        if not args.eager_attn:
            print("[Gemma4][O2b] Warning: --eager_attn not set. SDPA attention "
                  "does not expose weights — O2b will likely fail. "
                  "Re-run with --cross_attention --eager_attn.")
        try:
            image_positions, label_positions = find_image_and_label_positions(
                inputs, processor, args.label
            )
            attn_map, mean_cross_attn, layer_maps = extract_cross_attention(
                model, inputs, image_positions, label_positions,
                grid_h, grid_w, n_last_layers=args.n_attn_layers, device=device
            )
            make_figure_o2b(image, attn_map, layer_maps, grid_h, grid_w,
                            args.label, args.model_name, mean_cross_attn,
                            args.n_attn_layers, args.output_dir, image_stem)
        except RuntimeError as e:
            if "attn" in str(e).lower() or "attention" in str(e).lower():
                print(f"[Gemma4][O2b] Failed: {e}\n"
                      f"Try adding attn_implementation='eager' in the model load kwargs.")
            else:
                    raise

    print(f"\n── Gemma 4 Full Results ─────────────────────────────")
    print(f"  [O1 — Pre-projector  / Pure vision]")
    print(f"    Patch grid              : {grid_h}×{grid_w}")
    print(f"    Patch embedding entropy : {entropy:.4f}  (↑ = more uniform)")
    print(f"    Mean pairwise cosine sim: {mean_pw_sim:.4f}  (↑ = redundant patches)")
    print(f"  [O2 — Post-projector / Vision-language alignment]")
    if label_embed is not None:
        print(f"    Mean V↔T cosine sim     : {mean_vt_sim:.4f}  (↑ = better label alignment)")
        print(f"    Projected patch sim     : {projected_pw_sim:.4f}  (↑ = collapsed embedding space)")
    else:
        print(f"    Skipped (embedding lookup failed)")
    if args.cross_attention:
        print(f"  [O2b — Cross-attention / True LLM attention]")
        if not args.offload_lm and not np.isnan(mean_cross_attn):
            print(f"    Mean attn on image patches: {mean_cross_attn:.4f}  "
                  f"(↑ = model attends more to image)")
        else:
            print(f"    Skipped")
    print(f"─────────────────────────────────────────────────────\n")

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{image_stem}_gemma4_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"model: {args.model_name}\n")
        f.write(f"image: {args.image_path}\n")
        f.write(f"label: {args.label}\n")
        f.write(f"patch_grid: {grid_h}x{grid_w}\n")
        f.write(f"# O1 - pre-projector (pure vision, comparable to DINOv2/iNatAg)\n")
        f.write(f"patch_embedding_entropy: {entropy:.6f}\n")
        f.write(f"mean_pairwise_cosine_sim: {mean_pw_sim:.6f}\n")
        f.write(f"# O2 - post-projector cosine similarity (vision-language alignment proxy)\n")
        f.write(f"mean_vision_text_sim: {mean_vt_sim}\n")
        f.write(f"projected_mean_pairwise_cosine_sim: {projected_pw_sim}\n")
        f.write(f"# O2b - true LLM cross-attention (label tokens → image patches)\n")
        f.write(f"mean_cross_attention: {mean_cross_attn}\n")
        f.write(f"cross_attention_n_layers: {args.n_attn_layers if args.cross_attention else 'N/A'}\n")
    print(f"[Gemma4] Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
