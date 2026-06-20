"""
Stage 2: Linear Probe — Vision Encoder Separability

Freezes each backbone, extracts embeddings for all active AgML datasets,
trains logistic regression heads, and compares separability across backbones.

Backbone loading is batched per model: each backbone is loaded once, embeddings
are extracted for all datasets, then the backbone is unloaded before the next.

Backbones supported:
  --dino_model    DINOv3 self-supervised baseline (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
  --yolo_model    YOLOv11 classification backbone (default: yolo11x-cls.pt)
  --model         Gemma4 fine-tuned (optional)

Output: per-dataset results, per-task-type aggregates, overall summary, and plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import os
from collections import Counter, defaultdict
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageTextToText, AutoProcessor

from src.common.dataset_config import active_datasets, DatasetConfig, prefetch_agml_datasets
from src.common.io import save_json, update_pipeline_results

# Path to the layer selections config written by app/layer_explorer.py
_LAYER_SELECTIONS_PATH = Path(__file__).parent.parent / "config" / "layer_selections.yaml"


# ── Early-exit sentinel ────────────────────────────────────────────────────────

class _EarlyExit(Exception):
    """Raised by forward hooks to abort the forward pass once features are captured.

    This short-circuits the LLM layers of a VLM so extraction cost scales with
    the vision encoder (~460M params) rather than the full model (~26B params).
    """


# ── Embedding cache ────────────────────────────────────────────────────────────

def _layer_tag(layer_path: str | None) -> str:
    """Short stable tag derived from the layer module path, included in cache filenames.

    Changing the selected layer in layer_explorer.py changes this tag, so stale
    embeddings are never silently reused after a layer selection change.
    """
    if not layer_path:
        return "default"
    import hashlib
    return hashlib.sha1(layer_path.encode()).hexdigest()[:8]


def _cache_path(cache_dir: Path, backbone_key: str, dataset_name: str, split: str, layer_path: str | None = None) -> Path:
    slug = backbone_key.replace("/", "--").replace(":", "-")
    tag  = _layer_tag(layer_path)
    return cache_dir / f"{slug}__{dataset_name}__{split}__{tag}.npz"


def load_cached_embeddings(cache_dir: Path, backbone_key: str, dataset_name: str, split: str, layer_path: str | None = None):
    """Return (embeddings, labels) from cache, or None if not cached.

    layer_path is the module path used to extract embeddings (e.g. 'model.visual.blocks.26').
    It is hashed into the filename so that changing the layer selection auto-invalidates the cache.
    """
    p = _cache_path(cache_dir, backbone_key, dataset_name, split, layer_path)
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return d["embeddings"], list(d["labels"])
    return None


def save_cached_embeddings(
    cache_dir: Path, backbone_key: str, dataset_name: str, split: str,
    embeddings: np.ndarray, labels: list[str], layer_path: str | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, backbone_key, dataset_name, split, layer_path)
    np.savez(p, embeddings=embeddings, labels=np.array(labels))
    print(f"  Cached: {p.name}")


def _load_layer_selections() -> dict:
    """Load saved pre/post-projector layer paths per model ID, or return empty dict."""
    if _LAYER_SELECTIONS_PATH.exists():
        return yaml.safe_load(_LAYER_SELECTIONS_PATH.read_text()) or {}
    return {}


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_agml_samples(dataset_name: str) -> list[dict]:
    import agml
    loader = agml.data.AgMLDataLoader(dataset_name)
    contents = cast(Dict[str, Any], loader.export_contents())
    class_names = cast(list[str], loader.classes)

    samples = []
    for path, label_idx in contents.items():
        label_text = class_names[int(label_idx)]
        cleaned_label = label_text.replace("___", " ").replace("_", " ")
        samples.append({"path": str(path), "label": cleaned_label})
    return samples


def load_dg_samples(images_dir: Path, dg_descriptions_jsonl: Path) -> list[dict]:
    """Load Digital Green samples from local image directory + JSONL metadata.

    Each JSONL description record maps image_names → class_label (diagnosis).
    Returns same format as load_agml_samples: [{"path": str, "label": str}, ...]
    """
    import json
    filename_to_label: dict[str, str] = {}
    with open(dg_descriptions_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            label = meta.get("class_label", "")
            for fname in meta.get("image_names", []):
                if fname:
                    filename_to_label[fname] = label

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = filename_to_label.get(img_path.name)
        if label:
            samples.append({"path": str(img_path), "label": label})
    return samples


def subsample_per_class(samples: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Cap each class to n_per_class images (seeded shuffle) for a balanced comparison."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_class.setdefault(s["label"], []).append(i)
    selected: list[int] = []
    for indices in by_class.values():
        if len(indices) > n_per_class:
            indices = rng.choice(indices, size=n_per_class, replace=False).tolist()
        selected.extend(indices)
    selected.sort()
    return [samples[i] for i in selected]


# ── Gemma4 backbone ────────────────────────────────────────────────────────────

def resolve_mm_root(model):
    """Return the sub-module that holds the vision encoder.

    Gemma4: vision_tower lives on model or model.model
    Qwen VL: vision encoder lives as model.visual or model.model.visual
    """
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        if hasattr(root, "vision_tower"):
            return root
        if hasattr(root, "visual"):
            # Wrap so callers can always access root.vision_tower
            root.vision_tower = root.visual
            return root
    available = [n for n, _ in model.named_children()]
    raise AttributeError(f"Could not locate vision encoder. Top-level: {available}")


def extract_vlm_embeddings(
    samples, processor, model, hook_module: torch.nn.Module, desc: str = "VLM embeddings"
) -> tuple[np.ndarray, list[str]]:
    """
    Hook hook_module, run a forward pass per image, mean-pool the captured feature map.

    hook_module is resolved by the caller — either from layer_selections.yaml via
    model.get_submodule(path), or from the heuristic resolve_mm_root fallback.
    """
    from tqdm import tqdm

    model_device = next(model.parameters()).device
    embeddings, labels = [], []

    for sample in tqdm(samples, desc=f"  {desc}", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "."}]}]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            inputs = processor(text="<image>", images=image, return_tensors="pt")

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            t = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            if isinstance(t, (list, tuple)):
                t = t[0]
            if isinstance(t, torch.Tensor):
                captured.append(t.detach().float().cpu())
            raise _EarlyExit

        handle = hook_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(**inputs)
        except _EarlyExit:
            pass
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"Vision hook captured nothing for {sample['path']}")

        feat = captured[0]
        if feat.dim() == 3:
            feat = feat[0]          # [batch, seq, dim] → [seq, dim]
        emb = feat.mean(dim=0).numpy()   # mean-pool over sequence → [dim]
        embeddings.append(emb)
        labels.append(sample["label"])

    return np.stack(embeddings), labels


_PROJECTOR_NAMES = (
    "embed_vision",           # Gemma 4
    "multi_modal_projector",  # Gemma 3 / LLaVA-style
    "mm_projector",
    "vision_projector",
    "image_projector",
    "projector",
    "visual_merger",          # Qwen2-VL patch merger
    "visual_projection",
    "img_projection",
)


def resolve_projector(mm_root):
    """Return (projector_module, name) for the VLM's vision→LM projector, or (None, None)."""
    for name in _PROJECTOR_NAMES:
        if hasattr(mm_root, name):
            return getattr(mm_root, name), name
    return None, None


def extract_vlm_embeddings_postproj(
    samples, processor, model, hook_module: torch.nn.Module, desc: str = "post-proj embeddings"
) -> tuple[np.ndarray, list[str]]:
    """Hook hook_module (projector output), run a forward pass, mean-pool to one vector per image.

    hook_module is resolved by the caller — either from layer_selections.yaml via
    model.get_submodule(path), or from the heuristic resolve_projector fallback.
    """
    from tqdm import tqdm

    model_device = next(model.parameters()).device
    embeddings, labels = [], []

    for sample in tqdm(samples, desc=f"  {desc}", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "."}]}]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            inputs = processor(text="<image>", images=image, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if isinstance(t, torch.Tensor):
                captured.append(t.detach().float().cpu())
            raise _EarlyExit

        handle = hook_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(**inputs)
        except _EarlyExit:
            pass
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"Projector hook captured nothing for {sample['path']}")

        feat = captured[0]
        if feat.dim() == 3:
            feat = feat[0]          # (batch, seq, dim) → (seq, dim)
        emb = feat.mean(dim=0).numpy()
        embeddings.append(emb)
        labels.append(sample["label"])

    return np.stack(embeddings), labels


def load_gemma4(model_id: str, device: str, quantize: str | None = None):
    local = os.path.abspath(model_id)
    if os.path.exists(local):
        model_id = local
    print(f"Loading VLM: {model_id}" + (f" [{quantize}]" if quantize else ""))
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs: dict = {"device_map": "auto"}
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    mm_root = resolve_mm_root(model)
    vision_device = next(mm_root.vision_tower.parameters()).device
    print(f"  Vision tower: {type(mm_root.vision_tower).__name__} on {vision_device}")
    return processor, model, mm_root


# ── YOLO backbone ──────────────────────────────────────────────────────────────

def extract_yolo_embeddings(samples, yolo_model_name: str, device: str) -> np.ndarray:
    from tqdm import tqdm
    from ultralytics import YOLO
    import torchvision.transforms as T

    print(f"Loading YOLO: {yolo_model_name}")
    yolo = YOLO(yolo_model_name)
    yolo.model.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    captured = []

    def _hook(module, input, output):
        t = output[0] if isinstance(output, (list, tuple)) else output
        captured.append(t.detach().flatten(1).mean(0).cpu().float())

    penultimate = list(yolo.model.model.children())[-2]
    handle = penultimate.register_forward_hook(_hook)

    embeddings = []
    with torch.no_grad():
        for sample in tqdm(samples, desc="  YOLO embeddings", unit="img", leave=False):
            captured.clear()
            image = PILImage.open(sample["path"]).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            yolo.model(x)
            embeddings.append(captured[0].numpy())

    handle.remove()
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(embeddings)


# ── DINOv3 backbone ────────────────────────────────────────────────────────────

def load_dino(model_id: str, device: str):
    print(f"Loading DINOv3: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    return processor, model


@torch.no_grad()
def extract_dino_embeddings(samples, processor, model, device) -> tuple[np.ndarray, list[str]]:
    from tqdm import tqdm
    embeddings, labels = [], []
    for sample in tqdm(samples, desc="  DINOv3 embeddings", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[0, 0, :].cpu().float().numpy()
        embeddings.append(emb)
        labels.append(sample["label"])
    return np.stack(embeddings), labels


def load_bioclip(model_id: str, device: str):
    import open_clip
    print(f"Loading BioCLIP: {model_id}")
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{model_id}")
    model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def extract_bioclip_embeddings(samples, model, preprocess, device) -> tuple[np.ndarray, list[str]]:
    from tqdm import tqdm
    embeddings, labels = [], []
    for sample in tqdm(samples, desc="  BioCLIP embeddings", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        features = model.encode_image(tensor)
        emb = features[0].cpu().float().numpy()
        embeddings.append(emb)
        labels.append(sample["label"])
    return np.stack(embeddings), labels


# ── Logistic regression probe ──────────────────────────────────────────────────

def run_probe(
    X_train, X_test, y_train, y_test, le, label: str
) -> tuple[dict, list[str], list[str]]:
    print(f"Training logistic regression [{label}]...")
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_test_labels = le.inverse_transform(y_test).tolist()
    y_pred_labels = le.inverse_transform(y_pred).tolist()

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_test_labels, y_pred_labels, zero_division=0, output_dict=True)

    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    metrics = {
        "backbone": label,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_report_dict": report_dict,
    }
    return metrics, y_test_labels, y_pred_labels


# ── FC probe ───────────────────────────────────────────────────────────────────

def run_fc_probe(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    label: str,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[dict, list[str], list[str]]:
    """Single fully-connected linear layer trained with Adam + CrossEntropyLoss."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"Training FC probe [{label}]  epochs={epochs} lr={lr} batch={batch_size}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    fc = nn.Linear(input_dim, n_classes).to(device)
    opt = torch.optim.Adam(fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te_t = torch.tensor(y_test, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        fc.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(fc(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_tr))

        fc.eval()
        with torch.no_grad():
            val_loss = criterion(fc(X_te), y_te_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  train_loss={train_losses[-1]:.4f}  val_loss={val_loss:.4f}")

    fc.eval()
    with torch.no_grad():
        y_pred = fc(X_te).argmax(dim=1).cpu().numpy()

    y_test_labels = le.inverse_transform(y_test).tolist()
    y_pred_labels = le.inverse_transform(y_pred).tolist()

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_test_labels, y_pred_labels, zero_division=0, output_dict=True)

    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    metrics = {
        "backbone": label,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_report_dict": report_dict,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return metrics, y_test_labels, y_pred_labels


# ── Combined embeddings builder ────────────────────────────────────────────────

def get_combined_class_counts(all_embeddings: dict, backbone: str) -> Counter:
    """Raw class counts across all datasets for one backbone (before any filtering)."""
    counts: Counter = Counter()
    for _, (_, labels_all, *_) in all_embeddings.get(backbone, {}).items():
        counts.update(labels_all)
    return counts


def build_combined_split(
    all_embeddings: dict, backbone: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, list[str]] | None:
    """Concatenate embeddings from all datasets for one backbone into a single split."""
    X_parts, label_parts = [], []
    for agml_name, (X, labels_all, *_) in all_embeddings.get(backbone, {}).items():
        X_parts.append(X)
        label_parts.extend(labels_all)

    if not X_parts:
        return None

    X_all = np.concatenate(X_parts, axis=0)
    le = LabelEncoder()
    y_all = le.fit_transform(label_parts)

    # Drop classes with fewer than 2 samples
    counts = Counter(label_parts)
    keep = np.array([counts[label_parts[i]] >= 2 for i in range(len(label_parts))])
    X_all, y_all = X_all[keep], y_all[keep]
    label_parts = [label_parts[i] for i in range(len(label_parts)) if keep[i]]
    # Re-encode after filtering
    le = LabelEncoder()
    y_all = le.fit_transform(label_parts)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y_all)), y_all))
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx], le, label_parts


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Visualizations ─────────────────────────────────────────────────────────────

def _safe_name(label: str) -> str:
    return label.replace("/", "_").replace(":", "_").replace(" ", "_")


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], label: str, output_dir: Path):
    class_labels = sorted(set(y_true))
    n = len(class_labels)
    fig_size = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}", fontsize=11)
    plt.tight_layout()

    path = output_dir / f"confusion_matrix_{_safe_name(label)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne(X: np.ndarray, y_labels: list[str], label: str, output_dir: Path, max_samples: int = 2000):
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y_labels = [y_labels[i] for i in idx]

    print(f"  Running t-SNE for {label} ({len(X)} samples)...")
    perplexity = min(30, len(X) - 1)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

    unique_labels = sorted(set(y_labels))
    import matplotlib
    cmap = matplotlib.colormaps["tab20" if len(unique_labels) <= 20 else "hsv"].resampled(len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    handles = []
    for lbl in unique_labels:
        mask = [i for i, y in enumerate(y_labels) if y == lbl]
        sc = ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color_map[lbl]], label=lbl, s=12, alpha=0.7)
        handles.append(sc)

    # No legend on the scatter plot — saved separately to avoid overlap
    ax.set_title(f"t-SNE Embedding Space — {label}", fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()

    path = output_dir / f"tsne_{_safe_name(label)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save 2D coordinates so other stages (e.g. prompt-optim) can overlay markers
    # on the same layout without re-running t-SNE.
    coords_path = output_dir / f"tsne_{_safe_name(label)}_coords.npz"
    np.savez(coords_path, X_2d=X_2d, labels=np.array(y_labels))

    # Legend saved as a separate image
    legends_dir = output_dir.parent / "tsne_legends"
    legends_dir.mkdir(parents=True, exist_ok=True)
    ncol = max(1, len(unique_labels) // 20)
    rows = (len(unique_labels) + ncol - 1) // ncol
    fig_leg, ax_leg = plt.subplots(figsize=(max(3, ncol * 3.5), max(1.5, rows * 0.35)))
    ax_leg.set_axis_off()
    ax_leg.legend(handles=handles, labels=unique_labels, loc="center",
                  fontsize=7, ncol=ncol, markerscale=1.5)
    legend_path = legends_dir / f"legend_{_safe_name(label)}.png"
    fig_leg.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close(fig_leg)
    print(f"  Saved: {legend_path}")


def plot_per_class_f1(backbone_reports: dict[str, dict], dataset_name: str, output_dir: Path):
    skip = {"accuracy", "macro avg", "weighted avg"}
    all_classes = set()
    for report_dict in backbone_reports.values():
        all_classes.update(k for k in report_dict if k not in skip)
    all_classes = sorted(all_classes)

    backbones = list(backbone_reports.keys())
    n = len(backbones)
    x = np.arange(len(all_classes))
    width = 0.8 / n

    fig_w = max(14, len(all_classes) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    for i, key in enumerate(backbones):
        vals = [backbone_reports[key].get(c, {}).get("f1-score", 0.0) for c in all_classes]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 by Backbone — {dataset_name}")
    ax.legend()
    plt.tight_layout()

    path = output_dir / f"per_class_f1_{_safe_name(dataset_name)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overall_summary(results: dict, output_dir: Path):
    """Bar chart comparing backbone F1 using LR combined (one global classifier)."""
    backbones = list(results["lr_combined"].keys())
    if not backbones:
        return
    f1_macros = [results["lr_combined"][b]["f1_macro"] for b in backbones]
    accuracies = [results["lr_combined"][b]["accuracy"] for b in backbones]

    x = np.arange(len(backbones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(backbones) * 2.5), 5))
    b1 = ax.bar(x - width / 2, f1_macros, width, label="F1 Macro")
    b2 = ax.bar(x + width / 2, accuracies, width, label="Accuracy")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([b.split("/")[-1] for b in backbones], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe — Overall Backbone Comparison (LR combined, all classes)")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "summary_overall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_curve(train_losses: list[float], val_losses: list[float], label: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"FC Probe Loss — {label}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"fc_loss_{_safe_name(label)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_projector_before_after(
    pre_combined: dict[str, dict],   # {backbone: {f1_macro, ...}}  (lr_combined)
    post_combined: dict[str, dict],  # {backbone_postproj: {f1_macro, ...}}  (postproj_lr_combined)
    output_dir: Path,
):
    """Grouped bar chart: pre-projector vs post-projector LR F1 per backbone (LR combined)."""
    pre_means, post_means, ratios = {}, {}, {}
    for b, m in pre_combined.items():
        post_key = f"{b}_postproj"
        if post_key not in post_combined:
            continue
        pre_means[b]  = m["f1_macro"]
        post_means[b] = post_combined[post_key]["f1_macro"]
        ratios[b]     = round(post_means[b] / pre_means[b], 3) if pre_means[b] > 0 else 0.0

    if not pre_means:
        print("  [skip] No paired pre/post-projector results to plot.")
        return

    labels = list(pre_means)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.5), 5))
    b1 = ax.bar(x - width / 2, [pre_means[b] for b in labels],  width,
                label="Pre-projector (vision encoder)", color="#4C72B0")
    b2 = ax.bar(x + width / 2, [post_means[b] for b in labels], width,
                label="Post-projector (after LM projection)", color="#DD8452")

    for bar, val in zip(b1, [pre_means[b] for b in labels]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, b, val in zip(b2, labels, [post_means[b] for b in labels]):
        ratio = ratios[b]
        color = "#2ca02c" if ratio >= 1.0 else "#d62728"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}\n(×{ratio:.2f})", ha="center", va="bottom",
                fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([b.split("/")[-1] for b in labels], rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("LR F1 Macro (mean across datasets)")
    ax.set_title("Projector Separability: Pre vs Post-Projector\n"
                 "Ratio > 1 = projector preserves/improves class structure  "
                 "Ratio < 1 = projector destroys class structure")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = output_dir / "projector_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_long_tail(class_counts: Counter, output_dir: Path):
    """Bar chart of class sample counts sorted high → low."""
    classes = sorted(class_counts, key=lambda c: -class_counts[c])
    counts  = [class_counts[c] for c in classes]
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.35), 5))
    ax.bar(range(n), counts, color="#4C72B0")
    ax.set_xticks(range(n))
    fontsize = max(4, 8 - n // 20)
    ax.set_xticklabels(classes, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Class")
    ax.set_title("Class distribution — combined datasets (long-tail)")
    plt.tight_layout()
    path = output_dir / "class_distribution_long_tail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_accuracy_vs_frequency(
    class_counts: Counter,
    pre_reports:  dict,   # {backbone: per_class_report_dict}
    post_reports: dict,   # {backbone_postproj: per_class_report_dict}
    output_dir: Path,
):
    """Bar chart per backbone (pre vs post), plus summary averaging all backbones."""
    classes = sorted(class_counts, key=lambda c: -class_counts[c])
    n = len(classes)
    x = np.arange(n)
    fontsize = max(4, 8 - n // 20)

    cacc_dir = output_dir / "class_accuracy_vs_frequency"
    cacc_dir.mkdir(exist_ok=True)

    # --- Per-backbone figures ---
    for backbone, pre_report in pre_reports.items():
        post_key = f"{backbone}_postproj"
        post_report = post_reports.get(post_key)
        has_post = post_report is not None
        n_series = 2 if has_post else 1
        width = 0.8 / n_series
        offsets = [-(n_series - 1) * width / 2 + i * width for i in range(n_series)]

        pre_f1s = [pre_report.get(c, {}).get("f1-score", 0.0) for c in classes]

        fig, ax = plt.subplots(figsize=(max(14, n * 0.4), 6))
        ax.bar(x + offsets[0], pre_f1s, width, label="pre-proj", color="#4C72B0")
        if has_post:
            post_f1s = [post_report.get(c, {}).get("f1-score", 0.0) for c in classes]
            ax.bar(x + offsets[1], post_f1s, width, label="post-proj", color="#DD8452")

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=90, ha="right", fontsize=fontsize)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1 Score (per class)")
        ax.set_xlabel("Class (ordered by sample count, high → low)")
        short_name = backbone.split("/")[-1]
        ax.set_title(f"Per-class F1 vs. frequency — {short_name} (LR combined)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = cacc_dir / f"class_acc_vs_freq_{_safe_name(short_name)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Summary: average F1 per class across all backbones ---
    pre_avg = {
        cls: float(np.mean([pre_reports[b].get(cls, {}).get("f1-score", 0.0) for b in pre_reports]))
        for cls in classes
    }
    post_avg: dict[str, float] = {}
    if post_reports:
        post_avg = {
            cls: float(np.mean([post_reports[b].get(cls, {}).get("f1-score", 0.0) for b in post_reports]))
            for cls in classes
        }

    has_post_summary = bool(post_avg)
    n_series = 2 if has_post_summary else 1
    width = 0.8 / n_series
    offsets = [-(n_series - 1) * width / 2 + i * width for i in range(n_series)]

    fig, ax = plt.subplots(figsize=(max(14, n * 0.4), 6))
    ax.bar(x + offsets[0], [pre_avg[c] for c in classes], width,
           label="pre-proj (avg)", color="#4C72B0")
    if has_post_summary:
        ax.bar(x + offsets[1], [post_avg[c] for c in classes], width,
               label="post-proj (avg)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score (avg over backbones)")
    ax.set_xlabel("Class (ordered by sample count, high → low)")
    ax.set_title("Per-class F1 vs. frequency — summary (avg all backbones, LR combined)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = cacc_dir / "class_acc_vs_freq_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_master_comparison(results: dict, output_dir: Path):
    """One chart showing F1 macro for all 4 probe types side by side per backbone."""
    probe_keys = [
        ("LR per-dataset", "overall"),
        ("LR combined",    "lr_combined"),
        ("FC per-dataset", "fc_per_dataset_overall"),
        ("FC combined",    "fc_combined"),
    ]

    # Compute FC per-dataset mean F1 per backbone
    fc_pd_overall: dict[str, float] = {}
    for agml_data in results.get("fc_per_dataset", {}).values():
        for backbone, m in agml_data.items():
            fc_pd_overall.setdefault(backbone, []).append(m.get("f1_macro", 0.0))
    fc_pd_mean = {b: float(np.mean(vals)) for b, vals in fc_pd_overall.items()}
    results["fc_per_dataset_overall"] = {b: {"f1_macro": v} for b, v in fc_pd_mean.items()}

    backbones = sorted({
        b
        for key in ("overall", "lr_combined", "fc_combined")
        for b in results.get(key, {})
    } | set(fc_pd_mean))

    if not backbones:
        return

    n_probes = len(probe_keys)
    x = np.arange(len(backbones))
    width = 0.8 / n_probes
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(8, len(backbones) * 3), 5))
    for i, (probe_label, key) in enumerate(probe_keys):
        vals = [results.get(key, {}).get(b, {}).get("f1_macro", 0.0) for b in backbones]
        offset = (i - (n_probes - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=probe_label, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([b.split("/")[-1] for b in backbones], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Linear Probe — All 4 Experiments (F1 Macro)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = output_dir / "probe_comparison_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_summary(results: dict, output_dir: Path):
    """Bar chart comparing backbone F1 per task type using LR combined scores."""
    # Prefer by_task_lr_combined (consistent with LR combined classifier);
    # fall back to by_task (per-dataset mean) for old result files.
    by_task = results.get("by_task_lr_combined") or results.get("by_task", {})
    task_types = list(by_task.keys())
    if not task_types:
        return

    # Flatten nested {backbone: {f1_macro}} or {backbone: float} into floats
    def _f1(entry):
        return entry if isinstance(entry, float) else entry.get("f1_macro", 0.0)

    backbones = list(next(iter(by_task.values())).keys())
    x = np.arange(len(task_types))
    width = 0.8 / len(backbones)

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 2), 5))
    for i, backbone in enumerate(backbones):
        vals = [_f1(by_task[tt].get(backbone, 0.0)) for tt in task_types]
        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=backbone.split("/")[-1])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([tt.replace("_", " ") for tt in task_types], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Linear Probe — F1 by Task Type (LR combined)")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "summary_by_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.dino_model and args.dino_model.lower() == "none":
        args.dino_model = None
    if not hasattr(args, "models"):
        args.models = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_embedding_cache = getattr(args, "cache_embeddings", False)
    cache_dir = Path(getattr(args, "cache_dir", None) or (output_dir / "embedding_cache"))
    if use_embedding_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Embedding cache: {cache_dir}")

    use_dg = getattr(args, "dataset_source", "agml") == "digital_green"

    if use_dg:
        dg_config = getattr(args, "dg_config", None) or "outputs/data_generation/digital_green_datasets.yaml"
        dg_jsonl  = getattr(args, "dg_descriptions_jsonl", None) or "outputs/data_generation/dg_descriptions.jsonl"
        images_dir = Path(args.images_dir)
        datasets = active_datasets(Path(dg_config))
    else:
        datasets = active_datasets(Path(args.config))

    n_datasets = len(datasets)
    print(f"\nActive datasets: {n_datasets}")

    if not use_dg:
        prefetch_agml_datasets(datasets)

    # Phase 1: load each backbone once, extract embeddings for all datasets, unload
    # all_embeddings[backbone_key][agml_name] = (X, labels, train_idx, test_idx, le)
    all_embeddings: dict[str, dict[str, tuple]] = {}

    for ds_i, ds_cfg in enumerate(datasets):
        agml_name = ds_cfg.agml_name
        print(f"\n[{ds_i + 1}/{n_datasets}] Loading samples: {agml_name}")
        try:
            if use_dg:
                samples = load_dg_samples(images_dir, Path(dg_jsonl))
            else:
                samples = load_agml_samples(agml_name)
        except Exception as e:
            print(f"  [error] {e}")
            continue

        # Drop classes with fewer than 2 samples (need ≥1 train + ≥1 test)
        label_counts = Counter(s["label"] for s in samples)
        samples = [s for s in samples if label_counts[s["label"]] >= 2]
        if not samples:
            print(f"  [skip] {agml_name} — no classes with ≥2 samples after filtering")
            continue
        dropped = sum(1 for c, n in label_counts.items() if n < 2)
        if dropped:
            print(f"  [warn] {agml_name} — dropped {dropped} singleton class(es)")

        # Split per class: up to n_samples per class → train, all remaining → test
        # This ensures n_samples caps training data only; test uses all held-out images.
        rng = np.random.default_rng(42)
        by_class: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            by_class.setdefault(s["label"], []).append(i)
        train_idx_list, test_idx_list = [], []
        for indices in by_class.values():
            shuffled = rng.permutation(indices).tolist()
            n_train = min(args.n_samples, len(shuffled) - 1) if args.n_samples else len(shuffled) - 1
            n_train = max(n_train, 1)
            train_idx_list.extend(shuffled[:n_train])
            test_idx_list.extend(shuffled[n_train:])
        train_idx = np.array(sorted(train_idx_list))
        test_idx  = np.array(sorted(test_idx_list))

        if args.n_samples:
            n_cls = len(by_class)
            print(f"  Train: up to {args.n_samples} images/class ({len(train_idx)} total) | Test: all remaining ({len(test_idx)} total, {n_cls} classes)")

        labels_all = [s["label"] for s in samples]
        le = LabelEncoder()
        y_enc = le.fit_transform(labels_all)

        # Temporarily store samples with split indices keyed by dataset
        all_embeddings.setdefault("_samples", {})[agml_name] = (samples, labels_all, y_enc, train_idx, test_idx, le)

    # Save test split immediately after dataset loading so vision assessment can use it
    # even if the probe run is cancelled before backbone extraction completes.
    test_split_early: dict[str, dict[str, list[str]]] = {}
    for _dsname, (_samp, _, _, _, _tidx, _) in all_embeddings.get("_samples", {}).items():
        by_cls: dict[str, list[str]] = {}
        for i in _tidx:
            by_cls.setdefault(_samp[i]["label"], []).append(Path(_samp[i]["path"]).name)
        test_split_early[_dsname] = by_cls
    save_json(output_dir / "test_split.json", test_split_early)
    print(f"Test split saved to {output_dir / 'test_split.json'} ({len(test_split_early)} datasets)")

    # Count total backbone passes for progress display
    _backbone_names = (
        ([args.dino_model]    if args.dino_model    else [])
        + ([args.yolo_model]  if args.yolo_model    else [])
        + ([args.bioclip_model] if args.bioclip_model else [])
        + (args.models or [])
    )
    n_backbones   = len(_backbone_names)
    backbone_idx  = 0

    # DINOv3 pass
    if args.dino_model:
        from tqdm import tqdm
        backbone_idx += 1
        print(f"\n{'='*60}\n[Backbone {backbone_idx}/{n_backbones}] DINOv3: {args.dino_model}\n{'='*60}")
        dino_key = args.dino_model.split("/")[-1].replace(".pt", "")
        ds_items = list(all_embeddings["_samples"].items())
        ds_names = [dsn for dsn, _ in ds_items]
        all_embeddings[dino_key] = {}
        all_dino_cached = use_embedding_cache and all(
            load_cached_embeddings(cache_dir, dino_key, dsn, "emb") is not None for dsn in ds_names
        )
        if all_dino_cached:
            print(f"  [cache] All DINOv3 embeddings cached — skipping model load")
            for agml_name, (_, labels_all, y_enc, train_idx, test_idx, le) in ds_items:
                X, _ = load_cached_embeddings(cache_dir, dino_key, agml_name, "emb")
                all_embeddings[dino_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        else:
            dino_processor, dino_model = load_dino(args.dino_model, device)
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="DINOv3 datasets", unit="ds"):
                cached = load_cached_embeddings(cache_dir, dino_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, _ = cached
                    print(f"  [cache hit] {dino_key} / {agml_name}")
                else:
                    X, _ = extract_dino_embeddings(samples, dino_processor, dino_model, device)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, dino_key, agml_name, "emb", X, labels_all)
                all_embeddings[dino_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
            unload(dino_model)

    # YOLO pass
    if args.yolo_model:
        from tqdm import tqdm
        backbone_idx += 1
        print(f"\n{'='*60}\n[Backbone {backbone_idx}/{n_backbones}] YOLO: {args.yolo_model}\n{'='*60}")
        yolo_key = args.yolo_model.replace(".pt", "")
        ds_items = list(all_embeddings["_samples"].items())
        ds_names = [dsn for dsn, _ in ds_items]
        all_embeddings[yolo_key] = {}
        all_yolo_cached = use_embedding_cache and all(
            load_cached_embeddings(cache_dir, yolo_key, dsn, "emb") is not None for dsn in ds_names
        )
        if all_yolo_cached:
            print(f"  [cache] All YOLO embeddings cached — skipping model load")
            for agml_name, (_, labels_all, y_enc, train_idx, test_idx, le) in ds_items:
                X, _ = load_cached_embeddings(cache_dir, yolo_key, agml_name, "emb")
                all_embeddings[yolo_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        else:
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="YOLO datasets", unit="ds"):
                cached = load_cached_embeddings(cache_dir, yolo_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, _ = cached
                    print(f"  [cache hit] {yolo_key} / {agml_name}")
                else:
                    X = extract_yolo_embeddings(samples, args.yolo_model, device)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, yolo_key, agml_name, "emb", X, labels_all)
                all_embeddings[yolo_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)

    # BioCLIP pass
    if args.bioclip_model:
        from tqdm import tqdm
        backbone_idx += 1
        print(f"\n{'='*60}\n[Backbone {backbone_idx}/{n_backbones}] BioCLIP: {args.bioclip_model}\n{'='*60}")
        bioclip_key = args.bioclip_model.split("/")[-1]
        ds_items = list(all_embeddings["_samples"].items())
        ds_names = [dsn for dsn, _ in ds_items]
        all_embeddings[bioclip_key] = {}
        all_bioclip_cached = use_embedding_cache and all(
            load_cached_embeddings(cache_dir, bioclip_key, dsn, "emb") is not None for dsn in ds_names
        )
        if all_bioclip_cached:
            print(f"  [cache] All BioCLIP embeddings cached — skipping model load")
            for agml_name, (_, labels_all, y_enc, train_idx, test_idx, le) in ds_items:
                X, _ = load_cached_embeddings(cache_dir, bioclip_key, agml_name, "emb")
                all_embeddings[bioclip_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        else:
            bioclip_model, bioclip_preprocess = load_bioclip(args.bioclip_model, device)
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="BioCLIP datasets", unit="ds"):
                cached = load_cached_embeddings(cache_dir, bioclip_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, _ = cached
                    print(f"  [cache hit] {bioclip_key} / {agml_name}")
                else:
                    X, _ = extract_bioclip_embeddings(samples, bioclip_model, bioclip_preprocess, device)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, bioclip_key, agml_name, "emb", X, labels_all)
                all_embeddings[bioclip_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
            unload(bioclip_model)

    # VLM passes (one per --models entry, loaded and unloaded sequentially)
    layer_selections = _load_layer_selections()
    postproj_embeddings: dict[str, dict] = {}   # {backbone_postproj: {agml_name: (X, labels, ...)}}
    feature_dims: dict[str, dict] = {}          # {backbone: {"pre": int, "post": int}}
    postproj_only_outer = getattr(args, "postproj_only", False)
    for model_id in (args.models or []):
        from tqdm import tqdm
        backbone_idx += 1
        backbone_key = model_id.split("/")[-1]
        postproj_key = f"{backbone_key}_postproj"
        print(f"\n{'='*60}\n[Backbone {backbone_idx}/{n_backbones}] VLM: {model_id}\n{'='*60}")

        ds_items = list(all_embeddings["_samples"].items())
        ds_names = [dsn for dsn, _ in ds_items]

        # When caching is on, check if all datasets already have cached embeddings —
        # if so, skip loading the model entirely (saves ~20 min of load + extraction time).
        if use_embedding_cache:
            model_sels_check   = layer_selections.get(model_id, {})
            check_pre_path     = model_sels_check.get("pre")   # None → heuristic (can't verify w/o model)
            check_post_path    = model_sels_check.get("post")  # None → heuristic
            post_path_known    = check_post_path is not None

            all_pre_cached = postproj_only_outer or all(
                load_cached_embeddings(cache_dir, backbone_key, dsn, "pre", check_pre_path) is not None
                for dsn in ds_names
            )
            # Only check post caches if post path is explicitly configured in layer_selections;
            # otherwise we cannot know if post-projector exists without loading the model.
            all_post_cached = post_path_known and all(
                load_cached_embeddings(cache_dir, postproj_key, dsn, "post", check_post_path) is not None
                for dsn in ds_names
            )
            fully_cached = all_pre_cached and (all_post_cached or not post_path_known)
            if fully_cached:
                print(f"  [cache] All embeddings cached — skipping model load for {backbone_key}")
                all_embeddings[backbone_key] = {}
                postproj_embeddings[postproj_key] = {}
                for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in ds_items:
                    if not postproj_only_outer:
                        X, _ = load_cached_embeddings(cache_dir, backbone_key, agml_name, "pre", check_pre_path)
                        all_embeddings[backbone_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
                        if backbone_key not in feature_dims:
                            feature_dims[backbone_key] = {"pre": int(X.shape[1])}
                    if all_post_cached:
                        X_post, _ = load_cached_embeddings(cache_dir, postproj_key, agml_name, "post", check_post_path)
                        postproj_embeddings[postproj_key][agml_name] = (X_post, labels_all, y_enc, train_idx, test_idx, le)
                        if "post" not in feature_dims.get(backbone_key, {}):
                            feature_dims[backbone_key]["post"] = int(X_post.shape[1])
                continue

        processor, vlm, mm_root = load_gemma4(model_id, device, quantize=args.quantize)

        all_embeddings[backbone_key] = {}
        postproj_embeddings[postproj_key] = {}
        ds_items = list(all_embeddings["_samples"].items())

        # Resolve pre-projector hook module
        model_sels = layer_selections.get(model_id, {})
        if "pre" in model_sels:
            pre_layer_path: str | None = model_sels["pre"]
            pre_module = vlm.get_submodule(pre_layer_path)
            print(f"  Pre-projector (from layer_selections.yaml): {pre_layer_path}")
        else:
            pre_layer_path = None
            pre_module = mm_root.vision_tower
            print(f"  Pre-projector (heuristic fallback): vision_tower")
            if not layer_selections:
                print(f"  Tip: run 'streamlit run app/layer_explorer.py' to pick exact layers.")

        # Resolve post-projector hook module
        has_postproj = False
        post_module: torch.nn.Module | None = None
        post_layer_path: str | None = None
        if "post" in model_sels:
            try:
                post_layer_path = model_sels["post"]
                post_module = vlm.get_submodule(post_layer_path)
                has_postproj = True
                print(f"  Post-projector (from layer_selections.yaml): {post_layer_path}")
            except Exception as exc:
                print(f"  [warn] Could not resolve post-projector path '{model_sels['post']}': {exc}")
        else:
            proj, proj_name = resolve_projector(mm_root)
            if proj is not None:
                post_module = proj
                post_layer_path = proj_name
                has_postproj = True
                print(f"  Post-projector (heuristic fallback): {proj_name}")
            else:
                avail = [n for n, _ in mm_root.named_children()]
                print(f"  [warn] No post-projector found for {backbone_key}. Available: {avail}")

        postproj_only = getattr(args, "postproj_only", False)
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc=f"{backbone_key} datasets", unit="ds"):
            if not postproj_only:
                cached = load_cached_embeddings(cache_dir, backbone_key, agml_name, "pre", pre_layer_path) if use_embedding_cache else None
                if cached is not None:
                    X, _ = cached
                    print(f"  [cache hit] {backbone_key} / {agml_name} pre")
                else:
                    X, _ = extract_vlm_embeddings(samples, processor, vlm, pre_module, desc=backbone_key)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, backbone_key, agml_name, "pre", X, labels_all, pre_layer_path)
                all_embeddings[backbone_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
                if backbone_key not in feature_dims:
                    feature_dims[backbone_key] = {"pre": int(X.shape[1])}

            if has_postproj and post_module is not None:
                cached_post = load_cached_embeddings(cache_dir, postproj_key, agml_name, "post", post_layer_path) if use_embedding_cache else None
                if cached_post is not None:
                    X_post, _ = cached_post
                    print(f"  [cache hit] {postproj_key} / {agml_name} post")
                else:
                    X_post, _ = extract_vlm_embeddings_postproj(samples, processor, vlm, post_module, desc=postproj_key)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, postproj_key, agml_name, "post", X_post, labels_all, post_layer_path)
                postproj_embeddings[postproj_key][agml_name] = (X_post, labels_all, y_enc, train_idx, test_idx, le)
                if "post" not in feature_dims.get(backbone_key, {}):
                    feature_dims[backbone_key]["post"] = int(X_post.shape[1])

        # If all datasets were cache hits, skip VLM load on future runs
        unload(vlm)

    postproj_only = getattr(args, "postproj_only", False)
    available_backbones = [k for k in all_embeddings if k != "_samples" and (not postproj_only or all_embeddings[k])]
    if not available_backbones:
        print("No backbones to evaluate.")
        return

    fc_epochs     = getattr(args, "fc_epochs", 100)
    fc_lr         = getattr(args, "fc_lr", 1e-3)
    fc_batch_size = getattr(args, "fc_batch_size", 256)

    # Create subfolders up front so FC phases can write directly into them
    tsne_dir = output_dir / "tsne";             tsne_dir.mkdir(exist_ok=True)
    cm_dir   = output_dir / "confusion_matrices"; cm_dir.mkdir(exist_ok=True)
    f1_dir   = output_dir / "per_class_f1";    f1_dir.mkdir(exist_ok=True)
    loss_dir = output_dir / "loss_curves";      loss_dir.mkdir(exist_ok=True)

    all_experiments = getattr(args, "all_experiments", False)

    # Phase 2a: LR per-dataset (always runs — cheap after embeddings extracted)
    # Per-dataset plots (confusion matrix, t-SNE, per-class F1) gated behind --all_experiments
    results_per_dataset: dict[str, dict] = {}
    embeddings_store: dict[str, dict[str, tuple]] = defaultdict(dict)
    report_store: dict[str, dict[str, dict]] = defaultdict(dict)
    pred_store: dict[str, dict[str, tuple]] = defaultdict(dict)

    _lr_ds_names = list(all_embeddings["_samples"].keys())
    n_lr_ds = len(_lr_ds_names)
    for ds_probe_i, agml_name in enumerate(_lr_ds_names):
        _, _, _, train_idx, test_idx, le = all_embeddings["_samples"][agml_name]
        results_per_dataset[agml_name] = {"backbones": {}}

        for backbone in available_backbones:
            if agml_name not in all_embeddings.get(backbone, {}):
                continue
            X, labels_all, y_enc, train_idx, test_idx, le = all_embeddings[backbone][agml_name]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            print(f"\n=== LR probe [{ds_probe_i + 1}/{n_lr_ds}] {agml_name} / {backbone} ===")
            metrics, y_test_lbl, y_pred_lbl = run_probe(
                X[train_idx], X[test_idx], y_train, y_test, le, backbone
            )
            results_per_dataset[agml_name]["backbones"][backbone] = metrics
            embeddings_store[agml_name][backbone] = (X[test_idx], y_test_lbl)
            report_store[agml_name][backbone] = metrics["per_class_report_dict"]
            pred_store[agml_name][backbone] = (y_test_lbl, y_pred_lbl)

    # Phase 2b: FC per-dataset  (only with --all_experiments)
    fc_per_dataset: dict[str, dict] = {}
    fc_pred_store:  dict[str, dict[str, tuple]] = defaultdict(dict)
    fc_report_store: dict[str, dict[str, dict]] = defaultdict(dict)

    if all_experiments:
        for agml_name in all_embeddings["_samples"]:
            fc_per_dataset[agml_name] = {"backbones": {}}
            for backbone in available_backbones:
                if agml_name not in all_embeddings.get(backbone, {}):
                    continue
                X, labels_all, y_enc, train_idx, test_idx, le = all_embeddings[backbone][agml_name]
                y_train, y_test = y_enc[train_idx], y_enc[test_idx]

                print(f"\n=== FC probe (per-dataset): {agml_name} / {backbone} ===")
                metrics, y_test_lbl, y_pred_lbl = run_fc_probe(
                    X[train_idx], X[test_idx], y_train, y_test, le, backbone,
                    epochs=fc_epochs, lr=fc_lr, batch_size=fc_batch_size,
                )
                fc_per_dataset[agml_name]["backbones"][backbone] = {
                    k: v for k, v in metrics.items() if k not in ("train_losses", "val_losses")
                }
                fc_pred_store[agml_name][backbone]   = (y_test_lbl, y_pred_lbl)
                fc_report_store[agml_name][backbone] = metrics["per_class_report_dict"]
                plot_loss_curve(
                    metrics["train_losses"], metrics["val_losses"],
                    f"{agml_name}_{backbone}_fc", loss_dir,
                )

    # Phase 2c: LR combined (always runs — default mode)
    lr_combined: dict[str, dict] = {}
    lr_comb_pred_store:   dict[str, tuple] = {}   # backbone → (y_true, y_pred)
    lr_comb_embed_store:  dict[str, tuple] = {}   # backbone → (X_all, y_all_labels)
    lr_comb_report_store: dict[str, dict]  = {}
    for backbone in available_backbones:
        split = build_combined_split(all_embeddings, backbone)
        if split is None:
            continue
        X_tr, X_te, y_tr, y_te, le_comb, label_parts = split
        print(f"\n=== LR probe (combined): {backbone}  classes={len(le_comb.classes_)} ===")
        metrics, y_test_lbl, y_pred_lbl = run_probe(X_tr, X_te, y_tr, y_te, le_comb, backbone)
        lr_combined[backbone] = {k: v for k, v in metrics.items() if k != "per_class_report_dict"}
        lr_comb_pred_store[backbone]   = (y_test_lbl, y_pred_lbl)
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all_lbl = le_comb.inverse_transform(
            np.concatenate([y_tr, y_te], axis=0)
        ).tolist()
        lr_comb_embed_store[backbone]  = (X_all, y_all_lbl)
        lr_comb_report_store[backbone] = metrics["per_class_report_dict"]

        # Save intermediate metrics after each backbone so partial results survive crashes
        save_json(output_dir / "linear_probe_results_intermediate.json", {
            "lr_combined": lr_combined,
            "lr_combined_reports": lr_comb_report_store,
        })

    # Phase 2d: FC combined  (only with --all_experiments)
    fc_combined: dict[str, dict] = {}
    fc_comb_pred_store:   dict[str, tuple] = {}
    fc_comb_report_store: dict[str, dict]  = {}

    if all_experiments:
        for backbone in available_backbones:
            split = build_combined_split(all_embeddings, backbone)
            if split is None:
                continue
            X_tr, X_te, y_tr, y_te, le_comb, _ = split
            print(f"\n=== FC probe (combined): {backbone}  classes={len(le_comb.classes_)} ===")
            metrics, y_test_lbl, y_pred_lbl = run_fc_probe(
                X_tr, X_te, y_tr, y_te, le_comb, backbone,
                epochs=fc_epochs, lr=fc_lr, batch_size=fc_batch_size,
            )
            fc_combined[backbone] = {k: v for k, v in metrics.items() if k not in ("train_losses", "val_losses")}
            fc_comb_pred_store[backbone]   = (y_test_lbl, y_pred_lbl)
            fc_comb_report_store[backbone] = metrics["per_class_report_dict"]
            plot_loss_curve(
                metrics["train_losses"], metrics["val_losses"],
                f"combined_{backbone}_fc", loss_dir,
            )

    # Phase 2e: LR combined on post-projector embeddings (VLMs only — always runs)
    postproj_lr_combined: dict[str, dict] = {}          # {backbone_postproj: metrics}
    postproj_lr_comb_report_store: dict[str, dict] = {} # {backbone_postproj: per_class_report}
    postproj_lr_comb_embed_store:  dict[str, tuple] = {} # {backbone_postproj: (X_all, y_all_labels)}
    for postproj_key in postproj_embeddings:
        split = build_combined_split(postproj_embeddings, postproj_key)
        if split is None:
            continue
        X_tr, X_te, y_tr, y_te, le_comb, _ = split
        print(f"\n=== LR probe (post-proj combined): {postproj_key}  classes={len(le_comb.classes_)} ===")
        metrics, y_test_lbl, y_pred_lbl = run_probe(X_tr, X_te, y_tr, y_te, le_comb, postproj_key)
        postproj_lr_combined[postproj_key] = {k: v for k, v in metrics.items() if k != "per_class_report_dict"}
        postproj_lr_comb_report_store[postproj_key] = metrics["per_class_report_dict"]
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all_lbl = le_comb.inverse_transform(np.concatenate([y_tr, y_te], axis=0)).tolist()
        postproj_lr_comb_embed_store[postproj_key] = (X_all, y_all_lbl)

    # Phase 3: aggregate by task type and overall
    ds_to_task = {ds.agml_name: ds.task_type for ds in datasets}

    by_task_by_backbone: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall_by_backbone: dict[str, list[float]] = defaultdict(list)

    for agml_name, ds_results in results_per_dataset.items():
        tt = ds_to_task.get(agml_name)
        for backbone, metrics in ds_results["backbones"].items():
            f1 = metrics["f1_macro"]
            acc = metrics["accuracy"]
            overall_by_backbone[backbone].append(f1)
            if tt:
                by_task_by_backbone[tt][backbone].append(f1)

    # Build task-type breakdown from LR combined per-class F1 scores so the values
    # are consistent with the combined classifier rather than per-dataset means.
    # Map each class label → task type via datasets.yaml (last writer wins on overlap).
    _class_to_task: dict[str, str] = {}
    for agml_name, (samples, labels_all, *_) in all_embeddings["_samples"].items():
        tt = ds_to_task.get(agml_name)
        if tt:
            for lbl in set(labels_all):
                _class_to_task[lbl] = tt

    _skip = {"accuracy", "macro avg", "weighted avg"}
    by_task_combined_backbone: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for backbone, report in lr_comb_report_store.items():
        for cls, metrics in report.items():
            if cls in _skip:
                continue
            tt = _class_to_task.get(cls)
            if tt:
                by_task_combined_backbone[tt][backbone].append(metrics.get("f1-score", 0.0))

    by_task_lr_combined = {
        tt: {
            backbone: round(float(np.mean(vals)), 4)
            for backbone, vals in backbone_map.items()
        }
        for tt, backbone_map in by_task_combined_backbone.items()
    }

    results = {
        # LR per-dataset (always populated)
        "per_dataset": {
            agml_name: {
                backbone: {k: v for k, v in m.items() if k != "per_class_report_dict"}
                for backbone, m in ds_data["backbones"].items()
            }
            for agml_name, ds_data in results_per_dataset.items()
        },
        "by_task": {
            tt: {
                backbone: {
                    "f1_macro": round(float(np.mean(vals)), 4),
                    "n_datasets": len(vals),
                }
                for backbone, vals in backbone_map.items()
            }
            for tt, backbone_map in by_task_by_backbone.items()
        },
        "overall": {
            backbone: {
                "f1_macro": round(float(np.mean(vals)), 4),
                "accuracy": round(float(np.mean([
                    results_per_dataset[ds]["backbones"].get(backbone, {}).get("accuracy", 0.0)
                    for ds in results_per_dataset
                    if backbone in results_per_dataset[ds]["backbones"]
                ])), 4),
                "n_datasets": len(vals),
            }
            for backbone, vals in overall_by_backbone.items()
        },
        # FC per-dataset  (populated only with --all_experiments)
        "fc_per_dataset": {
            agml_name: {backbone: m for backbone, m in ds_data["backbones"].items()}
            for agml_name, ds_data in fc_per_dataset.items()
        },
        # LR combined (default — always runs)
        "lr_combined": lr_combined,
        "lr_combined_reports": lr_comb_report_store,
        "by_task_lr_combined": by_task_lr_combined,
        # FC combined  (populated only with --all_experiments)
        "fc_combined": fc_combined,
        # Post-projector LR combined (VLMs only — always runs)
        "postproj_lr_combined": postproj_lr_combined,
        "postproj_lr_combined_reports": postproj_lr_comb_report_store,
    }

    # Save results
    save_json(output_dir / "linear_probe_results.json", results)
    print(f"\nResults saved to {output_dir / 'linear_probe_results.json'}")

    # Save test split so vision assessment can evaluate the exact same images
    test_split: dict[str, dict[str, list[str]]] = {}
    for agml_name, (samples, _, _, _, test_idx, _) in all_embeddings.get("_samples", {}).items():
        by_class: dict[str, list[str]] = {}
        for i in test_idx:
            label = samples[i]["label"]
            fname = Path(samples[i]["path"]).name
            by_class.setdefault(label, []).append(fname)
        test_split[agml_name] = by_class
    save_json(output_dir / "test_split.json", test_split)
    print(f"Test split saved to {output_dir / 'test_split.json'}")

    # Summary tables
    print("\n=== LR Combined (all classes) ===")
    print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for backbone, m in results["lr_combined"].items():
        print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f}")

    if results["postproj_lr_combined"]:
        print("\n=== LR Combined — Post-projector (VLMs) ===")
        print(f"  {'Backbone':<40} {'F1 Macro':>10} {'Accuracy':>10}")
        print(f"  {'-'*62}")
        for backbone, m in results["postproj_lr_combined"].items():
            print(f"  {backbone:<40} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f}")

    if all_experiments:
        print("\n=== LR Per-Dataset — Overall Summary ===")
        print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10} {'N Datasets':>12}")
        print(f"  {'-'*64}")
        for backbone, m in results["overall"].items():
            print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} {m['n_datasets']:>12}")

        print("\n=== FC Combined (all classes) ===")
        print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10}")
        print(f"  {'-'*52}")
        for backbone, m in results["fc_combined"].items():
            print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f}")

    # Collect class counts from the first available backbone (same across all)
    _count_backbone = next(iter(available_backbones), None) if available_backbones else None
    combined_class_counts: Counter = (
        get_combined_class_counts(all_embeddings, _count_backbone)
        if _count_backbone else Counter()
    )

    # Plots
    print("\n=== Generating plots ===")

    # Overall backbone comparison + task-type summary (always — Phase 2a always runs now)
    if results["lr_combined"]:
        plot_overall_summary(results, output_dir)
    if results.get("by_task_lr_combined") or results.get("by_task"):
        plot_task_summary(results, output_dir)
    if all_experiments and results["overall"]:
        plot_master_comparison(results, output_dir)

    if postproj_lr_combined:
        plot_projector_before_after(results["lr_combined"], postproj_lr_combined, output_dir)

    # LR combined plots (always)
    for backbone, (X_te, y_te) in lr_comb_embed_store.items():
        plot_tsne(X_te, y_te, f"combined_{backbone}_lr", tsne_dir)
    for postproj_key, (X_all, y_all) in postproj_lr_comb_embed_store.items():
        plot_tsne(X_all, y_all, f"combined_{postproj_key}_lr", tsne_dir)
    if lr_comb_report_store:
        plot_per_class_f1(lr_comb_report_store, "combined_lr", f1_dir)
    for backbone, (y_true, y_pred) in lr_comb_pred_store.items():
        plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_lr", cm_dir)

    # Long-tail distribution
    if combined_class_counts:
        plot_long_tail(combined_class_counts, output_dir)

    # Per-class accuracy vs frequency (pre + post projector, LR combined)
    if lr_comb_report_store and combined_class_counts:
        plot_class_accuracy_vs_frequency(
            combined_class_counts,
            lr_comb_report_store,
            postproj_lr_comb_report_store,
            output_dir,
        )

    # --all_experiments plots
    if all_experiments:
        for agml_name in results_per_dataset:
            if len(report_store[agml_name]) > 1:
                plot_per_class_f1(report_store[agml_name], agml_name, f1_dir)
            for backbone, (y_true, y_pred) in pred_store[agml_name].items():
                plot_confusion_matrix(y_true, y_pred, f"{agml_name}_{backbone}_lr", cm_dir)
            for backbone, (X_test, y_test) in embeddings_store[agml_name].items():
                plot_tsne(X_test, y_test, f"{agml_name}_{backbone}", tsne_dir)
        for agml_name in fc_pred_store:
            if len(fc_report_store[agml_name]) > 1:
                plot_per_class_f1(fc_report_store[agml_name], f"{agml_name}_fc", f1_dir)
            for backbone, (y_true, y_pred) in fc_pred_store[agml_name].items():
                plot_confusion_matrix(y_true, y_pred, f"{agml_name}_{backbone}_fc", cm_dir)
        if fc_comb_report_store:
            plot_per_class_f1(fc_comb_report_store, "combined_fc", f1_dir)
        for backbone, (y_true, y_pred) in fc_comb_pred_store.items():
            plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_fc", cm_dir)

    # Update pipeline results CSV
    pipeline_csv = output_dir.parent / "pipeline_results.csv"
    for backbone, m in results["lr_combined"].items():
        update_pipeline_results(
            csv_path=pipeline_csv,
            model=backbone,
            model_type="vision_backbone",
            stage="linear_probe",
            metrics={"f1_macro": m["f1_macro"]},
        )

    # Write feature dimension summary
    if feature_dims:
        dim_lines = ["Feature dimensions (vision encoder output vs LM embedding space)\n", "=" * 60 + "\n"]
        for backbone, dims in sorted(feature_dims.items()):
            dim_lines.append(f"\n{backbone}\n")
            pre = dims.get("pre")
            post = dims.get("post")
            if pre is not None:
                dim_lines.append(f"  pre-projector  (vision encoder output): {pre}\n")
            if post is not None:
                dim_lines.append(f"  post-projector (LM embedding space):    {post}\n")
                if pre is not None:
                    dim_lines.append(f"  ratio post/pre:                         {post/pre:.3f}\n")
        dim_path = output_dir / "feature_dimensions.txt"
        dim_path.write_text("".join(dim_lines))
        print(f"Feature dimensions written to {dim_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe across all active AgML datasets."
    )
    parser.add_argument(
        "--dataset_source", default="agml", choices=["agml", "digital_green"],
        help="'agml' uses --config; 'digital_green' uses --dg_config + --dg_descriptions_jsonl + --images_dir",
    )
    parser.add_argument(
        "--dg_config", default="outputs/data_generation/digital_green_datasets.yaml",
        help="Digital Green datasets YAML (used when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--dg_descriptions_jsonl", default="outputs/data_generation/dg_descriptions.jsonl",
        help="Digital Green descriptions JSONL for filename→label mapping (used when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--images_dir", default=None,
        help="Local directory of Digital Green images (required when --dataset_source digital_green)",
    )
    parser.add_argument(
        "--config", default="src/config/datasets.yaml",
        help="Path to datasets.yaml",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="One or more VLM models (HF ID or local path). Each is loaded, evaluated, then unloaded.",
    )
    parser.add_argument(
        "--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 model ID. Pass 'none' to skip.",
    )
    parser.add_argument(
        "--bioclip_model", type=str, default=None,
        help="BioCLIP model HF repo ID (default: skipped). Pass 'imageomics/bioclip' to enable.",
    )
    parser.add_argument(
        "--yolo_model", type=str, default="yolo11x-cls.pt",
        help="YOLO classification model. Pass 'none' to skip.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/linear_probe/",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Max images per class for training (seeded). Test uses all remaining images. Default: all.",
    )
    parser.add_argument(
        "--quantize", choices=["4bit", "8bit"], default=None,
        help="Load VLMs in 4-bit or 8-bit quantization (requires bitsandbytes). Use 4bit for 48GB GPUs.",
    )
    parser.add_argument(
        "--fc_epochs", type=int, default=100,
        help="Training epochs for FC probe (default: 100)",
    )
    parser.add_argument(
        "--fc_lr", type=float, default=1e-3,
        help="Learning rate for FC probe Adam optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--fc_batch_size", type=int, default=256,
        help="Batch size for FC probe training (default: 256)",
    )
    parser.add_argument(
        "--postproj_only", action="store_true",
        help="For VLMs: extract only post-projector embeddings (skips vision-tower pre-projector pass). "
             "Runs LR on post-projector only; skips LR/FC per-dataset and combined phases for VLMs.",
    )
    parser.add_argument(
        "--all_experiments", action="store_true",
        help="Run all 4 probe variants (LR per-dataset, LR combined, FC per-dataset, FC combined). "
             "Default runs LR combined only.",
    )
    parser.add_argument(
        "--replot", action="store_true",
        help="Skip inference — load linear_probe_results.json from --output_dir and regenerate plots.",
    )
    parser.add_argument(
        "--cache_embeddings", action="store_true",
        help="Cache extracted embeddings to disk (.npz). On re-runs, cached datasets are loaded "
             "instantly and — when all datasets are cached — the VLM is not loaded at all.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory to store embedding cache files. Defaults to <output_dir>/embedding_cache/.",
    )
    args = parser.parse_args()

    if args.yolo_model and args.yolo_model.lower() == "none":
        args.yolo_model = None
    if args.bioclip_model and args.bioclip_model.lower() == "none":
        args.bioclip_model = None

    if args.replot:
        import json as _json
        output_dir = Path(args.output_dir)
        results_path = output_dir / "linear_probe_results.json"
        if not results_path.exists():
            # Fall back to intermediate file if full results not yet saved
            results_path = output_dir / "linear_probe_results_intermediate.json"
        if not results_path.exists():
            raise FileNotFoundError(f"No results file found in {output_dir}")
        results = _json.loads(results_path.read_text())
        output_dir.mkdir(parents=True, exist_ok=True)

        pre  = results.get("lr_combined", {})
        post = results.get("postproj_lr_combined", {})
        pre_reports  = results.get("lr_combined_reports", {})
        post_reports = results.get("postproj_lr_combined_reports", {})

        # Pre/post projector comparison
        if post:
            plot_projector_before_after(pre, post, output_dir)

        # Rebuild class counts from lr_combined_reports support field
        skip_keys = {"accuracy", "macro avg", "weighted avg"}
        counts: Counter = Counter()
        for report in pre_reports.values():
            for cls, m in report.items():
                if cls not in skip_keys:
                    counts[cls] += round(m.get("support", 0))

        if counts:
            plot_long_tail(counts, output_dir)
            if pre_reports:
                plot_class_accuracy_vs_frequency(counts, pre_reports, post_reports, output_dir)

        # Overall backbone comparison (LR combined)
        if results.get("lr_combined"):
            plot_overall_summary(results, output_dir)
        if results.get("by_task_lr_combined") or results.get("by_task"):
            plot_task_summary(results, output_dir)

        print(f"Plots saved to {output_dir}/")
    else:
        main(args)
