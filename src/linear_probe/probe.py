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
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

# Kimi-VL's custom modeling code (pulled in via trust_remote_code) still does
# `from transformers.utils.import_utils import is_torch_fx_available`, a name
# newer transformers removed — torch.fx has been unconditionally importable
# for a while now, so the availability check itself became dead weight and
# was deleted upstream. Shim it back as a trivial True (matches reality: the
# import always succeeds) rather than pinning an older transformers, which
# would risk breaking the other, newer-architecture backbones this script
# also loads (Qwen3.5, Gemma4). No-op for any repo that doesn't need it.
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
    import torch.fx  # noqa: F401  (import succeeding IS the check)
    transformers.utils.import_utils.is_torch_fx_available = lambda: True
    transformers.utils.is_torch_fx_available = lambda: True

from src.common.dataset_config import (
    active_datasets, DatasetConfig, subsample_class_balanced,
    load_manifest, load_hf_samples, missing_from_manifest,
)
from src.common.io import save_json, update_pipeline_results
from src.common.progress import ProgressBar

# This module (and check_vision_towers.py, which imports load_gemma4 from
# here) loads several backbones one after another in the same process —
# same repeated-reload pattern that hits a real bug on farm compute nodes
# without reliable outbound internet: transformers/huggingface_hub still
# issues a HEAD request on every from_pretrained call to check cache
# freshness, even when every needed file is already local. That request
# fails ("Network is unreachable"), and the retry path leaves the underlying
# httpx client closed, crashing the NEXT load attempt with "Cannot send a
# request, as the client has been closed" (see run_verifier_cot.py for the
# same fix applied to the same bug). HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
# skip that check and trust the local cache — safe for any model that has
# already been loaded successfully once (by this script or a prior run);
# a genuinely never-downloaded model instead gets a clear "not in cache"
# error rather than this confusing crash. setdefault() so an operator's own
# explicit choice isn't silently overridden.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Path to the layer selections config written by app/layer_explorer.py — lives
# alongside probe_config.yaml in this folder so all linear-probe-specific
# config is in one place, easy to find/edit.
_LAYER_SELECTIONS_PATH = Path(__file__).parent / "layer_selections.yaml"


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
# load_hf_samples / load_manifest / missing_from_manifest now live in
# src/common/dataset_config.py — shared across every stage that needs to
# materialize images from a scripts/agml_hf/download_and_sample.py manifest,
# not reimplemented per-script. The old agml-pip-package-based loader that
# used to live here is gone; it had no knowledge of datasets.yaml's
# HF-sourced crop/task_type/class_overrides or dataset/selected_images.json
# at all, so it was silently loading different (or no longer available) data
# than the rest of this pipeline uses.


def load_dg_samples(images_dir: Path, dg_descriptions_jsonl: Path) -> list[dict]:
    """Load Digital Green samples from local image directory + JSONL metadata.

    Each JSONL description record maps image_names → class_label (diagnosis).
    Returns the same [{"path": str, "label": str}, ...] shape as
    src.common.dataset_config.load_hf_samples().
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
    by_class = subsample_class_balanced(samples, lambda s: s["label"], n_per_class, seed)
    selected = sorted(i for indices in by_class.values() for i in indices)
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


def resolve_heuristic_pre_module(mm_root) -> tuple[torch.nn.Module, str | None, str]:
    """Pick the heuristic (no layer_selections.yaml entry) pre-projector hook
    module for a resolved mm_root. Returns (module, cache_layer_path, description).

    Default is mm_root.vision_tower's own output — correct for architectures
    like Gemma4, whose vision_tower ends on genuine vision features with a
    separate top-level projector afterward. cache_layer_path is None here,
    same as before this function existed, so existing Gemma-style embedding
    caches stay valid.

    Qwen-style towers are different: root.visual (aliased to vision_tower by
    resolve_mm_root) has both a `.blocks` (the actual transformer stack) and
    a `.merger` (projects into the LM's embedding space) as attributes on the
    SAME module — so hooking vision_tower's own output means hooking
    whatever ran last inside it, which for Qwen is the merger's own final
    Linear layer, not pre-projector vision features at all (confirmed via
    check_vision_towers.py: both Qwen3.5-9B/27B's heuristic hook landed on
    `merger.linear_fc2`, a raw unnormalized Linear). Auto-detected structurally
    (has both `.blocks` and `.merger`) rather than by matching "qwen" in the
    model name, so this also covers any future/other model sharing the same
    tower shape without needing a name-based special case. cache_layer_path is
    a real, distinct string here — deliberately, so any embeddings already
    cached under the old (wrong) merger-based heuristic get invalidated
    instead of silently reused.

    A manual layer_selections.yaml entry always overrides this — this
    function only runs when there isn't one."""
    vision_tower = mm_root.vision_tower
    if hasattr(vision_tower, "blocks") and hasattr(vision_tower, "merger"):
        blocks = vision_tower.blocks
        idx = len(blocks) - 1
        path = f"vision_tower.blocks.{idx}"
        desc = f"{path} (auto: Qwen-style merger detected in vision_tower — using the last pre-merger transformer block instead of the merger's own output)"
        return blocks[idx], path, desc
    return vision_tower, None, "vision_tower"


def extract_vlm_embeddings(
    samples, processor, model, hook_module: torch.nn.Module, desc: str = "VLM embeddings"
) -> tuple[np.ndarray, list[str]]:
    """
    Hook hook_module, run a forward pass per image, mean-pool the captured feature map.

    hook_module is resolved by the caller — either from layer_selections.yaml via
    model.get_submodule(path), or from the heuristic resolve_mm_root fallback.
    """
    model_device = next(model.parameters()).device
    embeddings, labels = [], []

    for sample in ProgressBar(samples, desc=f"  {desc}"):
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
    model_device = next(model.parameters()).device
    embeddings, labels = [], []

    for sample in ProgressBar(samples, desc=f"  {desc}"):
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


def load_gemma4(model_id: str, device: str, quantize: str | None = None, trust_remote_code: bool = True):
    """trust_remote_code defaults to True here so models that ship custom
    modeling code on their HF repo (e.g. Kimi-VL, Nemotron) load without an
    interactive prompt — this executes arbitrary Python from that repo, so
    it's a deliberate default, not an oversight; pass False (--no-trust-
    remote-code) to go back to transformers' normal opt-in-only behavior."""
    local = os.path.abspath(model_id)
    if os.path.exists(local):
        model_id = local
    print(f"Loading VLM: {model_id}" + (f" [{quantize}]" if quantize else ""))
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    load_kwargs: dict = {"device_map": "auto", "trust_remote_code": trust_remote_code}
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

    # Some custom-code VLM repos (observed with Kimi-VL, Nemotron-Omni) don't
    # register their model class under AutoModelForImageTextToText at all —
    # the checkpoint and its custom config load fine, this auto-class just
    # doesn't recognize that config type. Different repos land under
    # different fallback auto-classes (generic AutoModel, or — same pattern
    # DeepSeek-VL2 uses — AutoModelForCausalLM), so try each in turn and
    # re-raise the FIRST (most informative) "Unrecognized configuration
    # class" error if every tier fails, rather than whatever the last
    # attempted auto-class produced for a genuinely incompatible checkpoint.
    _AUTO_CLASSES = (AutoModelForImageTextToText, AutoModel, AutoModelForCausalLM)
    model = None
    first_error: ValueError | None = None
    for _auto_cls in _AUTO_CLASSES:
        try:
            model = _auto_cls.from_pretrained(model_id, **load_kwargs)
            if _auto_cls is not _AUTO_CLASSES[0]:
                print(f"  [note] loaded via {_auto_cls.__name__} "
                      f"(AutoModelForImageTextToText didn't recognize this config)")
            break
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            if first_error is None:
                first_error = e
            continue
    if model is None:
        assert first_error is not None
        raise first_error
    model.eval()
    mm_root = resolve_mm_root(model)
    vision_device = next(mm_root.vision_tower.parameters()).device
    print(f"  Vision tower: {type(mm_root.vision_tower).__name__} on {vision_device}")
    return processor, model, mm_root


# ── YOLO backbone ──────────────────────────────────────────────────────────────

def extract_yolo_embeddings(samples, yolo_model_name: str, device: str) -> np.ndarray:
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
        captured.append(t.detach().mean(dim=(-2, -1)).squeeze(0).cpu().float())

    penultimate = list(yolo.model.model.children())[-2]
    handle = penultimate.register_forward_hook(_hook)

    embeddings = []
    with torch.no_grad():
        for sample in ProgressBar(samples, desc="  YOLO embeddings"):
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
    embeddings, labels = [], []
    for sample in ProgressBar(samples, desc="  DINOv3 embeddings"):
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
    embeddings, labels = [], []
    for sample in ProgressBar(samples, desc="  BioCLIP embeddings"):
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

    train_losses = []
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

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  train_loss={train_losses[-1]:.4f}")

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
    }
    return metrics, y_test_labels, y_pred_labels


# ── Combined embeddings builder ────────────────────────────────────────────────

def get_combined_class_counts(all_embeddings: dict, backbone: str) -> Counter:
    """Raw class counts across all datasets for one backbone (before any filtering)."""
    counts: Counter = Counter()
    for _, (_, labels_all, *_) in all_embeddings.get(backbone, {}).items():
        counts.update(labels_all)
    return counts


def _split_from_labels(
    X: np.ndarray, labels_all: list[str], n_samples: int | None = None
) -> tuple:
    """Rebuild y_enc / train_idx / test_idx / le from X and labels_all.

    Used when loading embeddings from cache so that X and labels_all are
    always aligned (cache may have a different row count than _samples on a
    different machine).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(labels_all)
    rng = np.random.default_rng(42)
    by_class: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels_all):
        by_class.setdefault(lbl, []).append(i)
    _n = n_samples if n_samples is not None else 10
    train_idx_list, test_idx_list = [], []
    for indices in by_class.values():
        shuffled = rng.permutation(indices).tolist()
        n_train = min(_n, len(shuffled) - 1)
        n_train = max(n_train, 1)
        train_idx_list.extend(shuffled[:n_train])
        test_idx_list.extend(shuffled[n_train:])
    train_idx = np.array(sorted(train_idx_list))
    test_idx  = np.array(sorted(test_idx_list))
    return X, labels_all, y_enc, train_idx, test_idx, le


def build_combined_split(
    all_embeddings: dict, backbone: str, n_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, list[str]] | None:
    """Concatenate embeddings from all datasets for one backbone into a single split.

    Uses n_samples per class for training (default 10), rest goes to test —
    consistent with the per-dataset split logic.
    """
    X_parts, label_parts = [], []
    for agml_name, (X, labels_all, *_) in all_embeddings.get(backbone, {}).items():
        X_parts.append(X)
        label_parts.extend(labels_all)

    if not X_parts:
        return None

    X_all = np.concatenate(X_parts, axis=0)

    # Drop classes with fewer than 2 samples
    counts = Counter(label_parts)
    keep = np.array([counts[label_parts[i]] >= 2 for i in range(len(label_parts))])
    X_all = X_all[keep]
    label_parts = [label_parts[i] for i in range(len(label_parts)) if keep[i]]

    # Per-class split: n_samples → train, rest → test
    rng = np.random.default_rng(42)
    by_class: dict[str, list[int]] = {}
    for i, lbl in enumerate(label_parts):
        by_class.setdefault(lbl, []).append(i)
    train_idx_list, test_idx_list = [], []
    for indices in by_class.values():
        shuffled = rng.permutation(indices).tolist()
        n_train = min(n_samples, len(shuffled) - 1)
        n_train = max(n_train, 1)
        train_idx_list.extend(shuffled[:n_train])
        test_idx_list.extend(shuffled[n_train:])
    train_idx = np.array(sorted(train_idx_list))
    test_idx  = np.array(sorted(test_idx_list))

    le = LabelEncoder()
    y_all = le.fit_transform(label_parts)
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
    """Bar chart comparing backbone F1; uses LR combined if available, falls back to FC combined."""
    combined = results.get("lr_combined") or results.get("fc_combined", {})
    probe_label = "LR combined" if results.get("lr_combined") else "FC combined"
    backbones = list(combined.keys())
    if not backbones:
        return
    f1_macros = [combined[b]["f1_macro"] for b in backbones]
    accuracies = [combined[b]["accuracy"] for b in backbones]

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
    ax.set_title(f"Linear Probe — Overall Backbone Comparison ({probe_label}, all classes)")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "summary_overall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_curve(train_losses: list[float], label: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"FC Probe Training Loss — {label}")
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
    ax.set_ylabel("F1 Macro")
    ax.set_title("Projector Separability: Pre vs Post-Projector\n"
                 "Ratio > 1 = projector preserves/improves class structure  "
                 "Ratio < 1 = projector destroys class structure")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = output_dir / "projector_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_projector_task_comparison(
    by_task_pre: dict[str, dict[str, float]],   # {task: {backbone: f1}}
    by_task_post: dict[str, dict[str, float]],  # {task: {backbone_postproj: f1}}
    output_dir: Path,
):
    """Per-task grouped bar chart: pre vs post-projector F1 for each VLM backbone."""
    # Collect VLM backbones that have both pre and post entries
    backbones = []
    for b in next(iter(by_task_pre.values()), {}).keys():
        post_key = f"{b}_postproj"
        if any(post_key in by_task_post.get(tt, {}) for tt in by_task_pre):
            backbones.append(b)
    if not backbones or not by_task_pre:
        return

    task_types = sorted(by_task_pre.keys())
    x = np.arange(len(task_types))
    n_backbones = len(backbones)
    width = 0.8 / (n_backbones * 2)

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 3), 5))
    colors_pre  = plt.cm.tab10.colors[:n_backbones]
    colors_post = [tuple(min(c * 1.4, 1.0) for c in col) for col in colors_pre]

    for i, backbone in enumerate(backbones):
        post_key = f"{backbone}_postproj"
        short = backbone.split("/")[-1]
        pre_vals  = [by_task_pre.get(tt, {}).get(backbone, 0.0)  for tt in task_types]
        post_vals = [by_task_post.get(tt, {}).get(post_key, 0.0) for tt in task_types]
        base = (i * 2 - (n_backbones - 1)) * width
        b1 = ax.bar(x + base,         pre_vals,  width, label=f"{short} pre",  color=colors_pre[i])
        b2 = ax.bar(x + base + width,  post_vals, width, label=f"{short} post", color=colors_post[i], alpha=0.8)
        for bar, v in list(zip(b1, pre_vals)) + list(zip(b2, post_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([tt.replace("_", " ") for tt in task_types], rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Projector Separability by Task Type: Pre vs Post-Projector (FC combined)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = output_dir / "projector_by_task.png"
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
    """Bar chart comparing backbone F1 per task type; LR combined preferred, FC combined fallback."""
    by_task = (
        results.get("by_task_lr_combined")
        or results.get("by_task_fc_combined")
        or results.get("by_task", {})
    )
    probe_label = (
        "LR combined" if results.get("by_task_lr_combined")
        else "FC combined" if results.get("by_task_fc_combined")
        else "per-dataset mean"
    )
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
    ax.set_title(f"Linear Probe — F1 by Task Type ({probe_label})")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "summary_by_task.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _plots_only_mode(args, output_dir: Path) -> None:
    """Regenerate all plots from saved probe results without rerunning embeddings/training."""
    import json

    tsne_dir = output_dir / "tsne";              tsne_dir.mkdir(exist_ok=True, parents=True)
    cm_dir   = output_dir / "confusion_matrices"; cm_dir.mkdir(exist_ok=True, parents=True)
    f1_dir   = output_dir / "per_class_f1";     f1_dir.mkdir(exist_ok=True, parents=True)

    results_path = output_dir / "linear_probe_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the full probe first.")
        return

    with open(results_path) as _f:
        results = json.load(_f)

    # Reconstruct fc_comb_report_store from saved results
    fc_comb_report_store: dict[str, dict] = {
        b: m["per_class_report_dict"]
        for b, m in results.get("fc_combined", {}).items()
        if "per_class_report_dict" in m
    }

    # Load pred store
    preds_path = output_dir / "fc_comb_preds.json"
    fc_comb_pred_store: dict[str, tuple] = {}
    if preds_path.exists():
        with open(preds_path) as _f:
            _raw = json.load(_f)
        fc_comb_pred_store = {b: (v["y_true"], v["y_pred"]) for b, v in _raw.items()}
    else:
        print(f"  [warn] {preds_path} not found — confusion matrices skipped")

    # Load embed store (for t-SNE)
    fc_comb_embed_store: dict[str, tuple] = {}
    embed_dir = output_dir / "fc_comb_embeds"
    for backbone in results.get("fc_combined", {}):
        npz_path = embed_dir / f"{_safe_name(backbone)}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            fc_comb_embed_store[backbone] = (data["X"], list(data["y_labels"]))
        else:
            print(f"  [warn] embed file missing for {backbone} — t-SNE skipped")

    print("\n=== Generating plots (plots_only mode) ===")

    if results.get("lr_combined") or results.get("fc_combined"):
        plot_overall_summary(results, output_dir)
    if results.get("by_task_lr_combined") or results.get("by_task_fc_combined") or results.get("by_task"):
        plot_task_summary(results, output_dir)

    for backbone, (X_te, y_te) in fc_comb_embed_store.items():
        plot_tsne(X_te, y_te, f"combined_{backbone}_fc", tsne_dir)
    if fc_comb_report_store:
        plot_per_class_f1(fc_comb_report_store, "combined_fc", f1_dir)
    for backbone, (y_true, y_pred) in fc_comb_pred_store.items():
        plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_fc", cm_dir)

    lr_comb_report_store = results.get("lr_combined_reports", {})
    if lr_comb_report_store:
        plot_per_class_f1(lr_comb_report_store, "combined_lr", f1_dir)

    if results.get("lr_combined") and results.get("postproj_lr_combined"):
        plot_projector_before_after(results["lr_combined"], results["postproj_lr_combined"], output_dir)
    elif results.get("fc_combined") and results.get("postproj_fc_combined"):
        plot_projector_before_after(results["fc_combined"], results["postproj_fc_combined"], output_dir)
    if results.get("by_task_fc_combined") and results.get("by_task_postproj_fc_combined"):
        plot_projector_task_comparison(
            results["by_task_fc_combined"], results["by_task_postproj_fc_combined"], output_dir
        )

    print(f"\nPlots saved to {output_dir}/")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.dino_model and args.dino_model.lower() == "none":
        args.dino_model = None
    if not hasattr(args, "models"):
        args.models = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "plots_only", False):
        _plots_only_mode(args, output_dir)
        return

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

    manifest = None
    if not use_dg:
        manifest_path = Path(getattr(args, "manifest", None) or "dataset/selected_images.json")
        images_root = Path(getattr(args, "images_root", None) or "dataset/images")
        manifest = load_manifest(manifest_path)
        gaps = missing_from_manifest(datasets, manifest)
        if gaps:
            print(f"  [warn] {len(gaps)}/{n_datasets} active dataset(s) not in {manifest_path} "
                  f"(never downloaded, or download failed — see that manifest's summary.txt): "
                  f"{gaps}")

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
                samples = load_hf_samples(agml_name, manifest, images_root)
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

        # Split per class: up to n_samples per class → train, all remaining → test.
        # Default is 10 per class; pass --n_samples to override.
        # Combined probe always uses 80/20 stratified split regardless of this value.
        _n_samples_per_class = args.n_samples if args.n_samples is not None else 10
        rng = np.random.default_rng(42)
        by_class: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            by_class.setdefault(s["label"], []).append(i)
        train_idx_list, test_idx_list = [], []
        for indices in by_class.values():
            shuffled = rng.permutation(indices).tolist()
            n_train = min(_n_samples_per_class, len(shuffled) - 1)
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
            for agml_name, _ in ds_items:
                X, cached_labels = load_cached_embeddings(cache_dir, dino_key, agml_name, "emb")
                all_embeddings[dino_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
        else:
            dino_processor, dino_model = load_dino(args.dino_model, device)
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in ProgressBar(ds_items, desc="DINOv3 datasets"):
                cached = load_cached_embeddings(cache_dir, dino_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, cached_labels = cached
                    print(f"  [cache hit] {dino_key} / {agml_name}")
                    all_embeddings[dino_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
                else:
                    X, _ = extract_dino_embeddings(samples, dino_processor, dino_model, device)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, dino_key, agml_name, "emb", X, labels_all)
                    all_embeddings[dino_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
            unload(dino_model)

    # YOLO pass
    if args.yolo_model:
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
            for agml_name, _ in ds_items:
                X, cached_labels = load_cached_embeddings(cache_dir, yolo_key, agml_name, "emb")
                all_embeddings[yolo_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
        else:
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in ProgressBar(ds_items, desc="YOLO datasets"):
                cached = load_cached_embeddings(cache_dir, yolo_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, cached_labels = cached
                    print(f"  [cache hit] {yolo_key} / {agml_name}")
                    all_embeddings[yolo_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
                else:
                    X = extract_yolo_embeddings(samples, args.yolo_model, device)
                    if use_embedding_cache:
                        save_cached_embeddings(cache_dir, yolo_key, agml_name, "emb", X, labels_all)
                    all_embeddings[yolo_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)

    # BioCLIP pass
    if args.bioclip_model:
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
            for agml_name, _ in ds_items:
                X, cached_labels = load_cached_embeddings(cache_dir, bioclip_key, agml_name, "emb")
                all_embeddings[bioclip_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
        else:
            bioclip_model, bioclip_preprocess = load_bioclip(args.bioclip_model, device)
            for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in ProgressBar(ds_items, desc="BioCLIP datasets"):
                cached = load_cached_embeddings(cache_dir, bioclip_key, agml_name, "emb") if use_embedding_cache else None
                if cached is not None:
                    X, cached_labels = cached
                    print(f"  [cache hit] {bioclip_key} / {agml_name}")
                    all_embeddings[bioclip_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
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

        processor, vlm, mm_root = load_gemma4(
            model_id, device, quantize=args.quantize,
            trust_remote_code=getattr(args, "trust_remote_code", True),
        )

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
            pre_module, pre_layer_path, pre_desc = resolve_heuristic_pre_module(mm_root)
            print(f"  Pre-projector (heuristic fallback): {pre_desc}")
            if not layer_selections:
                print(f"  Tip: add an entry to {_LAYER_SELECTIONS_PATH} to pick exact layers.")

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
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in ProgressBar(ds_items, desc=f"{backbone_key} datasets"):
            if not postproj_only:
                cached = load_cached_embeddings(cache_dir, backbone_key, agml_name, "pre", pre_layer_path) if use_embedding_cache else None
                if cached is not None:
                    X, cached_labels = cached
                    print(f"  [cache hit] {backbone_key} / {agml_name} pre")
                    all_embeddings[backbone_key][agml_name] = _split_from_labels(X, cached_labels, args.n_samples)
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
                    X_post, cached_labels_post = cached_post
                    print(f"  [cache hit] {postproj_key} / {agml_name} post")
                    postproj_embeddings[postproj_key][agml_name] = _split_from_labels(X_post, cached_labels_post, args.n_samples)
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
    run_per_dataset = getattr(args, "per_dataset", False)

    # Phase 2a: LR per-dataset (opt-in via --per_dataset)
    results_per_dataset: dict[str, dict] = {}
    embeddings_store: dict[str, dict[str, tuple]] = defaultdict(dict)
    report_store: dict[str, dict[str, dict]] = defaultdict(dict)
    pred_store: dict[str, dict[str, tuple]] = defaultdict(dict)

    if run_per_dataset:
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
    else:
        print("\n[skip] LR per-dataset — pass --per_dataset to enable")

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
                    k: v for k, v in metrics.items() if k != "train_losses"
                }
                fc_pred_store[agml_name][backbone]   = (y_test_lbl, y_pred_lbl)
                fc_report_store[agml_name][backbone] = metrics["per_class_report_dict"]
                plot_loss_curve(
                    metrics["train_losses"],
                    f"{agml_name}_{backbone}_fc", loss_dir,
                )

    _n_combined = args.n_samples if args.n_samples is not None else 10

    # Phase 2c: FC combined (default mode — GPU-accelerated, fixed epoch budget)
    fc_combined: dict[str, dict] = {}
    fc_comb_pred_store:   dict[str, tuple] = {}
    fc_comb_report_store: dict[str, dict]  = {}
    fc_comb_embed_store:  dict[str, tuple] = {}  # {backbone: (X_te_np, y_test_labels)}

    for backbone in available_backbones:
        split = build_combined_split(all_embeddings, backbone, n_samples=_n_combined)
        if split is None:
            continue
        X_tr, X_te, y_tr, y_te, le_comb, _ = split
        print(f"\n=== FC probe (combined): {backbone}  classes={len(le_comb.classes_)}  train/class={_n_combined} ===")
        metrics, y_test_lbl, y_pred_lbl = run_fc_probe(
            X_tr, X_te, y_tr, y_te, le_comb, backbone,
            epochs=fc_epochs, lr=fc_lr, batch_size=fc_batch_size,
        )
        fc_combined[backbone] = {k: v for k, v in metrics.items() if k != "train_losses"}
        fc_comb_pred_store[backbone]   = (y_test_lbl, y_pred_lbl)
        fc_comb_report_store[backbone] = metrics["per_class_report_dict"]
        fc_comb_embed_store[backbone]  = (X_te, y_test_lbl)
        plot_loss_curve(
            metrics["train_losses"],
            f"combined_{backbone}_fc", loss_dir,
        )
        save_json(output_dir / "linear_probe_results_intermediate.json", {
            "fc_combined": fc_combined,
            "fc_combined_reports": fc_comb_report_store,
        })

    # Persist FC combined stores so --plots_only can regenerate plots without retraining
    _embed_dir = output_dir / "fc_comb_embeds"
    _embed_dir.mkdir(exist_ok=True)
    for _b, (_X, _y) in fc_comb_embed_store.items():
        np.savez(str(_embed_dir / f"{_safe_name(_b)}.npz"), X=_X, y_labels=np.array(_y, dtype=object))
    save_json(output_dir / "fc_comb_preds.json", {
        b: {"y_true": yt, "y_pred": yp}
        for b, (yt, yp) in fc_comb_pred_store.items()
    })

    # Phase 2d: LR combined (opt-in via --lr_combined)
    lr_combined: dict[str, dict] = {}
    lr_comb_pred_store:   dict[str, tuple] = {}
    lr_comb_embed_store:  dict[str, tuple] = {}
    lr_comb_report_store: dict[str, dict]  = {}

    if getattr(args, "lr_combined", False):
        for backbone in available_backbones:
            split = build_combined_split(all_embeddings, backbone, n_samples=_n_combined)
            if split is None:
                continue
            X_tr, X_te, y_tr, y_te, le_comb, label_parts = split
            print(f"\n=== LR probe (combined): {backbone}  classes={len(le_comb.classes_)}  train/class={_n_combined} ===")
            metrics, y_test_lbl, y_pred_lbl = run_probe(X_tr, X_te, y_tr, y_te, le_comb, backbone)
            lr_combined[backbone] = {k: v for k, v in metrics.items() if k != "per_class_report_dict"}
            lr_comb_pred_store[backbone]   = (y_test_lbl, y_pred_lbl)
            X_all = np.concatenate([X_tr, X_te], axis=0)
            y_all_lbl = le_comb.inverse_transform(np.concatenate([y_tr, y_te], axis=0)).tolist()
            lr_comb_embed_store[backbone]  = (X_all, y_all_lbl)
            lr_comb_report_store[backbone] = metrics["per_class_report_dict"]
    else:
        print("\n[skip] LR combined — pass --lr_combined to enable")

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

    # Phase 2e-FC: FC combined on post-projector embeddings (VLMs only — always runs)
    postproj_fc_combined: dict[str, dict] = {}
    postproj_fc_comb_report_store: dict[str, dict] = {}
    for postproj_key in postproj_embeddings:
        split = build_combined_split(postproj_embeddings, postproj_key, n_samples=_n_combined)
        if split is None:
            continue
        X_tr, X_te, y_tr, y_te, le_comb, _ = split
        print(f"\n=== FC probe (post-proj combined): {postproj_key}  classes={len(le_comb.classes_)} ===")
        metrics, y_test_lbl, y_pred_lbl = run_fc_probe(
            X_tr, X_te, y_tr, y_te, le_comb, postproj_key,
            epochs=fc_epochs, lr=fc_lr, batch_size=fc_batch_size,
        )
        postproj_fc_combined[postproj_key] = {k: v for k, v in metrics.items() if k != "train_losses"}
        postproj_fc_comb_report_store[postproj_key] = metrics["per_class_report_dict"]
        plot_loss_curve(metrics["train_losses"], f"postproj_{postproj_key}_fc", loss_dir)

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

    # Same breakdown for FC combined (pre-projector)
    _by_task_fc_backbone: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for backbone, report in fc_comb_report_store.items():
        for cls, metrics in report.items():
            if cls in _skip:
                continue
            tt = _class_to_task.get(cls)
            if tt:
                _by_task_fc_backbone[tt][backbone].append(metrics.get("f1-score", 0.0))

    by_task_fc_combined = {
        tt: {
            backbone: round(float(np.mean(vals)), 4)
            for backbone, vals in backbone_map.items()
        }
        for tt, backbone_map in _by_task_fc_backbone.items()
    }

    # FC post-projector task breakdown
    _by_task_postproj_fc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for backbone, report in postproj_fc_comb_report_store.items():
        for cls, metrics in report.items():
            if cls in _skip:
                continue
            tt = _class_to_task.get(cls)
            if tt:
                _by_task_postproj_fc[tt][backbone].append(metrics.get("f1-score", 0.0))

    by_task_postproj_fc_combined = {
        tt: {
            backbone: round(float(np.mean(vals)), 4)
            for backbone, vals in backbone_map.items()
        }
        for tt, backbone_map in _by_task_postproj_fc.items()
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
        # FC combined (always runs)
        "fc_combined": fc_combined,
        "by_task_fc_combined": by_task_fc_combined,
        # Post-projector LR combined (VLMs only — always runs)
        "postproj_lr_combined": postproj_lr_combined,
        "postproj_lr_combined_reports": postproj_lr_comb_report_store,
        # Post-projector FC combined (VLMs only — always runs)
        "postproj_fc_combined": postproj_fc_combined,
        "by_task_postproj_fc_combined": by_task_postproj_fc_combined,
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

    # Overall backbone comparison (uses fc_combined if lr_combined not run)
    if results["lr_combined"] or results["fc_combined"]:
        plot_overall_summary(results, output_dir)
    if results.get("by_task_lr_combined") or results.get("by_task_fc_combined") or results.get("by_task"):
        plot_task_summary(results, output_dir)
    if all_experiments and results["overall"]:
        plot_master_comparison(results, output_dir)

    # Projector before/after: prefer LR vs LR (apples-to-apples); fall back to FC vs FC
    if results["lr_combined"] and postproj_lr_combined:
        plot_projector_before_after(results["lr_combined"], postproj_lr_combined, output_dir)
    elif results["fc_combined"] and postproj_fc_combined:
        plot_projector_before_after(results["fc_combined"], postproj_fc_combined, output_dir)
    # Per-task projector comparison (FC)
    if by_task_fc_combined and by_task_postproj_fc_combined:
        plot_projector_task_comparison(by_task_fc_combined, by_task_postproj_fc_combined, output_dir)

    # FC combined plots (always — default probe mode)
    for backbone, (X_te, y_te) in fc_comb_embed_store.items():
        plot_tsne(X_te, y_te, f"combined_{backbone}_fc", tsne_dir)
    if fc_comb_report_store:
        plot_per_class_f1(fc_comb_report_store, "combined_fc", f1_dir)
    for backbone, (y_true, y_pred) in fc_comb_pred_store.items():
        plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_fc", cm_dir)

    # LR combined plots (opt-in via --lr_combined)
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

    # Per-class accuracy vs frequency (pre + post projector; FC default, LR if run)
    _pre_reports_for_freq = lr_comb_report_store or fc_comb_report_store
    _post_reports_for_freq = postproj_lr_comb_report_store or postproj_fc_comb_report_store
    if _pre_reports_for_freq and combined_class_counts:
        plot_class_accuracy_vs_frequency(
            combined_class_counts,
            _pre_reports_for_freq,
            _post_reports_for_freq,
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

    # Update pipeline results CSV (fc_combined is default; lr_combined overrides if run)
    pipeline_csv = output_dir.parent / "pipeline_results.csv"
    _probe_results_for_csv = results["lr_combined"] or results["fc_combined"]
    for backbone, m in _probe_results_for_csv.items():
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
    # Pre-parse just --probe_config so its values can seed argparse's own
    # defaults below, before the real parse happens — CLI flags a user
    # actually types still win, since parser.set_defaults() only changes
    # what "not passed" resolves to, not what an explicit flag resolves to.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--probe_config", default="src/linear_probe/probe_config.yaml")
    _pre_args, _ = _pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Linear probe across all active AgML datasets."
    )
    parser.add_argument(
        "--probe_config", default=_pre_args.probe_config,
        help="YAML file of defaults (hyperparameters, backbones, paths) — edit it "
             "directly for routine changes instead of passing flags every time. "
             "Any flag also passed on the command line overrides this file.",
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
        "--manifest", default="dataset/selected_images.json",
        help="scripts/agml_hf/download_and_sample.py manifest to materialize images "
             "from (used when --dataset_source agml, i.e. not digital_green).",
    )
    parser.add_argument(
        "--images_root", default="dataset/images",
        help="Local cache directory for images materialized from --manifest — "
             "images/<dataset>/<label>/<row_idx>.jpg, reused across runs.",
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
        help="Images per class used for training, in BOTH the per-dataset split and the "
             "combined (all datasets merged) split — default 10 either way. Test uses all "
             "remaining images in each class.",
    )
    parser.add_argument(
        "--quantize", choices=["4bit", "8bit"], default=None,
        help="Load VLMs in 4-bit or 8-bit quantization (requires bitsandbytes). Use 4bit for 48GB GPUs.",
    )
    parser.add_argument(
        "--trust_remote_code", action=argparse.BooleanOptionalAction, default=True,
        help="Allow loading models that ship custom modeling code on their HF repo "
             "(e.g. Kimi-VL, Nemotron) — this executes arbitrary Python from that "
             "repo. Default True; pass --no-trust_remote_code to require "
             "transformers-native architectures only.",
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
        help="Also run FC probe per-dataset (one probe per dataset), plus its plots. "
             "Default (no flags at all) runs FC combined only. Combine with --per_dataset "
             "and --lr_combined to get all 4 variants (LR per-dataset, LR combined, "
             "FC per-dataset, FC combined) — these three flags are independent, not "
             "one 'run everything' switch.",
    )
    parser.add_argument(
        "--per_dataset", action="store_true",
        help="Also run LR probe per-dataset (one probe per dataset). Default (no flags "
             "at all): FC combined only.",
    )
    parser.add_argument(
        "--lr_combined", action="store_true",
        help="Also run LR probe on combined embeddings (slower, CPU-bound). Default (no "
             "flags at all): FC combined only.",
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
    parser.add_argument(
        "--plots_only", action="store_true",
        help="Regenerate plots from a previous run without re-extracting embeddings or retraining. "
             "Requires linear_probe_results.json, fc_comb_preds.json, and fc_comb_embeds/ in --output_dir.",
    )

    if _pre_args.probe_config and Path(_pre_args.probe_config).exists():
        import yaml as _yaml
        with open(_pre_args.probe_config) as f:
            _cfg = _yaml.safe_load(f) or {}
        _valid_dests = {a.dest for a in parser._actions}
        _unknown = set(_cfg) - _valid_dests
        if _unknown:
            print(f"[warn] {_pre_args.probe_config}: unrecognized key(s) ignored: {sorted(_unknown)}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid_dests})

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
        if results.get("by_task_lr_combined") or results.get("by_task_fc_combined") or results.get("by_task"):
            plot_task_summary(results, output_dir)

        print(f"Plots saved to {output_dir}/")
    else:
        main(args)
