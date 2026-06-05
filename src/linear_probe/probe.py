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

from src.common.dataset_config import active_datasets, DatasetConfig
from src.common.io import save_json, update_pipeline_results


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


def subsample_stratified(samples: list[dict], n_samples: int) -> list[dict]:
    """Stratified subsample to preserve class distribution."""
    labels = [s["label"] for s in samples]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    sub_idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return [samples[i] for i in sub_idx]


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


def extract_vlm_embeddings(samples, processor, model, mm_root, desc: str = "VLM embeddings") -> tuple[np.ndarray, list[str]]:
    """
    Architecture-agnostic vision embedding extraction via forward hook.

    Registers a hook on mm_root.vision_tower, runs a full forward pass for each
    image, and mean-pools the captured feature map. Works for Gemma4, Qwen VL,
    and any model whose vision encoder is reachable via mm_root.vision_tower.
    """
    from tqdm import tqdm

    model_device = next(model.parameters()).device
    embeddings, labels = [], []

    for sample in tqdm(samples, desc=f"  {desc}", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")

        # Build a minimal chat-formatted input so the processor handles
        # image preprocessing correctly for each architecture
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

        handle = mm_root.vision_tower.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(**inputs)
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
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y_labels = [y_labels[i] for i in idx]

    print(f"  Running t-SNE for {label} ({len(X)} samples)...")
    perplexity = min(30, len(X) - 1)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

    unique_labels = sorted(set(y_labels))
    cmap = plt.cm.get_cmap("tab20" if len(unique_labels) <= 20 else "hsv", len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = [i for i, y in enumerate(y_labels) if y == lbl]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color_map[lbl]], label=lbl, s=12, alpha=0.7)

    ncol = max(1, len(unique_labels) // 20)
    ax.legend(loc="best", fontsize=6, ncol=ncol, markerscale=1.5)
    ax.set_title(f"t-SNE Embedding Space — {label}", fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()

    path = output_dir / f"tsne_{_safe_name(label)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


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
    """Bar chart comparing backbone F1 across all datasets (mean overall)."""
    backbones = list(results["overall"].keys())
    f1_macros = [results["overall"][b]["f1_macro"] for b in backbones]
    accuracies = [results["overall"][b]["accuracy"] for b in backbones]

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
    ax.set_xticklabels(backbones, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe — Overall Backbone Comparison (mean across all datasets)")
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
    """Bar chart comparing backbone F1 per task type."""
    task_types = list(results.get("by_task", {}).keys())
    if not task_types:
        return
    backbones = list(next(iter(results["by_task"].values())).keys())
    x = np.arange(len(task_types))
    width = 0.8 / len(backbones)

    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 2), 5))
    for i, backbone in enumerate(backbones):
        vals = [results["by_task"][tt].get(backbone, {}).get("f1_macro", 0.0) for tt in task_types]
        offset = (i - (len(backbones) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=backbone)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([tt.replace("_", " ") for tt in task_types], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Linear Probe — F1 by Task Type")
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

    use_dg = getattr(args, "dataset_source", "agml") == "digital_green"

    if use_dg:
        dg_config = getattr(args, "dg_config", None) or "outputs/data_generation/digital_green_datasets.yaml"
        dg_jsonl  = getattr(args, "dg_descriptions_jsonl", None) or "outputs/data_generation/dg_descriptions.jsonl"
        images_dir = Path(args.images_dir)
        datasets = active_datasets(Path(dg_config))
    else:
        datasets = active_datasets(Path(args.config))

    print(f"\nActive datasets: {len(datasets)}")

    # Phase 1: load each backbone once, extract embeddings for all datasets, unload
    # all_embeddings[backbone_key][agml_name] = (X, labels, train_idx, test_idx, le)
    all_embeddings: dict[str, dict[str, tuple]] = {}

    for ds_cfg in datasets:
        agml_name = ds_cfg.agml_name
        print(f"\nLoading samples: {agml_name}")
        try:
            if use_dg:
                samples = load_dg_samples(images_dir, Path(dg_jsonl))
            else:
                samples = load_agml_samples(agml_name)
        except Exception as e:
            print(f"  [error] {e}")
            continue

        if args.n_samples and args.n_samples < len(samples):
            samples = subsample_stratified(samples, args.n_samples)
            print(f"  Subsampled to {len(samples)} images")

        # Drop classes with fewer than 2 samples — StratifiedShuffleSplit requires ≥2
        label_counts = Counter(s["label"] for s in samples)
        samples = [s for s in samples if label_counts[s["label"]] >= 2]
        if not samples:
            print(f"  [skip] {agml_name} — no classes with ≥2 samples after filtering")
            continue
        dropped = sum(1 for c, n in label_counts.items() if n < 2)
        if dropped:
            print(f"  [warn] {agml_name} — dropped {dropped} singleton class(es)")

        labels_all = [s["label"] for s in samples]
        le = LabelEncoder()
        y_enc = le.fit_transform(labels_all)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y_enc)), y_enc))

        # Temporarily store samples with split indices keyed by dataset
        all_embeddings.setdefault("_samples", {})[agml_name] = (samples, labels_all, y_enc, train_idx, test_idx, le)

    # DINOv3 pass
    if args.dino_model:
        from tqdm import tqdm
        print(f"\n{'='*60}\nDINOv3: {args.dino_model}\n{'='*60}")
        dino_processor, dino_model = load_dino(args.dino_model, device)
        dino_key = args.dino_model.split("/")[-1].replace(".pt", "")
        all_embeddings[dino_key] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="DINOv3 datasets", unit="ds"):
            X, _ = extract_dino_embeddings(samples, dino_processor, dino_model, device)
            all_embeddings[dino_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        unload(dino_model)

    # YOLO pass
    if args.yolo_model:
        from tqdm import tqdm
        print(f"\n{'='*60}\nYOLO: {args.yolo_model}\n{'='*60}")
        yolo_key = args.yolo_model.replace(".pt", "")
        all_embeddings[yolo_key] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="YOLO datasets", unit="ds"):
            X = extract_yolo_embeddings(samples, args.yolo_model, device)
            all_embeddings[yolo_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)

    # VLM passes (one per --models entry, loaded and unloaded sequentially)
    for model_id in (args.models or []):
        from tqdm import tqdm
        backbone_key = model_id.split("/")[-1]
        print(f"\n{'='*60}\nVLM: {model_id}\n{'='*60}")
        processor, vlm, mm_root = load_gemma4(model_id, device, quantize=args.quantize)
        all_embeddings[backbone_key] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc=f"{backbone_key} datasets", unit="ds"):
            X, _ = extract_vlm_embeddings(samples, processor, vlm, mm_root, desc=backbone_key)
            all_embeddings[backbone_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        unload(vlm)

    available_backbones = [k for k in all_embeddings if k != "_samples"]
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

    # Phase 2a: LR per-dataset (existing behaviour)
    results_per_dataset: dict[str, dict] = {}
    embeddings_store: dict[str, dict[str, tuple]] = defaultdict(dict)
    report_store: dict[str, dict[str, dict]] = defaultdict(dict)
    pred_store: dict[str, dict[str, tuple]] = defaultdict(dict)

    for agml_name in all_embeddings["_samples"]:
        _, _, _, train_idx, test_idx, le = all_embeddings["_samples"][agml_name]
        results_per_dataset[agml_name] = {"backbones": {}}

        for backbone in available_backbones:
            if agml_name not in all_embeddings.get(backbone, {}):
                continue
            X, labels_all, y_enc, train_idx, test_idx, le = all_embeddings[backbone][agml_name]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            print(f"\n=== LR probe (per-dataset): {agml_name} / {backbone} ===")
            metrics, y_test_lbl, y_pred_lbl = run_probe(
                X[train_idx], X[test_idx], y_train, y_test, le, backbone
            )
            results_per_dataset[agml_name]["backbones"][backbone] = metrics
            embeddings_store[agml_name][backbone] = (X[test_idx], y_test_lbl)
            report_store[agml_name][backbone] = metrics["per_class_report_dict"]
            pred_store[agml_name][backbone] = (y_test_lbl, y_pred_lbl)

    # Phase 2b: FC per-dataset
    fc_per_dataset: dict[str, dict] = {}
    fc_pred_store:  dict[str, dict[str, tuple]] = defaultdict(dict)
    fc_report_store: dict[str, dict[str, dict]] = defaultdict(dict)
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

    # Phase 2c: LR combined (all datasets as one)
    lr_combined: dict[str, dict] = {}
    lr_comb_pred_store:   dict[str, tuple] = {}   # backbone → (y_true, y_pred)
    lr_comb_embed_store:  dict[str, tuple] = {}   # backbone → (X_te, y_true)
    lr_comb_report_store: dict[str, dict]  = {}
    for backbone in available_backbones:
        split = build_combined_split(all_embeddings, backbone)
        if split is None:
            continue
        X_tr, X_te, y_tr, y_te, le_comb, _ = split
        print(f"\n=== LR probe (combined): {backbone}  classes={len(le_comb.classes_)} ===")
        metrics, y_test_lbl, y_pred_lbl = run_probe(X_tr, X_te, y_tr, y_te, le_comb, backbone)
        lr_combined[backbone] = {k: v for k, v in metrics.items() if k != "per_class_report_dict"}
        lr_comb_pred_store[backbone]   = (y_test_lbl, y_pred_lbl)
        lr_comb_embed_store[backbone]  = (X_te, y_test_lbl)
        lr_comb_report_store[backbone] = metrics["per_class_report_dict"]

    # Phase 2d: FC combined (all datasets as one)
    fc_combined: dict[str, dict] = {}
    fc_comb_pred_store:   dict[str, tuple] = {}
    fc_comb_report_store: dict[str, dict]  = {}
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

    results = {
        # LR per-dataset (original behaviour — collate_results reads from here)
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
        # FC per-dataset
        "fc_per_dataset": {
            agml_name: {
                backbone: m
                for backbone, m in ds_data["backbones"].items()
            }
            for agml_name, ds_data in fc_per_dataset.items()
        },
        # LR combined (all classes at once)
        "lr_combined": lr_combined,
        # FC combined (all classes at once)
        "fc_combined": fc_combined,
    }

    # Save results
    save_json(output_dir / "linear_probe_results.json", results)
    print(f"\nResults saved to {output_dir / 'linear_probe_results.json'}")

    # Summary tables
    print("\n=== LR Per-Dataset — Overall Summary ===")
    print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10} {'N Datasets':>12}")
    print(f"  {'-'*64}")
    for backbone, m in results["overall"].items():
        print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} {m['n_datasets']:>12}")

    print("\n=== LR Combined (all classes) ===")
    print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for backbone, m in results["lr_combined"].items():
        print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f}")

    print("\n=== FC Combined (all classes) ===")
    print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for backbone, m in results["fc_combined"].items():
        print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f}")

    # Plots
    print("\n=== Generating plots ===")
    plot_master_comparison(results, output_dir)   # master chart → root
    plot_overall_summary(results, output_dir)     # LR summary → root
    plot_task_summary(results, output_dir)        # task summary → root

    # LR per-dataset plots
    for agml_name in results_per_dataset:
        if len(report_store[agml_name]) > 1:
            plot_per_class_f1(report_store[agml_name], agml_name, f1_dir)
        for backbone, (y_true, y_pred) in pred_store[agml_name].items():
            plot_confusion_matrix(y_true, y_pred, f"{agml_name}_{backbone}_lr", cm_dir)
        # t-SNE depends only on embeddings, not probe type — generate once per (dataset, backbone)
        for backbone, (X_test, y_test) in embeddings_store[agml_name].items():
            plot_tsne(X_test, y_test, f"{agml_name}_{backbone}", tsne_dir)

    # FC per-dataset plots (separate confusion matrix + per-class F1; t-SNE same embeddings so skip)
    for agml_name in fc_pred_store:
        if len(fc_report_store[agml_name]) > 1:
            plot_per_class_f1(fc_report_store[agml_name], f"{agml_name}_fc", f1_dir)
        for backbone, (y_true, y_pred) in fc_pred_store[agml_name].items():
            plot_confusion_matrix(y_true, y_pred, f"{agml_name}_{backbone}_fc", cm_dir)

    # LR combined plots
    for backbone, (X_te, y_te) in lr_comb_embed_store.items():
        plot_tsne(X_te, y_te, f"combined_{backbone}_lr", tsne_dir)
    if lr_comb_report_store:
        plot_per_class_f1(lr_comb_report_store, "combined_lr", f1_dir)
    for backbone, (y_true, y_pred) in lr_comb_pred_store.items():
        plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_lr", cm_dir)

    # FC combined plots (t-SNE same embeddings as LR combined — skip; confusion matrix + per-class F1)
    if fc_comb_report_store:
        plot_per_class_f1(fc_comb_report_store, "combined_fc", f1_dir)
    for backbone, (y_true, y_pred) in fc_comb_pred_store.items():
        plot_confusion_matrix(y_true, y_pred, f"combined_{backbone}_fc", cm_dir)

    # Update pipeline results CSV
    pipeline_csv = output_dir.parent / "pipeline_results.csv"
    for backbone, m in results["overall"].items():
        update_pipeline_results(
            csv_path=pipeline_csv,
            model=backbone,
            model_type="vision_backbone",
            stage="linear_probe",
            metrics={"f1_macro": m["f1_macro"]},
        )

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
        "--yolo_model", type=str, default="yolo11x-cls.pt",
        help="YOLO classification model. Pass 'none' to skip.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/linear_probe/",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Max samples per dataset (stratified). Default: all.",
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
    args = parser.parse_args()

    if args.yolo_model and args.yolo_model.lower() == "none":
        args.yolo_model = None

    main(args)
