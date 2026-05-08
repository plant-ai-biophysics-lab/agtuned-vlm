"""
H1: Linear Probe — Vision Encoder Separability
===============================================
Freezes each backbone, extracts embeddings, trains a logistic regression head,
and compares accuracy across backbones on the same train/test split.

Three backbones supported:
  --model          Fine-tuned Gemma4 (required)
  --baseline_model Base Gemma4 before fine-tuning (optional)
  --yolo_model     YOLOv8 classification backbone, e.g. yolov8s-cls.pt (optional)

All three use the same frozen backbone + logistic regression setup so the only
variable is feature quality.

Interpretation:
  YOLO >> Gemma4 base  → Gemma4 encoder is specifically bad for this domain
  YOLO ≈ Gemma4 base   → task is hard for linear probing in general (not encoder-specific)
  Fine-tuned > base    → fine-tuning improved the encoder's separability

Usage:
    python linear_probe.py --agml_dataset bean_disease_uganda \\
                           --model /path/to/finetuned \\
                           --baseline_model google/gemma-4-E2B-it \\
                           --yolo_model yolov8s-cls.pt
"""

import argparse
import gc
import inspect
import json
import os
from typing import Any, Dict, cast

import agml
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
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoImageProcessor, AutoModel


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_agml_samples(dataset_name: str) -> list[dict]:
    loader = agml.data.AgMLDataLoader(dataset_name)
    loader.split(train=0.8, val=0.1, test=0.1)

    contents = cast(Dict[str, Any], loader.export_contents())
    class_names = cast(list[str], loader.classes)

    samples = []
    for path, label_idx in contents.items():
        label_text = class_names[int(label_idx)]
        cleaned_label = label_text.replace("___", " ").replace("_", " ")
        samples.append({"path": str(path), "label": cleaned_label})

    return samples


# ── Gemma4 embedding extraction ───────────────────────────────────────────────

def resolve_mm_root(model):
    if hasattr(model, "vision_tower"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        return model.model
    available = [n for n, _ in model.named_children()]
    raise AttributeError(f"Could not locate vision_tower. Top-level: {available}")


@torch.no_grad()
def extract_gemma4_embedding(inputs: dict, mm_root) -> np.ndarray:
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError("No pixel_values in inputs.")

    if pixel_values.dim() == 5:
        pixel_values = pixel_values.squeeze(0)
    if pixel_values.dim() == 4 and pixel_values.shape[0] > 1:
        pixel_values = pixel_values[0:1]

    REMAP = {"image_position_ids": "pixel_position_ids"}
    VISION_KEYS = {"pixel_values", "pixel_position_ids", "pixel_attention_mask",
                   "image_position_ids", "spatial_shapes", "image_sizes"}
    forward_params = set(inspect.signature(mm_root.vision_tower.forward).parameters.keys())

    vision_kwargs = {"pixel_values": pixel_values}
    for k, v in inputs.items():
        if k == "pixel_values" or k not in VISION_KEYS:
            continue
        forward_key = REMAP.get(k, k)
        if forward_key in forward_params:
            vision_kwargs[forward_key] = v

    if "return_dict" in forward_params:
        vision_kwargs["return_dict"] = True

    vision_out = mm_root.vision_tower(**vision_kwargs)

    if hasattr(vision_out, "last_hidden_state"):
        hidden = vision_out.last_hidden_state
    elif isinstance(vision_out, tuple):
        hidden = vision_out[0]
    else:
        hidden = vision_out

    if hidden.dim() == 3:
        hidden = hidden[0]

    return hidden.mean(dim=0).cpu().float().numpy()


def extract_gemma4_embeddings(samples, processor, mm_root, device) -> tuple[np.ndarray, list[str]]:
    # Move inputs to wherever the vision tower's weights actually are.
    # device_map="auto" may offload the vision tower to CPU.
    vision_device = next(mm_root.vision_tower.parameters()).device

    embeddings, labels = [], []
    for i, sample in enumerate(samples):
        image = PILImage.open(sample["path"]).convert("RGB")
        inputs = processor(text="<image>", images=image, return_tensors="pt")
        inputs = {k: v.to(vision_device) for k, v in inputs.items()}
        emb = extract_gemma4_embedding(inputs, mm_root)
        embeddings.append(emb)
        labels.append(sample["label"])
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] extracted")
    return np.stack(embeddings), labels


def load_gemma4(model_id: str, device: str):
    local = os.path.abspath(model_id)
    if os.path.exists(local):
        model_id = local
    print(f"Loading Gemma4: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    mm_root = resolve_mm_root(model)
    vision_device = next(mm_root.vision_tower.parameters()).device
    print(f"  Vision tower: {type(mm_root.vision_tower).__name__} on {vision_device}")
    return processor, model, mm_root


# ── YOLO embedding extraction ─────────────────────────────────────────────────

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

    # Hook the penultimate layer (before the Classify head).
    captured = []
    def _hook(module, input, output):
        t = output[0] if isinstance(output, (list, tuple)) else output
        captured.append(t.detach().flatten(1).mean(0).cpu().float())

    penultimate = list(yolo.model.model.children())[-2]
    handle = penultimate.register_forward_hook(_hook)

    embeddings = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            captured.clear()
            image = PILImage.open(sample["path"]).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            yolo.model(x)
            embeddings.append(captured[0].numpy())
            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                print(f"  [{i+1}/{len(samples)}] extracted")

    handle.remove()
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(embeddings)


# ── DINOv3 embedding extraction ──────────────────────────────────────────────

def load_dino(model_id: str, device: str):
    print(f"Loading DINOv3: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    return processor, model


@torch.no_grad()
def extract_dino_embeddings(samples, processor, model, device) -> tuple[np.ndarray, list[str]]:
    embeddings, labels = [], []
    for i, sample in enumerate(samples):
        image = PILImage.open(sample["path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # CLS token is the standard image-level embedding for DINO models
        emb = outputs.last_hidden_state[0, 0, :].cpu().float().numpy()
        embeddings.append(emb)
        labels.append(sample["label"])
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] extracted")
    return np.stack(embeddings), labels


# ── Logistic regression probe ─────────────────────────────────────────────────

def run_probe(X_train, X_test, y_train, y_test, le, label: str) -> tuple[dict, list[str], list[str]]:
    print(f"Training logistic regression [{label}]...")
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_test_labels = le.inverse_transform(y_test).tolist()
    y_pred_labels = le.inverse_transform(y_pred).tolist()

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_str = classification_report(y_test_labels, y_pred_labels, zero_division=0)
    report_dict = classification_report(y_test_labels, y_pred_labels, zero_division=0, output_dict=True)

    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    metrics = {
        "backbone": label,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_report": report_str,
        "per_class_report_dict": report_dict,
    }
    return metrics, y_test_labels, y_pred_labels


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Visualizations ────────────────────────────────────────────────────────────

def _safe_name(label: str) -> str:
    return label.replace("/", "_").replace(":", "_").replace(" ", "_")


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], label: str, output_dir: str):
    class_labels = sorted(set(y_true))
    n = len(class_labels)
    fig_size = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}", fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, f"confusion_matrix_{_safe_name(label)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne(X: np.ndarray, y_labels: list[str], label: str, output_dir: str, max_samples: int = 2000):
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

    path = os.path.join(output_dir, f"tsne_{_safe_name(label)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_f1(backbone_reports: dict[str, dict], output_dir: str):
    # Collect classes from all backbones (excluding summary rows).
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
        bars = ax.bar(x + offset, vals, width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 by Backbone")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "per_class_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary(results: dict, output_dir: str):
    backbones = list(results["backbones"].keys())
    accuracies = [results["backbones"][b]["accuracy"] for b in backbones]
    f1_macros = [results["backbones"][b]["f1_macro"] for b in backbones]
    f1_weighted = [results["backbones"][b]["f1_weighted"] for b in backbones]

    x = np.arange(len(backbones))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(backbones) * 2.5), 5))
    b1 = ax.bar(x - width, accuracies, width, label="Accuracy")
    b2 = ax.bar(x,         f1_macros,  width, label="F1 Macro")
    b3 = ax.bar(x + width, f1_weighted, width, label="F1 Weighted")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(backbones, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Linear Probe — Backbone Comparison ({results['agml_dataset']})")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "summary_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.dino_model and args.dino_model.lower() == "none":
        args.dino_model = None

    print(f"\nLoading AgML dataset: {args.agml_dataset}")
    samples = load_agml_samples(args.agml_dataset)
    if args.n_samples and args.n_samples < len(samples):
        # Stratified subsample so class distribution is preserved
        labels_tmp = [s["label"] for s in samples]
        le_tmp = LabelEncoder()
        y_tmp = le_tmp.fit_transform(labels_tmp)
        splitter_tmp = StratifiedShuffleSplit(
            n_splits=1, train_size=args.n_samples, random_state=42
        )
        sub_idx, _ = next(splitter_tmp.split(np.zeros(len(y_tmp)), y_tmp))
        samples = [samples[i] for i in sub_idx]
        print(f"  Subsampled to {len(samples)} images")
    n_classes = len(set(s["label"] for s in samples))
    print(f"  {len(samples)} images | {n_classes} classes")

    labels_all = [s["label"] for s in samples]
    le = LabelEncoder()
    y_enc = le.fit_transform(labels_all)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y_enc)), y_enc))
    print(f"  Split: train={len(train_idx)}, test={len(test_idx)}")

    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    results = {
        "agml_dataset": args.agml_dataset,
        "n_images": len(samples),
        "n_classes": n_classes,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "backbones": {},
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Stores (X_all, labels_all) per backbone for t-SNE
    embeddings_store: dict[str, tuple[np.ndarray, list[str]]] = {}
    # Stores sklearn report dicts per backbone for per-class F1 chart
    report_store: dict[str, dict] = {}
    # Stores (y_test_labels, y_pred_labels) per backbone for confusion matrices
    pred_store: dict[str, tuple[list[str], list[str]]] = {}

    # --- DINOv3 self-supervised baseline ---
    if args.dino_model:
        print(f"\n=== DINOv3: {args.dino_model} ===")
        dino_processor, dino_model = load_dino(args.dino_model, device)
        X, _ = extract_dino_embeddings(samples, dino_processor, dino_model, device)
        metrics, y_test_lbl, y_pred_lbl = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, "dinov3"
        )
        results["backbones"]["dinov3"] = metrics
        embeddings_store["dinov3"] = (X, labels_all)
        report_store["dinov3"] = metrics["per_class_report_dict"]
        pred_store["dinov3"] = (y_test_lbl, y_pred_lbl)
        unload(dino_model)

    # --- YOLO supervised baseline ---
    if args.yolo_model:
        print(f"\n=== YOLO Backbone: {args.yolo_model} ===")
        X = extract_yolo_embeddings(samples, args.yolo_model, device)
        metrics, y_test_lbl, y_pred_lbl = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, args.yolo_model
        )
        results["backbones"]["yolo"] = metrics
        embeddings_store["yolo"] = (X, labels_all)
        report_store["yolo"] = metrics["per_class_report_dict"]
        pred_store["yolo"] = (y_test_lbl, y_pred_lbl)

    # --- Gemma4 base (pre-fine-tuning baseline) ---
    if args.baseline_model:
        print(f"\n=== Gemma4 Base: {args.baseline_model} ===")
        processor, model, mm_root = load_gemma4(args.baseline_model, device)
        X, _ = extract_gemma4_embeddings(samples, processor, mm_root, device)
        metrics, y_test_lbl, y_pred_lbl = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, args.baseline_model
        )
        results["backbones"]["gemma4_base"] = metrics
        embeddings_store["gemma4_base"] = (X, labels_all)
        report_store["gemma4_base"] = metrics["per_class_report_dict"]
        pred_store["gemma4_base"] = (y_test_lbl, y_pred_lbl)
        unload(model)

    # --- Gemma4 fine-tuned ---
    if args.model:
        print(f"\n=== Gemma4 Fine-tuned: {args.model} ===")
        processor, model, mm_root = load_gemma4(args.model, device)
        X, _ = extract_gemma4_embeddings(samples, processor, mm_root, device)
        metrics, y_test_lbl, y_pred_lbl = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, args.model
        )
        results["backbones"]["gemma4_finetuned"] = metrics
        embeddings_store["gemma4_finetuned"] = (X, labels_all)
        report_store["gemma4_finetuned"] = metrics["per_class_report_dict"]
        pred_store["gemma4_finetuned"] = (y_test_lbl, y_pred_lbl)
        unload(model)

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"  {'Backbone':<30} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*52}")
    for key, res in results["backbones"].items():
        print(f"  {key:<30} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}")

    out_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Visualizations ---
    print("\n=== Generating plots ===")

    plot_summary(results, args.output_dir)

    if len(report_store) > 1:
        plot_per_class_f1(report_store, args.output_dir)

    for key, (y_true, y_pred) in pred_store.items():
        plot_confusion_matrix(y_true, y_pred, key, args.output_dir)

    for key, (X_all, y_all) in embeddings_store.items():
        plot_tsne(X_all, y_all, key, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Fine-tuned Gemma4 (HF ID or local path). Omit to skip fine-tuned comparison.")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Base Gemma4 before fine-tuning (e.g. google/gemma-4-E2B-it)")
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="DINOv3 model ID (default: facebook/dinov3-vitb16-pretrain-lvd1689m). Pass 'none' to skip.")
    parser.add_argument("--yolo_model", type=str, default=None,
                        help="YOLO classification model (e.g. yolo11s-cls.pt)")
    parser.add_argument("--agml_dataset", type=str, required=True,
                        help="AgML dataset name (e.g. bean_disease_uganda)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max samples per dataset (stratified). Default: use all.")
    args = parser.parse_args()
    main(args)
