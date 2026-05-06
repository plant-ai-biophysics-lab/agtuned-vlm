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
import numpy as np
import torch
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import AutoProcessor, AutoModelForImageTextToText


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


# ── Logistic regression probe ─────────────────────────────────────────────────

def run_probe(X_train, X_test, y_train, y_test, le, label: str) -> dict:
    print(f"Training logistic regression [{label}]...")
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test_labels, y_pred_labels, zero_division=0)

    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    return {
        "backbone": label,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_report": report,
    }


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading AgML dataset: {args.agml_dataset}")
    samples = load_agml_samples(args.agml_dataset)
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

    # --- YOLO supervised baseline ---
    if args.yolo_model:
        print(f"\n=== YOLO Backbone: {args.yolo_model} ===")
        X = extract_yolo_embeddings(samples, args.yolo_model, device)
        results["backbones"]["yolo"] = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, args.yolo_model
        )

    # --- Gemma4 base (pre-fine-tuning baseline) ---
    if args.baseline_model:
        print(f"\n=== Gemma4 Base: {args.baseline_model} ===")
        processor, model, mm_root = load_gemma4(args.baseline_model, device)
        X, _ = extract_gemma4_embeddings(samples, processor, mm_root, device)
        results["backbones"]["gemma4_base"] = run_probe(
            X[train_idx], X[test_idx], y_train, y_test, le, args.baseline_model
        )
        unload(model)

    # --- Gemma4 fine-tuned ---
    print(f"\n=== Gemma4 Fine-tuned: {args.model} ===")
    processor, model, mm_root = load_gemma4(args.model, device)
    X, _ = extract_gemma4_embeddings(samples, processor, mm_root, device)
    results["backbones"]["gemma4_finetuned"] = run_probe(
        X[train_idx], X[test_idx], y_train, y_test, le, args.model
    )
    unload(model)

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"  {'Backbone':<30} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*52}")
    for key, res in results["backbones"].items():
        print(f"  {key:<30} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="Fine-tuned Gemma4 (HF ID or local path)")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Base Gemma4 before fine-tuning (e.g. google/gemma-4-E2B-it)")
    parser.add_argument("--yolo_model", type=str, default=None,
                        help="YOLO classification model (e.g. yolov8s-cls.pt)")
    parser.add_argument("--agml_dataset", type=str, required=True,
                        help="AgML dataset name (e.g. bean_disease_uganda)")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
