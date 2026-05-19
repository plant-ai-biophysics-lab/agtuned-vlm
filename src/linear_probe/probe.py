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
import inspect
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
    from tqdm import tqdm
    vision_device = next(mm_root.vision_tower.parameters()).device
    embeddings, labels = [], []
    for sample in tqdm(samples, desc="  Gemma4 embeddings", unit="img", leave=False):
        image = PILImage.open(sample["path"]).convert("RGB")
        inputs = processor(text="<image>", images=image, return_tensors="pt")
        inputs = {k: v.to(vision_device) for k, v in inputs.items()}
        emb = extract_gemma4_embedding(inputs, mm_root)
        embeddings.append(emb)
        labels.append(sample["label"])
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

    datasets = active_datasets(Path(args.config))
    print(f"\nActive datasets: {len(datasets)}")

    # Phase 1: load each backbone once, extract embeddings for all datasets, unload
    # all_embeddings[backbone_key][agml_name] = (X, labels, train_idx, test_idx, le)
    all_embeddings: dict[str, dict[str, tuple]] = {}

    for ds_cfg in datasets:
        agml_name = ds_cfg.agml_name
        print(f"\nLoading samples: {agml_name}")
        try:
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
        all_embeddings["dinov3"] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="DINOv3 datasets", unit="ds"):
            X, _ = extract_dino_embeddings(samples, dino_processor, dino_model, device)
            all_embeddings["dinov3"][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        unload(dino_model)

    # YOLO pass
    if args.yolo_model:
        from tqdm import tqdm
        print(f"\n{'='*60}\nYOLO: {args.yolo_model}\n{'='*60}")
        all_embeddings["yolo"] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc="YOLO datasets", unit="ds"):
            X = extract_yolo_embeddings(samples, args.yolo_model, device)
            all_embeddings["yolo"][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)

    # VLM passes (one per --models entry, loaded and unloaded sequentially)
    for model_id in (args.models or []):
        from tqdm import tqdm
        backbone_key = model_id.split("/")[-1]
        print(f"\n{'='*60}\nVLM: {model_id}\n{'='*60}")
        processor, vlm, mm_root = load_gemma4(model_id, device)
        all_embeddings[backbone_key] = {}
        ds_items = list(all_embeddings["_samples"].items())
        for agml_name, (samples, labels_all, y_enc, train_idx, test_idx, le) in tqdm(ds_items, desc=f"{backbone_key} datasets", unit="ds"):
            X, _ = extract_gemma4_embeddings(samples, processor, mm_root, device)
            all_embeddings[backbone_key][agml_name] = (X, labels_all, y_enc, train_idx, test_idx, le)
        unload(vlm)

    available_backbones = [k for k in all_embeddings if k != "_samples"]
    if not available_backbones:
        print("No backbones to evaluate.")
        return

    # Phase 2: train probes and collect results
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

            print(f"\n=== Probe: {agml_name} / {backbone} ===")
            metrics, y_test_lbl, y_pred_lbl = run_probe(
                X[train_idx], X[test_idx], y_train, y_test, le, backbone
            )
            results_per_dataset[agml_name]["backbones"][backbone] = metrics
            embeddings_store[agml_name][backbone] = (X[test_idx], y_test_lbl)
            report_store[agml_name][backbone] = metrics["per_class_report_dict"]
            pred_store[agml_name][backbone] = (y_test_lbl, y_pred_lbl)

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
    }

    # Save results
    save_json(output_dir / "linear_probe_results.json", results)
    print(f"\nResults saved to {output_dir / 'linear_probe_results.json'}")

    # Summary table
    print("\n=== Overall Summary ===")
    print(f"  {'Backbone':<30} {'F1 Macro':>10} {'Accuracy':>10} {'N Datasets':>12}")
    print(f"  {'-'*64}")
    for backbone, m in results["overall"].items():
        print(f"  {backbone:<30} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} {m['n_datasets']:>12}")

    # Plots
    print("\n=== Generating plots ===")
    plot_overall_summary(results, output_dir)
    plot_task_summary(results, output_dir)

    for agml_name in results_per_dataset:
        if len(report_store[agml_name]) > 1:
            plot_per_class_f1(report_store[agml_name], agml_name, output_dir)
        for backbone, (y_true, y_pred) in pred_store[agml_name].items():
            plot_confusion_matrix(y_true, y_pred, f"{agml_name}_{backbone}", output_dir)
        for backbone, (X_test, y_test) in embeddings_store[agml_name].items():
            plot_tsne(X_test, y_test, f"{agml_name}_{backbone}", output_dir)

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
    args = parser.parse_args()

    if args.yolo_model and args.yolo_model.lower() == "none":
        args.yolo_model = None

    main(args)
