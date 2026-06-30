"""
Chain-of-Thought Vision Assessment
====================================
Evaluates two CoT prompting strategies on agricultural image classification:

  self_cot   — model generates its own visual description first (2-turn history),
               then classifies using the description as context.

  cot_yaml   — structured multi-step CoT from cot_prompts.yaml; each step is its
               own conversation turn, then final classification bridges from analysis.

Neither condition uses any external descriptions — the model reasons from the image.
Aug-vision (pre-written descriptions injected) is evaluated separately via
desc_augmented_vision.py.

Uses HuggingFace transformers for inference (not vLLM) because CoT requires
sequential multi-turn generation per sample that cannot be trivially batched.

Outputs per model:
  cot_cls_{model}.csv      — one row per (image, condition)
  cot_vision_metrics.json  — F1 / accuracy per condition × dataset

Usage:
    uv run python src/cot_vision.py \\
        --model      google/gemma-4-26B-A4B-it \\
        --output_dir outputs/cot_vision/ \\
        --test_split outputs/linear_probe/test_split.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.metrics import compute_classification_metrics, contains_match, resolve_class, sample_class_images

# ── Dataset routing (26 datasets from test_split.json) ────────────────────────
# Maps agml dataset name → (task_type, crop)
DATASET_TASK_TYPES: dict[str, tuple[str, str | None]] = {
    "arabica_coffee_leaf_disease_classification": ("disease",            "arabica_coffee"),
    "banana_leaf_disease_classification":         ("disease",            "banana"),
    "bean_disease_uganda":                        ("disease",            "bean"),
    "betel_leaf_disease_classification":          ("disease",            "betel_leaf"),
    "blackgram_plant_leaf_disease_classification":("disease",            "blackgram"),
    "chilli_leaf_classification":                 ("disease",            "chilli"),
    "coconut_tree_disease_classification":        ("disease",            "coconut"),
    "corn_maize_leaf_disease":                    ("disease",            "corn"),
    "crop_weeds_greece":                          ("weed_identification", None),
    "cucumber_disease_classification":            ("disease",            "cucumber"),
    "guava_disease_pakistan":                     ("disease",            "guava"),
    "java_plum_leaf_disease_classification":      ("disease",            "java_plum"),
    "onion_leaf_classification":                  ("disease",            "onion"),
    "orange_leaf_disease_classification":         ("disease",            "orange"),
    "paddy_disease_classification":               ("disease",            "paddy"),
    "papaya_leaf_disease_classification":         ("disease",            "papaya"),
    "plant_seedlings_aarhus":                     ("weed_identification", None),
    "rangeland_weeds_australia":                  ("weed_identification", None),
    "rice_leaf_disease_classification":           ("disease",            "rice"),
    "soybean_insect_classification":              ("pest_damage",        "soybean"),
    "soybean_weed_uav_brazil":                    ("weed_identification", None),
    "sugarcane_damage_usa":                       ("pest_damage",        "sugarcane"),
    "sunflower_disease_classification":           ("disease",            "sunflower"),
    "tea_leaf_disease_classification":            ("disease",            "tea"),
    "tomato_leaf_disease":                        ("disease",            "tomato"),
    "vine_virus_photo_dataset":                   ("disease",            "grape"),
}

DEFAULT_DATASETS = list(DATASET_TASK_TYPES.keys())

_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)


# ── Prompt config ──────────────────────────────────────────────────────────────

_PROMPTS_YAML     = Path(__file__).parent / "config" / "prompts.yaml"
_COT_PROMPTS_YAML = Path(__file__).parent / "config" / "cot_prompts.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


_PROMPTS     = _load_yaml(_PROMPTS_YAML)
_COT_PROMPTS = _load_yaml(_COT_PROMPTS_YAML)


def _sys_msg() -> list[dict]:
    s = _PROMPTS.get("assessment_system", "")
    return [{"role": "system", "content": s}] if s else []


def _cls_question(class_names: list[str], task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        inner = _PROMPTS["visual_classification_weed"]
    else:
        inner = _PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    inner += f" Choose exactly one from: {', '.join(class_names)}."
    return _PROMPTS["assessment_classification"].format(question=inner)


def _desc_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        return _PROMPTS["visual_description_weed"]
    q = _PROMPTS["visual_description_disease"]
    return q.format(crop=crop or "plant") if "{crop}" in q else q


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, use_4bit: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    print(f"\n[cot_vision] Loading {model_id}  (4bit={use_4bit})")
    kwargs: dict = {}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"

    model     = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ── Inference helpers ──────────────────────────────────────────────────────────

def _tokenize(messages: list[dict], processor, device: str) -> dict:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


@torch.no_grad()
def _generate(model, processor, inputs: dict, max_new_tokens: int = 256) -> str:
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip thinking blocks and repetitions
    raw = _THINK_RE.sub("", raw).strip()
    m = _REPEAT_RE.search(raw)
    return raw[:m.start()].strip() if m else raw


@torch.no_grad()
def infer_self_cot(
    image: PILImage.Image, class_names: list[str],
    task_type: str, crop: str | None,
    model, processor, device: str,
) -> str:
    """Generate visual description, then classify using it as context."""
    desc_q = _desc_question(task_type, crop)

    # Step 1: generate description
    desc_msgs = _sys_msg() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": desc_q},
    ]}]
    desc_inputs = _tokenize(desc_msgs, processor, device)
    description = _generate(model, processor, desc_inputs, max_new_tokens=256)

    # Step 2: classify with description in history
    cls_q    = _cls_question(class_names, task_type, crop)
    messages = _sys_msg() + [
        {"role": "user",      "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": desc_q},
        ]},
        {"role": "assistant", "content": description},
        {"role": "user",      "content": cls_q},
    ]
    cls_inputs = _tokenize(messages, processor, device)
    return _generate(model, processor, cls_inputs, max_new_tokens=64)


@torch.no_grad()
def infer_cot_yaml(
    image: PILImage.Image, class_names: list[str],
    task_type: str, crop: str | None,
    model, processor, device: str,
) -> str:
    """Multi-step structured CoT from cot_prompts.yaml, then classify."""
    cot_key = _COT_PROMPTS["task_type_to_cot"].get(task_type, "disease_pest")
    steps   = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
    cot_sfx = _COT_PROMPTS["cot_classification_suffix"].strip()

    # First step includes the image
    messages = _sys_msg() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": steps[0]},
    ]}]
    for i, step in enumerate(steps):
        if i > 0:
            messages.append({"role": "user", "content": step})
        step_inputs   = _tokenize(messages, processor, device)
        step_response = _generate(model, processor, step_inputs, max_new_tokens=256)
        messages.append({"role": "assistant", "content": step_response})

    # Final classification turn
    cls_q  = _cls_question(class_names, task_type, crop) + " " + cot_sfx
    messages.append({"role": "user", "content": cls_q})
    cls_inputs = _tokenize(messages, processor, device)
    return _generate(model, processor, cls_inputs, max_new_tokens=64)


@torch.no_grad()
def _judge_one(gt: str, pred: str, model, processor, device: str) -> tuple[int, str]:
    system = _PROMPTS.get("judge_system_vision", "")
    template = _PROMPTS["judge_classification_template"]
    prompt = template.format(gt=str(gt)[:400], pred=str(pred)[:400])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw = _THINK_RE.sub("", raw).strip()
    is_match = _parse_judge_response(raw)
    return is_match, raw[:300]


_JUDGE_JSON_RE = re.compile(r'\{.*?\}', re.DOTALL)


def _parse_judge_response(raw: str) -> int:
    m = _JUDGE_JSON_RE.search(raw)
    if not m:
        return 0
    try:
        d = json.loads(m.group())
        return int(bool(d.get("is_match", False)))
    except Exception:
        return 0


def judge_rows(rows: list[dict], model, processor, device: str) -> list[dict]:
    """Run LLM-as-judge on all rows using the already-loaded model."""
    print(f"\n[cot_vision] Judging {len(rows)} predictions...")
    judged = []
    for row in tqdm(rows, desc="  judging", ncols=70):
        is_match, reasoning = _judge_one(
            row["ground_truth"], row["prediction"], model, processor, device
        )
        judged.append({**row, "judged_correct": is_match, "judge_reasoning": reasoning})
    return judged


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_classes(
    ds_name: str,
    n_samples: int | None,
    test_split: dict | None = None,
) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} restricted to test_split when provided."""
    import agml
    loader = agml.data.AgMLDataLoader(ds_name)

    class_name_map: list[str] | None = None
    for src in (loader, getattr(loader, "dataset_info", None)):
        if src is None:
            continue
        for attr in ("class_names", "class_labels", "classes"):
            val = getattr(src, attr, None)
            if val is not None:
                class_name_map = list(val)
                break
        if class_name_map is not None:
            break

    contents = loader.export_contents()
    by_class: dict[str, list[Path]] = {}
    for path, label in contents.items():
        if class_name_map is not None and isinstance(label, int):
            lbl = class_name_map[label] if label < len(class_name_map) else str(label)
        else:
            lbl = str(label) if not isinstance(label, str) else label
        by_class.setdefault(lbl, []).append(Path(str(path)))

    result: dict[str, list[Path]] = {}
    split_cls = (test_split or {}).get(ds_name, {})
    for cls, paths in by_class.items():
        if split_cls and cls in split_cls:
            # Restrict to the exact images used in the vision assessment test split
            allowed = set(split_cls[cls])
            paths = [p for p in paths if p.name in allowed]
        names = sample_class_images([p.name for p in paths], n_samples)
        name_set = set(names)
        result[cls] = [p for p in paths if p.name in name_set]
    return result


# ── Per-dataset run ────────────────────────────────────────────────────────────

CONDITIONS = ("self_cot", "cot_yaml")


def run_dataset(
    ds_name: str,
    model, processor,
    device: str,
    n_samples: int | None,
    test_split: dict | None = None,
) -> list[dict]:
    print(f"\n  {ds_name}")
    try:
        by_class = load_dataset_classes(ds_name, n_samples, test_split)
    except Exception as e:
        print(f"  [skip] {e}")
        return []

    task_type, crop = DATASET_TASK_TYPES.get(ds_name, ("disease", "plant"))
    class_names = sorted(by_class.keys())
    print(f"  task_type={task_type}  crop={crop}  {len(class_names)} classes")

    rows: list[dict] = []
    for cls, paths in by_class.items():
        pbar = tqdm(paths, desc=f"    {cls[:30]}", leave=False)
        for path in pbar:
            try:
                image = PILImage.open(path).convert("RGB")
            except Exception:
                continue

            img_name = path.name

            pred_self = infer_self_cot(image, class_names, task_type, crop,
                                       model, processor, device)
            pred_cot  = infer_cot_yaml(image, class_names, task_type, crop,
                                       model, processor, device)

            for condition, pred in [
                ("self_cot", pred_self),
                ("cot_yaml", pred_cot),
            ]:
                resolved = resolve_class(pred, class_names)
                correct  = (resolved or "").lower() == cls.lower()
                c_match  = contains_match(resolved or pred, cls)
                rows.append({
                    "agml_dataset":   ds_name,
                    "class_label":    cls,
                    "task_type":      task_type,
                    "crop":           crop or "",
                    "image_name":     img_name,
                    "condition":      condition,
                    "prediction":     pred,
                    "resolved_class": resolved or "",
                    "ground_truth":   cls,
                    "correct":        int(correct),
                    "contains_match": int(c_match),
                })

    return rows


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """Compute direct and judged F1 per (condition, dataset) and overall per condition."""
    from collections import defaultdict
    by_cond_ds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_ds[(r["condition"], r["agml_dataset"])].append(r)

    per_condition: dict[str, dict] = {}
    for (cond, ds), ds_rows in sorted(by_cond_ds.items()):
        refs  = [r["ground_truth"]   for r in ds_rows]
        preds = [r["resolved_class"] if r["correct"] else "__" for r in ds_rows]
        m     = compute_classification_metrics(refs, preds)

        judged = [r for r in ds_rows if r.get("judged_correct", "") != ""]
        if judged:
            j_refs  = [r["ground_truth"] for r in judged]
            j_preds = [r["resolved_class"] if int(r["judged_correct"]) else "__"
                       for r in judged]
            jm = compute_classification_metrics(j_refs, j_preds)
            m["judge_f1_macro"] = jm["f1_macro"]
            m["judge_accuracy"] = jm["accuracy"]

        per_condition.setdefault(cond, {})[ds] = m

    overall: dict[str, dict] = {}
    for cond, per_ds in per_condition.items():
        f1s  = [v["f1_macro"]       for v in per_ds.values()]
        jf1s = [v["judge_f1_macro"] for v in per_ds.values() if "judge_f1_macro" in v]
        overall[cond] = {
            "f1_macro":       float(np.mean(f1s))  if f1s  else 0.0,
            "judge_f1_macro": float(np.mean(jf1s)) if jf1s else None,
            "per_dataset":    per_ds,
        }

    return overall


# ── CSV / JSON I/O ─────────────────────────────────────────────────────────────

_CLS_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "condition", "prediction", "resolved_class", "ground_truth",
    "correct", "contains_match", "judged_correct", "judge_reasoning",
]


def write_cls_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CLS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_condition_comparison(metrics: dict, output_path: Path) -> None:
    """Bar chart: F1 per condition per dataset."""
    import matplotlib.pyplot as plt

    conditions = [c for c in ("self_cot", "cot_yaml") if c in metrics]
    if not conditions:
        return

    all_ds = sorted({ds for cond in conditions for ds in metrics[cond]["per_dataset"]})
    if not all_ds:
        return

    x = np.arange(len(all_ds))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(10, len(all_ds) * 1.4), 5))

    colors = {"self_cot": "#1f77b4", "cot_yaml": "#ff7f0e"}
    labels = {"self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}

    for i, cond in enumerate(conditions):
        f1s = [metrics[cond]["per_dataset"].get(ds, {}).get("f1_macro", 0.0) for ds in all_ds]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=labels[cond],
                      color=colors.get(cond, f"C{i}"), alpha=0.85)
        for bar, v in zip(bars, f1s):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    clean_ds = [d.replace("_classification", "").replace("_", " ") for d in all_ds]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_ds, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("CoT Vision Assessment — F1 Macro by Condition and Dataset")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoT Vision Assessment — self-description and structured CoT conditions"
    )
    parser.add_argument("--model",       default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output_dir",  default="outputs/cot_vision/")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max images per class (default: all — uses test_split when available)")
    parser.add_argument("--datasets",    nargs="+", default=DEFAULT_DATASETS,
                        help="AgML dataset names to evaluate")
    parser.add_argument("--test_split",  default="outputs/linear_probe/test_split.json",
                        help="Path to test_split.json from linear probe — restricts each class "
                             "to the same images used in vision assessment")
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    test_split: dict | None = None
    split_path = Path(args.test_split)
    if split_path.exists():
        import json as _json
        test_split = _json.loads(split_path.read_text())
        print(f"[cot_vision] test_split loaded from {split_path}")
    else:
        print(f"[cot_vision] test_split not found at {split_path} — using all available images")

    model, processor = load_model(args.model, args.use_4bit)
    _p     = next(model.parameters())
    device = _p.device if str(_p.device) != "meta" else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[cot_vision] input device: {device}")

    all_rows: list[dict] = []
    t0 = time.perf_counter()

    print("\n=== Running CoT conditions ===")
    for ds in args.datasets:
        rows = run_dataset(ds, model, processor, device, args.n_samples, test_split)
        all_rows.extend(rows)

    elapsed = time.perf_counter() - t0
    print(f"\nInference done in {elapsed:.1f}s  ({len(all_rows)} rows)")

    all_rows = judge_rows(all_rows, model, processor, device)

    # Save CSV
    cls_csv = output_dir / f"cot_cls_{safe_model}.csv"
    write_cls_csv(all_rows, cls_csv)

    # Compute + save metrics
    metrics = compute_metrics(all_rows)
    metrics_path = output_dir / "cot_vision_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, "n_samples": args.n_samples,
                   "conditions": metrics}, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Print summary
    print("\n── CoT Vision Assessment Summary ─────────────────")
    cond_labels = {"self_cot": "Self-CoT", "cot_yaml": "CoT YAML"}
    for cond, m in metrics.items():
        jf1 = m.get('judge_f1_macro')
        jf1_str = f"  judge_F1={jf1:.3f}" if jf1 is not None else ""
        print(f"  {cond_labels.get(cond, cond):<20}  F1={m['f1_macro']:.3f}{jf1_str}")
        for ds, dm in sorted(m["per_dataset"].items()):
            short = ds.replace("_classification", "").replace("_", " ")
            print(f"    {short:<45}  F1={dm['f1_macro']:.3f}")

    # Chart
    plot_condition_comparison(metrics, output_dir / "cot_comparison.png")

    unload_model(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
