"""
Logit Lens across Transformer Depth — Agricultural VLM
=======================================================
Tracks the probability of the correct class token at each LLM transformer
layer by projecting intermediate hidden states through the final layer norm
and LM head (no training, no vLLM — HuggingFace forward pass only).

Compares 3 hard datasets (near-zero VLM judged F1) against 3 easy datasets
to show where in the LLM the visual concept fails to become the answer token.

Usage:
    python experiments/logit_lens/logit_lens.py \\
        --model  google/gemma-4-26B-A4B-it \\
        --output_dir outputs/logit_lens/ \\
        --n_samples 8 \\
        --use_4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Defaults (from per-dataset difficulty analysis) ────────────────────────────

HARD_DATASETS = [
    "guava_disease_pakistan",
    "sugarcane_damage_usa",
    "plant_seedlings_aarhus",
]

EASY_DATASETS = [
    "bean_disease_uganda",
    "blackgram_plant_leaf_disease_classification",
    "corn_maize_leaf_disease",
]

_TASK_LABELS = {"pest_damage": "pest/damage"}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, use_4bit: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    print(f"[logit_lens] Loading {model_id}  (4bit={use_4bit})")
    kwargs: dict = {"attn_implementation": "eager"}
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


def discover_lm_components(model):
    """Return (lm_layers, final_norm, lm_head) from any Gemma-4 variant."""
    # Walk common attribute paths for Gemma-4
    candidates = [
        model,
        getattr(model, "model",          None),
        getattr(model, "language_model", None),
    ]
    lm = None
    for c in candidates:
        if c is not None and hasattr(c, "language_model"):
            lm = c.language_model
            break
    if lm is None:
        lm = model  # fallback: model itself is the LM

    # Final norm + lm_head
    inner = getattr(lm, "model", lm)
    final_norm = getattr(inner, "norm", None)
    lm_head    = (getattr(lm, "lm_head", None)
                  or getattr(model, "lm_head", None))

    if final_norm is None or lm_head is None:
        raise RuntimeError(
            f"Could not discover final_norm / lm_head. "
            f"Top-level attrs: {[n for n,_ in model.named_children()]}"
        )

    layers = getattr(inner, "layers", None)
    n_layers = len(layers) if layers is not None else "?"
    print(f"[logit_lens] n_lm_layers={n_layers}  "
          f"final_norm={type(final_norm).__name__}  lm_head={type(lm_head).__name__}")
    return final_norm, lm_head, n_layers


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_classes(ds_name: str, n_samples: int) -> dict[str, list[Path]]:
    """Return {class_label: [image_path, ...]} for a dataset via agml."""
    import agml
    loader = agml.data.AgMLDataLoader(ds_name)
    contents = loader.export_contents()
    by_class: dict[str, list[Path]] = {}
    for path, label in contents.items():
        lbl = str(label) if not isinstance(label, str) else label
        by_class.setdefault(lbl, []).append(Path(str(path)))
    # sample
    rng = np.random.default_rng(42)
    return {
        cls: [paths[i] for i in rng.choice(len(paths),
              min(n_samples, len(paths)), replace=False)]
        for cls, paths in by_class.items()
    }


# ── Prompt building ────────────────────────────────────────────────────────────

_SYS = "You are an expert agricultural scientist."
_SUFFIX = "\nClass:"  # makes first generated token = start of class name


def build_inputs(image: Image.Image, class_names: list[str],
                 processor, device: str) -> dict:
    question = (
        f"Identify the plant condition shown. "
        f"Choose exactly one from: {', '.join(class_names)}."
        + _SUFFIX
    )
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]},
    ]
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
    except Exception:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


# ── Logit lens forward pass ────────────────────────────────────────────────────

@torch.no_grad()
def logit_lens_pass(
    model, processor,
    final_norm, lm_head,
    image: Image.Image,
    class_names: list[str],
    correct_class: str,
    device: str,
) -> dict:
    """
    Single forward pass. Returns per-layer probability of the first token of
    the correct class name, plus the top predicted token at each layer.
    """
    inputs = build_inputs(image, class_names, processor, device)
    seq_len = inputs["input_ids"].shape[1]

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    # [0] = embedding output, [1..] = layer outputs
    hidden_states = outputs.hidden_states  # from the LM component

    # First token of correct class (strip leading space variants)
    correct_ids = processor.tokenizer.encode(
        correct_class, add_special_tokens=False
    )
    correct_ids_space = processor.tokenizer.encode(
        " " + correct_class, add_special_tokens=False
    )
    target_id = correct_ids[0] if correct_ids else None
    target_id_space = correct_ids_space[0] if correct_ids_space else None

    last_pos = seq_len - 1
    per_layer: list[dict] = []

    for layer_idx, h in enumerate(hidden_states):
        h_last = h[0, last_pos, :].unsqueeze(0)  # (1, hidden)
        try:
            h_normed = final_norm(h_last)
        except Exception:
            h_normed = h_last
        logits = lm_head(h_normed).squeeze(0).float()  # (vocab)
        probs  = torch.softmax(logits, dim=-1)

        prob_correct = 0.0
        if target_id is not None:
            prob_correct = float(probs[target_id])
        if target_id_space is not None:
            prob_correct = max(prob_correct, float(probs[target_id_space]))

        top_id  = int(torch.argmax(probs))
        top_tok = processor.tokenizer.decode([top_id])
        per_layer.append({
            "prob_correct": prob_correct,
            "top_token": top_tok,
            "top_prob": float(probs[top_id]),
        })

    return {
        "class":      correct_class,
        "n_layers":   len(per_layer),
        "per_layer":  per_layer,
        "target_tok": processor.tokenizer.decode([target_id]) if target_id else "?",
    }


# ── Per-dataset aggregation ────────────────────────────────────────────────────

def run_dataset(
    ds_name: str,
    model, processor,
    final_norm, lm_head,
    n_samples: int,
    device: str,
) -> dict | None:
    print(f"\n  Loading {ds_name}...")
    try:
        by_class = load_dataset_classes(ds_name, n_samples)
    except Exception as e:
        print(f"  [skip] {ds_name}: {e}")
        return None

    class_names = sorted(by_class.keys())
    print(f"  {len(class_names)} classes, up to {n_samples} images each")

    all_curves: list[list[float]] = []  # one list per image
    records: list[dict] = []

    for cls, paths in by_class.items():
        pbar = tqdm(paths, desc=f"    {cls[:30]}", leave=False)
        for path in pbar:
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            result = logit_lens_pass(
                model, processor, final_norm, lm_head,
                image, class_names, cls, device,
            )
            curve = [l["prob_correct"] for l in result["per_layer"]]
            all_curves.append(curve)
            records.append({
                "dataset": ds_name, "class": cls,
                "image": str(path.name),
                "target_tok": result["target_tok"],
                "curve": curve,
            })

    if not all_curves:
        return None

    max_layers = max(len(c) for c in all_curves)
    mean_curve = [
        float(np.mean([c[i] for c in all_curves if i < len(c)]))
        for i in range(max_layers)
    ]
    return {"dataset": ds_name, "mean_curve": mean_curve, "records": records}


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_logit_lens(
    hard_results: list[dict],
    easy_results: list[dict],
    output_path: Path,
) -> None:
    fig, (ax_hard, ax_easy) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    hard_colors = ["#d62728", "#e6550d", "#e7969c"]
    easy_colors = ["#2ca02c", "#1f77b4", "#9467bd"]

    def _plot_panel(ax, results, colors, title):
        for res, color in zip(results, colors):
            curve = res["mean_curve"]
            layers = list(range(len(curve)))
            ax.plot(layers, curve, color=color, linewidth=2,
                    label=res["dataset"].replace("_classification", "").replace("_", " "))
            ax.fill_between(layers, curve, alpha=0.08, color=color)

        # random baseline (1 / mean vocab size is too small; use 1/n_classes proxy)
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("Transformer Layer", fontsize=10)
        ax.set_ylabel("P(correct class first token)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    _plot_panel(ax_hard, hard_results, hard_colors, "Hard Datasets (low VLM F1)")
    _plot_panel(ax_easy, easy_results, easy_colors, "Easy Datasets (high VLM F1)")

    fig.suptitle(
        "Logit Lens: Correct Class Token Probability Across LLM Depth\n"
        "(post-projector hidden states → lm_head, last input position)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Logit lens across LLM depth for agricultural VLM")
    parser.add_argument("--model",       default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output_dir",  default="outputs/logit_lens/")
    parser.add_argument("--n_samples",   type=int, default=8,
                        help="Images per class")
    parser.add_argument("--hard_datasets", nargs="+", default=HARD_DATASETS)
    parser.add_argument("--easy_datasets", nargs="+", default=EASY_DATASETS)
    parser.add_argument("--use_4bit",    action="store_true",
                        help="Load in 4-bit NF4 (saves ~60%% VRAM)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, args.use_4bit)
    device = next(model.parameters()).device
    final_norm, lm_head, _ = discover_lm_components(model)

    hard_results, easy_results = [], []

    print("\n=== Hard datasets ===")
    for ds in args.hard_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head,
                          args.n_samples, device)
        if res:
            hard_results.append(res)

    print("\n=== Easy datasets ===")
    for ds in args.easy_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head,
                          args.n_samples, device)
        if res:
            easy_results.append(res)

    # Save raw curves
    all_records = []
    for r in hard_results + easy_results:
        all_records.extend(r["records"])
    raw_path = output_dir / "logit_lens_raw.json"
    with open(raw_path, "w") as f:
        json.dump({"hard": hard_results, "easy": easy_results,
                   "records": all_records}, f, indent=1)
    print(f"  Saved: {raw_path}")

    if hard_results or easy_results:
        plot_logit_lens(
            hard_results, easy_results,
            output_dir / "logit_lens.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
