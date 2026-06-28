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

    # Logit softcapping (Gemma-4 uses final_logit_softcapping=30.0)
    softcap = None
    for cfg_src in (getattr(lm, "config", None), getattr(model, "config", None)):
        if cfg_src is None:
            continue
        for attr in ("final_logit_softcapping",):
            v = getattr(cfg_src, attr, None) or getattr(
                getattr(cfg_src, "text_config", None) or {}, attr, None
            )
            if v:
                softcap = float(v)
                break
        if softcap:
            break

    print(f"[logit_lens] n_lm_layers={n_layers}  "
          f"final_norm={type(final_norm).__name__}  lm_head={type(lm_head).__name__}  "
          f"softcap={softcap}")
    return final_norm, lm_head, n_layers, softcap


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_classes(ds_name: str, n_samples: int) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} for a dataset via agml."""
    import agml
    loader = agml.data.AgMLDataLoader(ds_name)

    # Resolve integer labels → human-readable class name strings
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

    print(f"  [agml] class_name_map: {class_name_map}")

    contents = loader.export_contents()
    by_class: dict[str, list[Path]] = {}
    for path, label in contents.items():
        if class_name_map is not None and isinstance(label, int):
            lbl = class_name_map[label] if label < len(class_name_map) else str(label)
        else:
            lbl = str(label) if not isinstance(label, str) else label
        by_class.setdefault(lbl, []).append(Path(str(path)))

    rng = np.random.default_rng(42)
    return {
        cls: [paths[i] for i in rng.choice(len(paths),
              min(n_samples, len(paths)), replace=False)]
        for cls, paths in by_class.items()
    }


# ── Prompt building ────────────────────────────────────────────────────────────

_PROMPTS_YAML = Path(__file__).parent.parent.parent / "src" / "config" / "prompts.yaml"

def _load_prompts() -> dict:
    import yaml
    with open(_PROMPTS_YAML) as f:
        return yaml.safe_load(f)

_PROMPTS = _load_prompts()


def build_inputs(image: Image.Image, class_names: list[str],
                 processor, device: str, crop: str = "plant") -> dict:
    inner_q = _PROMPTS["visual_classification_disease"].format(crop=crop)
    inner_q += f" Choose exactly one from: {', '.join(class_names)}."
    question = _PROMPTS["assessment_classification"].format(question=inner_q)

    sys_msg = _PROMPTS.get("assessment_system", "")
    messages = (
        [{"role": "system", "content": sys_msg}] if sys_msg else []
    ) + [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


# ── Logit lens forward pass ────────────────────────────────────────────────────

_debug_printed = False  # print token diagnostics once per run

_VOCAB_SIZE = None  # filled on first pass


def _get_target_ids(processor, correct_class: str) -> tuple[int | None, int | None]:
    ids       = processor.tokenizer.encode(correct_class,       add_special_tokens=False)
    ids_space = processor.tokenizer.encode(" " + correct_class, add_special_tokens=False)
    return (ids[0] if ids else None), (ids_space[0] if ids_space else None)


def _rank_of(probs: torch.Tensor, token_id: int | None) -> int:
    """1-indexed rank of token_id in descending probability order."""
    if token_id is None:
        return int(probs.numel())
    # argsort descending: position 0 = highest prob
    return int((torch.argsort(probs, descending=True) == token_id).nonzero(as_tuple=False)[0, 0].item()) + 1


@torch.no_grad()
def logit_lens_pass(
    model, processor,
    final_norm, lm_head, softcap,
    image: Image.Image,
    class_names: list[str],
    correct_class: str,
    device: str,
) -> dict:
    """
    Generates exactly 1 token with output_hidden_states=True.
    At each layer, projects the hidden state at the generated position through
    final_norm + lm_head and records the rank of the correct class token.
    Rank is more robust than raw probability under 4-bit quantization.
    """
    global _debug_printed, _VOCAB_SIZE

    inputs = build_inputs(image, class_names, processor, device)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # hidden_states[0] = per-layer hidden states for the first generated token
    # Each element: (batch, 1, hidden_dim) — just the new token position
    gen_hs = gen_out.hidden_states[0]   # tuple of n_layers+1 tensors

    generated_id  = int(gen_out.sequences[0, -1])
    generated_tok = processor.tokenizer.decode([generated_id])

    target_id, target_id_space = _get_target_ids(processor, correct_class)
    norm_device = next(final_norm.parameters()).device

    per_layer: list[dict] = []
    for layer_idx, h in enumerate(gen_hs):
        h_tok    = h[0, -1, :].unsqueeze(0).to(norm_device)
        h_normed = final_norm(h_tok)
        logits   = lm_head(h_normed).squeeze(0).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        probs    = torch.softmax(logits, dim=-1)

        if _VOCAB_SIZE is None:
            _VOCAB_SIZE = int(probs.numel())

        rank = _rank_of(probs, target_id)
        if target_id_space is not None:
            rank = min(rank, _rank_of(probs, target_id_space))

        prob_correct = 0.0
        if target_id is not None:
            prob_correct = max(prob_correct, float(probs[target_id]))
        if target_id_space is not None:
            prob_correct = max(prob_correct, float(probs[target_id_space]))

        top5_ids  = torch.topk(probs, 5).indices.tolist()
        top5_toks = [processor.tokenizer.decode([i]) for i in top5_ids]

        if not _debug_printed:
            # Print header once at layer 0
            if layer_idx == 0:
                print(f"\n[debug] generated='{generated_tok}' (id={generated_id})")
                print(f"[debug] correct='{correct_class}'  "
                      f"target_id={target_id} "
                      f"('{processor.tokenizer.decode([target_id]) if target_id else '?'}')")
                print(f"[debug] vocab_size={_VOCAB_SIZE}")
                print(f"[debug] {'Layer':>5}  {'Rank':>8}  {'Score':>6}  Top-3 tokens")
                print(f"[debug] {'-----':>5}  {'----':>8}  {'-----':>6}  ----------")
            # Print every 5 layers + final
            if layer_idx % 5 == 0 or layer_idx == len(gen_hs) - 1:
                top3 = [(processor.tokenizer.decode([i]), f"{float(probs[i]):.3f}")
                        for i in top5_ids[:3]]
                vs = _VOCAB_SIZE or 262144
                score = 1.0 - rank / vs
                print(f"[debug] {layer_idx:>5}  {rank:>8}  {score:>6.4f}  "
                      f"{' | '.join(f'{t!r}({p})' for t, p in top3)}")
            if layer_idx == len(gen_hs) - 1:
                _debug_printed = True
                print()

        per_layer.append({
            "prob_correct": prob_correct,
            "rank":         rank,
            "top_token":    top5_toks[0],
            "top_prob":     float(probs[top5_ids[0]]),
        })

    return {
        "class":         correct_class,
        "n_layers":      len(per_layer),
        "per_layer":     per_layer,
        "target_tok":    processor.tokenizer.decode([target_id]) if target_id else "?",
        "generated_tok": generated_tok,
    }


# ── Per-dataset aggregation ────────────────────────────────────────────────────

def run_dataset(
    ds_name: str,
    model, processor,
    final_norm, lm_head, softcap,
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
                model, processor, final_norm, lm_head, softcap,
                image, class_names, cls, device,
            )
            curve      = [l["prob_correct"] for l in result["per_layer"]]
            rank_curve = [l["rank"]         for l in result["per_layer"]]
            all_curves.append(curve)
            records.append({
                "dataset":    ds_name, "class": cls,
                "image":      str(path.name),
                "target_tok": result["target_tok"],
                "generated_tok": result.get("generated_tok", "?"),
                "curve":      curve,
                "rank_curve": rank_curve,
            })

    if not all_curves:
        return None

    rank_curves: list[list[int]] = [r["rank_curve"] for r in records]
    max_layers = max(len(c) for c in all_curves)
    mean_prob_curve = [
        float(np.mean([c[i] for c in all_curves if i < len(c)]))
        for i in range(max_layers)
    ]
    # Normalised rank score: 1 - rank/vocab_size  (1.0 = top-1, 0.0 = last)
    vs = _VOCAB_SIZE or 262144
    mean_rank_score = [
        float(np.mean([1.0 - rank_curves[j][i] / vs
                        for j in range(len(rank_curves)) if i < len(rank_curves[j])]))
        for i in range(max_layers)
    ]
    return {
        "dataset":         ds_name,
        "mean_curve":      mean_prob_curve,   # kept for raw json
        "mean_rank_score": mean_rank_score,   # primary plot metric
        "records":         records,
    }


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
            curve  = res.get("mean_rank_score", res["mean_curve"])
            layers = list(range(len(curve)))
            label  = res["dataset"].replace("_classification", "").replace("_", " ")
            ax.plot(layers, curve, color=color, linewidth=2, label=label)
            ax.fill_between(layers, curve, alpha=0.08, color=color)

        ax.set_xlabel("Transformer Layer", fontsize=10)
        ax.set_ylabel("Normalised rank score  (↑ = closer to top-1)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=1)

    _plot_panel(ax_hard, hard_results, hard_colors, "Hard Datasets (low VLM F1)")
    _plot_panel(ax_easy, easy_results, easy_colors, "Easy Datasets (high VLM F1)")

    fig.suptitle(
        "Logit Lens: Correct Class Token Rank Across LLM Depth\n"
        "(generated-position hidden states → lm_head  |  1.0 = rank-1 in vocab)",
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
    # With device_map="auto" the model may span multiple GPUs; inputs go to
    # the first device (where the embedding layer lives).
    _p = next(model.parameters())
    device = _p.device if str(_p.device) != "meta" else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[logit_lens] input device: {device}")
    final_norm, lm_head, _, softcap = discover_lm_components(model)

    hard_results, easy_results = [], []

    print("\n=== Hard datasets ===")
    for ds in args.hard_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head, softcap,
                          args.n_samples, device)
        if res:
            hard_results.append(res)

    print("\n=== Easy datasets ===")
    for ds in args.easy_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head, softcap,
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
