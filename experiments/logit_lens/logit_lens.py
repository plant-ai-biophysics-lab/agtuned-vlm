"""
Logit Lens across Transformer Depth — Agricultural VLM
=======================================================
Tracks the probability of the correct class token at each LLM transformer
layer by projecting intermediate hidden states through the final layer norm
and LM head (no training, no vLLM — HuggingFace forward pass only).

Compares 3 hard datasets (near-zero VLM judged F1) against 3 easy datasets
to show where in the LLM the visual concept fails to become the answer token.

Four conditions compared:
  1. normal    — standard vision assessment prompt
  2. self_cot  — model generates its own description first (2-turn history)
  3. cot_yaml  — structured multi-step CoT from cot_prompts.yaml
  4. aug_vision — pre-written description injected (upper bound, requires --lang_dir)

Usage:
    python experiments/logit_lens/logit_lens.py \\
        --model  google/gemma-4-26B-A4B-it \\
        --output_dir outputs/logit_lens/ \\
        --n_samples 8 \\
        --use_4bit

    # With all 4 conditions:
    python experiments/logit_lens/logit_lens.py \\
        --model  google/gemma-4-26B-A4B-it \\
        --output_dir outputs/logit_lens/ \\
        --n_samples 4 \\
        --use_4bit \\
        --run_cot \\
        --lang_dir outputs/language_assessment/
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

# ── Dataset task-type registry ────────────────────────────────────────────────
# Maps agml dataset name → (task_type, crop)
# task_type determines which prompts to use (disease/pest_damage → disease path,
# weed_identification → weed path)

DATASET_TASK_TYPES: dict[str, tuple[str, str | None]] = {
    "guava_disease_pakistan":                      ("disease",              "guava"),
    "sugarcane_damage_usa":                        ("pest_damage",          "sugarcane"),
    "plant_seedlings_aarhus":                      ("weed_identification",  None),
    "bean_disease_uganda":                         ("disease",              "bean"),
    "blackgram_plant_leaf_disease_classification": ("disease",              "blackgram"),
    "corn_maize_leaf_disease":                     ("disease",              "corn"),
}

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
    """Return (final_norm, lm_head, n_layers, softcap) from any Gemma-4 variant."""
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
        lm = model

    inner      = getattr(lm, "model", lm)
    final_norm = getattr(inner, "norm", None)
    lm_head    = (getattr(lm, "lm_head", None)
                  or getattr(model, "lm_head", None))

    if final_norm is None or lm_head is None:
        raise RuntimeError(
            f"Could not discover final_norm / lm_head. "
            f"Top-level attrs: {[n for n,_ in model.named_children()]}"
        )

    layers   = getattr(inner, "layers", None)
    n_layers = len(layers) if layers is not None else "?"

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

_PROMPTS_YAML     = Path(__file__).parent.parent.parent / "src" / "config" / "prompts.yaml"
_COT_PROMPTS_YAML = Path(__file__).parent.parent.parent / "src" / "config" / "cot_prompts.yaml"


def _load_prompts() -> dict:
    import yaml
    with open(_PROMPTS_YAML) as f:
        return yaml.safe_load(f)


def _load_cot_prompts() -> dict:
    import yaml
    with open(_COT_PROMPTS_YAML) as f:
        return yaml.safe_load(f)


_PROMPTS     = _load_prompts()
_COT_PROMPTS = _load_cot_prompts()


def _sys_messages() -> list[dict]:
    sys_msg = _PROMPTS.get("assessment_system", "")
    return [{"role": "system", "content": sys_msg}] if sys_msg else []


def _classification_question(class_names: list[str], task_type: str,
                              crop: str | None, suffix: str = "") -> str:
    """Build the inner classification question, wrapped in assessment_classification."""
    if task_type == "weed_identification":
        inner_q = _PROMPTS["visual_classification_weed"]
    else:
        inner_q = _PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    inner_q += f" Choose exactly one from: {', '.join(class_names)}." + suffix
    return _PROMPTS["assessment_classification"].format(question=inner_q)


def _description_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        return _PROMPTS["visual_description_weed"]
    q = _PROMPTS["visual_description_disease"]
    return q.format(crop=crop or "plant") if "{crop}" in q else q


def _tokenize(messages: list[dict], processor, device: str) -> dict:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def build_inputs(image: Image.Image, class_names: list[str],
                 task_type: str, crop: str | None,
                 processor, device: str) -> dict:
    """Standard vision assessment prompt, task-type aware."""
    question = _classification_question(class_names, task_type, crop)
    messages = _sys_messages() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question},
    ]}]
    return _tokenize(messages, processor, device)


def build_aug_inputs(image: Image.Image, class_names: list[str], description: str,
                     task_type: str, crop: str | None,
                     processor, device: str) -> dict:
    """Augmented vision prompt with pre-written description as context."""
    if task_type == "weed_identification":
        inner_q = _PROMPTS["visual_classification_weed"]
    else:
        inner_q = _PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    inner_q += f" Choose exactly one from: {', '.join(class_names)}."
    question = _PROMPTS["desc_augmented_classification"].format(
        description=description, question=inner_q,
    )
    messages = _sys_messages() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question},
    ]}]
    return _tokenize(messages, processor, device)


@torch.no_grad()
def _generate_text(model, processor, inputs: dict, max_new_tokens: int = 256) -> str:
    """Run greedy decode and return decoded new tokens only."""
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def build_self_cot_inputs(
    image: Image.Image, class_names: list[str],
    task_type: str, crop: str | None,
    model, processor, device: str,
) -> dict:
    """
    Self-description CoT (2-turn):
      Turn 1: image + visual_description_* → model generates description
      Turn 2: classification question with description history
    Logit lens then runs on the generated position in turn 2.
    """
    desc_q = _description_question(task_type, crop)

    # Build and run description turn
    desc_messages = _sys_messages() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": desc_q},
    ]}]
    desc_inputs  = _tokenize(desc_messages, processor, device)
    description  = _generate_text(model, processor, desc_inputs, max_new_tokens=256)

    # Append description to history, then add classification turn
    cls_question = _classification_question(class_names, task_type, crop)
    messages = _sys_messages() + [
        {"role": "user",      "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": desc_q},
        ]},
        {"role": "assistant", "content": description},
        {"role": "user",      "content": cls_question},
    ]
    return _tokenize(messages, processor, device)


@torch.no_grad()
def build_cot_yaml_inputs(
    image: Image.Image, class_names: list[str],
    task_type: str, crop: str | None,
    model, processor, device: str,
) -> dict:
    """
    Structured multi-step CoT from cot_prompts.yaml:
      For each step in thinking_prompts[task_type_to_cot[task_type]]:
        - Append user step question, generate response, append to history
      Final turn: base classification question + cot_classification_suffix
    Logit lens runs on the generated position of the final turn.
    """
    cot_key  = _COT_PROMPTS["task_type_to_cot"].get(task_type, "disease_pest")
    steps    = _COT_PROMPTS["thinking_prompts"][cot_key]["steps"]
    cot_sfx  = _COT_PROMPTS["cot_classification_suffix"]

    # First turn includes the image + first step question
    messages = _sys_messages() + [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": steps[0]},
    ]}]

    for i, step in enumerate(steps):
        if i > 0:
            messages.append({"role": "user", "content": step})
        step_inputs   = _tokenize(messages, processor, device)
        step_response = _generate_text(model, processor, step_inputs, max_new_tokens=256)
        messages.append({"role": "assistant", "content": step_response})

    # Final classification turn — bridges from analysis to specific class name
    final_q = _classification_question(class_names, task_type, crop) + " " + cot_sfx.strip()
    messages.append({"role": "user", "content": final_q})
    return _tokenize(messages, processor, device)


def load_synthesized_descriptions(jsonl_path: Path) -> dict[tuple[str, str], str]:
    """Load {(dataset, class_label): synthesized_description} from agml_descriptions.jsonl.

    Uses the same synthesized_description that language assessment classification uses,
    so aug_vision sees no class-name leakage.
    """
    import json
    desc_map: dict[tuple[str, str], str] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec  = json.loads(line)
            meta = rec.get("metadata", {})
            ds   = meta.get("agml_dataset", "").strip()
            cls  = meta.get("class_label",  "").strip()
            desc = meta.get("synthesized_description", "").strip()
            if ds and cls and desc:
                desc_map[(ds, cls)] = desc
    print(f"[logit_lens] Loaded {len(desc_map)} synthesized descriptions from {jsonl_path.name}")
    return desc_map


def _lookup_description(desc_map: dict, ds_name: str, cls: str) -> str | None:
    for key_cls in (cls, cls.replace("_", " "), cls.replace(" ", "_")):
        v = desc_map.get((ds_name, key_cls))
        if v:
            return v
    return None


# ── Logit lens forward pass ────────────────────────────────────────────────────

_debug_printed = False
_VOCAB_SIZE    = None


def _get_target_ids(processor, correct_class: str) -> tuple[int | None, int | None]:
    ids       = processor.tokenizer.encode(correct_class,       add_special_tokens=False)
    ids_space = processor.tokenizer.encode(" " + correct_class, add_special_tokens=False)
    return (ids[0] if ids else None), (ids_space[0] if ids_space else None)


@torch.no_grad()
def logit_lens_pass(
    model, processor,
    final_norm, lm_head, softcap,
    image: Image.Image,
    class_names: list[str],
    correct_class: str,
    device: str,
    prebuilt_inputs: dict | None = None,
    task_type: str = "disease",
    crop: str | None = "plant",
    condition_label: str = "normal",
) -> dict:
    """
    Generates exactly 1 token with output_hidden_states=True.
    At each layer, projects the hidden state through final_norm + lm_head and
    records whether the top-1 token matches the first token of the correct class.

    Layer-wise top-1 accuracy (Wendler et al. 2024; LLM Microscope 2024):
      correct iff argmax matches either "ClassName" or " ClassName" first token.
    """
    global _debug_printed, _VOCAB_SIZE

    inputs = prebuilt_inputs if prebuilt_inputs is not None else \
             build_inputs(image, class_names, task_type, crop, processor, device)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    gen_hs        = gen_out.hidden_states[0]
    generated_id  = int(gen_out.sequences[0, -1])
    generated_tok = processor.tokenizer.decode([generated_id])

    target_id, target_id_space = _get_target_ids(processor, correct_class)
    norm_device = next(final_norm.parameters()).device

    per_layer: list[dict] = []
    for layer_idx, h in enumerate(gen_hs):
        h_tok = h[0, -1, :].unsqueeze(0).to(norm_device)
        # gen_hs[-1] is already post-final-norm (Gemma4TextModel.forward applies
        # self.norm before returning last_hidden_state). Skip double-norm for it.
        if layer_idx < len(gen_hs) - 1:
            h_normed = final_norm(h_tok)
        else:
            h_normed = h_tok
        logits = lm_head(h_normed).squeeze(0).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        probs    = torch.softmax(logits, dim=-1)

        if _VOCAB_SIZE is None:
            _VOCAB_SIZE = int(probs.numel())

        top1_id  = int(torch.argmax(probs))
        top1_tok = processor.tokenizer.decode([top1_id])

        top1_correct = int(
            top1_id == target_id or
            (target_id_space is not None and top1_id == target_id_space)
        )

        top5_ids = torch.topk(probs, 5).indices.tolist()

        if not _debug_printed and condition_label == "normal":
            if layer_idx == 0:
                print(f"\n[debug] condition={condition_label}  generated='{generated_tok}' (id={generated_id})")
                print(f"[debug] correct='{correct_class}'  "
                      f"target_id={target_id} "
                      f"('{processor.tokenizer.decode([target_id]) if target_id else '?'}')")
                print(f"[debug] vocab_size={_VOCAB_SIZE}")
                print(f"[debug] {'Layer':>5}  {'Top1':>8}  {'Match':>5}  Top-3 tokens")
                print(f"[debug] {'-----':>5}  {'----':>8}  {'-----':>5}  ----------")
            if layer_idx % 5 == 0 or layer_idx == len(gen_hs) - 1:
                top3 = [(processor.tokenizer.decode([i]), f"{float(probs[i]):.3f}")
                        for i in top5_ids[:3]]
                print(f"[debug] {layer_idx:>5}  {top1_tok!r:>8}  {'✓' if top1_correct else '✗':>5}  "
                      f"{' | '.join(f'{t!r}({p})' for t, p in top3)}")
            if layer_idx == len(gen_hs) - 1:
                _debug_printed = True
                print()

        per_layer.append({"top1_correct": top1_correct, "top1_tok": top1_tok})

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
    desc_map: dict | None = None,
    run_cot: bool = False,
) -> dict | None:
    print(f"\n  Loading {ds_name}...")
    try:
        by_class = load_dataset_classes(ds_name, n_samples)
    except Exception as e:
        print(f"  [skip] {ds_name}: {e}")
        return None

    task_type, crop = DATASET_TASK_TYPES.get(ds_name, ("disease", "plant"))
    class_names = sorted(by_class.keys())
    print(f"  task_type={task_type}  crop={crop}  "
          f"{len(class_names)} classes, up to {n_samples} images each")

    normal_curves:   list[list[int]] = []
    self_cot_curves: list[list[int]] = []
    cot_yaml_curves: list[list[int]] = []
    aug_curves:      list[list[int]] = []
    records: list[dict] = []

    for cls, paths in by_class.items():
        description = _lookup_description(desc_map, ds_name, cls) if desc_map else None
        pbar = tqdm(paths, desc=f"    {cls[:30]}", leave=False)
        for path in pbar:
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue

            # 1. Normal
            inputs_normal = build_inputs(image, class_names, task_type, crop, processor, device)
            res_normal = logit_lens_pass(
                model, processor, final_norm, lm_head, softcap,
                image, class_names, cls, device,
                prebuilt_inputs=inputs_normal,
                task_type=task_type, crop=crop, condition_label="normal",
            )
            acc_normal = [l["top1_correct"] for l in res_normal["per_layer"]]
            normal_curves.append(acc_normal)

            acc_self_cot = None
            acc_cot_yaml = None
            acc_aug      = None

            if run_cot:
                # 2. Self-description CoT
                inputs_self_cot = build_self_cot_inputs(
                    image, class_names, task_type, crop, model, processor, device,
                )
                res_self_cot = logit_lens_pass(
                    model, processor, final_norm, lm_head, softcap,
                    image, class_names, cls, device,
                    prebuilt_inputs=inputs_self_cot,
                    task_type=task_type, crop=crop, condition_label="self_cot",
                )
                acc_self_cot = [l["top1_correct"] for l in res_self_cot["per_layer"]]
                self_cot_curves.append(acc_self_cot)

                # 3. CoT YAML
                inputs_cot_yaml = build_cot_yaml_inputs(
                    image, class_names, task_type, crop, model, processor, device,
                )
                res_cot_yaml = logit_lens_pass(
                    model, processor, final_norm, lm_head, softcap,
                    image, class_names, cls, device,
                    prebuilt_inputs=inputs_cot_yaml,
                    task_type=task_type, crop=crop, condition_label="cot_yaml",
                )
                acc_cot_yaml = [l["top1_correct"] for l in res_cot_yaml["per_layer"]]
                cot_yaml_curves.append(acc_cot_yaml)

            # 4. Aug vision (requires lang_dir)
            if description:
                inputs_aug = build_aug_inputs(
                    image, class_names, description, task_type, crop, processor, device,
                )
                res_aug = logit_lens_pass(
                    model, processor, final_norm, lm_head, softcap,
                    image, class_names, cls, device,
                    prebuilt_inputs=inputs_aug,
                    task_type=task_type, crop=crop, condition_label="aug_vision",
                )
                acc_aug = [l["top1_correct"] for l in res_aug["per_layer"]]
                aug_curves.append(acc_aug)

            records.append({
                "dataset":           ds_name,
                "class":             cls,
                "image":             str(path.name),
                "target_tok":        res_normal["target_tok"],
                "generated_tok":     res_normal.get("generated_tok", "?"),
                "acc_curve":         acc_normal,
                "acc_curve_self_cot": acc_self_cot,
                "acc_curve_cot_yaml": acc_cot_yaml,
                "acc_curve_aug":     acc_aug,
            })

    if not normal_curves:
        return None

    max_layers = max(len(c) for c in normal_curves)

    def _mean_acc(curves: list[list[int]]) -> list[float]:
        return [
            float(np.mean([c[i] for c in curves if i < len(c)]))
            for i in range(max_layers)
        ]

    return {
        "dataset":                ds_name,
        "mean_accuracy":          _mean_acc(normal_curves),
        "mean_accuracy_self_cot": _mean_acc(self_cot_curves) if self_cot_curves else None,
        "mean_accuracy_cot_yaml": _mean_acc(cot_yaml_curves) if cot_yaml_curves else None,
        "mean_accuracy_aug":      _mean_acc(aug_curves)      if aug_curves      else None,
        "records":                records,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_CONDITIONS = [
    ("mean_accuracy",          "Normal Prompt"),
    ("mean_accuracy_self_cot", "Self-Description CoT"),
    ("mean_accuracy_cot_yaml", "Structured CoT (cot_prompts.yaml)"),
    ("mean_accuracy_aug",      "Aug Vision (upper bound)"),
]


def plot_logit_lens(
    hard_results: list[dict],
    easy_results: list[dict],
    output_path: Path,
    active_conditions: list[str] | None = None,
) -> None:
    """
    Plot layer-wise top-1 accuracy for each condition.
    active_conditions: list of curve_key strings to include; defaults to those
    with data in at least one result.
    """
    hard_colors = ["#d62728", "#e6550d", "#e7969c"]
    easy_colors = ["#2ca02c", "#1f77b4", "#9467bd"]

    all_results = hard_results + easy_results

    # Determine which conditions have data
    if active_conditions is None:
        active_conditions = [
            key for key, _ in _CONDITIONS
            if any(r.get(key) for r in all_results)
        ]

    n_rows = len(active_conditions)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), sharey=True)
    if n_rows == 1:
        axes = [axes]

    def _plot_panel(ax, results, colors, title, curve_key):
        for res, color in zip(results, colors):
            curve = res.get(curve_key)
            if not curve:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            layers = list(range(len(curve)))
            label  = res["dataset"].replace("_classification", "").replace("_", " ")
            ax.plot(layers, curve, color=color, linewidth=2, label=label)
            ax.fill_between(layers, curve, alpha=0.08, color=color)
        ax.set_xlabel("Transformer Layer", fontsize=10)
        ax.set_ylabel("Top-1 accuracy", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=1)

    for row_idx, (curve_key, cond_label) in enumerate(
        (k, l) for k, l in _CONDITIONS if k in active_conditions
    ):
        _plot_panel(axes[row_idx][0], hard_results, hard_colors,
                    f"Hard Datasets — {cond_label}", curve_key)
        _plot_panel(axes[row_idx][1], easy_results, easy_colors,
                    f"Easy Datasets — {cond_label}", curve_key)

    fig.suptitle(
        "Logit Lens: Layer-wise Top-1 Accuracy Across LLM Depth\n"
        "(generated-position hidden states → final_norm → lm_head  |  softcap=30)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens across LLM depth for agricultural VLM"
    )
    parser.add_argument("--model",           default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output_dir",      default="outputs/logit_lens/")
    parser.add_argument("--n_samples",       type=int, default=8,
                        help="Images per class")
    parser.add_argument("--hard_datasets",   nargs="+", default=HARD_DATASETS)
    parser.add_argument("--easy_datasets",   nargs="+", default=EASY_DATASETS)
    parser.add_argument("--use_4bit",        action="store_true")
    parser.add_argument("--run_cot",         action="store_true",
                        help="Also run self-description CoT and cot_prompts.yaml CoT conditions")
    parser.add_argument("--descriptions_jsonl", default="dataset/agml_descriptions.jsonl",
                        help="Path to agml_descriptions.jsonl — same synthesized descriptions "
                             "used by language assessment cls. Enables aug_vision condition.")
    parser.add_argument("--no_aug",            action="store_true",
                        help="Skip aug_vision condition.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desc_map = None
    jsonl_path = Path(args.descriptions_jsonl)
    if not args.no_aug and jsonl_path.exists():
        desc_map = load_synthesized_descriptions(jsonl_path)
    elif not args.no_aug:
        print(f"[logit_lens] {jsonl_path} not found — skipping aug_vision condition")

    model, processor = load_model(args.model, args.use_4bit)
    _p     = next(model.parameters())
    device = _p.device if str(_p.device) != "meta" else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[logit_lens] input device: {device}")
    final_norm, lm_head, _, softcap = discover_lm_components(model)

    hard_results, easy_results = [], []

    print("\n=== Hard datasets ===")
    for ds in args.hard_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head, softcap,
                          args.n_samples, device,
                          desc_map=desc_map, run_cot=args.run_cot)
        if res:
            hard_results.append(res)

    print("\n=== Easy datasets ===")
    for ds in args.easy_datasets:
        res = run_dataset(ds, model, processor, final_norm, lm_head, softcap,
                          args.n_samples, device,
                          desc_map=desc_map, run_cot=args.run_cot)
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
