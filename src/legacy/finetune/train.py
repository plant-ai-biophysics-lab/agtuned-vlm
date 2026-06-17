"""
Fine-tune a vision-language model on the Digital Green agricultural dataset
using unsloth LoRA.

Three training component modes:
  vision          — vis_cls + vis_desc tasks; trains vision layers only
  language        — lang_cls + lang_desc tasks; trains language layers only
  vision_language — all 4 tasks; trains both layers

The 4 task configurations mirror the language and vision assessment prompts
so that training signal is consistent with evaluation:
  vis_cls   image + crop  →  diagnosis
  vis_desc  image + crop  →  details (description)
  lang_cls  details + crop  →  diagnosis
  lang_desc diagnosis + crop  →  details (description)

Images are downloaded from S3 URLs on first run and cached locally.

Usage:
    python main.py finetune \
        --model          unsloth/gemma-4-E2B-it \
        --annotations    /path/to/annotations.csv \
        --images_dir     data/digital_green/images/ \
        --train_components vision_language \
        --output_dir     outputs/finetune/
"""

import argparse
import csv
import gc
import json
import os
import re
import random
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Text cleaning ──────────────────────────────────────────────────────────────

def _clean_details(text: str) -> str:
    """Collapse newlines/tabs/bullets into a single continuous paragraph."""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[•\-\*]\s+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _clean_diagnosis(text: str) -> str:
    """Replace semicolons with 'and' for multi-label diagnoses."""
    parts = [p.strip() for p in text.split(';')]
    return ' and '.join(p for p in parts if p)


# ── Image downloading ──────────────────────────────────────────────────────────

def _image_filename(url: str) -> str:
    """Derive a stable local filename from the URL's last path component."""
    return Path(url).name


def download_images(rows: list[dict], images_dir: Path) -> dict[str, Path]:
    """
    Download all images into images_dir. Skips existing files.
    Returns {url: local_path}.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    url_to_path: dict[str, Path] = {}
    missing = [(r["image_url"], images_dir / _image_filename(r["image_url"]))
               for r in rows if r["image_url"]]
    to_fetch = [(url, p) for url, p in missing if not p.exists()]

    if to_fetch:
        print(f"Downloading {len(to_fetch)} images to {images_dir} ...")
    for i, (url, path) in enumerate(to_fetch, 1):
        try:
            urllib.request.urlretrieve(url, path)
            if i % 50 == 0:
                print(f"  {i}/{len(to_fetch)}")
        except Exception as e:
            print(f"  WARNING: failed to download {url}: {e}")

    for url, path in missing:
        url_to_path[url] = path

    n_ok = sum(1 for p in url_to_path.values() if p.exists())
    print(f"Images ready: {n_ok}/{len(url_to_path)}")
    return url_to_path


# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_prompts(prompts_yaml: Path) -> dict:
    import yaml
    with open(prompts_yaml) as f:
        return yaml.safe_load(f)


# ── Sample builders ────────────────────────────────────────────────────────────

def _make_vision_sample(image: PILImage.Image, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def _make_text_sample(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def build_samples(
    rows: list[dict],
    url_to_path: dict[str, Path],
    prompts: dict,
    tasks: list[str],
) -> list[dict]:
    """
    Build training samples for the requested task subset.
    tasks is a subset of: vis_cls, vis_desc, lang_cls, lang_desc
    """
    vis_cls_q   = prompts["visual_classification_disease"]
    vis_desc_q  = prompts["visual_description_disease"]
    lang_cls_q  = prompts["text_classification_disease"]
    lang_desc_q = prompts["text_description_disease"]
    desc_wrap   = prompts["assessment_descriptive"]
    cls_suffix  = prompts["classification_suffix"]

    samples = []
    skipped = 0

    for row in rows:
        crop      = row["crop"].strip()
        diagnosis = _clean_diagnosis(row["diagnosis"])
        details   = _clean_details(row["details"])

        if not crop or not diagnosis or not details:
            skipped += 1
            continue

        img_path = url_to_path.get(row["image_url"])
        img_ok   = img_path and img_path.exists()

        # Vision tasks — require a loadable image
        if img_ok and ("vis_cls" in tasks or "vis_desc" in tasks):
            try:
                image = PILImage.open(img_path).convert("RGB")
            except Exception:
                img_ok = False

        if img_ok:
            if "vis_cls" in tasks:
                q = vis_cls_q.format(crop=crop) + cls_suffix
                samples.append(_make_vision_sample(image, q, diagnosis))

            if "vis_desc" in tasks:
                q = desc_wrap.format(question=vis_desc_q.format(crop=crop))
                samples.append(_make_vision_sample(image, q, details))

        # Language tasks — text only, no image required
        if "lang_cls" in tasks:
            q = lang_cls_q.format(crop=crop, description=details) + cls_suffix
            samples.append(_make_text_sample(q, diagnosis))

        if "lang_desc" in tasks:
            q = desc_wrap.format(question=lang_desc_q.format(cls=diagnosis, crop=crop))
            samples.append(_make_text_sample(q, details))

    if skipped:
        print(f"  Skipped {skipped} rows with missing crop/diagnosis/details.")

    # Print one sample per task for debugging
    shown: set[str] = set()
    print("\n=== Sample prompts ===")
    for task in tasks:
        for s in samples:
            content = s["messages"][0]["content"]
            has_image = any(c.get("type") == "image" for c in content)
            task_match = (
                (task in ("vis_cls", "vis_desc") and has_image) or
                (task in ("lang_cls", "lang_desc") and not has_image)
            )
            if task_match and task not in shown:
                user_text = next(c["text"] for c in content if c["type"] == "text")
                answer    = s["messages"][1]["content"][0]["text"]
                print(f"\n[{task}]")
                print(f"  input : {user_text[:200]}{'...' if len(user_text) > 200 else ''}")
                print(f"  output: {answer[:120]}{'...' if len(answer) > 120 else ''}")
                shown.add(task)
        if task not in shown:
            print(f"\n[{task}] — no samples (images missing?)")
    print()

    return samples


_ALL_TASKS = ["vis_cls", "vis_desc", "lang_cls", "lang_desc"]


# ── Dataset split ──────────────────────────────────────────────────────────────

def split_dataset(data: list, train: float = 0.9):
    shuffled = data.copy()
    random.shuffle(shuffled)
    t = int(len(shuffled) * train)
    return shuffled[:t], shuffled[t:]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a VLM on the Digital Green dataset with unsloth LoRA."
    )
    parser.add_argument("--model",            default="unsloth/gemma-4-E2B-it",
                        help="Unsloth-compatible model ID (HuggingFace or unsloth/ prefix)")
    parser.add_argument("--annotations",      default="data/digital_green/annotations.csv",
                        help="Path to annotations.csv")
    parser.add_argument("--images_dir",       default="data/digital_green/images/",
                        help="Directory to cache downloaded images")
    parser.add_argument("--prompts",          default="src/config/prompts.yaml",
                        help="Path to prompts.yaml")
    parser.add_argument("--train_layers", default="vision_language",
                        choices=["vision", "language", "vision_language"],
                        help="Which model layers receive gradient updates: "
                             "vision, language, or vision_language (both)")
    parser.add_argument("--tasks", nargs="+",
                        choices=_ALL_TASKS, default=_ALL_TASKS,
                        help="Prompt configurations to train on. "
                             f"Any subset of: {_ALL_TASKS}. Default: all four.")
    parser.add_argument("--output_dir",       default="outputs/finetune/",
                        help="Where to save adapters, merged model, and stats")
    parser.add_argument("--seed",             type=int, default=3407)
    parser.add_argument("--epochs",           type=int, default=1)
    parser.add_argument("--batch_size",       type=int, default=4)
    parser.add_argument("--grad_accum",       type=int, default=4)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--lora_r",           type=int, default=16)
    parser.add_argument("--max_length",       type=int, default=2048)
    parser.add_argument("--load_in_4bit",     action="store_true",
                        help="Load model in 4-bit (BnB). Default: bf16.")
    args = parser.parse_args()

    random.seed(args.seed)
    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) / args.train_layers / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    try:
        from unsloth import FastVisionModel, is_bf16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"ERROR: unsloth is not installed: {e}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    try:
        model, processor = FastVisionModel.from_pretrained(
            args.model,
            load_in_4bit=args.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
    except Exception as e:
        print(f"ERROR: could not load model '{args.model}' via unsloth: {e}")
        sys.exit(1)

    # ── Apply LoRA ──────────────────────────────────────────────────────────────
    train_vision_layers   = args.train_layers in ("vision", "vision_language")
    train_language_layers = args.train_layers in ("language", "vision_language")

    print(f"Train layers : {args.train_layers}")
    print(f"  vision layers  : {train_vision_layers}")
    print(f"  language layers: {train_language_layers}")
    print(f"Tasks        : {args.tasks}")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=train_vision_layers,
        finetune_language_layers=train_language_layers,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
    )

    # ── Load data ───────────────────────────────────────────────────────────────
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        print(f"ERROR: annotations file not found: {annotations_path}")
        sys.exit(1)

    with open(annotations_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {annotations_path}")

    url_to_path = download_images(rows, Path(args.images_dir))
    prompts = load_prompts(Path(args.prompts))

    samples = build_samples(rows, url_to_path, prompts, args.tasks)
    print(f"Built {len(samples)} training samples "
          f"({len(samples) // max(len(rows), 1):.1f}x per row)")

    if not samples:
        print("ERROR: no samples were built — check annotations and image downloads.")
        sys.exit(1)

    train_ds, val_ds = split_dataset(samples)
    print(f"Split: train={len(train_ds)}  val={len(val_ds)}")

    # ── Train ───────────────────────────────────────────────────────────────────
    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    max_mem   = round(gpu_stats.total_memory / 1024 ** 3, 3)

    eval_steps = max(1, int(0.25 * len(train_ds) / (args.batch_size * args.grad_accum)))

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(output_dir),
            report_to="trackio",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
        ),
    )

    trainer_stats = trainer.train()

    # ── GPU stats ───────────────────────────────────────────────────────────────
    used_mem  = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    runtime   = trainer_stats.metrics["train_runtime"]
    gpu_stats_dict = {
        "model":                  args.model,
        "train_layers":           args.train_layers,
        "tasks":                  args.tasks,
        "n_train_samples":        len(train_ds),
        "train_runtime_s":        round(runtime, 2),
        "train_runtime_min":      round(runtime / 60, 2),
        "gpu_name":               gpu_stats.name,
        "gpu_total_memory_gb":    max_mem,
        "peak_reserved_gb":       used_mem,
        "peak_lora_gb":           round(used_mem - start_mem, 3),
    }
    for k, v in gpu_stats_dict.items():
        print(f"  {k}: {v}")

    stats_path = output_dir / "train_stats.json"
    stats_path.write_text(json.dumps(gpu_stats_dict, indent=2))
    print(f"\nStats saved to {stats_path}")

    # ── Save adapters + merged model ────────────────────────────────────────────
    adapters_dir = output_dir / "lora_adapters"
    merged_dir   = output_dir / "vllm_model"

    model.save_pretrained(str(adapters_dir))
    processor.save_pretrained(str(adapters_dir))
    print(f"LoRA adapters saved to {adapters_dir}")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    model.save_pretrained_merged(str(merged_dir), processor, save_method="merged_16bit")
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
