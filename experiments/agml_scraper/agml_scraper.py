"""
AgML Scraper: Synthetic VQA Dataset Generation
===============================================
Generates two complementary datasets from AgML image classification benchmarks:

  agml_descriptions.jsonl   — visual description Q&A (open-ended, ROUGE-scored)
  agml_classification.jsonl — visual classification Q&A (short label, exact-match)

Both files are loadable by text_qa.py --idtools_jsonl for H2 evaluation.
Classification records are derived from description inference at no extra cost.

Usage:
    uv run python agml_scraper.py
    uv run python agml_scraper.py --n_samples_per_class 5 --output_dir outputs/
    uv run python agml_scraper.py --force --hub_repo_descriptions user/agml-desc
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image as PILImage
from vllm import LLM, SamplingParams


# ── Question templates ─────────────────────────────────────────────────────────

VISUAL_DESC_DISEASE = (
    "Can you describe visually the {cls} that is exhibited in this image of {crop}?"
)
TEXT_DESC_DISEASE = "Can you describe visually the {cls} of a {crop}?"
VISUAL_CLS_DISEASE = (
    "What disease, pest, damage type, or other stress, if any, "
    "is exhibited in this image of a {crop}?"
)
TEXT_CLS_DISEASE = (
    "What disease, pest, damage type, or other stress, if any, "
    "is exhibited in this description of a {crop}?\n\nDescription: {description}"
)

VISUAL_DESC_WEED = "Can you describe visually the {species} exhibited in this image?"
TEXT_DESC_WEED = "Can you describe visually the {species}?"
VISUAL_CLS_WEED = "What plant or weed is exhibited in this image?"
TEXT_CLS_WEED = (
    "What plant or weed is exhibited in this description?\n\nDescription: {description}"
)

DESCRIPTIVE_TASK_TYPES    = {"disease", "pest", "stress"}
IDENTIFICATION_TASK_TYPES = {"weed_identification", "crop_identification"}

# ── Sampling params ────────────────────────────────────────────────────────────

CAPTION_PARAMS = SamplingParams(
    temperature=0.2, max_tokens=256, repetition_penalty=1.1
)
SYNTHESIS_PARAMS = SamplingParams(
    temperature=0.0, max_tokens=384, repetition_penalty=1.05
)

_REPEAT_RE = re.compile(r'(\b\w+\b)(\s+\1){4,}', re.IGNORECASE)
_THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


# ── Response cleaning (adapted from text_qa.py) ───────────────────────────────

def _clean_response(text: str) -> str:
    text = _THINK_RE.sub('', text).strip()
    think_start = text.lower().find('<think>')
    if think_start != -1:
        text = text[:think_start].strip()
    bracket = text.find("[Answer")
    if bracket != -1:
        text = text[:bracket].strip()
    m = _REPEAT_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text


def _apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "/no_think"}, *messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ── Data loading ───────────────────────────────────────────────────────────────

def load_datasets_config(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return [d for d in config["datasets"] if not d.get("skip", False)]


def load_agml_images(
    agml_name: str, n_samples: int, seed: int = 42
) -> dict[str, list[str]]:
    import agml
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    classes  = loader.classes

    by_class: dict[str, list[str]] = defaultdict(list)
    for path, label_idx in contents.items():
        by_class[classes[int(label_idx)]].append(str(path))

    rng = random.Random(seed)
    return {
        cls: rng.sample(paths, min(n_samples, len(paths)))
        for cls, paths in by_class.items()
    }


def parse_class_label(raw_label: str, dataset_crop: str | None) -> tuple[str, str]:
    """Returns (crop, clean_class_label).

    Handles plant_village_classification-style 'Tomato___Early_blight' labels by
    splitting on '___'. For all other datasets, uses the dataset-level crop field.
    """
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        crop = parts[0].replace("_", " ").strip()
        cls  = parts[1].replace("_", " ").strip()
        return crop, cls
    crop = dataset_crop if dataset_crop and dataset_crop != "multiple" else "plant"
    return crop, raw_label.replace("_", " ").strip()


# ── Prompt building ────────────────────────────────────────────────────────────

def build_multimodal_prompt(tokenizer, image: PILImage.Image, question: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def build_questions(task_type: str, crop: str, cls: str) -> dict[str, str]:
    if task_type in DESCRIPTIVE_TASK_TYPES:
        return {
            "visual_desc": VISUAL_DESC_DISEASE.format(cls=cls, crop=crop),
            "text_desc":   TEXT_DESC_DISEASE.format(cls=cls, crop=crop),
            "visual_cls":  VISUAL_CLS_DISEASE.format(crop=crop),
        }
    return {
        "visual_desc": VISUAL_DESC_WEED.format(species=cls),
        "text_desc":   TEXT_DESC_WEED.format(species=cls),
        "visual_cls":  VISUAL_CLS_WEED,
    }


def build_text_cls_question(task_type: str, crop: str, description: str) -> str:
    if task_type in DESCRIPTIVE_TASK_TYPES:
        return TEXT_CLS_DISEASE.format(crop=crop, description=description)
    return TEXT_CLS_WEED.format(description=description)


# ── Inference ──────────────────────────────────────────────────────────────────

def load_llm(model_id: str) -> tuple[LLM, object]:
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def run_vision_pass(
    llm: LLM,
    tokenizer,
    images: list[PILImage.Image],
    question: str,
) -> list[str]:
    prompts = [build_multimodal_prompt(tokenizer, img, question) for img in images]
    outputs = llm.generate(prompts, sampling_params=CAPTION_PARAMS)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def run_synthesis_pass(
    llm: LLM,
    tokenizer,
    raw_captions: list[str],
    cls: str,
    crop: str,
) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(raw_captions))
    context  = f"{cls} on {crop}" if crop != "plant" else cls
    synthesis_prompt = (
        f"You are given {len(raw_captions)} visual descriptions of {context}, "
        f"each written from a separate photograph. Synthesize these into one clear, "
        f"representative 2-3 sentence description that captures the consensus visual "
        f"features. Ignore noise or one-off observations.\n\nDescriptions:\n{numbered}"
    )
    prompt  = _apply_chat_template(tokenizer, synthesis_prompt)
    outputs = llm.generate([prompt], sampling_params=SYNTHESIS_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


# ── Checkpointing ──────────────────────────────────────────────────────────────

def load_checkpoint(output_dir: Path) -> dict:
    path = output_dir / "checkpoint.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_checkpoint(output_dir: Path, checkpoint: dict) -> None:
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def make_description_record(
    text_desc_q: str,
    visual_desc_q: str,
    synthesized: str,
    raw_captions: list[str],
    image_paths: list[str],
    agml_name: str,
    crop: str,
    clean_cls: str,
    task_type: str,
    model_id: str,
) -> dict:
    return {
        "messages": [
            {"role": "user",      "content": text_desc_q},
            {"role": "assistant", "content": synthesized},
        ],
        "metadata": {
            "record_type":     "description",
            "agml_dataset":    agml_name,
            "crop":            crop,
            "class_label":     clean_cls,
            "task_type":       task_type,
            "visual_question": visual_desc_q,
            "raw_captions":    raw_captions,
            "n_images_used":   len(image_paths),
            "generator_model": model_id,
        },
    }


def make_classification_record(
    text_cls_q: str,
    clean_cls: str,
    agml_name: str,
    crop: str,
    task_type: str,
    model_id: str,
) -> dict:
    return {
        "messages": [
            {"role": "user",      "content": text_cls_q},
            {"role": "assistant", "content": clean_cls},
        ],
        "metadata": {
            "record_type":     "classification",
            "agml_dataset":    agml_name,
            "crop":            crop,
            "class_label":     clean_cls,
            "task_type":       task_type,
            "generator_model": model_id,
        },
    }


# ── HuggingFace push ───────────────────────────────────────────────────────────

def push_to_hub(jsonl_path: Path, hub_repo: str) -> None:
    from datasets import Dataset

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r    = json.loads(line)
            meta = r.get("metadata", {})
            records.append({
                "question":        r["messages"][0]["content"],
                "answer":          r["messages"][1]["content"],
                "agml_dataset":    meta.get("agml_dataset"),
                "crop":            meta.get("crop"),
                "class_label":     meta.get("class_label"),
                "task_type":       meta.get("task_type"),
                "record_type":     meta.get("record_type"),
                "generator_model": meta.get("generator_model"),
            })

    ds = Dataset.from_list(records)
    ds.push_to_hub(hub_repo)
    print(f"  Pushed {len(records)} records → {hub_repo}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desc_path = output_dir / "agml_descriptions.jsonl"
    cls_path  = output_dir / "agml_classification.jsonl"

    checkpoint      = load_checkpoint(output_dir)
    datasets_config = load_datasets_config(Path(args.datasets_config))

    print(f"Datasets: {len(datasets_config)} active entries from {args.datasets_config}")
    print(f"Model:    {args.model}")
    print(f"Samples:  {args.n_samples_per_class} per class\n")

    llm, tokenizer = load_llm(args.model)

    total_written = 0
    t_start = time.perf_counter()

    for ds_cfg in datasets_config:
        agml_name    = ds_cfg["agml_name"]
        task_type    = ds_cfg["task_type"]
        dataset_crop = ds_cfg.get("crop")

        print(f"\n=== {agml_name} [{task_type}] ===")
        try:
            sampled = load_agml_images(agml_name, args.n_samples_per_class, args.seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        checkpoint.setdefault(agml_name, {})

        for raw_label, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_label, dataset_crop)
            # Unique key disambiguates classes across multi-crop datasets
            class_key = (
                f"{clean_cls} ({crop})"
                if ("___" in raw_label or dataset_crop == "multiple")
                else clean_cls
            )

            if not args.force and checkpoint[agml_name].get(class_key, {}).get("status") == "done":
                print(f"  [skip] {class_key}")
                continue

            print(f"  {class_key}  ({len(image_paths)} images)")

            questions = build_questions(task_type, crop, clean_cls)
            images    = [PILImage.open(p).convert("RGB") for p in image_paths]

            # Pass 1 — per-image captions (vision)
            raw_captions = run_vision_pass(llm, tokenizer, images, questions["visual_desc"])

            # Pass 2 — synthesize to single description (text-only)
            synthesized = run_synthesis_pass(llm, tokenizer, raw_captions, clean_cls, crop)

            text_cls_q = build_text_cls_question(task_type, crop, synthesized)

            _append_jsonl(desc_path, make_description_record(
                text_desc_q=questions["text_desc"],
                visual_desc_q=questions["visual_desc"],
                synthesized=synthesized,
                raw_captions=raw_captions,
                image_paths=image_paths,
                agml_name=agml_name,
                crop=crop,
                clean_cls=clean_cls,
                task_type=task_type,
                model_id=args.model,
            ))
            _append_jsonl(cls_path, make_classification_record(
                text_cls_q=text_cls_q,
                clean_cls=clean_cls,
                agml_name=agml_name,
                crop=crop,
                task_type=task_type,
                model_id=args.model,
            ))

            checkpoint[agml_name][class_key] = {
                "status":                   "done",
                "synthesized_description":  synthesized,
                "raw_captions":             raw_captions,
                "image_paths":              image_paths,
            }
            save_checkpoint(output_dir, checkpoint)
            total_written += 1
            print(f"    → {synthesized[:100]}...")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {total_written} classes written in {elapsed:.1f}s")
    print(f"  Descriptions:   {desc_path}")
    print(f"  Classification: {cls_path}")

    if args.hub_repo_descriptions:
        print(f"\nPushing descriptions → {args.hub_repo_descriptions}")
        push_to_hub(desc_path, args.hub_repo_descriptions)

    if args.hub_repo_classification:
        print(f"\nPushing classification → {args.hub_repo_classification}")
        push_to_hub(cls_path, args.hub_repo_classification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic VQA datasets from AgML image classification benchmarks."
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.6-35B-A3B",
        help="vLLM-compatible VLM (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--datasets_config", default="datasets.yaml",
        help="Path to datasets.yaml registry",
    )
    parser.add_argument(
        "--n_samples_per_class", type=int, default=10,
        help="Number of images to sample per class (default: 10)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/",
        help="Directory for JSONL outputs, checkpoint, and logs",
    )
    parser.add_argument(
        "--hub_repo_descriptions", default=None,
        help="HuggingFace repo for description dataset (e.g. user/agml-descriptions)",
    )
    parser.add_argument(
        "--hub_repo_classification", default=None,
        help="HuggingFace repo for classification dataset (e.g. user/agml-classification)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoint and regenerate all entries",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for image sampling (default: 42)",
    )
    args = parser.parse_args()
    main(args)
