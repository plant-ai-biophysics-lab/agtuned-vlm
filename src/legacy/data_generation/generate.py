"""
Stage 1a: Generate synthetic VQA datasets from AgML image classification benchmarks.

Produces two complementary files:
  agml_descriptions.jsonl   — visual description Q&A (open-ended, ROUGE-scored)
  agml_classification.jsonl — visual classification Q&A (short label, exact-match)

A 3-pass pipeline is applied per class:
  Pass 1 — vision: N images → raw per-image captions
  Pass 2 — synthesis: N captions → 2-3 sentence consensus description
  Pass 3 — compression: synthesized description → 1 concise sentence

Checkpointing is done per dataset/class so runs are safely resumable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import random
import time
from collections import defaultdict

from PIL import Image as PILImage
from vllm import LLM, SamplingParams

from src.common.dataset_config import active_datasets, parse_class_label
from src.common.io import append_jsonl, load_prompts

CAPTION_PARAMS = SamplingParams(temperature=0.2, max_tokens=256, repetition_penalty=1.1)
SYNTHESIS_PARAMS = SamplingParams(temperature=0.0, max_tokens=384, repetition_penalty=1.05)
COMPRESSION_PARAMS = SamplingParams(temperature=0.0, max_tokens=96, repetition_penalty=1.05)

DISEASE_TASK_TYPES = {"disease", "pest_damage"}

import re
_REPEAT_RE = re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _class_slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def _clean_response(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    think_start = text.lower().find("<think>")
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


def load_agml_images(agml_name: str, n_samples: int, seed: int = 42) -> dict[str, list[str]]:
    import agml
    loader = agml.data.AgMLDataLoader(agml_name)
    contents = loader.export_contents()
    classes = loader.classes

    by_class: dict[str, list[str]] = defaultdict(list)
    for path, label_idx in contents.items():
        by_class[classes[int(label_idx)]].append(str(path))

    rng = random.Random(seed)
    return {
        cls: rng.sample(paths, min(n_samples, len(paths)))
        for cls, paths in by_class.items()
    }


def build_multimodal_prompt(tokenizer, image: PILImage.Image, question: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text, "multi_modal_data": {"image": image}}


def build_questions(task_type: str, crop: str, cls: str, prompts: dict[str, str]) -> dict[str, str]:
    """Build question strings for all prompt roles from the prompts dict."""
    if task_type in DISEASE_TASK_TYPES:
        return {
            "visual_desc": prompts["visual_description_disease"].format(cls=cls, crop=crop),
            "text_desc": prompts["text_description_disease"].format(cls=cls, crop=crop),
            "visual_cls": prompts["visual_classification_disease"].format(crop=crop),
        }
    return {
        "visual_desc": prompts["visual_description_weed"].format(species=cls),
        "text_desc": prompts["text_description_weed"].format(species=cls),
        "visual_cls": prompts["visual_classification_weed"],
    }


def build_text_cls_question(task_type: str, crop: str, description: str, prompts: dict[str, str]) -> str:
    """Build the text-only classification question from prompts dict."""
    if task_type in DISEASE_TASK_TYPES:
        return prompts["text_classification_disease"].format(crop=crop, description=description)
    return prompts["text_classification_weed"].format(description=description)


def load_llm(model_id: str) -> tuple[LLM, object]:
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def run_vision_pass(
    llm: LLM,
    tokenizer,
    images: list[PILImage.Image],
    question: str,
    no_label_suffix: str,
) -> list[str]:
    prompts = [
        build_multimodal_prompt(tokenizer, img, question + no_label_suffix)
        for img in images
    ]
    outputs = llm.generate(prompts, sampling_params=CAPTION_PARAMS)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def run_synthesis_pass(
    llm: LLM,
    tokenizer,
    raw_captions: list[str],
    cls: str,
    crop: str,
    prompts: dict[str, str],
) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(raw_captions))
    context = f"{cls} on {crop}" if crop != "plant" else cls
    synthesis_prompt = prompts["synthesis"].format(
        n=len(raw_captions),
        context=context,
        numbered_list=numbered,
    )
    prompt = _apply_chat_template(tokenizer, synthesis_prompt)
    outputs = llm.generate([prompt], sampling_params=SYNTHESIS_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


def run_compression_pass(
    llm: LLM,
    tokenizer,
    synthesized: str,
    prompts: dict[str, str],
) -> str:
    compression_prompt = prompts["compression"].format(description=synthesized)
    prompt = _apply_chat_template(tokenizer, compression_prompt)
    outputs = llm.generate([prompt], sampling_params=COMPRESSION_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


def load_checkpoint(output_dir: Path) -> dict:
    path = output_dir / "checkpoint.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_checkpoint(output_dir: Path, checkpoint: dict) -> None:
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))


def make_description_record(
    text_desc_q: str,
    visual_desc_q: str,
    visual_cls_q: str,
    synthesized: str,
    compressed: str,
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
            {"role": "user", "content": text_desc_q},
            {"role": "assistant", "content": compressed},
        ],
        "metadata": {
            "record_type": "description",
            "agml_dataset": agml_name,
            "crop": crop,
            "class_label": clean_cls,
            "task_type": task_type,
            "visual_question": visual_desc_q,
            "visual_cls_question": visual_cls_q,
            "synthesized_description": synthesized,
            "compressed_description": compressed,
            "raw_captions": raw_captions,
            "image_names": [Path(p).name for p in image_paths],
            "n_images_used": len(image_paths),
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
    synthesized_description: str,
    compressed_description: str,
) -> dict:
    return {
        "messages": [
            {"role": "user", "content": text_cls_q},
            {"role": "assistant", "content": clean_cls},
        ],
        "metadata": {
            "record_type": "classification",
            "agml_dataset": agml_name,
            "crop": crop,
            "class_label": clean_cls,
            "task_type": task_type,
            "synthesized_description": synthesized_description,
            "compressed_description": compressed_description,
            "generator_model": model_id,
        },
    }


def push_to_hub(jsonl_path: Path, hub_repo: str) -> None:
    from datasets import Dataset

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            meta = r.get("metadata", {})
            records.append({
                "question": r["messages"][0]["content"],
                "answer": r["messages"][1]["content"],
                "agml_dataset": meta.get("agml_dataset"),
                "crop": meta.get("crop"),
                "class_label": meta.get("class_label"),
                "task_type": meta.get("task_type"),
                "record_type": meta.get("record_type"),
                "generator_model": meta.get("generator_model"),
            })

    ds = Dataset.from_list(records)
    ds.push_to_hub(hub_repo)
    print(f"  Pushed {len(records)} records → {hub_repo}")


def backfill_images(output_dir: Path) -> None:
    from tqdm import tqdm

    checkpoint = load_checkpoint(output_dir)
    entries = [
        (agml_name, meta.get("clean_cls", class_key), meta.get("image_paths", []))
        for agml_name, classes in checkpoint.items()
        for class_key, meta in classes.items()
        if meta.get("status") == "done"
    ]

    saved = skipped = missing = 0
    for agml_name, clean_cls, image_paths in tqdm(entries, desc="Backfilling images", unit="class"):
        if not image_paths:
            missing += 1
            continue
        sample_dir = output_dir / "sample_images" / agml_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        dest = sample_dir / f"{_class_slug(clean_cls)}.jpg"
        if dest.exists():
            skipped += 1
            continue
        try:
            PILImage.open(image_paths[0]).convert("RGB").save(dest, "JPEG", quality=85)
            saved += 1
        except Exception as e:
            tqdm.write(f"  [warn] {agml_name}/{clean_cls}: {e}")
            missing += 1
    print(f"Backfill done — saved={saved}  skipped={skipped}  missing={missing}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backfill_images:
        backfill_images(output_dir)
        return

    desc_path = output_dir / "agml_descriptions.jsonl"
    cls_path = output_dir / "agml_classification.jsonl"

    prompts = load_prompts(Path(args.prompts))
    checkpoint = load_checkpoint(output_dir)
    datasets_cfg = active_datasets(Path(args.config))

    print(f"Datasets: {len(datasets_cfg)} active entries from {args.config}")
    print(f"Model:    {args.model}")
    print(f"Samples:  {args.n_samples_per_class} per class\n")

    llm, tokenizer = load_llm(args.model)

    total_written = 0
    t_start = time.perf_counter()

    for ds_cfg in datasets_cfg:
        agml_name = ds_cfg.agml_name
        task_type = ds_cfg.task_type
        dataset_crop = ds_cfg.crop

        print(f"\n=== {agml_name} [{task_type}] ===")
        try:
            sampled = load_agml_images(agml_name, args.n_samples_per_class, args.seed)
        except Exception as e:
            print(f"  [error] Could not load {agml_name}: {e}")
            continue

        checkpoint.setdefault(agml_name, {})

        for raw_label, image_paths in sampled.items():
            crop, clean_cls = parse_class_label(raw_label, dataset_crop)
            class_key = (
                f"{clean_cls} ({crop})"
                if ("___" in raw_label or dataset_crop == "multiple")
                else clean_cls
            )

            if not args.force and checkpoint[agml_name].get(class_key, {}).get("status") == "done":
                print(f"  [skip] {class_key}")
                continue

            print(f"  {class_key}  ({len(image_paths)} images)")

            questions = build_questions(task_type, crop, clean_cls, prompts)
            images = [PILImage.open(p).convert("RGB") for p in image_paths]

            sample_dir = output_dir / "sample_images" / agml_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"{_class_slug(clean_cls)}.jpg"
            if not sample_path.exists():
                images[0].save(sample_path, "JPEG", quality=85)

            raw_captions = run_vision_pass(
                llm, tokenizer, images, questions["visual_desc"], prompts["no_label_suffix"]
            )
            synthesized = run_synthesis_pass(llm, tokenizer, raw_captions, clean_cls, crop, prompts)
            compressed = run_compression_pass(llm, tokenizer, synthesized, prompts)
            text_cls_q = build_text_cls_question(task_type, crop, synthesized, prompts)

            append_jsonl(desc_path, make_description_record(
                text_desc_q=questions["text_desc"],
                visual_desc_q=questions["visual_desc"],
                visual_cls_q=questions["visual_cls"],
                synthesized=synthesized,
                compressed=compressed,
                raw_captions=raw_captions,
                image_paths=image_paths,
                agml_name=agml_name,
                crop=crop,
                clean_cls=clean_cls,
                task_type=task_type,
                model_id=args.model,
            ))
            append_jsonl(cls_path, make_classification_record(
                text_cls_q=text_cls_q,
                clean_cls=clean_cls,
                agml_name=agml_name,
                crop=crop,
                task_type=task_type,
                model_id=args.model,
                synthesized_description=synthesized,
                compressed_description=compressed,
            ))

            checkpoint[agml_name][class_key] = {
                "status": "done",
                "crop": crop,
                "clean_cls": clean_cls,
                "task_type": task_type,
                "synthesized_description": synthesized,
                "compressed_description": compressed,
                "raw_captions": raw_captions,
                "image_paths": image_paths,
                "visual_desc_question": questions["visual_desc"],
                "visual_cls_question": questions["visual_cls"],
                "text_desc_question": questions["text_desc"],
                "text_cls_question": text_cls_q,
            }
            save_checkpoint(output_dir, checkpoint)
            total_written += 1
            print(f"    → {synthesized[:80]}...")
            print(f"    ↓ {compressed}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {total_written} classes written in {elapsed:.1f}s")
    print(f"  Descriptions:   {desc_path}")
    print(f"  Classification: {cls_path}")

    if args.hub_repo_descriptions:
        print(f"\nPushing text descriptions → {args.hub_repo_descriptions}")
        push_to_hub(desc_path, args.hub_repo_descriptions)

    if args.hub_repo_classification:
        print(f"\nPushing text classification → {args.hub_repo_classification}")
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
        "--config", default="src/config/datasets.yaml",
        help="Path to datasets.yaml registry",
    )
    parser.add_argument(
        "--prompts", default="src/config/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument(
        "--n_samples_per_class", type=int, default=10,
        help="Number of images to sample per class (default: 10)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/data_generation/",
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
    parser.add_argument(
        "--backfill_images", action="store_true",
        help="Save sample images from checkpoint without rerunning inference",
    )
    args = parser.parse_args()
    main(args)
