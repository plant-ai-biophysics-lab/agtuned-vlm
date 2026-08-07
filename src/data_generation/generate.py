"""
Stage 1b: Synthesize grounded classification VQA from build_reference_db.py's fetched sources.

Produces agml_classification.jsonl — visual classification Q&A (short label,
exact-match), where the question embeds a grounded, synthesized description
of the class. There is no separate agml_descriptions.jsonl / open-ended
description-writing output — that task and its consumers
(desc_augmented_vision.py, the description half of language/vision
assessment) are being revisited separately; this script only produces what
the classification task needs.

Two grounding sources per class, combined:
  1. Web-source synthesis — reads reference_db.json (Wikipedia + live web
     search excerpts fetched by build_reference_db.py, which must be run
     first) and synthesizes a description purely from that fetched text.
  2. Vision-based synthesis — samples up to --n_vision_samples images for
     the class from the HF manifest, runs the same multimodal model as a
     captioner (one caption per image, then a consensus pass across them).

If both exist, a final pass merges them into one description. If only one
exists (most commonly: web source came up empty and vision fills in behind
it), that one is used as-is, no merge pass needed. Every stage's output is
kept in the record's metadata (web_description / vision_description /
synthesized_description) — not just the final merged text — so you can see
which source actually produced what.

A class with nothing from either source still gets a record, but with an
empty description and grounding_status="no_source_found" — meant to be
filled in by hand later, not silently dropped or backfilled with a guess.

Checkpointing is done per dataset/class so runs are safely resumable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import random
import re
import time
from collections import defaultdict

from PIL import Image as PILImage
from vllm import LLM, SamplingParams

from src.common.dataset_config import load_manifest, load_hf_samples
from src.common.io import append_jsonl, load_prompts
from src.common.progress import ProgressBar

CAPTION_PARAMS = SamplingParams(temperature=0.2, max_tokens=256, repetition_penalty=1.1)
SYNTHESIS_PARAMS = SamplingParams(temperature=0.0, max_tokens=384, repetition_penalty=1.05)
MERGE_PARAMS = SamplingParams(temperature=0.0, max_tokens=384, repetition_penalty=1.05)

DISEASE_TASK_TYPES = {"disease", "pest_damage"}
NOT_ENOUGH_DETAIL_SENTINEL = "NOT_ENOUGH_VISUAL_DETAIL"

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


def load_hf_images_by_class(
    agml_name: str, manifest: dict, images_root: Path, n_samples: int, seed: int = 42,
) -> dict[str, list[str]]:
    """Grouped by class (raw manifest label), capped at n_samples/class with a
    fixed seed, sourced from the HF-based manifest."""
    samples = load_hf_samples(agml_name, manifest, images_root)
    by_class: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(s["path"])

    rng = random.Random(seed)
    return {
        cls: rng.sample(paths, min(n_samples, len(paths)))
        for cls, paths in by_class.items()
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
        limit_mm_per_prompt={"image": 1},  # this model doubles as the vision
                                            # captioner (see run_vision_caption_pass)
                                            # — text-only prompts elsewhere in
                                            # this file are unaffected, vLLM only
                                            # engages the image path when a
                                            # prompt actually carries multi_modal_data
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def run_synthesis_from_sources_pass(
    llm: LLM,
    tokenizer,
    sources: list[dict],
    cls: str,
    crop: str,
    prompts: dict[str, str],
) -> str:
    numbered = "\n".join(
        f"{i + 1}. [{s.get('organization', 'source')}] {s['excerpt']}"
        for i, s in enumerate(sources)
    )
    synthesis_prompt = prompts["synthesis_from_sources"].format(
        n=len(sources),
        crop=crop,
        class_label=cls,
        numbered_sources=numbered,
    )
    prompt = _apply_chat_template(tokenizer, synthesis_prompt)
    outputs = llm.generate([prompt], sampling_params=SYNTHESIS_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


def run_vision_caption_pass(
    llm: LLM,
    tokenizer,
    images: list[PILImage.Image],
    cls: str,
    crop: str,
    prompts: dict[str, str],
) -> list[str]:
    """One caption per image, batched in a single llm.generate() call."""
    question = prompts["vision_caption_question"].format(cls=cls, crop=crop)
    question += prompts["vision_caption_no_label_suffix"]
    mm_prompts = [build_multimodal_prompt(tokenizer, img, question) for img in images]
    outputs = llm.generate(mm_prompts, sampling_params=CAPTION_PARAMS)
    return [_clean_response(o.outputs[0].text) for o in outputs]


def run_vision_synthesis_pass(
    llm: LLM,
    tokenizer,
    raw_captions: list[str],
    cls: str,
    crop: str,
    prompts: dict[str, str],
) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(raw_captions))
    synthesis_prompt = prompts["vision_synthesis_from_captions"].format(
        n=len(raw_captions),
        crop=crop,
        class_label=cls,
        numbered_captions=numbered,
    )
    prompt = _apply_chat_template(tokenizer, synthesis_prompt)
    outputs = llm.generate([prompt], sampling_params=SYNTHESIS_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


def run_merge_pass(
    llm: LLM,
    tokenizer,
    web_description: str,
    vision_description: str,
    cls: str,
    crop: str,
    prompts: dict[str, str],
) -> str:
    merge_prompt = prompts["merge_web_and_vision_description"].format(
        crop=crop,
        class_label=cls,
        web_description=web_description,
        vision_description=vision_description,
    )
    prompt = _apply_chat_template(tokenizer, merge_prompt)
    outputs = llm.generate([prompt], sampling_params=MERGE_PARAMS)
    return _clean_response(outputs[0].outputs[0].text)


def load_checkpoint(output_dir: Path) -> dict:
    path = output_dir / "checkpoint.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_checkpoint(output_dir: Path, checkpoint: dict) -> None:
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))


def make_classification_record(
    text_cls_q: str,
    clean_cls: str,
    agml_name: str,
    crop: str,
    task_type: str,
    model_id: str,
    final_description: str,
    web_description: str,
    vision_description: str,
    grounding_status: str,
    sources: list[dict],
    image_paths: list[str],
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
            "synthesized_description": final_description,
            "web_description": web_description,
            "vision_description": vision_description,
            "grounding_status": grounding_status,
            "source_urls": [s["url"] for s in sources],
            "image_names": [Path(p).name for p in image_paths],
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
                "grounding_status": meta.get("grounding_status"),
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

    cls_path = output_dir / "agml_classification.jsonl"

    prompts = load_prompts(Path(args.prompts))
    checkpoint = load_checkpoint(output_dir)
    manifest = load_manifest(Path(args.manifest))
    images_root = Path(args.images_root)

    reference_db = json.loads(Path(args.reference_db).read_text(encoding="utf-8"))
    entries = reference_db.get("entries", {})

    entries_by_dataset: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, meta in entries.items():
        entries_by_dataset[meta["agml_dataset"]].append((key, meta))

    n_classes = sum(len(v) for v in entries_by_dataset.values())
    print(f"Reference DB: {args.reference_db}  ({n_classes} classes across "
          f"{len(entries_by_dataset)} datasets)")
    print(f"Model: {args.model}  (also used as the vision captioner)")
    print(f"Vision samples/class: {args.n_vision_samples}\n")

    llm, tokenizer = load_llm(args.model)

    total_written = 0
    t_start = time.perf_counter()

    for agml_name, class_entries in ProgressBar(list(entries_by_dataset.items()), desc="Datasets"):
        print(f"\n=== {agml_name} ===")
        try:
            images_by_class = load_hf_images_by_class(
                agml_name, manifest, images_root, n_samples=args.n_vision_samples, seed=args.seed
            )
        except Exception as e:
            print(f"  [error] Could not load images for {agml_name}: {e}")
            images_by_class = {}

        checkpoint.setdefault(agml_name, {})

        for key, meta in ProgressBar(class_entries, desc=f"  {agml_name}"):
            crop = meta["crop"]
            clean_cls = meta["class_label"]
            task_type = meta["task_type"]
            raw_label = meta.get("raw_label", clean_cls)
            ref_status = meta.get("status", "no_source_found")
            sources = meta.get("sources", [])
            is_healthy_class = ref_status == "healthy_skip"

            if not args.force and checkpoint[agml_name].get(key, {}).get("status") == "done":
                print(f"  [skip] {clean_cls}")
                continue

            image_paths = images_by_class.get(raw_label, [])
            if image_paths:
                sample_dir = output_dir / "sample_images" / agml_name
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_path = sample_dir / f"{_class_slug(clean_cls)}.jpg"
                if not sample_path.exists():
                    try:
                        PILImage.open(image_paths[0]).convert("RGB").save(sample_path, "JPEG", quality=85)
                    except Exception as e:
                        print(f"  [warn] Could not save sample image for {clean_cls}: {e}")

            # ── Stage A: web-source synthesis ───────────────────────────────
            web_description = ""
            if not is_healthy_class and ref_status == "found" and sources:
                web_description = run_synthesis_from_sources_pass(
                    llm, tokenizer, sources, clean_cls, crop, prompts
                )
                if web_description.strip() == NOT_ENOUGH_DETAIL_SENTINEL:
                    web_description = ""

            # ── Stage B: vision-based synthesis (fallback / supplement) ─────
            vision_description = ""
            if not is_healthy_class and image_paths:
                try:
                    images = [PILImage.open(p).convert("RGB") for p in image_paths]
                    raw_captions = run_vision_caption_pass(llm, tokenizer, images, clean_cls, crop, prompts)
                    vision_description = run_vision_synthesis_pass(
                        llm, tokenizer, raw_captions, clean_cls, crop, prompts
                    )
                except Exception as e:
                    print(f"  [warn] Vision captioning failed for {clean_cls}: {e}")

            # ── Stage C: merge (only when both stages produced something) ───
            if is_healthy_class:
                final_description = ""
                grounding_status = "healthy_skip"
            elif web_description and vision_description:
                final_description = run_merge_pass(
                    llm, tokenizer, web_description, vision_description, clean_cls, crop, prompts
                )
                grounding_status = "grounded_web_and_vision"
            elif vision_description:
                final_description = vision_description
                grounding_status = "grounded_vision_only"
            elif web_description:
                final_description = web_description
                grounding_status = "grounded_web_only"
            else:
                final_description = ""
                grounding_status = "no_source_found"

            print(f"  {clean_cls}  [{grounding_status}]  "
                  f"({len(sources)} sources, {len(image_paths)} images)")

            text_cls_q = build_text_cls_question(task_type, crop, final_description, prompts)

            append_jsonl(cls_path, make_classification_record(
                text_cls_q=text_cls_q,
                clean_cls=clean_cls,
                agml_name=agml_name,
                crop=crop,
                task_type=task_type,
                model_id=args.model,
                final_description=final_description,
                web_description=web_description,
                vision_description=vision_description,
                grounding_status=grounding_status,
                sources=sources,
                image_paths=image_paths,
            ))

            checkpoint[agml_name][key] = {
                "status": "done",
                "crop": crop,
                "clean_cls": clean_cls,
                "task_type": task_type,
                "grounding_status": grounding_status,
                "synthesized_description": final_description,
                "web_description": web_description,
                "vision_description": vision_description,
                "image_paths": image_paths,
                "text_cls_question": text_cls_q,
            }
            save_checkpoint(output_dir, checkpoint)
            total_written += 1
            if final_description:
                print(f"    → {final_description[:80]}...")
            else:
                print(f"    → (empty — needs manual fill)")

    elapsed = time.perf_counter() - t_start
    by_status: dict[str, int] = defaultdict(int)
    for classes in checkpoint.values():
        for m in classes.values():
            by_status[m.get("grounding_status", "unknown")] += 1
    print(f"\n{'='*60}")
    print(f"Done: {total_written} classes written in {elapsed:.1f}s")
    for status, n in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {status}: {n}")
    print(f"  Classification: {cls_path}")

    if args.hub_repo_classification:
        print(f"\nPushing text classification → {args.hub_repo_classification}")
        push_to_hub(cls_path, args.hub_repo_classification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize grounded VQA datasets from build_reference_db.py's fetched "
                     "sources, backed up by vision-based captioning when web sources are thin."
    )
    parser.add_argument(
        "--model", default="google/gemma-4-26B-A4B-it",
        help="vLLM-compatible multimodal model (HuggingFace ID or local path) — used for "
             "both the text-only synthesis/merge passes and the vision captioning pass",
    )
    parser.add_argument(
        "--reference_db", default="outputs/data_generation/reference_db.json",
        help="Path to reference_db.json produced by build_reference_db.py (run that first)",
    )
    parser.add_argument(
        "--prompts", default="src/config/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument(
        "--manifest", default="dataset/selected_images.json",
        help="Path to the HF download manifest (from download-datasets)",
    )
    parser.add_argument(
        "--images_root", default="dataset/images",
        help="Local cache directory for materialized images from the manifest",
    )
    parser.add_argument(
        "--n_vision_samples", type=int, default=5,
        help="Images per class to run through the vision captioning pass (default: 5)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/data_generation/",
        help="Directory for JSONL outputs, checkpoint, and logs",
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
