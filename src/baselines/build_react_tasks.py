"""Builds a --tasks_jsonl file for react_baseline.py (src/baselines/
react_baseline.py) from REAL AgML images — the same sampling
(active_datasets/load_hf_samples/sample_class_images) and the same
classification question text (vision_baseline.py's own cls_question()) as
vision-baseline/verifier-cot use, so a react-baseline run is genuinely
comparable to those, not a different task in disguise. This is a separate
script rather than folding image-sampling into react_baseline.py itself —
keeps react_baseline.py's own job single ("run whatever's in this
--tasks_jsonl file"), same separation-of-concerns vision_baseline.py's
generation phase has from judge_leniency.py's own separate analysis phase.

Each output line: {"agml_dataset", "class_label", "image_name",
"image_path", "task"} — "image_path" is new (react_baseline.py's
--tasks_jsonl rows previously only ever had "task" for a text-only ad-hoc
episode; a row WITH "image_path" gets that image actually loaded and
attached to the episode, see react_baseline.py's main() and
react_agent.py's own `image` param).

--match_images_from (same convention and reasoning as verifier_cot.py's
own flag of the same name — see its _load_reference_images docstring):
point this at an existing vision_baseline/desc_aug/self_cot/cot_yaml
checkpoint to replay THAT run's own sampled image names instead of an
independent draw, guaranteeing 100% image-set overlap for a true
apples-to-apples comparison. Without it, --n_samples draws independently
per (dataset, class) via sample_class_images, same as vision_baseline.py's
own default behavior.

Usage:
    uv run python main.py react-build-tasks \\
        --n_samples 3 \\
        --output outputs_final/react_tasks_smoke3.jsonl

    # true apples-to-apples against an existing vision-baseline run:
    uv run python main.py react-build-tasks \\
        --match_images_from outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --n_samples 50 \\
        --output outputs_final/react_tasks_full.jsonl

Then:
    uv run python main.py react-baseline --tasks_jsonl outputs_final/react_tasks_smoke3.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv
import json

from src.common.dataset_config import (
    active_datasets, load_hf_samples, load_manifest, parse_class_label,
)
from src.common.io import load_prompts
from src.common.metrics import sample_class_images
from src.vision_baseline import cls_question


def _load_reference_images(path: Path) -> dict[tuple[str, str], list[str]]:
    """Same as verifier_cot.py's own _load_reference_images (see its
    docstring for the full rationale) — duplicated rather than imported
    across module boundaries since it's a small private helper there, not
    a shared public API."""
    out: dict[tuple[str, str], list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            names = out.setdefault(key, [])
            name = r.get("image_name", "")
            if name and name not in names:
                names.append(name)
    return out


def build_tasks(args: argparse.Namespace) -> list[dict]:
    prompts = load_prompts(Path(args.prompts))
    manifest = load_manifest(Path(args.manifest))
    reference_images = _load_reference_images(Path(args.match_images_from)) if args.match_images_from else None

    rows = []
    for ds in active_datasets(Path(args.config)):
        if ds.skip or not ds.task_type:
            continue
        if args.subsample is not None and ds.agml_name not in args.subsample:
            continue
        try:
            samples = load_hf_samples(
                ds.agml_name, manifest, Path(args.images_root), excluded_classes=ds.excluded_classes,
            )
        except KeyError:
            print(f"  [skip] {ds.agml_name} — not in manifest")
            continue
        by_label: dict[str, list[dict]] = {}
        for s in samples:
            by_label.setdefault(s["label"], []).append(s)
        for raw_label, items in by_label.items():
            path_by_name = {Path(it["path"]).name: it["path"] for it in items}
            crop_val, clean_label = parse_class_label(raw_label, ds.crop)
            if reference_images is not None:
                ref_names = [n for n in reference_images.get((ds.agml_name, clean_label), []) if n in path_by_name]
                if ref_names:
                    names = ref_names[: args.n_samples]
                else:
                    print(f"  [warn] --match_images_from: no overlapping images for "
                          f"{ds.agml_name}/{clean_label} — falling back to an independent sample")
                    names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            else:
                names = sample_class_images(list(path_by_name.keys()), args.n_samples)
            task_type = ds.effective_task_type(raw_label)
            question = cls_question(task_type, crop_val, prompts)
            for name in names:
                rows.append({
                    "agml_dataset": ds.agml_name,
                    "class_label": clean_label,
                    "image_name": name,
                    "image_path": path_by_name[name],
                    "task": question,
                })
    return rows


def main(args: argparse.Namespace) -> None:
    rows = build_tasks(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_datasets = len({r["agml_dataset"] for r in rows})
    n_classes = len({(r["agml_dataset"], r["class_label"]) for r in rows})
    print(f"Wrote {len(rows)} task(s) across {n_classes} (dataset, class) group(s) "
          f"in {n_datasets} dataset(s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="src/config/datasets.yaml")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--manifest", default="dataset/selected_images.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--n_samples", type=int, default=3,
                         help="Images/class. Kept small (3) by default — this file is meant to be "
                              "a deliberate, readable starting point, not vision_baseline.py's own "
                              "default of 10+; raise it once react-baseline itself is trusted.")
    parser.add_argument("--subsample", nargs="+", default=None,
                         help="Restrict to these dataset(s) only (agml_name, space-separated) — "
                              "same convention as verifier_cot.py's own --subsample.")
    parser.add_argument("--match_images_from", default=None,
                         help="Path to an existing single-prediction-per-row checkpoint (e.g. "
                              "outputs_final/vision_baseline_thinking/<model>/ckpt_<model>.csv) — "
                              "replays that run's own sampled image names for a true apples-to-apples "
                              "image set, same convention as verifier_cot.py's own flag of this name.")
    parser.add_argument("--output", default="outputs_final/react_tasks.jsonl")
    main(parser.parse_args())
