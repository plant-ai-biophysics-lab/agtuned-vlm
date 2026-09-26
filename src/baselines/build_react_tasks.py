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
        --output outputs_final/react_baseline/tasks/react_tasks_smoke3.jsonl

    # small varied smoke test — 3 DATASETS per task_type, not every active one:
    uv run python main.py react-build-tasks \\
        --n_samples 3 --num_datasets_per_task 3 \\
        --output outputs_final/react_baseline/tasks/react_tasks_smoke3.jsonl

    # true apples-to-apples against an existing vision-baseline run:
    uv run python main.py react-build-tasks \\
        --match_images_from outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \\
        --n_samples 10 \\
        --output outputs_final/react_baseline/tasks/react_tasks_full.jsonl

Then:
    uv run python main.py react-baseline --tasks_jsonl outputs_final/react_baseline/tasks/react_tasks_smoke3.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv
import json
import random
from collections import defaultdict

from src.common.dataset_config import (
    active_datasets, load_hf_samples, load_manifest, parse_class_label,
)
from src.common.io import load_prompts
from src.common.metrics import sample_class_images
from src.vision_baseline import apply_rubric, cls_question, load_rubric_templates


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


def _sample_datasets_per_task(datasets: list, n_per_task: int, seed: int) -> list:
    """Keep `n_per_task` DATASETS (not classes — a dataset with many classes
    still counts as one) per ds.task_type, randomly, seeded for
    reproducibility — for a smoke test that wants a small, varied spread
    across task types (disease/pest_damage/species_identification/
    quality_identification) rather than every active dataset. Same
    seeded-shuffle-then-slice pattern as verifier_cot.py's own
    _sample_random_classes_per_task, at the dataset level instead of the
    (dataset, class) group level — a task_type with fewer than n_per_task
    datasets keeps all of them."""
    rng = random.Random(seed)
    by_task: dict[str, list] = defaultdict(list)
    for ds in datasets:
        by_task[ds.task_type].append(ds)
    selected = []
    for task_type in sorted(by_task):
        task_datasets = by_task[task_type]
        rng.shuffle(task_datasets)
        selected.extend(task_datasets[:n_per_task])
    return selected


def build_tasks(args: argparse.Namespace) -> list[dict]:
    prompts = load_prompts(Path(args.prompts))
    manifest = load_manifest(Path(args.manifest))
    reference_images = _load_reference_images(Path(args.match_images_from)) if args.match_images_from else None

    active = [ds for ds in active_datasets(Path(args.config)) if not ds.skip and ds.task_type]
    if args.subsample is not None:
        active = [ds for ds in active if ds.agml_name in args.subsample]
    if getattr(args, "num_datasets_per_task", None):
        active = _sample_datasets_per_task(active, args.num_datasets_per_task, args.seed)
        by_task_counts = defaultdict(int)
        for ds in active:
            by_task_counts[ds.task_type] += 1
        print(f"  [num_datasets_per_task={args.num_datasets_per_task}, seed={args.seed}] "
              f"{len(active)} dataset(s) selected: {dict(by_task_counts)}")
    # --rubric: wrap cls_question()'s output through the SAME fill-in-the-
    # blank criteria chain vision_baseline.py's own --rubric flag uses (its
    # docstring covers the "Baseline with rubric" rationale) — this is ALSO
    # the exact prompt verifier_cot.py's own k-candidate generation step
    # always uses (verifier_config.yaml: "Candidate generation prompt is
    # ALWAYS the fill-in-the-blank structured_templates entry"), so
    # --rubric here is what makes a react-baseline task comparable to
    # verifier-cot's candidates too, not just vision-baseline's plain
    # question. Loaded once outside the per-class loop below, not per row.
    structured_templates = task_type_to_rubric = None
    if getattr(args, "rubric", False):
        structured_templates, task_type_to_rubric = load_rubric_templates()

    rows = []
    for ds in active:
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
            if structured_templates is not None:
                question = apply_rubric(question, task_type, structured_templates, task_type_to_rubric)
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
    parser.add_argument("--manifest", default="dataset/selected_images_10.json",
                         help="10/class, matching vision_baseline_config.yaml's own current default "
                              "(both switched back from the full 50/class dataset/selected_images.json "
                              "— a full-manifest verifier-cot run was projected at 1+ week). Pass the "
                              "full manifest explicitly if you want more than 10/class available to "
                              "sample from.")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--n_samples", type=int, default=3,
                         help="Images/class, capped by whatever --manifest actually has (10, by "
                              "default). Kept small (3) by default — this file is meant to be a "
                              "deliberate, readable starting point; raise it (up to 10) once "
                              "react-baseline itself is trusted.")
    parser.add_argument("--subsample", nargs="+", default=None,
                         help="Restrict to these dataset(s) only (agml_name, space-separated) — "
                              "same convention as verifier_cot.py's own --subsample. Applied BEFORE "
                              "--num_datasets_per_task if both are given.")
    parser.add_argument("--num_datasets_per_task", type=int, default=None,
                         help="Randomly keep this many DATASETS (not classes — one dataset can have "
                              "many classes and still counts once) per task_type (disease/pest_damage/"
                              "species_identification/quality_identification), for a small varied "
                              "smoke test instead of every active dataset. Seeded by --seed. Same "
                              "seeded-shuffle-then-slice pattern as verifier_cot.py's own "
                              "--random_classes_per_task, at the dataset level instead of the "
                              "(dataset, class) group level.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seeds --num_datasets_per_task's random selection, for reproducibility.")
    parser.add_argument("--match_images_from", default=None,
                         help="Path to an existing single-prediction-per-row checkpoint (e.g. "
                              "outputs_final/vision_baseline_thinking/<model>/ckpt_<model>.csv) — "
                              "replays that run's own sampled image names for a true apples-to-apples "
                              "image set, same convention as verifier_cot.py's own flag of this name.")
    parser.add_argument("--rubric", action="store_true",
                         help="Wrap the classification question through the same fill-in-the-blank "
                              "criteria chain vision_baseline.py's own --rubric flag uses — this is "
                              "ALSO the exact prompt verifier_cot.py's own k-candidate generation "
                              "step always uses, so pass this to compare react-baseline against "
                              "'Baseline with rubric' (Stage 2) or verifier-cot (Stage 4/legacy) "
                              "rather than plain vision-baseline. Off by default (plain question, "
                              "matching vision-baseline's own default).")
    parser.add_argument("--output", default="outputs_final/react_baseline/tasks/react_tasks.jsonl",
                         help="Nested under react-baseline's own output_dir (outputs_final/"
                              "react_baseline/) rather than a sibling path, so every react-baseline "
                              "artifact — task lists, per-model checkpoints, trajectory logs — lives "
                              "under one directory.")
    main(parser.parse_args())
