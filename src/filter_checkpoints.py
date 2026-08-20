"""
Write filtered COPIES of existing single-prediction-per-row checkpoints
(vision_baseline / desc_augmented_vision / self_cot / cot_yaml) restricted
to a smaller, reproducible image set — new `<condition>_filter/` directories,
originals untouched.

Why filter existing output instead of re-running at a smaller --n_samples:
np.random.Generator.choice(N, size=n, replace=False) (every stage's own
per-class sampler) is NOT nested across different n with the same seed, so
independently re-running at --n_samples 25 would NOT reproduce a subset of
an existing --n_samples 50 run — see scripts/agml_hf/truncate_manifest.py's
docstring for the full explanation and dataset/selected_images_25.json,
the reproducible manifest this should match. Confirmed empirically that
every one of these conditions' EXISTING 50-image checkpoints already store
identical per-class image order (all hit sample_class_images' own
short-circuit at n_samples==manifest's n_per_class==50), so taking each
one's own first --n_samples rows per class reproduces exactly what a fresh
run against the truncated manifest would produce, with zero re-computation.

Only each condition's primary ckpt_<model_safe>.csv is touched (the single-
prediction-per-row file verifier_summary.py's comparison chart reads) —
cot_yaml's auxiliary ckpt_steps_*/ckpt_swap_* files and any *.bak.csv
archives are left alone, out of scope for this comparison.

Usage:
    uv run python main.py filter-checkpoints \\
        --reference_dir outputs_final/vision_baseline/ \\
        --n_samples 25
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_CONDITION_DIRS = ["vision_baseline", "desc_augmented_vision", "self_cot", "cot_yaml"]


def reference_image_keys(reference_ckpt: Path, n_samples: int) -> set[tuple[str, str, str]]:
    """Same as verifier_summary.py's own reference_image_keys — duplicated
    (not imported) so this stays a standalone, dependency-light utility;
    both read the identical 'first n_samples image_names per (dataset,
    class), in the checkpoint's own row order' selection."""
    by_group: dict[tuple[str, str], list[str]] = {}
    with open(reference_ckpt, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            names = by_group.setdefault(key, [])
            name = r.get("image_name", "")
            if name and name not in names:
                names.append(name)
    return {(ds, cl, name) for (ds, cl), names in by_group.items() for name in names[:n_samples]}


def filter_ckpt(src: Path, dst: Path, keys: set[tuple[str, str, str]]) -> tuple[int, int]:
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    kept = [r for r in rows if (r.get("agml_dataset"), r.get("class_label"), r.get("image_name")) in keys]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    return len(rows), len(kept)


def main(args: argparse.Namespace) -> None:
    reference_dir = Path(args.reference_dir)
    suffix = args.out_suffix

    for cond in args.conditions:
        cond_dir = Path("outputs_final") / cond
        if not cond_dir.exists():
            print(f"[skip] {cond_dir} does not exist")
            continue
        out_cond_dir = Path("outputs_final") / f"{cond}{suffix}"
        for model_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            model_safe = model_dir.name
            ckpt = model_dir / f"ckpt_{model_safe}.csv"
            if not ckpt.exists():
                print(f"  [skip] {ckpt} not found")
                continue
            ref_ckpt = reference_dir / model_safe / f"ckpt_{model_safe}.csv"
            if not ref_ckpt.exists():
                print(f"  [skip] {cond}/{model_safe}: no reference checkpoint at {ref_ckpt}")
                continue
            keys = reference_image_keys(ref_ckpt, args.n_samples)
            dst = out_cond_dir / model_safe / f"ckpt_{model_safe}.csv"
            n_before, n_after = filter_ckpt(ckpt, dst, keys)
            print(f"  {cond}/{model_safe}: {n_before} -> {n_after} row(s)  ->  {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_dir", default="outputs_final/vision_baseline/",
                   help="Directory whose per-model checkpoints define the image order/selection "
                        "(same convention as verifier_cot.py's --match_images_from).")
    p.add_argument("--n_samples", type=int, default=25)
    p.add_argument("--conditions", nargs="+", default=_CONDITION_DIRS,
                   help="outputs_final/ subdirectory names to filter.")
    p.add_argument("--out_suffix", default="_filter")
    main(p.parse_args())
