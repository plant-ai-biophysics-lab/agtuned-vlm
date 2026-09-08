"""
Stage 0 utility — truncate an existing selected_images.json manifest to a
SMALLER --n_per_class as an explicit, publishable, reproducible artifact,
instead of relying on each downstream script's own sample_class_images()
draw to (maybe) line up across different --n_samples values.

Why this exists, not "just re-run download_and_sample.py with a smaller
--n_per_class": both subsample_class_balanced (Stage 0's own sampler) and
sample_class_images (every downstream script's per-run sampler) draw via
np.random.Generator.choice(N, size=n, replace=False), which is NOT nested
across different n with the same seed — confirmed empirically (choice(100,
size=25) is only ~64% overlapping with choice(100, size=50) from the same
seed). So a fresh Stage 0 run at --n_per_class 25, or any downstream
script's own --n_samples 25, is a DIFFERENT draw, not a guaranteed subset
of an existing 50-image run — see download_and_sample.py's own "Caveat
when changing --n_per_class" docstring section, which already flags this
for Stage 0 itself.

The one thing that IS trivially nested: an existing manifest's own stored
per-class index list is already in a fixed order, and (confirmed on real
checkpoints — see the paper's methods notes) every downstream script that
requested exactly --n_samples == that manifest's --n_per_class hit
sample_class_images' own short-circuit (`if n >= len(pool): return pool
unchanged`), so vision_baseline/desc_aug/self_cot/cot_yaml all preserved
that exact order with zero randomness. Truncating THAT stored order to its
first --n_per_class entries is therefore guaranteed consistent with what
those already-completed runs used — no re-sampling, no RNG, just a slice.

Downstream usage: point any stage's --manifest at the truncated file with
a matching --n_samples/--n_per_class (so its own sample_class_images call
also hits the short-circuit) to get the identical smaller image set with
zero risk of a fresh, non-nested draw.

Usage:
  uv run python scripts/agml_hf/truncate_manifest.py \
      --input dataset/selected_images.json \
      --n_per_class 25 \
      --output dataset/selected_images_25.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def truncate_manifest(input_path: Path, n_per_class: int) -> dict:
    manifest = json.loads(input_path.read_text())
    old_n = manifest.get("n_per_class")
    if old_n is not None and n_per_class > old_n:
        raise SystemExit(f"--n_per_class {n_per_class} exceeds the input manifest's own "
                          f"n_per_class={old_n} — nothing to truncate; a LARGER sample requires "
                          f"re-running Stage 0 (a fresh, non-nested draw), not truncation.")

    n_before = n_after = 0
    for ds_entry in manifest.get("datasets", {}).values():
        classes = ds_entry.get("classes", {})
        for label, indices in classes.items():
            n_before += len(indices)
            classes[label] = indices[:n_per_class]
            n_after += len(classes[label])

    manifest["n_per_class"] = n_per_class
    manifest["reproduced_from"] = str(input_path)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    print(f"  Truncated {n_before} -> {n_after} image indices "
          f"({len(manifest.get('datasets', {}))} dataset(s), n_per_class {old_n} -> {n_per_class})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="dataset/selected_images.json")
    ap.add_argument("--n_per_class", type=int, required=True)
    ap.add_argument("--output", required=True,
                     help="Never defaults to overwriting --input — the original larger manifest "
                          "stays intact for anything still referencing it.")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("--output must differ from --input — truncation is one-way and the "
                          "original manifest should stay recoverable.")

    manifest = truncate_manifest(input_path, args.n_per_class)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
