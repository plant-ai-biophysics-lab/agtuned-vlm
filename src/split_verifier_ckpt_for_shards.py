"""One-off helper: split an already-in-progress, UNSHARDED verifier-cot
checkpoint into N per-shard checkpoint files, so a resume with
--num_shards N can pick up the existing work instead of redoing it.

Why this is needed: verifier_cot.py's run_model() assigns groups to shards
via all_groups[shard_index::num_shards], computed from _build_groups()'s
deterministic order, BEFORE any checkpoint-based filtering. Each shard then
reads its OWN checkpoint file (ckpt_<model>_<type>_shard{i}of{N}.csv), never
the plain ckpt_<model>_<type>.csv an unsharded run writes to. So just adding
--num_shards to a resume starts every shard from zero. This script mirrors
_build_groups()'s own setup (same manifest/match_images_from auto-detect
logic run_model() uses) to compute each (dataset, class) group's target
shard under the SAME slicing rule, then splits the existing rows accordingly.

Must be run with the exact same manifest/--subsample/--random_datasets/
--random_classes_per_task/--match_images_from the original run used (all
left at default here, matching the command that was actually run), or the
group order won't match and rows will land in the wrong shard.

Usage:
    uv run python scripts/split_verifier_ckpt_for_shards.py \
        --model Qwen/Qwen3.8-27B-FP8 \
        --verifier_type ppt \
        --num_shards 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ -> repo root

from src.verifier_cot import _build_groups, VERIFIER_CKPT_FIELDS, _load_reference_images
from src.common.dataset_config import load_manifest
from src.common.vision_eval import safe_model_name, load_ckpt, rewrite_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--verifier_type", choices=["simple", "top3", "ppt"], default="ppt")
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--config", default="src/config/datasets.yaml")
    p.add_argument("--manifest", default="dataset/selected_images.json")
    p.add_argument("--images_root", default="dataset/images")
    p.add_argument("--output_dir", default="outputs_final/verifier_cot/")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--subsample", nargs="+", default=None)
    p.add_argument("--random_datasets", type=int, default=None)
    p.add_argument("--random_classes_per_task", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match_images_from", default=None)
    args = p.parse_args()

    model_dir = Path(args.output_dir) / safe_model_name(args.model)
    ckpt_stem = f"ckpt_{safe_model_name(args.model)}"
    if args.verifier_type != "simple":
        ckpt_stem += f"_{args.verifier_type}"
    src_ckpt_path = model_dir / f"{ckpt_stem}.csv"
    if not src_ckpt_path.exists():
        sys.exit(f"ERROR: no existing checkpoint at {src_ckpt_path} — nothing to split.")

    # Same auto-detect run_model() uses when --match_images_from isn't passed.
    match_images_from = args.match_images_from
    if match_images_from is None:
        candidate = Path("outputs_final/vision_baseline") / safe_model_name(args.model) / f"ckpt_{safe_model_name(args.model)}.csv"
        if candidate.exists():
            match_images_from = str(candidate)
            print(f"[match_images_from] auto-detected {match_images_from}")
    reference_images = None
    if match_images_from and Path(match_images_from).exists():
        reference_images = _load_reference_images(Path(match_images_from))

    manifest = load_manifest(Path(args.manifest))
    dataset_filter = set(args.subsample) if args.subsample else None
    all_groups = list(_build_groups(args, manifest, dataset_filter=dataset_filter, reference_images=reference_images))
    if args.random_datasets:
        from src.verifier_cot import _sample_random_datasets
        all_groups = _sample_random_datasets(all_groups, args.random_datasets, args.seed)
    if args.random_classes_per_task:
        from src.verifier_cot import _sample_random_classes_per_task
        all_groups = _sample_random_classes_per_task(all_groups, args.random_classes_per_task, args.seed)

    print(f"[groups] {len(all_groups)} (dataset, class) group(s) total")

    group_shard: dict[tuple[str, str], int] = {}
    for i, g in enumerate(all_groups):
        ds, clean_label = g[0], g[1]
        group_shard[(ds.agml_name, clean_label)] = i % args.num_shards

    rows, _ = load_ckpt(src_ckpt_path, fields=VERIFIER_CKPT_FIELDS)
    print(f"[source] {len(rows)} row(s) in {src_ckpt_path.name}")

    shard_rows: dict[int, list] = {i: [] for i in range(args.num_shards)}
    n_unmatched = 0
    for r in rows:
        key = (r["agml_dataset"], r["class_label"])
        shard = group_shard.get(key)
        if shard is None:
            n_unmatched += 1
            continue
        shard_rows[shard].append(r)

    if n_unmatched:
        print(f"[warn] {n_unmatched} row(s) didn't match any current group — "
              f"left out of every shard file (will be regenerated on resume).")

    for i in range(args.num_shards):
        dst = model_dir / f"{ckpt_stem}_shard{i}of{args.num_shards}.csv"
        rewrite_ckpt(dst, shard_rows[i], fields=list(VERIFIER_CKPT_FIELDS))
        print(f"  shard {i}: {len(shard_rows[i])} row(s) -> {dst}")

    print("\nDone. Launch each shard with --num_shards "
          f"{args.num_shards} --shard_index i (i in [0, {args.num_shards})), "
          "WITHOUT --rerun, same --model/--verifier_type/--output_dir.")


if __name__ == "__main__":
    main()
