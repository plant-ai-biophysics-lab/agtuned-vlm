"""Standalone check: prints the actual generate_class_name_variants() output
for a few real detection datasets' class names, so synonym quality can be
inspected directly. Unlike each K-candidate's own generation call, the
synonym-generation call itself is not written to detection_verifier.py's
verifier_log.jsonl, so this is the only way to see it after the fact.

Depends on the main repo's src/ package (not an isolated experiments/
sub-project) -- run from the repo root with the main env:

    uv run python experiments/synonym_check/print_synonyms.py --model google/gemma-4-E4B-it

Edit SAMPLE_DATASETS below to check different datasets/classes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.dataset_config import active_datasets, crop_display
from src.common.vlm_load import load_llm
from src.detection_verifier import generate_class_name_variants

SAMPLE_DATASETS = [
    "CocoaMFDB_detection",
    "mango_detection_australia",
    "RoCoLe_disease_detection",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--config", default="src/config/detection_datasets.yaml")
    parser.add_argument("--k_candidates", type=int, default=6)
    args = parser.parse_args()

    llm, tokenizer = load_llm(args.model, caller_tag="print_synonyms")

    by_name = {ds.agml_name: ds for ds in active_datasets(Path(args.config))}
    for name in SAMPLE_DATASETS:
        ds = by_name.get(name)
        if ds is None:
            print(f"[skip] {name} not found in {args.config}")
            continue
        class_names = ds.clean_classes
        print(f"\n=== {name} (crop={crop_display(ds.crop)}) ===")
        print(f"original class_names: {class_names}")
        variants = generate_class_name_variants(llm, tokenizer, class_names, args.k_candidates)
        for k, v in enumerate(variants):
            print(f"  variant {k}: {v}")


# vLLM spawns worker subprocesses that re-import this module -- everything
# that actually runs (arg parsing, model loading) must live behind this
# guard, or Python's multiprocessing "spawn" start method refuses to launch
# the workers ("attempt has been made to start a new process before the
# current process has finished its bootstrapping phase").
if __name__ == "__main__":
    main()
