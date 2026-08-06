"""Pipeline entry point — delegates to individual stage scripts."""

import runpy
import sys

STAGES = {
    "catalog-datasets":   "scripts.agml_hf.catalog_datasets",
    "migrate-datasets":   "scripts.agml_hf.migrate_datasets_yaml",
    "download-datasets":  "scripts.agml_hf.download_and_sample",
    "build-reference-db": "src.data_generation.build_reference_db",
    "data-gen":           "src.data_generation.generate",
    "check-vision-towers": "src.linear_probe.check_vision_towers",
    "probe":         "src.linear_probe.probe",
    "cluster":       "src.clustering_eval.eval",
    "cluster-viz":   "src.clustering_eval.sample_viz",
    "majority-vote": "src.majority_vote",
    "cot":           "src.cot_vision",
    "cot-step-analysis": "src.cot_step_analysis",
    "language":      "src.language_assessment.assess",
    "language-plot": "src.language_assessment.plot",
    "vision":        "src.vision_assessment.assess",
    "vision-plot":   "src.vision_assessment.plot",
    "aug-vision":    "src.desc_augmented_vision",
    "aug-vision-plot": "src.desc_augmented_vision_plot",
    "collate":       "src.collate_results",
}

_HELP = """
AgML VLM Pipeline

Usage:
    python main.py <stage> [stage-args...]

Stages:
    catalog-datasets   Stage 0a — catalog all Project-AgML datasets on HuggingFace
    migrate-datasets   Stage 0a — regenerate src/config/datasets.yaml from the HF catalog
    download-datasets  Stage 0a — download + class-balanced sample datasets, write selection manifest
    build-reference-db Stage 0  — fetch Wikipedia + live web-search sources per class (run this first)
    data-gen           Stage 0  — synthesize agml_classification.jsonl from build-reference-db's
                                   fetched sources (no vision pass)
    check-vision-towers      — pre-check: do candidate models expose a resolvable vision
                               tower before running the full probe pipeline on them
    probe           Stage 1 — linear probe: frozen-embedding separability across backbones
    cluster                 — unsupervised clustering eval of frozen embeddings (NMI, ARI, Silhouette)
    cluster-viz             — visualise easiest/hardest datasets per task with sample class images
    language        Stage 2 — language assessment: text-only agricultural VQA (ROUGE-SS + F1)
    language-plot           — regenerate language assessment charts from saved metrics JSON
    vision          Stage 3 — vision assessment: multimodal VQA with image input (ROUGE-SS + F1)
    vision-plot             — regenerate vision assessment charts from saved metrics JSON
    aug-vision      Stage 4 — description-augmented vision: test if language descriptions improve vision cls
    aug-vision-plot         — regenerate aug-vision charts from saved CSVs
    cot             Stage 4b — chain-of-thought vision assessment (self-cot + cot-yaml)
    cot-step-analysis        — per-step CoT accuracy/anchoring analysis
    collate         Stage 5 — collate all stage metrics into results_table.csv / .tex

Examples:
    python main.py probe \\
        --config src/config/datasets.yaml \\
        --output_dir outputs/linear_probe/

    python main.py language \\
        --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \\
        --classification_jsonl outputs/data_generation/agml_classification.jsonl \\
        --config src/config/datasets.yaml \\
        --prompts src/config/prompts.yaml \\
        --models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \\
        --output_dir outputs/language_assessment/

    python main.py vision \\
        --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \\
        --config src/config/datasets.yaml \\
        --prompts src/config/prompts.yaml \\
        --models google/gemma-4-26B-A4B-it Qwen/Qwen3.6-27B \\
        --output_dir outputs/vision_assessment/

Pass --help after the stage name for all available options:
    python main.py probe --help
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP.strip())
        sys.exit(0)

    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"Unknown stage: {stage!r}")
        print(f"Valid stages: {', '.join(STAGES)}")
        sys.exit(1)

    # Remove the stage name so the delegated script sees only its own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    runpy.run_module(STAGES[stage], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
