"""Pipeline entry point — delegates to individual stage scripts."""

import runpy
import sys

STAGES = {
    "generate":      "src.data_generation.generate",
    "build-db":      "src.data_generation.build_reference_db",
    "verify":        "src.data_generation.verify",
    "patch-healthy": "src.data_generation.patch_healthy",
    "probe":         "src.linear_probe.probe",
    "language":      "src.language_assessment.assess",
    "language-plot": "src.language_assessment.plot",
    "vision":        "src.vision_assessment.assess",
    "vision-plot":   "src.vision_assessment.plot",
    "collate":       "src.collate_results",
    "error-overlap": "src.error_overlap.analyze",
    "finetune":      "src.finetune.train",
}

_HELP = """
AgML VLM Pipeline

Usage:
    python main.py <stage> [stage-args...]

Stages:
    generate    Stage 1a — generate synthetic descriptions from AgML images
    build-db    Stage 1b — export CSV + build reference DB (Wikipedia / IDtools / CGIAR)
    verify      Stage 1c — rewrite descriptions grounded in reference sources
    patch-healthy       — patch healthy classes in descriptions_corrected.csv in-place
    probe       Stage 2  — linear probe across all AgML datasets
    language    Stage 3  — language assessment (text-only models)
    language-plot       — regenerate language assessment charts from saved metrics JSON
    vision      Stage 4  — vision assessment (multimodal models)

Examples:
    python main.py generate \\
        --model      Qwen/Qwen3.6-27B \\
        --config     src/config/datasets.yaml \\
        --prompts    src/config/prompts.yaml \\
        --output_dir outputs/data_generation/

    python main.py build-db \\
        --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \\
        --output_db          outputs/data_generation/reference_db.json

    python main.py verify \\
        --descriptions_csv outputs/data_generation/descriptions_export.csv \\
        --reference_db     outputs/data_generation/reference_db.json \\
        --output_csv       outputs/data_generation/descriptions_corrected.csv

    python main.py probe \\
        --config src/config/datasets.yaml \\
        --output_dir outputs/linear_probe/

    python main.py language \\
        --descriptions_jsonl outputs/data_generation/agml_descriptions.jsonl \\
        --classification_jsonl outputs/data_generation/agml_classification.jsonl \\
        --config src/config/datasets.yaml \\
        --prompts src/config/prompts.yaml \\
        --models openai/gpt-oss-20b meta-llama/Llama-4-Scout-17B-16E-Instruct \\
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
