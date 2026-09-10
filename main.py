"""Pipeline entry point — delegates to individual stage scripts."""

import runpy
import sys

STAGES = {
    "catalog-datasets":   "scripts.agml_hf.catalog_datasets",
    "migrate-datasets":   "scripts.agml_hf.migrate_datasets_yaml",
    "download-datasets":  "scripts.agml_hf.download_and_sample",
    "catalog-detection-datasets":  "scripts.agml_hf.catalog_detection_datasets",
    "report-detection-catalog":    "scripts.agml_hf.report_detection_catalog",
    "migrate-detection-datasets":  "scripts.agml_hf.migrate_detection_datasets_yaml",
    "download-detection-datasets": "scripts.agml_hf.download_and_sample_detection",
    "detection-baseline":          "src.detection_baseline",
    "detection-qc-plot":           "src.detection_qc_plot",
    "detection-verifier":          "src.detection_verifier",
    "detection-summary":           "src.detection_summary",
    "build-reference-db": "src.data_generation.build_reference_db",
    "data-gen":           "src.data_generation.generate",
    "check-vision-towers": "src.linear_probe.check_vision_towers",
    "probe":         "src.linear_probe.probe",
    "probe-collate-seeds": "src.linear_probe.collate_seeds",
    "cluster":       "src.clustering_eval.eval",
    "cluster-viz":   "src.clustering_eval.sample_viz",
    "majority-vote": "src.majority_vote",
    "cot-step-analysis": "src.cot_step_analysis",
    "vision-baseline": "src.vision_baseline",
    "desc-aug-v2":   "src.desc_augmented_vision_v2",
    "vision-baseline-collate": "src.vision_baseline_collate",
    "self-cot":      "src.self_cot",
    "cot-yaml":      "src.cot_yaml",
    "cot-yaml-step-report": "src.cot_yaml_step_report",
    "cot-compare":   "src.cot_compare",
    "verifier-cot":  "src.verifier_cot",
    "verifier-summary": "src.verifier_summary",
    "verifier-scaling-summary": "src.verifier_scaling_summary",
    "filter-checkpoints": "src.filter_checkpoints",
    "ablation":      "src.ablation",
    "judge-leniency": "src.judge_leniency",
    "judge-consistency": "src.judge_consistency",
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
    catalog-detection-datasets  Stage 5 (prep) — resolve class_names + bbox convention for
                               detection-type datasets already found by catalog-datasets
    migrate-detection-datasets  Stage 5 (prep) — regenerate src/config/detection_datasets.yaml
                               from catalog-detection-datasets' output
    download-detection-datasets Stage 5 — download + class-balanced sample detection datasets,
                               write selection manifest (dataset/selected_images_detection.json)
    build-reference-db Stage 0  — fetch Wikipedia + live web-search sources per class (run this first)
    data-gen           Stage 0  — synthesize agml_classification.jsonl from build-reference-db's
                                   fetched sources (no vision pass)
    check-vision-towers      — pre-check: do candidate models expose a resolvable vision
                               tower before running the full probe pipeline on them
    probe           Stage 1 — linear probe: frozen-embedding separability across backbones
    probe-collate-seeds     — combine multiple probe.py --seed runs into one summary_by_task
                               chart with 95% CI error bars (Student's t across seeds)
    cluster                 — unsupervised clustering eval of frozen embeddings (NMI, ARI, Silhouette)
    cluster-viz             — visualise easiest/hardest datasets per task with sample class images
    cot-step-analysis        — per-step CoT accuracy/anchoring analysis
    vision-baseline Stage 2a — vision lower bound: image + classification question only,
                               classification-only (F1 + ROUGE-SS), per-sample resumable
    desc-aug-v2     Stage 2b — vision upper bound: image + reference description +
                               classification question, same resumable per-model setup
    vision-baseline-collate  Stage 2c — combine vision-baseline + desc-aug-v2 across all
                               models into one summary + bar chart (95% CI over classes)
    self-cot        Stage 2d — self-generated description, then classify using it as
                               context (own reasoning, no external description)
    cot-yaml        Stage 2e — structured multi-step reasoning (cot_prompts.yaml) before
                               classifying; same resumable per-model setup as vision-baseline.
                               --per_step also classifies after every intermediate step;
                               --swap_datasets re-runs a dataset subset once per step-order
                               rotation (robustness check)
    cot-yaml-step-report     — per-step F1 report from cot-yaml --per_step / --swap_datasets
                               output (no GPU needed)
    cot-compare              — win rate / mean gain / mean loss / top-5 gains & losses for
                               desc-aug-v2, self-cot, cot-yaml vs. vision-baseline (no GPU needed)
    verifier-cot    Stage 4 — single-round K-candidate + PPT-tournament verifier
                               (docs/stage4_verifier.md). K candidates generated per image, ranked
                               once by the Probabilistic Pivot Tournament, argmax(win_rate) reported.
                               --k_candidates/--k_pivots are the main tunables. Compared against
                               src/verifier_lmv.py's full PPT tournament (run separately, not wired
                               into main.py — see its own --help) in `cot-compare`.
    ablation                 — reviewer-question ablation: isolates rubric vs. K-sampling
                               vs. verifier-selection (Q2/Q3), verifier confidence
                               calibration/ranking accuracy (Q4), oracle-description
                               vision-leakage tabulation (Q5), and whether judged_correct
                               is driven by the model's stated answer vs. the rubric's
                               descriptive fields around it (judge-recheck). --step
                               {generate,selection,calibration,oracle-leakage,judge-recheck,
                               all}; "generate" and "judge-recheck" need a GPU/model, the
                               rest reconstruct from existing outputs_final/ runs. Output
                               under ablation/.
    judge-leniency            — judge leniency baseline (needs judge model/GPU): at each of a
                               swept set of temperatures, freshly re-judges every resolvable
                               row twice (full model response vs. just the resolved_class
                               string, same judge/template both times) and compares the two
                               — isolates judge leniency from resolve_class's own string-
                               matching limits, and tracks whether leniency itself grows with
                               temperature. --ckpt (path/glob/dir), --output_dir, --label,
                               --temperatures (default 0.1 0.5 1.0, matches judge-consistency),
                               --judge_model. Writes leniency_gap_T<temp>.png,
                               leniency_gap_by_temperature.png, resolve_failures.txt,
                               rechecked_T<temp>.csv, leniency_gap.json.
    judge-consistency         — judge self-consistency (needs judge model/GPU): repeats the
                               current full-response judge call N times per row at each of a
                               swept set of temperatures (judge runs at temperature=0.0
                               normally, so this forces real sampling variance) and reports
                               flip_rate per task_type/temperature. --temperatures,
                               --n_repeats; --leniency_gap_json (a judge-leniency run's
                               leniency_gap.json for the SAME checkpoint) also writes a
                               leniency-vs-flip-rate scatter. Writes
                               flip_rate_by_temperature.png, flip_rate.json,
                               [leniency_vs_flip_rate.png].
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

    python main.py vision-baseline --model google/gemma-4-26B-A4B-it
    python main.py desc-aug-v2     --model google/gemma-4-26B-A4B-it
    python main.py self-cot        --model google/gemma-4-26B-A4B-it
    python main.py cot-yaml        --model google/gemma-4-26B-A4B-it --per_step
    # ... repeat --model per model, then once ALL models/both scripts are done:
    python main.py vision-baseline-collate
    python main.py cot-yaml-step-report
    python main.py cot-compare
    # then, for cot-compare's top gain/loss datasets:
    python main.py cot-yaml --model google/gemma-4-26B-A4B-it \\
        --swap_datasets ds_one_classification ds_two_classification
    python main.py cot-yaml-step-report

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
