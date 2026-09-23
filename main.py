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
    "verifier-score-analysis": "src.verifier_score_analysis",
    "filter-checkpoints": "src.filter_checkpoints",
    "ablation":      "src.ablation",
    "judge-leniency": "src.judge_leniency",
    "judge-consistency": "src.judge_consistency",
    "judge-template-sensitivity": "src.judge_template_sensitivity",
    "judge-qa-export": "src.judge_qa_export",
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
                               classification-only (F1 + ROUGE-SS), per-sample resumable.
                               --enable_thinking/--gen_max_tokens/--chat_template_path (generation-side
                               thinking), --judge_enable_thinking/--judge_max_tokens/
                               --judge_chat_template_path (judge-side thinking), --rubric ("Baseline with
                               rubric" — wrap the classification question through verifier_prompts.yaml's
                               fill-in-the-blank criteria chain before the final answer; combines freely
                               with --enable_thinking), --num_shards/--shard_index (data-parallel across
                               GPUs) + --merge_shards, --backfill_judge_results + --rechecked_csv (copy a
                               Phase 1-4 template's full-response verdict into this checkpoint's own
                               judged_correct), --compare_templates + --template_rechecked_csv — see
                               docs/stage1_judge_improvements.md.
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
    verifier-score-analysis   — diagnostic (no GPU needed): re-derives the paper's Appendix C
                               Figure 8/9 checks (score distribution + retained-set accuracy vs.
                               threshold, nearest-letter histogram) from an existing verifier-cot
                               run's own ckpt CSV + verifier_log.jsonl, to see whether the
                               fine-grained-reward score is discriminating (or inverted/collapsed)
                               BEFORE picking a fix. --model/--output_dir or --verifier_dir.
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
                               swept set of temperatures, freshly re-judges every taggable row
                               twice (full model response vs. just the raw <answer> tag text,
                               same judge/template both times) and compares the two — isolates
                               judge leniency from resolve_class's own string-matching limits,
                               and tracks whether leniency itself grows with temperature.
                               --ckpt (path/glob/dir), --output_dir, --label, --temperatures
                               (default 0.7, matches judge-consistency),
                               --template_key/--class_list (re-run under a different judge
                               template — see judge-template-sensitivity), --limit, --judge_model
                               (default RedHatAI/gemma-4-31B-it-FP8-dynamic),
                               --judge_enable_thinking/--judge_max_tokens/--judge_chat_template_path
                               (off/unset by default — see docs/stage1_judge_improvements.md).
                               Writes leniency_gap_T<temp>.png,
                               leniency_gap_by_temperature.png, disagreement_matrix_T<temp>.png
                               (the full-response/resolved-class 2x2 breakdown — raw counts in
                               each disagreement direction, not just leniency_gap's net rate),
                               missing_answer_tag.txt, rechecked_T<temp>.csv, leniency_gap.json.
    judge-consistency         — judge self-consistency (needs judge model/GPU, 2x the calls of
                               a single condition): repeats the judge call N times per row at
                               each of a swept set of temperatures, for BOTH the full-response
                               and <answer>-tag-only conditions (judge runs at temperature=0.0
                               normally, so this forces real sampling variance), and reports
                               flip_rate per condition/task_type/temperature. --temperatures,
                               --n_repeats, --template_key/--class_list, --judge_enable_thinking/
                               --judge_max_tokens (same as judge-leniency); --leniency_gap_json
                               (a judge-leniency run's leniency_gap.json for the SAME checkpoint)
                               also writes a leniency-vs-flip-rate scatter per condition. Writes
                               flip_rate_by_temperature_full.png,
                               flip_rate_by_temperature_answer_only.png, flip_rate.json,
                               [leniency_vs_flip_rate_full.png, leniency_vs_flip_rate_answer_only.png].
    judge-template-sensitivity — judge template sensitivity sweep (needs judge model/GPU,
                               the most expensive step in the plan — start with --limit):
                               for each of 4 templates (current, a specificity-strict rewrite
                               for plain classification, a +full-class-list variant, and Phase 4's
                               hybrid — class_list's context + specificity's guardrail, crediting
                               necessary category membership without requiring exact wording),
                               re-runs BOTH judge-leniency and judge-consistency's full
                               temperature sweeps under that template, writing each variant's
                               complete Phase 1+2 figure set to its own subfolder
                               (<output_dir>/<variant>/leniency/, /<variant>/consistency/) —
                               plus a cheap greedy-temperature summary of all 4 variants against
                               the checkpoint's own deterministic resolve_class match, and a
                               length/verbosity bias report. --template {all,current,specificity,
                               class_list,hybrid} (default all) runs just one variant this invocation —
                               re-run against the SAME --output_dir for each remaining variant
                               and the cross-variant summary progressively fills in from disk,
                               no re-judging. --temperatures, --n_repeats, --n_length_buckets,
                               --judge_enable_thinking/--judge_max_tokens/--judge_chat_template_path
                               (same as judge-leniency), --summary_temperature (default 0.7, must be
                               included in --temperatures for a variant to appear in
                               leniency_and_false_positive_summary.png), --skip_consistency (skip Phase 2
                               entirely for every variant — by far the most expensive part; nothing but
                               the LaTeX table's Flip rate column and the leniency_vs_flip_rate_*.png
                               plots need it).
                               Writes template_sensitivity.png,
                               length_bias.png (skipped until the "current" variant has run),
                               judge_accuracy_f1.png (judge accuracy + judge-scored macro-F1 per
                               variant), leniency_and_false_positive_summary.png (+ _stats.json;
                               leniency_gap and false-positive-only credit rate per variant, at
                               --summary_temperature), oracle_validation_sample.csv (hybrid vs.
                               current, sampled mostly-disagreement rows for manual human
                               validation — needs both variants present), template_sensitivity_table.tex
                               (LaTeX summary: leniency_gap, flip_rate, and both disagreement-direction
                               counts/rates per variant x temperature), template_sensitivity.json
                               (top-level summary), each variant's own greedy_summary_rechecked.csv, plus
                               each variant's own leniency_gap*.png / disagreement_matrix*.png /
                               flip_rate_by_temperature_{full,answer_only}.png /
                               leniency_vs_flip_rate_{full,answer_only}.png underneath.
    judge-qa-export            — no GPU needed: samples a review-ready CSV for human QA of judge
                               verdicts, comparing the "current" and "hybrid" templates' own
                               rechecked_T<temp>.csv files. Three pools per task_type
                               (--n_per_task, default 50): targeted_hybrid (hybrid disagrees with
                               a cheap extracted-answer-vs-ground-truth text check), disagreement
                               (current vs. hybrid differ), random_control (unbiased baseline).
                               --current_csv/--hybrid_csv, --seed (default 42), --output_csv
                               (default dataset/judge_qa_review/judge_qa_review.csv — dataset/, not
                               outputs_final/, so it's git-tracked for reproducibility).
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
