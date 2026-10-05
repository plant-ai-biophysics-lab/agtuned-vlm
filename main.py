"""Pipeline entry point — delegates to individual stage scripts."""

import runpy
import sys

STAGES = {
    "catalog-datasets":   "scripts.agml_hf.catalog_datasets",
    "migrate-datasets":   "scripts.agml_hf.migrate_datasets_yaml",
    "download-datasets":  "scripts.agml_hf.download_and_sample",
    "download-heldout-digitalgreen": "scripts.download_heldout_digitalgreen",
    "heldout-zeroshot": "src.heldout_zeroshot",
    "heldout-explain-traces": "src.heldout_explain_traces",
    "heldout-summarize": "src.heldout_summarize",
    "catalog-detection-datasets":  "scripts.agml_hf.catalog_detection_datasets",
    "report-detection-catalog":    "scripts.agml_hf.report_detection_catalog",
    "migrate-detection-datasets":  "scripts.agml_hf.migrate_detection_datasets_yaml",
    "download-detection-datasets": "scripts.agml_hf.download_and_sample_detection",
    "detection-baseline":          "src.detection_baseline",
    "detection-multiclass":        "src.detection_multiclass",
    "detection-box-verifier":      "src.detection_box_verifier",
    "detection-box-refine":        "src.detection_box_refine",
    "detection-final-metrics":     "src.detection_final_metrics",
    "detection-split-report":      "src.detection_split_report",
    "detection-comparison-summary": "src.detection_comparison_summary",
    "detection-box-verifier-summary": "src.detection_box_verifier_summary",
    "detection-qc-plot":           "src.detection_qc_plot",
    "detection-verifier":          "src.detection_verifier",
    "detection-summary":           "src.detection_summary",
    "detection-rubric-summary":    "src.detection_rubric_summary",
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
    "verifier-score-correlation-plot": "src.verifier_score_correlation_plot",
    "verifier-cot-qc-plot": "src.verifier_cot_qc_plot",
    "verifier-cot-qc-plot-top3": "src.verifier_cot_qc_plot_top3",
    "verifier-cot-top3-refine": "src.verifier_cot_top3_refine",
    "verifier-cot-comparison-summary": "src.verifier_cot_comparison_summary",
    "filter-checkpoints": "src.filter_checkpoints",
    "ablation":      "src.ablation",
    "judge-checkpoint": "src.judge_checkpoint",
    "judge-leniency": "src.judge_leniency",
    "judge-consistency": "src.judge_consistency",
    "judge-template-sensitivity": "src.judge_template_sensitivity",
    "judge-qa-export": "src.judge_qa_export",
    "collate":       "src.collate_results",
    "react-baseline": "src.baselines.react_baseline",
    "react-build-tasks": "src.baselines.build_react_tasks",
}

_HELP = """
AgML VLM Pipeline

Usage:
    python main.py <stage> [stage-args...]

Stages:
    catalog-datasets   Stage 0a — catalog all Project-AgML datasets on HuggingFace
    migrate-datasets   Stage 0a — regenerate src/config/datasets.yaml from the HF catalog
    download-datasets  Stage 0a — download + class-balanced sample datasets, write selection manifest
    download-heldout-digitalgreen  Paper 2 Stage 3 (docs/stage3_context_memory.md) — download +
                               cache DigiGreen/Crop_Disease_Images (989 real images, expert labels,
                               NOT under Project-AgML, not wired through DatasetConfig/datasets.yaml
                               — its own standalone manifest/image dir) as a HELD-OUT lesson-building
                               source, never sampled into the normal eval pipeline. Writes
                               dataset/images_heldout/ + dataset/heldout_selected_images.json, with a
                               per-row "split": "lessons"|"eval" assignment (--eval_frac, default 0.2,
                               stratified by crop, seeded by --seed) — the eval split must never be
                               used to write the skill-memory file this stage builds toward.
                               Resumable (an already-cached image/row is skipped).
    catalog-detection-datasets  Stage 5 (prep) — resolve class_names + bbox convention for
                               detection-type datasets already found by catalog-datasets
    migrate-detection-datasets  Stage 5 (prep) — regenerate src/config/detection_datasets.yaml
                               from catalog-detection-datasets' output
    download-detection-datasets Paper 2 Stage 2 / Paper 1 Stage 5 — download + class-balanced
                               sample detection datasets, write selection manifest (--output, e.g.
                               dataset/selected_images_detection_10.json for Paper 2 Stage 2's
                               10/class run; default dataset/selected_images_detection.json)
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
                               docs/stage0_judge_improvements.md.
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
    detection-baseline  Paper 2 Stage 2 — object detection: image + direct box-localization
                               question, no intermediate reasoning. Spatial-task analog of
                               vision-baseline (Paper 1 Stage 2a). Scored via IoU-based
                               precision/recall/F1 against ground-truth boxes
                               (src/common/bbox_metrics.py), not classification F1/ROUGE. See
                               docs/stage2_object_detection.md for the data-prep -> baseline
                               walkthrough. --rubric (fill-in-the-blank localization checklist
                               before printing boxes — same criteria chain detection-verifier's
                               structured generation uses, no tournament machinery — use a
                               different --output_dir than the plain run) also exists but isn't
                               part of Stage 2's current scope; rubric/verifier conditions for
                               detection are deferred to Stage 5 (docs/stage5_detection_verifier.md).
    detection-multiclass  Paper 2 Stage 5 (docs/stage5_detection_verifier.md) — same single-shot
                               box prediction as detection-baseline, but the prompt asks the model
                               to report EVERY visible instance at ANY growth stage/condition
                               (e.g. bud/open/dead flower), labeling each box with its own
                               specific stage instead of the dataset's one coarse class name.
                               Motivated by a real QC finding: ground truth often only labels ONE
                               stage as the target class, so the plain baseline's single-class
                               scoring counts correctly-detected other-stage instances as false
                               positives. Scoring is still class-agnostic IoU matching
                               (bbox_metrics.match_boxes), unchanged from detection-baseline — only
                               the prompt and each box's stage label differ. --smoketest restricts
                               to GEMINI_cowpea_flower_detection + GEMINI_cowpea_pod_detection.
    detection-box-verifier  Paper 2 Stage 5, Step 1 (docs/stage5_detection_verifier.md) — per-box
                               confidence grading over a detection-multiclass checkpoint: crops
                               each predicted box (padded, --crop_pad_frac) and shows ONLY that
                               crop to a verifier model, independent of the other boxes/full
                               image — graded against verifier_prompts_detection.yaml's box_rubric
                               (4 criteria) on a 1-3 scale, summed to a per-box confidence in
                               [0, 1] (src/verifier_cot.py's own criterion-scoring helpers, reused
                               unchanged). Boxes below --confidence_threshold (default 0.5, not
                               yet tuned) are dropped into a SEPARATE pred_boxes_filtered_json —
                               the original unfiltered pred_boxes_json/metrics are preserved
                               alongside, so metrics.json reports unfiltered vs. filtered
                               precision/recall/F1 side by side for direct comparison. Scoring
                               stays class-agnostic IoU matching, unchanged from detection-baseline.
    detection-box-verifier-summary  Paper 2 Stage 5, Step 1-report (docs/stage5_detection_verifier.md,
                               no GPU needed) — sweeps confidence_threshold (0.0-0.9) x iou_threshold
                               (0.3/0.5/0.75) directly from a detection-box-verifier checkpoint's
                               already-stored per-box confidence (pure CPU re-filtering/re-matching,
                               no new grading calls), to find which threshold combination actually
                               maximizes F1 instead of trusting whatever --confidence_threshold the
                               run happened to use. Writes (to --out_dir, default
                               <ckpt_dir>/threshold_sweep/) f1_vs_confidence_threshold.png +
                               confidence_distribution.png + criterion_score_distribution.png +
                               threshold_sweep.csv, prints the best combination. Also writes (to
                               --samples_dir, default <ckpt_dir>/refine_samples/) crop_gallery.png
                               (stratified by confidence) and, per dataset subfolder,
                               split/tighten/reclassify_timeline.png (one box's crop+scores per
                               round it went through, grouped by its first correction type —
                               detection-box-refine checkpoints only, needs per-box "history").
    detection-final-metrics  Paper 2 Stage 5-report (docs/stage5_detection_verifier.md, no GPU
                               needed) — the actual deliverable metrics at a CHOSEN
                               (--confidence_threshold, --iou_threshold) pair (e.g. whatever
                               detection-box-verifier-summary's sweep found best), in two variants:
                               Variant A (class-agnostic, every box pooled regardless of sub-class
                               — the real, deployable number) and Variant B (per-image ORACLE
                               sub-class selection — for each image, tries each locally-present
                               sub-class alone against that image's own ground truth and keeps
                               whichever wins; an upper bound, since the selection itself looks at
                               ground truth, not something a real filter could do). Writes
                               final_metrics_conf<X>_iou<Y>.json.
    detection-comparison-summary  Paper 2 Stage 5, Step 4-report (docs/stage5_detection_verifier.md,
                               no GPU needed) — grouped bar chart comparing 4 conditions at the SAME
                               --iou_threshold: Baseline (--baseline_ckpt, plain detection-multiclass/
                               -baseline, no confidence field — drawn as a dashed red REFERENCE LINE,
                               one segment per metric at that metric's own baseline value, not a bar),
                               Verifier no-refine (--verifier_ckpt, detection-box-verifier, filtered
                               at --confidence_threshold), Verifier+refine (--refine_ckpt,
                               detection-box-refine, unfiltered), Verifier+refine+filter
                               (--refine_ckpt again, filtered). Precision/Recall/F1 on the x-axis.
                               Writes comparison_conf<X>_iou<Y>.png/.json.
    detection-box-refine  Paper 2 Stage 5, Step 2 (docs/stage5_detection_verifier.md) — iterative
                               correct-and-reverify loop on top of detection-box-verifier's own
                               crop/grade mechanism (box_rubric, now 3 criteria). Any criterion
                               scoring <= 2 triggers a real corrective generation call (unlike
                               detection-box-verifier, which only grades): criterion 3 (multiple
                               instances merged) -> SPLIT the crop into separate boxes; criterion 2
                               (box not tight) -> TIGHTEN it; criterion 1 (not a real instance) ->
                               RECLASSIFY against this image's own other sub-class labels, or DROP
                               if none match. Split-then-tighten-then-reclassify order (not the
                               rubric's own 1/2/3), since splitting changes box count and would
                               waste a correction applied before it. A box gets AT MOST ONE
                               correction per round (whichever its current grade triggers), then is
                               re-graded fresh next round — --max_rounds (default 2) caps how many
                               rounds before a still-failing box is kept as-is and flagged
                               "exhausted" rather than looped forever.
    detection-qc-plot        — plot predicted vs. ground-truth boxes for a sample of images, to
                               visually catch a coordinate-alignment bug. --compare_ckpt plots a
                               second checkpoint (e.g. the --rubric run) side by side against the
                               same ground truth for the same images (Paper 2 Stage 2).
                               --confidence_threshold drops boxes below it before plotting (a
                               detection-box-refine/-verifier checkpoint's own per-box confidence;
                               boxes with no confidence at all are always kept) — when set, the
                               title's P/R/matched numbers are recomputed on the filtered set at
                               --iou_threshold rather than trusting the checkpoint's own stored
                               (unfiltered) columns. IMPORTANT: --out_dir defaults to <ckpt's own
                               dir>/qc_plots/ regardless of --confidence_threshold — an unfiltered
                               and a filtered run against the SAME checkpoint silently overwrite
                               each other's images unless you pass --out_dir explicitly to give
                               the filtered run its own folder.
    detection-rubric-summary Paper 2 Stage 2-report — bar charts comparing detection-baseline vs.
                               detection-baseline --rubric at an equal image budget (mean
                               precision/recall/F1/IoU with 95% CI, per-dataset breakdown, F1 vs.
                               IoU-threshold sweep) — no GPU needed.
    verifier-cot    --verifier_type simple (default, Paper 2 Stage 4, docs/stage4_classification_verifier.md):
                               ONE trajectory per image, graded against each rubric criterion
                               independently on a 1-3 scale, summed/normalized to verifier_score in
                               [0, 1] — no K-candidate pool, no tournament. Reports a point-biserial
                               correlation between verifier_score and judged correctness in
                               metrics.json, to see whether this cheaper signal is usable as a
                               filter/gate before adopting it elsewhere (e.g. react-baseline).
                               --verifier_type top3 (Paper 2 Stage 4, docs/stage4_classification_verifier.md):
                               same per-criterion grading, but draws --top3_n_candidates (default 3)
                               independent trajectories per image in ONE generation call and grades
                               each independently — no tournament, just N absolute scores. The
                               highest-verifier_score candidate wins (argmax); the other candidates'
                               answers/ranks are appended to the response as plain text, never inside
                               a second <answer> tag. Also reports top3_pool_analysis in
                               metrics.json: pool_coverage (does the right answer appear in ANY
                               candidate) and verifier_pick_accuracy_given_in_pool (when it does,
                               does the verifier's own grading actually pick it).
                               --top3_mode verifier_rubric (2026-10-03): a different division of
                               labor, aimed at the near-ceiling score=3 pattern seen in practice —
                               candidates are just N bare answers (no rubric-aligned observational
                               preamble to trivially "cite"), and the VERIFIER independently fills
                               in the rubric from the image ONCE (shared evidence, no candidate text
                               shown), then scores each candidate's bare answer against THAT
                               evidence in its own call. --enable_thinking applies to all three
                               calls in this mode (generation, evidence-fill, scoring) — unlike
                               "explicit"/"resample", whose grading call is always thinking-off.
                               Raise --grading_max_tokens well above its 300 default when combining
                               with --enable_thinking.
                               --verifier_type ppt (Paper 1 Stage 4, docs/stage4_verifier.md) — LEGACY
                               as of 2026-10-01, no longer actively tested/extended for Paper 2 (kept
                               for Paper 1 reproducibility only): the original single-round
                               K-candidate + PPT-tournament verifier — K candidates generated per
                               image, ranked once by the Probabilistic Pivot Tournament,
                               argmax(win_rate) reported. --k_candidates/--k_pivots are the main
                               tunables (ppt only). Compared against src/verifier_lmv.py's full PPT
                               tournament (run separately, not wired into main.py — see its own
                               --help) in `cot-compare`.
                               --num_shards/--shard_index (data-parallel across GPUs, same
                               convention as vision-baseline) + --merge_shards work with every mode.
    verifier-score-analysis   — diagnostic (no GPU needed): re-derives the paper's Appendix C
                               Figure 8/9 checks (score distribution + retained-set accuracy vs.
                               threshold, nearest-letter histogram) from an existing verifier-cot
                               run's own ckpt CSV + verifier_log.jsonl, to see whether the
                               fine-grained-reward score is discriminating (or inverted/collapsed)
                               BEFORE picking a fix. --model/--output_dir or --verifier_dir.
    verifier-score-correlation-plot — Paper 2 Stage 4 (docs/stage4_classification_verifier.md,
                               no GPU needed): turns metrics.json's verifier_score_correlation (a
                               handful of numbers) into ONE PNG — verifier_score split by judged
                               correctness, box+jittered-strip and an overlaid histogram side by
                               side, annotated with point-biserial r and both group means
                               (recomputed from --ckpt directly, not trusted from metrics.json).
                               Works for both --verifier_type simple and top3 checkpoints.
    verifier-cot-qc-plot       — Paper 2 Stage 4 (docs/stage4_classification_verifier.md, no GPU
                               needed): (1) a rubric gallery — --n_datasets (default 10) random
                               datasets x --n_per_dataset (default 3) random images, each shown
                               next to its ground_truth/verifier_score/criterion_scores (raw JSON
                               plus each criterion's own rubric question text), paginated via
                               --max_rows_per_figure; (2) a criterion score distribution — one
                               stacked bar chart (counts of score 1/2/3) per criterion, faceted
                               by task_type since different task_types use different rubrics
                               (criterion 1 means something different per task_type — see
                               verifier_prompts.yaml's task_type_to_rubric). Images resolved by
                               globbing whatever's already cached under --images_root (never
                               downloads/re-caches — read-only QC, not a generation script).
    verifier-cot-qc-plot-top3 — Paper 2 Stage 4 (docs/stage4_classification_verifier.md, no GPU
                               needed): same idea as verifier-cot-qc-plot but for a
                               --verifier_type top3 checkpoint (any --top3_mode). (1) a candidate
                               gallery — each sampled image next to ALL 3 candidate predictions
                               (verifier_score each, winner marked [CHOSEN]), the winner's own
                               criterion_scores (bare numbers, in order), and the judge's verdict;
                               (2) winner_score_correlation.png (reuses
                               verifier-score-correlation-plot's own plot_correlation against the
                               winning candidate's score vs. judged_correct); (3)
                               per_candidate_score_distribution.png — ALL 3 candidates per row
                               (not just winners), split by a cheap ground_truth TEXT-MATCH proxy
                               (not the real judge, which only ever grades the winner) — a broader
                               signal check than (2) alone; (4) pool_summary.png/.json —
                               pool_coverage / verifier_pick_accuracy_given_in_pool as a chart
                               (same numbers as metrics.json's top3_pool_analysis).
                               --n_datasets/--n_per_dataset/--max_rows_per_figure/--images_root
                               same meaning as verifier-cot-qc-plot.
    verifier-cot-top3-refine  — Paper 2 Stage 4 (docs/stage4_classification_verifier.md, needs
                               GPU/model): classification analog of detection-box-refine, applied to
                               an already-graded --verifier_type top3 --top3_mode verifier_rubric
                               checkpoint. For each row whose winning candidate scored <= 2 on any
                               rubric criterion: one correction call (shown to the MODEL, not the
                               verifier — the SAME round-1 evidence, which criteria failed with
                               their own rubric text, and the losing round-1 candidates) followed by
                               one re-scoring call against that same evidence. --max_rounds (default
                               2) — round 1 is the input checkpoint's own ALREADY-DONE grading (no
                               calls), so the default allows exactly one corrective round. A row
                               that passes needs no extra calls at all and keeps its original
                               judged_correct; only actually-corrected rows get re-judged. Writes a
                               checkpoint in the SAME schema as its input (VERIFIER_CKPT_FIELDS) plus
                               one extra refinement_info JSON column — round-1-only columns
                               (candidate_predictions/candidate_verifier_scores/answer_in_pool/
                               verifier_picked_correct) pass through UNCHANGED, so
                               verifier-cot-qc-plot-top3's pool_summary chart still reflects round 1
                               only when run against this output.
    verifier-cot-comparison-summary — Paper 2 Stage 4 (docs/stage4_classification_verifier.md, no
                               GPU needed): bar chart — vision-baseline (--baseline_ckpt, filtered to
                               just the datasets --verifier_rubric_ckpt actually covers, drawn as a
                               dashed red reference line) vs. --verifier_rubric_ckpt (top3
                               verifier_rubric, round 1) vs. --refined_ckpt (that same checkpoint
                               after verifier-cot-top3-refine). Metric: judge_accuracy (accuracy also
                               reported in the JSON for reference).
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
    judge-checkpoint  Paper 2 Stage 0 (docs/stage0_judge_improvements.md, needs judge model/GPU)
                               — judge an existing baseline checkpoint with the pipeline-wide
                               default hybrid template, ONE pass (run_judge_phase, the exact
                               mechanism vision-baseline's own inline judge phase uses) — no
                               leniency/consistency/template-sensitivity comparison. --ckpt is
                               never modified in place — copied into --output_dir first. Hand-edit
                               prompts.yaml's judge_classification_hybrid_template between runs and
                               point --output_dir at a fresh folder each time (with --rejudge, to
                               force every row re-graded) to compare wording variants without one
                               overwriting another's results.
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
                               (off/unset by default — see docs/stage0_judge_improvements.md).
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
    judge-qa-export            — no GPU needed: samples a review-ready CSV of DISAGREEMENTS between
                               two judge-checkpoint runs over the same classification checkpoint —
                               --baseline_ckpt (e.g. vision-baseline's own inline-judged checkpoint)
                               vs. --variant_ckpt (e.g. a judge-checkpoint --rejudge run after
                               hand-editing the hybrid template's wording). A row qualifies if
                               judged_correct differs between the two, stratified by task_type
                               (--n_per_task, default 50). --seed (default 42), --output_csv
                               (default dataset/judge_qa_review/judge_qa_review.csv — dataset/, not
                               outputs_final/, so it's git-tracked for reproducibility — name it
                               judge_qa_review_v<N>.csv per the --variant_ckpt you're reviewing).
    collate         Stage 6 — collate all stage metrics into results_table.csv / .tex
    react-build-tasks  Stage 6 (Paper 2/6) — builds a PERSISTED --tasks_jsonl file for
                               react-baseline (below) from REAL AgML images: same
                               active_datasets/load_hf_samples/sample_class_images sampling AND the
                               same cls_question() prompt text vision_baseline.py itself uses (pass
                               --rubric for the SAME fill-in-the-blank criteria chain
                               vision-baseline's own --rubric flag AND verifier_cot.py's own
                               k-candidate generation step always use, to compare against those
                               instead of plain vision-baseline), so react-baseline is genuinely
                               comparable, not a different task. --match_images_from an existing
                               vision_baseline checkpoint for a true apples-to-apples image set.
                               react-baseline's own --sample_images does the same sampling
                               IN-MEMORY (no file) for a one-off run — use this script instead when
                               you want the task list reviewable/reusable across multiple runs. See
                               docs/stage6_react_baseline.md.
    react-baseline  Stage 6 (Paper 2/6) — a tool-using ReAct agent (src/baselines/) over a
                               locally-served vLLM model's own native tool calling (same
                               load_llm/render_chat_template machinery as every other script here,
                               NOT a hosted API) — read/glob/grep/write (write sandboxed per
                               episode) plus agml-agent's retrieve/search_agml called directly as
                               Python (no MCP connection needed). Default model: RedHatAI's
                               quantized Gemma-4 + the vendored tool_chat_template_gemma4.jinja;
                               --tool_call_format auto-detects the matching parser
                               (tool_call_formats.py) by model name. Three ways to run (in priority
                               order): --sample_images (real AgML images sampled on the fly, same
                               code/flags as react-build-tasks — --rubric available here too);
                               --tasks_jsonl <path> (a row carrying "image_path" runs as a real
                               image episode, otherwise text-only — see
                               dataset/react_smoke_test_tasks.jsonl for a 3-task text-only smoke
                               test); --task "..." (single ad-hoc text-only episode). One checkpoint
                               row per episode (resumable) + a shared trajectory log under
                               outputs_final/react_baseline/<model>/, each row carrying
                               elapsed_seconds; browse with `streamlit run
                               app/react_baseline_viewer.py`. Defaults in
                               src/config/react_baseline_config.yaml — see
                               docs/stage6_react_baseline.md.

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
