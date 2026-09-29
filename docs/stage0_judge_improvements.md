# Paper 2, Stage 0 — Judge Improvements (Phase 1-3)

[← Back to README](../README.md)

Quantifies how lenient and how stable the LLM judge is when grading plain classification predictions, then tests whether a different judge template fixes it. Runs against the checkpoint [Stage 1](stage1_baseline_classification.md) generates — run that first if it doesn't exist yet.

- **Phase 1 — leniency**: does the judge give more credit to the full free-text response than to just the resolved class name?
- **Phase 2 — self-consistency**: does the judge's verdict flip across repeat calls at higher temperature?
- **Phase 3 — template sensitivity**: compares 4 judge templates by re-running Phase 1 + 2 under each. `hybrid` won and is now the pipeline-wide default (`DEFAULT_JUDGE_TEMPLATE_KEY`, `src/common/metrics.py`) — every judge call in the repo uses it unless a different template is explicitly requested.

## Phase 1 — leniency

```bash
uv run python main.py judge-leniency \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_leniency/vision_baseline_thinking/
```

Writes `leniency_gap_T<temp>.png`, `leniency_gap_by_temperature.png`, `disagreement_matrix_T<temp>.png`, `rechecked_T<temp>.csv`, `leniency_gap.json`.

## Phase 2 — self-consistency

```bash
uv run python main.py judge-consistency \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_consistency/vision_baseline_thinking/ \
    --leniency_gap_json ablation/judge_leniency/vision_baseline_thinking/leniency_gap.json
```

Runs both conditions (full response / `<answer>`-tag-only) over Phase 1's same rows. `--leniency_gap_json` (optional) adds a leniency-vs-flip-rate correlation plot.

## Phase 3 — template sensitivity

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir outputs_final/judge_template_sensitivity/vision_baseline_thinking/ \
    --judge_enable_thinking --judge_max_tokens 2048 \
    --judge_chat_template_path src/config/tool_chat_template_gemma4.jinja
```

`--judge_enable_thinking`/`--judge_max_tokens`/`--judge_chat_template_path` work identically on `judge-leniency`, `judge-consistency`, and `vision-baseline`'s own built-in judge phase — needed for a Gemma-4 judge, whose stock template has no thinking support at all.

`--template` (default `all`) runs every variant against the same `--output_dir`, backfilling the cross-variant summary (`template_sensitivity.png`/`.tex`) as each finishes — pass e.g. `--template hybrid` to run just one.

**End-of-run outputs** (once `current`/`hybrid` have data on disk): `judge_accuracy_f1.png`, `leniency_and_false_positive_summary.png` (+ `_stats.json`), `oracle_validation_sample.csv` (superseded by `judge-qa-export` below for a more deliberately-sampled version).

## Backfilling a checkpoint's own `judged_correct`

Once you've decided which template's **full-response** verdict to trust, copy it into the checkpoint (needed for a `--skip_judge` checkpoint, which starts with blank `judged_correct`):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/ \
    --backfill_judge_results \
    --rechecked_csv outputs_final/judge_template_sensitivity/vision_baseline_thinking/hybrid/leniency/rechecked_T0.7.csv
```

Joins on `(agml_dataset, class_label, image_name)`; unmatched rows are left unchanged. Also auto-generates `task_f1_summary.png` (judged vs. strict F1 per task_type, with CI).

**Comparing two templates directly** (e.g. before deciding which to backfill) — `--compare_templates --template_rechecked_csv current=<path> hybrid=<path>` writes `template_comparison.png` (judged F1 per task_type, one bar per template).

## Judge QA review export

Builds a review-ready CSV for a human reviewer to hand-check judge verdicts, sampling three pools per `task_type`: `targeted_hybrid` (hybrid disagrees with a cheap text check), `disagreement` (current vs. hybrid disagree), `random_control`. See `src/judge_qa_export.py`'s module docstring for the full pool logic.

```bash
uv run python main.py judge-qa-export
```

Writes to `dataset/judge_qa_review/judge_qa_review.csv` — `dataset/`, not `outputs_final/` (gitignored), so the sampled review set is reproducible from the repo.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--ckpt` | *(required)* | Checkpoint dir/CSV to evaluate |
| `--output_dir` | `ablation/judge_<phase>/` | Where charts/CSVs are written |
| `--temperatures` | `[0.7]` | Judge sampling temperature(s) |
| `--n_repeats` | `5` | Repeat judge calls per row (Phase 2 cost) |
| `--template` | `all` | Phase 3 variant(s): `current`/`specificity`/`class_list`/`hybrid`/`all` |
| `--skip_consistency` | off | Skip Phase 3's Phase-2 sub-pass (cheaper, no flip-rate columns) |
| `--judge_model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` | Judge model |
| `--judge_enable_thinking` | off | Turn on judge thinking (needs `--judge_chat_template_path` for Gemma-4) |
| `--judge_max_tokens` | `1024` | Judge generation budget (raise with thinking on) |
| `--judge_chat_template_path` | none | Custom chat template (required for Gemma-4 + thinking) |
| `--n_per_task` (`judge-qa-export`) | `50` | Rows sampled per pool per task_type |
| `--seed` (`judge-qa-export`) | `42` | Sampling seed |
| `--summary_temperature` | `0.7` | Which `--temperatures` value the end-of-run summary reads |
