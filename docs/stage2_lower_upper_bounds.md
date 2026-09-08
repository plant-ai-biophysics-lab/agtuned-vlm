# Stage 2 — Lower and Upper Bounds (Vision Baseline / Description-Augmented)

[← Back to README](../README.md)

Brackets how much of vision classification performance depends on a good text description vs. the model's own raw visual reasoning:

- **Lower bound** (`vision-baseline`, Stage 2a) — image + classification question only, no description.
- **Upper bound** (`desc-aug-v2`, Stage 2b) — same setup, plus a reference description (from `dataset/agml_classification.jsonl`) prepended to the question.

Both are classification-only (no separate description-quality metric): each sample is scored on exact/contains-match F1, ROUGE-SS against the ground-truth class label, and an LLM judge's accuracy. Both write a per-model, per-sample checkpoint CSV, flushed after every sample, so a run on a time-limited remote job resumes exactly where it left off.

Defaults for all three scripts on this page come from `src/config/vision_baseline_config.yaml` — edit it directly (model, sampling, judge settings, output dirs) instead of passing flags every run. Any flag passed on the command line overrides the config.

## Step 1 — lower bound

```bash
uv run python main.py vision-baseline --model google/gemma-4-26B-A4B-it
```

Run once per model (`--model` overrides the config's `model` key). Writes to `<baseline_output_dir>/<safe_model_name>/`:
- `ckpt_<safe_model_name>.csv` — per-sample rows, resumable
- `metrics.json` — per-dataset/task/overall F1, ROUGE-SS, judge accuracy

## Step 2 — upper bound

```bash
uv run python main.py desc-aug-v2 --model google/gemma-4-26B-A4B-it
```

Same per-model/resumable setup, writes under `<aug_output_dir>/<safe_model_name>/`. ~6% of classes have no synthesized reference description (mostly `healthy_skip` classes) — those fall back to a fixed, class-agnostic placeholder (*"This appears to be a normal image. No potential stresses can be seen in this image."*) rather than being skipped.

## The judge phase

Both scripts run an LLM-judge pass after classification (fixed judge model across every run, `judge_model` in the config, so `judge_accuracy` is comparable everywhere). It's a separate, resumable phase — checkpointed independently and chunked (`judge_chunk_size`) so a kill mid-judging only loses the in-flight chunk. Every judge call's full input/output is logged to `<model_dir>/judge_log.jsonl` for QA.

Useful flags:
- `--skip_judge` — skip the judge model entirely (no GPU needed for it), for a fast preliminary run of just F1/ROUGE-SS. Rerun later without the flag to backfill judging.
- `--rejudge` — clear `judged_correct`/`judge_reasoning` and re-judge every row from scratch (e.g. after a judge bug is fixed).

## Step 3 — run more models

Repeat Steps 1–2 for every model you want compared (`--model` per invocation, same two scripts).

## Step 4 — collate

Once every model has finished both scripts:

```bash
uv run python main.py vision-baseline-collate
```

Reads both output dirs and puts lower/upper bound side by side per model, into `<collated_output_dir>/`:
- `stage2_summary.csv` / `.json` — model, condition, scope (overall/per-task), F1 + 95% CI (bootstrap over classes)
- `stage2_f1_by_task.png` — grouped bar chart, one bar per (model, condition), with CI error bars
- `stage2_f1_percentiles.png` — supplementary box plot of each model/condition's raw per-class F1 spread (tight box = uniform across classes, wide box = some classes much harder)

## Related conditions

Two more conditions build on the same checkpoint/judge infrastructure but live in their own scripts, not this collation: `self-cot` (Stage 2d, own free-form description before classifying) and `cot-yaml` (Stage 2e, structured multi-step reasoning before classifying).
