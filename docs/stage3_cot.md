# Stage 3 — Chain-of-Thought (CoT Self Description / CoT User Defined)

[← Back to README](../README.md)

Two more conditions beyond the [lower/upper bound](stage2_lower_upper_bounds.md): the model reasons about the image before classifying it.

- **CoT Self Description** (`self-cot`) — the model describes the image in its own words, then classifies using its own description as context.
- **CoT User Defined** (`cot-yaml`) — the model works through a fixed, task-specific checklist of questions (`src/config/cot_prompts.yaml`), one turn each, before classifying with the whole checklist trail as context.

Both share the same checkpoint/judge infrastructure as [vision-baseline / desc-aug-v2](stage2_lower_upper_bounds.md) — resumable per-model CSVs, the same judge model, the same `--skip_judge`/`--rejudge` flags. Defaults come from `src/config/cot_config.yaml`.

## Step 1 — run both conditions

```bash
uv run python main.py self-cot --model google/gemma-4-26B-A4B-it
uv run python main.py cot-yaml --model google/gemma-4-26B-A4B-it
```

Run once per model. Writes to `<output_dir>/<safe_model_name>/`:
- `ckpt_<safe_model_name>.csv` — final prediction per image, resumable
- `metrics.json` — per-dataset/task/overall F1, ROUGE-SS, judge accuracy

Collate one script's models together the same way as every other stage (`--collate`, no `--model` needed).

## Step 2 — per-step F1 for CoT User Defined

`cot-yaml` normally only classifies once, after the *last* question in the checklist. `--per_step` also classifies after every intermediate question, so you can see how F1 changes as reasoning accumulates:

```bash
uv run python main.py cot-yaml --model google/gemma-4-26B-A4B-it --per_step
```

Writes a second checkpoint, `ckpt_steps_<safe_model_name>.csv` (one row per image per chain position, not one row per image). Roughly doubles generation cost (one extra, cheap classification pass per step) — pass `--per_step` from the start of a run rather than adding it on a resume, since images already finished under a plain run have no saved intermediate responses to backfill from.

View the results (no GPU needed):

```bash
uv run python main.py cot-yaml-step-report
```

Into `<output_dir>/<safe_model_name>/`:
- `step_f1_by_dataset.csv` / `step_f1_by_task.csv` — F1 at each step, broken down
- `step_f1_overall.png` — F1 vs. step index, one line per task type + overall

## Step 3 — win rate / mean gain / mean loss vs. the lower bound

Once `vision-baseline` and any of `desc-aug-v2` / `self-cot` / `cot-yaml` have run for a model:

```bash
uv run python main.py cot-compare
```

Per method, per dataset, compares judged F1 against `vision-baseline` (the lower bound) and reports, into `<output_dir>/<model>/`:
- `method_comparison.csv` / `.json` — win rate, mean gain (among datasets that improved), mean loss (among datasets that regressed)
- `top_movers.csv` — the top-N gain and top-N loss datasets per method (`--top_n`, default 5)
- `method_comparison.png` — win rate + gain/loss bar charts
- `cot_yaml_swap_datasets.txt` — `cot-yaml`'s top-N gains + top-N losses, space-separated, ready to feed into Step 4

## Step 4 — does the question order matter? (top movers only)

For `cot-compare`'s biggest `cot-yaml` gains and losses, re-run generation once per rotation of the step order (every question gets a turn going first), to check the per-step F1 curve isn't just an artifact of which question happened to lead:

```bash
uv run python main.py cot-yaml --model google/gemma-4-26B-A4B-it \
    --swap_datasets $(cat outputs_final/cot_compare/<safe_model_name>/cot_yaml_swap_datasets.txt)
```

Scoped to a handful of datasets deliberately — cost scales with the number of questions in that task's checklist (3–5x a normal `--per_step` run for just those datasets). Writes `ckpt_swap_<safe_model_name>.csv` instead of the normal checkpoint. View it the same way as Step 2:

```bash
uv run python main.py cot-yaml-step-report
```

Adds, per task type present in the swap data:

- `swap_step_f1.csv` / `swap_step_f1_<task>.png` — F1 at each step, one line per rotation
- `anchor_recovery.csv` / `anchor_recovery_<task>.png` — **anchor rate** (P(final correct | step-*k* correct) — does being right early "lock in" a right final answer?) and **recovery rate** (P(final correct | step-*k* wrong) — can a wrong early step still end up right?), at every step position, one line per rotation
- `diagnosis_stability.csv` / `diagnosis_stability_<task>.png` — buckets each image by whether its predicted class ever changed across the chain (**stable** = same prediction every step, **unstable** = changed at least once) and reports final judged F1 per bucket, per rotation — does a flip-flopping diagnosis predict a wrong final answer?

All three read the same `ckpt_swap_<safe_model_name>.csv` — no extra generation needed beyond Step 4's swap run.
