# Paper 2, Stage 3 — ReAct Baseline

[← Back to README](../README.md)

A tool-using agent (`src/baselines/`) as an alternative to the single-shot
`vision-baseline`/`verifier-cot` pipelines — the model does its own
Reason→Act→Observe loop over `read`/`glob`/`grep`/`write` (write sandboxed
per episode) plus `agml-agent`'s `retrieve`/`search_agml` (called directly
as Python, no MCP), via its own chat template's native tool calling. Same
vLLM model-serving path as every other stage — no external API. Full design
detail lives in the module docstrings of `src/baselines/react_baseline.py`,
`react_agent.py`, `tools.py`, and `tool_call_formats.py` — this doc is just
the "how do I run it" version.

Comparable to `vision-baseline`/`verifier-cot` by construction, not by
accident: `react-build-tasks` (Step 1 below) samples images and builds the
classification question the exact same way `vision_baseline.py` does
(`active_datasets`/`load_hf_samples`/`sample_class_images`/`cls_question`)
— a `react-baseline` run over its output is scoring the same task on the
same images, not a different one.

## Step 0 — smoke test (3 examples, text-only, no GPU-hungry image step)

```bash
uv run python main.py react-baseline \
    --tasks_jsonl dataset/react_smoke_test_tasks.jsonl
```

`dataset/react_smoke_test_tasks.jsonl` has 3 hand-picked tasks that each
exercise a different path through the loop: one that should call
`search_agml`, one that should call `retrieve`, and one simple enough the
model may just answer directly with no tool call at all. No `--model`/
`--chat_template_path` needed — both default correctly for the standard
checkpoint (`RedHatAI/gemma-4-31B-it-FP8-dynamic` +
`src/config/tool_chat_template_gemma4.jinja`, see
`src/config/react_baseline_config.yaml`).

This writes, under `outputs_final/react_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/`:

- `ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv` — one row per task (result,
  `stop_reason`, `n_steps`, `tools_used`, `elapsed_seconds`, ...).
  Resumable: re-running the same command skips any task already in this
  file, so a killed run or a `--tasks_jsonl` file you keep appending to
  never redoes finished work.
- `trajectory_log.jsonl` — the full turn-by-turn raw generations, one
  shared file, each line tagged with its `episode_key`.

Confirm the 3 rows all land with `stop_reason=finished` (not
`no_tool_call` on every row — that specific failure mode usually means
`--tool_call_format` picked the wrong parser for the served model, see
`tool_call_formats.py`'s module docstring) before trusting anything built
on top of this.

## Step 1 — real classification episodes (images included)

Build a `--tasks_jsonl` file from real AgML images — 3 images/class by
default, same sampling `vision_baseline.py` uses:

```bash
uv run python main.py react-build-tasks \
    --n_samples 3 \
    --output outputs_final/react_tasks_smoke3.jsonl
```

Then run it — each row's `image_path` gets loaded and attached to its
episode (`react_agent.py`'s `image` param), so the model actually sees the
image this time, not just the question text:

```bash
uv run python main.py react-baseline \
    --tasks_jsonl outputs_final/react_tasks_smoke3.jsonl
```

For a true apples-to-apples comparison against an existing
`vision-baseline` run (same exact images, not just the same sampling
*procedure*), point `--match_images_from` at its checkpoint:

```bash
uv run python main.py react-build-tasks \
    --match_images_from outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --n_samples 50 \
    --output outputs_final/react_tasks_full.jsonl
```

## Step 2 — view the results

```bash
streamlit run app/react_baseline_viewer.py -- \
    --output_dir outputs_final/react_baseline/
```

Browse every episode (filter by `stop_reason`, search task/result text),
click a row for its full task/result and the matched turn-by-turn
trajectory. `--images_root dataset/images` (the default) shows the source
image for any row from Step 1 (real `agml_dataset`/`class_label`/
`image_name`) — the Step 0 smoke-test rows won't show one, since they
aren't tied to a real image.

## Timing

Every checkpoint row's `elapsed_seconds` column is that one episode's own
wall-clock time (`run_episode()` alone — excludes the one-time model
load/destroy around the whole batch) — the viewer's "Avg. time/episode"
metric tile and the per-row detail view both surface it, so a slow outlier
(stuck near `--max_iterations`, or a slow `retrieve()`/`search_agml()`
call) is visible straight from the checkpoint without re-reading
`trajectory_log.jsonl` timestamps.

## Notes

- `--tasks_jsonl` is the batch mode — one JSON object per line, each with
  at least a `"task"` key, plus `"agml_dataset"`/`"class_label"`/
  `"image_name"`/`"image_path"` for a real image-tied episode
  (`react-build-tasks`'s own output shape). A row without `"image_path"`
  runs text-only, same as before. `--task "..."` runs a single ad-hoc
  text-only episode instead, for a quick one-off check.
- `--rerun` archives the existing checkpoint + trajectory log (timestamped,
  never deleted) and regenerates every episode from scratch — same
  convention as every other stage's `--rerun`.
- `--vision_token_budget` (default matches `vision_baseline.py`'s own)
  only matters for rows carrying `image_path` — it's a model-load-time
  vLLM kwarg (`mm_processor_kwargs`), so it's always set, not conditional
  on whether any given row actually has an image.
- Scoring `finish`'s answer against ground truth (F1/contains-match/judge,
  the way `vision_baseline.py`'s checkpoint gets scored) isn't built yet —
  `react-build-tasks`'s output carries `class_label` per row, so this is a
  straightforward next step once you're ready to compare accuracy, not a
  redesign of the images/task-format work above.
