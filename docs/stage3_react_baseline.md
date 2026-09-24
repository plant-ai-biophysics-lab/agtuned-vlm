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

The real heldout-AgML classification harness (sample images, build the
classification question, score `finish`'s answer against ground truth) is
**not built yet**. What exists today already supports a **text-only smoke
test** — a handful of freeform grounding/reasoning tasks, no images — to
verify the loop, the tool-calling parser, and the checkpoint plumbing
actually work end to end before building anything more elaborate on top.

## Step 0 — smoke test (3 examples)

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

## Step 1 — view the results

```bash
streamlit run app/react_baseline_viewer.py -- \
    --output_dir outputs_final/react_baseline/
```

Browse every episode (filter by `stop_reason`, search task/result text),
click a row for its full task/result and the matched turn-by-turn
trajectory. `--images_root dataset/images` (the default) shows the source
image for any row that does carry `agml_dataset`/`class_label`/
`image_name` — the ad-hoc smoke-test rows above won't have one, since they
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
  at least a `"task"` key (optional `"agml_dataset"`/`"class_label"`/
  `"image_name"` for a real image-tied episode, once that sampling step
  exists). `--task "..."` runs a single ad-hoc episode instead, for a
  quick one-off check.
- `--rerun` archives the existing checkpoint + trajectory log (timestamped,
  never deleted) and regenerates every episode from scratch — same
  convention as every other stage's `--rerun`.
- `run_episode()` is still **text-only** — no image content block is wired
  into the prompt yet. A `--tasks_jsonl` row's `agml_dataset`/`class_label`/
  `image_name` are bookkeeping only right now, not something the model
  actually sees.
