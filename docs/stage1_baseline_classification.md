# Paper 2, Stage 1 — Baseline Classification Runs

[← Back to README](../README.md)

The plain, thinking-enabled single-shot classification checkpoint — no rubric, no verifier, no tournament, just the model answering the classification question directly. Every later Paper 2 stage builds on this: Stage 0 tests the judge against it, Stage 4's `verifier-cot --from_ckpt` can grade its trajectories directly, and it's the reference point for Stage 4's verifier score and Stage 6's ReAct agent.

## Generate the checkpoint (thinking enabled)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/
```

Thinking is **on by default** (`vision_baseline_config.yaml`'s `enable_thinking: true` — the pipeline's standard workflow), along with the matching `gen_max_tokens: 4096` and `chat_template_path` (Gemma-4's own bundled template has no `enable_thinking` support at all, so this is required for it to actually produce a `<|channel>thought...<channel|>` trace). Pass `--no-enable_thinking` to opt out (see "Without thinking" below); for a non-Gemma model (e.g. Qwen, whose bundled template already supports thinking directly), also pass `--chat_template_path ""` to unset the Gemma-4-specific template.

The checkpoint's `raw_response` column holds the fully untouched generation; `reasoning` holds the extracted thinking text; `prediction` — the ONLY column the judge ever reads — is required to include a brief evidence statement before the answer (`assessment_classification_tagged`, `src/config/prompts.yaml`), not just a bare class name.

Judging runs automatically as part of this same command (the pipeline-wide `hybrid` template, `DEFAULT_JUDGE_TEMPLATE_KEY`) — pass `--skip_judge` to generate first and judge later instead.

## Without thinking (optional, for a thinking-vs-not comparison)

Add `--no-enable_thinking` — same command otherwise, a different `--output_dir` (e.g. `outputs_final/vision_baseline/`) so the two checkpoints don't collide.

## Rubric CoT (structured fill-in-the-blank, no verifier/tournament)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --rubric \
    --gen_max_tokens 1024 \
    --output_dir outputs_final/vision_baseline_rubric_thinking/
```

`--rubric` wraps the question through `verifier_prompts.yaml`'s structured fill-in-the-blank criteria chain (`task_type_to_rubric`) — the same model-facing template `verifier-cot` uses for its own K-candidate generation ([Stage 4](stage4_verifier.md)), but as a single plain answer with no candidate pool and no tournament ranking at all. This is the "Rubric-based CoT" condition — isolates whether the structured-criteria prompt alone helps, independent of PPT-tournament selection. Combine freely with `--enable_thinking`/`--no-enable_thinking` (independent knobs — this changes what's asked, thinking changes whether the model reasons before answering); raise `--gen_max_tokens` (e.g. 1024+) since the rubric prompt asks for several fields before the final answer, on top of any thinking trace.

## What's next

Stage 0 (judge tests), Stage 2 (same idea for detection), Stage 4 (`verifier-cot --from_ckpt` grades these trajectories directly), Stage 6 (ReAct agent answers the same question for comparison).

## Splitting the work across multiple GPUs

`--num_shards N --shard_index i` partitions the work into N independent single-GPU processes, each generating **and** judging its own shard — real multi-GPU parallelism for both phases, not just generation. Run one process per GPU with its own `--shard_index` (e.g. one array task per GPU on a cluster scheduler):

```bash
# 4 GPUs, one process each ($SLURM_ARRAY_TASK_ID = 0, 1, 2, 3):
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID
```

Each shard writes its own `ckpt_<model>_shard<i>of4.csv` (never the canonical filename) plus its own `judge_log_shard<i>of4.jsonl`, so all 4 can run concurrently with zero write contention. Once every shard finishes, merge them into the canonical checkpoint (single job, after all array tasks complete):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/ \
    --merge_shards
```

Don't leave shard files sitting alongside the merged canonical file when pointing Stage 0's tooling at that directory afterward — its `--ckpt` glob matches both and double-judges every row.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--model` | *(config: `RedHatAI/gemma-4-31B-it-FP8-dynamic`)* | Model to run — Qwen also works (e.g. `Qwen/Qwen3.8-27B-FP8`), pass `--chat_template_path ""` alongside it |
| `--enable_thinking` / `--no-enable_thinking` | **on** (config) | Extended thinking — `--no-enable_thinking` to opt out |
| `--gen_max_tokens` | `4096` (config) | Generation budget (`256` if thinking is off) |
| `--chat_template_path` | vendored Gemma-4 template (config) | Custom chat template — pass `""` to unset for a non-Gemma model |
| `--output_dir` | `outputs_final/vision_baseline/` | Checkpoint output dir |
| `--skip_judge` | off | Skip the judge phase (F1/ROUGE only) |
| `--num_shards` / `--shard_index` | `1` / `0` | Data-parallel sharding across GPUs |
| `--n_samples` | `10` | Images/class |
| `--rubric` | off | Wrap the question through the rubric fill-in-the-blank template |
